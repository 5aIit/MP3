"""Narrow repair proposals for unsupported filters and aggregate fan-out.

Deleting a JOIN or filter is intentionally not claimed to preserve the input
SQL's meaning. The question and schema support a candidate interpretation;
the controller must check every candidate before selecting it.
"""
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError


REQUEST_WORDS = set("what which are is was were the a an of all every each and or "
                    "list show give me please names name return find get their those "
                    "that who to".split())
TABLE_GENERIC = {"detail", "record", "data", "table", "link", "mapping", "and", "of"}


def _stem(word):
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _words(text):
    return {_stem(word) for word in re.findall(r"[a-z]+", text.casefold())}


def _parse(sql):
    if not isinstance(sql, str) or len(sql.encode("utf-8")) > 16000:
        return None
    try:
        parsed = sqlglot.parse(sql, read="sqlite")
    except (ParseError, TokenError, RecursionError):
        return None
    if len(parsed) != 1 or not isinstance(parsed[0], exp.Select):
        return None
    tree = parsed[0]
    if len(list(tree.find_all(exp.Select))) != 1 or any(
            tree.args.get(key) for key in ("with", "group", "having", "qualify", "limit", "offset", "into")):
        return None
    if any(isinstance(node, (exp.Subquery, exp.Window, exp.Command, exp.SetOperation)) for node in tree.walk()):
        return None
    return tree


def _context(tree, schema):
    if not isinstance(schema, list):
        return None
    metadata = {}
    for item in schema:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            return None
        key = item["name"].casefold()
        if key in metadata:
            return None
        metadata[key] = item
    source = tree.args.get("from")
    tables = ([source.this] if source is not None else []) + [j.this for j in tree.args.get("joins", [])]
    if not tables or not all(isinstance(t, exp.Table) and not t.db and not t.catalog
                             and isinstance(t.this, exp.Identifier) for t in tables):
        return None
    aliases = {}
    for table in tables:
        alias, key = table.alias_or_name.casefold(), table.name.casefold()
        if alias in aliases or key not in metadata:
            return None
        aliases[alias] = (table, metadata[key])
    return aliases


def _binding(column, aliases):
    if not isinstance(column, exp.Column) or column.is_star or column.db or column.catalog:
        return None
    choices = []
    for alias, (table, metadata) in aliases.items():
        columns = {item["name"].casefold() for item in metadata.get("columns", [])
                   if isinstance(item, dict) and isinstance(item.get("name"), str)}
        if (not column.table or column.table.casefold() == alias) and column.name.casefold() in columns:
            choices.append((alias, table.name.casefold(), column.name.casefold()))
    return choices[0] if len(choices) == 1 else None


def _foreign_key(left, right, aliases):
    for source, target in ((left, right), (right, left)):
        for foreign in aliases[source[0]][1].get("foreign_keys", []):
            if (isinstance(foreign, dict)
                    and str(foreign.get("table", "")).casefold() == target[1]
                    and str(foreign.get("from", "")).casefold() == source[2]
                    and str(foreign.get("to", "")).casefold() == target[2]):
                return True
    return False


def _aggregate_fanout(tree, aliases, question):
    if tree.args.get("where") or tree.args.get("order") or tree.args.get("distinct"):
        return None
    if len(tree.expressions) != 1:
        return None
    aggregate = tree.expressions[0].unalias()
    if not isinstance(aggregate, (exp.Sum, exp.Avg)) or not isinstance(aggregate.this, exp.Column):
        return None
    source = _binding(aggregate.this, aliases)
    base = tree.args["from"].this.alias_or_name.casefold()
    if source is None or source[0] != base:
        return None
    question_words = _words(question)
    base_words = _words(aliases[base][0].name) - TABLE_GENERIC
    measure_words = _words(aggregate.this.name) - {"id", "num", "number", "of", "total"}
    if not base_words & question_words or not measure_words & question_words:
        return None
    if not question_words & {"sum", "total", "average", "mean"}:
        return None
    # These cues can request an existence restriction represented by the JOIN.
    if re.search(r"\b(?:with|without|whose|having|related|associated|linked|for)\b", question, re.I):
        return None
    joins = tree.args.get("joins", [])
    if not joins or any(j.args.get("side") or j.args.get("method") or j.args.get("using")
                        or j.args.get("kind", "").upper() not in ("", "INNER") for j in joins):
        return None
    kept = list(joins)
    for join in reversed(joins):
        alias = join.this.alias_or_name.casefold()
        meaningful = _words(join.this.name) - TABLE_GENERIC
        if (not meaningful or meaningful & question_words
                or any(len(word) >= 4 and any(token.startswith(word[:4]) for token in question_words)
                       for word in meaningful)):
            continue
        on = join.args.get("on")
        if not isinstance(on, exp.EQ):
            continue
        left, right = _binding(on.this, aliases), _binding(on.expression, aliases)
        if left is None or right is None or left[0] == right[0] or alias not in {left[0], right[0]}:
            continue
        if not _foreign_key(left, right, aliases):
            continue
        # Resolve references rather than relying on spelling, including bare
        # columns in later ON clauses. Unknown binding keeps the JOIN intact.
        outside = [column for other in kept if other is not join for column in other.find_all(exp.Column)]
        bindings = [_binding(column, aliases) for column in outside]
        if any(item is None or item[0] == alias for item in bindings):
            continue
        kept.remove(join)
    if len(kept) == len(joins):
        return None
    changed = tree.copy()
    changed.set("joins", [join.copy() for join in kept] or None)
    return changed


def _entity_word_filter(tree, aliases, question):
    if tree.args.get("joins") or tree.args.get("order") or len(aliases) != 1:
        return None
    if not re.search(r"\ball\b", question, re.I):
        return None
    if not all(isinstance(item.unalias(), exp.Column) and _binding(item.unalias(), aliases)
               for item in tree.expressions):
        return None
    where = tree.args.get("where")
    if where is None or not isinstance(where.this, exp.EQ):
        return None
    condition = where.this
    if isinstance(condition.this, exp.Column) and isinstance(condition.expression, exp.Literal):
        column, literal = condition.this, condition.expression
    elif isinstance(condition.expression, exp.Column) and isinstance(condition.this, exp.Literal):
        column, literal = condition.expression, condition.this
    else:
        return None
    if not literal.is_string or _binding(column, aliases) is None or not re.fullmatch(r"[A-Za-z][A-Za-z _-]{2,}", literal.this):
        return None
    table = next(iter(aliases.values()))[0]
    entity_words = _words(table.name) - TABLE_GENERIC
    value_words = _words(literal.this)
    question_words = _words(question)
    if not entity_words or not value_words or not value_words <= entity_words or not entity_words <= question_words:
        return None
    # The request must be a schema-described all-entity enumeration, without
    # unrecognized adjective/value words that could express a real restriction.
    projection_words = set().union(*(_words(item.unalias().name) for item in tree.expressions))
    allowed = entity_words | projection_words | {_stem(word) for word in REQUEST_WORDS}
    if question_words - allowed:
        return None
    changed = tree.copy()
    changed.set("where", None)
    return changed


def _proposals(sql, question, schema):
    if not isinstance(question, str):
        return []
    tree = _parse(sql)
    if tree is None:
        return []
    aliases = _context(tree, schema)
    if aliases is None:
        return []
    return [candidate for candidate in (_aggregate_fanout(tree, aliases, question),
                                       _entity_word_filter(tree, aliases, question)) if candidate is not None]


def simplification_repairs(sql, feedback, example, question, schema):
    """Yield schema-supported interpretations; never assert equivalence."""
    for candidate in _proposals(sql, question, schema):
        yield candidate.sql(dialect="sqlite")


def simplification_penalty(sql, example, question, schema=None):
    """Rank only concrete proposed repairs; missing schema supplies no evidence."""
    return len(_proposals(sql, question, schema))
