"""Bind an explicitly named entity to its own name instead of a class name.

These schema-grounded interpretations are candidates, not equivalence claims.
There is no I/O, and every proposed SQL must be checked by the controller.
"""
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError


CLASS_WORDS = {"type", "category", "class", "kind", "classification", "style", "status"}


def _tokens(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return [word[:-3] + "y" if word.endswith("ies") else
            word[:-1] if word.endswith("s") and not word.endswith("ss") else word
            for word in words]


def _contains(sequence, fragment):
    return bool(fragment) and any(sequence[i:i + len(fragment)] == fragment
                                  for i in range(len(sequence) - len(fragment) + 1))


def _named(question, table, literal):
    words, entity, value = _tokens(question), _tokens(table), _tokens(literal)
    if not entity or not value or set(entity) & CLASS_WORDS:
        return False
    return any(_contains(words, entity + link + value) for link in ([], ["named"], ["called"]))


def _parse(sql):
    if not isinstance(sql, str) or len(sql.encode("utf-8")) > 16000:
        return None
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except (ParseError, TokenError, RecursionError):
        return None
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        return None
    tree = statements[0]
    if (len(list(tree.find_all(exp.Select))) != 1
            or any(tree.args.get(key) for key in ("with", "into", "group", "having", "qualify"))
            or any(not isinstance(item.unalias(), exp.Column) for item in tree.expressions)
            or any(isinstance(node, (exp.Subquery, exp.Window, exp.SetOperation)) for node in tree.walk())):
        return None
    return tree


def _context(tree, schema):
    if not isinstance(schema, list):
        return None
    metadata = {}
    for table in schema:
        if not isinstance(table, dict) or not isinstance(table.get("name"), str):
            return None
        key = table["name"].casefold()
        if key in metadata:
            return None
        metadata[key] = table
    base = tree.args.get("from")
    tables = ([base.this] if base is not None else []) + [join.this for join in tree.args.get("joins", [])]
    sources = {}
    for table in tables:
        if (not isinstance(table, exp.Table) or table.db or table.catalog
                or table.name.casefold() not in metadata or table.alias_or_name.casefold() in sources):
            return None
        columns = {column["name"].casefold(): column["name"]
                   for column in metadata[table.name.casefold()].get("columns", [])
                   if isinstance(column, dict) and isinstance(column.get("name"), str)}
        sources[table.alias_or_name.casefold()] = (table.alias_or_name, table.name, columns)
    return sources or None


def _binding(column, sources):
    if not isinstance(column, exp.Column) or column.is_star or column.db or column.catalog:
        return None
    choices = [alias for alias, (_, _, columns) in sources.items()
               if (not column.table or column.table.casefold() == alias)
               and column.name.casefold() in columns]
    return choices[0] if len(choices) == 1 else None


def _prune_class_joins(tree, sources, question):
    """Drop only unused class-table leaf joins; retain all other relationships."""
    used = set()
    for key, clause in tree.args.items():
        if key in ("from", "joins"):
            continue
        for child in clause if isinstance(clause, list) else [clause]:
            if not isinstance(child, exp.Expression):
                continue
            for column in child.find_all(exp.Column):
                alias = _binding(column, sources)
                if alias is None:
                    return tree
                used.add(alias)
    kept = []
    question_words = set(_tokens(question))
    for join in reversed(tree.args.get("joins", [])):
        alias = join.this.alias_or_name.casefold()
        words = set(_tokens(sources[alias][1]))
        classes = words & CLASS_WORDS
        pure = (not join.args.get("side") and not join.args.get("method")
                and not join.args.get("using") and join.args.get("kind", "").upper() in ("", "INNER")
                and isinstance(join.args.get("on"), exp.EQ))
        if alias not in used and classes and not classes & question_words and pure:
            continue
        kept.append(join.copy())
        for column in join.find_all(exp.Column):
            binding = _binding(column, sources)
            if binding is None:
                return tree
            used.add(binding)
    changed = tree.copy()
    changed.set("joins", list(reversed(kept)))
    return changed


def _proposals(sql, question, schema):
    if not isinstance(question, str) or len(question) > 20000:
        return []
    tree = _parse(sql)
    sources = _context(tree, schema) if tree is not None else None
    if sources is None or not tree.args.get("where"):
        return []
    where = tree.args["where"]
    if any(isinstance(node, (exp.Or, exp.Not)) for node in where.walk()):
        return []
    proposals = []
    for condition in where.find_all(exp.EQ):
        column, literal = condition.this, condition.expression
        if isinstance(column, exp.Literal):
            column, literal = literal, column
        if not isinstance(literal, exp.Literal) or not literal.is_string:
            continue
        source_alias = _binding(column, sources)
        if source_alias is None:
            continue
        source = sources[source_alias]
        if not (set(_tokens(source[1])) & CLASS_WORDS and set(_tokens(column.name)) & {"name", "title", "label"}):
            continue
        targets = []
        for alias, (spelling, table, columns) in sources.items():
            if alias == source_alias or not _named(question, table, literal.this):
                continue
            names = [name for name in columns.values() if set(_tokens(name)) & {"name", "title"}
                     and not set(_tokens(name)) & CLASS_WORDS]
            if len(names) == 1:
                targets.append((spelling, names[0]))
        if len(targets) != 1:
            continue
        changed = tree.copy()
        target_index = next(index for index, node in enumerate(tree.walk()) if node is condition)
        target = list(changed.walk())[target_index]
        alias, name = targets[0]
        target.set("this", exp.column(name, table=alias, quoted=True))
        target.set("expression", literal.copy())
        proposals.append(_prune_class_joins(changed, sources, question))
    return proposals


def entity_repairs(sql, feedback, example, question, schema):
    """Yield at most two explicitly grounded name-binding proposals."""
    for candidate in _proposals(sql, question, schema)[:2]:
        text = candidate.sql(dialect="sqlite")
        if len(text.encode("utf-8")) <= 16000:
            yield text


def entity_penalty(sql, example, question, schema=None):
    return min(len(_proposals(sql, question, schema)), 2) * 2
