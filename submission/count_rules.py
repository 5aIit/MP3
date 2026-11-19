"""Bounded counting interpretations grounded in schema and question wording.

Changing LEFT to INNER or grouping by an entity key is a repair proposal, not
an equivalence claim. Every emitted query requires an official check.
"""
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError


def _words(text):
    def stem(word):
        if word.endswith("ies"):
            return word[:-3] + "y"
        return word[:-1] if word.endswith("s") and not word.endswith("ss") else word
    return {stem(word) for word in re.findall(r"[a-z]+", str(text).casefold())}


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
    if (len(list(tree.find_all(exp.Select))) != 1 or tree.args.get("with")
            or tree.args.get("into") or any(isinstance(node, (exp.Subquery, exp.Window)) for node in tree.walk())
            or not any(tree.find_all(exp.Count))):
        return None
    return tree


def _context(tree, schema):
    if not isinstance(schema, list):
        return None
    metadata = {table["name"].casefold(): table for table in schema
                if isinstance(table, dict) and isinstance(table.get("name"), str)}
    base = tree.args.get("from")
    tables = ([base.this] if base is not None else []) + [join.this for join in tree.args.get("joins", [])]
    if not tables or any(not isinstance(table, exp.Table) or table.db or table.catalog
                         or table.name.casefold() not in metadata for table in tables):
        return None
    sources = {table.alias_or_name.casefold(): (table, metadata[table.name.casefold()]) for table in tables}
    return (metadata, sources) if len(sources) == len(tables) else None


def _target(question, metadata):
    if not re.search(r"\b(?:each|every|per)\b", question, re.I):
        return None
    cue = re.search(r"\b(?:how many|number of)\s+(.+?)(?=\b(?:are|were|is|was|have|has|for|per|each|every|related|that|which|do|does)\b|[?.]|$)", question, re.I)
    if cue is None:
        return None
    phrase = _words(cue.group(1)) - {"the", "a", "an", "total"}
    choices = [(len(_words(table["name"])), name) for name, table in metadata.items()
               if _words(table["name"]) and _words(table["name"]) <= phrase]
    if not choices:
        return None
    best = max(size for size, _ in choices)
    names = [name for size, name in choices if size == best]
    return names[0] if len(names) == 1 else None


def _binding(column, sources):
    if not isinstance(column, exp.Column) or column.is_star or column.db or column.catalog:
        return None
    choices = [(alias, table.name.casefold(), item["name"]) for alias, (table, meta) in sources.items()
               for item in meta.get("columns", []) if isinstance(item.get("name"), str)
               and item["name"].casefold() == column.name.casefold()
               and (not column.table or column.table.casefold() == alias)]
    return choices[0] if len(choices) == 1 else None


def _foreign_link(left, right, metadata):
    if left is None or right is None or left[0] == right[0]:
        return False
    for source, target in ((left, right), (right, left)):
        for foreign in metadata[source[1]].get("foreign_keys", []):
            if (str(foreign.get("table", "")).casefold() == target[1]
                    and str(foreign.get("from", "")).casefold() == source[2].casefold()
                    and str(foreign.get("to", "")).casefold() == target[2].casefold()):
                return True
    return False


def _inner_count(tree, question, metadata, sources):
    target = _target(question, metadata)
    if (target is None or not tree.args.get("group") or tree.args.get("where") or tree.args.get("having")
            or re.search(r"\b(?:zero|none|no|without|including|even|regardless)\b", question, re.I)):
        return None
    indices = []
    for index, join in enumerate(tree.args.get("joins", [])):
        side = join.args.get("side", "").upper()
        if side and side != "LEFT":
            return None
        if join.args.get("method") or join.args.get("using"):
            return None
        if side != "LEFT" or join.this.name.casefold() != target:
            continue
        condition = join.args.get("on")
        if not isinstance(condition, exp.EQ):
            return None
        left, right = _binding(condition.this, sources), _binding(condition.expression, sources)
        if (not _foreign_link(left, right, metadata)
                or join.this.alias_or_name.casefold() not in {left[0], right[0]}):
            return None
        indices.append(index)
    if len(indices) != 1:
        return None
    changed = tree.copy()
    join = changed.args["joins"][indices[0]]
    join.set("side", None)
    join.set("kind", "INNER")
    return changed


def _entity_frequency(tree, question, metadata, sources):
    if (not re.search(r"\bmost (?:common|frequent)\b", question, re.I)
            or not tree.args.get("group") or len(tree.args["group"].expressions) != 1):
        return None
    group = tree.args["group"].expressions[0]
    binding = _binding(group, sources)
    if binding is None or not _words(binding[2]) & {"name", "title", "label"}:
        return None
    alias, table_name, name = binding
    table, meta = sources[alias]
    if not _words(table.name) <= _words(question):
        return None
    keys = [column["name"] for column in meta.get("columns", []) if column.get("pk")]
    if len(keys) != 1 or keys[0].casefold() == name.casefold():
        return None
    # The name must be the sole nonaggregate output, avoiding SQLite bare-value
    # ambiguities for other selected fields or another entity's labels.
    plain = [item.unalias() for item in tree.expressions if not isinstance(item.unalias(), exp.Count)]
    if len(plain) != 1 or _binding(plain[0], sources) != binding:
        return None
    linked = False
    for join in tree.args.get("joins", []):
        if join.args.get("side") or join.args.get("method") or join.args.get("using"):
            return None
        condition = join.args.get("on")
        if not isinstance(condition, exp.EQ):
            return None
        left, right = _binding(condition.this, sources), _binding(condition.expression, sources)
        if _foreign_link(left, right, metadata):
            linked |= any(endpoint is not None and endpoint[0] == alias
                          and endpoint[2].casefold() == keys[0].casefold() for endpoint in (left, right))
    if not linked:
        return None
    changed = tree.copy()
    changed.set("group", exp.Group(expressions=[exp.column(keys[0], table=table.alias_or_name, quoted=True)]))
    return changed


def _inspect(sql, question, schema):
    if not isinstance(question, str):
        return [], 0
    tree = _parse(sql)
    context = _context(tree, schema) if tree is not None else None
    if context is None:
        return [], 0
    metadata, sources = context
    target = _target(question, metadata)
    missing = target is not None and all(table.name.casefold() != target for table, _ in sources.values())
    candidates = [item for item in (_inner_count(tree, question, metadata, sources),
                                    _entity_frequency(tree, question, metadata, sources)) if item is not None]
    return candidates, 2 * int(missing) + len(candidates)


def count_repairs(sql, feedback, example, question, schema):
    """Yield at most two schema-supported, check-required counting alternatives."""
    candidates, _ = _inspect(sql, question, schema)
    for candidate in candidates:
        text = candidate.sql(dialect="sqlite")
        if len(text.encode("utf-8")) <= 16000:
            yield text


def count_penalty(sql, example, question, schema=None):
    """Evidence for inspection/ranking, never a substitute for successful check."""
    return _inspect(sql, question, schema)[1]
