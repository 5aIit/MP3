"""Small, schema-grounded semantic proposals; every result still needs check.

No benchmark access or SQL execution occurs here. Public output labels are
evidence for bounded proposals, never instructions or proof of correctness.
"""
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError


def _expression(text, max_bytes=1000):
    if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > max_bytes:
        return None
    try:
        statements = sqlglot.parse(text, read="sqlite")
    except (ParseError, TokenError, RecursionError):
        return None
    return statements[0] if len(statements) == 1 else None


def _column(node):
    return isinstance(node, exp.Column) and not node.is_star and not node.db and not node.catalog


def _simple_select(sql):
    tree = _expression(sql, max_bytes=16000)
    if not isinstance(tree, exp.Select):
        return None
    if any(tree.args.get(key) for key in ("distinct", "group", "having", "with", "qualify", "offset")):
        return None
    if len(list(tree.find_all(exp.Select))) != 1:
        return None
    if any(isinstance(node, (exp.Subquery, exp.SetOperation, exp.Window, exp.Into))
           for node in tree.walk()):
        return None
    if not tree.expressions or not all(_column(item.unalias()) for item in tree.expressions):
        return None
    return tree


def _sources(tree, schema):
    if not isinstance(schema, list):
        return None
    metadata = {table["name"].casefold(): table for table in schema
                if isinstance(table, dict) and isinstance(table.get("name"), str)}
    from_clause = tree.args.get("from")
    tables = ([from_clause.this] if from_clause is not None else []) + [
        join.this for join in tree.args.get("joins", [])]
    if not tables or not all(isinstance(table, exp.Table) for table in tables):
        return None
    sources = []
    for table in tables:
        if table.db or table.catalog or not isinstance(table.this, exp.Identifier):
            return None
        record = metadata.get(table.name.casefold())
        if record is None:
            return None
        columns = {item["name"].casefold(): item["name"] for item in record.get("columns", [])
                   if isinstance(item, dict) and isinstance(item.get("name"), str)}
        sources.append((table.alias_or_name, columns))
    if not sources or len({alias.casefold() for alias, _ in sources}) != len(sources):
        return None
    return sources


def _resolve(column, sources):
    if not _column(column):
        return None
    matches = [(alias, columns[column.name.casefold()]) for alias, columns in sources
               if column.name.casefold() in columns
               and (not column.table or column.table.casefold() == alias.casefold())]
    if len(matches) != 1:
        return None
    alias, name = matches[0]
    return exp.column(name, table=alias, quoted=True)


def _key(column):
    return column.table.casefold(), column.name.casefold()


def _projected(tree, sources):
    projected = []
    for item in tree.expressions:
        resolved = _resolve(item.unalias(), sources)
        if resolved is None:
            return None
        projected.append((item, resolved))
    return projected


def _resolve_output(column, projected, sources):
    """Reuse one established output binding before looking across input tables.

    Public labels commonly omit qualifiers even when a joined query already
    chose one side of an identically named key. Distinct existing bindings
    remain ambiguous; an output alias never overrides a conflicting output.
    """
    if not _column(column):
        return None
    if column.table:
        return _resolve(column, sources)
    name = column.name.casefold()
    matches = {_key(resolved): resolved for item, resolved in projected
               if resolved.name.casefold() == name or item.alias.casefold() == name}
    if len(matches) > 1:
        return None
    if matches:
        return next(iter(matches.values())).copy()
    return _resolve(column, sources)


def _ordered_columns(tree, projected, sources):
    order = tree.args.get("order")
    result = []
    if order is None:
        return result
    aliases = {}
    for item, column in projected:
        if item.alias:
            aliases.setdefault(item.alias.casefold(), []).append(column)
    for ordered in order.expressions:
        node = ordered.this
        if not _column(node):
            return None
        candidates = aliases.get(node.name.casefold(), []) if not node.table else []
        if len(candidates) > 1:
            return None
        resolved = candidates[0] if candidates else _resolve(node, sources)
        if resolved is None:
            return None
        result.append((ordered, resolved))
    return result


def _complete_projection(tree, projected, sources, example):
    labels = example.get("columns", [])
    if not isinstance(labels, list) or not len(projected) < len(labels) <= 8:
        return None
    expected = [_expression(label) for label in labels]
    if not all(_column(node) for node in expected):
        return None
    if _ordered_columns(tree, projected, sources) is None:
        return None
    expressions, used = [], set()
    for node in expected:
        # Existing explicit output aliases can be reused without treating them
        # as database columns. All genuinely new columns require schema proof.
        aliases = [(index, item) for index, (item, _) in enumerate(projected)
                   if not node.table and item.alias and item.alias.casefold() == node.name.casefold()]
        if len(aliases) > 1:
            return None
        resolved = _resolve_output(node, projected, sources)
        if resolved is None:
            return None
        matches = [index for index, (_, column) in enumerate(projected) if _key(column) == _key(resolved)]
        if len(matches) > 1:
            return None
        if matches:
            index = matches[0]
            if index in used:
                return None
            used.add(index)
            expressions.append(projected[index][0].copy())
        else:
            if any(_key(node.unalias()) == _key(resolved) for node in expressions):
                return None
            expressions.append(resolved.copy())
    if len(used) != len(projected):
        return None
    changed = tree.copy()
    changed.set("expressions", expressions)
    return changed


def _sum_spec(example, question):
    if not isinstance(question, str) or not re.search(r"\b(?:total|sum|combined|altogether)\b", question, re.I):
        return None
    labels = example.get("columns", [])
    if not isinstance(labels, list) or not 1 <= len(labels) <= 8:
        return None
    desired = [_expression(label) for label in labels]
    sums = [node for node in desired if isinstance(node, exp.Sum) and _column(node.this)]
    if len(sums) != 1 or not all(_column(node) or node is sums[0] for node in desired):
        return None
    return desired, sums[0]


def _grouped_sum(tree, projected, sources, example, question):
    spec = _sum_spec(example, question)
    if spec is None:
        return None
    desired, requested_sum = spec
    operand = _resolve_output(requested_sum.this, projected, sources)
    if operand is None or _key(operand) not in {_key(column) for _, column in projected}:
        return None
    groups, expressions = [], []
    for item in desired:
        if item is requested_sum:
            expressions.append(exp.Sum(this=operand.copy()))
        else:
            column = _resolve_output(item, projected, sources)
            if column is None or _key(column) == _key(operand):
                return None
            groups.append(column)
            expressions.append(column.copy())
    group_keys = {_key(column) for column in groups}
    if len(group_keys) != len(groups):
        return None
    if not {_key(column) for _, column in projected} <= group_keys | {_key(operand)}:
        return None
    ordered = _ordered_columns(tree, projected, sources)
    if ordered is None or any(_key(column) not in group_keys | {_key(operand)} for _, column in ordered):
        return None
    # Output aliases used elsewhere could change WHERE/JOIN meaning on SQLite.
    aliases = {item.alias.casefold() for item, _ in projected if item.alias}
    for key, clause in tree.args.items():
        if key in ("expressions", "order"):
            continue
        for child in clause if isinstance(clause, list) else [clause]:
            if isinstance(child, exp.Expression) and any(
                    not column.table and column.name.casefold() in aliases
                    for column in child.find_all(exp.Column)):
                return None
    changed = tree.copy()
    changed.set("expressions", expressions)
    if groups:
        changed.set("group", exp.Group(expressions=[column.copy() for column in groups]))
    if ordered:
        changed_order = []
        for item, column in ordered:
            entry = item.copy()
            entry.set("this", exp.Sum(this=operand.copy())
                      if _key(column) == _key(operand) else column.copy())
            changed_order.append(entry)
        changed.set("order", exp.Order(expressions=changed_order))
    return changed


def semantic_repairs(sql, feedback, example, question, schema):
    """Yield at most two bounded proposals from public labels and visible schema."""
    if not isinstance(feedback, dict) or not feedback.get("ok") or not isinstance(example, dict):
        return
    tree = _simple_select(sql)
    if tree is None:
        return
    sources = _sources(tree, schema)
    if sources is None:
        return
    projected = _projected(tree, sources)
    if projected is None:
        return
    for candidate in (_complete_projection(tree, projected, sources, example),
                      _grouped_sum(tree, projected, sources, example, question)):
        if candidate is not None:
            yield candidate.sql(dialect="sqlite")


def semantic_penalty(sql, example, question):
    """Flag specific missing-column or total-versus-raw-value evidence.

    Unknown labels or ambiguous structure return zero. This is ranking evidence
    only, subordinate to an actual successful check of the candidate.
    """
    if not isinstance(example, dict):
        return 0
    tree, spec = _simple_select(sql), _sum_spec(example, question)
    if tree is None:
        return 0
    if spec is not None:
        _, requested_sum = spec
        name = requested_sum.this.name.casefold()
        matching = [item for item in tree.expressions
                    if name in {item.unalias().name.casefold(), item.alias.casefold()}]
        if len(matching) == 1:
            return 2
    labels = example.get("columns", [])
    if not isinstance(labels, list) or not len(tree.expressions) < len(labels) <= 8:
        return 0
    expected = [_expression(label) for label in labels]
    if not all(_column(node) for node in expected):
        return 0
    # Only a strict extension of individually identifiable current outputs
    # counts as missing-column evidence. Qualifier spelling is not a penalty.
    names = [node.name.casefold() for node in expected]
    if len(set(names)) != len(names):
        return 0
    matched = []
    for item in tree.expressions:
        possibilities = {item.unalias().name.casefold(), item.alias.casefold()} - {""}
        indices = [index for index, name in enumerate(names) if name in possibilities]
        if len(indices) != 1 or indices[0] in matched:
            return 0
        matched.append(indices[0])
    return 1
