"""Conservative projection candidates derived from public column metadata.

These functions propose queries, never execute them or treat column names as
proof of correctness. The controller must check every proposed query.
"""
from collections import Counter
from itertools import permutations

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError


def _select(sql):
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except (ParseError, TokenError, RecursionError, TypeError):
        return None
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        return None
    tree = statements[0]
    # A wildcard's expanded output cannot be mapped without schema information.
    if any(item.is_star or isinstance(item.unalias(), exp.Star)
           for item in tree.expressions):
        return None
    return tree


def _label(value):
    return value.strip().casefold() if isinstance(value, str) else ""


def _expression_key(value):
    """Compare expressions across harmless whitespace and qualifier differences."""
    try:
        tree = sqlglot.parse_one(value, read="sqlite")
    except (ParseError, TokenError, RecursionError, TypeError):
        return None
    if tree is None or isinstance(tree, (exp.Query, exp.Command)):
        return None
    tree = tree.copy()
    for column in tree.find_all(exp.Column):
        column.set("table", None)
        column.set("db", None)
        column.set("catalog", None)
    # normalize changes identifier case, but preserves string literal contents.
    return tree.sql(dialect="sqlite", normalize=True, comments=False)


def _mapping(tree, expected, actual=()):
    """Map each expected output to exactly one existing projection expression."""
    if not expected or len(expected) > len(tree.expressions):
        return None
    if any(not isinstance(name, str) or not name.strip() for name in expected):
        return None
    sources = []
    for index, item in enumerate(tree.expressions):
        raw = item.unalias()
        names = {_label(item.alias), _label(raw.sql(dialect="sqlite"))}
        if index < len(actual):
            names.add(_label(actual[index]))
        # Keep qualified spelling as well as bare names, so a fully qualified
        # expected label can resolve two otherwise ambiguous columns.
        bare = _label(raw.name) if isinstance(raw, exp.Column) else ""
        sources.append((names - {""}, bare, _expression_key(raw.sql(dialect="sqlite"))))
    mapping = []
    for name in expected:
        label, expression_key = _label(name), _expression_key(name)
        scores = []
        for direct, bare, key in sources:
            scores.append(3 if label in direct else 2 if label == bare else
                          1 if expression_key is not None and expression_key == key else 0)
        best = max(scores, default=0)
        indices = [index for index, score in enumerate(scores) if score == best]
        if not best or len(indices) != 1 or indices[0] in mapping:
            return None
        mapping.append(indices[0])
    return mapping


def _safe_to_project(tree, mapping):
    # Removing an aggregate can change cardinality or the row supplying bare
    # columns (SQLite's MIN/MAX rule), even when a one-row example still fits.
    # Permit removal only when an explicit GROUP BY still defines the rows and
    # every retained item is a group key or direct aggregate. Subquery/window
    # aggregates are not responsible for grouping the outer SELECT.
    group = tree.args.get("group")
    grouped = {node.sql(dialect="sqlite", normalize=True, comments=False)
               for node in group.expressions} if group else set()
    retained_are_grouped = bool(grouped) and all(
        isinstance(tree.expressions[index].unalias(), exp.AggFunc)
        or tree.expressions[index].unalias().sql(
            dialect="sqlite", normalize=True, comments=False) in grouped
        for index in mapping)
    for index, item in enumerate(tree.expressions):
        if index not in mapping and not retained_are_grouped and any(
                isinstance(node, exp.AggFunc)
                and node.find_ancestor(exp.Select) is tree
                and node.find_ancestor(exp.Window) is None for node in item.walk()):
            return False
    # Reordering/removing SELECT expressions changes positional ORDER/GROUP BY.
    # Do not touch those queries, including positional expressions in subqueries.
    for clause in (tree.args.get("order"), tree.args.get("group")):
        if clause and any(isinstance(node, exp.Literal) and not node.is_string
                          for node in clause.walk()):
            return False
    removed_aliases = {item.alias.casefold() for index, item in enumerate(tree.expressions)
                       if index not in mapping and item.alias}
    if removed_aliases:
        for key, clause in tree.args.items():
            if key == "expressions":
                continue
            children = clause if isinstance(clause, list) else [clause]
            for child in children:
                if isinstance(child, exp.Expression) and any(
                        not column.table and column.name.casefold() in removed_aliases
                        for column in child.find_all(exp.Column)):
                    return False
    return True


def _expand_removed_order_aliases(tree, mapping):
    """Expand only direct outer ORDER BY aliases with unambiguous bindings.

    SQLite resolves a bare ORDER BY alias against the output columns, while an
    alias embedded in an expression can bind to an input column instead. Keep
    those expressions, nested scopes, and references in other clauses outside
    this rule. The existing safety check still validates the resulting tree.
    """
    removed = {item.alias.casefold(): item.unalias()
               for index, item in enumerate(tree.expressions)
               if index not in mapping and item.alias}
    order = tree.args.get("order")
    if not removed or order is None:
        return tree
    references = [column for column in order.find_all(exp.Column)
                  if not column.table and column.name.casefold() in removed]
    if not references:
        return tree

    output_names = Counter(item.alias_or_name.casefold() for item in tree.expressions
                           if item.alias_or_name)
    explicit_aliases = {item.alias.casefold() for item in tree.expressions if item.alias}
    for column in references:
        name = column.name.casefold()
        if column.find_ancestor(exp.Select) is not tree or output_names[name] != 1:
            return None
        term = column.parent
        if isinstance(term, exp.Collate) and term.this is column:
            term = term.parent
        if not isinstance(term, exp.Ordered) or term.parent is not order:
            return None
        source = removed[name]
        # Restrict expansion to columns and ordinary aggregates. In particular,
        # avoid re-evaluating volatile functions, windows, and scalar subqueries.
        if not isinstance(source, (exp.Column, exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)):
            return None
        if any(isinstance(node, (exp.Query, exp.Window))
               or (node is not source and isinstance(node, exp.Func))
               for node in source.walk()):
            return None
        # An expanded column must not newly resolve to another SELECT alias.
        if any(not node.table and node.name.casefold() in explicit_aliases
               for node in source.find_all(exp.Column)):
            return None

    changed = tree.copy()
    for column in list(changed.args["order"].find_all(exp.Column)):
        if not column.table and column.name.casefold() in removed:
            column.replace(removed[column.name.casefold()].copy())
    return changed


def _value_mapping(tree, feedback, example):
    """Use complete, nonempty rows only when exactly one projection fits.

    This is a bounded proposal mechanism, not semantic evidence for selecting
    between public-example matches. The controller still checks its output.
    """
    actual_count, expected_count = feedback.get("row_count"), example.get("row_count")
    actual_rows, expected_rows = feedback.get("rows", []), example.get("rows", [])
    actual_width, expected_width = len(tree.expressions), len(example.get("columns", []))
    if (not isinstance(actual_count, int) or actual_count <= 0
            or actual_count != expected_count
            or feedback.get("truncated") or example.get("truncated")
            or len(actual_rows) != actual_count or len(expected_rows) != expected_count
            or not 1 <= expected_width <= actual_width <= 5):
        return None
    if (any(not isinstance(row, (list, tuple)) or len(row) != actual_width for row in actual_rows)
            or any(not isinstance(row, (list, tuple)) or len(row) != expected_width for row in expected_rows)):
        return None

    def value_key(value):
        if value is None:
            return ("null", None)
        if isinstance(value, (int, float)):
            return ("number", round(value, 6))
        if isinstance(value, str):
            return ("text", value)
        raise TypeError("Not a scalar SQL result")

    try:
        actual = [[value_key(value) for value in row] for row in actual_rows]
        expected = Counter(tuple(value_key(value) for value in row) for row in expected_rows)
    except (TypeError, ValueError, OverflowError):
        return None
    found = None
    for mapping in permutations(range(actual_width), expected_width):
        observed = Counter(tuple(row[index] for index in mapping) for row in actual)
        if observed == expected:
            if found is not None:
                return None
            found = list(mapping)
    return found


def projection_repairs(sql, feedback, example, *, expand_order_aliases=False):
    """Yield at most one unambiguous reordering/subset of existing SELECT items."""
    if not feedback.get("ok"):
        return
    tree = _select(sql)
    if tree is None:
        return
    actual = feedback.get("columns", [])
    if len(actual) != len(tree.expressions):
        return
    mapping = _mapping(tree, example.get("columns", []), actual)
    if mapping is None:
        mapping = _value_mapping(tree, feedback, example)
    if mapping is None or mapping == list(range(len(tree.expressions))):
        return
    if expand_order_aliases:
        tree = _expand_removed_order_aliases(tree, mapping)
        if tree is None:
            return
    if not _safe_to_project(tree, mapping):
        return
    tree.set("expressions", [tree.expressions[index].copy() for index in mapping])
    yield tree.sql(dialect="sqlite")


def projection_penalty(sql, example, *, expand_order_aliases=False):
    """Return a small penalty only for an unambiguous, repairable mismatch.

    Unknown or ambiguous column metadata produces zero, not a guessed mismatch.
    This score must be subordinate to real execution/example feedback.
    """
    tree = _select(sql)
    if tree is None:
        return 0
    mapping = _mapping(tree, example.get("columns", []))
    if mapping is None:
        return 0
    if expand_order_aliases:
        tree = _expand_removed_order_aliases(tree, mapping)
        if tree is None:
            return 0
    if not _safe_to_project(tree, mapping):
        return 0
    return len(tree.expressions) - len(mapping) + sum(index != target
                                                   for target, index in enumerate(mapping))
