"""Construct a projection from the observed column names and rebind its inputs."""
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from relation_rules import _graph, _schema, _select, _unique_path


def _projection(headers, metadata):
    columns = {name for table in metadata.values() for name in table["columns"]}
    allowed = (exp.Column, exp.Identifier, exp.Star, exp.Count, exp.Sum, exp.Avg,
               exp.Min, exp.Max, exp.Distinct, exp.Paren, exp.Add, exp.Sub,
               exp.Mul, exp.Div, exp.Literal)
    output = []
    for header in headers:
        if not isinstance(header, str) or not header or len(header) > 140:
            return None
        if header.casefold() in columns:
            expr = exp.column(header)
        else:
            try:
                parsed = sqlglot.parse(header, read="sqlite")
            except (ParseError, TokenError, RecursionError):
                return None
            if len(parsed) != 1:
                return None
            expr = parsed[0]
            if not isinstance(expr, exp.AggFunc) or any(not isinstance(node, allowed) for node in expr.walk()):
                return None
        for col in expr.find_all(exp.Column):
            if col.name.casefold() not in columns or col.is_star:
                return None
            col.set("table", None)
        output.append(expr)
    return output


def _words(text):
    parts = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower())
    return {word[:-3] + "y" if word.endswith("ies") and len(word) > 4
            else word.rstrip("s") for word in parts}


def _sql_table(name):
    return exp.Table(this=exp.to_identifier(name, quoted=True))


def _sql_col(name, table):
    return exp.column(name, table=table, quoted=True)


def _reconstruct(tree, projection, metadata, question, count_order=None):
    root = tree.args.get("from")
    tables = ([root.this] if root else []) + [j.this for j in tree.args.get("joins", [])]
    if not tables or any(not isinstance(t, exp.Table) for t in tables):
        return
    aliases = {t.alias_or_name.casefold(): t.name.casefold() for t in tables}
    if len(set(aliases.values())) != len(tables) or any(
            j.args.get("side") for j in tree.args.get("joins", [])):
        return
    used = list(dict.fromkeys(k for k in aliases.values() if k in metadata))
    selected = {}
    qwords = _words(question)
    for item in tree.expressions:
        for col in item.find_all(exp.Column):
            if col.table.casefold() in aliases:
                selected.setdefault(col.name.casefold(), set()).add(aliases[col.table.casefold()])
    rebuilt = exp.Select(expressions=[p.copy() for p in projection])
    for clause in ("where", "group", "having", "order", "limit", "offset", "distinct"):
        source = tree.args.get(clause)
        if source is not None:
            rebuilt.set(clause, source.copy())
    has_aggregate = any(p.find(exp.AggFunc) for p in projection)
    bare = [p for p in projection if isinstance(p, exp.Column)]
    if count_order:
        # Ordering by a count implies grouping even when that count is not a returned column.
        if not bare:
            return
        rebuilt.set("group", exp.Group(expressions=[p.copy() for p in bare]))
        rebuilt.set("order", exp.Order(expressions=[
            exp.Ordered(this=exp.Count(this=exp.Star()), desc=count_order[0])]))
        rebuilt.set("limit", exp.Limit(expression=exp.Literal.number(count_order[1])))
        if rebuilt.args.get("having") and rebuilt.args["having"].find(exp.Select):
            rebuilt.set("having", None)
    elif has_aggregate and bare and not rebuilt.args.get("group"):
        rebuilt.set("group", exp.Group(expressions=[p.copy() for p in bare]))
    # Replace projected aliases in ORDER/GROUP before discarding draft projections.
    projection_aliases = {item.alias.casefold(): item.unalias() for item in tree.expressions if item.alias}
    for clause in (rebuilt.args.get("order"), rebuilt.args.get("group"), rebuilt.args.get("having")):
        if clause:
            for col in list(clause.find_all(exp.Column)):
                replacement = projection_aliases.get(col.name.casefold())
                if replacement is not None and not col.table:
                    col.replace(replacement.copy())
    needs = []
    owners = {}
    for col in rebuilt.find_all(exp.Column):
        name = col.name.casefold()
        if col.is_star or col.db or col.catalog:
            return
        options = [k for k in metadata if name in metadata[k]["columns"]]
        if not options:
            return
        bound = aliases.get(col.table.casefold(), col.table.casefold())
        def rank(t):
            return (bool(bound) and bound == t,
                    t in selected.get(name, ()), t in used, t in needs,
                    len(_words(metadata[t]["name"]) & qwords),
                    -used.index(t) if t in used else -999)
        candidates = sorted(options, key=rank, reverse=True)
        if len(candidates) > 1 and rank(candidates[0]) == rank(candidates[1]):
            return
        owner = candidates[0]
        owners[id(col)] = owner
        if owner not in needs:
            needs.append(owner)
    if not needs:
        if len(used) != 1:
            return
        needs.append(used[0])
    # Scalar aggregate or unfiltered grouped queries can count joined child rows.
    # Preserve explicitly named source tables in the first candidate.
    extras = [k for k in used if k not in needs and _words(metadata[k]["name"]) <= qwords]
    sets = [needs + extras, needs] if extras else [needs]
    for wanted in sets:
        connected = {wanted[0]}
        edges = []
        graph = _graph(metadata)
        for target in wanted[1:]:
            if target in connected:
                continue
            paths = [path for s in sorted(connected) if (path := _unique_path(graph, s, target))]
            if not paths:
                break
            best = min(paths, key=len)
            for source, dest, left, right in best:
                if dest not in connected:
                    edges.append((source, dest, left, right))
                    connected.add(dest)
        if not set(wanted) <= connected:
            continue
        query = rebuilt.copy()
        original_cols = list(rebuilt.find_all(exp.Column))
        for col, old in zip(query.find_all(exp.Column), original_cols):
            table = metadata[owners[id(old)]]
            col.replace(_sql_col(table["columns"][old.name.casefold()], table["name"]))
        query.set("from", exp.From(this=_sql_table(metadata[wanted[0]]["name"])))
        joins = []
        for source, dest, left, right in edges:
            first, other = metadata[source], metadata[dest]
            cond = exp.EQ(this=_sql_col(first["columns"][left], first["name"]),
                          expression=_sql_col(other["columns"][right], other["name"]))
            joins.append(exp.Join(this=_sql_table(other["name"]), on=cond))
        query.set("joins", joins)
        yield query.sql(dialect="sqlite")


def header_candidates(history, example, question, schema):
    metadata = _schema(schema)
    if not metadata or not (0 < len(example.get("columns", [])) <= 10):
        return
    projection = _projection(example["columns"], metadata)
    if not projection:
        return
    trees = []
    seen = set()
    for event in history:
        if event.get("tool") != "check":
            continue
        tree = _select(event["sql"])
        if tree is not None:
            trees.append(tree)
    text = question.casefold()
    count_order = None
    if re.search(r"\b(most|fewest|least|largest|greatest|smallest|highest|lowest|maximum|minimum)\b", text):
        if (re.search(r"\b(number|many|how often|times|count|fewest)\b", text)
                or any(t.find(exp.Count) for t in trees)):
            limit = re.search(r"\btop\s+(\d+)\b", text)
            count_order = (not bool(re.search(r"\b(fewest|least|smallest|lowest|minimum)\b", text)),
                           int(limit[1]) if limit else 1)
    for order in ([count_order, None] if count_order else [None]):
        for tree in trees:
            for sql in _reconstruct(tree, projection, metadata, question, order):
                if sql not in seen:
                    seen.add(sql)
                    yield sql
