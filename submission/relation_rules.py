"""Bounded relation repairs grounded in visible schema and foreign keys.

These are candidates, not guaranteed answers. There is no benchmark access or
SQL execution; the controller must check every emitted SQL with official tools.
"""
from collections import deque

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError


def _select(sql):
    if not isinstance(sql, str) or len(sql.encode("utf-8")) > 16000:
        return None
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except (ParseError, TokenError, RecursionError):
        return None
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        return None
    tree = statements[0]
    if tree.args.get("with") or len(list(tree.find_all(exp.Select))) != 1:
        return None
    if any(isinstance(node, (exp.Subquery, exp.SetOperation, exp.Into)) for node in tree.walk()):
        return None
    return tree


def _schema(schema):
    if not isinstance(schema, list):
        return None
    result = {}
    for table in schema:
        if not isinstance(table, dict) or not isinstance(table.get("name"), str):
            return None
        key = table["name"].casefold()
        if key in result:
            return None
        columns = {column["name"].casefold(): column["name"]
                   for column in table.get("columns", [])
                   if isinstance(column, dict) and isinstance(column.get("name"), str)}
        result[key] = {"name": table["name"], "columns": columns,
                       "foreign_keys": table.get("foreign_keys", [])}
    return result


def _sources(tree, metadata):
    from_clause = tree.args.get("from")
    tables = ([from_clause.this] if from_clause is not None else []) + [
        join.this for join in tree.args.get("joins", [])]
    if not tables or not all(isinstance(table, exp.Table) for table in tables):
        return None
    result = {}
    for table in tables:
        if table.db or table.catalog or not isinstance(table.this, exp.Identifier):
            return None
        key, alias = table.name.casefold(), table.alias_or_name.casefold()
        if key not in metadata or alias in result:
            return None
        result[alias] = {"table": key, "alias": table.alias_or_name}
    return result


def _graph(metadata):
    """Keep parallel FK alternatives distinct so ambiguous paths are rejected."""
    graph = {key: [] for key in metadata}
    seen = set()
    for source, table in metadata.items():
        for foreign in table["foreign_keys"]:
            if not isinstance(foreign, dict) or not all(isinstance(foreign.get(key), str)
                                                       for key in ("table", "from", "to")):
                continue
            target = foreign["table"].casefold()
            left, right = foreign["from"].casefold(), foreign["to"].casefold()
            if (target not in metadata or left not in table["columns"]
                    or right not in metadata[target]["columns"] or source == target):
                continue
            identity = tuple(sorted(((source, left), (target, right))))
            if identity in seen:
                continue
            seen.add(identity)
            graph[source].append((target, left, right))
            graph[target].append((source, right, left))
    return graph


def _unique_path(graph, source, target, max_edges=3):
    if source == target:
        return None
    queue = deque([(source, [], {source})])
    found, shortest = [], max_edges + 1
    while queue:
        node, path, visited = queue.popleft()
        if len(path) >= min(shortest, max_edges):
            continue
        for neighbor, left, right in graph.get(node, []):
            if neighbor in visited:
                continue
            current = path + [(node, neighbor, left, right)]
            if neighbor == target:
                if len(current) < shortest:
                    found, shortest = [current], len(current)
                elif len(current) == shortest:
                    found.append(current)
            elif len(current) < shortest:
                queue.append((neighbor, current, visited | {neighbor}))
    return found[0] if len(found) == 1 else None


def _rebind_columns(tree, sources, metadata):
    changed, count = tree.copy(), 0
    for column in changed.find_all(exp.Column):
        if column.is_star or column.db or column.catalog or not column.table:
            continue
        if column.find_ancestor(exp.Join) is not None:
            # Moving a broken ON endpoint onto its other endpoint creates a
            # tautology. Repair join relationships through the FK graph instead.
            continue
        alias, name = column.table.casefold(), column.name.casefold()
        current = sources.get(alias)
        if current and name in metadata[current["table"]]["columns"]:
            continue
        choices = [(value["alias"], metadata[value["table"]]["columns"][name])
                   for value in sources.values() if name in metadata[value["table"]]["columns"]]
        if len(choices) != 1:
            continue
        replacement_alias, replacement_name = choices[0]
        column.set("table", exp.to_identifier(replacement_alias, quoted=True))
        column.set("this", exp.to_identifier(replacement_name, quoted=True))
        count += 1
    return changed, count


def _reference(column, sources, metadata):
    if (not isinstance(column, exp.Column) or column.is_star
            or column.db or column.catalog):
        return None
    name = column.name.casefold()
    options = [(key, value["table"], name) for key, value in sources.items()
               if (not column.table or column.table.casefold() == key)
               and name in metadata[value["table"]]["columns"]]
    return options[0] if len(options) == 1 else None


def _join_endpoint(column, sources, metadata):
    known = _reference(column, sources, metadata)
    if known is not None:
        return known
    if isinstance(column, exp.Column) and column.table.casefold() in sources:
        alias = column.table.casefold()
        return alias, sources[alias]["table"], column.name.casefold()
    return None


def _qualify_equivalent_columns(tree, sources, metadata):
    joins = tree.args.get("joins", [])
    if any(join.args.get("side") for join in joins):
        return tree, 0
    links = {}
    for join in joins:
        condition = join.args.get("on")
        if not isinstance(condition, exp.EQ):
            continue
        left, right = (_reference(condition.this, sources, metadata),
                       _reference(condition.expression, sources, metadata))
        if left and right:
            a, b = (left[0], left[2]), (right[0], right[2])
            links.setdefault(a, set()).add(b)
            links.setdefault(b, set()).add(a)
    aliases = {item.alias.casefold() for item in tree.expressions if item.alias}
    changed, count = tree.copy(), 0
    for column in changed.find_all(exp.Column):
        if column.table or column.is_star or column.name.casefold() in aliases:
            continue
        name = column.name.casefold()
        choices = [(alias, name) for alias, item in sources.items()
                   if name in metadata[item["table"]]["columns"]]
        if len(choices) < 2:
            continue
        connected, pending = set(), [choices[0]]
        while pending:
            node = pending.pop()
            if node not in connected:
                connected.add(node)
                pending.extend(links.get(node, set()) - connected)
        if not set(choices) <= connected:
            continue
        alias = choices[0][0]
        column.set("table", exp.to_identifier(sources[alias]["alias"], quoted=True))
        count += 1
    return changed, count


def _equality(left_alias, left_name, right_alias, right_name):
    return exp.EQ(this=exp.column(left_name, table=left_alias, quoted=True),
                  expression=exp.column(right_name, table=right_alias, quoted=True))


def _repair_joins(tree, sources, metadata):
    joins = tree.args.get("joins", [])
    # Replacing an ON condition on an outer/natural/USING join can change which
    # rows survive. Keep those queries outside this bounded graph proposal.
    if any(join.args.get("side") or join.args.get("method") or join.args.get("using")
           or join.args.get("kind", "").upper() not in ("", "INNER") for join in joins):
        return tree, 0
    graph = _graph(metadata)
    original_tables = {value["table"] for value in sources.values()}
    first = tree.args["from"].this.alias_or_name.casefold()
    prefix = {first: sources[first]}
    new_joins, repaired, alias_number = [], 0, 0
    used_aliases = set(sources)
    for join in joins:
        target_alias = join.this.alias_or_name.casefold()
        scope = {**prefix, target_alias: sources[target_alias]}
        condition = join.args.get("on")
        binding = None
        if isinstance(condition, exp.EQ):
            left = _join_endpoint(condition.this, scope, metadata)
            right = _join_endpoint(condition.expression, scope, metadata)
            if left and right:
                if left[0] == target_alias and right[0] in prefix:
                    left, right = right, left
                if left[0] in prefix and right[0] == target_alias:
                    binding = left, right
        path = None
        if binding:
            left, right = binding
            # A valid declared relationship is not a repair target, including
            # FK direction reversed in the SQL equality.
            if (right[1], left[2], right[2]) not in graph.get(left[1], []):
                candidate = _unique_path(graph, left[1], right[1])
                if candidate and not any(edge[1] in original_tables for edge in candidate[:-1]):
                    path = candidate
        if not path:
            new_joins.append(join.copy())
        else:
            previous_alias = sources[binding[0][0]]["alias"]
            for index, (left_table, right_table, left_column, right_column) in enumerate(path):
                last = index == len(path) - 1
                if last:
                    entry = join.copy()
                    next_alias = sources[target_alias]["alias"]
                else:
                    while True:
                        alias_number += 1
                        next_alias = f"_repair_fk_{alias_number}"
                        if next_alias.casefold() not in used_aliases:
                            break
                    used_aliases.add(next_alias.casefold())
                    table = exp.Table(this=exp.to_identifier(metadata[right_table]["name"], quoted=True),
                                      alias=exp.TableAlias(this=exp.to_identifier(next_alias, quoted=True)))
                    entry = exp.Join(this=table)
                entry.set("on", _equality(previous_alias, metadata[left_table]["columns"][left_column],
                                          next_alias, metadata[right_table]["columns"][right_column]))
                new_joins.append(entry)
                previous_alias = next_alias
            repaired += 1
        prefix[target_alias] = sources[target_alias]
    if not repaired:
        return tree, 0
    changed = tree.copy()
    changed.set("joins", new_joins)
    return changed, repaired


def _propose(sql, schema):
    tree, metadata = _select(sql), _schema(schema)
    if tree is None or metadata is None:
        return None, 0, 0
    sources = _sources(tree, metadata)
    if sources is None:
        return None, 0, 0
    rebound, column_count = _rebind_columns(tree, sources, metadata)
    changed, join_count = _repair_joins(rebound, sources, metadata)
    expanded_sources = _sources(changed, metadata)
    if expanded_sources is not None:
        changed, ambiguity_count = _qualify_equivalent_columns(changed, expanded_sources, metadata)
        column_count += ambiguity_count
    return changed, column_count, join_count


def relation_repairs(sql, feedback, example, question, schema):
    """Propose one combined, unambiguous rebinding/FK-path repair.

    Execution failures can be repaired as well as executable non-FK joins.
    Question/example text is deliberately not interpolated into SQL.
    """
    changed, columns, joins = _propose(sql, schema)
    if changed is not None and (columns or joins):
        candidate = changed.sql(dialect="sqlite")
        if len(candidate.encode("utf-8")) <= 16000:
            yield candidate


def relation_penalty(sql, example, question, schema=None):
    """Small evidence score only when the schema supports a concrete repair."""
    _, columns, joins = _propose(sql, schema)
    return min(columns, 2) * 2 + min(joins, 2)
