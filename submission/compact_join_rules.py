"""Try minimal two-table FK joins for grounded, flat query expressions."""
from collections import defaultdict
import re

from sqlglot import exp
from relation_rules import _equality, _graph, _schema, _select
from single_table_rules import _outputs, _words


def _bind(node, pair, aliases):
    node = node.copy()
    owners = set()
    for column in node.find_all(exp.Column):
        key = column.name.casefold()
        qualifier = column.table.casefold()
        if column.is_star or column.db or column.catalog:
            return None
        choices = [source for source, table in pair.items() if key in table["columns"]]
        preferred = aliases.get(qualifier, qualifier)
        if preferred in choices:
            choices = [preferred]
        if len(choices) != 1:
            return None
        source = choices[0]
        table = pair[source]
        column.set("this", exp.to_identifier(table["columns"][key], quoted=True))
        column.set("table", exp.to_identifier(table["name"], quoted=True))
        owners.add(source)
    return node, owners


def _queries(tree, outputs, metadata, edges, words):
    refs = {t.alias_or_name.casefold(): t.name.casefold() for t in tree.find_all(exp.Table)}
    before = set(refs.values())
    if not before or tree.args.get("offset") or tree.find(exp.Window):
        return
    clauses = {key:tree.args[key] for key in ("where", "group", "having", "order", "limit") if tree.args.get(key)}
    if any(isinstance(node, exp.Ordered) and isinstance(node.this, exp.Literal)
           for clause in clauses.values() for node in clause.walk()):
        return
    for left, right, lcol, rcol in edges:
        pair = {left: metadata[left], right: metadata[right]}
        sources = set(pair)
        if not before & sources:
            continue
        # For an existing two-relation join, the locked relation rules already
        # offer FK endpoint correction for the same two source tables.
        if before == sources:
            continue
        bound_nodes = [_bind(item, pair, refs) for item in [*outputs,*clauses.values()]]
        if any(item is None for item in bound_nodes):
            continue
        owners = set().union(*(entry[1] for entry in bound_nodes))
        if owners != sources:
            # A grouped COUNT(*) can be defined by rows on the other side of
            # the join when that table is explicitly named by the question.
            implicit = sources - owners
            if (len(implicit) != 1 or not tree.args.get("group") or not any(
                    isinstance(item, exp.Count) for item in outputs)
                    or not _words(metadata[next(iter(implicit))]["name"]) & words):
                continue
        nodes = [entry[0] for entry in bound_nodes]
        query = exp.Select(expressions=nodes[:len(outputs)]).from_(
            exp.Table(this=exp.to_identifier(metadata[left]["name"], quoted=True)))
        condition = _equality(metadata[left]["name"], metadata[left]["columns"][lcol],
                              metadata[right]["name"], metadata[right]["columns"][rcol])
        query.append("joins", exp.Join(this=exp.Table(this=exp.to_identifier(metadata[right]["name"], quoted=True)),
                                       on=condition))
        for key, clause in zip(clauses, nodes[len(outputs):]):
            query.set(key, clause)
        if tree.args.get("distinct"):
            query.set("distinct", tree.args["distinct"].copy())
        overlap = sum(len(_words(pair[key]["name"]) & words) / max(1,len(_words(pair[key]["name"]))) for key in pair)
        score = 2 * len(before & sources) + overlap + len(owners) + min(3,len(clauses))
        yield query.sql(dialect="sqlite"), score


def compact_join_repairs(history, example, question, schema):
    if not isinstance(question, str) or len(question) > 4096 or re.search(r"\b(?:without|except|either|both)\b",question,re.I):
        return
    outputs, metadata = _outputs(example.get("columns")), _schema(schema)
    if not metadata or not outputs or len(metadata) > 150:
        return
    edges = [(left, right, lcol, rcol) for left, paths in _graph(metadata).items()
             for right, lcol, rcol in paths if left < right]
    if len(edges) > 256:
        return
    words = _words(question)
    candidates = defaultdict(list)
    checked = set()
    for event in history:
        if event["tool"] != "check":
            continue
        checked.add(event["sql"])
        tree = _select(event["sql"])
        if tree is None:
            continue
        for sql, strength in _queries(tree, outputs, metadata, edges, words):
            if len(sql.encode("utf-8")) <= 16000:
                candidates[sql].append(strength)
    queries = sorted(set(candidates) - checked,
                     key=lambda sql:(-max(candidates[sql])-min(3,len(candidates[sql]))*.5,len(sql),sql))
    yield from queries[:6]
