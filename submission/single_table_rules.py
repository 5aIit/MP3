"""Reconstruct single-table candidates using only schema and observed drafts."""
from collections import defaultdict
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from relation_rules import _schema, _select


def _words(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    words = re.findall(r"[a-z0-9]+", text.lower())
    def stem(word):
        if word.endswith("ies"):
            return word[:-3] + "y"
        if word.endswith("ses") and word != "uses":
            return word[:-2]
        if word.endswith("s") and len(word) > 2 and not word.endswith(("ss", "us")):
            return word[:-1]
        return word
    return {stem(word) for word in words}


def _outputs(labels):
    if not isinstance(labels, list) or not 1 <= len(labels) <= 8:
        return None
    result = []
    for label in labels:
        if not isinstance(label, str) or len(label) > 1000:
            return None
        try:
            parsed = sqlglot.parse(label, read="sqlite")
        except (ParseError, TokenError, RecursionError):
            return None
        if len(parsed) != 1:
            return None
        item = parsed[0]
        if isinstance(item, exp.Column) and not item.is_star and not item.db and not item.catalog:
            result.append(item)
        elif isinstance(item, (exp.Count, exp.Min, exp.Max, exp.Avg, exp.Sum)):
            if isinstance(item.this, exp.Star) and isinstance(item, exp.Count) and not any(item.this.args.values()):
                result.append(item)
            elif isinstance(item.this, exp.Column) and not item.this.is_star and not item.this.db and not item.this.catalog:
                result.append(item)
            else:
                return None
            if any(value for key, value in item.args.items() if key != "this"):
                # SQLGlot can set boolean hints such as big_int on COUNT.
                if set(key for key, value in item.args.items() if key != "this" and value) - {"big_int"}:
                    return None
        else:
            return None
    return result


def _bind(clause, table):
    """Require all columns of the clause to exist in the proposed source."""
    node = clause.copy()
    for column in node.find_all(exp.Column):
        name = table["columns"].get(column.name.casefold())
        if not name or column.db or column.catalog or column.is_star:
            return None
        column.set("this", exp.to_identifier(name, quoted=True))
        column.set("table", None)
    return node


def _syntheses(tree, outputs, metadata, question):
    if tree.args.get("offset") or tree.args.get("qualify") or tree.find(exp.Window):
        return
    if any(not isinstance(table.this, exp.Identifier) for table in tree.find_all(exp.Table)):
        return
    tables = {table.name.casefold() for table in tree.find_all(exp.Table)}
    if not tables:
        return
    clauses = {name: tree.args[name] for name in ("where", "having", "group", "order", "limit")
               if tree.args.get(name)}
    for key, table in metadata.items():
        # COUNT(*) alone contains no evidence for moving to an unrelated table.
        if key not in tables and not any(list(item.find_all(exp.Column)) for item in outputs):
            continue
        items = [_bind(item, table) for item in outputs]
        bound = {name: _bind(clause, table) for name, clause in clauses.items()}
        if any(item is None for item in [*items, *bound.values()]):
            continue
        names = _words(table["name"])
        named = len(names & question) / max(len(names), 1)
        # A non-named source absent from every SQL clause needs two independent
        # references in order to justify replacing the FROM clause.
        referenced = {c.name.casefold() for expr in [*items, *bound.values()] for c in expr.find_all(exp.Column)}
        if not referenced or (key not in tables and not named and len(referenced) < 2):
            continue
        expected = exp.Select(expressions=items).from_(exp.Table(this=exp.to_identifier(table["name"], quoted=True)))
        for name, clause in bound.items():
            expected.set(name, clause)
        if tree.args.get("distinct"):
            expected.set("distinct", tree.args["distinct"].copy())
        grouped = expected.args.get("group")
        aggregate = any(isinstance(item, exp.AggFunc) for item in items)
        plain = [item for item in items if isinstance(item, exp.Column)]
        if plain and aggregate and not grouped:
            if not question & {"each", "every", "per"}:
                continue
            expected.set("group", exp.Group(expressions=[item.copy() for item in plain]))
        # Exact agreement on output labels and WHERE column names is significant
        # even if an original qualifier referred to an unrelated source.
        rank = 2 * named + (key in tables) + min(3, len(referenced)) * 0.5
        if bound.get("where") is not None:
            condition = bound["where"]
            grounded = [lit for lit in condition.find_all(exp.Literal)
                        if _words(lit.this) and _words(lit.this) <= question]
            rank += min(2, len(grounded))
        yield expected, rank


def _identity(tree):
    tables = list(tree.find_all(exp.Table))
    if len(tables) != 1:
        return None
    tree = tree.copy()
    # Ignore simple output alias spelling only when no clause references it.
    aliases = {item.alias.casefold() for item in tree.expressions if item.alias}
    for key in ("where", "group", "order", "having"):
        clause = tree.args.get(key)
        if clause and any(column.name.casefold() in aliases for column in clause.find_all(exp.Column)):
            return None
    tree.set("expressions", [item.unalias().copy() for item in tree.expressions])
    table = tree.find(exp.Table)
    qualifiers = {table.alias_or_name.casefold(), table.name.casefold()}
    table.set("alias", None)
    for column in tree.find_all(exp.Column):
        if column.table and column.table.casefold() not in qualifiers:
            return None
        column.set("table", None)
    for identifier in tree.find_all(exp.Identifier):
        identifier.set("quoted", False)
    return tree.sql(dialect="sqlite", normalize=True, comments=False)


def single_table_repairs(history, example, question, schema):
    """Pool corroborating source reconstructions from multiple observed drafts."""
    if not isinstance(question, str) or len(question) > 4096:
        return
    outputs = _outputs(example.get("columns"))
    metadata = _schema(schema)
    if not outputs or not metadata or len(metadata) > 150:
        return
    # A set-theoretic question needs more than a single flat SELECT.
    if re.search(r"\b(?:except|without|intersect|either|both)\b", question, re.I):
        return
    question_words = _words(question)
    candidates = defaultdict(list)
    checked = set()
    identities = {}
    for event in history:
        if event.get("tool") != "check":
            continue
        tree = _select(event["sql"])
        if tree is None:
            continue
        if event.get("observation", {}).get("ok"):
            checked.add(_identity(tree))
        for candidate, strength in _syntheses(tree, outputs, metadata, question_words):
            sql = candidate.sql(dialect="sqlite")
            if len(sql.encode("utf-8")) <= 16000:
                candidates[sql].append(strength)
                identities[sql] = _identity(candidate)
    ranked = sorted(candidates, key=lambda sql: (-max(candidates[sql]) - min(3, len(candidates[sql])) * 0.5,
                                                    len(sql), sql))
    yield from [sql for sql in ranked if sql not in {e.get("sql") for e in history}
                and identities[sql] not in checked][:6]
