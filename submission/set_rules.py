"""Repairs for collective requests, unions of sources and universal comparisons."""
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError
from relation_rules import _schema


def _tokens(text):
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", text.lower())}


def _union(columns, metadata, question):
    if not re.search(r"\band\b", question, re.I):
        return
    keys = [c.casefold() for c in columns]
    words = _tokens(question)
    sources = [t for t in metadata.values() if _tokens(t["name"]) <= words
               and all(k in t["columns"] for k in keys)]
    if len(sources) == 2:
        branches = [exp.select(*(exp.column(t["columns"][k], quoted=True) for k in keys))
                    .from_(exp.Table(this=exp.to_identifier(t["name"], quoted=True))) for t in sources]
        yield exp.Union(this=branches[0], expression=branches[1], distinct=True).sql(dialect="sqlite")


def _intersection(tree, columns, question):
    if not re.search(r"\bboth\b.+\band\b", question, re.I):
        return
    if any(tree.args.get(k) for k in ("order", "group", "limit", "having", "with")):
        return
    if len(tree.expressions) != len(columns) or any(
            not isinstance(e.unalias(), exp.Column) or e.unalias().name.casefold() != c.casefold()
            for e, c in zip(tree.expressions, columns)):
        return
    where = tree.args.get("where")
    if not where:
        return
    predicate = where.this
    if isinstance(predicate, exp.In) and len(predicate.expressions) == 2:
        column = predicate.this
        values = predicate.expressions
    elif isinstance(predicate, (exp.Or, exp.And)):
        left, right = predicate.this, predicate.expression
        if not (isinstance(left, exp.EQ) and isinstance(right, exp.EQ) and left.this == right.this):
            return
        column, values = left.this, [left.expression, right.expression]
    else:
        return
    if not isinstance(column, exp.Column) or not all(isinstance(v, exp.Literal) and
            v.this.casefold() in question.casefold() for v in values) or values[0] == values[1]:
        return
    branches = []
    for value in values:
        branch = tree.copy()
        branch.set("distinct", None)
        branch.set("where", exp.Where(this=exp.EQ(this=column.copy(), expression=value.copy())))
        branches.append(branch)
    yield exp.Intersect(this=branches[0], expression=branches[1], distinct=True).sql(dialect="sqlite")


def _universal(tree, question):
    cue = re.search(r"\b(greater|more|older|higher|larger|longer|less|lower|younger|smaller|shorter)"
                    r"\s+than\s+(?:the\s+[\w ]{1,35}?\s+of\s+)?(?:all|every)\b", question, re.I)
    if not cue:
        return
    greater = cue.group(1).casefold() in {"greater", "more", "older", "higher", "larger", "longer"}
    for op in tree.find_all(exp.GT if greater else exp.LT):
        if not isinstance(op.this, exp.Column) or not isinstance(op.expression, exp.Subquery):
            continue
        inner = op.expression.this
        if (not isinstance(inner, exp.Select) or len(inner.expressions) != 1 or
                not isinstance(inner.expressions[0], exp.Column) or
                inner.expressions[0].name.casefold() != op.this.name.casefold() or
                any(inner.args.get(k) for k in ("group", "having", "limit", "offset", "order"))):
            continue
        src = inner.args.get("from")
        if not src or not isinstance(src.this, exp.Table) or inner.args.get("joins"):
            continue
        changed = inner.copy()
        agg = exp.Max if greater else exp.Min
        changed.set("expressions", [agg(this=inner.expressions[0].copy())])
        for column in changed.find_all(exp.Column):
            if column.table and column.table.casefold() != src.this.alias_or_name.casefold():
                column.set("table", src.this.this.copy() if not src.this.alias else exp.to_identifier(src.this.alias))
        yield tree.transform(lambda n: changed.copy() if n is inner else n, copy=False).sql(dialect="sqlite")
        return


def set_candidates(history, example, question, schema, selected, matched):
    columns = example.get("columns", [])
    if not columns or not all(isinstance(c, str) for c in columns):
        return
    if not matched and schema and len(columns) <= 5:
        yield from _union(columns, _schema(schema) or {}, question)
    if not re.search(r"\b(?:both|all|every)\b", question, re.I):
        return
    for event in history:
        if event["tool"] != "check" or (matched and event["sql"] != selected):
            continue
        try:
            tree = sqlglot.parse_one(event["sql"], read="sqlite")
        except (ParseError, TokenError, RecursionError):
            continue
        if isinstance(tree, exp.Select):
            yield from _intersection(tree, columns, question)
            yield from _universal(tree, question)
