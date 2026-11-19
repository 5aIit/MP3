"""Check cardinality changes supported by the question and observed row counts."""
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError


def cardinality_candidates(event, example, question):
    feedback = event["observation"]
    if not feedback.get("ok") or len(feedback.get("columns", [])) != len(example.get("columns", [])):
        return
    try:
        tree = sqlglot.parse_one(event["sql"], read="sqlite")
    except (ParseError, TokenError, RecursionError):
        return
    if not isinstance(tree, exp.Select) or any(tree.args.get(k) for k in ("group", "having", "limit", "offset", "with")):
        return
    if not all(isinstance(e.unalias(), exp.Column) and not e.unalias().is_star for e in tree.expressions):
        return
    actual, expected = feedback.get("row_count"), example.get("row_count")
    if not isinstance(actual, int) or not isinstance(expected, int):
        return
    unique_requested = re.search(r"\b(?:distinct|different|unique|once|duplicates?)\b", question, re.I)
    if tree.args.get("distinct") and actual <= expected and not unique_requested:
        changed = tree.copy()
        changed.set("distinct", None)
        yield changed.sql(dialect="sqlite")
    elif not tree.args.get("distinct") and (actual > expected or unique_requested and feedback.get("matches_example")):
        changed = tree.copy()
        changed.set("distinct", exp.Distinct())
        yield changed.sql(dialect="sqlite")
    where = tree.args.get("where")
    if (not where or actual > expected or
            re.search(r"\b(?:null|missing|known|recorded|available|empty|nonempty|unknown)\b", question, re.I)):
        return
    retained = []
    predicates = list(where.this.flatten()) if isinstance(where.this, exp.And) else [where.this]
    for term in predicates:
        child = term.this
        if (isinstance(term, exp.Not) and isinstance(child, exp.Is) and
                isinstance(child.expression, exp.Null) and isinstance(child.this, exp.Column)):
            continue
        retained.append(term)
    if len(retained) != len(predicates):
        changed = tree.copy()
        changed.set("where", exp.Where(this=exp.and_(*(p.copy() for p in retained))) if retained else None)
        yield changed.sql(dialect="sqlite")
