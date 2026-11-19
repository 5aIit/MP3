"""Schema-grounded date extrema; candidates are checked by the controller."""
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError


def _words(text):
    words = re.findall(r"[a-z]+", text.lower())
    return {re.sub(r"(?:ing|ed|s)$", "", word).rstrip("e") for word in words}


def _cue(question):
    if not isinstance(question, str):
        return None
    # Names contain ordinal words without requesting row order.
    text = re.sub(r"\b(?:first|last|middle)\s+names?\b", "", question, flags=re.I)
    latest = re.search(r"\b(?:latest|newest|most recently|most recent)\b", text, re.I)
    earliest = re.search(r"\b(?:earliest|oldest|first)\b", text, re.I)
    if bool(latest) == bool(earliest):
        return None
    return bool(latest), text


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
    if any(tree.args.get(key) for key in ("with", "group", "having", "offset", "qualify", "into")):
        return None
    if any(not isinstance(item.unalias(), exp.Column) for item in tree.expressions):
        return None
    return tree


def _propose(sql, question, schema):
    cue, tree = _cue(question), _parse(sql)
    if cue is None or tree is None or not isinstance(schema, list):
        return None
    descending, text = cue
    metadata = {table["name"].casefold(): table for table in schema}
    base = tree.args.get("from")
    tables = ([base.this] if base is not None else []) + [j.this for j in tree.args.get("joins", [])]
    if not tables or any(not isinstance(t, exp.Table) or t.name.casefold() not in metadata for t in tables):
        return None
    question_words = _words(text)
    time_words = _words("date year time launch release publish start begin open establish found create contact")
    scored = []
    for table in tables:
        for column in metadata[table.name.casefold()].get("columns", []):
            name, kind = column["name"], column.get("type", "").casefold()
            words = _words(name)
            action = words & question_words - _words("date year time name id number")
            if not (words & time_words or "date" in kind or "time" in kind):
                continue
            # Require an event word or an explicitly asked date/year/time.
            evidence = len(action) * 4 + len(words & question_words & _words("date year time")) * 2
            if evidence == 0:
                continue
            evidence += bool(words & _words("from start begin"))
            evidence -= bool(words & _words("to end finish"))
            scored.append((evidence, table.alias_or_name, name))
    if not scored:
        return None
    best = max(item[0] for item in scored)
    choices = [item for item in scored if item[0] == best]
    if len(choices) != 1:
        return None
    _, alias, column = choices[0]
    chosen = exp.column(column, table=alias, quoted=True)
    changed = tree.copy()
    where = changed.args.get("where")
    if where and isinstance(where.this, exp.In):
        sub = where.this.args.get("query")
        inner = sub.this if isinstance(sub, exp.Subquery) else None
        # Replace only an unfiltered top-one membership used to encode an
        # extreme. Preserve other explicit conditions and nested predicates.
        if (not isinstance(inner, exp.Select) or not inner.args.get("order")
                or not inner.args.get("limit") or inner.args["limit"].expression.name != "1"
                or any(inner.args.get(key) for key in ("where", "group", "having"))):
            return None
        changed.set("where", None)
    elif where and any(isinstance(node, exp.In) for node in where.walk()):
        return None
    limit = changed.args.get("limit")
    if limit and (not isinstance(limit.expression, exp.Literal) or limit.expression.name != "1"):
        return None
    changed.set("order", exp.Order(expressions=[exp.Ordered(this=chosen, desc=descending)]))
    changed.set("limit", exp.Limit(expression=exp.Literal.number(1)))
    # Canonical keys disregard harmless quoting for stable deduplication.
    def key(node):
        node = node.copy()
        for identifier in node.find_all(exp.Identifier):
            identifier.set("quoted", False)
        return node.sql(dialect="sqlite", normalize=True, comments=False)
    return changed if key(changed) != key(tree) else None


def superlative_repairs(sql, feedback, example, question, schema):
    candidate = _propose(sql, question, schema)
    if candidate is not None:
        yield candidate.sql(dialect="sqlite")


def superlative_penalty(sql, example, question, schema=None):
    if schema is not None:
        return 2 if _propose(sql, question, schema) is not None else 0
    tree = _parse(sql)
    return int(_cue(question) is not None and tree is not None and not tree.args.get("order"))
