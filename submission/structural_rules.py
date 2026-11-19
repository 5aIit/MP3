"""Bounded structural proposals from language cues and visible schema.

These are proposals, not answer assertions: the controller must check them.
No files, database handles, task identities or learned benchmark labels are used.
"""
import re
from collections import Counter

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError


NUMBERS = dict(zip("one two three four five six seven eight nine ten".split(), range(1, 11)))
AVERAGE = re.compile(r"\b(?:older|younger|higher|lower|larger|smaller|greater|less|more|longer|shorter|taller)\s+than\s+(?:the\s+)?average\b", re.I)
AT_LEAST = re.compile(r"\bat least\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:distinct\s+)?([a-z_]+)", re.I)


def _parse(sql):
    if not isinstance(sql, str) or len(sql.encode("utf-8")) > 16000:
        return None
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except (ParseError, TokenError, RecursionError):
        return None
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        return None
    tree = statements[0]
    if any(isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Into, exp.Command,
                             exp.Create, exp.Drop)) for node in tree.walk()):
        return None
    return tree


def _stem(word):
    word = word.casefold()
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _words(value):
    return {_stem(word) for word in re.findall(r"[a-z]+", value.casefold())}


def _key(node):
    node = node.unalias().copy()
    for column in node.find_all(exp.Column):
        column.set("table", None)
        column.set("db", None)
        column.set("catalog", None)
    for identifier in node.find_all(exp.Identifier):
        identifier.set("quoted", False)
    return node.sql(dialect="sqlite", normalize=True, comments=False)


def _sources(tree, schema):
    if not isinstance(schema, list):
        return None
    metadata = {table["name"].casefold(): table for table in schema
                if isinstance(table, dict) and isinstance(table.get("name"), str)}
    base = tree.args.get("from")
    tables = ([base.this] if base is not None else []) + [j.this for j in tree.args.get("joins", [])]
    if not tables or any(not isinstance(table, exp.Table) or table.db or table.catalog for table in tables):
        return None
    sources = []
    for table in tables:
        record = metadata.get(table.name.casefold())
        if record is None:
            return None
        columns = {item["name"].casefold(): item["name"] for item in record.get("columns", [])
                   if isinstance(item, dict) and isinstance(item.get("name"), str)}
        sources.append((table, columns))
    if len({table.alias_or_name.casefold() for table, _ in sources}) != len(sources):
        return None
    return sources


def _resolve(node, sources, current=()):
    if not isinstance(node, exp.Column) or node.is_star or node.db or node.catalog:
        return None
    choices = [(table, columns[node.name.casefold()]) for table, columns in sources
               if node.name.casefold() in columns
               and (not node.table or node.table.casefold() in
                    {table.name.casefold(), table.alias_or_name.casefold()})]
    if not choices and node.table:
        # Public labels can use an evaluator alias such as t1. It is ignored
        # only when the unqualified name resolves uniquely in this query.
        choices = [(table, columns[node.name.casefold()]) for table, columns in sources
                   if node.name.casefold() in columns]
    if len(choices) != 1:
        projected = [item.unalias() for item in current
                     if isinstance(item.unalias(), exp.Column)
                     and item.unalias().name.casefold() == node.name.casefold()]
        if len(projected) == 1 and projected[0].table:
            choices = [(table, name) for table, name in choices
                       if table.alias_or_name.casefold() == projected[0].table.casefold()]
    if len(choices) != 1:
        return None
    table, name = choices[0]
    return exp.column(name, table=table.alias_or_name, quoted=True)


def _expected(tree, sources, example):
    labels = example.get("columns", [])
    if not isinstance(labels, list) or not 1 <= len(labels) <= 8:
        return None
    expressions = []
    for label in labels:
        if not isinstance(label, str) or len(label) > 1000:
            return None
        try:
            nodes = sqlglot.parse(label, read="sqlite")
        except (ParseError, TokenError, RecursionError):
            return None
        if len(nodes) != 1:
            return None
        node = nodes[0]
        if isinstance(node, exp.Column):
            matches = [item for item in tree.expressions
                       if item.alias and item.alias.casefold() == node.name.casefold() and not node.table]
            resolved = matches[0].copy() if len(matches) == 1 else _resolve(node, sources, tree.expressions)
        elif isinstance(node, (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)):
            if isinstance(node, exp.Count) and isinstance(node.this, exp.Star):
                resolved = node.copy()
            elif isinstance(node.this, exp.Column):
                operand = _resolve(node.this, sources, tree.expressions)
                resolved = node.copy() if operand is not None else None
                if resolved is not None:
                    resolved.set("this", operand)
            else:
                return None
        else:
            return None
        if resolved is None:
            return None
        expressions.append(resolved)
    return expressions


def _project(tree, expected):
    if expected is None or [_key(e) for e in tree.expressions] == [_key(e) for e in expected]:
        return tree.copy()
    if tree.args.get("distinct") or tree.args.get("with") or tree.args.get("qualify"):
        return None
    if len(list(tree.find_all(exp.Select))) != 1:
        return None
    if any(not isinstance(item.unalias(), (exp.Column, exp.AggFunc)) for item in tree.expressions):
        return None
    for key in ("group", "order"):
        clause = tree.args.get(key)
        if clause and any(isinstance(n, exp.Literal) and not n.is_string for n in clause.walk()):
            return None
    current_aggregates = [e for e in tree.expressions if isinstance(e.unalias(), exp.AggFunc)]
    expected_aggregates = [e for e in expected if isinstance(e.unalias(), exp.AggFunc)]
    if bool(current_aggregates) != bool(expected_aggregates):
        return None
    # Dropping/replacing a scalar aggregate affects row cardinality. Without a
    # GROUP BY, require the same collection of aggregates (only reorder them).
    if current_aggregates and not tree.args.get("group") and sorted(map(_key, current_aggregates)) != sorted(map(_key, expected_aggregates)):
        return None
    aliases = {e.alias.casefold(): e.unalias() for e in tree.expressions if e.alias}
    output_names = Counter(e.alias_or_name.casefold() for e in tree.expressions if e.alias_or_name)
    changed = tree.copy()
    for key, clause in changed.args.items():
        if key == "expressions":
            continue
        for child in clause if isinstance(clause, list) else [clause]:
            if not isinstance(child, exp.Expression):
                continue
            for column in list(child.find_all(exp.Column)):
                if not column.table and column.name.casefold() in aliases:
                    if (key != "order" or not isinstance(column.parent, exp.Ordered)
                            or output_names[column.name.casefold()] != 1):
                        return None
                    source = aliases[column.name.casefold()]
                    if any(not node.table and node.name.casefold() in aliases
                           for node in source.find_all(exp.Column)):
                        return None
                    column.replace(source.copy())
    changed.set("expressions", [e.copy() for e in expected])
    return changed


def _average_filter(tree, sources, expected, question):
    cue = AVERAGE.search(question)
    if cue is None or len(sources) != 1 or any(isinstance(node, exp.Avg) for node in tree.walk()):
        return None
    if len(list(tree.find_all(exp.Select))) != 1:
        return None
    table, columns = sources[0]
    adjective = cue.group(0).split()[0].casefold()
    specific = {"older": "age", "younger": "age", "taller": "height",
                "longer": "length", "shorter": "length"}.get(adjective)
    window = question[max(0, cue.start() - 60):min(len(question), cue.end() + 50)]
    scores = []
    for name in columns.values():
        tokens = _words(name)
        score = 10 if specific in tokens else len(tokens & _words(window))
        if score and not tokens <= {"id", "name", "code", "type"}:
            scores.append((score, name))
    if not scores:
        return None
    best = max(score for score, _ in scores)
    names = [name for score, name in scores if score == best]
    if len(names) != 1:
        return None
    name = names[0]
    query = exp.select(exp.Avg(this=exp.column(name, quoted=True))).from_(
        exp.Table(this=exp.to_identifier(table.name, quoted=True)))
    operator = exp.LT if adjective in {"younger", "lower", "smaller", "less", "shorter"} else exp.GT
    comparison = operator(this=exp.column(name, table=table.alias_or_name, quoted=True),
                          expression=exp.Subquery(this=query))
    changed = _project(tree, expected)
    if changed is None:
        return None
    where = changed.args.get("where")
    changed.set("where", exp.Where(this=exp.and_(where.this.copy(), comparison) if where else comparison))
    return changed


def _quantifier(tree, question):
    cue = re.search(r"\b(after|before|greater than|more than|less than|smaller than)\s+(any|all)\b", question, re.I)
    if cue is None:
        return None
    relation, quantifier = cue.group(1).lower(), cue.group(2).lower()
    greater = relation in {"after", "greater than", "more than"}
    kind = exp.Min if (greater == (quantifier == "any")) else exp.Max
    changes = []
    for node in tree.walk():
        if not isinstance(node, (exp.GT, exp.GTE) if greater else (exp.LT, exp.LTE)):
            continue
        rhs = node.expression
        if not isinstance(rhs, exp.Subquery) or not isinstance(rhs.this, exp.Select):
            continue
        query = rhs.this
        if len(query.expressions) != 1 or query.args.get("group") or query.args.get("having"):
            continue
        aggregate = query.expressions[0].unalias()
        if isinstance(aggregate, (exp.Min, exp.Max)) and not isinstance(aggregate, kind):
            changes.append(aggregate)
    if len(changes) != 1:
        return None
    target_index = next(index for index, node in enumerate(tree.walk()) if node is changes[0])
    changed = tree.copy()
    target = list(changed.walk())[target_index]
    target.replace(kind(this=target.this.copy()))
    return changed


def _existential_one(tree, question):
    """An OR alternative with an indefinite one denotes existence.

    Restrict this proposal to UNION branches used by membership tests. It is
    not a general rewrite of exact-count questions or standalone COUNT = 1.
    """
    if re.search(r"\b(?:exactly|only|precisely)\s+(?:one|1)\b", question, re.I):
        return None
    cue = re.search(r"\bor\s+(?:one|an?|1)\s+([a-z_]+)\b", question, re.I)
    if cue is None:
        return None
    changed = tree.copy()
    targets = []
    for query in changed.find_all(exp.Select):
        if not isinstance(query.parent, exp.Union) or query.find_ancestor(exp.In) is None:
            continue
        base, having = query.args.get("from"), query.args.get("having")
        if (base is None or not isinstance(base.this, exp.Table) or query.args.get("joins")
                or having is None or not isinstance(having.this, exp.EQ)
                or not isinstance(having.this.this, exp.Count)
                or not isinstance(having.this.expression, exp.Literal)
                or having.this.expression.name != "1"
                or _stem(cue.group(1)) not in _words(base.this.name)):
            continue
        targets.append(having.this)
    if len(targets) != 1:
        return None
    target = targets[0]
    target.replace(exp.GTE(this=target.this.copy(), expression=exp.Literal.number(1)))
    return changed


def _frequency_group(tree, sources, expected, question):
    cue = AT_LEAST.search(question)
    common = re.search(r"\bmost (?:common|frequent)\b", question, re.I)
    if not (cue or common) or len(sources) != 1 or expected is None:
        return None
    if not all(isinstance(item.unalias(), exp.Column) for item in expected):
        return None
    table, _ = sources[0]
    if cue and _stem(cue.group(2)) not in _words(table.name):
        return None
    if common:
        right = re.split(r"\b(?:across|among|of|for|in|with)\b", question[common.end():], maxsplit=1, flags=re.I)[0]
        before = re.search(r"([a-z_]+)\s+(?:(?:that|which)\s+)?(?:is|are|was|were)\s*$", question[:common.start()], re.I)
        mentioned = _words(right) | (_words(before.group(1)) if before else set())
        if len(expected) != 1 or not _words(expected[0].unalias().name) & mentioned:
            return None
    if tree.args.get("with") or tree.args.get("distinct"):
        return None
    where = tree.args.get("where")
    if where and any(isinstance(n, exp.Select) for n in where.walk()):
        # Only a single membership test for an aggregate is replaced. Other
        # filters/subqueries must not be silently discarded.
        if not cue or not isinstance(where.this, exp.In) or not any(isinstance(n, exp.Count) for n in where.walk()):
            return None
        where = None
    changed = exp.select(*[item.copy() for item in expected]).from_(table.copy())
    if where:
        changed.set("where", where.copy())
    changed.set("group", exp.Group(expressions=[item.unalias().copy() for item in expected]))
    if cue:
        raw = cue.group(1).lower()
        number = int(raw) if raw.isdigit() else NUMBERS[raw]
        if not 1 <= number <= 1000000:
            return None
        changed.set("having", exp.Having(this=exp.GTE(this=exp.Count(this=exp.Star()),
                                                       expression=exp.Literal.number(number))))
        if tree.args.get("order"):
            # Preserve ordering only if it refers directly to retained columns.
            keys = {_key(item) for item in expected}
            if any(_key(item.this) not in keys for item in tree.args["order"].expressions):
                return None
            changed.set("order", tree.args["order"].copy())
    else:
        changed.set("order", exp.Order(expressions=[exp.Ordered(this=exp.Count(this=exp.Star()), desc=True)]))
        changed.set("limit", exp.Limit(expression=exp.Literal.number(1)))
    return changed


def structural_repairs(sql, feedback, example, question, schema):
    """Yield at most four distinct, check-required structural proposals."""
    if not isinstance(question, str) or not isinstance(example, dict):
        return
    tree = _parse(sql)
    if tree is None:
        return
    sources = _sources(tree, schema)
    expected = _expected(tree, sources, example) if sources else None
    candidates = [_quantifier(tree, question), _existential_one(tree, question)]
    if sources:
        candidates += [_average_filter(tree, sources, expected, question),
                       _frequency_group(tree, sources, expected, question),
                       _project(tree, expected)]
    seen = {tree.sql(dialect="sqlite", normalize=True, comments=False)}
    for candidate in candidates:
        if candidate is None:
            continue
        normalized = candidate.sql(dialect="sqlite", normalize=True, comments=False)
        if normalized not in seen:
            seen.add(normalized)
            yield candidate.sql(dialect="sqlite")


def structural_penalty(sql, example, question, schema=None):
    """Flag only explicit, recognized evidence of a missing structural cue."""
    if not isinstance(question, str):
        return 0
    tree = _parse(sql)
    if tree is None:
        return 0
    penalty = 0
    if _quantifier(tree, question) is not None:
        penalty += 2
    if _existential_one(tree, question) is not None:
        penalty += 1
    if AVERAGE.search(question) and not any(isinstance(n, exp.Avg) for n in tree.walk()):
        penalty += 2
    if re.search(r"\bmost (?:common|frequent)\b", question, re.I) and not tree.args.get("group"):
        penalty += 1
    cue = AT_LEAST.search(question)
    if cue and re.search(r"\bat least\s+\w+\s+distinct\b", question, re.I):
        entity = _stem(cue.group(2))
        for clause in tree.find_all(exp.Having):
            for count in clause.find_all(exp.Count):
                if not isinstance(count.this, exp.Distinct):
                    penalty += 1
                columns = list(count.find_all(exp.Column))
                if columns and not any(entity in _words(column.name) for column in columns):
                    penalty += 1
    if cue and not tree.args.get("group"):
        base = tree.args.get("from")
        if base is not None and isinstance(base.this, exp.Table) and _stem(cue.group(2)) in _words(base.this.name):
            penalty += 1
    return penalty
