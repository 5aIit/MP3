"""Schema-grounded anti-existence proposals; candidates still require check.

Supports parent/child relations, a two-edge location relation, and categories
within one table.
There is no database, filesystem, model, or task-identity access in this module.
"""
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError


NEGATION = re.compile(r"\b(?:never|not\s+ever|(?:do|does|did)\s+not\s+(?:have|own|contain|include)|"
                      r"(?:have|has|had)\s+no|without\s+any|not\s+participated\s+in\s+any)\b", re.I)
CATEGORY_NEGATION = re.compile(r"\b(?:never|not\s+ever|no(?:\s+[a-z]+){0,6}\s+any)\b", re.I)
LOCATION_NEGATION = re.compile(r"\b(?:not\s+(?:living|located|based)\s+in|"
                               r"(?:do|does|did)\s+not\s+live\s+in)\b", re.I)


def _parse(sql):
    if not isinstance(sql, str) or len(sql.encode("utf-8")) > 16000:
        return None
    try:
        nodes = sqlglot.parse(sql, read="sqlite")
    except (ParseError, TokenError, RecursionError):
        return None
    if len(nodes) != 1 or not isinstance(nodes[0], exp.Select):
        return None
    tree = nodes[0]
    if (len(list(tree.find_all(exp.Select))) != 1
            or any(tree.args.get(key) for key in ("with", "group", "having", "qualify", "into"))
            or any(isinstance(node, (exp.Subquery, exp.Window, exp.Command)) for node in tree.walk())):
        return None
    return tree


def _metadata(schema):
    if not isinstance(schema, list):
        return None
    result = {}
    for table in schema:
        if not isinstance(table, dict) or not isinstance(table.get("name"), str):
            return None
        name = table["name"].casefold()
        if name in result:
            return None
        columns = {item["name"].casefold(): item["name"] for item in table.get("columns", [])
                   if isinstance(item, dict) and isinstance(item.get("name"), str)}
        result[name] = {"name": table["name"], "columns": columns,
                        "foreign_keys": table.get("foreign_keys", [])}
    return result


def _sources(tree, metadata, max_tables=2):
    from_clause = tree.args.get("from")
    tables = ([from_clause.this] if from_clause else []) + [j.this for j in tree.args.get("joins", [])]
    if not 1 <= len(tables) <= max_tables:
        return None
    result = {}
    for table in tables:
        if not isinstance(table, exp.Table) or table.db or table.catalog or table.name.casefold() not in metadata:
            return None
        alias = table.alias_or_name.casefold()
        if alias in result or not isinstance(table.this, exp.Identifier):
            return None
        result[alias] = (table, metadata[table.name.casefold()])
    return result


def _binding(column, sources):
    if not isinstance(column, exp.Column) or column.is_star or column.db or column.catalog:
        return None
    matches = [(alias, record["columns"][column.name.casefold()]) for alias, (_, record) in sources.items()
               if column.name.casefold() in record["columns"]
               and (not column.table or column.table.casefold() == alias)]
    return matches[0] if len(matches) == 1 else None


def _tokens(text):
    parts = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower())
    irregular = {"people": "person", "men": "man", "women": "woman", "children": "child"}
    return {irregular.get(word, word[:-3] + "y" if word.endswith("ies") else
                         word[:-1] if word.endswith("s") and not word.endswith("ss") else word)
            for word in parts}


def _foreign_keys(child, metadata):
    result = set()
    for foreign in child["foreign_keys"]:
        if not isinstance(foreign, dict) or not all(isinstance(foreign.get(k), str) for k in ("table", "from", "to")):
            continue
        parent, own, target = (foreign[k].casefold() for k in ("table", "from", "to"))
        if (parent in metadata and own in child["columns"] and target in metadata[parent]["columns"]
                and parent != child["name"].casefold()):
            result.add((parent, own, target))
    return sorted(result)


def _conjuncts(node):
    if isinstance(node, exp.Paren):
        return _conjuncts(node.this)
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def _forbidden(predicate, question):
    """Convert a row-level exclusion into the child condition that must not exist."""
    if isinstance(predicate, exp.NEQ):
        return exp.EQ(this=predicate.this.copy(), expression=predicate.expression.copy())
    if isinstance(predicate, exp.Not) and isinstance(predicate.this, (exp.In, exp.Like)):
        return predicate.this.copy()
    if isinstance(predicate, (exp.GT, exp.GTE, exp.LT, exp.LTE)):
        value = predicate.expression
        if not isinstance(value, exp.Literal) or value.is_string:
            return None
        pattern = (r"\b(longer than|shorter than|greater than|less than|higher than|lower than|"
                   r"more than|fewer than|larger than|smaller than|older than|younger than|"
                   r"at least|at most|over|under)\s+(-?\d+(?:\.\d+)?)\b")
        cues = [match for match in re.finditer(pattern, question, re.I)
                if float(match.group(2)) == float(value.this)]
        if len(cues) != 1:
            return None
        cue = cues[0].group(1).lower()
        cls = exp.GTE if cue == "at least" else exp.LTE if cue == "at most" else (
            exp.LT if cue in {"shorter than", "less than", "lower than", "fewer than", "smaller than", "younger than", "under"}
            else exp.GT)
        return cls(this=predicate.this.copy(), expression=value.copy())
    if isinstance(predicate, exp.EQ) and isinstance(predicate.expression, exp.Literal):
        return predicate.copy()
    if (isinstance(predicate, exp.In) and not predicate.args.get("query")
            and predicate.expressions and all(isinstance(item, exp.Literal) for item in predicate.expressions)):
        return predicate.copy()
    return None


def _projection(tree, sources, parent_alias, parent, example, inferred):
    if not inferred:
        for item in tree.expressions:
            node = item.unalias()
            if isinstance(node, exp.Count) and isinstance(node.this, exp.Star):
                continue
            if not isinstance(node, exp.Column) or (_binding(node, sources) or (None,))[0] != parent_alias:
                return None
        return [item.copy() for item in tree.expressions]
    labels = example.get("columns", []) if isinstance(example, dict) else []
    if not 1 <= len(labels) <= 8:
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
        if isinstance(node, exp.Column) and not node.is_star and node.name.casefold() in parent["columns"]:
            expressions.append(exp.column(parent["columns"][node.name.casefold()], table=parent_alias, quoted=True))
        elif (isinstance(node, exp.Count) and isinstance(node.this, exp.Star)
              and len(tree.expressions) == 1 and isinstance(tree.expressions[0].unalias(), exp.Count)
              and isinstance(tree.expressions[0].unalias().this, exp.Star)):
            expressions.append(exp.Count(this=exp.Star()))
        else:
            return None
    return expressions


def _category_negation(tree, sources, question):
    """A category must have no forbidden row, rather than one allowed row.

    The outer table supplies the universe of observed categories. NULL category
    values form one group, so the correlation uses SQLite's null-safe IS.
    """
    if (len(sources) != 1 or len(tree.expressions) != 1 or not CATEGORY_NEGATION.search(question)
            or tree.args.get("joins") or tree.args.get("limit") or tree.args.get("offset")):
        return None
    projection = tree.expressions[0]
    group_column = projection.unalias()
    binding = _binding(group_column, sources)
    if binding is None or not _tokens(group_column.name) <= _tokens(question):
        return None
    outer_key, category = binding
    predicate = tree.args["where"].this
    if not isinstance(predicate, exp.EQ):
        return None
    column, literal = predicate.this, predicate.expression
    if isinstance(column, exp.Literal):
        column, literal = literal, column
    filter_binding = _binding(column, sources)
    if (not isinstance(literal, exp.Literal) or filter_binding is None
            or filter_binding[0] != outer_key or filter_binding[1].casefold() == category.casefold()):
        return None
    order = tree.args.get("order")
    if order:
        for item in order.expressions:
            value = item.this
            is_alias = (isinstance(value, exp.Column) and not value.table and projection.alias
                        and value.name.casefold() == projection.alias.casefold())
            if not is_alias and _binding(value, sources) != binding:
                return None
    outer_table, metadata = sources[outer_key]
    outer_alias = outer_table.alias_or_name
    inner_alias = "_neg_category"
    while inner_alias.casefold() in sources:
        inner_alias += "_"
    inner_table = exp.Table(this=exp.to_identifier(metadata["name"], quoted=True),
                           alias=exp.TableAlias(this=exp.to_identifier(inner_alias, quoted=True)))
    correlation = exp.NullSafeEQ(
        this=exp.column(category, table=inner_alias, quoted=True),
        expression=exp.column(category, table=outer_alias, quoted=True))
    forbidden = exp.EQ(this=exp.column(filter_binding[1], table=inner_alias, quoted=True),
                       expression=literal.copy())
    inner = exp.select(exp.Literal.number(1)).from_(inner_table).where(exp.and_(correlation, forbidden))
    changed = tree.copy()
    changed.set("where", exp.Where(this=exp.Not(this=exp.Exists(this=inner))))
    changed.set("distinct", exp.Distinct())
    candidate = changed.sql(dialect="sqlite")
    return candidate if len(candidate.encode("utf-8")) <= 16000 else None


def _location_negation(tree, metadata, question):
    """Exclude parents with a forbidden location along one declared FK path.

    Requiring the original joins to be exactly that path prevents unrelated
    relationships from being silently removed. NULL child keys do not match
    the correlation, while parents with no relationship remain in the answer.
    """
    if not LOCATION_NEGATION.search(question):
        return None
    sources = _sources(tree, metadata, max_tables=3)
    joins = tree.args.get("joins", [])
    if sources is None or len(sources) != 3 or len(joins) != 2:
        return None
    if len({record["name"].casefold() for _, record in sources.values()}) != 3:
        return None
    if any(join.args.get("side") or join.args.get("method") or join.args.get("using")
           or join.args.get("kind", "").upper() not in ("", "INNER") for join in joins):
        return None
    projected = [_binding(item.unalias(), sources) for item in tree.expressions]
    if not projected or any(binding is None for binding in projected):
        return None
    parent_aliases = {binding[0] for binding in projected}
    if len(parent_aliases) != 1:
        return None
    parent_alias = next(iter(parent_aliases))
    parent_terms, location_terms = [], []
    for term in _conjuncts(tree.args["where"].this):
        if any(isinstance(node, exp.Or) for node in term.walk()):
            return None
        bindings = [_binding(column, sources) for column in term.find_all(exp.Column)]
        if not bindings or any(binding is None for binding in bindings):
            return None
        if {binding[0] for binding in bindings} == {parent_alias}:
            parent_terms.append(term.copy())
            continue
        if not isinstance(term, exp.NEQ):
            return None
        column, literal = term.this, term.expression
        if isinstance(column, exp.Literal):
            column, literal = literal, column
        target = _binding(column, sources)
        if (target is None or target[0] == parent_alias or not isinstance(literal, exp.Literal)
                or not literal.is_string or not literal.this
                or literal.this.casefold() not in question.casefold()):
            return None
        location_words = {"state", "province", "county", "city", "country", "region",
                          "place", "location", "address", "town"}
        target_words = _tokens(target[1]) | _tokens(sources[target[0]][1]["name"])
        if not target_words & location_words:
            return None
        location_terms.append((target, literal))
    if len(location_terms) != 1:
        return None
    (location_alias, location_column), literal = location_terms[0]
    middle_alias = next(alias for alias in sources if alias not in {parent_alias, location_alias})

    def edge_key(left, right):
        return frozenset(((left[0], left[1].casefold()), (right[0], right[1].casefold())))

    edges = {}
    for own_alias, (_, record) in sources.items():
        for target_name, own_column, target_column in _foreign_keys(record, metadata):
            for target_alias, (_, target_record) in sources.items():
                if target_record["name"].casefold() == target_name:
                    left = (own_alias, record["columns"][own_column])
                    right = (target_alias, target_record["columns"][target_column])
                    edges[edge_key(left, right)] = (left, right)

    def between(start, end):
        result = []
        for left, right in edges.values():
            if (left[0], right[0]) == (start, end):
                result.append((left, right))
            elif (right[0], left[0]) == (start, end):
                result.append((right, left))
        return result

    first = between(parent_alias, middle_alias)
    second = between(middle_alias, location_alias)
    if len(first) != 1 or len(second) != 1 or between(parent_alias, location_alias):
        return None
    expected_edges = {edge_key(*first[0]), edge_key(*second[0])}
    actual_edges = set()
    for join in joins:
        condition = join.args.get("on")
        if not isinstance(condition, exp.EQ):
            return None
        left, right = _binding(condition.this, sources), _binding(condition.expression, sources)
        if left is None or right is None:
            return None
        actual_edges.add(edge_key(left, right))
    if actual_edges != expected_edges:
        return None
    output_aliases = {item.alias.casefold() for item in tree.expressions if item.alias}
    order = tree.args.get("order")
    if order:
        for column in order.find_all(exp.Column):
            if not column.table and column.name.casefold() in output_aliases:
                continue
            if (_binding(column, sources) or (None,))[0] != parent_alias:
                return None

    occupied = set(sources)
    inner_aliases = []
    for base_name in ("_neg_link", "_neg_location"):
        alias = base_name
        while alias.casefold() in occupied:
            alias += "_"
        occupied.add(alias.casefold())
        inner_aliases.append(alias)
    link_alias, place_alias = inner_aliases
    parent_table = sources[parent_alias][0]
    link_table, place_table = sources[middle_alias][0].copy(), sources[location_alias][0].copy()
    link_table.set("alias", exp.TableAlias(this=exp.to_identifier(link_alias, quoted=True)))
    place_table.set("alias", exp.TableAlias(this=exp.to_identifier(place_alias, quoted=True)))
    parent_key, link_parent_key = first[0][0][1], first[0][1][1]
    link_place_key, place_key = second[0][0][1], second[0][1][1]
    correlation = exp.EQ(this=exp.column(link_parent_key, table=link_alias, quoted=True),
                         expression=exp.column(parent_key, table=parent_table.alias_or_name, quoted=True))
    relation = exp.EQ(this=exp.column(link_place_key, table=link_alias, quoted=True),
                      expression=exp.column(place_key, table=place_alias, quoted=True))
    forbidden = exp.EQ(this=exp.column(location_column, table=place_alias, quoted=True),
                       expression=literal.copy())
    inner = exp.select(exp.Literal.number(1)).from_(link_table)
    inner.set("joins", [exp.Join(this=place_table, on=relation)])
    inner.set("where", exp.Where(this=exp.and_(correlation, forbidden)))
    anti = exp.Not(this=exp.Exists(this=inner))
    changed = tree.copy()
    changed.set("from", exp.From(this=parent_table.copy()))
    changed.set("joins", None)
    changed.set("where", exp.Where(this=exp.and_(*parent_terms, anti)))
    candidate = changed.sql(dialect="sqlite")
    return candidate if len(candidate.encode("utf-8")) <= 16000 else None


def _propose(sql, example, question, schema):
    if (not isinstance(question, str)
            or not (NEGATION.search(question) or CATEGORY_NEGATION.search(question)
                    or LOCATION_NEGATION.search(question))):
        return None
    tree, metadata = _parse(sql), _metadata(schema)
    if tree is None or metadata is None or not tree.args.get("where"):
        return None
    location_candidate = _location_negation(tree, metadata, question)
    if location_candidate is not None:
        return location_candidate
    if not (NEGATION.search(question) or CATEGORY_NEGATION.search(question)):
        return None
    sources = _sources(tree, metadata)
    if sources is None:
        return None
    category_candidate = _category_negation(tree, sources, question)
    if category_candidate is not None:
        return category_candidate
    joins = tree.args.get("joins", [])
    if any(join.args.get("side", "").upper() not in ("", "LEFT") or join.args.get("method")
           or join.args.get("using") or join.args.get("kind", "").upper() not in ("", "INNER", "OUTER") for join in joins):
        return None
    options = []
    for child_alias, (child_table, child) in sources.items():
        for parent_name, child_key, parent_key in _foreign_keys(child, metadata):
            parents = [(alias, table, record) for alias, (table, record) in sources.items()
                       if table.name.casefold() == parent_name]
            if len(parents) == 1 and len(sources) == 2:
                parent_alias, parent_table, parent = parents[0]
                inferred = False
                on = joins[0].args.get("on") if len(joins) == 1 else None
                if not isinstance(on, exp.EQ):
                    continue
                bindings = {_binding(on.this, sources), _binding(on.expression, sources)}
                if bindings != {(parent_alias, parent["columns"][parent_key]),
                                 (child_alias, child["columns"][child_key])}:
                    continue
            elif not parents and len(sources) == 1:
                parent = metadata[parent_name]
                if not _tokens(parent["name"]) <= _tokens(question):
                    continue
                parent_alias = parent["name"]
                if parent_alias.casefold() in sources:
                    parent_alias = "_neg_parent"
                parent_table = exp.Table(this=exp.to_identifier(parent["name"], quoted=True))
                if parent_alias != parent["name"]:
                    parent_table.set("alias", exp.TableAlias(this=exp.to_identifier(parent_alias, quoted=True)))
                inferred = True
            else:
                continue
            expressions = _projection(tree, sources, parent_alias, parent, example, inferred)
            if expressions is not None:
                options.append((child_alias, child_table, child, child_key,
                                parent_alias, parent_table, parent, parent_key, expressions, inferred))
    if len(options) != 1:
        return None
    child_alias, child_table, child, child_key, parent_alias, parent_table, parent, parent_key, expressions, inferred = options[0]
    parent_terms, child_terms = [], []
    for term in _conjuncts(tree.args["where"].this):
        bindings = [_binding(column, sources) for column in term.find_all(exp.Column)]
        if not bindings or any(binding is None for binding in bindings):
            return None
        aliases = {binding[0] for binding in bindings}
        if aliases == {child_alias}:
            forbidden = _forbidden(term, question)
            if forbidden is None:
                return None
            child_terms.append(forbidden)
        elif not inferred and aliases == {parent_alias}:
            parent_terms.append(term.copy())
        else:
            return None
    # Multiple negated predicates need a language-aware AND/OR decision.
    if len(child_terms) != 1:
        return None
    for key in ("order",):
        clause = tree.args.get(key)
        if clause and any((_binding(column, sources) or (None,))[0] != parent_alias
                          for column in clause.find_all(exp.Column)):
            return None
        if inferred and clause:
            return None
    inner_alias = "_neg_child"
    while inner_alias.casefold() in sources or inner_alias.casefold() == parent_alias.casefold():
        inner_alias += "_"
    forbidden = child_terms[0]
    for column in forbidden.find_all(exp.Column):
        column.set("table", exp.to_identifier(inner_alias, quoted=True))
    relation = exp.EQ(this=exp.column(child["columns"][child_key], table=inner_alias, quoted=True),
                      expression=exp.column(parent["columns"][parent_key], table=parent_alias, quoted=True))
    inner_table = exp.Table(this=exp.to_identifier(child["name"], quoted=True),
                           alias=exp.TableAlias(this=exp.to_identifier(inner_alias, quoted=True)))
    inner = exp.select(exp.Literal.number(1)).from_(inner_table).where(exp.and_(relation, forbidden))
    anti = exp.Not(this=exp.Exists(this=inner))
    changed = tree.copy()
    changed.set("from", exp.From(this=parent_table.copy()))
    changed.set("joins", None)
    changed.set("expressions", expressions)
    changed.set("where", exp.Where(this=exp.and_(*parent_terms, anti)))
    candidate = changed.sql(dialect="sqlite")
    return candidate if len(candidate.encode("utf-8")) <= 16000 else None


def negation_repairs(sql, feedback, example, question, schema):
    """Yield at most one schema-supported correlated NOT EXISTS query."""
    candidate = _propose(sql, example, question, schema)
    if candidate is not None:
        yield candidate


def negation_penalty(sql, example, question, schema=None):
    """Request inspection for explicit anti-existence wording; rank checked repairs."""
    if (not isinstance(question, str)
            or not (NEGATION.search(question) or CATEGORY_NEGATION.search(question)
                    or LOCATION_NEGATION.search(question))):
        return 0
    tree = _parse(sql)
    if tree is None or not tree.args.get("where"):
        return 0
    if schema is None:
        return 1
    return 2 if _propose(sql, example, question, schema) is not None else 0
