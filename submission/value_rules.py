"""Propose bounded value corrections from one official, read-only data probe.

This module performs no I/O. Every identifier comes from visible schema and
every replacement value from an observed probe row. Proposed SQL still needs
an independent check; observed text is never interpreted as SQL syntax.
"""
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError


PROBE_PREFIX = "/* value-grounding */ "
PROBE_COLUMNS = ["table_name", "column_name", "value"]


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
    if any(isinstance(node, (exp.Subquery, exp.SetOperation, exp.Into, exp.Command))
           for node in tree.walk()):
        return None
    return tree


def _schema(schema):
    if not isinstance(schema, list):
        return None
    result = {}
    for table in schema:
        if not isinstance(table, dict) or not isinstance(table.get("name"), str):
            return None
        name = table["name"]
        columns = {}
        for column in table.get("columns", []):
            if not isinstance(column, dict) or not isinstance(column.get("name"), str):
                return None
            key = column["name"].casefold()
            if key in columns:
                return None
            columns[key] = (column["name"], str(column.get("type", "")).upper())
        if name.casefold() in result:
            return None
        result[name.casefold()] = (name, columns)
    return result


def _sources(tree, metadata):
    base = tree.args.get("from")
    tables = ([base.this] if base is not None else []) + [
        join.this for join in tree.args.get("joins", [])]
    result = {}
    for table in tables:
        if (not isinstance(table, exp.Table) or not isinstance(table.this, exp.Identifier)
                or table.db or table.catalog or table.name.casefold() not in metadata):
            return None
        alias = table.alias_or_name.casefold()
        if alias in result:
            return None
        result[alias] = (table.alias_or_name, table.name.casefold())
    return result or None


def _resolve(column, sources, metadata):
    if not isinstance(column, exp.Column) or column.is_star or column.db or column.catalog:
        return None
    choices = [(alias, table, column.name.casefold()) for alias, table in sources.values()
               if (not column.table or alias.casefold() == column.table.casefold())
               and column.name.casefold() in metadata[table][1]]
    return choices[0] if len(choices) == 1 else None


def _filters(tree, sources, metadata):
    where = tree.args.get("where")
    if where is None:
        return []
    result = []
    for node in where.walk():
        if isinstance(node, (exp.EQ, exp.NEQ, exp.Like, exp.ILike)):
            column, literal = node.this, node.expression
            if isinstance(column, exp.Literal) and isinstance(literal, exp.Column):
                if not isinstance(node, (exp.EQ, exp.NEQ)):
                    continue
                column, literal = literal, column
            binding = _resolve(column, sources, metadata)
            if binding and isinstance(literal, exp.Literal) and literal.is_string:
                if 0 < len(literal.this) <= 100:
                    result.append((node, column, [literal], binding))
        elif isinstance(node, exp.In) and not node.args.get("query"):
            binding = _resolve(node.this, sources, metadata)
            literals = node.expressions
            if (binding and 1 <= len(literals) <= 6 and all(
                    isinstance(value, exp.Literal) and value.is_string and 0 < len(value.this) <= 100
                    for value in literals)):
                result.append((node, node.this, literals, binding))
    return result[:8]


def _text(value):
    return exp.Literal.string(value).sql(dialect="sqlite")


def _normal(value):
    return "".join(character.casefold() for character in value if character.isalnum())


def _needle(literal):
    return literal.strip("%_")


def value_probe(queries, question, schema):
    """Return one read-only SQL probe, or None if no supported filter exists.

    The controller must execute this SQL through the official execute tool and
    exclude that event from its answer-candidate list. A full eight-row result
    is treated as potentially incomplete by the repair functions below.
    """
    metadata = _schema(schema)
    if metadata is None or not isinstance(question, str) or not isinstance(queries, list):
        return None
    pairs, literals, used_tables = [], [], set()
    for event in queries[:8]:
        tree = _select(event.get("sql")) if isinstance(event, dict) else None
        sources = _sources(tree, metadata) if tree is not None else None
        if not sources:
            continue
        filters = _filters(tree, sources, metadata)
        if not filters:
            continue
        used_tables.update(table for _, table in sources.values())
        for _, _, values, (_, table, column) in filters:
            if (table, column) not in pairs:
                pairs.append((table, column))
            for literal in values:
                needle = _needle(literal.this)
                if needle and needle not in literals:
                    literals.append(needle)
    if not pairs:
        return None
    # Only already referenced tables are eligible for alternate column binding.
    # Explicit text types and familiar textual names cover schemas without types.
    for table in sorted(used_tables):
        for column, (_, kind) in metadata[table][1].items():
            textual = any(token in kind for token in ("TEXT", "CHAR", "CLOB"))
            familiar = bool(re.search(r"(?:name|title|label|description|details|type|status|format|team|state)", column))
            if (textual or familiar) and (table, column) not in pairs:
                pairs.append((table, column))
    clauses = []
    for table, column in pairs[:18]:
        table_name, columns = metadata[table]
        column_name = columns[column][0]
        reference = exp.column(column_name, table=table_name, quoted=True).sql(dialect="sqlite")
        table_sql = exp.Table(this=exp.to_identifier(table_name, quoted=True)).sql(dialect="sqlite")
        value = f"CAST({reference} AS TEXT)"
        conditions = [f"INSTR(LOWER({value}), LOWER({_text(literal)})) > 0" for literal in literals[:6]]
        conditions.append(f"(LENGTH({value}) >= 3 AND INSTR(LOWER({_text(question[:1000])}), LOWER({value})) > 0)")
        clause = (f"SELECT {_text(table_name)} AS table_name, {_text(column_name)} AS column_name, "
                  f"{value} AS value FROM {table_sql} WHERE {reference} IS NOT NULL AND "
                  f"({' OR '.join(conditions)}) GROUP BY {reference}")
        candidate = PROBE_PREFIX + "SELECT table_name, column_name, value FROM (" + " UNION ALL ".join(clauses + [clause]) + ") LIMIT 8"
        if len(candidate.encode("utf-8")) > 16000:
            break
        clauses.append(clause)
    if not clauses:
        return None
    return PROBE_PREFIX + "SELECT table_name, column_name, value FROM (" + " UNION ALL ".join(clauses) + ") LIMIT 8"


def _observed(probe_feedback, metadata):
    if (not isinstance(probe_feedback, dict) or not probe_feedback.get("ok")
            or probe_feedback.get("truncated") or probe_feedback.get("columns") != PROBE_COLUMNS):
        return None
    rows = probe_feedback.get("rows")
    count = probe_feedback.get("row_count")
    if not isinstance(rows, list) or type(count) is not int or count != len(rows) or count >= 8:
        return None
    result = {}
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 3 or not all(isinstance(value, str) for value in row):
            return None
        table, column, value = row
        key = table.casefold(), column.casefold()
        if key[0] not in metadata or key[1] not in metadata[key[0]][1] or len(value) > 500:
            return None
        result.setdefault(key, set()).add(value)
    return result


def _similar(old, value, question):
    needle, actual = _normal(_needle(old)), _normal(value)
    if not needle or not actual:
        return False
    if needle == actual:
        return True
    # A containment correction needs either corroboration in the question or
    # a substantial overlap, so a tiny fragment cannot choose an arbitrary name.
    if needle in actual or actual in needle:
        return actual in _normal(question) or min(len(needle), len(actual)) / max(len(needle), len(actual)) >= .5
    return actual in _normal(question) and len(actual) >= 3 and needle not in _normal(question)


def _literal_replacement(node, old, value):
    if isinstance(node, (exp.Like, exp.ILike)):
        # Keep the original wildcard semantics; do not turn a contains query
        # into an exact match just because one observed value currently fits.
        prefix = old[:len(old) - len(old.lstrip("%_"))]
        suffix = old[len(old.rstrip("%_")):]
        return prefix + value + suffix
    return value


def _proposals(sql, question, schema, probe_feedback):
    metadata, tree = _schema(schema), _select(sql)
    if metadata is None or tree is None or not isinstance(question, str):
        return []
    sources = _sources(tree, metadata)
    observed = _observed(probe_feedback, metadata)
    if not sources:
        return []
    result = _duration_proposals(tree, question, metadata, sources, probe_feedback)
    if observed is None:
        return result
    for node, column, literals, (alias, table, name) in _filters(tree, sources, metadata):
        values = observed.get((table, name), set())
        old_values = [literal.this for literal in literals]
        replacements = []
        for old in old_values:
            core = _needle(old)
            if core in values:
                replacements.append(old)
                continue
            choices = [value for value in values if _similar(old, value, question)]
            exact = [value for value in choices if _normal(value) == _normal(core)]
            choices = exact or choices
            replacements.append(_literal_replacement(node, old, choices[0]) if len(choices) == 1 else None)
        # For an IN enumeration, an observed list explicitly named in the
        # question may replace a guessed encoding as one set operation.
        if isinstance(node, exp.In) and any(value is None for value in replacements):
            named = sorted(value for value in values if len(_normal(value)) >= 3
                           and _normal(value) in _normal(question))
            if len(named) == len(old_values) and not set(old_values) & values:
                replacements = named
        if (all(value is not None for value in replacements) and replacements != old_values
                and len(set(replacements)) == len(replacements)):
            changed = tree.copy()
            target = list(changed.walk())[next(index for index, value in enumerate(tree.walk()) if value is node)]
            if isinstance(target, exp.In):
                target.set("expressions", [exp.Literal.string(value) for value in replacements])
            elif isinstance(target.this, exp.Literal):
                target.set("this", exp.Literal.string(replacements[0]))
            else:
                target.set("expression", exp.Literal.string(replacements[0]))
            result.append(changed)
        # Rebinding needs an exact observed value in one alternative column.
        # Existing successful grounding in the original column takes priority.
        if len(literals) != 1 or isinstance(node, exp.In):
            continue
        old = _needle(old_values[0])
        if any(_similar(old_values[0], value, question) for value in values):
            continue
        alternatives = []
        for (other_table, other_column), candidates in observed.items():
            aliases = [source_alias for source_alias, source_table in sources.values() if source_table == other_table]
            matches = [value for value in candidates if _normal(value) == _normal(old)]
            if len(aliases) == 1 and len(matches) == 1 and (other_table, other_column) != (table, name):
                alternatives.append((aliases[0], metadata[other_table][1][other_column][0], matches[0]))
        if len(alternatives) == 1:
            new_alias, new_column, new_value = alternatives[0]
            changed = tree.copy()
            target = list(changed.walk())[next(index for index, value in enumerate(tree.walk()) if value is node)]
            replacement = exp.column(new_column, table=new_alias, quoted=True)
            if isinstance(target.this, exp.Column):
                target.set("this", replacement)
                target.set("expression", exp.Literal.string(_literal_replacement(node, old_values[0], new_value)))
            else:
                target.set("expression", replacement)
                target.set("this", exp.Literal.string(new_value))
            result.append(changed)
    return result


def _duration_proposals(tree, question, metadata, sources, feedback):
    """Use an observed minute:second representation for a minute bucket.

    Unlike choosing one replacement value, learning a representation does not
    require an exhaustive value probe. Exact-duration wording is excluded.
    """
    cue = re.search(r"\b(\d+)\s*(?:minutes?|mins?)\b", question, re.I)
    if (cue is None or re.search(r"\b(?:exactly|precisely)\b", question, re.I)
            or not isinstance(feedback, dict) or not feedback.get("ok")
            or feedback.get("columns") != PROBE_COLUMNS):
        return []
    rows = feedback.get("rows", [])
    if not isinstance(rows, list) or not all(isinstance(row, list) and len(row) == 3 for row in rows):
        return []
    result = []
    for node, column, literals, (_, table, name) in _filters(tree, sources, metadata):
        if (not isinstance(node, exp.EQ) or len(literals) != 1
                or literals[0].this != cue.group(1)
                or not re.search(r"duration|time|length", name, re.I)):
            continue
        values = [row[2] for row in rows if isinstance(row[0], str) and isinstance(row[1], str)
                  and row[0].casefold() == table and row[1].casefold() == name]
        if len(values) < 2 or not all(isinstance(value, str)
                and re.fullmatch(r"\d{1,4}:[0-5]\d", value) for value in values):
            continue
        changed = tree.copy()
        target = list(changed.walk())[next(index for index, value in enumerate(tree.walk()) if value is node)]
        position = exp.func("INSTR", column.copy(), exp.Literal.string(":"))
        prefix = exp.func("SUBSTR", column.copy(), exp.Literal.number(1),
                          exp.Sub(this=position, expression=exp.Literal.number(1)))
        target.replace(exp.EQ(this=exp.Cast(this=prefix, to=exp.DataType.build("INT")),
                              expression=exp.Literal.number(cue.group(1))))
        result.append(changed)
    return result


def value_repairs(sql, feedback, example, question, schema, probe_feedback):
    """Yield at most three distinct, observed-value proposals requiring check."""
    seen = set()
    for tree in _proposals(sql, question, schema, probe_feedback):
        candidate = tree.sql(dialect="sqlite")
        if candidate not in seen and len(candidate.encode("utf-8")) <= 16000:
            seen.add(candidate)
            yield candidate
            if len(seen) == 3:
                return


def value_penalty(sql, example, question, schema=None, probe_feedback=None):
    """A small ranking cue only when observed values support a specific repair."""
    return min(len(_proposals(sql, question, schema, probe_feedback)), 2)
