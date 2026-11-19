"""Small question/schema lexical repairs, proposed for official checking.

No files or databases are read. A stronger column-name match is evidence for a
candidate, not proof that the query has the intended meaning.
"""
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError


def _words(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    result = set()
    for word in re.findall(r"[a-z]+", text.lower()):
        if word.endswith("ies"):
            word = word[:-3] + "y"
        elif word.endswith(("sses", "ches", "shes", "xes")):
            word = word[:-2]
        elif word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        word = re.sub(r"(?:ing|ed)$", "", word).rstrip("e")
        result.add(word)
    return result


GENERIC = _words("total number average mean sum count id code key name")
MEASURES = _words("amount value price cost quantity duration length height age weight score rating temperature pressure")
DESCRIPTIVE = _words("name label title description details address")


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
    if (tree.args.get("with") or tree.args.get("into")
            or len(list(tree.find_all(exp.Select))) != 1
            or any(isinstance(node, (exp.Subquery, exp.Window, exp.SetOperation)) for node in tree.walk())):
        return None
    return tree


def _context(tree, schema):
    if not isinstance(schema, list):
        return None
    metadata = {}
    for table in schema:
        if not isinstance(table, dict) or not isinstance(table.get("name"), str):
            return None
        key = table["name"].casefold()
        if key in metadata:
            return None
        metadata[key] = table
    base = tree.args.get("from")
    tables = ([base.this] if base is not None else []) + [join.this for join in tree.args.get("joins", [])]
    sources = {}
    for table in tables:
        if (not isinstance(table, exp.Table) or table.db or table.catalog
                or not isinstance(table.this, exp.Identifier) or table.name.casefold() not in metadata
                or table.alias_or_name.casefold() in sources):
            return None
        sources[table.alias_or_name.casefold()] = (table.alias_or_name, table.name.casefold())
    return (metadata, sources) if sources else None


def _columns(metadata, key):
    return {column["name"].casefold(): column for column in metadata[key].get("columns", [])
            if isinstance(column, dict) and isinstance(column.get("name"), str)}


def _resolve(column, metadata, sources):
    if not isinstance(column, exp.Column) or column.is_star or column.db or column.catalog:
        return None
    choices = [(alias, table, _columns(metadata, table)[column.name.casefold()])
               for alias, table in sources.values()
               if (not column.table or alias.casefold() == column.table.casefold())
               and column.name.casefold() in _columns(metadata, table)]
    return choices[0] if len(choices) == 1 else None


def _operand(aggregate):
    value = aggregate.this
    if isinstance(value, exp.Cast):
        value = value.this
    return value if isinstance(value, exp.Column) and not value.is_star else None


def _aggregate_columns(tree, question, metadata, sources):
    question_words = _words(question) - GENERIC
    replacements = {}
    for aggregate in tree.walk():
        if not isinstance(aggregate, (exp.Avg, exp.Sum)):
            continue
        cues = r"\b(?:average|mean)\b" if isinstance(aggregate, exp.Avg) else r"\b(?:total|sum|combined)\b"
        if not re.search(cues, question, re.I):
            continue
        operand = _operand(aggregate)
        resolved = _resolve(operand, metadata, sources)
        if resolved is None:
            continue
        alias, table, current = resolved
        current_words = _words(current["name"]) - GENERIC
        score = len(current_words & question_words)
        candidates = []
        for name, column in _columns(metadata, table).items():
            words = _words(column["name"]) - GENERIC
            if name == current["name"].casefold() or _words(column["name"]) & _words("id code key name"):
                continue
            missing, extra = words - current_words, current_words - words
            # Keep the measured subject stable. Single-token measurement
            # substitutions require both words to be ordinary measure terms.
            related = bool(words & current_words) or (words <= MEASURES and current_words <= MEASURES)
            if (not related or len(missing) != 1 or len(extra) != 1
                    or not missing <= question_words or extra & question_words):
                continue
            new_score = len(words & question_words)
            if new_score > score:
                candidates.append((new_score, column["name"]))
        if not candidates:
            continue
        best = max(value[0] for value in candidates)
        selected = [name for value, name in candidates if value == best]
        if len(selected) == 1:
            replacements[(alias.casefold(), current["name"].casefold())] = (alias, selected[0])
    if not replacements:
        return None
    changed = tree.copy()
    for aggregate in changed.walk():
        if isinstance(aggregate, (exp.Avg, exp.Sum)):
            operand = _operand(aggregate)
            resolved = _resolve(operand, metadata, sources)
            if resolved:
                alias, _, column = resolved
                replacement = replacements.get((alias.casefold(), column["name"].casefold()))
                if replacement:
                    operand.replace(exp.column(replacement[1], table=replacement[0], quoted=True))
    return changed


def _alphabetical(tree, question, metadata, sources):
    if not re.search(r"\balphabetical(?:ly)?\b", question, re.I):
        return None
    order = tree.args.get("order")
    if order is None or tree.args.get("group") or tree.args.get("having") or tree.args.get("distinct"):
        return None
    targets = []
    question_words = _words(question)
    for index, term in enumerate(order.expressions):
        column = term.this
        resolved = _resolve(column, metadata, sources)
        if resolved is None or "id" not in _words(column.name):
            continue
        alias, table, source_column = resolved
        links = []
        for fk in metadata[table].get("foreign_keys", []):
            if not all(isinstance(fk.get(key), str) for key in ("from", "table", "to")):
                continue
            target = fk["table"].casefold()
            if (fk["from"].casefold() != source_column["name"].casefold() or target not in metadata
                    or fk["to"].casefold() not in _columns(metadata, target)):
                continue
            if not _words(metadata[target]["name"]) & question_words:
                continue
            links.append((target, fk["to"]))
        if len(set(links)) != 1:
            continue
        target, target_key = links[0]
        candidates = []
        for candidate in _columns(metadata, target).values():
            words = _words(candidate["name"])
            kind = candidate.get("type", "").lower()
            if words & _words("id code key") or (kind and not any(x in kind for x in ("text", "char", "clob"))):
                continue
            if not words & DESCRIPTIVE:
                continue
            score = len(words & question_words) * 3 + len(words & DESCRIPTIVE)
            candidates.append((score, candidate["name"]))
        if not candidates:
            continue
        best = max(value for value, _ in candidates)
        names = [name for value, name in candidates if value == best]
        if len(names) == 1:
            targets.append((index, alias, source_column["name"], target, target_key, names[0]))
    if len(targets) != 1:
        return None
    index, source_alias, source_key, target, target_key, description = targets[0]
    existing = [alias for alias, table in sources.values() if table == target]
    if len(existing) > 1:
        return None
    changed = tree.copy()
    if existing:
        target_alias = existing[0]
    else:
        suffix = 1
        while f"_lexical_{suffix}".casefold() in sources:
            suffix += 1
        target_alias = f"_lexical_{suffix}"
        destination = exp.Table(this=exp.to_identifier(metadata[target]["name"], quoted=True),
                                alias=exp.TableAlias(this=exp.to_identifier(target_alias, quoted=True)))
        condition = exp.EQ(this=exp.column(source_key, table=source_alias, quoted=True),
                           expression=exp.column(target_key, table=target_alias, quoted=True))
        changed.set("joins", [join.copy() for join in tree.args.get("joins", [])]
                    + [exp.Join(this=destination, on=condition)])
    changed.args["order"].expressions[index].set("this", exp.column(description, table=target_alias, quoted=True))
    return changed


def _proposals(sql, question, schema):
    if not isinstance(question, str):
        return []
    tree = _parse(sql)
    context = _context(tree, schema) if tree is not None else None
    if context is None:
        return []
    metadata, sources = context
    return [candidate for candidate in (_aggregate_columns(tree, question, metadata, sources),
                                        _alphabetical(tree, question, metadata, sources)) if candidate is not None]


def lexical_repairs(sql, feedback, example, question, schema):
    for candidate in _proposals(sql, question, schema):
        text = candidate.sql(dialect="sqlite")
        if len(text.encode("utf-8")) <= 16000:
            yield text


def lexical_penalty(sql, example, question, schema=None):
    if schema is not None:
        return len(_proposals(sql, question, schema)) * 2
    tree = _parse(sql)
    if tree is None or not isinstance(question, str) or not re.search(r"\balphabetical(?:ly)?\b", question, re.I):
        return 0
    order = tree.args.get("order")
    return int(order is not None and any(isinstance(item.this, exp.Column) and "id" in _words(item.this.name)
                                        for item in order.expressions))
