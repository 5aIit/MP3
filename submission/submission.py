"""Offline SQL repair controller using only observations and official tool actions.

No benchmark files, database connections, network calls, or learned assets are
used here. The complete decision state is reconstructed from tool history.
"""
from difflib import SequenceMatcher
from functools import lru_cache
import json
from pathlib import Path
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from repair_rules import projection_repairs, projection_penalty


MAX_SQL_BYTES = 16_000


def _parse(sql):
    """Filter generated candidates; the original draft still gets checked verbatim."""
    if not isinstance(sql, str) or not sql.strip():
        return None
    if len(sql.encode("utf-8")) > MAX_SQL_BYTES:
        return None
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except (ParseError, TokenError, RecursionError):
        return None
    if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.SetOperation)):
        return None
    tree = statements[0]
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Command, exp.Into)
    if any(isinstance(node, forbidden) for node in tree.walk()):
        return None
    return tree


@lru_cache(maxsize=256)
def _key(sql):
    tree = _parse(sql)
    if tree is None:
        return ("raw", sql)
    # In a single SQLite source, an unquoted bound qualifier identifies the
    # same column as the bare name. Keep aliases used by later clauses, quoted
    # identifiers (SQLite also accepts double-quoted strings), and nested SQL.
    if (isinstance(tree, exp.Select) and not tree.args.get("with") and not tree.args.get("joins")
            and isinstance(tree.args.get("from"), exp.From)
            and isinstance(tree.args["from"].this, exp.Table)
            and not any(isinstance(n, (exp.Subquery, exp.SetOperation)) or
                        isinstance(n, exp.Identifier) and n.args.get("quoted") for n in tree.walk())):
        table = tree.args["from"].this
        columns = list(tree.find_all(exp.Column))
        if (not table.db and not table.catalog and isinstance(table.this, exp.Identifier)
                and all(not c.db and not c.catalog and
                        (not c.table or c.table.casefold()==table.alias_or_name.casefold()) for c in columns)):
            aliases = {p.alias.casefold() for p in tree.expressions if p.alias}
            used_alias = any(not c.table and c.name.casefold() in aliases
                             for k in ("order", "group", "having", "where", "qualify")
                             if tree.args.get(k) for c in tree.args[k].find_all(exp.Column))
            if not used_alias and not tree.find(exp.Window):
                for col in columns:
                    col.set("table",None)
                tree.set("expressions",[p.unalias().copy() for p in tree.expressions])
                table.set("alias",None)
    # Identifier case and whitespace are immaterial; string literal case is not.
    return ("sql", tree.sql(dialect="sqlite", normalize=True, comments=False))


def _candidate(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    fence = re.fullmatch(r"```(?:sql|sqlite)?\s*\n?(.*?)\n?```", text, re.I | re.S)
    if fence:
        text = fence.group(1).strip()
    return text if _parse(text) is not None else None


def _error_message(event):
    return (event.get("observation", {}).get("error") or {}).get("message", "")


def _needs_schema(events):
    return any(re.search(r"(?:no such (?:column|table)|ambiguous column name):", _error_message(e), re.I)
               for e in events)


def _closest(name, choices):
    """Only accept a strong, unambiguous spelling correction."""
    scores = sorted(((SequenceMatcher(None, name.casefold(), c.casefold()).ratio(), c)
                     for c in set(choices)), reverse=True)
    if not scores or scores[0][0] < 0.8:
        return None
    if len(scores) > 1 and scores[0][0] - scores[1][0] < 0.15:
        return None
    return scores[0][1]


def _identifier_repairs(event, schema):
    """Repair one reported identifier in a single SELECT scope using tool schema."""
    tree = _parse(event["sql"])
    if tree is None or not isinstance(tree, exp.Select):
        return
    # Avoid changing a similarly named identifier inside another query scope.
    if len(list(tree.find_all(exp.Select))) != 1 or tree.args.get("with"):
        return
    match = re.search(r"(no such column|no such table|ambiguous column name):\s*(\S+)",
                      _error_message(event), re.I)
    if not match:
        return
    kind, missing = match.group(1).lower(), match.group(2).strip('`"[]')
    tables = {table["name"].casefold(): table for table in schema}
    if kind == "no such table":
        replacement = _closest(missing, [t["name"] for t in schema])
        if replacement:
            changed = tree.copy()
            for node in changed.find_all(exp.Table):
                if node.name.casefold() == missing.casefold():
                    node.set("this", exp.to_identifier(replacement, quoted=True))
            yield changed.sql(dialect="sqlite")
        return
    aliases = {}
    for table in tree.find_all(exp.Table):
        metadata = tables.get(table.name.casefold())
        if metadata:
            aliases[table.alias_or_name] = [column["name"] for column in metadata["columns"]]
    parts = missing.rsplit(".", 1)
    qualifier, name = (parts if len(parts) == 2 else (None, parts[0]))
    options = []
    for alias, columns in aliases.items():
        if qualifier and qualifier.casefold() != alias.casefold():
            continue
        if kind == "ambiguous column name":
            options.extend((alias, c) for c in columns if c.casefold() == name.casefold())
        else:
            replacement = _closest(name, columns)
            if replacement:
                options.append((alias, replacement))
    # Ambiguity can have multiple tool-checked alternatives; spelling repairs
    # must identify a unique column across all tables in this SELECT scope.
    if kind == "no such column" and len(options) != 1:
        return
    for alias, replacement in options:
        changed = tree.copy()
        for node in changed.find_all(exp.Column):
            if node.name.casefold() == name.casefold() and (
                    (not qualifier and not node.table)
                    or (qualifier and node.table.casefold() == qualifier.casefold())):
                node.set("this", exp.to_identifier(replacement, quoted=True))
                node.set("table", exp.to_identifier(alias, quoted=True))
        yield changed.sql(dialect="sqlite")


def _multiplicity_repairs(event, example):
    """Use row-count feedback to propose DISTINCT changes, then check them."""
    feedback = event["observation"]
    if not feedback.get("ok") or len(feedback.get("columns", [])) != len(example.get("columns", [])):
        return
    tree = _parse(event["sql"])
    expected = example.get("row_count")
    actual = feedback.get("row_count")
    if not isinstance(tree, exp.Select) or not isinstance(expected, int) or not isinstance(actual, int):
        return
    if tree.args.get("limit") or tree.args.get("offset"):
        return
    if tree.args.get("distinct") and expected > actual:
        tree.set("distinct", None)
        yield tree.sql(dialect="sqlite")
    elif not tree.args.get("distinct") and expected < actual:
        rows = feedback.get("rows", [])
        if len({repr(row) for row in rows}) < len(rows):
            tree.set("distinct", exp.Distinct())
            yield tree.sql(dialect="sqlite")


def _fallback(events, original, example):
    """Prefer a checked executable result with the public result's shape.

    This is only a fallback heuristic, never a claim of semantic correctness.
    Ties prefer the original draft, then the earliest candidate checked.
    """
    successful = [event for event in events if event["observation"].get("ok")]
    if not successful:
        return original

    def rank(event):
        feedback = event["observation"]
        expected_rows = example.get("row_count", 0)
        return (len(feedback.get("columns", [])) == len(example.get("columns", [])),
                -abs(feedback.get("row_count", 0) - expected_rows),
                event["sql"] == original)

    return max(successful, key=rank)["sql"]


def _weak_example(example):
    """An empty/all-NULL example carries little evidence about query semantics."""
    rows = example.get("rows", [])
    return example.get("row_count") == 0 or (
        example.get("row_count") == 1 and len(rows) == 1
        and bool(rows[0]) and all(value is None for value in rows[0]))


def _zero_count_example(sql, example):
    """A scalar zero COUNT is weak evidence about filters, unlike an arbitrary zero."""
    rows = example.get("rows", [])
    if (example.get("row_count") != 1 or len(rows) != 1 or len(rows[0]) != 1
            or type(rows[0][0]) not in (int, float) or rows[0][0] != 0):
        return False
    tree = _parse(sql)
    return (isinstance(tree, exp.Select) and len(tree.expressions) == 1
            and isinstance(tree.expressions[0].unalias(), exp.Count)
            and not tree.args.get("group") and not tree.args.get("having"))


class RepairAgent:
    def __init__(self, *, audit_weak_matches=True, repair_projection=True,
                 repair_identifiers=True, repair_distinct=True,
                 expand_order_aliases=False, audit_zero_counts=False,
                 audit_all_matches=False, prefer_hint_matches=False,
                 repair_semantics=False, repair_structures=False, repair_relations=False,
                 repair_superlatives=False, repair_values=False, repair_negation=False,
                 combine_projection=False, repair_lexical=False, repair_simplification=False,
                 repair_entities=False, repair_counts=False, rank_candidates=False,
                 prioritize_values=False, audit_early_hints=False):
        self.audit_weak_matches = audit_weak_matches
        self.repair_projection = repair_projection
        self.repair_identifiers = repair_identifiers
        self.repair_distinct = repair_distinct
        self.expand_order_aliases = expand_order_aliases
        self.audit_zero_counts = audit_zero_counts
        self.audit_all_matches = audit_all_matches
        self.prefer_hint_matches = prefer_hint_matches
        self.repair_semantics = repair_semantics
        self.repair_structures = repair_structures
        self.repair_relations = repair_relations
        self.repair_superlatives = repair_superlatives
        self.repair_values = repair_values
        self.repair_negation = repair_negation
        self.combine_projection = combine_projection
        self.repair_lexical = repair_lexical
        self.repair_simplification = repair_simplification
        self.repair_entities = repair_entities
        self.repair_counts = repair_counts
        self.rank_candidates = rank_candidates
        self.prioritize_values = prioritize_values
        self.audit_early_hints = audit_early_hints

    def act(self, observation):
        original = observation["issue_sql"]
        history = observation.get("history", [])
        remaining = observation["remaining"]
        example = observation.get("example", {})
        question = observation.get("task", {}).get("question", "")
        probe_event = next((event for event in history if event["tool"] == "execute"
                            and event.get("sql", "").startswith("/* value-grounding */ ")), None)
        probe_feedback = probe_event["observation"] if probe_event is not None else None
        queries = [event for event in history if event["tool"] in ("check", "execute")
                   and event is not probe_event]
        schema_event = next((e for e in history if e["tool"] == "schema"), None)
        schema = schema_event["observation"] if schema_event is not None else None

        if self.repair_semantics:
            from semantic_rules import semantic_penalty, semantic_repairs
        if self.repair_structures:
            from structural_rules import structural_penalty, structural_repairs
        if self.repair_relations:
            from relation_rules import relation_penalty, relation_repairs
        if self.repair_superlatives:
            from superlative_rules import superlative_penalty, superlative_repairs
        if self.repair_values:
            from value_rules import value_probe, value_penalty, value_repairs
        if self.repair_negation:
            from negation_rules import negation_penalty, negation_repairs
        if self.repair_lexical:
            from lexical_rules import lexical_penalty, lexical_repairs
        if self.repair_simplification:
            from simplification_rules import simplification_penalty, simplification_repairs
        if self.repair_entities:
            from entity_rules import entity_penalty, entity_repairs
        if self.repair_counts:
            from count_rules import count_penalty, count_repairs

        def prepare_candidate(sql):
            if not self.combine_projection or not self.repair_projection:
                return sql
            tree = _parse(sql)
            if not isinstance(tree, exp.Select):
                return sql
            # AST labels propose a safe projection before paying for execution.
            # This is synthetic metadata for proposal generation only; it is
            # never written to history or used as a claimed tool observation.
            labels = [item.alias or item.unalias().name or item.unalias().sql(dialect="sqlite")
                      for item in tree.expressions]
            proposed = projection_repairs(sql, {"ok": True, "columns": labels}, example,
                expand_order_aliases=self.expand_order_aliases)
            return next(proposed, sql)

        def semantic_score(sql):
            penalty = semantic_penalty(sql, example, question) if self.repair_semantics else 0
            if self.repair_structures:
                penalty += structural_penalty(sql, example, question, schema)
            if self.repair_relations:
                penalty += relation_penalty(sql, example, question, schema)
            if self.repair_superlatives:
                penalty += superlative_penalty(sql, example, question, schema)
            if self.repair_values:
                penalty += value_penalty(sql, example, question, schema, probe_feedback)
            if self.repair_negation:
                penalty += negation_penalty(sql, example, question, schema)
            if self.repair_lexical:
                penalty += lexical_penalty(sql, example, question, schema)
            if self.repair_simplification:
                penalty += simplification_penalty(sql, example, question, schema)
            if self.repair_entities:
                penalty += entity_penalty(sql, example, question, schema)
            if self.repair_counts:
                penalty += count_penalty(sql, example, question, schema)
            return penalty

        matches = [event for event in queries if event["tool"] == "check"
                   and event["observation"].get("ok")
                   and event["observation"].get("matches_example")]
        weak_audit = self.audit_weak_matches and _weak_example(example)
        zero_audit = self.audit_zero_counts and _zero_count_example(original, example)
        semantic_audit = bool(matches) and min(semantic_score(e["sql"]) for e in matches) > 0
        audit = weak_audit or zero_audit or self.audit_all_matches or semantic_audit
        hint_event = next((e for e in history if e["tool"] == "repair_hint"), None)

        def finish():
            if matches:
                # Headers are only ranking evidence, never an unchecked answer.
                # A checked shared repair provides independent evidence when the
                # public example is too weak to distinguish it from the draft.
                preferred = max(matches, key=lambda event: (
                    -semantic_score(event["sql"]),
                    -projection_penalty(event["sql"], example, **(
                        {"expand_order_aliases": True} if self.expand_order_aliases else {}))
                    if self.repair_projection else 0,
                    event["sql"] != original if (weak_audit or zero_audit or self.prefer_hint_matches)
                    else event["sql"] == original))
                return {"tool": "submit", "sql": preferred["sql"]}
            return {"tool": "submit", "sql": _fallback(queries, original, example)}

        if matches and not audit:
            # A draft may match the public rows by coincidence. When it would
            # otherwise finish immediately, try the shared repairs in their
            # original order and require a genuine check of any replacement.
            if (self.audit_early_hints and schema_event is None and queries
                    and queries[0]["sql"] == original
                    and queries[0]["observation"].get("matches_example")
                    and not semantic_score(original)):
                if hint_event is None and remaining >= 2:
                    return {"tool": "repair_hint"}
                if hint_event is not None:
                    checked_hints = {_key(e["sql"]): e
                                     for e in queries if e["tool"] == "check"}
                    draft_key = _key(original)
                    hints = hint_event["observation"]
                    for text in hints if isinstance(hints, list) else ():
                        hint = _candidate(text)
                        if hint is None:
                            continue
                        key = _key(hint)
                        if key == draft_key:
                            continue
                        if key not in checked_hints:
                            if remaining > 0:
                                return {"tool": "check", "sql": hint}
                        elif checked_hints[key]["observation"].get("matches_example"):
                            return {"tool": "submit", "sql": checked_hints[key]["sql"]}
                return {"tool": "submit", "sql": original}
            return finish()
        if remaining <= 0:
            return finish()
        if not any(event["sql"] == original for event in queries):
            return {"tool": "check", "sql": original}

        if hint_event is None:
            if matches and remaining < 2:
                return finish()
            return {"tool": "repair_hint"}

        checked = {_key(event["sql"]) for event in queries}

        def checked_value_candidate():
            if not self.repair_values or probe_event is None:
                return None
            for event in reversed(queries):
                if not event["observation"].get("ok"):
                    continue
                for sql in value_repairs(event["sql"], event["observation"], example,
                                         question, schema, probe_feedback):
                    sql = prepare_candidate(sql)
                    if _parse(sql) is not None and _key(sql) not in checked:
                        return {"tool": "check", "sql": sql}
            return None
        candidates = hint_event["observation"]
        if not isinstance(candidates, list):
            candidates = []
        for text in candidates:
            sql = _candidate(text)
            if sql is not None and _key(sql) not in checked:
                return {"tool": "check", "sql": sql}

        if self.repair_projection:
            for event in queries:
                if not event["observation"].get("ok"):
                    continue
                for sql in projection_repairs(event["sql"], event["observation"], example, **(
                        {"expand_order_aliases": True} if self.expand_order_aliases else {})):
                    if _parse(sql) is not None and _key(sql) not in checked:
                        return {"tool": "check", "sql": sql}

        extra_rules = (self.repair_semantics or self.repair_structures or self.repair_relations
                       or self.repair_superlatives or self.repair_values or self.repair_negation
                       or self.repair_lexical or self.repair_simplification
                       or self.repair_entities or self.repair_counts)
        inspect_matches = (self.repair_structures or self.repair_relations or self.repair_values
                           or self.repair_negation) and (
            weak_audit or zero_audit or self.audit_all_matches)
        if extra_rules and (not matches or semantic_audit or inspect_matches):
            if schema_event is None and remaining >= 2:
                return {"tool": "schema"}
            if schema_event is not None:
                if self.prioritize_values and self.repair_values and not matches:
                    if probe_event is not None:
                        action = checked_value_candidate()
                        if action is not None:
                            return action
                    elif 2 <= remaining <= 3 and any(event["observation"].get("ok")
                            and event["observation"].get("row_count") == 0 for event in queries):
                        probe = value_probe(queries, question, schema)
                        if probe is not None:
                            return {"tool": "execute", "sql": probe}
                generators = []
                if self.repair_relations:
                    generators.append(relation_repairs)
                if self.repair_superlatives:
                    generators.append(superlative_repairs)
                if self.repair_negation:
                    generators.append(negation_repairs)
                if self.repair_lexical:
                    generators.append(lexical_repairs)
                if self.repair_simplification:
                    generators.append(simplification_repairs)
                if self.repair_entities:
                    generators.append(entity_repairs)
                if self.repair_counts:
                    generators.append(count_repairs)
                if self.repair_structures:
                    generators.append(structural_repairs)
                if self.repair_semantics:
                    generators.append(semantic_repairs)
                pool = {}
                for event in queries:
                    for generate in generators:
                        for sql in generate(event["sql"], event["observation"], example, question, schema):
                            sql = prepare_candidate(sql)
                            if _parse(sql) is not None and _key(sql) not in checked:
                                if not self.rank_candidates:
                                    return {"tool": "check", "sql": sql}
                                pool.setdefault(_key(sql), sql)
                if pool:
                    def candidate_rank(sql):
                        tree = _parse(sql)
                        return (semantic_score(sql),
                                projection_penalty(sql, example, expand_order_aliases=self.expand_order_aliases),
                                len(list(tree.find_all(exp.Table))), len(list(tree.walk())))
                    return {"tool": "check", "sql": min(pool.values(), key=candidate_rank)}
                if self.repair_values:
                    if probe_event is None and remaining >= 2:
                        probe = value_probe(queries, question, schema)
                        if probe is not None:
                            return {"tool": "execute", "sql": probe}
                    if probe_event is not None:
                        action = checked_value_candidate()
                        if action is not None:
                            return action

        if matches:
            return finish()

        if (self.repair_identifiers and schema_event is None
                and remaining >= 2 and _needs_schema(queries)):
            return {"tool": "schema"}
        for event in queries:
            repairs = list(_multiplicity_repairs(event, example)) if self.repair_distinct else []
            if self.repair_identifiers and schema_event is not None:
                repairs.extend(_identifier_repairs(event, schema_event["observation"]))
            for sql in repairs:
                if _parse(sql) is not None and _key(sql) not in checked:
                    return {"tool": "check", "sql": sql}
        return finish()


def create_agent(checkpoint=None):
    config = {}
    if checkpoint is not None:
        try:
            config = json.loads(Path(checkpoint).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("Checkpoint must be a valid local strategy JSON") from error
        allowed = {"audit_weak_matches", "repair_projection",
                   "repair_identifiers", "repair_distinct", "expand_order_aliases",
                   "audit_zero_counts", "audit_all_matches", "prefer_hint_matches",
                   "repair_semantics", "repair_structures", "repair_relations",
                   "repair_superlatives", "repair_values", "repair_negation",
                   "combine_projection", "repair_lexical", "repair_simplification",
                   "repair_entities", "repair_counts", "rank_candidates", "prioritize_values",
                   "audit_early_hints", "audit_cardinality", "repair_sets", "reconstruct_sources",
                   "audit_relational_intent"}
        if (not isinstance(config, dict) or set(config) - allowed
                or any(type(value) is not bool for value in config.values())):
            raise ValueError("Strategy JSON accepts only boolean feature switches")
    followups = {key: config.pop(key, False) for key in
                 ("audit_cardinality", "repair_sets", "reconstruct_sources", "audit_relational_intent")}
    agent = RepairAgent(**config)
    if any(followups.values()):
        from final_checks import FinalChecks
        return FinalChecks(agent, _key, **followups)
    return agent
