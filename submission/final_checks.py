"""Spend remaining calls on proposed alternatives after the base policy finishes."""
from cardinality_rules import cardinality_candidates


class FinalChecks:
    def __init__(self, agent, sql_key, *, audit_cardinality=False, repair_sets=False,
                 reconstruct_sources=False, audit_relational_intent=False):
        self.agent = agent
        self.sql_key = sql_key
        self.audit_cardinality = audit_cardinality
        self.repair_sets = repair_sets
        self.reconstruct_sources = reconstruct_sources
        self.audit_relational_intent = audit_relational_intent
        self.waiting_for_schema = None
        self.pending = None

    def proposals(self, obs, original):
        checks = [event for event in obs["history"] if event["tool"] == "check"]
        matches = [event for event in checks if event["observation"].get("matches_example")]
        schema = next((e["observation"] for e in obs["history"] if e["tool"] == "schema"), None)
        if self.audit_relational_intent and schema:
            from absent_relation_rules import absent_relation_repairs, relationship_cue
            from group_repair import group_action
            question = obs["task"]["question"]
            if relationship_cue(question):
                yield from absent_relation_repairs(obs["history"], obs["example"], question, schema)
            group = group_action(obs, original, bool(matches), self.sql_key)
            if group and group["tool"] == "check":
                yield group["sql"]
        if self.repair_sets:
            from set_rules import set_candidates
            yield from set_candidates(obs["history"], obs["example"], obs["task"]["question"],
                                      schema, original, bool(matches))
        if self.reconstruct_sources and not matches and schema:
            from single_table_rules import single_table_repairs
            from compact_join_rules import compact_join_repairs
            from header_repair import header_candidates
            from itertools import zip_longest
            args = (obs["history"], obs["example"], obs["task"]["question"], schema)
            # Give each distinct construction one opportunity before a second variant.
            for group in zip_longest(single_table_repairs(*args), header_candidates(*args),
                                     compact_join_repairs(*args)):
                yield from (sql for sql in group if sql is not None)
        if self.audit_cardinality:
            for event in checks:
                if matches and event["sql"] != original:
                    continue
                yield from cardinality_candidates(event, obs["example"], obs["task"]["question"])

    def after_submit(self, obs, original):
        if obs["remaining"] <= 0:
            return {"tool": "submit", "sql": original}
        if (self.audit_relational_intent and obs["remaining"] >= 2
                and not any(e["tool"] == "schema" for e in obs["history"])):
            from absent_relation_rules import relationship_cue
            from group_repair import group_action
            matched = any(e["tool"] == "check" and e["observation"].get("matches_example") for e in obs["history"])
            if relationship_cue(obs["task"]["question"]) or group_action(obs, original, matched, self.sql_key):
                self.waiting_for_schema = original
                return {"tool": "schema"}
        by_key = {self.sql_key(e["sql"]): e for e in obs["history"] if e["tool"] == "check"}
        offers = []
        seen = set()
        for sql in self.proposals(obs, original):
            if len(sql.encode("utf-8")) > 16000:
                continue
            key = self.sql_key(sql)
            if key in by_key or key in seen:
                continue
            seen.add(key)
            offers.append(sql)
            if len(offers) >= obs["remaining"]:
                break
        if not offers:
            return {"tool": "submit", "sql": original}
        self.pending = (original, iter(offers))
        return {"tool": "check", "sql": next(self.pending[1])}

    def act(self, observation):
        history = observation.get("history", [])
        if not history:
            self.pending = None
            self.waiting_for_schema = None
        if self.waiting_for_schema is not None:
            original = self.waiting_for_schema
            self.waiting_for_schema = None
            return self.after_submit(observation, original)
        if self.pending is None:
            action = self.agent.act(observation)
            if action["tool"] == "submit":
                return self.after_submit(observation, action["sql"])
            return action
        original, offers = self.pending
        last = history[-1]
        if last["observation"].get("matches_example"):
            self.pending = None
            return {"tool": "submit", "sql": last["sql"]}
        next_sql = next(offers, None) if observation["remaining"] > 0 else None
        if next_sql is None:
            self.pending = None
            return {"tool": "submit", "sql": original}
        return {"tool": "check", "sql": next_sql}
