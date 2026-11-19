"""Spend otherwise unused calls on additional schema-grounded repairs."""
from rescue import rescue_queries
from submission import create_agent as create_base


class RemainingChecks:
    def __init__(self, base):
        self.base = base
        self.fallback = None
        self.offers = None

    def act(self, obs):
        if not obs["history"]:
            self.fallback = self.offers = None
        if self.offers is None:
            action = self.base.act(obs)
            if action["tool"] != "submit" or obs["remaining"] == 0:
                return action
            self.fallback = action["sql"]
            self.offers = rescue_queries(obs["history"], obs["remaining"], self.fallback,
                                         obs["task"]["question"], obs["example"], obs["ordered"])
        else:
            last = obs["history"][-1]
            if last["observation"].get("matches_example"):
                self.offers = None
                return {"tool":"submit","sql":last["sql"]}
        sql = next(self.offers, None) if obs["remaining"] else None
        if sql is not None:
            return {"tool":"check","sql":sql}
        self.offers = None
        return {"tool":"submit","sql":self.fallback}


def create_agent(checkpoint=None):
    return RemainingChecks(create_base(checkpoint))
