"""Plan a bounded count/group query from projection labels, wording, and declared keys."""
import re

from sqlglot import exp
from header_repair import _projection, _sql_col, _sql_table
from relation_rules import _graph, _schema, _select, _unique_path


def _stem(w):
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "xes", "sses")):
        return w[:-2]
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _words(text):
    return [_stem(t) for t in re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).casefold())]


_NUMBERS = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}


def _intent(question, count_projected):
    text = question.casefold()
    threshold = re.search(r"\b(at least|at most|more than|fewer than|less than|over)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(\w+)", text)
    frequency = re.search(r"\b(most|least)\s+(?:(?:used|seen|recorded)\s+)?(?:common|popular|frequent(?:ly)?|often)\b", text)
    top = re.search(r"\b(most|least|fewest|smallest|highest|greatest|largest|maximum|minimum)\s+(?:number (?:of )?)?([a-z]+)", text)
    group = count_projected and re.search(r"\b(each|per|every|number of|how many|count of|times)\b", text)
    if not (threshold or frequency or top or group):
        return None
    if re.search(r'''['"][^'"]{2,}['"]|\b(?:before|after|since|between|starts? with|ends? with|except|without)\b''', text):
        return None
    if threshold:
        token = threshold[2]
        number = int(token) if token.isdigit() else _NUMBERS[token]
        if number > 10000000:
            return None
        return ("threshold", threshold[1], number, _stem(threshold[3]))
    if frequency or top:
        if not frequency and "number" not in top[0] and not top[2].endswith("s"):
            return None
        desc = (frequency or top)[1] in {"most","highest","greatest","largest","maximum"}
        limit = re.search(r"\btop\s+([0-9]+)\b", text)
        return ("top", desc, int(limit[1]) if limit else 1, "" if frequency else _stem(top[2]))
    return ("group", None, None, "")


def group_candidates(history, example, question, schema, previous_match=None):
    metadata = _schema(schema)
    if not metadata:
        return
    projection = _projection(example.get("columns", []), metadata)
    if not projection or not any(isinstance(p, exp.Column) for p in projection):
        return
    if not all(isinstance(p, exp.Column) or isinstance(p, exp.Count) and isinstance(p.this, exp.Star)
               for p in projection):
        return
    intent = _intent(question, any(isinstance(p, exp.Count) for p in projection))
    if not intent:
        return
    mode, operator, number, noun = intent
    question_words = _words(question)
    table_words = {k: set(_words(t["name"])) for k,t in metadata.items()}
    count_nouns = {w for names in table_words.values() for w in names}
    if mode in ("top","threshold") and noun and noun not in count_nouns | {"time","that","occurrence"}:
        return
    if re.search(r"\b(?:day|month|year|old|price|age|salary|rate|duration|higher|lower)\b", question.casefold()) and mode == "threshold":
        return
    if re.search(r"\b(whose|with|where)\b", question.casefold()) and re.search(r"\b(?:named|called|equals?|below|above|majoring|starting)\b", question.casefold()):
        return
    previous = _select(previous_match) if previous_match else None
    if previous_match and previous is None:
        return
    check_trees = [t for e in history if e.get("tool") == "check" and (t := _select(e["sql"])) is not None]
    if previous is not None:
        check_trees.insert(0,previous)
    header_names = [p.name.casefold() for p in projection if isinstance(p,exp.Column)]
    owners = [k for k,t in metadata.items() if all(n in t["columns"] for n in header_names)]
    if not owners:
        return
    def source_evidence(table):
        for i,t in enumerate(check_trees):
            sources = {s.alias_or_name.casefold():s.name.casefold() for s in t.find_all(exp.Table)}
            if any(sources.get(c.table.casefold()) == table and c.name.casefold() in header_names
                   for item in t.expressions for c in item.find_all(exp.Column)):
                return len(check_trees)-i
        return 0
    def rank(k):
        return (source_evidence(k), table_words[k] <= set(question_words), len(table_words[k] & set(question_words)))
    owners.sort(key=rank,reverse=True)
    if len(owners)>1 and rank(owners[0]) == rank(owners[1]):
        return
    owner = owners[0]
    other_tables = [k for k in metadata if k != owner and table_words[k] <= set(question_words)]
    # Strongest lexical evidence is the counted noun after a superlative/threshold,
    # then a table named as the object of an explicit count or frequency phrase.
    count_span = re.search(r"\b(?:number of(?: times)?|across|among|often on|most|fewest|least)\b(.*)",question.casefold())
    scope = set(_words(count_span[1])) if count_span else set(question_words)
    def other_score(k):
        return (bool(noun) and noun in table_words[k], table_words[k] <= scope,
                len(table_words[k] & scope), len(table_words[k]))
    graph = _graph(metadata)
    labeled = [k for k in other_tables if (path := _unique_path(graph, owner, k)) and len(path) <= 3]
    labeled.sort(key=other_score, reverse=True)
    options = []
    if labeled:
        path = _unique_path(graph, owner, labeled[0])
        # The linking table itself stores each association. A final unused
        # lookup join can discard valid facts when the corresponding row is absent.
        options.append(path[:-1] if len(path)>1 and path[-2][1] not in labeled else path)
        if path not in options:
            options.append(path)
    incoming = []
    for k,t in metadata.items():
        if any(f.get("table","").casefold()==owner for f in t["foreign_keys"]):
            path = _unique_path(graph, owner, k)
            if path and len(path)==1:
                incoming.append(path)
    if not options and len(incoming)==1 and (noun in {"that","time"} or "frequent" in question.casefold()):
        options.append(incoming[0])
    options.append([])
    pk = [c["name"] for t in schema if t["name"].casefold()==owner for c in t["columns"] if c.get("pk")]
    seen = set()
    for path in options:
        if previous is not None:
            tables = {t.name.casefold() for t in previous.find_all(exp.Table)}
            wrong_relation = bool(path) and path[-1][1] not in tables
            order = previous.args.get("order")
            wrong_order = mode == "top" and (order is None or order.find(exp.Count) is None)
            group = previous.args.get("group")
            having = previous.args.get("having")
            wrong_threshold = mode == "threshold" and (having is None or not having.find(exp.Count))
            if not (wrong_relation or wrong_order or wrong_threshold or group is None):
                continue
            if previous.args.get("where"):
                continue
        head = metadata[owner]
        full = [_sql_col(head["columns"][p.name.casefold()], head["name"]) if isinstance(p,exp.Column)
                else p.copy() for p in projection]
        groups = [c.copy() for c in full if isinstance(c,exp.Column)]
        if path and len(pk) == 1:
            groups = [_sql_col(pk[0],head["name"])]
        q = exp.Select(expressions=full)
        q.set("from", exp.From(this=_sql_table(head["name"])))
        joins = []
        for a,b,left,right in path:
            first,other=metadata[a],metadata[b]
            condition=exp.EQ(this=_sql_col(first["columns"][left],first["name"]),
                             expression=_sql_col(other["columns"][right],other["name"]))
            joins.append(exp.Join(this=_sql_table(other["name"]),on=condition))
        q.set("joins", joins)
        q.set("group", exp.Group(expressions=groups))
        count = exp.Count(this=exp.Star())
        if mode == "top":
            q.set("order", exp.Order(expressions=[exp.Ordered(this=count,desc=operator)]))
            q.set("limit", exp.Limit(expression=exp.Literal.number(number)))
        elif mode == "threshold":
            kind = {"at least":exp.GTE,"at most":exp.LTE,"more than":exp.GT,"over":exp.GT,
                    "less than":exp.LT,"fewer than":exp.LT}[operator]
            q.set("having",exp.Having(this=kind(this=count,expression=exp.Literal.number(number))))
            if re.search(r"\b(?:alphabetical|ascending|descending) order\b", question.casefold()):
                q.set("order",exp.Order(expressions=[exp.Ordered(this=full[0].copy(),
                    desc="descending" in question.casefold())]))
        text=q.sql(dialect="sqlite")
        if text not in seen:
            seen.add(text)
            yield text


def group_action(observation, incumbent, has_match, sql_key):
    """Continue only after the current policy has chosen its terminal SQL."""
    if observation["remaining"] < 1:
        return None
    question = observation["task"]["question"]
    example = observation["example"]
    history = observation["history"]
    intent = _intent(question, any("count(" in s.lower() for s in example.get("columns", [])))
    if not intent:
        return None
    if has_match:
        tree = _select(incumbent)
        if tree is None or tree.args.get("where"):
            return None
        if intent[0] == "top" and tree.args.get("order") and tree.args["order"].find(exp.Count):
            return None
        if intent[0] == "threshold" and tree.args.get("having") and tree.args["having"].find(exp.Count):
            return None
        if intent[0] == "group" and not ("number of times" in question.casefold() or "how many" in question.casefold()):
            return None
    schema = next((e["observation"] for e in history if e["tool"] == "schema"), None)
    if not schema:
        return {"tool":"schema"} if observation["remaining"] >= 2 else None
    seen = {sql_key(e["sql"]) for e in history if e["tool"] == "check"}
    for sql in group_candidates(history, example, question, schema, incumbent if has_match else None):
        if len(sql.encode()) <= 16000 and sql_key(sql) not in seen:
            return {"tool":"check", "sql":sql}
    return None
