"""Construct direct extrema when the measured column and outputs share a table."""
import re
from sqlglot import exp
from header_repair import _sql_table, _sql_col
from relation_rules import _schema
from single_table_rules import _words, _outputs


def extreme_candidates(sql, question, schema, example):
    metadata = _schema(schema)
    outputs = _outputs(example.get("columns", []))
    if not metadata or not outputs or not all(isinstance(c,exp.Column) for c in outputs):
        return
    text = question.casefold()
    cue = re.search(r"\b(highest|largest|greatest|maximum|lowest|smallest|minimum|fewest|least|most|"
                    r"oldest|youngest|latest|earliest|newest)\b", text)
    if not cue or re.search(r"""\b(?:except|without|between|before|after|over|under|above|below|more than|less than|who have)\b|["']|\d""", text):
        return
    key = cue[1]
    count = bool(re.search(r"\b(?:number of|fewest|frequently|frequency|popular|most common|least common)\b",text))
    rank_desc = key in {"highest","largest","greatest","maximum","most","oldest","latest","newest"}
    words = _words(question)
    candidates = []
    for t,table in metadata.items():
        if not all(c.name.casefold() in table["columns"] for c in outputs):
            continue
        named = _words(table["name"]) <= words
        if count:
            if not named:
                continue
            candidates.append(((2,0),t,None))
            continue
        next_words = _words(" ".join(text[cue.end():].strip().split()[:3]))
        for col in table["columns"]:
            cw=_words(col)
            if not cw:
                continue
            direct = cw <= next_words
            age = key in {"youngest","oldest"} and "age" in cw
            temporal = key in {"latest","earliest","newest"} or "recently" in next_words
            dated = temporal and bool(cw & {"year","date","time"})
            if not (direct or age or dated):
                continue
            candidates.append(((3*direct+2*age+dated,int(named),len(cw & words)),t,col))
    candidates.sort(key=lambda r:r[0],reverse=True)
    if not candidates or len(candidates)>1 and candidates[0][0]==candidates[1][0]:
        return
    _,t,col = candidates[0]
    table = metadata[t]
    fields = [_sql_col(table["columns"][c.name.casefold()],table["name"]) for c in outputs]
    q = exp.select(*fields).from_(_sql_table(table["name"]))
    if count:
        q.set("group",exp.Group(expressions=[c.copy() for c in fields]))
        sort = exp.Count(this=exp.Star())
    else:
        sort = _sql_col(table["columns"][col],table["name"])
    q.set("order",exp.Order(expressions=[exp.Ordered(this=sort,desc=rank_desc,nulls_first=not rank_desc)]))
    q.set("limit",exp.Limit(expression=exp.Literal.number(1)))
    yield q.sql(dialect="sqlite")
