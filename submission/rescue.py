"""Try bounded repairs only after the incumbent has no public match."""
import re
from sqlglot import exp
from submission import _key, _parse
from cardinality_rules import cardinality_candidates
from extreme_rules import extreme_candidates
from filter_rules import binding_candidates
from terminal_rules import order_default, prune


def literal_candidates(sql,question):
    tree=_parse(sql)
    if tree is None:
        return
    for index,node in enumerate(tree.walk()):
        if not isinstance(node,(exp.EQ,exp.NEQ,exp.In)) or not isinstance(node.this,exp.Column):
            continue
        vals=node.expressions if isinstance(node,exp.In) else [node.expression]
        if not vals or not all(isinstance(v,exp.Literal) and v.is_string for v in vals):
            continue
        col=node.this.name.casefold()
        value=vals[0].this
        if len(vals)==1 and isinstance(node,(exp.EQ,exp.NEQ)):
            # SQLite flags can store a binary integer although the draft
            # spells out the state (e.g. a status word rather than TRUE).
            norm=value.casefold().replace("_"," ")
            positive={"true","yes","active","enabled","logged in"}
            negative={"false","no","inactive","disabled","logged out"}
            if norm in positive|negative and re.search(r"(?<!\w)"+re.escape(norm)+r"(?!\w)",question,re.I):
                changed=tree.copy()
                list(changed.walk())[index].set("expression",exp.Literal.number(int(norm in positive)))
                yield changed.sql(dialect="sqlite")
            if value.isdigit() and re.fullmatch(r"[a-z]+",col) and re.search(
                    r"\b"+re.escape(col)+r"\b.{0,20}\b"+re.escape(value)+r"\b",question,re.I):
                changed=tree.copy()
                target=list(changed.walk())[index]
                target.set("this",exp.Collate(this=target.this.copy(),expression=exp.Var(this="NOCASE")))
                target.set("expression",exp.Literal.string(col+" "+value))
                yield changed.sql(dialect="sqlite")
        if not all(v.this.casefold() in question.casefold() and re.search(r"[a-z]",v.this,re.I) for v in vals):
            continue
        changed=tree.copy()
        target=list(changed.walk())[index]
        target.set("this",exp.Collate(this=target.this.copy(),expression=exp.Var(this="NOCASE")))
        yield changed.sql(dialect="sqlite")


def rescue_queries(history, remaining, incumbent, question, example, ordered):
    checks=[e for e in history if e["tool"]=="check"]
    if not remaining or any(e["observation"]["matches_example"] for e in checks):
        return
    schema=next((e["observation"] for e in history if e["tool"]=="schema"),None)
    checked={_key(e["sql"]) for e in checks}
    seen=set(checked)
    items=sorted(checks,key=lambda e:(e["sql"]!=incumbent,not e["observation"]["ok"]))
    for gen in (
        lambda e: extreme_candidates(e["sql"],question,schema,example),
        lambda e: order_default(e["sql"],question,ordered),
        lambda e: binding_candidates(e["sql"],question,schema,example),
        lambda e: literal_candidates(e["sql"],question),
        lambda e: cardinality_candidates(e,example,question),
        lambda e: prune(e["sql"],question,schema),
    ):
        for event in items:
            for sql in gen(event):
                if len(sql.encode())>16000:
                    continue
                key=_key(sql)
                if key not in seen:
                    seen.add(key)
                    yield sql
                    if len(seen)-len(checked)>=remaining:
                        return
