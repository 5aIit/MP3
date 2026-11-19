"""Checked repairs for gratuitous joins and implied ordering."""
import re
from sqlglot import exp
from relation_rules import _select
from simplification_rules import _binding, _context
from single_table_rules import _words


def prune(sql, question, schema):
    tree = _select(sql)
    if not tree or not tree.args.get("joins") or not tree.args.get("where"):
        return
    if any(tree.args.get(k) for k in ("group","having","limit","offset")):
        return
    if not all(isinstance(e.unalias(),exp.Column) and not e.unalias().is_star for e in tree.expressions):
        return
    aliases = _context(tree,schema)
    if not aliases:
        return
    required = set()
    for node in [*tree.expressions,tree.args["where"],tree.args.get("order")]:
        if node is None:
            continue
        for col in node.find_all(exp.Column):
            resolved = _binding(col,aliases)
            if resolved is None:
                return
            required.add(resolved[0])
    # Restrict to explicit value filters so unmentioned relationship semantics
    # cannot be discarded solely because a table is absent from the projection.
    values = [v.this for v in tree.args["where"].find_all(exp.Literal)]
    if not values or not all(v.casefold() in question.casefold() for v in values):
        return
    words = _words(question)
    for alias,(table,_) in aliases.items():
        name_words = _words(table.name) - {"in","the","of","to","ref","detail","record"}
        if name_words and name_words <= words:
            required.add(alias)
    if len(required)!=1:
        return
    owner = next(iter(required))
    for join in tree.args["joins"]:
        on = join.args.get("on")
        if (join.args.get("side") or join.args.get("kind","").upper() not in ("","INNER")
                or not isinstance(on,exp.EQ) or not isinstance(on.this,exp.Column)
                or not isinstance(on.expression,exp.Column)):
            return
    tree.set("from",exp.From(this=aliases[owner][0].copy()))
    tree.set("joins",[])
    yield tree.sql(dialect="sqlite")


def order_default(sql,question,ordered):
    if not ordered or not re.search(r"\b(?:order(?:ed)?|sort(?:ed)?)\s+by\b",question,re.I):
        return
    if re.search(r"\b(?:asc\w*|desc\w*|decreas\w*|increas\w*|highest|lowest|most|least|top|first|"
                 r"last|latest|oldest|newest|youngest|largest|smallest|reverse|high|low|best|worst|earliest)\b",question,re.I):
        return
    tree=_select(sql)
    if not tree or tree.args.get("limit") or not tree.args.get("order"):
        return
    terms=tree.args["order"].expressions
    if not any(t.args.get("desc") for t in terms) or not all(isinstance(t.this,exp.Column) for t in terms):
        return
    for term in terms:
        term.set("desc",False)
        term.set("nulls_first",True)
    yield tree.sql(dialect="sqlite")
