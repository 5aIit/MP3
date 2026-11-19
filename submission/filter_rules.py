"""Bind a comparison literal to a schema field explicitly named beside it."""
import re
from sqlglot import exp
from header_repair import _projection, _reconstruct, _sql_col
from relation_rules import _schema, _select, _sources
from single_table_rules import _words


FILLER = {"is","are","was","were","equal","equals","to","of","as","has","have",
          "the","a","an","with","at","over","above","below","than","greater","less",
          "more","fewer","not","least","most","whose","value"}
COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like)


def binding_candidates(sql, question, schema, example):
    tree = _select(sql)
    metadata = _schema(schema)
    if not tree or not metadata or not tree.args.get("where"):
        return
    sources = _sources(tree, metadata)
    projection = _projection(example.get("columns", []),metadata)
    if not sources or not projection:
        return
    used = {s["table"] for s in sources.values()}
    for index,node in enumerate(tree.walk()):
        if not isinstance(node,COMPARISONS) or not isinstance(node.this,exp.Column) or not isinstance(node.expression,exp.Literal):
            continue
        if node.find_ancestor(exp.Where) is None:
            continue
        literal = node.expression.this.strip("%")
        if not literal or len(literal)>120:
            continue
        positions = list(re.finditer(r"(?<!\w)"+re.escape(literal)+r"(?!\w)",question,re.I))
        if len(positions)!=1:
            continue
        point=positions[0]
        left = re.findall(r"[\w]+",question[:point.start()].lower())[-6:]
        right = re.findall(r"[\w]+",question[point.end():].lower())[:3]
        # Keep 'value' if it is part of the actual field (e.g. attribute value).
        nearby = [w for w in left if w not in FILLER-{"value"}][-3:]
        after = [w for w in right if w not in FILLER][:2]
        if not nearby and not after:
            continue
        ranked = []
        for table,record in metadata.items():
            for key,name in record["columns"].items():
                words = _words(name)
                if not words:
                    continue
                matches = [len(words) for seq in (nearby,after) if words <= _words(" ".join(seq)) and
                           (seq and (_words(seq[-1]) <= words if seq is nearby else _words(seq[0]) <= words))]
                if not matches:
                    continue
                table_words = _words(record["name"])
                if words <= {"name","id","code","value","number"} and not (table_words & _words(" ".join(nearby+after))):
                    continue
                ranked.append(((max(matches),len(table_words & _words(question)),table in used),table,key))
        ranked.sort(reverse=True)
        if not ranked or len(ranked)>1 and ranked[0][0]==ranked[1][0]:
            continue
        _,owner,key=ranked[0]
        old_table=sources.get(node.this.table.casefold(),{}).get("table")
        if (key==node.this.name.casefold() and (owner==old_table or not old_table and owner in used)):
            continue
        modified=tree.copy()
        target=list(modified.walk())[index]
        alias=next((s["alias"] for s in sources.values() if s["table"]==owner),metadata[owner]["name"])
        target.set("this",_sql_col(metadata[owner]["columns"][key],alias))
        if owner in used:
            yield modified.sql(dialect="sqlite")
        yield from _reconstruct(modified,projection,metadata,question)
