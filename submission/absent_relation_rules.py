"""Propose missing-relationship queries from parent and child schema evidence."""
import re
from sqlglot import exp

from relation_rules import _schema
from single_table_rules import _outputs, _words


NEGATIVE = re.compile(r"\b(?:without|never|(?:do(?:es)?|did|is|are|was|were|has|have)\s+not|"
                      r"(?:do|does|did|is|are|was|were|has|have)n['’]t|(?:has|have|had)\s+no)\b", re.I)
STOP = _words("any a an in to the that who which have has are is does do did record been had "
              "use used using belong belonging involved participate contain include participating all with of")


def relationship_cue(question):
    if not isinstance(question, str) or len(question) > 1600:
        return None
    cue = NEGATIVE.search(question)
    if not cue or re.search(r"\d|[\"“”]|\b(?:more|less|before|after|than|except|both|either)\b", question, re.I):
        return None
    left = _words(question[:cue.start()])
    tail = question[cue.end():]
    if not re.search(r"\b(?:any|have|has|own|belong|include|involved|participate|used|enrolled|friend)\b", tail, re.I) and cue.group().lower() not in ("without","has no","have no","never"):
        return None
    return left, _words(tail) - STOP


def _terms(text):
    terms = set()
    for word in _words(text):
        # Align grammatical variants of a role in FK names and question verbs.
        word = word.replace("iz", "is")
        if len(word) > 5:
            word = re.sub(r"(?:ing|ed|er|or|e)$", "", word)
        terms.add(word)
    return terms


def _qcol(table, name):
    return exp.column(name, table=table, quoted=True)


def _source(name):
    return exp.Table(this=exp.to_identifier(name, quoted=True))


def absent_relation_repairs(history, example, question, schema):
    cue = relationship_cue(question)
    outputs = _outputs(example.get("columns"))
    metadata = _schema(schema)
    if not cue or not outputs or not metadata or len(metadata) > 150:
        return
    left, right = cue
    if not right:
        return
    role_words = {w for word in right for w in _terms(word)}
    is_count = len(outputs) == 1 and isinstance(outputs[0], exp.Count) and isinstance(outputs[0].this, exp.Star)
    if not is_count and not all(isinstance(item, exp.Column) for item in outputs):
        return
    columns = [item.name.casefold() for item in outputs if isinstance(item, exp.Column)]
    record = {t['name'].casefold():t for t in schema}
    pairs = []
    for pk, parent in metadata.items():
        match = _words(parent['name']) & left
        if not match or any(c not in parent['columns'] for c in columns):
            continue
        for ck, child in metadata.items():
            if ck == pk:
                continue
            childterms = _words(child['name'])
            target = childterms & right
            if not target:
                continue
            role = role_words - _terms(child['name']) - _terms(parent['name']) - {'id','code','key'}
            refs = [(f['to'], f['from'], 3) for f in child['foreign_keys']
                    if isinstance(f,dict) and str(f.get('table','')).casefold() == pk
                    and str(f.get('to','')).casefold() in parent['columns']
                    and str(f.get('from','')).casefold() in child['columns']]
            if not refs and len(childterms & right) == len(childterms):
                # When there is no declared FK, only a strongly grounded role
                # with the same schema type as a single-column PK may join it.
                primary = [c for c in record[pk]['columns'] if c.get('pk')]
                if len(primary) == 1:
                    key = primary[0]
                    refs = [(key['name'], c['name'], 0) for c in record[ck]['columns']
                            if not c.get('pk') and c.get('type') == key.get('type')
                            and _terms(c['name']) & role]
            for pcol, ccol, declared in refs:
                rank = (declared + 3 * len(_terms(ccol) & role) + len(target)/len(childterms)
                        + len(match)/len(_words(parent['name'])))
                pairs.append((rank,parent,child,pcol,ccol))
    if not pairs:
        return
    best = max(entry[0] for entry in pairs)
    winners = [entry for entry in pairs if entry[0] == best]
    if len(winners) != 1:
        return
    _, parent, child, own, foreign = winners[0]
    pn, cn = parent['name'], child['name']
    result = [exp.Count(this=exp.Star())] if is_count else [
        _qcol(pn,parent['columns'][name]) for name in columns]
    base = exp.select(*result).from_(_source(pn))
    childkeys = exp.select(_qcol(cn, foreign)).from_(_source(cn))
    negative = base.copy().where(exp.Not(this=_qcol(pn, own).isin(query=childkeys)))
    offers = [negative]
    if not is_count:
        joined = base.copy().join(_source(cn), on=_qcol(pn,own).eq(_qcol(cn,foreign)))
        difference = base.except_(joined)
        if columns == [own.casefold()]:
            difference = base.except_(childkeys)
            offers.insert(0, difference)
        else:
            offers.append(difference)
    checked = {e.get('sql') for e in history if e.get('tool') == 'check'}
    for query in offers:
        sql = query.sql(dialect='sqlite')
        if sql not in checked and len(sql.encode('utf-8')) <= 16000:
            yield sql
