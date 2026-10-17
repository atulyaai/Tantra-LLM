"""Prepare a provenance-preserving expert review queue, never automatic approval."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
import re
from pathlib import Path
from .expert_workbench import catalog, messages_of, validate_record
from .public_sources import sources, location, sha256

# Suggestions only. Multi-intent or ambiguous records remain unassigned.
PATTERNS = {
 'greeting':r'^(?:hi|hello|hey|good morning|good evening|good afternoon)[!.\s]*$',
 'farewell':r'^(?:bye|goodbye|see you|good night)[!.\s]*$',
 'identity_self':r'^(?:who are you|what is your name)[?!.\s]*$',
 'identity_capabilities':r'^(?:what can you do|how can you help me)[?!.\s]*$',
 'empathy':r'\b(?:i feel sad|i am lonely|i feel lonely|feeling overwhelmed|i am grieving)\b',
 'clarification':r'^(?:help me|can you help|i need help)[?!.\s]*$',
 'summarization':r'\b(?:summari[sz]e|write a summary|give a summary)\b',
 'translation':r'\btranslate\b',
 'coding':r'\b(?:write|implement) (?:a |an )?(?:python|javascript|function|program)\b',
 'debugging':r'\b(?:debug|traceback|fix this code)\b',
 'math':r'\b(?:calculate|solve the equation|compute the value)\b',
 'logic':r'\b(?:logical puzzle|logic puzzle|syllogism)\b',
 'creative_writing':r'\b(?:write a poem|write a story|write a short story)\b',
 'api_schema':r'\b(?:function call|api schema|tool call)\b',
 'json_schema':r'\b(?:json schema|output json|return json)\b',
 'gk_geography':r'\b(?:capital of|highest mountain|largest continent)\b',
}


def suggest(text):
    hits=[name for name,pattern in PATTERNS.items() if re.search(pattern,text,re.I)]
    return hits[0] if len(hits)==1 else None


def accepted_node(node):
    if node.get('deleted') or node.get('synthetic') or node.get('review_result') is not True:
        return False
    labels=node.get('labels') or {}
    if any((labels.get(k) or {}).get('value',0)>0 for k in ('spam','pii','lang_mismatch','not_appropriate')):
        return False
    return isinstance(node.get('text'),str) and bool(node['text'].strip())


def paths(node, history=(), ancestors=frozenset()):
    identity=node.get('message_id')
    if identity in ancestors or len(history)>=32 or not accepted_node(node): return
    role={'prompter':'user','assistant':'assistant'}.get(node.get('role'))
    if role is None or (not history and role!='user') or (history and history[-1]['role']==role): return
    messages=history+({'role':role,'content':node['text'].strip()},)
    if role=='assistant' and node.get('rank') in (None,0): yield node,messages
    for child in node.get('replies',[]):
        yield from paths(child,messages,ancestors|{identity})


def prepare(output):
    output=Path(output)
    if output.exists(): raise ValueError('Use a new immutable review queue directory')
    source=next(s for s in sources() if s['id']=='oasst1')
    raw=location(source)
    if sha256(raw)!=source['sha256']: raise ValueError('Source checksum mismatch')
    output.mkdir(parents=True)
    counts=Counter(); languages=Counter(); seen=set(); writers={}
    try:
        with gzip.open(raw,'rt',encoding='utf-8') as reader:
            for line in reader:
                tree=json.loads(line); root=tree.get('prompt') or {}
                family='oasst1:'+tree['message_tree_id']
                for node,messages in paths(root):
                    lang=node.get('lang'); languages[lang]+=1
                    if lang not in ('hi','en'): counts['excluded_language']+=1; continue
                    if any(len(m['content'])>16000 for m in messages): counts['too_long']+=1; continue
                    h=hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
                    if h in seen: counts['duplicate']+=1; continue
                    seen.add(h)
                    expert=suggest(next(m['content'] for m in reversed(messages) if m['role']=='user'))
                    name=expert or 'unassigned'
                    if name not in writers: writers[name]=(output/f'{name}.jsonl').open('w',encoding='utf-8')
                    row={'schema_version':1,'record_id':h,'expert':expert,'language':lang,'family_id':family,
                         'template_id':'oasst1-human-conversation','messages':list(messages),
                         'source':{'id':'OpenAssistant/oasst1','revision':source['revision'],'license':source['license'],
                                   'kind':'community_conversation','url':'https://huggingface.co/datasets/OpenAssistant/oasst1',
                                   'record_id':node['message_id']},
                         'source_review':{'review_result':node.get('review_result'),'review_count':node.get('review_count')},
                         'review':{'status':'pending','reviewer':None},
                         'note':'Community annotation is not Tantra expert, correctness or fluent-language review.'}
                    writers[name].write(json.dumps(row,ensure_ascii=False)+'\n'); counts[name]+=1
    finally:
        for writer in writers.values(): writer.close()
    index={'source':source,'source_languages_seen':dict(languages),'counts':dict(counts),
           'experts':{name:{'suggested_rows':counts[name],'state':'needs_review' if counts[name] else 'needs_source',
                           'goal':spec['goal'],'minimum':spec['minimum'],
                           'queue':f'{name}.jsonl' if counts[name] else None} for name,spec in catalog().items()},
           'files':{p.name:sha256(p) for p in output.glob('*.jsonl')},
           'warning':'No record approved or expert marked data_ready; no training is launched by this command.'}
    (output/'index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
    return index


def approve(queue, decisions, output):
    """Import externally supplied row-specific review decisions into canonical incoming."""
    queue,decisions,output=Path(queue),Path(decisions),Path(output)
    if output.exists(): raise ValueError('Reviewed export must be a new directory')
    index=json.loads((queue/'index.json').read_text(encoding='utf-8'))
    for name,expected in index['files'].items():
        if sha256(queue/name)!=expected: raise ValueError('Review queue checksum mismatch')
    reviews={}
    for line in decisions.read_text(encoding='utf-8').splitlines():
        decision=json.loads(line)
        if decision['record_id'] in reviews: raise ValueError('Duplicate review decision')
        reviews[decision['record_id']]=decision
    selected=[]; found=set()
    for path in [queue/name for name in sorted(index['files'])]:
        for line in path.open(encoding='utf-8'):
            row=json.loads(line); decision=reviews.get(row['record_id'])
            if not decision: continue
            found.add(row['record_id'])
            if decision.get('status')!='approved': continue
            if not decision.get('reviewer') or not decision.get('rationale'):
                raise ValueError('Approval requires real reviewer attribution and rationale')
            expert=decision.get('expert')
            if expert not in catalog(): raise ValueError('Unknown reviewed expert')
            if row['language'] in ('hi','hinglish') and not decision.get('fluent_language_review'):
                raise ValueError('Hindi/Hinglish require fluent review')
            domain=catalog()[expert]['domain']
            if domain=='safety' or expert=='safety_refusal':
                if not decision.get('domain_review'): raise ValueError('Safety requires domain review')
            row={**row,'expert':expert,'review':{'status':'reviewed','reviewer':decision['reviewer'],
                  'rationale':decision['rationale'],'fluent_language_review':bool(decision.get('fluent_language_review')),
                  'domain_review':bool(decision.get('domain_review')),'decision_sha256':sha256(decisions)}}
            selected.append((domain,expert,validate_record(row,expert)))
    if set(reviews)-found: raise ValueError('Review contains unknown record IDs')
    output.mkdir(parents=True)
    for domain,expert,row in selected:
        directory=output/domain; directory.mkdir(exist_ok=True)
        with (directory/f'{expert}.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    return {'approved':len(selected),'rejected_or_skipped':len(reviews)-len(selected)}


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    make=sub.add_parser('prepare'); make.add_argument('--output',type=Path,required=True)
    review=sub.add_parser('approve'); review.add_argument('--queue',type=Path,required=True)
    review.add_argument('--decisions',type=Path,required=True); review.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=prepare(a.output) if a.command=='prepare' else approve(a.queue,a.decisions,a.output)
    print(json.dumps({'counts':result['counts'],'registered':len(result['experts'])} if 'experts' in result else result,indent=2))

if __name__=='__main__': main()
