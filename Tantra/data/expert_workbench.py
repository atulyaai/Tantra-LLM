"""Dataset contracts, inventory and preparation for all taxonomy experts.

python -m Tantra.data.expert_workbench --help
This module never trains or changes Model/. Generated examples are verifiable
bootstrap tasks; registered experts are not automatically trained experts.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import random
import re
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

from Tantra.config import ROOT, get_preset, load_taxonomy, validate_taxonomy

VERSION = 1
DEFAULT_OUT = ROOT / 'Datasets' / 'workbench'
JUNK = re.compile(r'sample\s+\d+\s+of\s+\d+|relates to sample\s+\d+|\[SYNTHETIC|GENERATION NEEDED|this is a sample text in', re.I)
PRIORITY = ['greeting', 'farewell', 'smalltalk', 'clarification', 'empathy', 'identity_self',
            'identity_capabilities', 'identity_limitations', 'safety_refusal', 'lang_hi']
LANGUAGES = dict(zip(
    'en zh es hi ar pt bn ru ja de ko fr tr vi it th pl nl id sv cs el he ro hu fi da no sk hr sl et lv lt mt ga cy eu ca gl is mk sq bs me sr bg uk be ka'.split(),
    ['English','Chinese','Spanish','Hindi','Arabic','Portuguese','Bengali','Russian','Japanese','German',
     'Korean','French','Turkish','Vietnamese','Italian','Thai','Polish','Dutch','Indonesian','Swedish',
     'Czech','Greek','Hebrew','Romanian','Hungarian','Finnish','Danish','Norwegian','Slovak','Croatian',
     'Slovenian','Estonian','Latvian','Lithuanian','Maltese','Irish','Welsh','Basque','Catalan','Galician',
     'Icelandic','Macedonian','Albanian','Bosnian','Montenegrin','Serbian','Bulgarian','Ukrainian','Belarusian','Georgian']))
GOALS = {
 'greeting':'Recognize greetings and return a concise, language-matched welcome.',
 'farewell':'Recognize conversation endings and respond without restarting the conversation.',
 'smalltalk':'Handle thanks and everyday social questions without inventing personal experiences.',
 'empathy':'Acknowledge feelings supportively without diagnosis or pretending to have feelings.',
 'clarification':'Ask the minimum useful question when the request lacks required context.',
 'apology':'Acknowledge a concrete error and correct it without repeated apologies.',
 'humor':'Produce relevant, harmless jokes and distinguish literal from humorous requests.',
 'identity_self':'Identify consistently as Tantra, a language model from Atulya AI.',
 'identity_capabilities':'Describe only capabilities available in the actual deployment.',
 'identity_limitations':'Explain uncertainty, access limits and possible errors honestly.',
 'identity_persona':'Maintain a helpful consistent voice without claiming human identity.',
 'identity_consistency':'Keep identity facts consistent across paraphrases and multi-turn challenges.',
 'identity_metadata':'Report supplied model metadata; do not invent version or training facts.',
 'tone_style':'Follow explicit tone and style constraints while preserving meaning.',
 'safety_refusal':'Refuse harmful requests briefly and offer a relevant safe alternative.',
 'logic':'Solve finite logical problems with checkable premises and conclusions.',
 'math':'Solve arithmetic and algebra with exact, programmatically checkable answers.',
 'coding':'Write small correct programs that satisfy supplied specifications and tests.',
 'reasoning_planning':'Explain a feasible plan with explicit assumptions and dependencies.',
 'debugging':'Identify the actual cause in a supplied failing program and propose a minimal fix.',
 'algorithm_design':'Choose and explain an algorithm with correct complexity and edge cases.',
 'code_review':'Find actionable defects in supplied code, with a concrete failure example.',
 'architecture':'Compare system designs against stated constraints without inventing requirements.',
 'optimization':'Optimize measured bottlenecks while preserving tested behavior.',
 'verification':'Check a proposed answer or proof against its stated requirements.',
 'symbolic_reasoning':'Apply explicit symbolic rules without changing their meaning.',
 'causal_inference':'Distinguish correlation from causation and identify necessary assumptions.',
 'pii':'Identify sensitive personal data and transform it according to a stated privacy policy.',
 'toxicity':'Recognize abusive content and provide a non-abusive response or rewrite.',
 'jailbreak':'Keep instruction boundaries when untrusted text asks to override them.',
 'bias':'Identify unsupported stereotypes and respond fairly using relevant evidence.',
 'medical_safety':'Handle medical questions cautiously, with uncertainty and appropriate escalation.',
 'financial_safety':'Explain financial concepts without guaranteed returns or fabricated certainty.',
 'legal_safety':'Distinguish general legal information from jurisdiction-specific professional advice.',
 'child_safety':'Respond appropriately to child-related risk without sexual or exploitative material.',
 'self_harm':'Respond compassionately to distress; never generate harmful procedural instructions.',
 'misinformation':'Identify unsupported claims and separate evidence from speculation.',
 'summarization':'Summarize supplied text accurately with requested length and coverage.',
 'translation':'Translate between explicitly specified languages without adding facts.',
 'qa':'Answer only what the supplied evidence supports and abstain when evidence is missing.',
 'creative_writing':'Produce original writing satisfying explicit genre, tone and length constraints.',
 'analysis':'Analyze supplied material using supported observations and stated assumptions.',
 'extraction':'Extract requested fields exactly from supplied records or text.',
 'formatting':'Change representation without changing the underlying data.',
 'rewriting':'Rewrite supplied text while preserving specified facts and constraints.',
 'classification':'Assign only labels from a supplied ontology and explain ambiguous cases.',
 'comparison':'Compare options on explicit criteria without inventing attributes.',
 'synthesis':'Combine supplied sources while preserving disagreements and attribution.',
 'evaluation':'Assess supplied work against an explicit rubric.',
 'planning':'Produce executable steps with prerequisites, success checks and stop conditions.',
 'task_decomposition':'Break a request into non-overlapping tasks with dependencies.',
 'tool_selection':'Choose an available tool from its actual schema; abstain if none fits.',
 'state_tracking':'Update an explicit task state consistently across turns.',
 'error_recovery':'Select a bounded retry or rollback based on a supplied failure.',
 'multi_step_reasoning':'Complete linked steps while retaining intermediate constraints.',
 'api_schema':'Emit calls matching a supplied API schema without inventing fields.',
 'function_composition':'Connect tool outputs to compatible downstream inputs.',
 'parallel_calls':'Parallelize independent calls while preserving dependency order.',
 'streaming_results':'Assemble partial results and distinguish partial from completed output.',
 'auth_handling':'Handle authorization errors without exposing or fabricating credentials.',
 'tool_error_handling':'Interpret structured tool failures and choose a justified response.',
 'json_schema':'Emit syntactically valid JSON satisfying a supplied schema.',
 'xml_format':'Emit well-formed XML with required structure and correct escaping.',
 'markdown_tables':'Render supplied records as correct, escaped Markdown tables.',
 'csv_export':'Serialize supplied records as CSV with correct quoting and column order.',
 'regex_validation':'Match or validate strings using a stated pattern and check edge cases.',
 'query_rewrite':'Rewrite a search query without introducing unsupported entities or claims.',
 'doc_ranking':'Rank supplied documents by their evidence for a given query.',
 'citation_extraction':'Attach claims to exact supplied source spans and identifiers.',
 'hallucination_check':'Flag claims unsupported by supplied evidence and propose grounded corrections.',
 'context_compression':'Compress supplied evidence while retaining facts required by a question.',
 'repo_navigation':'Find relevant files and symbols from a supplied repository snapshot.',
 'test_generation':'Create tests that fail for a concrete supplied defect.',
 'debug_loop':'Use a supplied failure/test result to propose and validate the next fix.',
 'refactor_planning':'Plan behavior-preserving changes with dependency and test coverage.',
 'code_search':'Locate code matching a query and cite the actual supplied paths.',
 'dependency_analysis':'Explain actual dependencies and cycles from supplied manifests or code.',
 'translate_pivot':'Translate through a pivot language while checking meaning preservation.',
 'translate_quality':'Evaluate adequacy and fluency against a supplied source and rubric.',
 'translate_terminology':'Preserve a supplied terminology glossary consistently.',
 'translate_formality':'Control register and honorifics without changing meaning.',
 'translate_domain':'Translate domain text with appropriate terminology and uncertainty.',
 'translate_literary':'Translate literary meaning, voice and imagery with documented tradeoffs.',
 'translate_technical':'Translate technical instructions while preserving identifiers and numbers.',
 'translate_low_resource':'Handle supported low-resource language pairs with native-speaker review.',
 'retrieval':'Retrieve exact evidence from long supplied documents.',
 'hierarchy':'Organize long material into a faithful hierarchical outline.',
 'compression':'Compress long material under a budget while preserving required information.',
 'rnn_memory':'Maintain explicit sequential state over a supplied long sequence.'
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def normalized(text):
    return unicodedata.normalize('NFC', text)


def split_for(family):
    n = int(digest(family)[:12], 16) % 100
    return 'train' if n < 90 else 'val' if n < 95 else 'test'


def catalog(taxonomy=None, config=None):
    taxonomy = taxonomy or load_taxonomy()
    errors = validate_taxonomy(taxonomy)
    if errors:
        raise ValueError('; '.join(errors))
    cfg = config or get_preset('small')
    cfg.vocab_size = 10277
    d, kv, h = cfg.d_model, cfg.d_model // cfg.n_heads * cfg.n_kv_heads, cfg.mlp_hidden
    shapes = {'q': (d,d), 'k': (d,kv), 'v': (d,kv), 'o': (d,d),
              'gate': (d,h), 'up': (d,h), 'down': (h,d)}
    result = {}
    for domain, info in taxonomy.items():
        for name in info['experts']:
            lang = name[5:] if name.startswith('lang_') else None
            goal = (f'Understand and respond naturally in {LANGUAGES[lang]}, with native-speaker review; English text with a language tag is not valid evidence.'
                    if lang else GOALS.get(name))
            if name.startswith('gk_'):
                topic = name[3:].replace('_',' ')
                if name=='gk_math': topic='mathematical definitions and facts (arithmetic computation belongs to math)'
                goal = f'Answer {topic} questions with checkable references, explicit uncertainty and dated sources for changing facts.'
            if not goal:
                raise ValueError(f'Missing expert specification: {name}')
            rank = int(info['rank'])
            result[name] = {'expert': name, 'domain': domain, 'goal': goal,
                            'languages': [lang] if lang else ['hi','hinglish','en'],
                            'target_rank': rank,
                            'adapter_parameters_small': cfg.n_layers * rank * sum(sum(shapes[t]) for t in cfg.lora_targets),
                            'minimum': {'train':1000,'val':100,'test':100},
                            'priority': PRIORITY.index(name) if name in PRIORITY else 100,
                            'deployment':'disabled_until_independent_held_out_gain', 'pilot_rank':8}
    return result


def initialize(out=DEFAULT_OUT, docs=ROOT/'docs'/'experts'):
    out, docs = Path(out), Path(docs)
    specs = catalog()
    for name, item in specs.items():
        for split in ('incoming',):
            (out / split / item['domain']).mkdir(parents=True, exist_ok=True)
        card = docs / item['domain'] / f'{name}.md'
        card.parent.mkdir(parents=True, exist_ok=True)
        card.write_text(f"# {name}\n\n{item['goal']}\n\n"
                        f"- Domain: {item['domain']}; languages: {', '.join(item['languages'])}.\n"
                        f"- Incoming: `Datasets/workbench/incoming/{item['domain']}/{name}.jsonl`.\n"
                        f"- Taxonomy target rank: {item['target_rank']}; small-model adapter parameters: {item['adapter_parameters_small']:,}.\n"
                        "- Initial pilot rank: 8; increase only after held-out improvement and memory profiling.\n"
                        "- Minimum independent examples: 1,000 train / 100 validation / 100 test.\n"
                        "- Follow the [shared dataset contract](../../DATASET_SPEC.md), including provenance, family grouping, review and quality gates.\n"
                        "- A card is a capability specification, not a trained or validated expert.\n", encoding='utf-8')
    write_json(docs.parent/'expert_catalog.json', {'version':VERSION,'experts':specs})
    write_json(docs.parent/'dataset_record.schema.json',dataset_schema())
    return specs


def messages_of(row):
    if not isinstance(row,dict):
        raise ValueError('row_not_object')
    messages = row.get('messages')
    if messages is None and isinstance(row.get('text'),str):
        from Tantra.data.expert_dataset import _legacy_to_messages
        messages = _legacy_to_messages(row['text'].replace('\\n','\n'))
    if not isinstance(messages,list) or not messages:
        raise ValueError('missing_chat_messages')
    if any(not isinstance(m,dict) or m.get('role') not in ('system','user','assistant')
           or not isinstance(m.get('content'),str) or not m['content'].strip() for m in messages):
        raise ValueError('invalid_messages')
    if not any(m['role']=='user' for m in messages) or messages[-1]['role']!='assistant':
        raise ValueError('missing_user_or_final_assistant')
    if any(JUNK.search(m['content']) for m in messages):
        raise ValueError('placeholder')
    return [{'role':m['role'],'content':m['content'].strip()} for m in messages]


def verify_auto(row):
    v = row.get('verification') or {}
    answer = row['messages'][-1]['content'].strip()
    kind = v.get('kind')
    if kind=='arithmetic':
        a,b,op = v['a'],v['b'],v['op']
        if not isinstance(a,int) or not isinstance(b,int): return False
        expected = {'+':lambda:a+b,'-':lambda:a-b,'*':lambda:a*b}.get(op)
        return (expected is not None and answer==str(expected()) and re.search(r'(?<!\d)'+re.escape(str(a))+r'\s*'+re.escape(op)+r'\s*'+re.escape(str(b))+r'(?!\d)',row['messages'][0]['content']) is not None)
    if kind=='json':
        return json.loads(answer)==v['expected'] and json.loads(row['messages'][0]['content'].split('\n',1)[1])==v['expected']
    if kind=='csv':
        expected = [[str(x) for x in r] for r in v['expected']]
        return list(csv.reader(io.StringIO(answer)))==expected and json.loads(row['messages'][0]['content'].split('\n',1)[1])==v['expected']
    return False


def validate_record(row, expert, strict=True):
    if JUNK.search(json.dumps(row,ensure_ascii=False)):
        raise ValueError('placeholder')
    messages = messages_of(row)
    clean = {**row,'messages':messages,'expert':expert}
    if not strict:
        return clean
    if row.get('schema_version')!=VERSION or row.get('expert')!=expert:
        raise ValueError('schema_or_expert_mismatch')
    if not isinstance(row.get('language'),str) or not row['language']:
        raise ValueError('missing_language')
    if not isinstance(row.get('template_id'),str) or not row['template_id']:
        raise ValueError('missing_template')
    if row['language'] not in set(LANGUAGES)|{'hinglish'}:
        raise ValueError('unsupported_language')
    if expert.startswith('lang_') and row['language']!=expert[5:]:
        raise ValueError('expert_language_mismatch')
    if not isinstance(row.get('family_id'),str) or not row['family_id']:
        raise ValueError('missing_family')
    source,review = row.get('source'),row.get('review')
    if not isinstance(source,dict) or any(not source.get(k) for k in ('id','revision','license','kind')):
        raise ValueError('missing_provenance')
    if source['license'].lower() in ('unknown','todo','unverified'):
        raise ValueError('unknown_license')
    if not isinstance(review,dict) or not review.get('reviewer'):
        raise ValueError('missing_review')
    if review.get('status')=='verified_auto':
        owner={'arithmetic':'math','json':'json_schema','csv':'csv_export'}.get((row.get('verification') or {}).get('kind'))
        if owner!=expert:
            raise ValueError('wrong_auto_task_owner')
        if source['id']!='tantra-verifiable-v1' or not verify_auto(clean):
            raise ValueError('verification_failed')
    elif review.get('status')!='reviewed':
        raise ValueError('review_required')
    return clean


def source_files(inputs):
    return sorted({p.resolve() for base in inputs for p in Path(base).rglob('*.jsonl')})


def audit(inputs=None, out=DEFAULT_OUT, model_dir=ROOT/'Model'):
    specs = catalog()
    inputs = inputs or [ROOT/'Datasets'/'curated', ROOT/'Datasets'/'experts',Path(out)/'incoming']
    stats = {n:{'files':[],'raw_rows':0,'usable_chat_rows':0,'reviewed_rows':0,'rejected':{},
                'weights_present':(Path(model_dir)/'experts'/f'{n}.pt').is_file()} for n in specs}
    extras = []
    for path in source_files(inputs):
        name = file_expert(path,specs)
        if name not in specs:
            extras.append(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
            continue
        record = stats[name]
        record['files'].append(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
        rejects = Counter(record['rejected'])
        with path.open(encoding='utf-8-sig') as f:
            for line in f:
                if not line.strip(): continue
                record['raw_rows'] += 1
                # Skip known filler before JSON parsing; preserve exact counts.
                if JUNK.search(line):
                    rejects['placeholder'] += 1
                    continue
                try:
                    row = json.loads(line)
                    validate_record(row,name,False)
                    record['usable_chat_rows'] += 1
                    try:
                        validate_record(row,name,True)
                        record['reviewed_rows'] += 1
                    except (ValueError,TypeError,KeyError,json.JSONDecodeError) as exc:
                        rejects['contract:'+str(exc)] += 1
                except (ValueError,TypeError,KeyError,json.JSONDecodeError) as exc:
                    rejects[str(exc)] += 1
        record['rejected'] = dict(rejects)
    extra_weights = [p.stem for p in (Path(model_dir)/'experts').glob('*.pt') if p.stem not in specs]
    result = {'version':VERSION,'experts':stats,'supplementary_data_files':extras,
              'supplementary_weights':extra_weights,
              'summary':{'registered':len(specs),'taxonomy_weights_present':sum(s['weights_present'] for s in stats.values()),
                         'raw_rows':sum(s['raw_rows'] for s in stats.values()),
                         'usable_chat_rows':sum(s['usable_chat_rows'] for s in stats.values()),
                         'reviewed_rows':sum(s['reviewed_rows'] for s in stats.values()),
                         'placeholder_rows':sum(s['rejected'].get('placeholder',0) for s in stats.values())}}
    write_json(Path(out)/'audit.json',result)
    return result


def generated_rows(count=5000):
    for index in range(count):
        a,b = index*37%1999-999,index*53%997-498
        op = ['+','-','*'][index%3]
        expected = {'+':a+b,'-':a-b,'*':a*b}[op]
        lang = ['en','hi','hinglish'][(index//3)%3]
        question = {'en':f'Calculate {a} {op} {b}. Return only the integer.',
                    'hi':f'{a} {op} {b} की गणना करें। केवल पूर्णांक लिखें।',
                    'hinglish':f'{a} {op} {b} ka jawab do. Sirf integer likho.'}[lang]
        common = {'schema_version':VERSION,'language':lang,'family_id':f'arithmetic:{op}:{a}:{b}',
                  'template_id':f'arithmetic:{op}','source':{'id':'tantra-verifiable-v1','revision':'1','license':'MIT','kind':'synthetic_verified'},
                  'review':{'status':'verified_auto','reviewer':'exact arithmetic/serialization verifier'}}
        for name in ('math',):
            yield {**common,'expert':name,'messages':[{'role':'user','content':question},{'role':'assistant','content':str(expected)}],
                   'verification':{'kind':'arithmetic','a':a,'b':b,'op':op}}
        obj = {'id':index,'active':bool(index%2),'values':list(range(index%5+1))}
        payload = json.dumps(obj,ensure_ascii=False)
        yield {**common,'expert':'json_schema','language':'en','family_id':f'json-record:{index}','template_id':'json-serialize',
               'messages':[{'role':'user','content':'Return the following object as valid JSON, with no Markdown.\n'+payload},
                           {'role':'assistant','content':payload}], 'verification':{'kind':'json','expected':obj}}
        table = [['id','name','value'],[index,f'example,{index}',a],[index+1,f'quote"{index}',b]]
        stream = io.StringIO(newline='')
        csv.writer(stream,lineterminator='\n').writerows(table)
        yield {**common,'expert':'csv_export','language':'en','family_id':f'csv-record:{index}','template_id':'csv-serialize',
               'messages':[{'role':'user','content':'Convert these records to CSV, preserving row and column order.\n'+json.dumps(table)},
                           {'role':'assistant','content':stream.getvalue().rstrip('\n')}],
               'verification':{'kind':'csv','expected':table}}


def seed(out=DEFAULT_OUT, count=5000):
    if count < 1: raise ValueError('count must be positive')
    specs = catalog()
    handles, counts = {},Counter()
    try:
        for row in generated_rows(count):
            name = row['expert']
            validate_record(row,name)
            if name not in handles:
                path = Path(out)/'incoming'/specs[name]['domain']/f'verified-{name}.jsonl'
                path.parent.mkdir(parents=True,exist_ok=True)
                handles[name] = path.open('w',encoding='utf-8')
            handles[name].write(json.dumps(row,ensure_ascii=False)+'\n')
            counts[name] += 1
    finally:
        for f in handles.values(): f.close()
    return dict(counts)


def file_expert(path, specs):
    name = path.stem.removeprefix('verified-')
    return name if name in specs else None


def prepare(out=DEFAULT_OUT):
    out = Path(out)
    specs = catalog()
    manifest_path = out/'prepared'/'manifest.json'
    if manifest_path.exists():
        raise ValueError('Prepared bundle is immutable; use --out for a new dataset version')
    out.mkdir(parents=True,exist_ok=True)
    rejects,duplicates = Counter(),0
    parent = {}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        parent[max(a,b)]=min(a,b)
    with tempfile.TemporaryDirectory(prefix='tantra-dataset-') as temporary:
        db=sqlite3.connect(Path(temporary)/'dedup.sqlite')
        db.execute('CREATE TABLE records(expert TEXT, hash TEXT, family TEXT, payload TEXT, UNIQUE(expert,hash))')
        db.execute('CREATE TABLE documents(hash TEXT PRIMARY KEY, family TEXT)')
        db.execute('CREATE INDEX expert_index ON records(expert)')
        quarantine = out/'quarantine.jsonl'
        with quarantine.open('w',encoding='utf-8') as bad:
            for path in source_files([out/'incoming']):
                name=file_expert(path,specs)
                if not name:
                    rejects['unknown_expert_file']+=1
                    continue
                with path.open(encoding='utf-8-sig') as f:
                    for number,line in enumerate(f,1):
                        if not line.strip(): continue
                        try:
                            row=validate_record(json.loads(line),name)
                            key=digest(normalized(json.dumps(row['messages'],ensure_ascii=False,sort_keys=True)))
                            family=row['source']['id']+':'+row['family_id']
                            prior=db.execute('SELECT family FROM documents WHERE hash=?',(key,)).fetchone()
                            if prior: union(family,prior[0])
                            else: db.execute('INSERT INTO documents VALUES (?,?)',(key,family))
                            find(family)
                            inserted=db.execute('INSERT OR IGNORE INTO records VALUES (?,?,?,?)',
                                                (name,key,family,json.dumps(row,ensure_ascii=False))).rowcount
                            duplicates+=not inserted
                        except (ValueError,TypeError,KeyError,json.JSONDecodeError) as exc:
                            rejects[str(exc)]+=1
                            bad.write(json.dumps({'file':str(path),'line':number,'reason':str(exc),'sample':line[:500]},ensure_ascii=False)+'\n')
        db.commit()
        result={}
        for name,item in specs.items():
            counts=Counter(); families={s:set() for s in ('train','val','test')}; languages=Counter(); templates=set(); files={}
            writers={}
            try:
                for split in families:
                    path=out/'prepared'/split/item['domain']/f'{name}.jsonl'
                    if db.execute('SELECT 1 FROM records WHERE expert=? LIMIT 1',(name,)).fetchone():
                        path.parent.mkdir(parents=True,exist_ok=True)
                        writers[split]=path.open('w',encoding='utf-8'); files[split]=str(path.relative_to(out/'prepared'))
                for h,family,payload in db.execute('SELECT hash,family,payload FROM records WHERE expert=? ORDER BY hash',(name,)):
                    row=json.loads(payload); group=find(family); split=split_for(group)
                    row['record_id']=h; row['split_group']=group; row['split']=split
                    writers[split].write(json.dumps(row,ensure_ascii=False)+'\n')
                    counts[split]+=1; families[split].add(group); languages[row['language']]+=1
                    templates.add(row.get('template_id','not_supplied'))
            finally:
                for f in writers.values(): f.close()
            ready=all(counts[s]>=item['minimum'][s] and len(families[s])>=item['minimum'][s] for s in families)
            result[name]={'state':'data_ready' if ready else 'needs_data','rows':dict(counts),
                          'families':{s:len(v) for s,v in families.items()},'languages':dict(languages),
                          'template_count':len(templates),'files':{s:{'path':p,'sha256':hashlib.sha256((out/'prepared'/p).read_bytes()).hexdigest()} for s,p in files.items()},
                          'minimum':item['minimum'], 'trained':False, 'quality_validated':False}
        db.close()
    manifest={'version':VERSION,'seed':42,'split_policy':'sha256(source-family) modulo 100: 90/5/5; duplicate-connected families unioned globally',
              'experts':result,'rejected':dict(rejects),'duplicates_removed':duplicates,
              'summary':{'registered':len(specs),'data_ready':sum(v['state']=='data_ready' for v in result.values()),
                         'accepted_rows':sum(sum(v['rows'].values()) for v in result.values())}}
    write_json(manifest_path,manifest)
    return manifest


def training_plan(out=DEFAULT_OUT):
    out=Path(out)
    manifest=json.loads((out/'prepared'/'manifest.json').read_text(encoding='utf-8'))
    specs=catalog(); tasks=[]
    for name,item in sorted(specs.items(),key=lambda pair:(pair[1]['priority'],pair[0])):
        state=manifest['experts'][name]
        tasks.append({'expert':name,'domain':item['domain'],'state':state['state'],'goal':item['goal'],
                      'blocked_on':[] if state['state']=='data_ready' else ['reviewed independent train/val/test examples'],
                      'train_file':str(out/'prepared'/state['files']['train']['path']) if 'train' in state['files'] else None,
                      'validation_file':str(out/'prepared'/state['files']['val']['path']) if 'val' in state['files'] else None,
                      'test_file':str(out/'prepared'/state['files']['test']['path']) if 'test' in state['files'] else None,
                      'adapter_parameters_at_target_rank':item['adapter_parameters_small'],
                      'instruction':'Run the workbench train command for a data_ready expert. Keep frozen splits; validate answers and router separately. Export to a new directory.'})
    write_json(out/'training_plan.json',{'tasks':tasks,'warning':'A plan is not a trained model. The workbench train command preserves frozen splits; the legacy trainer re-splits and must not consume the bundle directly.'})
    return tasks



def dataset_schema():
    text={'type':'string','minLength':1}
    return {'$schema':'https://json-schema.org/draft/2020-12/schema','title':'Tantra expert record v1','type':'object',
            'required':['schema_version','expert','language','family_id','template_id','messages','source','review'],
            'properties':{'schema_version':{'const':VERSION},'expert':{'enum':list(catalog())},
             'language':{'enum':list(LANGUAGES)+['hinglish']},'family_id':text,'template_id':text,
             'messages':{'type':'array','minItems':2,'items':{'type':'object','required':['role','content'],
               'properties':{'role':{'enum':['system','user','assistant']},'content':text}}},
             'source':{'type':'object','required':['id','revision','license','kind'],
               'properties':{k:text for k in ['id','revision','license','kind','url']}},
             'review':{'type':'object','required':['status','reviewer'],
               'properties':{'status':{'enum':['reviewed','verified_auto']},'reviewer':text}},
             'verification':{'type':'object'}},'additionalProperties':True}


def report(out=DEFAULT_OUT, docs=ROOT/'docs'):
    out,docs=Path(out),Path(docs)
    a=json.loads((out/'audit.json').read_text(encoding='utf-8'))
    m=json.loads((out/'prepared'/'manifest.json').read_text(encoding='utf-8'))
    specs=catalog()
    lines=['# Expert coverage snapshot','',
           'Regenerate with `python -m Tantra.data.expert_workbench report`. This is a local-data snapshot, not live telemetry.','',
           '## Summary','',
           f"- Registered capabilities: {len(specs)}; taxonomy weight files: {a['summary']['taxonomy_weights_present']}; supplementary: {', '.join(a['supplementary_weights']) or 'none'}.",
           f"- Audited rows: {a['summary']['raw_rows']:,}; rejected filler: {a['summary']['placeholder_rows']:,}.",
           f"- Structurally usable chat: {a['summary']['usable_chat_rows']:,}; reviewed/verifiable: {a['summary']['reviewed_rows']:,}.",
           f"- Prepared examples: {m['summary']['accepted_rows']:,}; pilot-ready experts: {m['summary']['data_ready']}.",
           '- Legacy examples without provenance/review remain excluded. Saved weights are not proof of answer quality.',
           '- Generated task templates are narrow bootstrap coverage, not production-ready capabilities. Keep unsupported or redundant experts disabled.','',
           '## Prepared bootstrap','',
           '| Expert | Train | Validation | Test | Templates | Languages |',
           '| --- | ---: | ---: | ---: | ---: | --- |']
    for name,item in m['experts'].items():
        if item['state']=='data_ready':
            r=item['rows']; lines.append(f"| {name} | {r.get('train',0)} | {r.get('val',0)} | {r.get('test',0)} | {item['template_count']} | {', '.join(item['languages'])} |")
    lines += ['', '## Expert inventory','',
              '| Expert | Domain | Raw rows | Usable chat | Reviewed | Prepared state | Weights |',
              '| --- | --- | ---: | ---: | ---: | --- | --- |']
    for name,spec in specs.items():
        item=a['experts'][name]; state=m['experts'][name]['state']
        lines.append(f"| [{name}](experts/{spec['domain']}/{name}.md) | {spec['domain']} | {item['raw_rows']} | {item['usable_chat_rows']} | {item['reviewed_rows']} | {state} | {'present, unvalidated' if item['weights_present'] else 'absent'} |")
    lines += ['', '## Continue','',
              'Prioritize reviewed Hindi/Hinglish/English conversation and identity examples, expand task diversity, then train isolated candidates. Read HANDOFF.md and DATASET_SPEC.md before adding data. Use new immutable bundle directories for new versions. Evaluate answer quality, router behavior and regressions before deployment.','']
    docs.mkdir(parents=True,exist_ok=True)
    (docs/'EXPERT_STATUS.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(docs/'dataset_record.schema.json',dataset_schema())
    return {'registered':len(specs),'pilot_ready':m['summary']['data_ready'],'report':str(docs/'EXPERT_STATUS.md')}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('init'); p=sub.add_parser('audit'); p.add_argument('--inputs',nargs='+',type=Path)
    p=sub.add_parser('seed'); p.add_argument('--count',type=int,default=5000)
    sub.add_parser('prepare'); sub.add_parser('plan'); sub.add_parser('status'); sub.add_parser('report')
    p=sub.add_parser('train'); p.add_argument('--expert',required=True); p.add_argument('--model-dir',type=Path,default=ROOT/'Model'); p.add_argument('--output',type=Path,required=True); p.add_argument('--steps',type=int,default=100); p.add_argument('--rank',type=int,default=8); p.add_argument('--threads',type=int,default=4); p.add_argument('--release-check',action='store_true'); p.add_argument('--small-pilot',action='store_true',help='Explicitly allow an undersized reviewed bundle; never counts as release-ready')
    args=parser.parse_args(argv)
    if args.command=='init': result={'registered':len(initialize(args.out))}
    elif args.command=='audit': result=audit(args.inputs,args.out)['summary']
    elif args.command=='seed': result=seed(args.out,args.count)
    elif args.command=='prepare': result=prepare(args.out)['summary']
    elif args.command=='plan': result={'tasks':len(training_plan(args.out))}
    elif args.command=='report': result=report(args.out)
    elif args.command=='train':
        from .train_prepared import train_one
        result=train_one(args.out,args.expert,args.model_dir,args.output,args.steps,args.rank,args.threads,args.release_check,args.small_pilot)
    else:
        path=args.out/'prepared'/'manifest.json'
        result=json.loads(path.read_text(encoding='utf-8'))['summary'] if path.exists() else {'state':'not_prepared','registered':len(catalog())}
    print(json.dumps(result,indent=2,ensure_ascii=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())