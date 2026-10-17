"""Acquire licensed human-authored instruction candidates; never auto-approve."""
import argparse
from collections import Counter
import hashlib,json
from pathlib import Path
from Tantra.config import ROOT
from Tantra.data.public_sources import fetch,location,sha256

RAW=ROOT/'Datasets/raw/conversation-sources-v1'
LOCK=ROOT/'docs/data/conversation_sources.lock.json'


def run(output):
    import pyarrow.parquet as pq
    output=Path(output)
    if output.exists():raise ValueError('Use new immutable review output')
    sources=json.loads(LOCK.read_text(encoding='utf-8'))['sources']
    output.mkdir(parents=True);counts=Counter();seen={};overlap=0
    # Process reserved test FIRST so any matching training content remains excluded.
    for source in reversed(sources):
        path=fetch(source,RAW);reserved=source['kind']=='reserved_source_test'
        filename='reserved-source-test.jsonl' if reserved else 'pending-review.jsonl'
        with (output/filename).open('w',encoding='utf-8') as writer:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                for raw in batch.to_pylist():
                    code=raw.get('language_code');language={'eng':'en','hin':'hi'}.get(code)
                    if language is None:continue
                    question,answer=raw.get('inputs'),raw.get('targets')
                    if not isinstance(question,str) or not isinstance(answer,str) or not question.strip() or not answer.strip():
                        counts['malformed']+=1;continue
                    key=hashlib.sha256(json.dumps([question,answer],ensure_ascii=False).encode()).hexdigest()
                    if key in seen:
                        counts['duplicate']+=1
                        if not reserved and seen[key]=='test':overlap+=1
                        continue
                    seen[key]='test' if reserved else 'candidate'
                    row={'candidate_id':key,'language':language,'messages':[{'role':'user','content':question.strip()},{'role':'assistant','content':answer.strip()}],
                         'source':{'id':source['repo'],'revision':source['revision'],'license':source['license'],'file':source['path'],
                                   'annotation_type':raw.get('annotation_type')},
                         'expert':None,'review':{'status':'pending','reviewer':None},
                         'source_split':'reserved_test' if reserved else 'candidate',
                         'warning':'Human source annotation does not certify Tantra identity, expert ownership or fluent-review approval.'}
                    writer.write(json.dumps(row,ensure_ascii=False)+'\n');counts[('reserved_' if reserved else 'candidate_')+language]+=1
    report={'counts':dict(counts),'source_test_duplicates_excluded_from_candidates':overlap,'source_lock_sha256':sha256(LOCK),
            'review_status':'pending','approved_training_examples':0,'hinglish':'Not inferred from Hindi or fabricated by translation',
            'limitations':['Source may contain mislabeled language or unsafe/inaccurate content','Source IDs do not establish semantic family independence; review grouping before splitting','No expert is marked data_ready by acquisition']}
    (output/'index.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)

if __name__=='__main__':main()
