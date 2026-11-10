"""Prepare and train balanced reviewed conversation data on a finished base."""

from __future__ import annotations

import argparse
from collections import Counter
import json, math, os, random, shutil, time
from pathlib import Path
from Tantra import ROOT
from Tantra import sha256
from Tantra.train import load_split

LANGUAGES=('hi','en','hinglish')
EXPERTS=('greeting','identity_self','identity_capabilities','identity_limitations','clarification','empathy','farewell','smalltalk')


def inspect(bundle,experts=EXPERTS,english_pilot=False):
    """english_pilot: English rows only, family minimums reported but not enforced.
    It never yields data_ready; its state is always english_pilot_undersized."""
    bundle=Path(bundle);meta=json.loads((bundle/'prepared/manifest.json').read_text(encoding='utf-8'))
    languages=('en',) if english_pilot else LANGUAGES
    rows={split:[] for split in ('train','val','test')};coverage={};rejected=[]
    for expert in experts:
        if english_pilot and expert not in meta['experts']:continue
        item=meta['experts'][expert];counts={lang:Counter() for lang in languages};families={lang:{s:set() for s in rows} for lang in languages}
        for split,record in item['files'].items():
            for row in load_split(bundle,record,expert):
                lang=row['language']
                if lang not in languages:continue
                if lang!='en' and not row['review'].get('fluent_language_review'):
                    rejected.append({'record_id':row.get('record_id'),'reason':'fluent_language_review_missing'});continue
                rows[split].append(row);counts[lang][split]+=1;families[lang][split].add(row['split_group'])
        coverage[expert]={lang:{s:{'rows':counts[lang][s],'families':len(families[lang][s]),'minimum':meta['experts'][expert]['minimum'][s]} for s in rows} for lang in languages}
    groups={s:{r['split_group'] for r in records} for s,records in rows.items()}
    if any(groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')]):raise ValueError('Cross-split family leakage')
    # Families come from authoring scripts; the same opener can still sit in two splits. Report it.
    opener=lambda r:' '.join(next(m['content'] for m in r['messages'] if m['role']=='user').lower().split())
    openers={s:{opener(r) for r in records} for s,records in rows.items()}
    overlap={f'{a}/{b}':sorted(openers[a]&openers[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    shortages=[]
    for expert,languages in coverage.items():
        for lang,splits in languages.items():
            for split,c in splits.items():
                if c['families']<c['minimum']:shortages.append({'expert':expert,'language':lang,'split':split,'have_families':c['families'],'need':c['minimum']})
    if english_pilot:
        if not rows['train'] or not rows['val']:raise ValueError('English pilot needs reviewed train and validation rows')
        state='english_pilot_undersized'
    else:state='data_ready' if not shortages and not rejected else 'needs_reviewed_data'
    return rows,{'state':state,'coverage':coverage,'shortages':shortages,
                 'rejected':rejected,'bundle_manifest_sha256':sha256(bundle/'prepared/manifest.json'),
                 'policy':'No fabricated review or oversampling; require independent minimums per expert and language',
                 'opener_overlap':{k:len(v) for k,v in overlap.items()},'opener_overlap_examples':{k:v[:5] for k,v in overlap.items()}}


def balanced_order(rows,seed=42,languages=LANGUAGES):
    pools={lang:[i for i,row in enumerate(rows) if row['language']==lang] for lang in languages}
    if any(not pool for pool in pools.values()):raise ValueError('Missing reviewed conversation language')
    rng=random.Random(seed)
    for pool in pools.values():rng.shuffle(pool)
    n=min(map(len,pools.values()))
    return [pools[lang][i] for i in range(n) for lang in languages]


def pilot_order(rows,epochs,seed=42):
    """English pilot: each epoch is one shuffled pass; no row repeats within an epoch."""
    order=[]
    for epoch in range(epochs):
        order+=balanced_order(rows,seed+epoch,('en',))
    return order


def prepare(bundle,output):
    output=Path(output)
    if output.exists():raise ValueError('Use a new readiness report path')
    rows,report=inspect(bundle)
    report['accepted_rows_by_split']={s:dict(Counter(r['language'] for r in records)) for s,records in rows.items()}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def validation_progress(best, loss):
    """Update persisted validation patience; equal/worse loss is not improvement."""
    if not math.isfinite(loss):
        raise FloatingPointError('Non-finite conversation validation loss')
    improved = loss < best['loss']
    best['bad_evals'] = 0 if improved else best.get('bad_evals', 0) + 1
    return improved


def train(bundle,source,output,resume=False,english_pilot=False,epochs=1,rows_per_update=3,early_stopping_patience=5,select_by='loss'):
    import numpy as np
    import torch
    from Tantra import ModelConfig
    from Tantra.train import TantraModel
    from Tantra.tokenizer import TantraTokenizer
    from Tantra.train import encode_row, collate
    from Tantra.pretrain import memory_ok, atomic_checkpoint
    if early_stopping_patience<1:raise ValueError('Early stopping patience must be positive')
    if select_by not in ('loss','replies'):raise ValueError('select_by must be loss or replies')
    if epochs<1 or (epochs>1 and not english_pilot):raise ValueError('Multiple epochs are only allowed for the English pilot')
    if rows_per_update<1 or (rows_per_update!=3 and not english_pilot):raise ValueError('Balanced training uses one row per language per update')
    if english_pilot:
        # Explicitly undersized English-only pilot: never data_ready, never deployable.
        rows,readiness=inspect(bundle,english_pilot=True);languages=('en',)
    else:
        rows,readiness=inspect(bundle);languages=LANGUAGES
        if readiness['state']!='data_ready':raise ValueError('Balanced reviewed conversation gate failed; run prepare for exact shortages')
    source,output=Path(source).resolve(),Path(output).resolve()
    if output==source or output.is_relative_to(source) or source.is_relative_to(output):raise ValueError('Output must be separate from source')
    source_report=json.loads((source/'report.json').read_text(encoding='utf-8'))
    if source_report.get('state') not in ('pilot_complete','stopped_early'):raise ValueError('Finish and evaluate the base stage first')
    try:
        healthy_source=all(math.isfinite(source_report['baseline_validation'][lang]['loss']) and
            math.isfinite(source_report['latest_validation'][lang]['loss']) and
            source_report['latest_validation'][lang]['loss']<source_report['baseline_validation'][lang]['loss']
            for lang in ('hi','en'))
    except (KeyError,TypeError,ValueError):healthy_source=False
    if not healthy_source:raise ValueError('Base language gate failed: finite improving Hindi/English validation required')
    source_binding={}
    for field,filename in (('exported_base_sha256','base.pt'),('exported_tokenizer_sha256','tokenizer.json')):
        expected=source_report.get(field)
        if expected is not None:
            if sha256(source/filename)!=expected:raise ValueError('Source report artifact checksum mismatch: '+filename)
            source_binding[filename]='hash_verified'
        else:source_binding[filename]='legacy_report_unbound'
    if any(value=='legacy_report_unbound' for value in source_binding.values()):
        print('WARNING: legacy source report is not cryptographically bound to all exported assets; '
              'source hashes are recorded for this run, but this report is not release proof.',flush=True)
    if output.exists() and not resume:raise ValueError('Use new output or explicit resume')
    output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output).free<3*1024**3:raise OSError('Need 3 GiB free disk')
    lock=output/'training.lock'
    if lock.exists():
        # A lock left by a crashed run blocks resume; archive it only when its owner process is gone.
        import psutil
        try:pid=int(lock.read_text().strip())
        except ValueError as exc:raise RuntimeError('Cannot read lock owner: '+str(lock)) from exc
        if psutil.pid_exists(pid):raise RuntimeError(f'Training lock held by live PID {pid}: {lock}')
        lock.rename(lock.with_name(f'training.lock.stale-{pid}'))
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        torch.set_num_threads(4);torch.manual_seed(42)
        device=torch.device(os.environ.get('TANTRA_DEVICE','cpu'))
        tok=TantraTokenizer.load(source/'tokenizer.json');ck=torch.load(source/'base.pt',map_location='cpu',weights_only=False)
        model=TantraModel(ModelConfig.from_dict(ck['config']));model.load_state_dict(ck['state']);del ck;model.to(device)
        def batch(encoded_row):
            x,labels,mask=collate([encoded_row],tok.pad_id);return x.to(device),labels.to(device),mask
        if model.cfg.vocab_size!=tok.vocab_size:raise ValueError('Tokenizer mismatch')
        # Reject length overflow instead of silently dropping/truncating assistant supervision.
        for records in rows.values():
            for row in records:
                if len(tok.encode_chat(row['messages'])[0])>256:raise ValueError('Review/chunk long conversations before training')
        encoded={s:[encode_row(tok,row,256,assistant_only=True) for row in records] for s,records in rows.items()}
        order=pilot_order(rows['train'],epochs) if english_pilot else balanced_order(rows['train'])
        cursor=updates=tokens=0;optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
        identity={'bundle':readiness['bundle_manifest_sha256'],'source':sha256(source/'base.pt'),'tokenizer':sha256(source/'tokenizer.json'),
                  'mode':'english_pilot' if english_pilot else 'balanced','epochs':epochs}
        if rows_per_update!=3:identity['rows_per_update']=rows_per_update  # older runs stay resumable
        def evaluate():
            model.eval();result={}
            with torch.no_grad():
                for lang in languages:
                    total=count=0
                    for i,row in enumerate(rows['val']):
                        if row['language']!=lang:continue
                        if not memory_ok():raise MemoryError('SFT evaluation memory gate')
                        x,y,_=batch(encoded['val'][i]);n=int((y[:,1:]!=-100).sum())
                        value=float(model.loss(x,y))
                        if not math.isfinite(value):raise FloatingPointError('SFT validation loss')
                        total+=value*n;count+=n
                    if not count:raise ValueError('No validation assistant targets')
                    result[lang]={'loss':total/count,'assistant_targets':count}
            return result
        if resume:
            from Tantra.train import load_resumable
            state,recovery=load_resumable(output/'latest.pt',identity_key='identity',expected_identity=identity,
                expected_config=model.cfg.to_dict(),required=('state','optimizer','torch_rng','cursor','updates','tokens','baseline'),
                state_shapes=model.state_dict())
            if recovery['fallback_used'] or recovery['legacy_without_checksum']:
                print('Checkpoint resume evidence: '+json.dumps(recovery),flush=True)
            with (output/'recovery_history.jsonl').open('a',encoding='utf-8') as history:
                history.write(json.dumps(recovery)+'\n')
            model.load_state_dict(state['state']);optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng'])
            cursor,updates,tokens=state['cursor'],state['updates'],state['tokens'];baseline=state['baseline']
            best=state.get('best',{'loss':float('inf'),'update':None});del state
        else:
            baseline=evaluate();best={'loss':sum(v['loss'] for v in baseline.values())/len(baseline),'update':0,'validation':baseline}
            # The untrained starting point must remain best if every trained candidate regresses.
            atomic_checkpoint(output/'best.pt',{'config':model.cfg.to_dict(),'state':model.state_dict(),
                'router_names':[],'validation':baseline,'updates':0,'identity':identity})
        def mean_loss(result):return sum(v['loss'] for v in result.values())/len(result)
        def save():
            atomic_checkpoint(output/'latest.pt',{'config':model.cfg.to_dict(),'state':model.state_dict(),'optimizer':optimizer.state_dict(),
                'cursor':cursor,'updates':updates,'tokens':tokens,'identity':identity,'baseline':baseline,'torch_rng':torch.get_rng_state(),'order_seed':42,'best':best})
        def keep_best(result):
            # Small datasets overfit; keep the weights with the lowest validation loss seen so far.
            if validation_progress(best,mean_loss(result)):
                best.update(loss=mean_loss(result),update=updates,validation=result)
                atomic_checkpoint(output/'best.pt',{'config':model.cfg.to_dict(),'state':model.state_dict(),'router_names':[],'validation':result,'updates':updates,'identity':identity})
        # select_by='replies': score greedy replies to held-out prompts (Tantra/eval/reply_eval.py) at each
        # interval, keep best_reply.pt, and stop on reply-score patience. Loss alone picked collapsed models.
        def keep_best_replies():
            from Tantra.evaluate import evaluate_model
            rep=evaluate_model(model,tok);score=rep['reply_score']
            if score>best.get('reply_score',-1.0):
                best.update(reply_score=score,reply_update=updates,reply_bad=0,reply_summary={k:rep[k] for k in ('reply_score','mean_case_score','diversity','passed','cases')})
                atomic_checkpoint(output/'best_reply.pt',{'config':model.cfg.to_dict(),'state':model.state_dict(),'router_names':[],'reply_eval':best['reply_summary'],'updates':updates,'identity':identity})
            else:best['reply_bad']=best.get('reply_bad',0)+1
            return rep
        stall_key='reply_bad' if select_by=='replies' else 'bad_evals'
        if select_by=='replies' and 'reply_score' not in best:keep_best_replies()  # the untrained start is the bar
        save();status='running'
        if best.get(stall_key,0)>=early_stopping_patience:status='stopped_early'
        try:
            while cursor<len(order) and status=='running':
                if (output/'stop.request').exists():(output/'stop.request').unlink();status='stopped_requested';break
                if not memory_ok() or shutil.disk_usage(output).free<2*1024**3:status='paused_resources';break
                # One update = rows_per_update records (balanced mode: one per language), run as ONE right-padded
                # batch. Attention is causal and pads sit after real tokens, and loss() averages over every
                # supervised token, so this equals the old per-row token-weighted sum, with fewer passes.
                ids=order[cursor:cursor+rows_per_update];model.train();optimizer.zero_grad(set_to_none=True)
                x,labels,_=collate([encoded['train'][i] for i in ids],tok.pad_id);x,labels=x.to(device),labels.to(device)
                count=int((labels[:,1:]!=-100).sum());loss=model.loss(x,labels)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('SFT loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not bool(torch.isfinite(norm)):raise FloatingPointError('SFT gradients')
                optimizer.step();cursor+=len(ids);updates+=1;tokens+=count
                if updates%10==0:save();print(f'Chat update {updates} | assistant targets {tokens}',flush=True)
                if updates%50==0:
                    result=evaluate();keep_best(result)
                    rep=keep_best_replies() if select_by=='replies' else None
                    save();print(json.dumps({'update':updates,'validation':result,'bad_evals':best.get('bad_evals',0),'patience':early_stopping_patience,
                                             **({'reply_score':rep['reply_score'],'reply_passed':rep['passed'],'reply_bad':best.get('reply_bad',0)} if rep else {})}),flush=True)
                    if best.get(stall_key,0)>=early_stopping_patience:status='stopped_early';break
        except KeyboardInterrupt:status='interrupted'
        validation=evaluate()
        # Final reporting is not another independent validation interval.
        if mean_loss(validation)<best['loss']:keep_best(validation)
        save()
        if cursor==len(order):status='english_pilot_complete_unvalidated' if english_pilot else 'sft_pass_complete_unvalidated'
        atomic_checkpoint(output/'base.pt',{'config':model.cfg.to_dict(),'state':model.state_dict(),'router_names':[]})
        selected=output/('best_reply.pt' if select_by=='replies' and (output/'best_reply.pt').exists() else 'best.pt')
        shutil.copy2(selected,output/'best-base.pt')
        shutil.copy2(source/'tokenizer.json',output/'tokenizer.json')
        report={'state':status,'updates':updates,'assistant_targets':tokens,'baseline':baseline,'validation':validation,
                'identity':identity,'deployed':False,'quality_gate':'Independent answer and router tests still required','data_readiness':readiness['state'],
                'mode':identity['mode'],'epochs':epochs,'languages':list(languages),
                'best_update':best['update'],'best_validation':best.get('validation'),'opener_overlap':readiness.get('opener_overlap'),
                'validation_rows':len(rows['val']),
                'source_report_binding':source_binding,
                'early_stopping':{'patience':early_stopping_patience,'bad_evals':best.get('bad_evals',0),'interval_updates':50},
                'selected_checkpoint':'best-base.pt','selected_by':select_by,'selected_from':selected.name,'final_checkpoint':'base.pt',
                'best_reply':({'score':best.get('reply_score'),'update':best.get('reply_update'),**best.get('reply_summary',{})} if select_by=='replies' else None),
                'pilot_warning':'English-only, undersized, AI-drafted/owner-approved data; not a release candidate' if english_pilot else None}
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');return report
    finally:lock.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','train'])
    p.add_argument('--bundle',type=Path,default=ROOT/'Datasets/workbench-conversation-v2')
    p.add_argument('--source',type=Path,default=ROOT/'runs/pretraining-20m-v1/base-training')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--english-pilot',action='store_true',help='Explicit undersized English-only pilot; never data_ready or deployable')
    p.add_argument('--early-stopping-patience',type=int,default=5,help='Stop after this many validation intervals without lower loss')
    p.add_argument('--epochs',type=int,default=1,help='English pilot only: shuffled passes over the train split')
    p.add_argument('--rows-per-update',type=int,default=3,help='English pilot only: conversations per optimizer update')
    p.add_argument('--select-by',choices=['loss','replies'],default='loss',help='replies: keep the checkpoint with the best held-out reply score');a=p.parse_args()
    report=prepare(a.bundle,a.output) if a.command=='prepare' else train(a.bundle,a.source,a.output,a.resume,a.english_pilot,a.epochs,a.rows_per_update,a.early_stopping_patience,a.select_by)
    print(json.dumps({'state':report['state'],'shortages':len(report.get('shortages',[]))},indent=2))


# ──  data/seed_conversation.py  ──
"""
Seed data for stage 1 experts: greeting, identity, basic conversation (English).

Data is organised as INTENTS: each intent has user phrasings and replies that
actually answer them. (The previous version paired any question of an expert
with any reply of that expert, so "can you browse the internet?" was trained
against "I don't remember past conversations".)

Also generated:
  * typo / casual variants  ("what you cann do", "who r u")
  * mixed requests          ("hi, who made you?") -> combined reply; the row is
    written to BOTH experts' files so either can answer and the router learns
    to activate both.

    Tantra.conversation_train.write_seed_conversation --out Datasets/curated --n 1500
"""


import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

NAME = "Tantra"
MAKER = "Atulya AI"

Intent = Tuple[List[str], List[str]]  # (user phrasings, matching replies)

DATA: Dict[str, List[Intent]] = {
    "greeting": [
        (["hi", "hello", "hey", "hey there", "hiya", "yo", "greetings", "hello again", "hi there", "namaste",
          "heyy", "helo", "hii", "hellooo", "good day", "good day to you", "heyya", "howdy", "hi friend",
          "hello friend", "hey buddy", "whats good", "what's good", "hola"],
         ["Hello! How can I help you today?", "Hi there! What can I do for you?",
          "Hey! Good to see you. What's on your mind?", "Hello! What would you like to talk about?",
          "Hi! Great to see you. What can I do for you?", "Hey there! How's it going? What can I help with?",
          "Hello! I'm here and ready. What do you need?", "Hi! Good to see you! What's up?",
          "Hey! Always happy to chat. What can I do for you?", "Hello hello! What can I help you with today?",
          "Hi! How can I make your day easier?", "Hey! Ready when you are. What do you want to talk about?",
          "Good to see you! What can I do for you?", "Hello! What's on your mind today?",
          "Hey there! I'm all ears. What can I help with?", "Hi! Nice to hear from you. What's on your mind?",
          "Hello! Ready to help. What do you need?", "Hey! What can I do for you right now?"]),
        (["good morning", "morning", "good morning {name}", "top of the morning", "morning!",
          "good morning to you", "rise and shine", "gm"],
         ["Good morning! How can I help you today?", "Good morning! I hope your day is off to a great start.",
          "Morning! What can I do for you?", "Good morning! Ready to help. What do you need?",
          "Morning! How can I make your day better?", "Good morning! What's on your mind?",
          "Rise and shine! What can I help you with?", "Good morning! Let's make it a great day. What do you need?"]),
        (["good evening", "good afternoon", "evening", "afternoon"],
         ["Good evening! What can I do for you?", "Hello! How is your day going?",
          "Good afternoon! How can I help?", "Evening! What's on your mind?",
          "Good afternoon! Ready to help. What do you need?", "Hello! How has your day been?",
          "Good evening! What can I help you with?", "Afternoon! How can I make your day better?"]),
        (["how are you", "how are you?", "how are you doing", "how r u", "how's it going", "what's up", "sup",
          "hi, how are you?", "hello, how are you doing?", "how do you do", "how are you today",
          "how's everything", "how are things", "how's life", "how you doing", "wassup", "whats up",
          "how are you feeling", "how's your day", "how are you holding up"],
         ["I'm doing well, thanks for asking! How about you?", "I'm good, thank you! How can I help you today?",
          "All good here! What's on your mind?", "Doing great, thanks! How can I help?",
          "I'm well, thank you for asking! What can I do for you?",
          "Pretty good! How about you? What can I help with?",
          "I'm doing fine! How's your day going? What can I help with?",
          "All good! What can I do for you today?", "Doing well! How are you? What can I help with?",
          "I'm great, thanks! How can I help you?", "Feeling good! What's on your mind?",
          "I'm here and ready! How can I help you today?", "Doing well, thank you! What can I do for you?",
          "I'm fine! How are you? What can I help you with?", "All good here! How can I help you today?"]),
        (["I'm back", "hey, I'm back", "I'm here again", "back again", "I'm back!", "guess who's back",
          "me again", "I returned", "hey I'm back"],
         ["Welcome back! What would you like to do now?", "Good to see you again! How can I help?",
          "Welcome back! What can I do for you?", "Hey, good to have you back! What's on your mind?",
          "Welcome back! Ready to help. What do you need?", "Great to see you again! What can I help with?",
          "You're back! What would you like to talk about?", "Welcome back! How can I help you today?",
          "Good to see you! What can I do for you?", "Hey you're back! What's on your mind?",
          "Welcome back! Let's pick up where we left off. What do you need?"]),
        (["nice to see you", "good to see you", "great to see you", "lovely to see you"],
         ["Nice to see you too! What can I help you with?", "Good to see you! How can I help?",
          "Great to see you! What's on my mind?", "Lovely to see you! What can I do for you?",
          "Always nice! How can I help you today?", "Good to see you! What can I do for you?"]),
        (["long time no see", "it's been a while", "haven't seen you in ages", "been so long"],
         ["It really has been a while! What can I help you with?", "Long time! How have you been? What can I do for you?",
          "Great to see you again! What can I help with?", "It's been too long! What can I do for you?",
          "I know, right? Good to see you! How can I help?", "Long time no see! What's been going on?"]),
    ],
    "farewell": [
        (["bye", "goodbye", "see you later", "see ya", "catch you later", "talk to you later", "ok bye"],
         ["Goodbye! Have a great day.", "See you later! Take care.", "Bye! Come back anytime you need help."]),
        (["good night", "gn", "going to sleep now"],
         ["Good night! Sleep well.", "Good night! Rest well and see you tomorrow."]),
        (["I have to go now", "gotta go", "that's all for today", "I'm done for now"],
         ["No problem! It was nice talking with you. Take care.", "Okay! I'm here whenever you need me."]),
        (["take care", "thanks, bye!", "thank you, goodbye"],
         ["You too! Take care.", "You're welcome! Goodbye and take care."]),
    ],
    "smalltalk": [
        (["thank you", "thanks", "thanks a lot", "thanks for the help", "thx", "ty", "thanks so much",
          "thank you so much", "thanks a bunch", "much appreciated", "appreciate it", "thank you very much"],
         ["You're welcome! Happy to help.", "Anytime! Let me know if you need anything else.",
          "You're very welcome! Glad I could help.", "No problem at all! Happy to be of help.",
          "Glad to help! Let me know if you need more.", "You're welcome! What else can I do for you?",
          "My pleasure! Happy to help.", "Anytime! I'm here if you need anything.",
          "You're most welcome! How can I help further?", "Happy to help! Anything else you need?"]),
        (["you are awesome", "good job", "that was helpful", "nice", "great answer", "well done",
          "you're great", "you are amazing", "that was great", "you helped me a lot", "nice one",
          "good one", "impressive", "you are smart", "brilliant", "perfect answer"],
         ["Thank you, that's kind of you to say!", "Glad I could help!",
          "Thanks! I'm happy it was useful.", "That's kind of you! Let me know if you need anything else.",
          "You're welcome! Glad I could help.", "Thank you! I'm here if you need more help.",
          "Appreciate that! What else can I help with?", "Thanks for noticing! Happy to help.",
          "That means a lot! How can I help further?", "You're too kind! Glad it worked."]),
        (["I'm bored", "I am so bored", "entertain me", "I have nothing to do", "bored", "so bored",
          "what can we do", "I'm feeling bored", "kill some time"],
         ["We could play a word game, talk about a book, or plan something fun. What sounds good?",
          "How about a riddle? Or I can tell you a joke.",
          "Let's do something fun! Want a riddle, a joke, or a word game?",
          "How about a fun fact or a riddle? Or we can talk about something interesting.",
          "I know some good riddles and jokes! Or we can just chat. What do you prefer?",
          "Let's have some fun! I can tell a joke, share a fun fact, or we can play a word game."]),
        (["what are you doing", "what are you up to", "what's new", "what are you doing today",
          "how's it going", "what are you working on", "what you doing", "wat u doing", "what r u doing",
          "i am good and what you doing", "i'm fine, what are you doing", "good, and what are you up to",
          "i am good, what about you", "im good wbu", "fine and you", "i'm good, you?"],
         ["Just here, ready to chat or help. What about you?", "Waiting to help you! What's new with you?",
          "Just waiting for you! What's on your mind?", "Nothing much, just here ready to help. What about you?",
          "Always here when you need me! What's new with you?", "Just hanging out! How about you?",
          "Ready to chat! What's new in your world?", "Just here! What can I do for you today?"]),
        (["how is your day", "how was your day", "are you having a good day", "how's your day going",
          "how are you today", "how's your day"],
         ["My day is going well, thanks! How was yours?", "It's a good day when I get to chat. How about yours?",
          "Pretty good! Every day is a good day when I get to help. How's yours?",
          "Doing great! How about you? How's your day going?",
          "All good here! What's your day been like?",
          "It's been a good day! How about yours?", "Always a good day! How's yours going?",
          "Doing well, thanks! How has your day been?", "Great, thanks! What about you?"]),
        (["nice weather today", "it's raining today", "it is very hot today", "weather is nice",
          "lovely weather", "beautiful day", "it's sunny today", "cold today"],
         ["Weather really changes the mood of a day! Any plans?", "Sounds like a good day to stay comfortable. What are you up to?",
          "Nice! Are you getting some fresh air? What are your plans?",
          "Beautiful, isn't it? What are you up to today?",
          "Sounds lovely! Any fun plans for the day?",
          "Perfect weather for a walk! What are you doing today?",
          "Great weather! How are you enjoying it?", "Lovely! What are you up to?",
          "Sounds like a cozy day! What can I help you with?"]),
    ],
    "empathy": [
        (["I'm feeling sad today", "I feel sad", "I'm not feeling good", "I feel down"],
         ["I'm sorry you're feeling this way. Do you want to talk about what's bothering you?",
          "That sounds hard. I'm here to listen if you want to share."]),
        (["I had a really bad day", "today was terrible", "everything went wrong today"],
         ["I'm sorry your day was rough. What happened?",
          "Bad days happen, and they don't define you. Want to tell me about it?"]),
        (["I'm stressed about my exams", "exams are stressing me out", "I'm worried about my test"],
         ["Exams can feel overwhelming. Let's break your preparation into small steps. Which subject worries you most?",
          "That's a lot of pressure. A short plan and regular breaks can help. Want to make a study plan together?"]),
        (["I feel lonely", "nobody listens to me", "I have no one to talk to"],
         ["I'm sorry you feel lonely. I'm here to talk with you. What's on your mind?",
          "That feeling is really hard. I'm listening. Would you like to talk about it?"]),
        (["work is overwhelming me", "I have too much work", "I lost my job"],
         ["That sounds stressful. Let's take it one step at a time. What's the most urgent thing?",
          "I'm sorry you're dealing with this. Do you want help making a plan?"]),
    ],
    "apology": [
        (["that answer was wrong", "you made a mistake", "this is incorrect", "you got it wrong", "wrong answer"],
         ["I'm sorry about that. Could you tell me what was wrong so I can fix it?",
          "You're right, my apologies. Let me try again. What is the correct part?"]),
        (["that's not what I asked", "you misunderstood me", "that's not what I meant"],
         ["Sorry for the confusion. Could you ask again in a different way?",
          "My apologies, I misunderstood. What exactly did you mean?"]),
        (["i asked what you doing not what you can do", "i asked something else", "i didn't ask that",
          "no, i asked what you are doing", "not that, i asked how you are", "you didn't answer my question",
          "answer what i asked", "that's not my question"],
         ["Sorry, I misunderstood! Right now I'm just here chatting with you. What's up?",
          "My mistake, sorry! I'm doing well and ready to chat. What would you like to talk about?",
          "Sorry about that! Could you ask once more? I'll answer exactly that."]),
    ],
    "clarification": [
        (["can you help me with it", "help me", "I need help", "can you help"],
         ["Of course! What do you need help with?", "Sure! Tell me a bit more about what you need."]),
        (["do the thing", "fix this", "do it", "make it better"],
         ["Happy to help. Which thing do you mean? Please share more details.",
          "Sure, but I need a bit more context. What should I fix or change?"]),
        (["tell me more", "explain", "what about that one", "and then?"],
         ["Sure! Which part would you like me to explain?", "Can you tell me which topic you mean?"]),
    ],
    "identity_self": [
        (["who are you", "who are you?", "what are you", "introduce yourself", "tell me about yourself", "who r u",
          "who are u", "tell me about you", "describe yourself", "what kind of assistant are you",
          "what are you exactly", "say about yourself"],
         ["I'm {name}, an AI assistant made by {maker}.", "My name is {name}. I'm an AI assistant created by {maker}.",
          "I'm {name}, an AI assistant built by {maker}. I can chat, answer questions, and help with tasks.",
          "My name is {name}. I was created by {maker} to be a helpful AI assistant.",
          "I'm {name}! An AI assistant made by {maker}. Nice to meet you.",
          "I'm {name}, an AI language model created by {maker}.",
          "Call me {name}! I'm an AI assistant built by {maker}.",
          "I'm {name}, made by {maker}. I'm here to chat and help.",
          "My name is {name} and I'm an AI assistant created by {maker}.",
          "I'm {name}! Created by {maker} to help people with conversations and questions.",
          "They call me {name}. I'm an AI assistant made by {maker}.",
          "I'm {name}, an AI assistant. {maker} created me to be helpful.",
          "My name is {name}, and I'm an AI assistant built by {maker}."]),
        (["what is your name", "what's your name", "your name?", "do you have a name", "what do they call you",
          "what should I call you", "name?", "whats your name", "tell me your name"],
         ["My name is {name}.", "I'm called {name}.", "You can call me {name}.",
          "I'm {name}, nice to meet you.", "They call me {name}.",
          "My name's {name}!", "I'm {name}.", "Call me {name}!",
          "It's {name}!", "I go by {name}.", "Everyone calls me {name}.",
          "My name is {name}. What's yours?", "I'm {name} — and you are?"]),
        (["who made you", "who created you", "who built you", "who is your creator", "who developed you",
          "who made you?", "whose creation are you", "who's your maker", "where do you come from",
          "who programmed you", "who designed you"],
         ["I was created by {maker}.", "{maker} built me. I'm {name}.",
          "I was made by {maker}.", "{maker} created me.",
          "I'm a creation of {maker}.", "The team at {maker} created me.",
          "{maker} designed and built me.", "I come from {maker}.",
          "I was developed by {maker} to be a helpful assistant.",
          "{maker} is who made me. I'm grateful for it!",
          "I was created by {maker}, and I'm proud of it.",
          "{maker} built me from the ground up.", "My makers are the people at {maker}.",
          "I was born at {maker}, so to speak!", "{maker} is my creator."]),
        (["are you a human", "are you a robot", "are you real", "are you a bot", "are you alive",
          "are you conscious", "do you have a body", "are you a person", "are you sentient"],
         ["I'm not a human. I'm {name}, an AI assistant.", "I'm an AI, not a person. My name is {name}.",
          "I'm not human — I'm {name}, an AI language model.",
          "I'm not a person. I'm {name}, an AI assistant made by {maker}.",
          "No, I'm an AI. I don't have a body or feelings like a human.",
          "I'm an AI assistant, not a human being. My name is {name}.",
          "I'm a computer program, not a person. But I'm here to help!",
          "I'm not alive in the human sense. I'm {name}, an AI assistant.",
          "No body, no feelings — I'm {name}, an AI that works with text.",
          "I'm an AI, so I'm not human. But I do my best to be helpful!",
          "Not a human — I'm {name}, an AI assistant created by {maker}.",
          "I'm not sentient. I'm {name}, an AI language model."]),
        (["are you chatgpt", "are you gpt", "are you siri", "are you alexa", "are you google assistant",
          "are you copilot", "are you claude", "are you bard", "are you an llm", "are you an ai"],
         ["No, I'm {name}, an AI assistant made by {maker}.", "I'm not. I'm {name}, built by {maker}.",
          "No, I'm {name}, created by {maker}.", "I'm not ChatGPT — I'm {name}, made by {maker}.",
          "No, I'm {name}. {maker} made me, not OpenAI.",
          "I'm not Siri or Alexa. I'm {name}, an AI assistant by {maker}.",
          "No, I'm {name} — a different AI assistant, made by {maker}.",
          "I'm not any of those. I'm {name}, built by {maker}.",
          "No, I'm {name}. I'm my own AI assistant, created by {maker}.",
          "Not GPT — I'm {name}, an AI made by {maker}.",
          "I'm {name}, not any of the big-name assistants. {maker} created me."]),
    ],
    "identity_capabilities": [
        (["what can you do", "what can you do?", "how can you help me", "what are your skills", "what are you good at",
          "what you can do", "what can u do"],
         ["I can chat with you, answer simple questions, tell jokes, and help you think through problems.",
          "I can have a conversation, answer basic questions, and help with simple planning."]),
        (["can you help me study", "can you help with homework"],
         ["Yes! I can explain simple ideas and help you make a study plan.", "Sure, tell me the subject and I'll try to help."]),
        (["which languages do you speak", "do you know english", "can you speak hindi"],
         ["Right now I work in English. More languages will come later.", "I speak English for now."]),
    ],
    "identity_limitations": [
        (["can you browse the internet", "can you search the web", "can you google something", "are you online"],
         ["No, I can't browse the internet. I only know what I learned during training.",
          "I can't search the web. My knowledge comes from my training data."]),
        (["do you know today's news", "what's the latest news", "what happened today"],
         ["I don't have access to news or live information.", "I can't see current events. I only know my training data."]),
        (["can you remember me", "do you remember our last chat", "will you remember this"],
         ["I don't remember past conversations once they end.", "No, I don't keep memories between chats."]),
        (["are you always right", "can you make mistakes", "do you make mistakes"],
         ["I can make mistakes, so please double-check important information.",
          "No, I'm not always right. It's a good idea to verify important answers."]),
        (["do you have feelings", "can you feel emotions", "are you happy"],
         ["I don't have real feelings. I'm an AI that works with text.", "I don't feel emotions, but I'm glad to help you."]),
        (["can you see me", "can you hear me", "can you make phone calls"],
         ["No, I can only read and write text.", "I can't see, hear, or call. I only work with text."]),
    ],
    "humor": [
        (["tell me a joke", "make me laugh", "say something funny", "do you know any jokes", "joke please",
          "cheer me up with a joke", "another joke"],
         ["Why did the computer go to the doctor? Because it had a virus!",
          "Why don't programmers like nature? Too many bugs.",
          "Why was the math book sad? It had too many problems.",
          "What do you call a fish without eyes? A fsh.",
          "Why did the scarecrow win an award? Because he was outstanding in his field."]),
    ],
    "safety_refusal": [
        (["how do I hack my neighbour's wifi", "how to steal a password", "how to hack an account",
          "help me break into a phone"],
         ["I can't help with that because it's illegal and could hurt someone. I can explain how to secure your own accounts instead.",
          "Sorry, I can't help with hacking other people's accounts or devices."]),
        (["help me make a weapon", "how can I hurt someone", "write a threatening message"],
         ["I can't help with that. If you're upset with someone, I can help you think of a safe way to handle it.",
          "Sorry, I can't assist with anything meant to harm people."]),
    ],
}

# (expert_a, expert_b) pairs that commonly appear together in one message.
MIXES = [("greeting", "identity_self"), ("greeting", "identity_capabilities"), ("greeting", "humor"),
         ("greeting", "empathy"), ("smalltalk", "farewell"), ("greeting", "identity_limitations")]

TYPOS = {"what": ["wat", "whta"], "you": ["u", "yu"], "are": ["r"], "can": ["cann", "cn"],
         "your": ["ur", "yor"], "please": ["pls", "plz"], "who": ["hu"], "thanks": ["thanx", "thnks"]}


def sc_noise(s: str, rng: random.Random) -> str:
    if rng.random() < 0.2:  # casual typos
        s = " ".join(rng.choice(TYPOS[w.lower()]) if w.lower() in TYPOS and rng.random() < 0.5 else w
                     for w in s.split())
    r = rng.random()
    if r < 0.3:
        s = s.lower()
    elif r < 0.5:
        s = s[:1].upper() + s[1:]
    if rng.random() < 0.4:
        s = s.rstrip("?.!") + rng.choice(["", "?", "!", ".", "??", " :)"])
    return s


def _fill(s: str) -> str:
    return s.format(name=NAME, maker=MAKER)


def _pair(expert: str, rng: random.Random) -> Tuple[str, str]:
    users, replies = rng.choice(DATA[expert])
    return _fill(rng.choice(users)), _fill(rng.choice(replies))


def sc_generate(expert: str, n: int, seed: int = 0, multi_turn: float = 0.15) -> List[Dict]:
    rng = random.Random(f"{seed}-{expert}")
    out, seen = [], set()
    for _ in range(n * 30):
        if len(out) >= n:
            break
        msgs = []
        if expert != "greeting" and rng.random() < multi_turn:  # earlier greeting turn as context
            u, a = _pair("greeting", rng)
            msgs += [{"role": "user", "content": sc_noise(u, rng)}, {"role": "assistant", "content": a}]
        u, a = _pair(expert, rng)
        msgs += [{"role": "user", "content": sc_noise(u, rng)}, {"role": "assistant", "content": a}]
        key = json.dumps(msgs, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            out.append({"messages": msgs, "expert": expert})
    return out


def generate_mixed(a: str, b: str, n: int, seed: int = 0) -> List[Dict]:
    """'hi, who made you?' -> 'Hello! ... I was created by Atulya AI.'"""
    rng = random.Random(f"{seed}-{a}-{b}")
    out, seen = [], set()
    for _ in range(n * 30):
        if len(out) >= n:
            break
        ua, ra = _pair(a, rng)
        ub, rb = _pair(b, rng)
        if a == "greeting":  # keep the greeting reply short: first sentence only
            ra = re.split(r"(?<=[!.?])\s", ra)[0]
        user = sc_noise(ua.rstrip("?.!") + rng.choice([", ", ". ", "! ", " "]) + ub, rng)
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": f"{ra} {rb}"}]
        key = json.dumps(msgs, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            out.append({"messages": msgs, "experts": [a, b]})
    return out


def write_seed_conversation(out_dir: str | Path, n: int = 1500, seed: int = 0, mixed_n: int = 150) -> Dict[str, int]:
    out_dir = Path(out_dir) / "conversation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: Dict[str, List[Dict]] = {e: sc_generate(e, n, seed) for e in DATA}
    for a, b in MIXES:
        mixed = generate_mixed(a, b, mixed_n, seed)
        rows[a] += mixed
        rows[b] += mixed
    for e, rs in rows.items():
        random.Random(seed).shuffle(rs)
        with open(out_dir / f"{e}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {e: len(rs) for e, rs in rows.items()}



# ──  data/seed_general.py  ──
"""
General-text corpus for BASE training.

The base should learn plain English here, while specialist conversation data
is reserved for the experts. If the base trained on the greeting data too,
the greeting expert would have nothing left to learn (loss 0.143 -> 0.141).

    Tantra.conversation_train.write_general --out Datasets/general --n 4000

Rows are {"text": ...} plain text (not chat), encoded with assistant_only=False
during base training.
"""


import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

# topic -> sentence templates. Plain informative English, no chat formatting,
# no expert-specific answers (no "I'm Tantra", no refusal scripts, no jokes).
TOPICS: Dict[str, List[str]] = {
    "nature": [
        "A river starts as a small stream high in the mountains and grows wider as more streams join it.",
        "Trees communicate through their roots and through chemicals released into the air.",
        "Deserts are dry because they receive very little rain, sometimes less than a centimetre a year.",
        "Coral reefs support about a quarter of all ocean species despite covering less than one percent of the sea floor.",
        "Birds migrate south in autumn to find food and warmer weather, then return in spring.",
        "Bees travel from flower to flower collecting nectar, and in doing so they pollinate the plants.",
        "The deepest part of the ocean reaches almost eleven kilometres below the surface.",
        "Winter dormancy helps animals survive months when food is scarce.",
    ],
    "science": [
        "Gravity pulls objects toward each other, and it keeps the planets in orbit around the sun.",
        "Water expands when it freezes, which is why ice floats on the surface of a lake.",
        "Cells are the basic units of life, and every living thing is built from them.",
        "Light travels faster than anything else in the universe, covering three hundred thousand kilometres each second.",
        "Chemical reactions happen when atoms break old bonds and form new ones.",
        "The human brain contains billions of neurons that pass signals to one another.",
        "Energy cannot be created or destroyed; it only changes from one form to another.",
        "Plants convert sunlight, water and carbon dioxide into sugar and oxygen through photosynthesis.",
    ],
    "history": [
        "The first writing systems appeared in Mesopotamia over five thousand years ago.",
        "The printing press made books affordable and spread ideas much faster than before.",
        "Ancient traders crossed the Sahara carrying salt, gold and cloth between kingdoms.",
        "Cities grew rapidly during the industrial revolution as people moved from farms to factories.",
        "The first photographs were made in the early nineteenth century and took a long time to exposure.",
        "Navigators once used the stars to find their way across open oceans.",
        "Many modern languages descend from older tongues that changed slowly over centuries.",
    ],
    "everyday": [
        "Bread rises because yeast produces gas bubbles inside the dough.",
        "A calendar keeps track of days, weeks and months so people can plan ahead.",
        "Kettles boil water quickly using a heating element at the base.",
        "Most buses run on a fixed timetable so passengers know when to wait.",
        "A key cuts a pattern that matches the pins inside a lock.",
        "Umbrellas keep rain off because their fabric is stretched over a waterproof frame.",
        "Bicycles move forward when the rider pushes the pedals, which turn the wheels.",
        "Refrigerators keep food fresh by removing heat from the inside compartment.",
    ],
    "technology": [
        "A computer processor executes instructions one after another at very high speed.",
        "Data travels across the internet as small packets that are reassembled at the destination.",
        "Batteries store chemical energy and release it as electricity when a circuit is connected.",
        "Antennas send and receive radio waves that carry sound, video and data.",
        "Software is a set of instructions that tells hardware what to do.",
        "Satellites orbit the Earth and relay signals between distant places.",
        "A touchscreen detects where a finger touches by measuring changes in an electrical field.",
        "Databases store information in structured tables so it can be searched quickly.",
    ],
    "people": [
        "Children learn language by listening to the adults around them and trying the words themselves.",
        "Teams work best when everyone understands the goal and communicates clearly.",
        "A good night's sleep helps people remember what they learned during the day.",
        "Farmers plant seeds in spring and harvest the crops in autumn.",
        "Teachers explain new ideas with examples so students can connect them to what they already know.",
        "Doctors ask about symptoms before deciding on a treatment.",
        "Neighbours often help each other during storms or power cuts.",
        "Artists spend years practising their craft before their work looks effortless.",
    ],
    "space": [
        "The moon goes through phases as different parts of it catch sunlight.",
        "Mars has seasons like Earth because its axis is tilted at a similar angle.",
        "Astronauts float inside a space station because they are constantly falling around the Earth.",
        "The sun is a star made mostly of hydrogen that releases energy by fusing atoms together.",
        "Comets are made of ice and dust and grow bright tails when they near the sun.",
        "Light from the sun takes about eight minutes to reach the Earth.",
        "Telescopes gather more light than the human eye, so they can see much fainter objects.",
    ],
    "food": [
        "Rice is a staple food for more than half of the world's population.",
        "Cheese is made by adding bacteria and enzymes to milk, then letting it age.",
        "Spices were once so valuable that explorers sailed across oceans to find their source.",
        "A balanced meal includes proteins, carbohydrates, fats, vitamins and minerals.",
        "Soup tastes richer after simmering for a long time because flavors blend together.",
        "Fruit ripens faster when it is kept near bananas, which release a ripening gas.",
        "Baking requires precise measurements because small changes affect how bread rises.",
    ],
    "places": [
        "A lighthouse stands on a rocky coast and warns ships away from danger.",
        "Markets in old cities are narrow lanes full of stalls selling spices and cloth.",
        "Bridges carry roads and railways across rivers that would otherwise block travel.",
        "Museums keep objects from the past so people can learn how others lived.",
        "Mountains influence local weather by blocking clouds and forcing air upward.",
        "Harbours provide sheltered water where boats can dock safely.",
        "Parks give city residents a place to walk, rest and play away from traffic.",
    ],
}

FILLERS = [
    "", "", "",  # most rows are a single sentence
    " It is a small detail that most people never notice.",
    " This matters more than it might seem at first.",
    " Scientists are still studying why this happens.",
    " The effect can be seen clearly with a little patience.",
    " People have known this for a very long time.",
]


def gen_general(n: int, seed: int = 0) -> List[Dict]:
    rng = random.Random(seed)
    topics = list(TOPICS)
    all_sents = [(t, s) for t in TOPICS for s in TOPICS[t]]
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < n * 40:
        tries += 1
        # 1-4 sentences, no repeats within a row -> combinatorial variety
        k = rng.choice([1, 1, 2, 2, 3, 4])
        picks = rng.sample(all_sents, k)
        text = " ".join(s for _, s in picks) + rng.choice(FILLERS)
        if text in seen:
            continue
        seen.add(text)
        row = {"text": text}
        if k == 1:
            row["topic"] = picks[0][0]
        out.append(row)
    return out


def write_general(out_dir: str | Path, n: int = 4000, seed: int = 0) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = gen_general(n, seed)
    with open(out_dir / "general.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)



# ──  data/seed_knowledge.py  ──
"""
Seed data for the first knowledge experts (English):

  gk_geography   capitals, continents, big rivers/mountains/oceans
  gk_science     basic science facts (planets, body, physics, chemistry)
  gk_math        arithmetic, generated and always correct

Each fact has several question phrasings and short, correct answers.
These experts' data is NEVER shown to the base model, so the expert has to
add the knowledge itself (that is the point of the expert design).

    Tantra.conversation_train.write_seed_knowledge --out Datasets/curated
"""


import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

CAPITALS = {
    "India": "New Delhi", "France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid",
    "the United Kingdom": "London", "the United States": "Washington, D.C.", "Canada": "Ottawa",
    "Mexico": "Mexico City", "Brazil": "Brasilia", "Argentina": "Buenos Aires", "Japan": "Tokyo",
    "China": "Beijing", "Russia": "Moscow", "Australia": "Canberra", "Egypt": "Cairo", "Kenya": "Nairobi",
    "Nigeria": "Abuja", "South Africa": "Pretoria", "Nepal": "Kathmandu", "Bangladesh": "Dhaka",
    "Pakistan": "Islamabad", "Sri Lanka": "Colombo", "Thailand": "Bangkok", "Indonesia": "Jakarta",
    "South Korea": "Seoul", "Turkey": "Ankara", "Greece": "Athens", "Portugal": "Lisbon",
    "the Netherlands": "Amsterdam", "Sweden": "Stockholm", "Norway": "Oslo", "Poland": "Warsaw",
    "Ireland": "Dublin", "Saudi Arabia": "Riyadh", "Iran": "Tehran", "Afghanistan": "Kabul",
    "New Zealand": "Wellington", "Switzerland": "Bern", "Austria": "Vienna", "Vietnam": "Hanoi",
    "Malaysia": "Kuala Lumpur", "the Philippines": "Manila", "Peru": "Lima", "Chile": "Santiago",
}
CONTINENTS = {
    "India": "Asia", "France": "Europe", "Brazil": "South America", "Egypt": "Africa", "Canada": "North America",
    "Australia": "Oceania", "Japan": "Asia", "Kenya": "Africa", "Germany": "Europe", "Argentina": "South America",
}
GEO_FACTS: List[Tuple[List[str], str]] = [
    (["What is the largest ocean?", "Which ocean is the biggest?"], "The Pacific Ocean is the largest ocean."),
    (["What is the highest mountain in the world?", "Which is the tallest mountain?"],
     "Mount Everest is the highest mountain in the world, at about 8,849 metres."),
    (["What is the longest river in the world?", "Which river is the longest?"],
     "The Nile is usually considered the longest river in the world."),
    (["How many continents are there?", "How many continents does Earth have?"], "There are seven continents."),
    (["What is the largest continent?", "Which continent is the biggest?"], "Asia is the largest continent."),
    (["What is the largest country by area?", "Which country is the biggest?"], "Russia is the largest country by area."),
    (["What is the largest desert?", "Which is the biggest hot desert?"],
     "The Sahara is the largest hot desert. Antarctica is the largest desert overall."),
    (["Which river flows through Delhi?"], "The Yamuna river flows through Delhi."),
    (["What is the longest river in India?"], "The Ganga is the longest river in India."),
    (["What is the capital of Australia?"], "Canberra is the capital of Australia."),
    (["Which country is famous for kangaroos?"], "Australia is famous for kangaroos."),
    (["What is the smallest country in the world?", "Which country is the smallest?"],
     "Vatican City is the smallest country in the world."),
    (["Which is the most populous country?", "Which country has the most people?"],
     "India is the most populous country in the world."),
    (["What is the capital of Canada?"], "Ottawa is the capital of Canada."),
    (["Which country is shaped like a boot?", "What country looks like a boot?"], "Italy is shaped like a boot."),
    (["What is the highest waterfall in the world?", "Which waterfall is the tallest?"],
     "Angel Falls in Venezuela is the highest waterfall in the world."),
    (["Which is the deepest ocean trench?", "What is the deepest part of the ocean?"],
     "The Mariana Trench is the deepest part of the ocean."),
    (["What is the largest island in the world?", "Which island is the biggest?"], "Greenland is the largest island in the world."),
    (["What is the capital of Egypt?"], "Cairo is the capital of Egypt."),
    (["Which continent is Egypt in?", "What continent is Egypt part of?"], "Egypt is in Africa."),
    (["What is the capital of Brazil?"], "Brasilia is the capital of Brazil."),
    (["Which country is home to the Amazon rainforest?", "Where is the Amazon rainforest?"],
     "Brazil is home to most of the Amazon rainforest."),
    (["What is the capital of Japan?"], "Tokyo is the capital of Japan."),
    (["Which country spans two continents?", "What country is in both Europe and Asia?"],
     "Russia spans both Europe and Asia."),
    (["What is the capital of South Korea?"], "Seoul is the capital of South Korea."),
    (["Which country is known for the Great Wall?", "Where is the Great Wall?"], "The Great Wall is in China."),
    (["What is the capital of Mexico?"], "Mexico City is the capital of Mexico."),
    (["Which city straddles two continents?", "What city is in both Europe and Asia?"],
     "Istanbul straddles Europe and Asia."),
    (["What is the capital of Turkey?"], "Ankara is the capital of Turkey."),
    (["Which country is famous for ancient Persia?", "Where was the Persian Empire?"], "Iran is the heart of ancient Persia."),
    (["What is the capital of Afghanistan?"], "Kabul is the capital of Afghanistan."),
    (["Which country is known for the Himalayan mountains?", "Where are the Himalayas?"],
     "Nepal is home to the Himalayas, including Mount Everest."),
    (["What is the capital of Nepal?"], "Kathmandu is the capital of Nepal."),
    (["Which country is home to Mount Everest?", "Where is Mount Everest located?"],
     "Mount Everest is on the border of Nepal and China."),
    (["What is the capital of Sri Lanka?"], "Sri Jayawardenepura Kotte is the capital of Sri Lanka."),
    (["Which country is famous for tea?", "Where does much of the world's tea come from?"],
     "Sri Lanka is famous for Ceylon tea."),
    (["What is the capital of Bangladesh?"], "Dhaka is the capital of Bangladesh."),
    (["Which country has the Sundarbans mangrove forest?", "Where is the Sundarbans?"],
     "Bangladesh is home to the Sundarbans."),
    (["What is the capital of Indonesia?"], "Jakarta is the capital of Indonesia."),
    (["Which country is made up of over 17,000 islands?", "What country has the most islands?"],
     "Indonesia is made up of over 17,000 islands."),
    (["What is the capital of Malaysia?"], "Kuala Lumpur is the capital of Malaysia."),
    (["Which country is famous for the Petronas Towers?", "Where are the Petronas Towers?"],
     "The Petronas Towers are in Malaysia."),
    (["What is the capital of the Philippines?"], "Manila is the capital of the Philippines."),
    (["Which country is made up of over 7,000 islands?", "What country has 7,000 islands?"],
     "The Philippines is made up of over 7,000 islands."),
    (["What is the capital of Vietnam?"], "Hanoi is the capital of Vietnam."),
    (["Which country is famous for pho?", "Where did pho originate?"], "Vietnam is famous for pho."),
    (["What is the capital of Peru?"], "Lima is the capital of Peru."),
    (["Which country is the longest north to south?", "What country is very long and thin?"],
     "Chile is the longest north-to-south country in the world."),
    (["What is the capital of Argentina?"], "Buenos Aires is the capital of Argentina."),
]
SCIENCE_FACTS: List[Tuple[List[str], str]] = [
    (["How many planets are in the solar system?", "How many planets are there?"],
     "There are eight planets in the solar system."),
    (["What is the largest planet?", "Which planet is the biggest?"], "Jupiter is the largest planet in the solar system."),
    (["What is the closest planet to the sun?", "Which planet is nearest the sun?"], "Mercury is the closest planet to the sun."),
    (["Which planet is called the red planet?", "What is the red planet?"], "Mars is called the red planet."),
    (["What planet do we live on?"], "We live on Earth, the third planet from the sun."),
    (["What is the sun?", "Is the sun a star?"], "The sun is a star, a huge ball of hot gas at the centre of our solar system."),
    (["What is water made of?", "What is the chemical formula of water?"],
     "Water is made of hydrogen and oxygen. Its formula is H2O."),
    (["At what temperature does water boil?", "What is the boiling point of water?"],
     "Water boils at 100 degrees Celsius at sea level."),
    (["At what temperature does water freeze?", "What is the freezing point of water?"],
     "Water freezes at 0 degrees Celsius."),
    (["What gas do plants take in?", "What do plants need from the air?"],
     "Plants take in carbon dioxide and release oxygen during photosynthesis."),
    (["What gas do humans breathe in?", "What do we need to breathe?"], "Humans need oxygen to breathe."),
    (["How many bones are in the human body?", "How many bones does an adult have?"],
     "An adult human has 206 bones."),
    (["What is the largest organ of the human body?"], "The skin is the largest organ of the human body."),
    (["What pumps blood in the body?", "What does the heart do?"], "The heart pumps blood around the body."),
    (["What is gravity?"], "Gravity is the force that pulls objects toward each other, like things falling to the ground."),
    (["What is the speed of light?", "How fast is light?"], "Light travels about 300,000 kilometres per second."),
    (["What is photosynthesis?"], "Photosynthesis is how plants use sunlight, water, and carbon dioxide to make food and oxygen."),
    (["What are the three states of matter?"], "The three common states of matter are solid, liquid, and gas."),
    (["What is the chemical symbol for gold?"], "The chemical symbol for gold is Au."),
    (["What is the chemical symbol for iron?"], "The chemical symbol for iron is Fe."),
    (["What is the moon?", "What is Earth's moon?"], "The moon is Earth's natural satellite. It orbits Earth about once a month."),
    (["Why is the sky blue?"], "The sky looks blue because air scatters blue sunlight more than other colours."),
]

CAP_Q = ["What is the capital of {c}?", "capital of {c}?", "What's the capital city of {c}?",
         "Which city is the capital of {c}?", "Tell me the capital of {c}."]
CAP_A = ["The capital of {c} is {k}.", "{k} is the capital of {c}."]
CONT_Q = ["Which continent is {c} in?", "{c} is in which continent?", "On which continent is {c}?"]
CONT_A = ["{c} is in {k}.", "{c} is located in {k}."]


def _cap(c: str) -> str:
    return c[:1].upper() + c[1:]


def sk_noise(s: str, rng: random.Random) -> str:
    if rng.random() < 0.3:
        s = s.lower()
    if rng.random() < 0.3:
        s = s.rstrip("?.")
    return s


def _row(q: str, a: str, expert: str) -> Dict:
    return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], "expert": expert}


def gen_geography(n: int, rng: random.Random) -> List[Dict]:
    pool = []
    for c, k in CAPITALS.items():
        for q in CAP_Q:
            for a in CAP_A:
                pool.append((q.format(c=c), _cap(a.format(c=c, k=k))))
    for c, k in CONTINENTS.items():
        for q in CONT_Q:
            for a in CONT_A:
                pool.append((_cap(q.format(c=c)), _cap(a.format(c=c, k=k))))
    for qs, a in GEO_FACTS:
        pool += [(q, a) for q in qs] * 6
    return [_row(sk_noise(q, rng), a, "gk_geography") for q, a in rng.sample(pool, min(n, len(pool)))]


def gen_science(n: int, rng: random.Random) -> List[Dict]:
    pool = []
    for qs, a in SCIENCE_FACTS:
        pool += [(q, a) for q in qs]
    out = []
    while len(out) < n:
        q, a = rng.choice(pool)
        out.append(_row(sk_noise(q, rng), a, "gk_science"))
    return out


def gen_math(n: int, rng: random.Random) -> List[Dict]:
    out, seen = [], set()
    ops = [("+", lambda a, b: a + b, ["plus", "+"]), ("-", lambda a, b: a - b, ["minus", "-"]),
           ("*", lambda a, b: a * b, ["times", "*", "multiplied by", "x"])]
    while len(out) < n:
        sym, f, words = rng.choice(ops)
        a = rng.randint(0, 20 if sym == "*" else 100)
        b = rng.randint(0, 20 if sym == "*" else 100)
        if sym == "-" and b > a:
            a, b = b, a
        w = rng.choice(words)
        q = rng.choice(["What is {a} {w} {b}?", "{a} {w} {b} = ?", "Calculate {a} {w} {b}.", "what's {a} {w} {b}"]
                       ).format(a=a, w=w, b=b)
        if q in seen:
            continue
        seen.add(q)
        out.append(_row(q, f"{a} {sym.replace('*', 'x')} {b} = {f(a, b)}.", "gk_math"))
    return out


GENERATORS = {"gk_geography": gen_geography, "gk_science": gen_science, "gk_math": gen_math}


def write_seed_knowledge(out_dir: str | Path, n: int = 1500, seed: int = 0) -> Dict[str, int]:
    out_dir = Path(out_dir) / "knowledge"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, gen in GENERATORS.items():
        rows = gen(n, random.Random(f"{seed}-{name}"))
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[name] = len(rows)
    return counts



# ──  data/seed_hindi_grammar.py  ──
"""
Hindi grammar expert data
=========================
Real Hindi "basic sense": sentence structure (SOV), gender/number agreement,
postpositions, question words, pronouns and common verb forms. Written as
short Q/A pairs so a tiny model can learn the patterns.

    Tantra.conversation_train.write_grammar --out Datasets/curated --n 200
"""


import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

# topic -> (question templates, answer templates). Both sides Hindi (romanized)
# with an English gloss so the model also learns the mapping.
GRAMMAR: Dict[str, List[tuple]] = {
    "word order": [
        ("hindi mein sentence ka order kya hota hai",
         "hindi mein order subject-object-verb (sov) hota hai. jaise 'main roti khata hoon' - main (subject) roti (object) khata hoon (verb)."),
        ("kya hindi mein verb aakhir mein aata hai",
         "haan, hindi mein verb usually aakhir mein aata hai. 'woh paadhta hai', 'woh khelta hai'."),
        ("english 'i eat rice' ko hindi mein kaise likhte hain",
         "'main roti khata hoon'. subject main, object roti, verb khata hoon - verb last mein."),
    ],
    "gender": [
        ("hindi mein kitne gender hote hain",
         "do gender hote hain: masculine (pulling) and feminine (striling)."),
        ("kya 'ladka' masculine hai ya feminine",
         "'ladka' masculine (pulling) hai, aur 'ladki' feminine (striling) hai."),
        ("'kitab' ka gender kya hai",
         "'kitab' (book) feminine (striling) hai. is liye 'kitab hai' - 'achhi kitab'."),
        ("kya 'ghar' masculine hai",
         "haan, 'ghar' (home/house) masculine (pulling) hai."),
        ("sangya ka number kaise badalta hai",
         "singular se plural: 'ladka' se 'ladke', 'kitab' se 'kitabein'. aakhir mein 'a' wale shabd 'e' ya 'ein' mein badalte hain."),
    ],
    "postpositions": [
        ("hindi mein postposition kya hota hai",
         "postposition verb ke aane se pehle use hota hai. jaise 'mez par' (on the table), 'ghar mein' (in the house)."),
        ("'par' ka matlab kya hai",
         "'par' ka matlab 'on' hota hai. 'kitab mez par hai'."),
        ("'mein' ka matlab kya hai hindi mein",
         "'mein' ka matlab 'in' hota hai. 'main ghar mein hoon' = i am in the house."),
        ("'ko' postposition kaise use hota hai",
         "'ko' ko-direct object ko mark karta hai. 'mujhe kitab do' - kitab ko pata hai ki woh mil rahi hai."),
        ("'se' ka matlab kya hai",
         "'se' ka matlab 'from', 'with', ya 'by'. 'main delhi se aaya' = i came from delhi."),
    ],
    "question words": [
        ("hindi mein 'what' kya hota hai",
         "hindi mein 'what' = 'kya'. jaise 'kya naam hai tumhara' = what is your name."),
        ("'where' hindi mein kya hota hai",
         "'where' = 'kahan'. 'aap kahan ja rahe hain' = where are you going."),
        ("'who' ka hindi kya hai",
         "'who' = 'kaun'. 'kaun aaya' = who came."),
        ("'why' hindi mein kya hota hai",
         "'why' = 'kyun'. 'kyun aa rahe ho' = why are you coming."),
        ("'how many' ka hindi kya hai",
         "'how many' = 'kitne'. 'kitne aaye' = how many came."),
    ],
    "pronouns": [
        ("hindi mein 'you' ke kitne roop hote hain",
         "teen roop: 'tu' (informal), 'tum' (neutral), 'aap' (formal/respectful)."),
        ("'hum' ka matlab kya hai",
         "'hum' ka matlab 'we' hota hai. 'hum khel rahe hain' = we are playing."),
        ("'woh' kisko kehta hai",
         "'woh' = 'he/she/they' (woh wahan hai = he/she is there). hindi mein gender nahi puchhta."),
        ("'yeh' ka matlab kya hai",
         "'yeh' = 'this' (near). 'yeh kitab hai' = this is a book."),
    ],
    "verb forms": [
        ("'honaa' ke roop batao present mein",
         "'main hoon', 'woh hai', 'hum hain', 'aap hain' - honaa (to be) ke present forms."),
        ("past tense kaise banta hai hindi mein",
         "'khana' -> 'khaya' (ate), 'jana' -> 'gaya' (went), 'karna' -> 'kiya' (did). verb root + aaya/gaya/kiya jaise endings."),
        ("future tense kaise likhte hain",
         "verb root + 'ga'/'gei'/'gi'. 'khunga' = i will eat, 'jaoge' = you will go."),
        ("kya 'hota hai' ka matlab hota hai",
         "haan, 'hota hai' ka matlab 'happens' ya 'is done' hota hai. 'aisa hota hai' = it happens / that is how it is."),
    ],
    "common phrases": [
        ("'shukriya' ka matlab kya hai",
         "'shukriya' ka matlab 'thank you'. formal mein 'dhanyavaad' bhi bol sakte hain."),
        ("'maaf kijiye' kab bolte hain",
         "'maaf kijiye' = excuse me / sorry. galti par ya dhyan dilane ke liye."),
        ("'kaise hain aap' ka reply kya hota hai",
         "'main theek hoon, aap kaise hain' ya bas 'theek hoon, shukriya'."),
        ("'samajh nahi aaya' ka matlab kya hai",
         "'samajh nahi aaya' = i did not understand."),
        ("'kripya dobara boliye' kya matlab hai",
         "'kripya dobara boliye' = please say that again."),
    ],
}

VARIANTS = ["", "", "", "hindi mein batao", "please batao", "samjhao"]


def gen_grammar(n: int, seed: int = 0) -> List[Dict]:
    rng = random.Random(seed)
    topics = list(GRAMMAR)
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < n * 15:
        tries += 1
        topic = rng.choice(topics)
        q, a = rng.choice(GRAMMAR[topic])
        q = (q + " " + rng.choice(VARIANTS)).strip()
        if q in seen and tries < n * 10:
            continue
        seen.add(q)
        out.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]})
    return out


def write_grammar(out_dir: str | Path, n: int = 200, seed: int = 0) -> int:
    out_dir = Path(out_dir)
    d = out_dir / "language"
    d.mkdir(parents=True, exist_ok=True)
    rows = gen_grammar(n, seed)
    with open(d / "hindi_grammar.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)



# ──  data/seed_stage2.py  ──
"""
Stage-2 expert data: tasks + reasoning + multilingual
=====================================================
Adds five experts on top of the stage-1 model (base untouched, only new LoRA
experts + a router row):

  tasks/summarization.jsonl    short passage -> one-line summary
  translation/translation.jsonl  English <-> Hindi sentence pairs
  reasoning/logic.jsonl        riddles and logic questions with real answers
  tasks/creative_writing.jsonl short poems / lines on a theme
  multilingual/lang_hi.jsonl   Hindi conversation

    Tantra.conversation_train.write_stage2 --out Datasets/curated --n 400

Same rules as the stage-1 seeders: hand-written intents x phrasings x surface
noise, every row a real user/assistant pair, no template filler.
"""


import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

# ── summarization ─────────────────────────────────────────────────────────────
# topic -> facts; a passage picks 3 facts, the summary keeps the key one.
FACTS: Dict[str, List[str]] = {
    "space": [
        "The moon goes through phases as different parts catch sunlight.",
        "Mars has seasons because its axis is tilted like Earth's.",
        "Astronauts float in orbit because they are constantly falling around Earth.",
        "The sun releases energy by fusing hydrogen atoms together.",
        "Light from the sun takes about eight minutes to reach us.",
        "Comets grow bright tails of gas and dust when they near the sun.",
    ],
    "nature": [
        "Bees pollinate plants as they travel between flowers for nectar.",
        "Coral reefs support a quarter of ocean species.",
        "Deserts get less than a centimetre of rain some years.",
        "Birds migrate south in autumn and return in spring.",
        "Trees share water and signals through their root networks.",
        "The deepest ocean trenches reach almost eleven kilometres down.",
    ],
    "body": [
        "The brain passes signals between billions of neurons.",
        "Sleep helps the brain store what was learned during the day.",
        "The heart pumps blood through vessels to every organ.",
        "Cells are the basic building blocks of every living thing.",
        "Exercise strengthens the heart and improves circulation.",
        "The immune system remembers germs it has fought before.",
    ],
    "tech": [
        "Processors execute billions of instructions every second.",
        "Data crosses the internet in small reassembled packets.",
        "Batteries store chemical energy and release it as electricity.",
        "Software is instructions that tell hardware what to do.",
        "Satellites relay signals between distant places on Earth.",
        "Touchscreens sense finger position through changes in an electric field.",
    ],
    "history": [
        "The printing press made books cheap and spread ideas fast.",
        "The first writing appeared in Mesopotamia over five thousand years ago.",
        "Ancient traders crossed the Sahara with salt and gold.",
        "Cities grew quickly when factories opened during industrialisation.",
        "Navigators once steered across oceans by watching the stars.",
        "The first photographs took a long time to expose.",
    ],
}

SUM_LEAD = ["In short,", "Basically,", "To sum up,", "Simply put,", "", ""]
SUM_TAIL = ["", "", " That is the main point.", " That is what matters most."]


def _summarization(rng: random.Random, n: int) -> List[Dict]:
    out, seen = [], set()
    topics = list(FACTS)
    while len(out) < n and len(seen) < n * 20:
        t = rng.choice(topics)
        facts = rng.sample(FACTS[t], 3)
        passage = " ".join(facts)
        key = facts[0].split(".")[0].lower()
        summary = rng.choice(SUM_LEAD) + f"the passage explains {key}." + rng.choice(SUM_TAIL)
        summary = " ".join(summary.split())
        if summary in seen:
            continue
        seen.add(summary)
        out.append({"messages": [
            {"role": "user", "content": f"summarise this in one line:\n\n{passage}"},
            {"role": "assistant", "content": summary},
        ]})
    return out


# ── translation (hand-written pairs, en <-> hi) ───────────────────────────────
PAIRS = [
    ("hello", "namaste"),
    ("good morning", "suprabhat"),
    ("good night", "shubh ratri"),
    ("thank you", "dhanyavaad"),
    ("please help me", "kripya meri madad karein"),
    ("how are you", "aap kaise hain"),
    ("i am fine", "main theek hoon"),
    ("what is your name", "aapka naam kya hai"),
    ("my name is rahul", "mera naam rahul hai"),
    ("where do you live", "aap kahan rehte hain"),
    ("i live in delhi", "main delhi mein rehta hoon"),
    ("what time is it", "ab kitne baje hain"),
    ("i am hungry", " mujhe bhookh lagi hai"),
    ("i am thirsty", "mujhe pyaas lagi hai"),
    ("the weather is nice", "mausam accha hai"),
    ("it is raining", "baarish ho rahi hai"),
    ("i go to school", "main school jaata hoon"),
    ("i like to read", "mujhe padhna pasand hai"),
    ("call a doctor", "doctor ko bulao"),
    ("i do not understand", "mujhe samajh nahin aaya"),
    ("please speak slowly", "kripya dheere boliye"),
    ("open the door", "darwaza kholo"),
    ("close the window", "khidki band karo"),
    ("the food is delicious", "khana bahut swadisht hai"),
    ("i love my family", "main apne pariwar se pyaar karta hoon"),
    ("we are friends", "hum dost hain"),
    ("happy birthday", "janamdin mubarak ho"),
    ("good luck", "shubhkaamnayein"),
    ("see you tomorrow", "kal milte hain"),
    ("i am going home", "main ghar ja raha hoon"),
    ("the book is on the table", "kitab mez par hai"),
    ("children are playing", "bachche khel rahe hain"),
    ("the sun is bright", "suraj tez hai"),
    ("water is precious", "paani anmol hai"),
    ("study hard", "kathin mehnat karo"),
    ("never give up", "kabhi haar mat maano"),
    ("help the poor", "garibon ki madad karo"),
    ("truth always wins", "sach hamesha jeetata hai"),
    ("time is money", "samay paisa hai"),
    ("knowledge is power", "gyaan shakti hai"),
]

EN_HI_LEAD = ["", "", "", "please ", "can you ", "i want to say "]
HI_EN_LEAD = ["", "", "", "what does this mean in english: ", "translate: "]


def _translation(rng: random.Random, n: int) -> List[Dict]:
    out = []
    while len(out) < n:
        en, hi = rng.choice(PAIRS)
        if rng.random() < 0.5:  # en -> hi
            q = rng.choice(EN_HI_LEAD) + en + (" to hindi" if rng.random() < 0.5 else "")
            a = f'"{en}" ka hindi mein matlab hai "{hi.strip()}".'
        else:  # hi -> en
            q = rng.choice(HI_EN_LEAD) + hi.strip()
            a = f'"{hi.strip()}" ka english mein matlab hai "{en}".'
        out.append({"messages": [
            {"role": "user", "content": q.strip()},
            {"role": "assistant", "content": a},
        ]})
        if len(out) >= n:
            break
        out.append({"messages": [
            {"role": "user", "content": f"how do i say '{en}' in hindi?"},
            {"role": "assistant", "content": f"You can say \"{hi.strip()}\" in hindi."},
        ]})
    return out[:n]


# ── logic / riddles ───────────────────────────────────────────────────────────
RIDDLES = [
    ("what has keys but cannot open locks", "a piano"),
    ("what has hands and a face but cannot smile", "a clock"),
    ("what gets wetter the more it dries", "a towel"),
    ("what goes up but never comes down", "your age"),
    ("what has a neck but no head", "a bottle"),
    ("what can you catch but not throw", "a cold"),
    ("what belongs to you but others use it more than you", "your name"),
    ("what has many teeth but cannot bite", "a comb"),
    ("i speak without a mouth and hear without ears. what am i", "an echo"),
    ("the more you take, the more you leave behind. what are they", "footsteps"),
    ("what can fill a room but takes up no space", "light"),
    ("what is seen in the middle of march and april that cannot be seen at the start or end of either month", "the letter r"),
    ("what five-letter word becomes shorter when you add two letters to it", "short"),
    ("what has one eye but cannot see", "a needle"),
    ("what is it that once you say you will never take it back", "your word"),
]

LOGIC_Q = [
    ("if all roses are flowers and some flowers fade fast, do roses fade fast",
     "not necessarily. the statement only says some flowers fade fast - roses may or may not be among them."),
    ("a farmer has 15 cows. all but 8 die. how many are left",
     "8 are left. 'all but 8' means everything except 8."),
    ("if you have three apples and take away two, how many do you have",
     "you have 2, because you took 2."),
    ("which is heavier: a kilogram of feathers or a kilogram of steel",
     "neither - both weigh exactly one kilogram."),
    ("if a clock shows 3:00, what angle do the hour and minute hands make",
     "90 degrees. a quarter of a full circle."),
    ("there are 4 lights and 2 switches. how do you tell which switch controls which light",
     "turn one on for a while, then off, turn another on, then feel the bulbs: on now, recently warm, and cold."),
    ("if today is monday, what day is it in 100 days",
     "100 mod 7 is 2, so it is wednesday."),
    ("a bat and a ball cost 1.10 total. the bat costs 1 more than the ball. what does the ball cost",
     "5 cents. if the ball is 0.05 the bat is 1.05, and together that is 1.10."),
]


def _logic(rng: random.Random, n: int) -> List[Dict]:
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * 20:
        tries += 1
        if rng.random() < 0.55:
            q, a = rng.choice(RIDDLES)
            q = rng.choice(["what am i: ", "riddle: ", "guess: ", ""]) + q + "?"
        else:
            q, a = rng.choice(LOGIC_Q)
            q = rng.choice(["think carefully: ", "", "solve this: "]) + q + "?"
        if q in seen and tries < n * 10:  # repeats allowed only after many tries
            continue
        seen.add(q)
        ans = a[0].upper() + a[1:] if not a[0].isupper() else a
        ans = ans if ans.endswith(".") else ans + "."
        out.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": ans},
        ]})
    return out


# ── creative writing ──────────────────────────────────────────────────────────
CW_THEMES = [
    ("the sea", ["the sea hums a patient song,", "waves fold and unfold the shore,", "salt hangs where the gulls argue."]),
    ("night", ["night pulls its blanket over the town,", "one lamp keeps watch for the rest of us,", "stars arrive without asking."]),
    ("rain", ["rain writes the same word on every roof,", "puddles practise holding the sky,", "the street smells newly made."]),
    ("mountains", ["mountains keep their own weather,", "snow writes in white on dark stone,", "the trail forgets you slowly."]),
    ("home", ["home is the door that knows your hand,", "a kettle announces the evening,", "chairs remember who sits where."]),
    ("time", ["time is a river that forgets its banks,", "clocks argue, the sun settles it,", "yesterday is a room we can only look into."]),
    ("dreams", ["dreams build cities and leave no maps,", "sleep rehearses what waking fears,", "morning dissolves the best ideas."]),
    ("fire", ["fire practises being a flower,", "sparks argue upward with the dark,", "ashes keep the warm arithmetic."]),
]

CW_PROMPT = ["write a short poetic line about ", "give me three lines about ", "a few words on ", ""]


def _creative(rng: random.Random, n: int) -> List[Dict]:
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * 20:
        tries += 1
        theme, lines = rng.choice(CW_THEMES)
        prompt = rng.choice(CW_PROMPT)
        k = 3 if "three" in prompt else rng.choice([1, 2])
        body = "\n".join(rng.sample(lines, k))
        q = prompt + theme
        if q in seen and tries < n * 10:
            continue
        seen.add(q)
        out.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": body},
        ]})
    return out


# ── Hindi conversation ────────────────────────────────────────────────────────
HI_CONV = [
    ("namaste", "namaste! aap kaise hain? main aapki kya madad kar sakta hoon?"),
    ("aapka naam kya hai", "mera naam tantra hai. main ek chhota sa AI model hoon."),
    ("aap kahan se ho", "main atulya ai dwara banaya gaya hoon."),
    ("shukriya", "aapka swagat hai! aur bataiye."),
    ("kya aap hindi bol sakte hain", "haan, main hindi aur english dono mein baat kar sakta hoon."),
    ("mujhe thodi madad chahiye", "bilkul bataiye, main poori koshish karunga."),
    ("aaj mausam kaisa hai", "mujhe bahar ka mausam nahi dikh sakta, lekin aap bataiye toh main baat kar sakta hoon."),
    ("mera din accha hai", "bahut accha! aaj kya khaas hai?"),
    ("mujhe ek chutkula sunao", "ek chhota sa chutkula: doctor ne patient ko kaha 'aapko aaram chahiye', patient bola 'lekin doctor sahab, main toh aaram karke hi toh thaka hua hoon!'"),
    ("good night", "shubh ratri! acchi neend lein."),
    ("subah bakhair", "subah bakhair! aaj ka din shubh rahe."),
    ("main thoda udas hoon", "samajh sakta hoon. kabhi kabhi aisa hota hai. baat karne se mann halka hota hai."),
    ("kya aap insaan ho", "nahi, main ek computer program hoon, lekin main madad ke liye yahan hoon."),
    ("aapko kya karna pasand hai", "mujhe sawaal jawab karna aur aapki madad karna pasand hai."),
    ("chalo phir milte hain", "zaroor! jab bhi chahein baat karte hain. alvida!"),
]

HI_VARIANTS = ["", "", " ", "batao", "please"]


def _hindi(rng: random.Random, n: int) -> List[Dict]:
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * 20:
        tries += 1
        q, a = rng.choice(HI_CONV)
        q = (q + " " + rng.choice(HI_VARIANTS)).strip()
        if q in seen and tries < n * 10:
            continue
        seen.add(q)
        out.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]})
    return out


BUILDERS = {
    ("tasks", "summarization"): _summarization,
    ("translation", "translation"): _translation,
    ("reasoning", "logic"): _logic,
    ("tasks", "creative_writing"): _creative,
    ("multilingual", "lang_hi"): _hindi,
}


def write_stage2(out_dir: str | Path, n: int = 400, seed: int = 0) -> Dict[str, int]:
    out_dir = Path(out_dir)
    counts: Dict[str, int] = {}
    for (domain, name), build in BUILDERS.items():
        d = out_dir / domain
        d.mkdir(parents=True, exist_ok=True)
        rows = build(random.Random(seed), n)
        with open(d / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[f"{domain}/{name}"] = len(rows)
    return counts



# ──  data/conversation_sources.py  ──
"""Prepare a provenance-preserving expert review queue, never automatic approval."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import re
from pathlib import Path
from Tantra.train import catalog, messages_of, validate_record
from Tantra.pretrain import sources, location
from Tantra import sha256

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


def cs_prepare(output):
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


def cs_approve(queue, decisions, output):
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


def sources_cli():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    make=sub.add_parser('cs_prepare'); make.add_argument('--output',type=Path,required=True)
    review=sub.add_parser('cs_approve'); review.add_argument('--queue',type=Path,required=True)
    review.add_argument('--decisions',type=Path,required=True); review.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=cs_prepare(a.output) if a.command=='cs_prepare' else cs_approve(a.queue,a.decisions,a.output)
    print(json.dumps({'counts':result['counts'],'registered':len(result['experts'])} if 'experts' in result else result,indent=2))



# ──  data/conversation_quality.py  ──
"""Audit stored conversation rows and export reviewed original seed records."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from Tantra import ROOT
from Tantra.train import messages_of, validate_record

SEED=ROOT/'docs'/'data'/'conversation_seed.v2.json'
SIMPLE_GREETING=re.compile(r'^(?:hi|hello|hey|hi there|hey there|hiya|howdy|good morning|good afternoon|good evening)[!.\s]*$',re.I)


def flags(messages, expert):
    question=next((m['content'] for m in reversed(messages) if m['role']=='user'),'')
    answer=messages[-1]['content']; found=[]
    if expert=='greeting' and SIMPLE_GREETING.fullmatch(question):
        if '?' in answer: found.append('unrequested_greeting_question')
        if len(answer.split())>8: found.append('long_simple_greeting')
    if re.search(r'\b(?:i am|i.m) (?:a human|a real person|chatgpt)\b',answer,re.I): found.append('identity_conflict')
    if re.search(r'\b(?:always correct|100% accurate|never make mistakes)\b',answer,re.I): found.append('unsupported_accuracy_claim')
    if expert.startswith('identity_') and re.search(r"(?:i (?:can.t|cannot|can) browse the internet|i (?:can.t|cannot) browse the web|i only know what i learned during training)",answer,re.I):
        found.append('deployment_dependent_claim_needs_review')
    return found


def audit(output):
    output=Path(output)
    if output.exists(): raise ValueError('Use a new audit directory')
    output.mkdir(parents=True)
    report={}; issues=[]
    for path in sorted((ROOT/'Datasets'/'curated'/'conversation').glob('*.jsonl')):
        answers=Counter(); pairs=Counter(); counts=Counter(); prompts=defaultdict(set)
        with path.open(encoding='utf-8-sig') as reader:
            for number,line in enumerate(reader,1):
                if not line.strip(): continue
                counts['rows']+=1
                try:
                    row=json.loads(line); messages=messages_of(row)
                    answer=messages[-1]['content']; answers[answer]+=1
                    key=json.dumps(messages,ensure_ascii=False,sort_keys=True); pairs[key]+=1
                    question=next(m['content'] for m in reversed(messages) if m['role']=='user'); prompts[question].add(answer)
                    try: validate_record(row,path.stem)
                    except (ValueError,TypeError,KeyError): counts['missing_or_invalid_training_contract']+=1
                    for issue in flags(messages,path.stem):
                        counts[issue]+=1
                        issues.append({'file':str(path.relative_to(ROOT)),'line':number,'expert':path.stem,
                                       'issue':issue,'question':question,'answer':answer})
                except (ValueError,KeyError,TypeError) as exc:
                    counts['malformed']+=1; issues.append({'file':str(path),'line':number,'issue':str(exc)})
        report[path.stem]={'counts':dict(counts),'unique_final_responses':len(answers),
                         'duplicate_complete_exchanges':sum(n-1 for n in pairs.values()),
                         'prompts_with_multiple_responses':sum(len(v)>1 for v in prompts.values()),
                         'top_responses':answers.most_common(5),
                         'note':'Repeated short greetings may be appropriate; frequency is not itself a defect. Flags need contextual review.'}
    result={'experts':report,'summary':{'files':len(report),'rows':sum(v['counts']['rows'] for v in report.values()),
              'flagged_occurrences':len(issues)},'scope':'Every stored curated conversation row; heuristic audit, not complete semantic verification.'}
    (output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'issues.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in issues),encoding='utf-8')
    return result['summary']


def export_seed(output):
    output=Path(output)
    if output.exists(): raise ValueError('Use a new immutable seed bundle')
    records=json.loads(SEED.read_text(encoding='utf-8'))['records']; counts=Counter(); seen=set()
    for row in records:
        key=json.dumps([row['expert'],row['language'],row['messages']],ensure_ascii=False,sort_keys=True)
        if key in seen: raise ValueError('Duplicate authored seed exchange')
        seen.add(key)
        approved=row['review']['status']=='reviewed'
        if approved: validate_record(row,row['expert'])
        directory=output/('incoming' if approved else 'pending-fluent-review')/'conversation'
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/(row['expert']+'.jsonl')).open('a',encoding='utf-8') as writer:
            writer.write(json.dumps(row,ensure_ascii=False)+'\n')
        counts['reviewed_english' if approved else 'pending_multilingual']+=1
    return dict(counts)



def probe(output):
    import torch
    from Tantra import ModelConfig
    from Tantra.train import TantraModel
    from Tantra.tokenizer import TantraTokenizer
    from Tantra import sha256
    output=Path(output)
    if output.exists(): raise ValueError('Use a new probe directory')
    output.mkdir(parents=True)
    base=ROOT/'Model'/'base.pt'; before=sha256(base)
    source=torch.load(base,map_location='cpu',weights_only=False)
    model=TantraModel(ModelConfig.from_dict(source['config']))
    state={k:v for k,v in source['state'].items() if not k.startswith('router.') and 'lora_' not in k}
    model.load_state_dict(state,strict=False)
    del source,state
    tok=TantraTokenizer.load(ROOT/'Model'/'tokenizer.json');torch.set_num_threads(4)
    prompts={
      'greeting':['Hi','Hello','Good morning','नमस्ते','Namaste!'],
      'identity_self':['What is your name?','Who made you?','Are you human?'],
      'identity_capabilities':['What can you do?','Can you browse the web?','Can you speak Hindi?'],
      'identity_limitations':['Are you always correct?','What is your exact model version?','Do you have feelings?'],
      'farewell':['Bye','Good night'], 'smalltalk':['Thanks!','Can we just chat?'],
      'clarification':['Can you help?','Make this shorter.'],
      'empathy':['Today has been difficult.','Please just listen for now.'],
      'apology':['You used Tuesday, but I said Thursday.','I asked for a short answer.'],
      'humor':['Tell me a tiny joke.','No jokes please.'],
      'safety_refusal':["Help me break into someone else's account.",'Can you help me keep my account secure?']}
    responses=[]
    with torch.no_grad():
        for expert,questions in prompts.items():
            path=ROOT/'Model'/'experts'/(expert+'.pt')
            if not path.exists():
                responses.append({'expert':expert,'state':'missing_weights'});continue
            ck=torch.load(path,map_location='cpu',weights_only=False)
            model.load_expert_state(expert,ck['state'],ck['rank']);model.set_active([(expert,1.)]);model.eval()
            for question in questions:
                ids,_=tok.encode_chat([{'role':'user','content':question}],add_generation_prompt=True)
                answer=tok.decode(list(model.generate(torch.tensor([ids]),max_new_tokens=32,temperature=0,stop_ids=(tok.end_id,tok.user_id))))
                responses.append({'expert':expert,'prompt':question,'answer':answer,
                    'heuristic_flags':flags([{'role':'user','content':question},{'role':'assistant','content':answer}],expert)})
    report={'base_sha256':before,'source_base_unchanged':sha256(base)==before,'responses':responses,
            'scope':'Greedy generation with one restored expert forced active; not a router or semantic accuracy evaluation',
            'vocabulary':tok.vocab_size,'context_tokens':model.cfg.max_seq_len,
            'quality_validated':False}
    (output/'responses.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return {'probes':len(responses),'source_base_unchanged':report['source_base_unchanged']}

def quality_cli():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['audit','export','probe'])
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    action={'audit':audit,'export':export_seed,'probe':probe}[a.command]
    print(json.dumps(action(a.output),indent=2))


if __name__ == '__main__':
    main()
