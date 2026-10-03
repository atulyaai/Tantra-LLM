"""
Tantra/rag_data.py — Teach the model to answer from looked-up facts (python main.py --mode rag).

A 100M model cannot memorise all general knowledge, but it can learn to read. The WebUI already
looks questions up in Smriti and puts the facts in the system message; until now the model was
never trained on that layout, so it mostly ignored them. This writes Datasets/sft_rag.jsonl:

  * 85%  a stored question + the facts Smriti finds for it (the right fact mixed with a
         distractor, in random order) -> the stored answer. Teaches: find the fact, use it.
  * 15%  only distractor facts -> the stored answer. Teaches: unrelated facts are not the answer.

The system message is built by Smriti.knowledge_note(), the same function the WebUI uses.
Only factual questions (who / what / when / कौन / क्या है / कब ...) with short answers are used:
facts do not help chit-chat or creative writing. Probe questions, refusals and broken text are left out.
"""
from __future__ import annotations

import json
import os
import random
import re
from typing import Dict, List, Optional

from Tantra.data_prep import _Buckets, bad_text, foreign_assistant, norm
from Tantra.smriti import Smriti, _unpack
from Tantra.utils import get_logger

log = get_logger("tantra.rag")

# Fact questions only: looked-up facts help "who/what/when/where", not chit-chat, poems or puzzles.
FACTUAL = re.compile(
    r"^(who|what|when|where|which|how many|how much|in which|name the)\b|"
    r"(कौन|क्या है|क्या हैं|क्या था|कब|कहाँ|कहां|किस|कितन|किसे कहते|किसको कहते|का नाम बताइए|का नाम बताओ)", re.I)
CREATIVE = re.compile(r"(कविता|कहानी|पहेली|लिखें|लिखिए|बनाएं|बनाइए|poem|story|write|generate|create|puzzle)", re.I)


def _probe_questions(data_dir: str) -> set:
    path = os.path.join(data_dir, "probe_50.jsonl")
    if not os.path.isfile(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {norm(json.loads(l)["question"]) for l in f if l.strip()}


def example(store: Smriti, gold: Dict, rng: random.Random, distractor_only: bool) -> Optional[dict]:
    """One training conversation for a stored question/answer, or None when nothing useful is found."""
    q, a = gold["question"], gold["text"]
    others = [h for h in store.search(q, k=4) if h["id"] != gold["id"] and h["text"].strip() != a.strip()]
    if distractor_only:
        hits = others[:2]
        if not hits:
            return None
    else:
        hits = others[:1] + [gold]
        rng.shuffle(hits)
    return {"messages": [{"role": "system", "content": Smriti.knowledge_note(hits)},
                         {"role": "user", "content": q}, {"role": "assistant", "content": a}]}


_store: Optional[Smriti] = None


def _worker_init(path: str) -> None:
    global _store
    _store = Smriti(path, readonly=True)


def _worker(job: tuple) -> Optional[dict]:
    rid, q, a, distractor_only, seed = job
    return example(_store, {"id": rid, "question": q, "text": a}, random.Random(seed), distractor_only)


def build(data_dir: str, smriti_path: str, n: int = 20000, seed: int = 0, workers: int = 0) -> Dict:
    """Each Smriti search takes ~1-4 s on a multi-GB store, so searches run in parallel processes."""
    import multiprocessing as mp
    if not os.path.isfile(smriti_path):
        raise FileNotFoundError(f"{smriti_path} not found — run  python main.py --mode smriti  first")
    rng = random.Random(seed)
    store = Smriti(smriti_path, readonly=True)
    probe = _probe_questions(data_dir)
    stats = {"with_fact": 0, "distractors_only": 0, "skipped": 0}
    # random stored Q&A rows (ids first: sorting multi-GB blobs at random would be slow)
    ids = [r[0] for r in store.db.execute("SELECT id FROM docs WHERE kind='qa'")]
    rng.shuffle(ids)
    jobs = []
    for rid in ids:
        if len(jobs) >= int(n * 1.15) + 10:          # a few searches find nothing useful
            break
        d = _unpack(store.db.execute("SELECT blob FROM docs WHERE id = ?", (rid,)).fetchone()[0])
        q, a = norm(d["head"]), norm(d["body"])
        if (not q or q in probe or not FACTUAL.search(q) or CREATIVE.search(q) or len(a) < 2 or bad_text(a)
                or foreign_assistant(a) or len(q) > 200 or len(a) > 600):
            stats["skipped"] += 1
            continue
        jobs.append((rid, q, a, rng.random() < 0.15, rng.randrange(1 << 30)))
    store.close()
    out = _Buckets(os.path.join(data_dir, "sft_rag.jsonl"), n=8, seed=seed)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    log.info(f"Looking up {len(jobs):,} questions with {workers} processes ...")
    with mp.get_context("spawn").Pool(workers, initializer=_worker_init, initargs=(smriti_path,)) as pool:
        for job, rec in zip(jobs, pool.imap(_worker, jobs, chunksize=4)):
            if stats["with_fact"] + stats["distractors_only"] >= n:
                pool.terminate()
                break
            if rec is None:
                stats["skipped"] += 1
                continue
            out.add(rec)
            stats["distractors_only" if job[3] else "with_fact"] += 1
            done = stats["with_fact"] + stats["distractors_only"]
            if done % 1000 == 0:
                log.info(f"  {done:,}/{n:,}")
    out.close()
    with open(os.path.join(data_dir, "rag_report.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    log.info(f"sft_rag.jsonl: {stats}")
    return stats
