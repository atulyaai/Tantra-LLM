"""
Tantra/distill.py — A big open model writes GK questions and answers for Tantra (python main.py --mode distill).

Runs on a GPU (Kaggle 2x T4 / Colab). Takes Hindi and English Wikipedia passages, asks a teacher
model (default Qwen2.5-3B-Instruct; --teacher Qwen/Qwen2.5-7B-Instruct is better at Hindi but ~2x
slower) for question/answer pairs that the passage supports, and writes them as SFT conversations
to sft_distill.jsonl. This is how Phi and SmolLM made small models know much more than their size.

  * The rules in the teacher prompt keep out "As an AI…" text and "according to the passage".
  * Every pair is checked: same language as the passage, not a refusal, answer not too short/long,
    not a probe question.
  * Output is appended and the run resumes where it stopped, so it can span several sessions.
    On Kaggle it is written to /kaggle/working (download it, or add it to your dataset).
"""
from __future__ import annotations

import json
import os
import random
import re
from typing import Callable, Dict, Iterator, List, Optional

from Tantra.data_prep import _wiki_text, foreign_assistant, lang_of, norm
from Tantra.utils import get_logger

log = get_logger("tantra.distill")

PROMPT = {
    "hindi": ("नीचे दिए गए लेख को पढ़कर उससे {k} सामान्य-ज्ञान प्रश्न और उनके उत्तर हिंदी में लिखें।\n"
              "नियम: हर उत्तर पूरा वाक्य हो और लेख के तथ्यों पर आधारित हो। 'लेख के अनुसार' या 'इस पाठ में' न लिखें। "
              "प्रश्न ऐसे हों जो कोई भी बिना लेख देखे पूछ सके।\n"
              "केवल JSON लौटाएँ: [{{\"q\": \"प्रश्न\", \"a\": \"उत्तर\"}}, ...]\n\nलेख:\n{passage}"),
    "english": ("Read the article below and write {k} general-knowledge questions with answers in English.\n"
                "Rules: every answer is a complete sentence based on facts in the article. Never say "
                "'according to the passage' or 'the text'. Questions must make sense without seeing the article.\n"
                "Return only JSON: [{{\"q\": \"question\", \"a\": \"answer\"}}, ...]\n\nArticle:\n{passage}"),
}


def passages(rng: random.Random, hindi_share: float = 0.6) -> Iterator[str]:
    """Wikipedia passages, ~60% Hindi, streamed (no full download)."""
    from datasets import load_dataset
    hi = iter(load_dataset("wikimedia/wikipedia", "20231101.hi", split="train", streaming=True).shuffle(seed=rng.randrange(1 << 30)))
    en = iter(load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True).shuffle(seed=rng.randrange(1 << 30)))
    while True:
        row = next(hi if rng.random() < hindi_share else en)
        for piece in _wiki_text(row)[:1]:            # the lead of the article holds the key facts
            if len(piece) >= 400:
                yield piece[:2500]


def parse_pairs(reply: str, language: str, probe: set) -> List[Dict[str, str]]:
    """Teacher reply -> checked [{'q','a'}]; anything malformed is dropped."""
    m = re.search(r"\[.*\]", reply or "", re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except ValueError:
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        q, a = norm(str(it.get("q") or "")), norm(str(it.get("a") or ""))
        if not q or not a or len(a) < 8 or len(a) > 800 or q in probe:
            continue
        if foreign_assistant(a) or re.search(r"according to the (passage|article|text)|लेख के अनुसार|इस (लेख|पाठ) में", a, re.I):
            continue
        if lang_of(q + " " + a) != language:
            continue
        out.append({"q": q, "a": a})
    return out


class Teacher:
    """Batched generation with Hugging Face transformers (fp16, spread over all GPUs)."""

    def __init__(self, name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto").eval()
        self.torch = torch

    def __call__(self, prompts: List[str], max_new_tokens: int = 600) -> List[str]:
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                 for p in prompts]
        batch = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)


def build(data_dir: str, teacher: str = "Qwen/Qwen2.5-3B-Instruct", n: int = 20000, batch_size: int = 16,
          per_passage: int = 3, seed: Optional[int] = None,
          generate: Optional[Callable[[List[str]], List[str]]] = None,
          source: Optional[Iterator[str]] = None) -> Dict:
    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else data_dir
    out_path = os.path.join(out_dir, "sft_distill.jsonl")
    done = sum(1 for _ in open(out_path, encoding="utf-8")) if os.path.isfile(out_path) else 0
    rng = random.Random(seed if seed is not None else done)     # a resumed run reads new passages
    probe = set()
    pf = os.path.join(data_dir, "probe_50.jsonl")
    if os.path.isfile(pf):
        with open(pf, encoding="utf-8") as f:
            probe = {norm(json.loads(l)["question"]) for l in f if l.strip()}
    generate = generate or Teacher(teacher)
    source = source or passages(rng)
    stats = {"pairs": done, "passages": 0, "replies_without_pairs": 0}
    log.info(f"Distilling with {teacher} -> {out_path} ({done:,} pairs already there, target {n:,})")
    with open(out_path, "a", encoding="utf-8") as f:
        while stats["pairs"] < n:
            batch = []
            for p in source:
                lang = lang_of(p)
                if lang in PROMPT:
                    batch.append((p, lang))
                if len(batch) >= batch_size:
                    break
            if not batch:
                break
            replies = generate([PROMPT[lang].format(k=per_passage, passage=p) for p, lang in batch])
            for (p, lang), reply in zip(batch, replies):
                stats["passages"] += 1
                pairs = parse_pairs(reply, lang, probe)
                if not pairs:
                    stats["replies_without_pairs"] += 1
                for qa in pairs:
                    f.write(json.dumps({"messages": [{"role": "user", "content": qa["q"]},
                                                     {"role": "assistant", "content": qa["a"]}]}, ensure_ascii=False) + "\n")
                    stats["pairs"] += 1
            f.flush()
            log.info(f"  {stats['pairs']:,}/{n:,} pairs from {stats['passages']:,} passages")
    return stats
