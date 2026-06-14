"""NP-DNA v3 — curriculum by data volume, dynamic vocab, fresh start."""
import argparse, sys, os, json, time, random, math, gc
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from npdna import NpDnaCore
from npdna.config import CONFIGS

torch.set_num_threads(mp.cpu_count())

# Config
CONFIG_NAME = "seed"
USE_ATTENTION = True
BATCH_SIZE = 4
SEQ_LEN = 128
LR = 5e-3
WARMUP_STEPS = 200            # shorter warmup
LOG_EVERY = 25
SAVE_EVERY = 500
EVAL_EVERY = 250
LATEST_EVERY = 25
MTP_DEPTH = 3
MTP_WEIGHT = 0.25
DEFAULT_TARGET_STEPS = 100_000

CKPT_DIR = Path("model/npdna_v3")
ASSETS_DIR = Path("model/tokenizer")
SEED_CHAT_PATH = Path("data/seed_chat.jsonl")
DEFAULT_SEED_CHAT_RATIO = 0.35
DEFAULT_SEED_RATIO_MIN = 0.10
DEFAULT_SEED_RATIO_DECAY_STEPS = 30_000
DEFAULT_SYSTEM_PROMPT = "You are Atulya. Be warm, thoughtful, and direct."
IGNORE_INDEX = -100

# Auto-calculate steps per dataset based on MB size
DATASET_SIZES = {
    "samples": 0,
    "agentic": 151,
    "factual": 220,
    "code": 556,
    "reasoning": 559,
    "translation": 1814,
    "general": 7601,
    "math": 9127,
}

def calc_steps(mb, base=200, max_steps=2000):
    if mb == 0:
        return 500        # tiny samples get 500 steps to overfit
    return min(max_steps, base + int((mb / 100) * 50))

# Build curriculum weights from data volume, then scale to target steps.
BASE_CURRICULUM = []
cumul = 0
all_folders = ["samples","agentic","factual","code","reasoning",
               "translation","general","math"]
for i in range(len(all_folders)):
    folders = all_folders[:i+1]
    name = all_folders[i]
    mb = DATASET_SIZES[name]
    steps = calc_steps(mb, base=200, max_steps=1500)
    cumul += steps
    BASE_CURRICULUM.append({
        "name": name,
        "folders": list(folders),
        "steps": cumul,
        "mb": mb,
    })

BASE_TOTAL_STEPS = cumul


def build_curriculum(target_steps: int) -> list[dict]:
    target_steps = max(len(BASE_CURRICULUM), int(target_steps))
    scaled = []
    previous = 0
    for idx, stage in enumerate(BASE_CURRICULUM):
        if idx == len(BASE_CURRICULUM) - 1:
            step_limit = target_steps
        else:
            ratio = stage["steps"] / max(1, BASE_TOTAL_STEPS)
            step_limit = max(previous + 1, int(round(target_steps * ratio)))
        scaled.append({
            **stage,
            "steps": min(step_limit, target_steps),
        })
        previous = scaled[-1]["steps"]
    return scaled


TOTAL_STEPS = DEFAULT_TARGET_STEPS
CURRICULUM = build_curriculum(TOTAL_STEPS)


def print_curriculum(curriculum: list[dict], total_steps: int) -> None:
    print(f"Auto curriculum: {total_steps} total steps")
    for idx, stage in enumerate(curriculum):
        prev = 0 if idx == 0 else curriculum[idx - 1]["steps"]
        print(f"  Stage {idx:02d} steps {prev:6d}-{stage['steps']:6d} "
              f"(+{stage['steps']-prev:6d}, folders={len(stage['folders'])})")


def stage_index_for_step(step: int, curriculum: list[dict]) -> int:
    for idx, stage in enumerate(curriculum):
        if step <= stage["steps"]:
            return idx
    return max(0, len(curriculum) - 1)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def get_chunks(data_dir, folders):
    chunks = []
    for f in folders:
        fp = data_dir / f
        if fp.exists():
            for jf in sorted(fp.glob("*.jsonl")):
                chunks.append(jf)
    return chunks


def load_texts(fp, max_lines=None):
    texts = []
    with open(fp, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                t = (d.get("text") or d.get("output") or "").strip()
                if d.get("instruction") or d.get("input"):
                    inp = (d.get("instruction") or d.get("input") or "").strip()
                    out = (d.get("response") or d.get("output") or "").strip()
                    t = f"{inp} {out}".strip()
                if len(t) > 10:
                    texts.append(t)
                    if max_lines and len(texts) >= max_lines:
                        break
            except:
                pass
    return texts


def format_chat_prompt(user: str, system: str = "") -> str:
    system = (system or DEFAULT_SYSTEM_PROMPT).strip()
    user = user.strip()
    return f"System: {system}\nUser: {user}\nAssistant:"


def format_chat_example(user: str, assistant: str, system: str = "") -> str:
    return f"{format_chat_prompt(user, system)} {assistant.strip()}"


def load_seed_chat(path=SEED_CHAT_PATH):
    examples = []
    path = Path(path)
    if not path.exists():
        return examples
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                user = (d.get("user") or d.get("instruction") or d.get("prompt") or "").strip()
                assistant = (d.get("assistant") or d.get("response") or d.get("output") or "").strip()
                system = (d.get("system") or "").strip()
                if user and assistant:
                    examples.append(format_chat_example(user, assistant, system))
            except Exception:
                pass
    return examples


def load_seed_chat_records(path=SEED_CHAT_PATH):
    records = []
    path = Path(path)
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                user = (d.get("user") or d.get("instruction") or d.get("prompt") or "").strip()
                assistant = (d.get("assistant") or d.get("response") or d.get("output") or "").strip()
                system = (d.get("system") or "").strip()
                if user and assistant:
                    records.append({
                        "prompt": format_chat_prompt(user, system),
                        "assistant": assistant,
                        "text": format_chat_example(user, assistant, system),
                    })
            except Exception:
                pass
    return records


class Dataset:
    def __init__(self, data_dir, folders, tokenizer, seq_len, seed_chat_path=SEED_CHAT_PATH,
                 seed_chat_ratio=DEFAULT_SEED_CHAT_RATIO,
                 seed_ratio_min=DEFAULT_SEED_RATIO_MIN,
                 seed_ratio_decay_steps=DEFAULT_SEED_RATIO_DECAY_STEPS,
                 max_seed_per_batch_pct=0.50):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed_chat_peak = max(0.0, min(1.0, float(seed_chat_ratio)))
        self.seed_chat_ratio = self.seed_chat_peak
        self.seed_ratio_min = max(0.0, min(1.0, float(seed_ratio_min)))
        self.seed_ratio_decay_steps = max(1, int(seed_ratio_decay_steps))
        self.max_seed_per_batch_pct = max(0.0, min(1.0, float(max_seed_per_batch_pct)))
        self._current_step = 0
        self.seed_chat_records = load_seed_chat_records(seed_chat_path)
        self.seed_chat = [record["text"] for record in self.seed_chat_records]
        self._cache = {}
        self._chunks = get_chunks(data_dir, folders)

    def set_step(self, step: int) -> None:
        """Update effective seed ratio with linear decay."""
        self._current_step = max(0, step)
        fraction = min(1.0, self._current_step / self.seed_ratio_decay_steps)
        self.seed_chat_ratio = self.seed_chat_peak - (self.seed_chat_peak - self.seed_ratio_min) * fraction
        self.seed_chat_ratio = max(self.seed_ratio_min, self.seed_chat_ratio)

    def set_folders(self, folders):
        self._chunks = get_chunks(self.data_dir, folders)
        self._cache = {}

    @property
    def chunk_count(self):
        return len(self._chunks)

    def sample_batch(self, batch_size, seq_len, allow_growth=True):
        x_list, y_list = [], []
        max_seed = max(1, int(batch_size * self.max_seed_per_batch_pct))
        seed_count = 0
        for _ in range(batch_size):
            use_seed = (self.seed_chat and seed_count < max_seed
                        and random.random() < self.seed_chat_ratio)
            if use_seed:
                record = random.choice(self.seed_chat_records)
                seed_count += 1
                chunk, target = self._encode_seed_chat(record, seq_len, allow_growth)
                x_list.append(chunk[:-1])
                y_list.append(target[1:])
                continue
            else:
                if not self._chunks:
                    continue
                fp = random.choice(self._chunks)
                if str(fp) not in self._cache:
                    self._cache[str(fp)] = load_texts(fp)
                texts = self._cache[str(fp)]
                if not texts:
                    continue
                t = random.choice(texts)
                encode_growth = allow_growth
            ids = self.tokenizer.encode(t, allow_growth=encode_growth)
            if len(ids) < seq_len + 1:
                ids = ids + [0] * (seq_len + 1 - len(ids))
            ms = max(0, len(ids) - seq_len - 1)
            start = random.randint(0, ms) if ms else 0
            chunk = ids[start:start + seq_len + 1]
            x_list.append(chunk[:-1]); y_list.append(chunk[1:])
        if not x_list:
            x_list.append([0] * seq_len); y_list.append([0] * seq_len)
        return torch.tensor(x_list, dtype=torch.long), torch.tensor(y_list, dtype=torch.long)

    def _encode_seed_chat(self, record, seq_len, allow_growth=True):
        prompt_ids = self.tokenizer.encode(record["prompt"], allow_growth=allow_growth)
        answer_ids = self.tokenizer.encode(" " + record["assistant"], allow_growth=allow_growth)
        ids = prompt_ids + answer_ids
        targets = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
        if len(ids) < seq_len + 1:
            pad = seq_len + 1 - len(ids)
            ids = ids + [0] * pad
            targets = targets + [IGNORE_INDEX] * pad
        elif len(ids) > seq_len + 1:
            max_start = max(0, min(len(prompt_ids), len(ids) - seq_len - 1))
            start = random.randint(0, max_start) if max_start else 0
            ids = ids[start:start + seq_len + 1]
            targets = targets[start:start + seq_len + 1]
        return ids, targets

    def eval_set(self, num_samples=2000):
        ids_list = []
        sample_chunks = get_chunks(self.data_dir, ["samples"])
        buffer = []
        for t in self.seed_chat:
            ids = self.tokenizer.encode(t, allow_growth=False)
            buffer.extend(ids)
            while len(buffer) >= self.seq_len + 1:
                ids_list.append(buffer[:self.seq_len + 1])
                buffer = buffer[self.seq_len + 1:]
                if len(ids_list) >= num_samples:
                    return ids_list
        for fp in sample_chunks:
            texts = load_texts(fp)
            for t in texts:
                ids = self.tokenizer.encode(t, allow_growth=False)
                buffer.extend(ids)
                while len(buffer) >= self.seq_len + 1:
                    ids_list.append(buffer[:self.seq_len + 1])
                    buffer = buffer[self.seq_len + 1:]
                    if len(ids_list) >= num_samples:
                        return ids_list
        if len(ids_list) == 0:
            ids_list.append([0] * (self.seq_len + 1))
        return ids_list


def eval_model(model, ids_list, batch_size=4, seq_len=128):
    model.eval()
    tl, tt = 0.0, 0
    with torch.no_grad():
        for _ in range(min(20, len(ids_list) // batch_size)):
            batch = random.sample(ids_list, min(batch_size, len(ids_list)))
            x_list, y_list = [], []
            for ids in batch:
                ms = max(0, len(ids) - seq_len - 1)
                start = random.randint(0, ms) if ms else 0
                ch = ids[start:start + seq_len + 1]
                x_list.append(ch[:-1]); y_list.append(ch[1:])
            x = torch.tensor(x_list); y = torch.tensor(y_list)
            logits, bal = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            tl += float(loss) * x.numel(); tt += x.numel()
    model.train()
    av = tl / max(tt, 1)
    return av, math.exp(min(av, 20))


def mtp_aux_loss(logits, targets, depth: int = MTP_DEPTH) -> torch.Tensor:
    """Auxiliary multi-token prediction loss for offsets 2..depth."""
    if depth <= 1:
        return logits.new_tensor(0.0)
    seq_len = targets.size(1)
    losses = []
    for offset in range(2, depth + 1):
        if seq_len < offset:
            break
        pred = logits[:, : seq_len - offset + 1, :]
        tgt = targets[:, offset - 1 :]
        if (tgt != IGNORE_INDEX).any():
            losses.append(F.cross_entropy(
                pred.reshape(-1, pred.size(-1)),
                tgt.reshape(-1),
                ignore_index=IGNORE_INDEX,
            ))
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def save_tokenizer_assets(core, tag=""):
    name = f"tokenizer{'_'+tag if tag else ''}"
    core.tokenizer.save(str(ASSETS_DIR / f"{name}.json"))
    torch.save({
        "vocab_size": core.tokenizer.size,
        "capacity": core.tokenizer.capacity,
        "merges": core.tokenizer.merges,
    }, ASSETS_DIR / f"{name}.pt")


def save_training_checkpoint(core, name, losses, step, best_val, stage, mtp_depth, total_tokens=0):
    core.save(str(CKPT_DIR / name), losses=losses,
              metadata_extra={"step": step,
                             "best_val": best_val,
                             "stage": stage,
                             "mtp_depth": mtp_depth,
                             "total_tokens": total_tokens})


def train(
    max_steps: int | None = None,
    target_steps: int = DEFAULT_TARGET_STEPS,
    mtp_depth: int = MTP_DEPTH,
    threads: int | None = None,
    compile_model: bool = False,
    freeze_backbone: bool = False,
    train_embeddings: bool = False,
    seed_chat_ratio: float = DEFAULT_SEED_CHAT_RATIO,
):
    global CURRICULUM, TOTAL_STEPS
    TOTAL_STEPS = max(1, int(target_steps))
    CURRICULUM = build_curriculum(TOTAL_STEPS)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR = Path("Download")
    if threads:
        torch.set_num_threads(max(1, threads))

    print(f"  NP-DNA v3: {CONFIG_NAME}, attn={USE_ATTENTION}")
    print(f"  {TOTAL_STEPS} planned steps, batch={BATCH_SIZE}, seq={SEQ_LEN}, "
          f"mtp_depth={mtp_depth}, seed_chat_ratio={seed_chat_ratio:.2f} "
          f"(decay->{DEFAULT_SEED_RATIO_MIN:.2f} over {DEFAULT_SEED_RATIO_DECAY_STEPS} steps)")
    print_curriculum(CURRICULUM, TOTAL_STEPS)

    base_cfg = CONFIGS[CONFIG_NAME]
    if USE_ATTENTION:
        for spec in base_cfg.mesh_specs:
            spec.strand.strand_type = "attention"
        base_cfg.mesh.strand.strand_type = "attention"

    from npdna.model import NpDnaModel
    from npdna.tokenizer import AtulyaTokenizer

    start_step = 1
    core = None
    current_stage = 0

    resume_dir = CKPT_DIR / "latest"
    if not resume_dir.exists():
        resume_dir = CKPT_DIR / "best"

    if resume_dir.exists():
        core = NpDnaCore.load(str(resume_dir))
        meta = json.loads((resume_dir / "metadata.json").read_text())
        start_step = meta.get("step", 0) + 1
        current_stage = stage_index_for_step(start_step - 1, CURRICULUM)
        print(f"\n  Resumed from {resume_dir.name}: step {start_step-1}, stage {current_stage}")

    if core is None:
        tok = AtulyaTokenizer(initial_capacity=base_cfg.initial_vocab,
                               max_capacity=base_cfg.max_vocab)
        model = NpDnaModel(base_cfg)
        core = NpDnaCore(model=model, tokenizer=tok, config=base_cfg)
        print(f"\n  Fresh: {model.parameter_count():,} params "
              f"({model.active_parameter_count():,} active)")

    # Load or train tokenizer
    if core.tokenizer.merges == []:
        tok_files = sorted(ASSETS_DIR.glob("tokenizer*.json"))
        if tok_files:
            print(f"  Loading tokenizer from {tok_files[-1].name}")
            tok2 = AtulyaTokenizer.load(str(tok_files[-1]))
            core.tokenizer = tok2
            core.model.resize_embeddings(core.tokenizer.capacity)
        else:
            print("  Training BPE tokenizer on all categories...")
            bpe_texts = []
            for folder in all_folders:
                if folder == "math":  # skip huge math folder for BPE
                    continue
                chunks = get_chunks(DATA_DIR, [folder])
                for fp in chunks[:2]:
                    bpe_texts.extend(load_texts(fp, max_lines=1000))
            bpe_texts = bpe_texts[:20000]
            bpe_texts.extend(load_seed_chat())
            print(f"  BPE on {len(bpe_texts):,} texts")
            core.tokenizer.train_bpe(bpe_texts, target_merges=8000, min_pair_freq=2)
            core.model.resize_embeddings(core.tokenizer.capacity)
            print(f"  Vocab: {core.tokenizer.size} tokens, cap={core.tokenizer.capacity}")
            save_tokenizer_assets(core, tag=f"stage0_v{core.tokenizer.size}")

    print(f"\n  Vocab: {core.tokenizer.size} tokens, cap={core.tokenizer.capacity}, "
          f"fill={core.tokenizer.fill_ratio:.1%}")

    # Dataset
    current_stage = stage_index_for_step(start_step - 1, CURRICULUM)
    dataset = Dataset(
        DATA_DIR,
        CURRICULUM[current_stage]["folders"],
        core.tokenizer,
        SEQ_LEN,
        seed_chat_ratio=seed_chat_ratio,
        seed_ratio_min=DEFAULT_SEED_RATIO_MIN,
        seed_ratio_decay_steps=min(TOTAL_STEPS // 2, DEFAULT_SEED_RATIO_DECAY_STEPS),
    )
    dataset.set_step(start_step - 1)
    eval_ids = dataset.eval_set(num_samples=2000)
    print(f"  Seed chat: {len(dataset.seed_chat)} examples")
    print(f"  Eval: {len(eval_ids)} sequences from held-out local/seed data")

    # Optimizer
    model = core.model
    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            core.model = model
            print("  torch.compile enabled")
        except Exception as exc:
            print(f"  torch.compile skipped: {str(exc)[:120]}")

    if freeze_backbone:
        from npdna.quant_turbo import freeze_for_partial_training

        trainable = freeze_for_partial_training(
            core,
            train_strands=True,
            train_embeddings=train_embeddings,
        )
        print(f"  partial training enabled: {trainable:,} trainable params")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01,
                             betas=(0.9, 0.95))

    losses = []
    best_val = float('inf')
    t_start = time.time()
    smooth_loss = 0.0
    total_tok = 0

    if start_step > 1 and (CKPT_DIR / "best" / "metadata.json").exists():
        meta = json.loads((CKPT_DIR / "best" / "metadata.json").read_text())
        losses = meta.get("losses", [])
        best_val = meta.get("best_val", float('inf'))
    if start_step > 1 and (resume_dir / "metadata.json").exists():
        meta = json.loads((resume_dir / "metadata.json").read_text())
        losses = meta.get("losses", losses)
        best_val = meta.get("best_val", best_val)

    end_step = TOTAL_STEPS if max_steps is None else min(TOTAL_STEPS, start_step + max_steps - 1)
    print(f"\n  Stage {current_stage}/7 ({dataset.chunk_count} chunks)")
    if max_steps is not None:
        print(f"  Smoke run: steps {start_step}-{end_step}\n")
    else:
        print()

    last_step = start_step - 1
    try:
        for step in range(start_step, end_step + 1):
            last_step = step
            # Curriculum stage switch
            new_stage = current_stage
            for si, stage in enumerate(CURRICULUM):
                if step <= stage["steps"]:
                    new_stage = si
                    break
                new_stage = si

            if new_stage != current_stage:
                current_stage = new_stage
                stage = CURRICULUM[current_stage]
                dataset.set_folders(stage["folders"])
                print(f"\n  >>> Stage {current_stage}/7 ({dataset.chunk_count} chunks) <<<\n")

                # Grow vocab if needed at stage transitions
                if core.tokenizer.fill_ratio > 0.9:
                    old_cap = core.tokenizer.capacity
                    more_texts = []
                    for fp in random.sample(dataset._chunks,
                                             min(3, len(dataset._chunks))):
                        more_texts.extend(load_texts(fp, max_lines=1000))
                    core.tokenizer.train_bpe(more_texts, target_merges=2000,
                                              min_pair_freq=2)
                    core.model.resize_embeddings(core.tokenizer.capacity)
                    if core.tokenizer.capacity > old_cap:
                        print(f"  Vocab grew: {old_cap} -> {core.tokenizer.capacity} "
                              f"(size={core.tokenizer.size})")
                        save_tokenizer_assets(core, tag=f"stage{current_stage}")

            # Warmup
            if step <= WARMUP_STEPS:
                lr = LR * step / max(WARMUP_STEPS, 1)
                for g in opt.param_groups:
                    g['lr'] = lr

            dataset.set_step(step)
            x, y = dataset.sample_batch(BATCH_SIZE, SEQ_LEN, allow_growth=True)
            total_tok += x.numel()

            model.train()
            logits, bal = model(x)
            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            mtp_loss = mtp_aux_loss(logits, y, depth=mtp_depth)
            loss = ce_loss + (MTP_WEIGHT * mtp_loss) + bal * 0.01

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step > WARMUP_STEPS:
                decay = 1.0 - step / TOTAL_STEPS
                for g in opt.param_groups:
                    g['lr'] = max(1e-5, LR * decay)

            loss_val = float(ce_loss.detach())
            losses.append(loss_val)
            smooth_loss = 0.95 * smooth_loss + 0.05 * loss_val if smooth_loss else loss_val

            # Log
            if step % LOG_EVERY == 0 or step == start_step:
                elapsed = time.time() - t_start
                rate = total_tok / max(elapsed, 1)
                steps_done = max(1, step - start_step + 1)
                seconds_per_step = elapsed / steps_done
                eta = seconds_per_step * max(0, end_step - step)
                cur_lr = opt.param_groups[0]['lr']
                best = min(losses) if losses else 0
                print(f"  step {step:5d}/{TOTAL_STEPS} | "
                      f"stage {current_stage:02d} | "
                       f"loss {smooth_loss:.2f} | mtp {float(mtp_loss.detach()):.2f} | best {best:.2f} | "
                       f"seed_r {dataset.seed_chat_ratio:.2f} | "
                       f"lr {cur_lr:.2e} | {rate:.0f} tok/s | eta {format_duration(eta)}")

            if step % LATEST_EVERY == 0:
                save_training_checkpoint(core, "latest", losses, step, best_val,
                                         current_stage, mtp_depth, total_tok)

            # Eval
            force_eval = max_steps is not None and step == end_step
            if step % EVAL_EVERY == 0 or force_eval:
                vl, vp = eval_model(model, eval_ids, BATCH_SIZE, SEQ_LEN)
                gen = core.generate("Hello.", max_tokens=20, temperature=0.3,
                                    top_k=30, top_p=0.85, repetition_penalty=1.2)
                safe = gen.encode('ascii', 'replace').decode('ascii')
                print(f"  VAL loss={vl:.4f} ppl={vp:.1f} | GEN: {safe[:80]}")
                if vl < best_val:
                    best_val = vl
                    core.save(str(CKPT_DIR / "best"), losses=losses,
                              metadata_extra={"step": step, "val_loss": vl,
                                             "stage": current_stage,
                                             "mtp_depth": mtp_depth})
                    save_tokenizer_assets(core)

            # Generation check every 1000 steps
            if step % 1000 == 0 or step == start_step:
                for p in ["Hi! How are you?", "What is gravity?"]:
                    o = core.generate(p, max_tokens=25, temperature=0.3,
                                      top_k=30, top_p=0.85, repetition_penalty=1.2)
                    safe = o.encode('ascii', 'replace').decode('ascii')
                    print(f"  GEN [{step}] {p[:20]} -> {safe[:70]}")

            # Checkpoint
            if step % SAVE_EVERY == 0:
                core.save(str(CKPT_DIR / f"step_{step}"), losses=losses,
                          metadata_extra={"step": step, "best_val": best_val,
                                         "stage": current_stage})

            if step % 500 == 0:
                gc.collect()
    except KeyboardInterrupt:
        if last_step >= start_step:
            print(f"\n  Interrupted. Saving latest checkpoint at step {last_step}...")
            save_training_checkpoint(core, "latest", losses, last_step, best_val,
                                     current_stage, mtp_depth, total_tok)
            save_tokenizer_assets(core)
        raise

    # Final
    elapsed = time.time() - t_start
    fv, fp = eval_model(model, eval_ids, BATCH_SIZE, SEQ_LEN)
    if max_steps is None:
        core.save(str(CKPT_DIR / "final"), losses=losses,
                  metadata_extra={"step": TOTAL_STEPS, "val_loss": fv,
                                 "total_tokens": total_tok,
                                 "total_time_sec": elapsed,
                                 "mtp_depth": mtp_depth})
        save_tokenizer_assets(core, tag="final")

    print(f"\n  DONE: {TOTAL_STEPS} steps in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"  Final val loss: {fv:.4f} | Best val: {best_val:.4f}")

    print("\n  --- Generation ---")
    for p in ["Hello. How are you?",
              "What is gravity?",
              "Tell me something interesting.",
              "Write a Python function.",
              "Who was Chanakya?",
              "What is machine learning?"]:
        o = core.generate(p, max_tokens=50, temperature=0.3,
                          top_k=30, top_p=0.85, repetition_penalty=1.2)
        safe = o.encode('ascii', 'replace').decode('ascii')
        print(f"  Q: {p}\n  A: {safe}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NP-DNA.")
    parser.add_argument("--steps", type=int, default=None, help="Run only this many steps for smoke testing.")
    parser.add_argument("--target-steps", type=int, default=DEFAULT_TARGET_STEPS,
                        help="Full training target. Omit --steps to train to this value.")
    parser.add_argument("--mtp-depth", type=int, default=MTP_DEPTH, help="Multi-token prediction depth.")
    parser.add_argument("--threads", type=int, default=None, help="PyTorch CPU thread count.")
    parser.add_argument("--compile", action="store_true", help="Try torch.compile for repeated training steps.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only genome seeds by default.")
    parser.add_argument("--train-embeddings", action="store_true", help="When freezing, also train embeddings.")
    parser.add_argument("--seed-chat-ratio", type=float, default=DEFAULT_SEED_CHAT_RATIO,
                        help="Fraction of batches sampled from data/seed_chat.jsonl.")
    parser.add_argument("--seed-ratio-min", type=float, default=DEFAULT_SEED_RATIO_MIN,
                        help="Floor for seed chat ratio after decay.")
    parser.add_argument("--seed-ratio-decay", type=int, default=DEFAULT_SEED_RATIO_DECAY_STEPS,
                        help="Steps over which seed ratio decays from initial to min.")
    args = parser.parse_args()
    train(
        max_steps=args.steps,
        target_steps=args.target_steps,
        mtp_depth=args.mtp_depth,
        threads=args.threads,
        compile_model=args.compile,
        freeze_backbone=args.freeze_backbone,
        train_embeddings=args.train_embeddings,
        seed_chat_ratio=args.seed_chat_ratio,
    )
