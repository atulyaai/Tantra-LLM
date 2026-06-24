"""NP-DNA — curriculum by data volume, dynamic vocab, fresh start."""
import argparse, sys, os, json, time, random, math, gc, threading
from pathlib import Path
from copy import deepcopy
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from npdna import NpDnaCore
from npdna.config import CONFIGS

try:
    import psutil
    physical_cores = psutil.cpu_count(logical=False) or mp.cpu_count()
except ImportError:
    import os
    if os.name == 'nt':
        physical_cores = max(1, mp.cpu_count() // 2)
    else:
        physical_cores = mp.cpu_count()
torch.set_num_threads(physical_cores)

# Config
CONFIG_NAME = "seed"
USE_ATTENTION = True
BATCH_SIZE = 4
SEQ_LEN = 256
LR = 3e-3
WARMUP_STEPS = 1000           # ~1% of 100k steps for stable warmup
LOG_EVERY = 25
SAVE_EVERY = 500
EVAL_EVERY = 250
LATEST_EVERY = 250
MTP_DEPTH = 3
MTP_WEIGHT = 0.25
DEFAULT_TARGET_STEPS = 100_000

CKPT_DIR = Path("model/npdna")
ASSETS_DIR = Path("model/tokenizer")
SEED_CHAT_PATH = Path("Download/seed")
DEFAULT_SEED_CHAT_RATIO = 0.35
DEFAULT_SEED_RATIO_MIN = 0.10
DEFAULT_SEED_RATIO_DECAY_STEPS = 30_000
DEFAULT_SYSTEM_PROMPT = "You are Atulya. Be warm, thoughtful, and direct."
IGNORE_INDEX = -100
MAX_CACHED_TEXTS_PER_CHUNK = 512
MAX_SCAN_LINES_PER_SAMPLE = 2048
MAX_PREENCODE_SEED_RECORDS = 50_000
LARGE_SEED_CHAT_BYTES = 256 * 1024 * 1024
SEED_VOCAB_SAMPLE_SIZE = 250_000
SEED_VOCAB_MERGE_ROUNDS = 12_000
SEED_TARGET_VOCAB_SIZE = 64_000

GENERATION_PROBE_PROMPTS = [
    "Hi! How are you?",
    "Can you help me plan my study session?",
    "Explain gravity simply.",
    "Why do things fall down?",
    "What is machine learning?",
    "Explain photosynthesis in one paragraph.",
    "Who was Chanakya?",
    "Tell me one interesting fact.",
    "Write a Python function to add two numbers.",
    "Write a Python function to multiply two numbers.",
    "What is 17 plus 29?",
    "If I have 12 apples and give away 5, how many are left?",
    "Give me three tips for learning faster.",
    "Write a short paragraph about discipline.",
    "Explain why clean data matters for training.",
    "What should I do if I feel stressed?",
]

FINAL_GENERATION_PROMPTS = [
    "Hello. How are you?",
    "Explain gravity to a 10 year old.",
    "Why do things fall down?",
    "Tell me something interesting.",
    "Write a Python function to add two numbers.",
    "Write a Python function to multiply two numbers.",
    "Who was Chanakya?",
    "What is machine learning?",
    "Explain photosynthesis.",
    "Give me study tips.",
    "Write a short paragraph about focus.",
    "If I have 9 apples and give away 4, how many are left?",
]


def sample_generation_prompts(step: int, count: int = 4) -> list[str]:
    """Pick stable random generation probes so logs test more than seed prompts."""
    if count >= len(GENERATION_PROBE_PROMPTS):
        return list(GENERATION_PROBE_PROMPTS)
    rng = random.Random(int(step))
    return rng.sample(GENERATION_PROBE_PROMPTS, count)

# Auto-calculate steps per dataset based on local dataset size.
DATASET_SIZES = {
    "factual": 679,
    "instruction": 195,
    "experts": 7,
    "reasoning": 164,
    "code": 363,
    "general": 139,
    "chat": 113,
    "emotion": 6,
    "spatial": 6,
    "action": 7,
}

def calc_steps(mb, base=200, max_steps=2000):
    if mb == 0:
        return 500        # tiny samples get 500 steps to overfit
    return min(max_steps, base + int((mb / 100) * 50))

# Build curriculum weights from data volume, then scale to target steps.
BASE_CURRICULUM = []
cumul = 0
all_folders = [
    "instruction",
    "code",
    "chat",
    "reasoning",
    "factual",
    "general",
    "experts",
    "emotion",
    "spatial",
    "action",
]
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


def scheduled_lr(step: int, peak_lr: float, target_steps: int) -> float:
    if step <= WARMUP_STEPS:
        return peak_lr * step / max(WARMUP_STEPS, 1)
    progress = (step - WARMUP_STEPS) / max(1, target_steps - WARMUP_STEPS)
    progress = min(1.0, max(0.0, progress))
    decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(1e-6, peak_lr * decay)


def get_chunks(data_dir, folders):
    chunks = []
    for f in folders:
        fp = data_dir / f
        if fp.exists():
            for jf in sorted(fp.glob("*.jsonl")):
                chunks.append(jf)

    # Fallback to consolidated train pack if folders are missing
    if not chunks:
        for fallback_name in ["train_pack_all_expanded_1040k.jsonl", "train_pack_core_20k.jsonl"]:
            fallback_path = data_dir / "train_pack" / fallback_name
            if fallback_path.exists():
                chunks.append(fallback_path)
                break
    return chunks


def _extract_training_text(line: str) -> str:
    d = json.loads(line.strip())
    user = (d.get("user") or d.get("instruction") or d.get("input") or d.get("prompt") or "").strip()
    out = (d.get("assistant") or d.get("response") or d.get("output") or "").strip()
    if user and out:
        return format_chat_example(user, out, (d.get("system") or "").strip())

    t = (d.get("text") or d.get("content") or "").strip()
    if t and len(t) > 80:
        sentences = t.split(". ")
        if len(sentences) >= 2:
            q = sentences[0].strip()
            body = ". ".join(sentences[1:]).strip()
            return format_chat_example(f"Tell me about: {q}.", body[:1200])
    return t


def load_texts(fp, max_lines=None, start_line=0):
    texts = []
    with open(fp, 'r', encoding='utf-8', errors='replace') as f:
        for idx, line in enumerate(f):
            if idx < start_line:
                continue
            try:
                t = _extract_training_text(line)
                if len(t) > 10:
                    texts.append(t)
                    if max_lines and len(texts) >= max_lines:
                        break
            except Exception:
                pass
    return texts


def sample_texts_from_chunk(fp, max_texts=MAX_CACHED_TEXTS_PER_CHUNK):
    """Read a random bounded window of lines from a possibly multi-GB JSONL chunk via byte seeking."""
    try:
        file_size = fp.stat().st_size
    except OSError:
        return []

    if file_size < 4096:
        return load_texts(fp, max_lines=max_texts, start_line=0)

    texts = []
    offset = random.randint(0, max(0, file_size - 128 * 1024))

    with open(fp, 'rb') as f:
        if offset > 0:
            f.seek(offset)
            f.readline()  # Skip partial line

        for _ in range(max_texts * 2):
            line_bytes = f.readline()
            if not line_bytes:
                break
            try:
                line = line_bytes.decode('utf-8', errors='replace').strip()
                t = _extract_training_text(line)
                if len(t) > 10:
                    texts.append(t)
                    if len(texts) >= max_texts:
                        break
            except Exception:
                pass

    if not texts:
        texts = load_texts(fp, max_lines=max_texts, start_line=0)

    random.shuffle(texts)
    return texts


def format_chat_prompt(user: str, system: str = "") -> str:
    system = (system or DEFAULT_SYSTEM_PROMPT).strip()
    user = user.strip()
    return f"System: {system}\nUser: {user}\nAssistant:"


def format_chat_example(user: str, assistant: str, system: str = "") -> str:
    return f"{format_chat_prompt(user, system)} {assistant.strip()}"


def _parse_qa_line(line: str):
    """Parse a single JSONL line into (user, assistant, system) or None."""
    try:
        d = json.loads(line.strip())
        user = (d.get("user") or d.get("instruction") or d.get("prompt") or "").strip()
        assistant = (d.get("assistant") or d.get("response") or d.get("output") or "").strip()
        system = (d.get("system") or "").strip()
        if user and assistant:
            return user, assistant, system
    except Exception:
        pass
    return None


def _load_qa_dir(path):
    """Yield parsed records from all .jsonl files in a directory."""
    path = Path(path)
    if not path.exists():
        # Fallback to train_pack_core_20k.jsonl if seed path is missing
        fallback = Path("Download/train_pack/train_pack_core_20k.jsonl")
        if fallback.exists():
            path = fallback
        else:
            fallback = Path("Download/train_pack/train_pack_all_expanded_1040k.jsonl")
            if fallback.exists():
                path = fallback
            else:
                return
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob("*.jsonl"))
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                result = _parse_qa_line(line)
                if result:
                    yield result


def load_seed_chat(path=SEED_CHAT_PATH):
    examples = []
    for user, assistant, system in _load_qa_dir(path):
        examples.append(format_chat_example(user, assistant, system))
    return examples


def load_seed_chat_records(path=SEED_CHAT_PATH):
    records = []
    for user, assistant, system in _load_qa_dir(path):
        records.append({
            "prompt": format_chat_prompt(user, system),
            "assistant": assistant,
            "text": format_chat_example(user, assistant, system),
        })
    return records


class JsonlSeedRecordStore:
    """Random-access seed chat records without loading the full JSONL into RAM."""

    def __init__(self, path: Path, eval_count: int = 2000):
        self.path = Path(path)
        self.files: list[Path] = []
        self._file_offsets: list[list[int]] = []
        self._counts: list[int] = []
        self.eval_records: list[dict] = []
        self._build_index(eval_count)

    def _build_index(self, eval_count: int) -> None:
        files = [self.path] if self.path.is_file() else sorted(self.path.rglob("*.jsonl"))
        eval_stride = None
        seen_valid = 0
        for fp in files:
            offsets: list[int] = []
            with open(fp, "rb") as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    parsed = _parse_qa_line(line.decode("utf-8", errors="replace"))
                    if not parsed:
                        continue
                    if eval_stride is None and eval_count > 0:
                        eval_stride = max(1, 20)
                    if len(self.eval_records) < eval_count and seen_valid % max(1, eval_stride or 1) == 0:
                        user, assistant, system = parsed
                        self.eval_records.append({
                            "prompt": format_chat_prompt(user, system),
                            "assistant": assistant,
                            "text": format_chat_example(user, assistant, system),
                        })
                    else:
                        offsets.append(offset)
                    seen_valid += 1
            if offsets:
                self.files.append(fp)
                self._file_offsets.append(offsets)
                self._counts.append(len(offsets))

        self._cumulative: list[int] = []
        total = 0
        for count in self._counts:
            total += count
            self._cumulative.append(total)

    def __len__(self) -> int:
        return self._cumulative[-1] if self._cumulative else 0

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        import bisect

        file_i = bisect.bisect_right(self._cumulative, idx)
        prev = 0 if file_i == 0 else self._cumulative[file_i - 1]
        offset = self._file_offsets[file_i][idx - prev]
        with open(self.files[file_i], "rb") as f:
            f.seek(offset)
            line = f.readline().decode("utf-8", errors="replace")
        parsed = _parse_qa_line(line)
        if not parsed:
            raise ValueError(f"Indexed seed record at {self.files[file_i]}:{offset} no longer parses")
        user, assistant, system = parsed
        return {
            "prompt": format_chat_prompt(user, system),
            "assistant": assistant,
            "text": format_chat_example(user, assistant, system),
        }

    def sample_texts(self, sample_size: int) -> Iterable[str]:
        total = len(self)
        if total <= 0 or sample_size <= 0:
            return
        stride = max(1, total // sample_size)
        yielded = 0
        for idx in range(0, total, stride):
            yield self[idx]["text"]
            yielded += 1
            if yielded >= sample_size:
                break


class Dataset:
    def __init__(self, data_dir, folders, tokenizer, seq_len, seed_chat_path=SEED_CHAT_PATH,
                 seed_chat_ratio=DEFAULT_SEED_CHAT_RATIO,
                 seed_ratio_min=DEFAULT_SEED_RATIO_MIN,
                 seed_ratio_decay_steps=DEFAULT_SEED_RATIO_DECAY_STEPS,
                 max_seed_per_batch_pct=0.50,
                 proportional_mix=True):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed_chat_peak = max(0.0, min(1.0, float(seed_chat_ratio)))
        self.seed_chat_ratio = self.seed_chat_peak
        self.seed_ratio_min = max(0.0, min(1.0, float(seed_ratio_min)))
        self.seed_ratio_decay_steps = max(1, int(seed_ratio_decay_steps))
        self.max_seed_per_batch_pct = max(0.0, min(1.0, float(max_seed_per_batch_pct)))
        self._current_step = 0
        self._proportional_mix = proportional_mix
        self._new_chunks = []
        self._prev_chunks = []
        seed_chat_path = Path(seed_chat_path)
        self.eval_seed_chat_records = []
        if seed_chat_path.exists() and seed_chat_path.is_file() and seed_chat_path.stat().st_size > LARGE_SEED_CHAT_BYTES:
            self.seed_chat_records = JsonlSeedRecordStore(seed_chat_path)
            self.eval_seed_chat_records = self.seed_chat_records.eval_records
        else:
            all_seed_chat_records = load_seed_chat_records(seed_chat_path)
            if len(all_seed_chat_records) > 20:
                eval_count = min(2000, max(1, len(all_seed_chat_records) // 20))
                stride = max(1, len(all_seed_chat_records) // eval_count)
                eval_indices = set(range(0, len(all_seed_chat_records), stride))
                eval_indices = set(list(eval_indices)[:eval_count])
                self.eval_seed_chat_records = [
                    record for i, record in enumerate(all_seed_chat_records)
                    if i in eval_indices
                ]
                self.seed_chat_records = [
                    record for i, record in enumerate(all_seed_chat_records)
                    if i not in eval_indices
                ]
            else:
                self.seed_chat_records = all_seed_chat_records
        self.seed_chat = self.seed_chat_records
        # Pre-encode modest seed sets once. Large overnight datasets are encoded
        # lazily so startup does not stall for minutes.
        self._seed_encoded = []
        self._preencode_seed = len(self.seed_chat_records) <= MAX_PREENCODE_SEED_RECORDS
        self._seed_vocab_signature = (self.tokenizer.size, self.tokenizer.capacity)
        self._vocab_changed = False
        if self._preencode_seed:
            for r in self.seed_chat_records:
                p = tokenizer.encode(r["prompt"], allow_growth=False)
                a = tokenizer.encode(" " + r["assistant"], allow_growth=False)
                ids = p + a
                targets = [IGNORE_INDEX] * len(p) + a
                self._seed_encoded.append((ids, targets))
        self._cache = {}
        self.set_folders(folders)

    def _current_vocab_signature(self) -> tuple[int, int]:
        return self.tokenizer.size, self.tokenizer.capacity

    def note_vocab_changed(self) -> None:
        self._vocab_changed = True

    def _seed_cache_is_valid(self) -> bool:
        if self._current_vocab_signature() != self._seed_vocab_signature:
            self._vocab_changed = True
        return self._preencode_seed and not self._vocab_changed

    def seed_vocab_texts(self, sample_size: int = SEED_VOCAB_SAMPLE_SIZE) -> Iterable[str]:
        if isinstance(self.seed_chat_records, JsonlSeedRecordStore):
            yield from self.seed_chat_records.sample_texts(sample_size)
            return
        for record in self.seed_chat_records[:sample_size]:
            yield record["text"]

    def set_step(self, step: int) -> None:
        """Update effective seed ratio with linear decay."""
        self._current_step = max(0, step)
        fraction = min(1.0, self._current_step / self.seed_ratio_decay_steps)
        self.seed_chat_ratio = self.seed_chat_peak - (self.seed_chat_peak - self.seed_ratio_min) * fraction
        self.seed_chat_ratio = max(self.seed_ratio_min, self.seed_chat_ratio)

    def set_folders(self, folders):
        if self._current_vocab_signature() != self._seed_vocab_signature:
            self._vocab_changed = True
        self._new_chunks = get_chunks(self.data_dir, folders[-1:]) if folders else []
        self._prev_chunks = get_chunks(self.data_dir, folders[:-1]) if len(folders) > 1 else []
        self._chunks = get_chunks(self.data_dir, folders)
        self._cache = {}

    @property
    def chunk_count(self):
        return len(self._chunks)

    def sample_batch(self, batch_size, seq_len, allow_growth=True):
        x_list, y_list = [], []
        max_seed = max(1, int(batch_size * self.max_seed_per_batch_pct))
        seed_count = 0
        n_seed = len(self.seed_chat_records)
        for _ in range(batch_size):
            use_seed = (n_seed > 0 and seed_count < max_seed
                        and random.random() < self.seed_chat_ratio)
            if use_seed and n_seed:
                idx = random.randrange(n_seed)
                seed_count += 1
                chunk, target = self._encode_seed_chat(idx, seq_len, allow_growth)
                x_list.append(chunk[:-1])
                y_list.append(target[1:])
                continue
            else:
                if not self._chunks:
                    continue
                if self._proportional_mix and self._prev_chunks and random.random() < 0.30:
                    fp = random.choice(self._prev_chunks)
                elif self._new_chunks:
                    fp = random.choice(self._new_chunks)
                else:
                    fp = random.choice(self._chunks)
                cache_key = str(fp)
                if cache_key not in self._cache or not self._cache[cache_key]:
                    self._cache[cache_key] = sample_texts_from_chunk(fp)
                texts = self._cache[str(fp)]
                if not texts:
                    continue
                t = texts.pop()
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

    def _encode_seed_chat(self, idx, seq_len, allow_growth=True):
        if self._seed_cache_is_valid() and not allow_growth:
            ids, targets = self._seed_encoded[idx]
            ids = list(ids); targets = list(targets)
        else:
            ids, targets = self._encode_seed_chat_record(self.seed_chat_records[idx], allow_growth)
        return self._fit_seed_window(ids, targets, seq_len)

    def _encode_seed_chat_record(self, record, allow_growth=True):
        prompt_ids = self.tokenizer.encode(record["prompt"], allow_growth=allow_growth)
        answer_ids = self.tokenizer.encode(" " + record["assistant"], allow_growth=allow_growth)
        ids = prompt_ids + answer_ids
        targets = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
        return ids, targets

    def _fit_seed_window(self, ids, targets, seq_len):
        if len(ids) < seq_len + 1:
            pad = seq_len + 1 - len(ids)
            ids = ids + [0] * pad
            targets = targets + [IGNORE_INDEX] * pad
        elif len(ids) > seq_len + 1:
            answer_start = next((i for i, t in enumerate(targets) if t != IGNORE_INDEX), len(ids) - 1)
            max_start = max(0, len(ids) - seq_len - 1)
            # Ensure the sampled window includes at least one assistant token.
            low = max(0, answer_start - seq_len)
            high = min(answer_start, max_start)
            if low > high:
                low = high = min(max_start, max(0, answer_start - 1))
            start = random.randint(low, high) if high > low else low
            ids = ids[start:start + seq_len + 1]
            targets = targets[start:start + seq_len + 1]
        return ids, targets

    def eval_set(self, num_samples=2000):
        ids_list = []
        for record in self.eval_seed_chat_records[:num_samples]:
            ids, targets = self._encode_seed_chat_record(record, allow_growth=False)
            ids, targets = self._fit_seed_window(ids, targets, self.seq_len)
            ids_list.append((ids, targets))
        if ids_list:
            return ids_list
        sample_chunks = get_chunks(self.data_dir, ["samples"])
        buffer = []
        for record in self.seed_chat:
            text = record["text"] if isinstance(record, dict) else str(record)
            ids = self.tokenizer.encode(text, allow_growth=False)
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
        for fp in self._chunks[: min(8, len(self._chunks))]:
            texts = sample_texts_from_chunk(fp, max_texts=64)
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
        if not ids_list:
            model.train()
            return float("inf"), float("inf")
        eval_batches = min(20, max(1, math.ceil(len(ids_list) / max(1, batch_size))))
        for _ in range(eval_batches):
            batch = random.sample(ids_list, min(batch_size, len(ids_list)))
            x_list, y_list = [], []
            for item in batch:
                if isinstance(item, tuple):
                    ids, targets = item
                    x_list.append(ids[:-1])
                    y_list.append(targets[1:])
                else:
                    ids = item
                    ms = max(0, len(ids) - seq_len - 1)
                    start = random.randint(0, ms) if ms else 0
                    ch = ids[start:start + seq_len + 1]
                    x_list.append(ch[:-1]); y_list.append(ch[1:])
            x = torch.tensor(x_list); y = torch.tensor(y_list)
            active_targets = (y != IGNORE_INDEX).sum().item()
            if active_targets == 0:
                continue
            logits, bal = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            tl += float(loss) * active_targets; tt += active_targets
    model.train()
    if tt == 0:
        return float("inf"), float("inf")
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


class PrefetchLoader:
    """Background-thread data prefetcher.

    While the GPU/CPU processes the current batch, the next batch is
    being prepared on a background thread.  Hides I/O + tokenisation
    latency behind the forward/backward compute.
    """

    def __init__(self, dataset, batch_size: int, seq_len: int, allow_growth: bool = True):
        import queue
        self._dataset = dataset
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._allow_growth = allow_growth
        self._queue = queue.Queue(maxsize=1)
        self._error: BaseException | None = None
        self._running = True
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()

    def _prefetch_loop(self) -> None:
        while self._running:
            try:
                x, y = self._dataset.sample_batch(
                    self._batch_size, self._seq_len, allow_growth=self._allow_growth,
                )
                if not self._running:
                    break
                self._queue.put((x, y))
            except BaseException as exc:
                self._error = exc
                self._queue.put((None, None))
                break

    def kick(self) -> None:
        """No-op: prefetch loop runs continuously."""
        pass

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Block until the prefetched batch is ready, then return it."""
        if self._error is not None:
            raise self._error
        res = self._queue.get()
        if res == (None, None) and self._error is not None:
            raise self._error
        return res

    def update_step(self, step: int) -> None:
        """Update the dataset's step counter (controls seed ratio decay)."""
        self._dataset.set_step(step)

    def stop(self) -> None:
        self._running = False
        import queue
        try:
            while not self._queue.empty():
                self._queue.get_nowait()
            self._queue.put_nowait((None, None))
        except queue.Full:
            pass


def save_tokenizer_assets(core, tag=""):
    name = f"tokenizer{'_'+tag if tag else ''}"
    core.tokenizer.save(str(ASSETS_DIR / f"{name}.json"))
    torch.save({
        "vocab_size": core.tokenizer.size,
        "capacity": core.tokenizer.capacity,
        "merges": core.tokenizer.merges,
    }, ASSETS_DIR / f"{name}.pt")


def save_training_checkpoint(
    core,
    name,
    losses,
    step,
    best_val,
    stage,
    mtp_depth,
    total_tokens=0,
    batch_size=BATCH_SIZE,
    seq_len=SEQ_LEN,
    mtp_weight=MTP_WEIGHT,
    grad_accum_steps=1,
    ema_loss=None,
    best_ema_loss=None,
    target_steps=None,
    peak_lr=None,
    warmup_steps=WARMUP_STEPS,
):
    metadata = {"step": step,
                "best_val": best_val,
                "stage": stage,
                "mtp_depth": mtp_depth,
                "total_tokens": total_tokens,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "mtp_weight": mtp_weight,
                "grad_accum_steps": grad_accum_steps,
                "target_steps": int(target_steps) if target_steps is not None else DEFAULT_TARGET_STEPS,
                "peak_lr": float(peak_lr) if peak_lr is not None else LR,
                "warmup_steps": int(warmup_steps),
                "lr_schedule": "cosine"}
    if ema_loss is not None:
        metadata["ema_loss"] = ema_loss
    if best_ema_loss is not None:
        metadata["best_ema_loss"] = best_ema_loss
    core.save(str(CKPT_DIR / name), losses=losses, metadata_extra=metadata)


def train(
    max_steps: int | None = None,
    target_steps: int = DEFAULT_TARGET_STEPS,
    lr: float = LR,
    mtp_depth: int = MTP_DEPTH,
    threads: int | None = None,
    compile_model: bool = False,
    freeze_backbone: bool = False,
    train_embeddings: bool = False,
    seed_chat_ratio: float = DEFAULT_SEED_CHAT_RATIO,
    seed_ratio_min: float = DEFAULT_SEED_RATIO_MIN,
    seed_ratio_decay_steps: int = DEFAULT_SEED_RATIO_DECAY_STEPS,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    mtp_weight: float = MTP_WEIGHT,
    grad_accum_steps: int = 1,
    seed_only: bool = False,
    log_every: int = LOG_EVERY,
    save_every: int = SAVE_EVERY,
    eval_every: int = EVAL_EVERY,
    latest_every: int = LATEST_EVERY,
    fresh_start: bool = False,
    reset_step_on_resume: bool = False,
    tokenizer_path: str | None = None,
    seed_chat_path: str | None = None,
    seq_preset: str | None = None,
    complexity: float | None = None,
    amp: bool = False,
    proportional_mix: bool = True,
    vocab_growth_sample: int = SEED_VOCAB_SAMPLE_SIZE,
    vocab_growth_merges: int = SEED_VOCAB_MERGE_ROUNDS,
    target_vocab_size: int = SEED_TARGET_VOCAB_SIZE,
    distill: bool = False,
    use_attention: bool = USE_ATTENTION,
    all_folders_now: bool = False,
):
    global CURRICULUM, TOTAL_STEPS
    TOTAL_STEPS = max(1, int(target_steps))
    CURRICULUM = build_curriculum(TOTAL_STEPS)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR = Path("Download")
    if threads:
        torch.set_num_threads(max(1, threads))

    print(f"  NP-DNA: {CONFIG_NAME}, attn={use_attention}")
    if seq_preset:
        presets = {
            "short": 64,
            "default": SEQ_LEN,
            "medium": 128,
            "long": 192,
        }
        if seq_preset not in presets:
            raise ValueError(f"Unknown --seq-preset {seq_preset!r}; choose {sorted(presets)}")
        seq_len = presets[seq_preset]

    batch_size = max(1, int(batch_size))
    seq_len = max(16, int(seq_len))
    seed_ratio_min = max(0.0, min(1.0, float(seed_ratio_min)))
    seed_ratio_decay_steps = max(1, int(seed_ratio_decay_steps))
    mtp_weight = max(0.0, float(mtp_weight))
    grad_accum_steps = max(1, int(grad_accum_steps))
    seed_only = bool(seed_only)
    all_folders_now = bool(all_folders_now)
    lr = max(1e-6, float(lr))
    log_every = max(1, int(log_every))
    save_every = max(0, int(save_every))
    eval_every = max(0, int(eval_every))
    latest_every = max(0, int(latest_every))
    fresh_start = bool(fresh_start)
    tokenizer_path_obj = Path(tokenizer_path) if tokenizer_path else None
    seed_chat_path_obj = Path(seed_chat_path) if seed_chat_path else SEED_CHAT_PATH

    print(f"  {TOTAL_STEPS} planned steps, batch={batch_size}, seq={seq_len}, "
          f"mtp_depth={mtp_depth}, seed_chat_ratio={seed_chat_ratio:.2f} "
          f"(decay->{seed_ratio_min:.2f} over {seed_ratio_decay_steps} steps), "
          f"lr={lr:.2e}, mtp_weight={mtp_weight:.2f}, grad_accum={grad_accum_steps}, "
          f"seed_only={seed_only}, all_folders={all_folders_now}, latest_every={latest_every}, "
          f"eval_every={eval_every}, save_every={save_every}, "
          f"fresh_start={fresh_start}, reset_step_on_resume={reset_step_on_resume}, "
          f"seed_chat_path={seed_chat_path_obj}")
    warmup_first_lr = lr / max(WARMUP_STEPS, 1)
    if lr < 1e-4:
        print(f"  WARNING: peak lr={lr:.2e} is very small; "
              f"warmup step 1 will be {warmup_first_lr:.2e}. "
              "Use --lr 1e-3 or --lr 5e-4 for the larger smoke run unless you intend a slow fine-tune.")
    print_curriculum(CURRICULUM, TOTAL_STEPS)

    base_cfg = deepcopy(CONFIGS[CONFIG_NAME])
    if complexity is not None:
        from npdna.config import NpDnaConfig
        base_cfg = NpDnaConfig(complexity=max(0.5, float(complexity)))
    if use_attention:
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

    if resume_dir.exists() and not fresh_start:
        core = NpDnaCore.load(str(resume_dir))
        meta = json.loads((resume_dir / "metadata.json").read_text())
        if "target_steps" in meta and int(target_steps) == DEFAULT_TARGET_STEPS:
            TOTAL_STEPS = max(1, int(meta["target_steps"]))
            CURRICULUM = build_curriculum(TOTAL_STEPS)
            print(f"  Restored target_steps={TOTAL_STEPS} from checkpoint metadata")
        loaded_step = meta.get("step", 0)
        start_step = 1 if reset_step_on_resume else loaded_step + 1
        current_stage = stage_index_for_step(start_step - 1, CURRICULUM)
        reset_note = " (step counter reset)" if reset_step_on_resume else ""
        print(f"\n  Resumed from {resume_dir.name}: loaded step {loaded_step}, "
              f"starting step {start_step}, stage {current_stage}{reset_note}")

    if core is None:
        tok = AtulyaTokenizer(initial_capacity=base_cfg.initial_vocab,
                               max_capacity=base_cfg.max_vocab)
        model = NpDnaModel(base_cfg)
        core = NpDnaCore(model=model, tokenizer=tok, config=base_cfg)
        print(f"\n  Fresh: {model.parameter_count():,} params "
              f"({model.active_parameter_count():,} active), "
              f"hidden={base_cfg.hidden_size}, state={base_cfg.state_size}")

    # Load or train tokenizer
    if core.tokenizer.merges == []:
        tok_files = [tokenizer_path_obj] if tokenizer_path_obj else sorted(ASSETS_DIR.glob("tokenizer*.json"))
        tok_files = [path for path in tok_files if path and path.exists()]
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
            save_tokenizer_assets(core)

    print(f"\n  Vocab: {core.tokenizer.size} tokens, cap={core.tokenizer.capacity}, "
          f"fill={core.tokenizer.fill_ratio:.1%}")

    # Dataset
    current_stage = stage_index_for_step(start_step - 1, CURRICULUM)
    if seed_only:
        dataset_folders = []
    elif all_folders_now:
        dataset_folders = list(all_folders)
    else:
        dataset_folders = CURRICULUM[current_stage]["folders"]
    dataset = Dataset(
        DATA_DIR,
        dataset_folders,
        core.tokenizer,
        seq_len,
        seed_chat_path=seed_chat_path_obj,
        seed_chat_ratio=seed_chat_ratio,
        seed_ratio_min=seed_ratio_min,
        seed_ratio_decay_steps=seed_ratio_decay_steps,
        max_seed_per_batch_pct=1.0 if seed_only else 0.50,
        proportional_mix=proportional_mix,
    )
    dataset.set_step(start_step - 1)
    vocab_growth_sample = max(0, int(vocab_growth_sample))
    vocab_growth_merges = max(0, int(vocab_growth_merges))
    target_vocab_size = max(0, int(target_vocab_size))
    if len(dataset.seed_chat_records) and seed_chat_ratio > 0 and vocab_growth_sample and vocab_growth_merges:
        old_vocab_size = core.tokenizer.size
        old_vocab_cap = core.tokenizer.capacity
        if target_vocab_size > old_vocab_size:
            vocab_growth_merges = max(vocab_growth_merges, target_vocab_size - old_vocab_size)
        print(
            f"  Vocab pre-growth: sampling up to {vocab_growth_sample:,} seed records, "
            f"target_vocab={target_vocab_size:,}, max_merges={vocab_growth_merges:,}..."
        )
        added_vocab = core.tokenizer.dynamic_vocab_growth(
            dataset.seed_vocab_texts(vocab_growth_sample),
            sample_size=vocab_growth_sample,
            merge_rounds=vocab_growth_merges,
            min_pair_freq=2,
        )
        if core.tokenizer.fill_ratio >= core.tokenizer.growth_threshold:
            reserve_capacity = math.ceil(core.tokenizer.size / 0.75)
            core.tokenizer.ensure_capacity(reserve_capacity)
        if core.tokenizer.capacity != old_vocab_cap:
            core.model.resize_embeddings(core.tokenizer.capacity)
        if core.tokenizer.size != old_vocab_size or core.tokenizer.capacity != old_vocab_cap:
            dataset.note_vocab_changed()
            save_tokenizer_assets(core)
        print(
            f"  Vocab pre-growth: size {old_vocab_size} -> {core.tokenizer.size} "
            f"(+{added_vocab}), cap {old_vocab_cap} -> {core.tokenizer.capacity}, "
            f"fill={core.tokenizer.fill_ratio:.1%}"
        )
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
        from npdna.brain import freeze_for_partial_training

        trainable = freeze_for_partial_training(
            core,
            train_strands=True,
            train_embeddings=train_embeddings,
        )
        print(f"  partial training enabled: {trainable:,} trainable params")

    # Exclude unused multimodal projectors from optimizer to save memory
    _base_model = model.module if hasattr(model, "module") else getattr(model, "_orig_mod", model)
    _excluded_ids = set()
    for _proj_name in ("vision_projector", "audio_projector"):
        _proj = getattr(_base_model, _proj_name, None)
        if _proj is not None:
            for _p in _proj.parameters():
                _excluded_ids.add(id(_p))
                _p.requires_grad = False
    _train_params = [p for p in model.parameters() if p.requires_grad and id(p) not in _excluded_ids]

    opt = torch.optim.AdamW(_train_params, lr=lr, weight_decay=0.01)

    losses = []
    best_val = float('inf')
    t_start = time.time()
    smooth_loss = 0.0
    ema_loss = None
    best_ema_loss = float("inf")
    total_tok = 0

    if distill:
        try:
            from npdna.distill import DistillationTeacher, compute_distillation_loss
            teacher = DistillationTeacher(device="cpu")
        except ImportError as e:
            print(f"Distillation failed to initialize: {e}")
            teacher = None
    else:
        teacher = None

    if start_step > 1 and not reset_step_on_resume and (CKPT_DIR / "best" / "metadata.json").exists():
        meta = json.loads((CKPT_DIR / "best" / "metadata.json").read_text())
        losses = meta.get("losses", [])
        best_val = meta.get("best_val", float('inf'))
        ema_loss = meta.get("ema_loss", ema_loss)
        best_ema_loss = meta.get("best_ema_loss", best_ema_loss)
    if start_step > 1 and not reset_step_on_resume and (resume_dir / "metadata.json").exists():
        meta = json.loads((resume_dir / "metadata.json").read_text())
        losses = meta.get("losses", losses)
        best_val = meta.get("best_val", best_val)
        ema_loss = meta.get("ema_loss", ema_loss)
        best_ema_loss = meta.get("best_ema_loss", best_ema_loss)

    end_step = TOTAL_STEPS if max_steps is None else min(TOTAL_STEPS, start_step + max_steps - 1)
    print(f"\n  Stage {current_stage}/{len(CURRICULUM)-1} ({dataset.chunk_count} chunks)")
    if max_steps is not None:
        print(f"  Smoke run: steps {start_step}-{end_step}\n")
    else:
        print()

    last_step = start_step - 1
    prefetcher = PrefetchLoader(dataset, batch_size, seq_len, allow_growth=True)
    prefetcher.update_step(start_step)

    try:
        for step in range(start_step, end_step + 1):
            last_step = step
            # LR schedule — set at top of loop for all steps
            step_lr = scheduled_lr(step, lr, TOTAL_STEPS)
            for g in opt.param_groups:
                g['lr'] = step_lr

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
                if seed_only:
                    next_folders = []
                elif all_folders_now:
                    next_folders = list(all_folders)
                else:
                    next_folders = stage["folders"]
                dataset.set_folders(next_folders)
                print(f"\n  >>> Stage {current_stage}/{len(CURRICULUM)-1} ({dataset.chunk_count} chunks) <<<\n")

                # Grow vocab if needed at stage transitions
                if core.tokenizer.fill_ratio > 0.9:
                    old_cap = core.tokenizer.capacity
                    old_size = core.tokenizer.size
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
                    if core.tokenizer.size != old_size or core.tokenizer.capacity != old_cap:
                        dataset.note_vocab_changed()

                eval_ids = dataset.eval_set(num_samples=2000)

                # Reset prefetcher to load from new folders
                if prefetcher is not None:
                    prefetcher.stop()
                prefetcher = PrefetchLoader(dataset, batch_size, seq_len, allow_growth=True)
                prefetcher.update_step(step)

            # (LR schedule is now set at the top of the loop)

            model.train()
            opt.zero_grad(set_to_none=True)
            ce_parts = []
            mtp_parts = []
            for micro_i in range(grad_accum_steps):
                x, y = prefetcher.get()
                prefetcher.update_step(step)
                total_tok += x.numel()
                with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=amp):
                    logits, bal = model(x)
                    ce_loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1),
                        ignore_index=IGNORE_INDEX,
                        label_smoothing=0.1,
                    )
                    mtp_loss = mtp_aux_loss(logits, y, depth=mtp_depth)

                    # Adaptive Depth (Early Exit) loss
                    # Each exit head learns to predict whether ITS OWN layer's
                    # logits would match the final target, not the full model's.
                    exit_loss = 0.0
                    base_model = model.module if hasattr(model, "module") else getattr(model, "_orig_mod", model)
                    if getattr(base_model, "_last_exit_logits", None) and getattr(base_model, "_last_layer_xs", None):
                        valid_mask = (y != IGNORE_INDEX).unsqueeze(-1)
                        if valid_mask.any():
                            for conf_logit, layer_x in zip(base_model._last_exit_logits, base_model._last_layer_xs):
                                with torch.no_grad():
                                    layer_normed = base_model.final_norm(layer_x)
                                    layer_logits = base_model.lm_head(layer_normed)
                                    correct = (layer_logits.argmax(dim=-1) == y).float().unsqueeze(-1)
                                exit_loss += F.binary_cross_entropy_with_logits(
                                    conf_logit[valid_mask], correct[valid_mask]
                                )
                            exit_loss = exit_loss / max(1, len(base_model._last_exit_logits))

                    distill_loss = 0.0
                    if teacher is not None:
                        # Decode input_ids to text, get teacher logits
                        with torch.no_grad():
                            texts = [core.decode(x[b].tolist()) for b in range(x.size(0))]
                            t_logits = teacher.get_teacher_logits(texts)
                        distill_loss = compute_distillation_loss(logits, t_logits)

                    loss = ce_loss + (mtp_weight * mtp_loss) + bal * 0.1 + exit_loss * 0.5 + distill_loss * 2.0
                (loss / grad_accum_steps).backward()
                ce_parts.append(ce_loss.detach())
                mtp_parts.append(mtp_loss.detach())
            ce_loss = torch.stack(ce_parts).mean()
            mtp_loss = torch.stack(mtp_parts).mean()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # Strand Evolution (DNA cloning/mutation/pruning)
            if step > WARMUP_STEPS and step % 2000 == 0:
                actions = {}
                for i, mesh in enumerate(core.model.mesh_layers):
                    if hasattr(mesh, "evolve_strands"):
                        mesh_actions = mesh.evolve_strands()
                        if mesh_actions:
                            actions[f"layer_{i}"] = mesh_actions
                if actions:
                    print(f"  [Evolution] step {step} - {actions}")
            # Cortex write: store hidden states every 1000 steps
            if step > WARMUP_STEPS and step % 1000 == 0:
                model.eval()
                with torch.no_grad():
                    sample_x = x[:2, :32]  # small sample from current batch
                    h = model.embedding(sample_x)
                    for mesh, norm in zip(model.mesh_layers, model.layer_norms):
                        out, _ = mesh(h)
                        h = norm(h + out)
                    h = model.final_norm(h)
                    vectors = h[:, -1].float()
                    core.cortex.store_batch(
                        vectors,
                        vectors,
                        topic=f"step_{step}",
                        source=f"train_step_{step}",
                    )
                model.train()

            # (LR schedule is now set at the top of the loop)

            loss_val = float(ce_loss.detach())
            losses.append(loss_val)
            smooth_loss = 0.95 * smooth_loss + 0.05 * loss_val if smooth_loss else loss_val
            ema_loss = loss_val if ema_loss is None else 0.99 * ema_loss + 0.01 * loss_val
            best_ema_loss = min(best_ema_loss, ema_loss)

            # Log
            if step % log_every == 0 or step == start_step:
                elapsed = time.time() - t_start
                rate = total_tok / max(elapsed, 1)
                steps_done = max(1, step - start_step + 1)
                seconds_per_step = elapsed / steps_done
                eta = seconds_per_step * max(0, end_step - step)
                cur_lr = opt.param_groups[0]['lr']
                best = best_ema_loss if math.isfinite(best_ema_loss) else (min(losses) if losses else 0)
                print(f"  step {step:5d}/{TOTAL_STEPS} | "
                      f"stage {current_stage:02d} | "
                       f"loss {smooth_loss:.2f} | ema {ema_loss:.2f} | mtp {float(mtp_loss.detach()):.2f} | best_ema {best:.2f} | "
                       f"seed_r {dataset.seed_chat_ratio:.2f} | "
                       f"lr {cur_lr:.2e} | {rate:.0f} tok/s | eta {format_duration(eta)}")

            if latest_every and step % latest_every == 0:
                save_training_checkpoint(core, "latest", losses, step, best_val,
                                         current_stage, mtp_depth, total_tok,
                                         batch_size=batch_size, seq_len=seq_len,
                                         mtp_weight=mtp_weight,
                                         grad_accum_steps=grad_accum_steps,
                                         ema_loss=ema_loss,
                                         best_ema_loss=best_ema_loss,
                                         target_steps=TOTAL_STEPS,
                                         peak_lr=lr)

            # Eval
            force_eval = max_steps is not None and step == end_step
            if (eval_every and step % eval_every == 0) or force_eval:
                vl, vp = eval_model(model, eval_ids, batch_size, seq_len)
                gen = core.generate("Hello.", max_tokens=20, temperature=0.3,
                                    top_k=30, top_p=0.85, repetition_penalty=1.2,
                                    context_window=256)
                safe = gen.encode('ascii', 'replace').decode('ascii')
                print(f"  VAL loss={vl:.4f} ppl={vp:.1f} | GEN: {safe[:80]}")
                if vl < best_val:
                    best_val = vl
                    core.save(str(CKPT_DIR / "best"), losses=losses,
                              metadata_extra={"step": step, "val_loss": vl,
                                             "stage": current_stage,
                                             "mtp_depth": mtp_depth,
                                             "batch_size": batch_size,
                                             "seq_len": seq_len,
                                             "mtp_weight": mtp_weight,
                                             "grad_accum_steps": grad_accum_steps,
                                             "ema_loss": ema_loss,
                                             "best_ema_loss": best_ema_loss,
                                             "target_steps": TOTAL_STEPS,
                                             "peak_lr": lr})
                    save_tokenizer_assets(core)

            # Generation check every 1000 steps
            if step % 1000 == 0 or step == start_step:
                for p in sample_generation_prompts(step):
                    o = core.generate(p, max_tokens=25, temperature=0.3,
                                      top_k=30, top_p=0.85, repetition_penalty=1.2,
                                      context_window=256)
                    safe = o.encode('ascii', 'replace').decode('ascii')
                    print(f"  GEN [{step}] {p[:20]} -> {safe[:70]}")

            # Checkpoint
            if save_every and step % save_every == 0:
                core.save(str(CKPT_DIR / f"step_{step}"), losses=losses,
                          metadata_extra={"step": step, "best_val": best_val,
                                         "stage": current_stage,
                                         "batch_size": batch_size,
                                         "seq_len": seq_len,
                                         "mtp_weight": mtp_weight,
                                         "grad_accum_steps": grad_accum_steps,
                                         "ema_loss": ema_loss,
                                         "best_ema_loss": best_ema_loss,
                                         "target_steps": TOTAL_STEPS,
                                         "peak_lr": lr})

            if step % 500 == 0:
                gc.collect()
    except KeyboardInterrupt:
        if last_step >= start_step:
            print(f"\n  Interrupted. Saving latest checkpoint at step {last_step}...")
            save_training_checkpoint(core, "latest", losses, last_step, best_val,
                                     current_stage, mtp_depth, total_tok,
                                     batch_size=batch_size, seq_len=seq_len,
                                     mtp_weight=mtp_weight,
                                     grad_accum_steps=grad_accum_steps,
                                     ema_loss=ema_loss,
                                     best_ema_loss=best_ema_loss,
                                     target_steps=TOTAL_STEPS,
                                     peak_lr=lr)
            save_tokenizer_assets(core)
        raise

    # Final
    elapsed = time.time() - t_start
    fv, fp = eval_model(model, eval_ids, batch_size, seq_len)
    if max_steps is None:
        core.save(str(CKPT_DIR / "final"), losses=losses,
                  metadata_extra={"step": TOTAL_STEPS, "val_loss": fv,
                                 "total_tokens": total_tok,
                                 "total_time_sec": elapsed,
                                 "mtp_depth": mtp_depth,
                                 "batch_size": batch_size,
                                 "seq_len": seq_len,
                                 "mtp_weight": mtp_weight,
                                 "grad_accum_steps": grad_accum_steps,
                                 "ema_loss": ema_loss,
                                 "best_ema_loss": best_ema_loss,
                                 "target_steps": TOTAL_STEPS,
                                 "peak_lr": lr})
        save_tokenizer_assets(core, tag="final")

    print(f"\n  DONE: {TOTAL_STEPS} steps in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"  Final val loss: {fv:.4f} | Best val: {best_val:.4f}")

    print("\n  --- Generation ---")
    for p in FINAL_GENERATION_PROMPTS:
        o = core.generate(p, max_tokens=50, temperature=0.3,
                          top_k=30, top_p=0.85, repetition_penalty=1.2,
                          context_window=256)
        safe = o.encode('ascii', 'replace').decode('ascii')
        print(f"  Q: {p}\n  A: {safe}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NP-DNA.")
    parser.add_argument("--steps", type=int, default=None, help="Run only this many steps for smoke testing.")
    parser.add_argument("--target-steps", type=int, default=DEFAULT_TARGET_STEPS,
                        help="Full training target. Omit --steps to train to this value.")
    parser.add_argument("--lr", type=float, default=LR,
                        help="Peak AdamW learning rate.")
    parser.add_argument("--mtp-depth", type=int, default=MTP_DEPTH, help="Multi-token prediction depth.")
    parser.add_argument("--threads", type=int, default=None, help="PyTorch CPU thread count.")
    parser.add_argument("--compile", action="store_true", help="Try torch.compile for repeated training steps.")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--train-embeddings", action="store_true")
    parser.add_argument("--distill", action="store_true", help="Enable Knowledge Distillation from GPT-2")
    parser.add_argument("--attention", action="store_true", dest="use_attention",
                        default=USE_ATTENTION,
                        help="Use attention strands for fresh-start models.")
    parser.add_argument("--no-attention", action="store_false", dest="use_attention",
                        help="Use faster SSM strands for fresh-start CPU training.")

    parser.add_argument("--seed-chat-ratio", type=float, default=DEFAULT_SEED_CHAT_RATIO,
                        help="Fraction of batches sampled from data/seed_chat.jsonl.")
    parser.add_argument("--seed-ratio-min", type=float, default=DEFAULT_SEED_RATIO_MIN,
                        help="Floor for seed chat ratio after decay.")
    parser.add_argument("--seed-ratio-decay", type=int, default=DEFAULT_SEED_RATIO_DECAY_STEPS,
                        help="Steps over which seed ratio decays from initial to min.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Training batch size. Higher values train more tokens per step but need more RAM.")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN,
                        help="Training sequence length. Higher values train longer context but are slower.")
    parser.add_argument("--seq-preset", choices=["short", "default", "medium", "long"], default=None,
                        help="Named sequence length preset: short=64, default/medium=128, long=192.")
    parser.add_argument("--mtp-weight", type=float, default=MTP_WEIGHT,
                        help="Weight applied to auxiliary MTP loss.")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Accumulate this many micro-batches before each optimizer step.")
    parser.add_argument("--seed-only", action="store_true",
                        help="Train only on data/seed_chat.jsonl for short chat-correction runs.")
    parser.add_argument("--all-folders", action="store_true",
                        help="Sample all dataset folders immediately instead of following curriculum stages.")
    parser.add_argument("--log-every", type=int, default=LOG_EVERY,
                        help="Print training progress every N steps.")
    parser.add_argument("--latest-every", type=int, default=LATEST_EVERY,
                        help="Save model/npdna/latest every N steps. Use 0 to disable.")
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY,
                        help="Run validation/generation every N steps. Use 0 for final eval only.")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY,
                        help="Save milestone step_N checkpoints every N steps. Use 0 to disable.")
    parser.add_argument("--fresh-start", action="store_true",
                        help="Ignore latest/best checkpoints and initialize a new model.")
    parser.add_argument("--reset-step-on-resume", action="store_true",
                        help="Resume latest/best weights but start this training phase from step 1.")
    parser.add_argument("--tokenizer-path", default=None,
                        help="Tokenizer JSON to load for a fresh model, for example model/tokenizer/tokenizer_seed_clean.json.")
    parser.add_argument("--seed-chat-path", default=None,
                        help="Seed chat JSONL file or directory. Defaults to Download/seed.")
    parser.add_argument("--complexity", type=float, default=None,
                        help="Fresh-start model scale. 1.0=hidden 64, 2.0=128, 4.0=256. Ignored when resuming.")
    parser.add_argument("--amp", action="store_true", help="Enable CPU mixed precision (bfloat16) training.")
    parser.add_argument("--no-proportional-mix", action="store_false", dest="proportional_mix",
                        help="Disable proportional stage data mixing.")
    parser.add_argument("--vocab-growth-sample", type=int, default=SEED_VOCAB_SAMPLE_SIZE,
                        help="Seed records sampled for startup BPE vocab improvement. Use 0 to disable.")
    parser.add_argument("--vocab-growth-merges", type=int, default=SEED_VOCAB_MERGE_ROUNDS,
                        help="Maximum BPE merge rounds to add before training starts. Use 0 to disable.")
    parser.add_argument("--target-vocab-size", type=int, default=SEED_TARGET_VOCAB_SIZE,
                        help="Aggressively grow BPE vocabulary toward this many tokens before training.")
    args = parser.parse_args()

    train(
        max_steps=args.steps,
        target_steps=args.target_steps,
        lr=args.lr,
        mtp_depth=args.mtp_depth,
        threads=args.threads,
        compile_model=args.compile,
        freeze_backbone=args.freeze_backbone,
        train_embeddings=args.train_embeddings,
        seed_chat_ratio=args.seed_chat_ratio,
        seed_ratio_min=args.seed_ratio_min,
        seed_ratio_decay_steps=args.seed_ratio_decay,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        mtp_weight=args.mtp_weight,
        grad_accum_steps=args.grad_accum_steps,
        seed_only=args.seed_only,
        log_every=args.log_every,
        save_every=args.save_every,
        eval_every=args.eval_every,
        latest_every=args.latest_every,
        fresh_start=args.fresh_start,
        reset_step_on_resume=args.reset_step_on_resume,
        tokenizer_path=args.tokenizer_path,
        seed_chat_path=args.seed_chat_path,
        seq_preset=args.seq_preset,
        complexity=args.complexity,
        amp=args.amp,
        proportional_mix=args.proportional_mix,
        vocab_growth_sample=args.vocab_growth_sample,
        vocab_growth_merges=args.vocab_growth_merges,
        target_vocab_size=args.target_vocab_size,
        distill=args.distill,
        use_attention=args.use_attention,
        all_folders_now=args.all_folders,
    )
