"""NP-DNA — curriculum by data volume, dynamic vocab, fresh start, optimization utilities."""
from __future__ import annotations

import argparse, statistics, sys, os, json, time, random, math, gc, threading
from pathlib import Path
from copy import deepcopy
from npdna.policy import AdaptiveTrainingPolicy, SelfDistillationTeacher, build_vocab_map
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import nn
import numpy as np

from npdna import NpDnaCore
from npdna.architecture import CONFIGS
from npdna.model import inject_lora, mark_only_lora_trainable

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
USE_ATTENTION = False
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
MAX_NONFINITE_SKIPS = 5

_COLOR_ENABLED = False
_ANSI = {
    "reset": "\033[0m", "bold": "\033[1m", "cyan": "\033[96m",
    "green": "\033[92m", "yellow": "\033[93m", "magenta": "\033[95m",
    "red": "\033[91m", "dim": "\033[2m",
}


def _configure_terminal_color(mode: str) -> None:
    """Enable ANSI decoration only for an interactive capable terminal."""
    global _COLOR_ENABLED
    _COLOR_ENABLED = mode == "always" or (mode == "auto" and sys.stdout.isatty())
    if _COLOR_ENABLED and os.name == "nt":
        try:
            import ctypes
            handle = ctypes.windll.kernel32.GetStdHandle(-11)
            current_mode = ctypes.c_uint32()
            if ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(current_mode)):
                ctypes.windll.kernel32.SetConsoleMode(handle, current_mode.value | 0x0004)
        except Exception:
            pass


def _paint(text: str, color: str, *, bold: bool = False) -> str:
    if not _COLOR_ENABLED:
        return text
    prefix = _ANSI[color] + (_ANSI["bold"] if bold else "")
    return f"{prefix}{text}{_ANSI['reset']}"


def _progress_bar(current: int, total: int, width: int = 24) -> str:
    fraction = min(1.0, max(0.0, current / max(1, total)))
    filled = int(round(width * fraction))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {fraction:6.2%}"

CKPT_DIR = Path("model")
ASSETS_DIR = Path("model/latest")
SEED_CHAT_PATH = Path("Download/seed")
DEFAULT_SEED_CHAT_RATIO = 0.50
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
SEED_TARGET_VOCAB_SIZE = 131072

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
    """Pick varied generation probes so logs test question conditioning."""
    topics = [
        "gravity",
        "photosynthesis",
        "machine learning",
        "clean training data",
        "time management",
        "binary search",
        "emotional intelligence",
        "Chanakya",
        "Python functions",
        "basic arithmetic",
        "study planning",
        "focus",
        "the human heart",
        "GPS",
    ]
    forms = [
        "Explain {topic} simply.",
        "Give a beginner-friendly answer about {topic}.",
        "What should I know about {topic}?",
        "Teach me {topic} with one example.",
        "Summarize {topic} in a focused paragraph.",
    ]
    rng = random.Random(int(step))
    probes = list(GENERATION_PROBE_PROMPTS)
    for _ in range(max(count * 4, 16)):
        probes.append(rng.choice(forms).format(topic=rng.choice(topics)))
    return rng.sample(probes, min(count, len(probes)))

# DATASET_SIZES and all_folders are populated at startup by discover_training_folders().
# Do NOT hardcode these — new folders in Download/ are picked up automatically.
DATASET_SIZES: dict[str, int] = {}
all_folders: list[str] = []

# Folders to skip even if they exist in the data directory
_SKIP_FOLDERS = {"seed", "train_pack", "samples", "archived_before_clean_500k"}


def discover_training_folders(data_dir: Path) -> None:
    """Auto-discover all training folders from the data directory.
    
    Scans data_dir for subdirectories containing .jsonl files,
    excludes reserved folders (seed, train_pack, etc.), sorts by
    dataset size descending so the curriculum learns large domains first.
    """
    global all_folders, DATASET_SIZES
    found: dict[str, int] = {}
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        return
    for folder_path in sorted(data_dir.iterdir()):
        if not folder_path.is_dir():
            continue
        name = folder_path.name
        if name in _SKIP_FOLDERS or name.startswith("."):
            continue
        total_bytes = sum(
            fp.stat().st_size
            for fp in folder_path.rglob("*.jsonl")
            if "archived_before_clean_500k" not in fp.parts
        )
        if total_bytes > 0:
            found[name] = max(1, int(total_bytes / (1024 * 1024)))
    if not found:
        return
    # Sort: larger datasets first (more data = more training time allocated)
    sorted_folders = sorted(found.items(), key=lambda x: -x[1])
    all_folders = [name for name, _ in sorted_folders]
    DATASET_SIZES.update(found)
    print(f"  Auto-discovered {len(all_folders)} training folders: {', '.join(all_folders)}")


def calc_steps(mb, base=200, max_steps=2000):
    if mb == 0:
        return 500        # tiny samples get 500 steps to overfit
    return min(max_steps, base + int((mb / 100) * 50))


def base_curriculum() -> list[dict]:
    stages = []
    cumul = 0
    for i, name in enumerate(all_folders):
        folders = all_folders[:i + 1]
        mb = DATASET_SIZES[name]
        steps = calc_steps(mb, base=200, max_steps=1500)
        cumul += steps
        stages.append({
            "name": name,
            "folders": list(folders),
            "steps": cumul,
            "mb": mb,
        })
    if not stages:
        # Pack-only mode (category folders absent): one default stage whose
        # empty folder list makes get_chunks use the train_pack fallback.
        stages.append({"name": "train_pack", "folders": [], "steps": 2000, "mb": 0})
    return stages


def build_curriculum(target_steps: int) -> list[dict]:
    base = base_curriculum()
    base_total_steps = base[-1]["steps"] if base else 1
    target_steps = max(len(base), int(target_steps))
    scaled = []
    previous = 0
    for idx, stage in enumerate(base):
        if idx == len(base) - 1:
            step_limit = target_steps
        else:
            ratio = stage["steps"] / max(1, base_total_steps)
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


def stage_index_for_step(
    step: int,
    curriculum: list[dict],
    max_stage: int | None = None,
) -> int:
    """Return the scheduled stage, optionally capped to a safe maximum."""
    for idx, stage in enumerate(curriculum):
        if step <= stage["steps"]:
            selected = idx
            break
    else:
        selected = max(0, len(curriculum) - 1)
    if max_stage is not None:
        selected = min(selected, max(0, int(max_stage)))
    return selected


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
    min_lr = peak_lr * 0.02
    return min_lr + (peak_lr - min_lr) * decay


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
            for candidate in (data_dir / "train_pack" / fallback_name, data_dir / fallback_name):
                if candidate.exists():
                    chunks.append(candidate)
                    break
            if chunks:
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
            except (ValueError, TypeError, json.JSONDecodeError):
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
            except (ValueError, TypeError, json.JSONDecodeError):
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
    except (ValueError, TypeError, json.JSONDecodeError):
        pass
    return None


def _load_qa_dir(path):
    """Yield parsed records from all .jsonl files in a directory."""
    path = Path(path)
    if not path.exists():
        # Fallback to train pack if seed path is missing
        for candidate in (
            Path("Download/train_pack/train_pack_core_20k.jsonl"),
            Path("Download/train_pack/train_pack_all_expanded_1040k.jsonl"),
            Path("Download/train_pack_core_20k.jsonl"),
            Path("Download/train_pack_all_expanded_1040k.jsonl"),
        ):
            if candidate.exists():
                path = candidate
                break
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


class TokenMemmapDataset(torch.utils.data.Dataset):
    """Fixed-token memory-mapped windows for frozen-tokenizer training."""

    def __init__(self, path: str | Path, seq_len: int, stride: int | None = None):
        self.path = Path(path)
        self.meta_path = self.path.with_suffix(self.path.suffix + ".json")
        if not self.path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(f"Token memmap and manifest are required: {self.path}")
        self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.seq_len = max(1, int(seq_len))
        self.stride = max(1, int(stride or self.seq_len))
        self._tokens = np.memmap(self.path, mode="r", dtype=np.dtype(self.metadata["dtype"]), shape=(int(self.metadata["token_count"]),))
        self._length = max(0, (len(self._tokens) - self.seq_len - 1) // self.stride + 1)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self._length:
            raise IndexError(index)
        start = index * self.stride
        window = np.asarray(self._tokens[start:start + self.seq_len + 1], dtype=np.int64)
        return {"input_ids": torch.from_numpy(window[:-1].copy()), "labels": torch.from_numpy(window[1:].copy())}


def make_token_memmap_loader(
    path: str | Path,
    *,
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
    stride: int | None = None,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """Create a multiprocessing-safe loader for a frozen uint32 token corpus.

    This deliberately reads only pre-tokenized IDs. Use it only when tokenizer
    growth is disabled, otherwise worker processes could observe different vocabularies.
    """
    dataset = TokenMemmapDataset(path, seq_len=seq_len, stride=stride)
    if not len(dataset):
        raise ValueError(f"Token corpus is too short for seq_len={seq_len}: {path}")
    workers = max(0, int(num_workers))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=False,
    )


def build_token_memmap(texts: Iterable[str], tokenizer, output: str | Path, *, allow_growth: bool = False) -> dict:
    """Tokenize a corpus once into a portable uint32 memory-mapped file."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    token_count = 0
    with temp.open("wb") as stream:
        for text in texts:
            ids = tokenizer.encode(str(text), allow_growth=allow_growth)
            if ids:
                np.asarray(ids, dtype=np.uint32).tofile(stream)
                token_count += len(ids)
    os.replace(temp, output)
    metadata = {"version": 1, "dtype": "uint32", "token_count": token_count, "vocab_size": int(tokenizer.size), "vocab_capacity": int(tokenizer.capacity)}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


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

    def dataset_vocab_texts(self, sample_size: int = SEED_VOCAB_SAMPLE_SIZE) -> Iterable[str]:
        """Yield bounded texts from active dataset chunks for startup vocab growth."""
        if sample_size <= 0 or not self._chunks:
            return
        yielded = 0
        chunks = list(self._chunks)
        random.shuffle(chunks)
        while yielded < sample_size:
            made_progress = False
            for fp in chunks:
                remaining = sample_size - yielded
                if remaining <= 0:
                    return
                for text in sample_texts_from_chunk(fp, max_texts=min(512, remaining)):
                    yield text
                    yielded += 1
                    made_progress = True
                    if yielded >= sample_size:
                        return
            if not made_progress:
                return

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


def _resolve_device(device: str | None = None) -> torch.device:
    choice = (device or "auto").lower()
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        print("  WARNING: --device cuda requested but CUDA is unavailable; using CPU")
        choice = "cpu"
    return torch.device(choice)


def eval_model(model, ids_list, batch_size=4, seq_len=128, device: torch.device | None = None):
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration):
            device = torch.device("cpu")
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
            x = torch.tensor(x_list, dtype=torch.long, device=device)
            y = torch.tensor(y_list, dtype=torch.long, device=device)
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


def _chunked_cross_entropy(
    pred: "torch.Tensor",
    tgt: "torch.Tensor",
    ignore_index: int = IGNORE_INDEX,
    label_smoothing: float = 0.1,
    chunk_size: int = 512,
) -> "torch.Tensor":
    """Compute cross-entropy in chunks to avoid materialising the full [N, V]
    tensor at once.  Peak VRAM is O(chunk_size * V) instead of O(N * V),
    which prevents OOM crashes with large vocabularies (65k-131k tokens).
    """
    flat_pred = pred.reshape(-1, pred.size(-1))   # [N, V]
    flat_tgt  = tgt.reshape(-1)                   # [N]
    total_loss = flat_pred.new_tensor(0.0)
    valid_tokens = 0
    for start in range(0, flat_pred.size(0), chunk_size):
        end = start + chunk_size
        chunk_p = flat_pred[start:end]             # [chunk, V]
        chunk_t = flat_tgt[start:end]              # [chunk]
        mask = chunk_t != ignore_index
        n = mask.sum()
        if n == 0:
            continue
        loss = F.cross_entropy(
            chunk_p,
            chunk_t,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction="sum",
        )
        total_loss = total_loss + loss
        valid_tokens += int(n)
    if valid_tokens == 0:
        return flat_pred.new_tensor(0.0)
    return total_loss / valid_tokens


def mtp_aux_loss(logits, targets, depth: int = MTP_DEPTH) -> torch.Tensor:
    """Auxiliary multi-token prediction loss for offsets 2..depth.
    Uses chunked cross-entropy so large vocabularies never cause OOM."""
    if depth <= 1:
        return logits.new_tensor(0.0)
    seq_len = targets.size(1)
    losses = []
    for offset in range(2, depth + 1):
        if seq_len < offset:
            break
        pred = logits[:, : seq_len - offset + 1, :]
        tgt  = targets[:, offset - 1 :]
        if (tgt != IGNORE_INDEX).any():
            losses.append(_chunked_cross_entropy(pred, tgt))
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def _scalar_loss(loss: torch.Tensor) -> torch.Tensor:
    """Collapse DataParallel's per-device scalar losses before backward."""
    return loss.mean() if loss.ndim > 0 else loss


def _nonfinite_loss_report(**parts) -> str | None:
    bad = []
    for name, value in parts.items():
        if torch.is_tensor(value):
            tensor = value.detach()
            if not torch.isfinite(tensor).all():
                bad.append(f"{name}={float(torch.nanmean(tensor.float())):.4g}")
        elif isinstance(value, (int, float)) and not math.isfinite(float(value)):
            bad.append(f"{name}={float(value):.4g}")
    return ", ".join(bad) if bad else None


class PrefetchLoader:
    """Background-thread data prefetcher.

    While the GPU/CPU processes the current batch, the next batch is
    being prepared on a background thread.  Hides I/O + tokenisation
    latency behind the forward/backward compute.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        seq_len: int,
        allow_growth: bool = True,
        prefetch_size: int = 1,
    ):
        import queue
        self._dataset = dataset
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._allow_growth = allow_growth
        # Note: Uses a threaded queue, not a DataLoader.
        self._queue = queue.Queue(maxsize=max(1, int(prefetch_size)))
        self._error: BaseException | None = None
        self._running = True
        # daemon=True so a worker blocked on a full queue or slow I/O can
        # never prevent the interpreter from exiting after training finishes.
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()

    def __del__(self):
        self.stop()

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
            # Unblock a worker stuck on a full queue so it can observe
            # _running == False and exit.
            self._queue.put_nowait((None, None))
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait((None, None))
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)


def save_tokenizer_assets(core, tag=""):
    name = f"tokenizer{'_'+tag if tag else ''}"
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    core.tokenizer.save(str(ASSETS_DIR / f"{name}.json"))
    torch.save({
        "vocab_size": core.tokenizer.size,
        "capacity": core.tokenizer.capacity,
        "merges": core.tokenizer.merges,
    }, ASSETS_DIR / f"{name}.pt")


def save_training_state(path: Path, opt=None, scaler=None, step=None) -> None:
    """Save optimizer/AMP state next to a model checkpoint for warm resume."""
    if opt is None:
        return
    path.mkdir(parents=True, exist_ok=True)
    payload = {"optimizer": opt.state_dict()}
    if step is not None:
        payload["step"] = int(step)
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    tmp = path / "training_state.pt.tmp"
    torch.save(payload, tmp)
    tmp.replace(path / "training_state.pt")


def load_training_state(
    path: Path, opt, scaler=None, device: torch.device | str = "cpu", expected_step=None
) -> bool:
    """Restore optimizer/AMP state if present; old checkpoints simply skip it.

    If the persisted optimizer step disagrees with the checkpoint's metadata step,
    the stale momentum is discarded (fresh optimizer) so resume stays consistent."""
    state_path = path / "training_state.pt"
    if not state_path.exists():
        return False
    # Optimizer checkpoints are tensors and plain containers; do not execute
    # pickle payloads when resuming a checkpoint supplied from outside.
    state = torch.load(state_path, map_location=device, weights_only=True)
    ts_step = state.get("step")
    if expected_step is not None and ts_step is not None and int(ts_step) != int(expected_step):
        print(
            f"  training_state.pt step ({ts_step}) != checkpoint step ({expected_step}): "
            "discarding stale optimizer momentum for a consistent resume"
        )
        return False
    try:
        opt.load_state_dict(state["optimizer"])
        if scaler is not None and "scaler" in state:
            scaler.load_state_dict(state["scaler"])
    except (ValueError, RuntimeError) as exc:
        print(f"  Optimizer state skipped (model shape changed after vocab resize): {str(exc)[:120]}")
        return False
    return True


def _rmtree_with_retries(path: Path, attempts: int = 12, delay: float = 0.5) -> bool:
    """Remove a directory tree, retrying transient PermissionError (Windows file locks,
    antivirus scanners, or a lingering handle from a just-finished read).
    Returns True if removed (or already absent), False if still locked after retries."""
    import shutil
    import time
    for attempt in range(attempts):
        try:
            if not path.exists():
                return True
            shutil.rmtree(path)
            return True
        except PermissionError as exc:
            if attempt == attempts - 1:
                print(f"  WARNING: could not remove {path} after {attempts} retries: {exc}")
                return False
            time.sleep(delay)
    return False


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
    opt=None,
    scaler=None,
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
    ckpt_path = CKPT_DIR / name
    
    # Keep rolling backups of the 'latest' checkpoint (keep 3 old versions).
    # Best-effort only: a stuck backup (e.g. Windows/AV file lock) must never
    # prevent the current checkpoint from being written.
    if name == "latest":
        import shutil
        max_backups = 3
        for i in range(max_backups - 1, 0, -1):
            src = CKPT_DIR / f"{name}.{i}"
            dst = CKPT_DIR / f"{name}.{i+1}"
            if src.exists():
                if dst.exists() and not _rmtree_with_retries(dst):
                    continue  # dst still busy; leave src in place, avoid nesting
                try:
                    shutil.move(str(src), str(dst))
                except (OSError, shutil.Error) as exc:
                    print(f"  WARNING: backup rotation {src} -> {dst} failed: {exc}")
        if ckpt_path.exists():
            dst = CKPT_DIR / f"{name}.1"
            if dst.exists() and not _rmtree_with_retries(dst):
                pass  # leave the new latest in place; core.save below overwrites it
            else:
                try:
                    shutil.move(str(ckpt_path), str(dst))
                except (OSError, shutil.Error) as exc:
                    print(f"  WARNING: latest backup {ckpt_path} -> {dst} failed: {exc}")

    core.save(str(ckpt_path), losses=losses, metadata_extra=metadata)
    save_training_state(ckpt_path, opt=opt, scaler=scaler, step=metadata.get("step"))

    # Auto-backup to Google Drive (or any path) if env var is set
    # Set NPDNA_BACKUP_DIR=/content/drive/MyDrive/Tantra_Checkpoints before training
    backup_dir = os.environ.get("NPDNA_BACKUP_DIR", "").strip()
    if backup_dir:
        import shutil as _shutil
        try:
            backup_path = Path(backup_dir) / name
            backup_path.mkdir(parents=True, exist_ok=True)
            _shutil.copytree(str(ckpt_path), str(backup_path), dirs_exist_ok=True)
            print(f"  Backup saved: {backup_path}")
        except Exception as _e:
            print(f"  Backup warning: {_e}")


class LionOptimizer(torch.optim.Optimizer):
    """Lion optimizer (Symbolic Discovery of Optimization Algorithms).
    
    Uses sign-gradient updates — 3x more memory efficient than AdamW on CPU.
    Enables larger effective batch sizes at same RAM cost.
    Ref: https://arxiv.org/abs/2302.06675
    """
    def __init__(self, params, lr: float = 1e-4, betas=(0.9, 0.99), weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                # Update step: sign(beta1 * m + (1-beta1) * grad)
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                p.data.add_(update.sign_(), alpha=-group['lr'])
                # Update momentum: beta2 * m + (1-beta2) * grad
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


def train(
    max_steps: int | None = None,
    target_steps: int = DEFAULT_TARGET_STEPS,
    lr: float = LR,
    mtp_depth: int = MTP_DEPTH,
    threads: int | None = None,
    use_compile: bool = False,
    lora_rank: int = 0,
    lora_alpha: float | None = None,
    freeze_backbone: bool = False,
    train_embeddings: bool = False,
    optimizer_name: str = "adamw",
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
    retrain_tokenizer: bool = False,
    data_dir: str | None = None,
    tokenizer_path: str | None = None,
    seed_chat_path: str | None = None,
    seq_preset: str | None = None,
    complexity: float | None = None,
    amp: bool = False,
    proportional_mix: bool = True,
    vocab_growth_sample: int = SEED_VOCAB_SAMPLE_SIZE,
    vocab_growth_merges: int = SEED_VOCAB_MERGE_ROUNDS,
    target_vocab_size: int = SEED_TARGET_VOCAB_SIZE,
    vocab_min_pair_freq: int = 2,
    distill: bool = False,
    teacher_model: str = "gpt2",
    seed: int = 42,
    distill_weight: float = 2.0,
    distill_warmup: int = 0,
    use_attention: bool = USE_ATTENTION,
    all_folders_now: bool = False,
    max_stage: int | None = None,
    device: str | None = None,
    multi_gpu: bool = False,
    freeze_vocab: bool = False,
    prefetch_size: int = 1,
    resume_from: str = "best",
    token_corpus: str | None = None,
    num_workers: int = 0,
    lora_output: str | None = None,
    skip_final_eval: bool = False,
    auto_grow: bool = False,
    growth_patience: int = 2,
    growth_min_delta: float = 0.005,
    growth_strands_before_layer: int = 3,
    unbounded_growth: bool = False,
    grad_checkpoint: bool = False,
    color: str = "auto",
    fusion_ratio: float = 0.0,
    fusion_cache: str | None = None,
    self_distill: bool = False,
    self_distill_decay: float = 0.999,
    self_distill_temperature: float = 2.0,
    self_distill_warmup: int = 200,
):
    global CURRICULUM, TOTAL_STEPS
    _configure_terminal_color(color)
    TOTAL_STEPS = max(1, int(target_steps))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR = Path(data_dir) if data_dir else Path("Download")
    discover_training_folders(DATA_DIR)
    CURRICULUM = build_curriculum(TOTAL_STEPS)
    if max_stage is not None:
        max_stage = max(0, min(int(max_stage), len(CURRICULUM) - 1))
    if all_folders and max_stage is not None:
        raise ValueError("--all-folders and --max-stage cannot be used together")
    if threads:
        torch.set_num_threads(max(1, threads))
    device_obj = _resolve_device(device)
    if device_obj.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif device_obj.type == "cuda":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.bfloat16
    scaler = torch.amp.GradScaler(
        device=device_obj.type,
        enabled=bool(amp and device_obj.type == "cuda" and amp_dtype == torch.float16),
    )

    print(_paint(f"  NP-DNA TRAINING | {CONFIG_NAME} | device={device_obj} | attention={use_attention}", "cyan", bold=True))
    if device_obj.type == "cuda":
        print(f"  CUDA: {torch.cuda.get_device_name(device_obj)}")
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
    retrain_tokenizer = bool(retrain_tokenizer)
    freeze_vocab = bool(freeze_vocab)
    prefetch_size = max(1, int(prefetch_size))
    vocab_min_pair_freq = max(1, int(vocab_min_pair_freq))
    train_allow_growth = not freeze_vocab
    if token_corpus and train_allow_growth:
        raise ValueError("--token-corpus requires --freeze-vocab because worker processes use frozen token IDs")
    tokenizer_path_obj = Path(tokenizer_path) if tokenizer_path else None
    seed_chat_path_obj = Path(seed_chat_path) if seed_chat_path else SEED_CHAT_PATH

    print(f"  {TOTAL_STEPS} planned steps, batch={batch_size}, seq={seq_len}, "
          f"mtp_depth={mtp_depth}, seed_chat_ratio={seed_chat_ratio:.2f} "
          f"(decay->{seed_ratio_min:.2f} over {seed_ratio_decay_steps} steps), "
          f"lr={lr:.2e}, mtp_weight={mtp_weight:.2f}, grad_accum={grad_accum_steps}, "
          f"seed_only={seed_only}, all_folders={all_folders_now}, max_stage={max_stage}, "
          f"latest_every={latest_every}, "
          f"eval_every={eval_every}, save_every={save_every}, "
          f"fresh_start={fresh_start}, reset_step_on_resume={reset_step_on_resume}, "
          f"retrain_tokenizer={retrain_tokenizer}, "
          f"freeze_vocab={freeze_vocab}, prefetch_size={prefetch_size}, "
          f"vocab_min_pair_freq={vocab_min_pair_freq}, "
          f"data_dir={DATA_DIR}, seed_chat_path={seed_chat_path_obj}")
    warmup_first_lr = lr / max(WARMUP_STEPS, 1)
    if lr < 1e-4:
        print(f"  WARNING: peak lr={lr:.2e} is very small; "
              f"warmup step 1 will be {warmup_first_lr:.2e}. "
              "Use --lr 1e-3 or --lr 5e-4 for the larger smoke run unless you intend a slow fine-tune.")
    print_curriculum(CURRICULUM, TOTAL_STEPS)

    base_cfg = deepcopy(CONFIGS[CONFIG_NAME])
    if complexity is not None:
        from npdna.architecture import NpDnaConfig
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

    resume_from = str(resume_from or "best").lower()
    if resume_from not in {"latest", "best"}:
        raise ValueError("--resume-from must be 'latest' or 'best'")
    resume_dir = CKPT_DIR / resume_from
    if not resume_dir.exists():
        fallback = "latest" if resume_from == "best" else "best"
        resume_dir = CKPT_DIR / fallback

    if resume_dir.exists() and not fresh_start:
        core = NpDnaCore.load(str(resume_dir))
        meta = json.loads((resume_dir / "metadata.json").read_text())
        if "target_steps" in meta and int(target_steps) == DEFAULT_TARGET_STEPS:
            TOTAL_STEPS = max(1, int(meta["target_steps"]))
            CURRICULUM = build_curriculum(TOTAL_STEPS)
            print(f"  Restored target_steps={TOTAL_STEPS} from checkpoint metadata")
        loaded_step = meta.get("step", 0)
        start_step = 1 if reset_step_on_resume else loaded_step + 1
        current_stage = stage_index_for_step(start_step - 1, CURRICULUM, max_stage)
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
        tok_files = [] if retrain_tokenizer else ([tokenizer_path_obj] if tokenizer_path_obj else sorted(ASSETS_DIR.glob("tokenizer*.json")))
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
            if not bpe_texts:
                raise FileNotFoundError(
                    f"No training text found under {DATA_DIR!s}. Expected category folders "
                    f"({', '.join(all_folders)}) or train_pack/*.jsonl. On Kaggle, pass "
                    "--data-dir /kaggle/input/<your-dataset-folder> or copy/link data into Download/."
                )
            print(f"  BPE on {len(bpe_texts):,} texts")
            initial_target_merges = max(8000, target_vocab_size - core.tokenizer.size)
            core.tokenizer.train_bpe(
                bpe_texts,
                target_merges=initial_target_merges,
                min_pair_freq=vocab_min_pair_freq,
            )
            core.model.resize_embeddings(core.tokenizer.capacity)
            print(f"  Vocab: {core.tokenizer.size} tokens, cap={core.tokenizer.capacity}")
            save_tokenizer_assets(core)

    print(f"\n  Vocab: {core.tokenizer.size} tokens, cap={core.tokenizer.capacity}, "
          f"fill={core.tokenizer.fill_ratio:.1%}")

    # Dataset
    current_stage = stage_index_for_step(start_step - 1, CURRICULUM, max_stage)
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
    if not seed_only and dataset.chunk_count == 0 and len(dataset.seed_chat_records) == 0:
        raise FileNotFoundError(
            f"No dataset chunks found under {DATA_DIR!s}. Expected JSONL files in category folders "
            f"({', '.join(dataset_folders)}) or train_pack/train_pack_*.jsonl."
        )
    vocab_growth_sample = max(0, int(vocab_growth_sample))
    vocab_growth_merges = max(0, int(vocab_growth_merges))
    target_vocab_size = max(0, int(target_vocab_size))
    _embedding_resized = False
    if freeze_vocab:
        print(
            f"  Vocab pre-growth skipped: --freeze-vocab is set "
            f"(size={core.tokenizer.size}, cap={core.tokenizer.capacity}, "
            f"fill={core.tokenizer.fill_ratio:.1%})"
        )
    elif target_vocab_size and core.tokenizer.size >= target_vocab_size:
        print(
            f"  Vocab pre-growth skipped: size {core.tokenizer.size} "
            f">= target_vocab={target_vocab_size:,}"
        )
    elif vocab_growth_sample and vocab_growth_merges:
        old_vocab_size = core.tokenizer.size
        old_vocab_cap = core.tokenizer.capacity
        if target_vocab_size > old_vocab_size:
            vocab_growth_merges = max(vocab_growth_merges, target_vocab_size - old_vocab_size)
        if len(dataset.seed_chat_records) and dataset.chunk_count:
            # Blend seed + dataset chunks: seed alone misses code/factual/reasoning tokens
            import itertools
            half = max(1, vocab_growth_sample // 2)
            vocab_texts = itertools.chain(
                dataset.seed_vocab_texts(half),
                dataset.dataset_vocab_texts(half),
            )
            vocab_source = "seed records + dataset chunks"
        elif len(dataset.seed_chat_records):
            vocab_texts = dataset.seed_vocab_texts(vocab_growth_sample)
            vocab_source = "seed records"
        else:
            vocab_texts = dataset.dataset_vocab_texts(vocab_growth_sample)
            vocab_source = "dataset chunks"
        print(
            f"  Vocab pre-growth: sampling up to {vocab_growth_sample:,} {vocab_source}, "
            f"target_vocab={target_vocab_size:,}, max_merges={vocab_growth_merges:,}..."
        )
        added_vocab, vocab_stats = core.tokenizer.dynamic_vocab_growth(
            vocab_texts,
            sample_size=vocab_growth_sample,
            merge_rounds=vocab_growth_merges,
            min_pair_freq=vocab_min_pair_freq,
            target_vocab_size=target_vocab_size,
            return_stats=True,
        )
        if core.tokenizer.fill_ratio >= core.tokenizer.growth_threshold:
            reserve_capacity = math.ceil(core.tokenizer.size / 0.75)
            core.tokenizer.ensure_capacity(reserve_capacity)
        if core.tokenizer.capacity != old_vocab_cap:
            core.model.resize_embeddings(core.tokenizer.capacity)
            _embedding_resized = True
        if core.tokenizer.size != old_vocab_size or core.tokenizer.capacity != old_vocab_cap:
            dataset.note_vocab_changed()
            save_tokenizer_assets(core)
        print(
            f"  Vocab pre-growth: sampled={vocab_stats['sampled_texts']:,}, "
            f"target_merges={vocab_stats['target_merges']:,}, "
            f"forced_tokens={vocab_stats.get('forced_tokens', 0):,}, "
            f"size {old_vocab_size} -> {core.tokenizer.size} "
            f"(+{added_vocab}), cap {old_vocab_cap} -> {core.tokenizer.capacity}, "
            f"fill={core.tokenizer.fill_ratio:.1%}"
        )
        # Always sync expanded tokenizer into best checkpoint so resuming
        # from --resume-from best never loses vocabulary growth
        if added_vocab > 0:
            import shutil
            best_ckpt = CKPT_DIR / "best"
            if best_ckpt.exists():
                shutil.copy2(
                    str(ASSETS_DIR / "tokenizer.json"),
                    str(best_ckpt / "tokenizer.json")
                )
                print("  Tokenizer synced to best checkpoint.")
    eval_ids = dataset.eval_set(num_samples=2000)
    print(f"  Seed chat: {len(dataset.seed_chat)} examples")
    print(f"  Eval: {len(eval_ids)} sequences from held-out local/seed data")

    # Optimizer
    model = core.model
    model.to(device_obj)
    model.config.growth_unbounded = bool(unbounded_growth)
    if unbounded_growth:
        print("  capacity growth: no software strand cap (host resources remain the safety bound)")
    core.cortex = core.model.cortex
    lora_rank = max(0, int(lora_rank))
    if lora_rank:
        replaced = inject_lora(model, rank=lora_rank, alpha=lora_alpha)
        if not replaced:
            raise RuntimeError("No linear layers matched the LoRA target modules")
        trainable = mark_only_lora_trainable(model)
        print(f"  LoRA: rank={lora_rank}, modules={len(replaced)}, trainable={trainable:,}")
    if use_compile and hasattr(torch, 'compile'):
        # Check if C++ compiler (cl.exe on Windows, gcc/clang on Linux) is available
        import shutil
        compiler_ok = True
        if sys.platform == "win32":
            compiler_ok = shutil.which("cl.exe") is not None
            if not compiler_ok:
                print("[Compile] MSVC compiler (cl.exe) not found in PATH. Skipping torch.compile on Windows.")
        
        if compiler_ok:
            try:
                model = torch.compile(model, mode='default')
                print('[Compile] torch.compile enabled (default CPU mode)')
            except Exception as e:
                print(f'[Compile] torch.compile failed: {e}')

    if freeze_backbone and not lora_rank:
        from npdna.brain import freeze_for_partial_training

        trainable = freeze_for_partial_training(
            core,
            train_strands=True,
            train_embeddings=train_embeddings,
        )
        print(f"  partial training enabled: {trainable:,} trainable params")

    # LoRA-only training freezes the whole genome.  Cache its generated,
    # detached direct weights once; this is invalidated by strand evolution.
    if not any(parameter.requires_grad for parameter in model.genome.parameters()):
        model.genome.enable_frozen_weight_cache()
        print("  frozen genome: direct-weight cache enabled")

    if multi_gpu and device_obj.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"  DataParallel enabled on {torch.cuda.device_count()} GPUs")
    elif multi_gpu:
        print("  DataParallel skipped: requires CUDA and more than one GPU")

    if grad_checkpoint:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("  Gradient checkpointing enabled")
        else:
            print("  Gradient checkpointing not supported by model")

    def _unwrap_model():
        return model.module if hasattr(model, "module") else getattr(model, "_orig_mod", model)

    # Multimodal projectors are trainable only in --fusion mode; otherwise we
    # exclude them from the optimizer to save memory (they stay frozen/dormant).
    _base_model = _unwrap_model()
    _excluded_ids = set()
    _enable_fusion = float(fusion_ratio or 0.0) > 0.0
    if not _enable_fusion:
        for _proj_name in ("vision_projector", "audio_projector"):
            _proj = getattr(_base_model, _proj_name, None)
            if _proj is not None:
                for _p in _proj.parameters():
                    _excluded_ids.add(id(_p))
                    _p.requires_grad = False
    _train_params = [p for p in model.parameters() if p.requires_grad and id(p) not in _excluded_ids]

    decay_params = [p for p in _train_params if p.dim() >= 2]
    nodecay_params = [p for p in _train_params if p.dim() < 2]

    _opt_name = optimizer_name.lower()
    if _opt_name == "lion":
        opt = LionOptimizer([
            {"params": decay_params, "weight_decay": 0.01},
            {"params": nodecay_params, "weight_decay": 0.0}
        ], lr=lr)
        print(f"  Optimizer: Lion (memory-efficient sign-gradient updates)")
    else:
        opt = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": 0.01},
            {"params": nodecay_params, "weight_decay": 0.0}
        ], lr=lr)
        print(f"  Optimizer: AdamW")

    def _resync_optimizer_params() -> None:
        """Refresh optimizer params after mid-training module/parameter growth."""
        fresh_params = [
            p for p in model.parameters()
            if p.requires_grad and id(p) not in _excluded_ids
        ]
        fresh_decay = [p for p in fresh_params if p.dim() >= 2]
        fresh_nodecay = [p for p in fresh_params if p.dim() < 2]
        if len(opt.param_groups) >= 2:
            opt.param_groups[0]["params"] = fresh_decay
            opt.param_groups[1]["params"] = fresh_nodecay
        else:
            opt.param_groups[0]["params"] = fresh_params

    _vocab_was_resized = _embedding_resized
    if _vocab_was_resized:
        print("  Optimizer state skipped: vocab/embedding was resized, starting fresh optimizer.")
    elif start_step > 1 and not reset_step_on_resume:
        if load_training_state(resume_dir, opt, scaler=scaler, device=device_obj, expected_step=loaded_step):
            print(f"  Restored optimizer state from {resume_dir / 'training_state.pt'}")

    losses = []
    best_val = float('inf')
    t_start = time.time()
    smooth_loss = 0.0
    ema_loss = None
    best_ema_loss = float("inf")
    total_tok = 0

    if distill:
        try:
            teacher = DistillationTeacher(model_name=teacher_model, device=device_obj.type,
                                          student_tokenizer=core.tokenizer)
        except ImportError as e:
            print(f"  [Distill] transformers not installed; skipping distillation ({e}).")
            teacher = None
        except (OSError, RuntimeError) as e:
            print(f"  [Distill] GPT-2 teacher unavailable (network/Hub error); "
                  f"continuing without distillation:\n    {e}")
            teacher = None
    else:
        teacher = None

    self_distill_teacher: SelfDistillationTeacher | None = None
    if self_distill:
        self_distill_teacher = SelfDistillationTeacher(
            model, decay=self_distill_decay,
            temperature=self_distill_temperature, warmup_steps=self_distill_warmup,
        )
        print(f"  [Self-Distill] EMA shadow teacher enabled "
              f"(decay={self_distill_decay}, T={self_distill_temperature}, "
              f"warmup={self_distill_warmup} steps).")

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
    growth_controller = DynamicGrowthController(
        patience=growth_patience,
        min_delta=growth_min_delta,
        max_strands_per_layer=None if unbounded_growth else None,
        strands_before_layer=growth_strands_before_layer,
    )
    growth_controller.best_val_loss = best_val
    if auto_grow and use_compile:
        print("  Capacity growth disabled with torch.compile because compiled graphs cannot safely absorb new modules.")
        auto_grow = False
    if auto_grow and (lora_rank or freeze_backbone or multi_gpu):
        print("  Capacity growth disabled for LoRA/partial training/DataParallel because new strand weights would be unfrozen.")
        auto_grow = False
    consecutive_nonfinite = 0

    end_step = TOTAL_STEPS if max_steps is None else min(TOTAL_STEPS, start_step + max_steps - 1)
    stage_display_max = max_stage if max_stage is not None else len(CURRICULUM) - 1
    print(f"\n  Stage {current_stage}/{stage_display_max} ({dataset.chunk_count} chunks)")
    if max_steps is not None:
        print(f"  Smoke run: steps {start_step}-{end_step}\n")
    else:
        print()

    last_step = start_step - 1
    token_loader = None
    token_iterator = None
    if token_corpus:
        token_loader = make_token_memmap_loader(
            token_corpus, seq_len=seq_len, batch_size=batch_size,
            num_workers=num_workers,
        )
        token_iterator = iter(token_loader)
        prefetcher = None
        print(f"  Token corpus: {token_corpus} (workers={num_workers})")
    else:
        prefetcher = PrefetchLoader(
            dataset, batch_size, seq_len,
            allow_growth=train_allow_growth,
            prefetch_size=prefetch_size,
        )
        prefetcher.update_step(start_step)

    # Unified multimodal fusion dataset (vision/audio embeddings → text)
    _fusion_ds = None
    _fusion_rng = random.Random(int(os.environ.get("PYTHONHASHSEED", "0")))
    if _enable_fusion:
        try:
            from npdna.fusion import MultimodalDataset as _MMD
        except Exception:
            _MMD = None
        if _MMD is not None:
            if fusion_cache and Path(fusion_cache).is_dir():
                _fusion_ds = _MMD(cache_dir=fusion_cache)
                print(f"  Fusion mode: cache={fusion_cache} ({len(_fusion_ds)} samples)")
            else:
                _vdim = getattr(getattr(core.model, "vision_projector", None), "in_features", 4096)
                _adim = getattr(getattr(core.model, "audio_projector", None), "in_features", 4096)
                _voc = getattr(core.tokenizer, "vocab_size", None) or getattr(core.tokenizer, "capacity", 131308)
                _fusion_ds = _MMD.generate_synthetic(
                    num_samples=64, vision_dim=int(_vdim), audio_dim=int(_adim), target_dim=int(_voc)
                )
                print("  Fusion mode: synthetic vision/audio embeddings "
                      f"(projector dims {int(_vdim)}/{int(_adim)}, vocab {int(_voc)}; "
                      "pass --fusion-cache for real precomputed embeddings)")
        else:
            print("  WARNING: --fusion requested but MultimodalDataset import failed; ignoring")
        if _fusion_ds is not None:
            print(f"  Training built-in projectors jointly (fusion_ratio={float(fusion_ratio):.2f})")

    try:
        for step in range(start_step, end_step + 1):
            last_step = step
            # LR schedule — set at top of loop for all steps
            step_lr = scheduled_lr(step, lr, TOTAL_STEPS)
            for g in opt.param_groups:
                g['lr'] = step_lr

            # Curriculum stage switch
            new_stage = stage_index_for_step(step, CURRICULUM, max_stage)

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
                print(f"\n  >>> Stage {current_stage}/{stage_display_max} ({dataset.chunk_count} chunks) <<<\n")

                # Grow vocab if needed at stage transitions
                if train_allow_growth and core.tokenizer.fill_ratio > 0.9:
                    old_cap = core.tokenizer.capacity
                    old_size = core.tokenizer.size
                    more_texts = []
                    for fp in random.sample(dataset._chunks,
                                             min(3, len(dataset._chunks))):
                        more_texts.extend(load_texts(fp, max_lines=1000))
                    new_target = len(core.tokenizer.merges) + 500
                    core.tokenizer.train_bpe(more_texts, target_merges=new_target,
                                              min_pair_freq=2)
                    core.model.resize_embeddings(core.tokenizer.capacity)
                    core.model.to(device_obj)
                    _resync_optimizer_params()
                    if core.tokenizer.capacity > old_cap:
                        print(f"  Vocab grew: {old_cap} -> {core.tokenizer.capacity} "
                              f"(size={core.tokenizer.size})")
                        save_tokenizer_assets(core)
                        # Sync updated tokenizer into best checkpoint so
                        # resuming from best never loses vocabulary progress
                        best_ckpt = CKPT_DIR / "best"
                        if best_ckpt.exists():
                            import shutil
                            shutil.copy2(
                                str(ASSETS_DIR / "tokenizer.json"),
                                str(best_ckpt / "tokenizer.json")
                            )
                    if core.tokenizer.size != old_size or core.tokenizer.capacity != old_cap:
                        dataset.note_vocab_changed()

                eval_ids = dataset.eval_set(num_samples=2000)

                # Reset prefetcher to load from new folders
                if prefetcher is not None:
                    prefetcher.stop()
                if token_loader is None:
                    prefetcher = PrefetchLoader(
                        dataset, batch_size, seq_len,
                        allow_growth=train_allow_growth,
                        prefetch_size=prefetch_size,
                    )
                    prefetcher.update_step(step)

            # (LR schedule is now set at the top of the loop)

            model.train()
            opt.zero_grad(set_to_none=True)
            ce_parts = []
            mtp_parts = []
            skip_step_reason = None
            for micro_i in range(grad_accum_steps):
                if token_iterator is not None:
                    try:
                        batch = next(token_iterator)
                    except StopIteration:
                        token_iterator = iter(token_loader)
                        batch = next(token_iterator)
                    x, y = batch["input_ids"], batch["labels"]
                else:
                    x, y = prefetcher.get()
                    prefetcher.update_step(step)
                # Dynamic vocab growth can outgrow the model embedding table
                # between stage-transition resize calls. Resize just before
                # this batch touches the model.
                if train_allow_growth and core.tokenizer.size > core.model.vocab_size:
                    core.model.resize_embeddings(core.tokenizer.capacity)
                    core.model.to(device_obj)
                    _resync_optimizer_params()
                total_tok += x.numel()
                x = x.to(device_obj, non_blocking=True)
                y = y.to(device_obj, non_blocking=True)
                _emb = None
                _modality = None
                if _enable_fusion and _fusion_ds is not None:
                    if _fusion_rng.random() < fusion_ratio:
                        _b = x.size(0)
                        _s = _fusion_ds[_fusion_rng.randrange(len(_fusion_ds))]
                        _mm = _base_model
                        _v = _s.get("vision_embeds")
                        if _v is not None and getattr(_mm, "vision_projector", None) is not None:
                            _src = _v.detach().to(x.device, dtype=torch.float32)
                            _emb = _src.unsqueeze(0).expand(_b, 1, -1).contiguous()
                            _modality = "vision"
                        else:
                            _au = _s.get("audio_embeds")
                            if _au is not None and getattr(_mm, "audio_projector", None) is not None:
                                _src = _au.detach().to(x.device, dtype=torch.float32)
                                _emb = _src.unsqueeze(0).expand(_b, 1, -1).contiguous()
                                _modality = "audio"
                        if _emb is not None:
                            # Prepending a fused token makes logits length seq+1;
                            # prepend an ignored label so it adds no CE mass.
                            y = torch.cat([y.new_full((_b, 1), IGNORE_INDEX), y], dim=1)
                with torch.amp.autocast(device_type=device_obj.type, dtype=amp_dtype, enabled=amp):
                    logits, bal = model(x, multimodal_embeddings=_emb, modality=_modality)
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
                    base_model = _unwrap_model()
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
                    if teacher is not None and step >= distill_warmup:
                        # Decode input_ids to text, get teacher logits
                        with torch.no_grad():
                            texts = [core.decode(x[b].tolist()) for b in range(x.size(0))]
                            t_logits = teacher.get_teacher_logits(texts, x)
                            if t_logits is not None:
                                t_logits = t_logits.to(logits.device)
                        if t_logits is not None:
                            distill_loss = compute_distillation_loss(logits, t_logits)
                    elif (self_distill_teacher is not None
                          and self_distill_teacher.active(step)):
                        distill_loss = self_distill_teacher.distill(
                            logits, x, lambda m, inp: m(inp)
                        )

                    loss = ce_loss + (mtp_weight * mtp_loss) + bal * 0.1 + exit_loss * 0.5 + distill_loss * distill_weight
                    loss = _scalar_loss(loss)
                    skip_step_reason = _nonfinite_loss_report(
                        ce=ce_loss,
                        mtp=mtp_loss,
                        balance=bal,
                        exit=exit_loss,
                        distill=distill_loss,
                        total=loss,
                    )
                if skip_step_reason:
                    break
                scaler.scale(loss / grad_accum_steps).backward()
                ce_parts.append(ce_loss.detach())
                mtp_parts.append(mtp_loss.detach())
            if skip_step_reason:
                consecutive_nonfinite += 1
                opt.zero_grad(set_to_none=True)
                print(
                    f"  WARNING: non-finite loss at step {step}, micro {micro_i}: "
                    f"{skip_step_reason}; skipping optimizer step"
                )
                if consecutive_nonfinite >= MAX_NONFINITE_SKIPS:
                    raise RuntimeError(
                        f"Training produced non-finite losses for {consecutive_nonfinite} "
                        f"consecutive steps; last report: {skip_step_reason}"
                    )
                continue
            ce_loss = torch.stack(ce_parts).mean()
            mtp_loss = torch.stack(mtp_parts).mean()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                consecutive_nonfinite += 1
                opt.zero_grad(set_to_none=True)
                print(
                    f"  WARNING: non-finite grad norm at step {step}: "
                    f"{float(grad_norm):.4g}; skipping optimizer step"
                )
                if consecutive_nonfinite >= MAX_NONFINITE_SKIPS:
                    raise RuntimeError(
                        f"Training produced non-finite gradients for {consecutive_nonfinite} "
                        f"consecutive steps; last grad_norm={float(grad_norm):.4g}"
                    )
                continue
            scaler.step(opt)
            scaler.update()
            consecutive_nonfinite = 0

            if self_distill_teacher is not None:
                self_distill_teacher.step(model)

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
                    _resync_optimizer_params()
            # Cortex write: store hidden states every 1000 steps
            if step > WARMUP_STEPS and step % 1000 == 0:
                model.eval()
                with torch.no_grad():
                    base_model = _unwrap_model()
                    sample_x = x[:2, :32]  # small sample from current batch
                    h = base_model.embedding(sample_x)
                    for mesh, norm in zip(base_model.mesh_layers, base_model.layer_norms):
                        out, _ = mesh(h)
                        h = norm(h + out)
                    h = base_model.final_norm(h)
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
                status = (
                    f"  {_progress_bar(step, TOTAL_STEPS)}  step {step:,}/{TOTAL_STEPS:,}  "
                    f"stage {current_stage:02d}  loss {smooth_loss:.3f}  ema {ema_loss:.3f}  "
                    f"mtp {float(mtp_loss.detach()):.3f}  best {best:.3f}\n"
                    f"  rate {rate:.0f} tok/s  lr {cur_lr:.2e}  seed {dataset.seed_chat_ratio:.0%}  "
                    f"ETA {format_duration(eta)}"
                )
                print(_paint(status, "green", bold=True))

            if latest_every and step % latest_every == 0:
                save_training_checkpoint(core, "latest", losses, step, best_val,
                                         current_stage, mtp_depth, total_tok,
                                         batch_size=batch_size, seq_len=seq_len,
                                         mtp_weight=mtp_weight,
                                         grad_accum_steps=grad_accum_steps,
                                         ema_loss=ema_loss,
                                         best_ema_loss=best_ema_loss,
                                         target_steps=TOTAL_STEPS,
                                         peak_lr=lr,
                                         opt=opt,
                                         scaler=scaler)

            # Eval
            force_eval = not skip_final_eval and max_steps is not None and step == end_step
            if not skip_final_eval and ((eval_every and step % eval_every == 0) or force_eval):
                vl, vp = eval_model(model, eval_ids, batch_size, seq_len, device_obj)
                gen = core.generate("Hello.", max_tokens=20, temperature=0.3,
                                    top_k=30, top_p=0.85, repetition_penalty=1.2,
                                    context_window=256)
                safe = gen.encode('ascii', 'replace').decode('ascii')
                print(_paint(f"  VALIDATION | loss={vl:.4f} | ppl={vp:.1f} | sample: {safe[:80]}", "magenta", bold=True))
                previous_best_val = best_val
                improved = vl < previous_best_val
                if improved:
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
                    save_training_state(CKPT_DIR / "best", opt=opt, scaler=scaler)
                    save_tokenizer_assets(core)

                if auto_grow:
                    try:
                        growth_controller.evaluate_checkpoint(step, vl, core, opt)
                        _resync_optimizer_params()
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        print(f"  [GrowthController] stopped: {exc}")
                        auto_grow = False

            # Generation check every 1000 steps
            if not skip_final_eval and (step % 1000 == 0 or step == start_step):
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
                save_training_state(CKPT_DIR / f"step_{step}", opt=opt, scaler=scaler)

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
                                     peak_lr=lr,
                                     opt=opt,
                                     scaler=scaler)
            save_tokenizer_assets(core)
        raise

    # Final
    elapsed = time.time() - t_start
    if skip_final_eval:
        print("  Smoke run complete; skipped final evaluation and generation.")
        return
    fv, fp = eval_model(model, eval_ids, batch_size, seq_len, device_obj)
    if fv < best_val:
        best_val = fv
        core.save(str(CKPT_DIR / "best"), losses=losses,
                  metadata_extra={"step": last_step, "val_loss": fv,
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
        save_training_state(CKPT_DIR / "best", opt=opt, scaler=scaler)
        save_tokenizer_assets(core)
    if max_steps is None:
        core.save(str(CKPT_DIR / "final"), losses=losses,
                  metadata_extra={"step": last_step, "val_loss": fv,
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
        save_training_state(CKPT_DIR / "final", opt=opt, scaler=scaler)
        save_tokenizer_assets(core, tag="final")
        if lora_rank:
            from npdna.model import save_lora_adapter  # noqa: local import for clarity
            adapter_path = Path(lora_output) if lora_output else CKPT_DIR / "final" / "lora_adapter.pt"
            save_lora_adapter(_unwrap_model(), adapter_path)
            print(f"  LoRA adapter saved: {adapter_path}")

    steps_executed = max(0, last_step - start_step + 1)
    print(f"\n  DONE: {steps_executed} steps executed "
          f"({start_step}-{last_step}) in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"  Final val loss: {fv:.4f} | Best val: {best_val:.4f}")

    print("\n  --- Generation ---")
    for p in FINAL_GENERATION_PROMPTS:
        o = core.generate(p, max_tokens=50, temperature=0.3,
                          top_k=30, top_p=0.85, repetition_penalty=1.2,
                          context_window=256)
        safe = o.encode('ascii', 'replace').decode('ascii')
        print(f"  Q: {p}\n  A: {safe}\n")


# ── Dynamic Growth Controller (from growth_controller.py) ──────────────────

import logging
from dataclasses import dataclass as _dataclass

_logger = logging.getLogger(__name__)


@_dataclass
class GrowthEvent:
    step: int
    val_loss_before: float
    old_param_count: int
    new_param_count: int
    action: str


class DynamicGrowthController:
    """Automated growth controller for NP-DNA models."""

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.005,
        max_strands_per_layer: int | None = None,
        growth_step_strands: int = 1,
        strands_before_layer: int = 3,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.max_strands_per_layer = max_strands_per_layer
        self.growth_step_strands = max(1, growth_step_strands)
        self.strands_before_layer = max(1, strands_before_layer)

        self.best_val_loss = float("inf")
        self.no_improve_count = 0
        self.growth_stage_count = 0
        self.growth_events: list[GrowthEvent] = []

    def evaluate_checkpoint(
        self,
        step: int,
        val_loss: float,
        core,
        opt: torch.optim.Optimizer | None = None,
    ) -> bool:
        relative_gain = (self.best_val_loss - val_loss) / max(abs(self.best_val_loss), 1e-8)
        if val_loss < self.best_val_loss and (
            not torch.isfinite(torch.tensor(self.best_val_loss)) or relative_gain >= self.min_delta
        ):
            self.best_val_loss = val_loss
            self.no_improve_count = 0
            return False

        self.no_improve_count += 1
        _logger.info(
            "[GrowthController] Validation loss plateau (%d/%d): val_loss=%.4f (best=%.4f)",
            self.no_improve_count,
            self.patience,
            val_loss,
            self.best_val_loss,
        )

        if self.no_improve_count >= self.patience:
            self.no_improve_count = 0
            return self.trigger_growth(step, val_loss, core, opt)

        return False

    def trigger_growth(
        self,
        step: int,
        val_loss: float,
        core,
        opt: torch.optim.Optimizer | None = None,
    ) -> bool:
        model = core.model
        old_params = model.parameter_count()

        mesh_layers = getattr(model, "mesh_layers", None)
        if not mesh_layers:
            _logger.warning("[GrowthController] Model does not have mesh_layers. Cannot expand strands.")
            return False

        before_trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        grow_layer = (self.growth_stage_count + 1) % self.strands_before_layer == 0
        strands_per_layer = sum(len(getattr(m, "strands", [])) for m in mesh_layers) // len(mesh_layers)
        can_grow_strands = self.max_strands_per_layer is None or strands_per_layer < self.max_strands_per_layer
        if not grow_layer and can_grow_strands and hasattr(model, "grow_strands"):
            model.grow_strands(self.growth_step_strands)
            action_msg = f"Grew {self.growth_step_strands} strands per layer"
        elif hasattr(model, "add_layer"):
            reference = getattr(model, "layer_specs", [None])[-1]
            num_s = getattr(reference, "num_strands", 2)
            top_k = getattr(reference, "top_k", 1)
            model.add_layer(name=f"grown_{len(mesh_layers)}", num_strands=num_s, top_k=top_k)
            action_msg = f"Added layer 'grown_{len(mesh_layers)}' with {num_s} strands"
        new_params = model.parameter_count()
        if new_params <= old_params and action_msg.startswith("Grew") and hasattr(model, "add_layer"):
            reference = getattr(model, "layer_specs", [None])[-1]
            model.add_layer(
                name=f"grown_{len(model.mesh_layers)}",
                num_strands=getattr(reference, "num_strands", 2),
                top_k=getattr(reference, "top_k", 1),
            )
            action_msg = f"Added layer 'grown_{len(model.mesh_layers) - 1}' after strand capacity was reached"
            new_params = model.parameter_count()
        if new_params <= old_params:
            _logger.warning("[GrowthController] Growth request added no parameters; stopping this event.")
            return False
        if opt is not None:
            new_trainable = [
                parameter for parameter in model.parameters()
                if parameter.requires_grad and id(parameter) not in before_trainable
            ]
            if new_trainable:
                new_decay = [p for p in new_trainable if p.dim() >= 2]
                new_nodecay = [p for p in new_trainable if p.dim() < 2]
                if new_decay:
                    opt.add_param_group({"params": new_decay, "weight_decay": 0.01})
                if new_nodecay:
                    opt.add_param_group({"params": new_nodecay, "weight_decay": 0.0})
        self.growth_stage_count += 1

        event = GrowthEvent(
            step=step,
            val_loss_before=val_loss,
            old_param_count=old_params,
            new_param_count=new_params,
            action=action_msg,
        )
        self.growth_events.append(event)

        print(
            f"\n🌱 [GrowthController] Validation plateaued at {val_loss:.4f}! "
            f"Auto-expanded model: {old_params:,} → {new_params:,} params ({action_msg})"
        )

        return True


# ── Optimization Utilities (from optimization.py) ──────────────────────────

# ── CPU Benchmark (from cpu_benchmark.py) ───────────────────────────────────

def _mean_ms(values: list[float]) -> float:
    return 1000.0 * statistics.mean(values) if values else 0.0


def _run_step(
    model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor, *, record: bool,
    precision: str, inference_only: bool,
) -> dict[str, float]:
    timings: dict[str, float] = {}
    if not inference_only:
        model.zero_grad(set_to_none=True)
    forward_started = time.perf_counter()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "bf16"):
        if inference_only:
            with torch.no_grad():
                logits, _ = model(inputs, timings=timings if record else None)
        else:
            logits, _ = model(inputs, timings=timings if record else None)
    forward_seconds = time.perf_counter() - forward_started
    backward_seconds = 0.0
    if not inference_only:
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
        backward_started = time.perf_counter()
        loss.backward()
        backward_seconds = time.perf_counter() - backward_started
    if record:
        timings["forward"] = forward_seconds
        timings["backward"] = backward_seconds
    return timings


def benchmark(
    *, batches: list[int], seq_lens: list[int], warmup: int, iterations: int,
    threads: int | None, lora_rank: int, precision: str, inference_only: bool, dynamic_int8: bool,
    direct_weights: bool,
) -> list[dict[str, float | int]]:
    if threads:
        torch.set_num_threads(max(1, threads))
    torch.set_default_dtype(torch.float32)
    core = NpDnaCore.from_config("seed")
    model = core.model.to(device="cpu", dtype=torch.float32)
    if lora_rank:
        inject_lora(model, rank=lora_rank)
        mark_only_lora_trainable(model)
        model.genome.enable_frozen_weight_cache(direct_write=direct_weights)
    if dynamic_int8:
        if not inference_only:
            raise ValueError("Dynamic INT8 is inference-only; use --inference-only")
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.train(not inference_only)

    print(
        f"CPU {precision.upper()} | threads={torch.get_num_threads()} | warmup={warmup} | "
        f"iterations={iterations} | precision={precision} | inference_only={inference_only} | "
        f"dynamic_int8={dynamic_int8} | lora_rank={lora_rank} | "
        f"frozen_genome_cache={model.genome._frozen_weight_cache} | direct_weights={direct_weights}"
    )
    rows: list[dict[str, float | int]] = []
    for batch in batches:
        for seq_len in seq_lens:
            inputs = torch.randint(0, model.vocab_size, (batch, seq_len), dtype=torch.long)
            targets = torch.randint(0, model.vocab_size, (batch, seq_len), dtype=torch.long)
            for _ in range(warmup):
                _run_step(model, inputs, targets, record=False, precision=precision, inference_only=inference_only)
            samples = [
                _run_step(model, inputs, targets, record=True, precision=precision, inference_only=inference_only)
                for _ in range(iterations)
            ]
            row: dict[str, float | int] = {"batch": batch, "seq_len": seq_len}
            for phase in ("forward", "backward", "genome", "routing", "output_head"):
                row[f"{phase}_ms"] = _mean_ms([sample.get(phase, 0.0) for sample in samples])
            elapsed_ms = row["forward_ms"] if inference_only else row["forward_ms"] + row["backward_ms"]
            row["tokens_per_second"] = batch * seq_len / (elapsed_ms / 1000.0)
            rows.append(row)
            print(
                "batch={batch} seq={seq_len} | fwd={forward_ms:.1f} ms | bwd={backward_ms:.1f} ms | "
                "genome={genome_ms:.1f} ms | routing={routing_ms:.1f} ms | head={output_head_ms:.1f} ms | "
                "{tokens_per_second:.0f} tok/s".format(**row)
            )
    return rows


def benchmark_main() -> None:
    parser = argparse.ArgumentParser(description="Warm CPU FP32 NP-DNA training benchmark")
    parser.add_argument("--batches", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[192, 256])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=0,
                        help="LoRA rank for benchmark (default 0 to match default training); use --lora-rank 8 to benchmark LoRA-on.")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--dynamic-int8", action="store_true")
    parser.add_argument("--no-direct-weights", action="store_false", dest="direct_weights")
    parser.set_defaults(direct_weights=True)
    args = parser.parse_args()
    benchmark(
        batches=args.batches, seq_lens=args.seq_lens, warmup=max(0, args.warmup),
        iterations=max(1, args.iterations), threads=args.threads, lora_rank=max(0, args.lora_rank),
        precision=args.precision, inference_only=args.inference_only, dynamic_int8=args.dynamic_int8,
        direct_weights=args.direct_weights,
    )


# ── Knowledge Distillation (from distill.py) ───────────────────────────────

class DistillationTeacher:
    """Wrapper for a pre-trained teacher model (e.g., GPT-2 Small)."""

    def __init__(self, model_name: str = "gpt2", device: str = "cpu",
                 student_tokenizer=None):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Please install transformers to use distillation: pip install transformers")

        self.device = device
        self.student_tokenizer = student_tokenizer
        self._stu2tea: torch.Tensor | None = None
        self._disabled = False
        print(f"  [Distill] Loading teacher model: {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if len(self.tokenizer) != self.model.get_input_embeddings().num_embeddings:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def get_teacher_logits(self, text_batch: list[str],
                           student_input_ids: torch.Tensor | None = None) -> torch.Tensor | None:
        if self._disabled:
            return None
        inputs = self.tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        tea_logits = outputs.logits
        if self.student_tokenizer is not None and student_input_ids is not None:
            tea_logits = self._project_to_student(tea_logits, student_input_ids)
            if tea_logits is None:
                return None
        return tea_logits

    def _project_to_student(self, tea_logits: torch.Tensor,
                            student_input_ids: torch.Tensor) -> torch.Tensor | None:
        if self._stu2tea is None:
            self._stu2tea = build_vocab_map(self.student_tokenizer, self.tokenizer).to(tea_logits.device)
            overlap = (self._stu2tea >= 0).float().mean().item()
            if overlap < 0.9:
                print(f"  [Distill] WARNING: teacher/student token overlap only {overlap:.1%}; "
                      f"external distillation needs a shared vocabulary (positions align only "
                      f"with the same tokenizer). Disabling external-teacher distillation.")
                self._disabled = True
                return None
        T = min(tea_logits.size(1), student_input_ids.size(1))
        idx = self._stu2tea.clamp(min=0)
        cols = torch.index_select(tea_logits[:, :T, :], 2, idx)
        mask = (self._stu2tea != -1).to(tea_logits.dtype).reshape(1, 1, -1)
        return cols * mask


def compute_distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """Compute KL-divergence loss between student and teacher logits."""
    min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
    min_seq = min(student_logits.size(1), teacher_logits.size(1))

    s_logits = student_logits[:, :min_seq, :min_vocab] / temperature
    t_logits = teacher_logits[:, :min_seq, :min_vocab] / temperature

    s_log_probs = F.log_softmax(s_logits, dim=-1)
    t_probs = F.softmax(t_logits, dim=-1)

    loss = F.kl_div(s_log_probs, t_probs, reduction='batchmean') * (temperature ** 2)
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NP-DNA.")
    parser.add_argument("--steps", type=int, default=None, help="Run only this many steps for smoke testing.")
    parser.add_argument("--target-steps", type=int, default=DEFAULT_TARGET_STEPS,
                        help="Full training target. Omit --steps to train to this value.")
    parser.add_argument("--lr", type=float, default=LR,
                        help="Peak AdamW learning rate.")
    parser.add_argument("--mtp-depth", type=int, default=MTP_DEPTH, help="Multi-token prediction depth.")
    parser.add_argument("--threads", type=int, default=None, help="PyTorch CPU thread count.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Training device. auto uses CUDA when available.")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use torch.nn.DataParallel when multiple CUDA GPUs are visible.")
    parser.add_argument("--freeze-vocab", action="store_true",
                        help="Disable tokenizer growth during batch prefetch/training after startup vocab setup.")
    parser.add_argument("--prefetch-size", type=int, default=1,
                        help="Number of prepared batches to queue ahead.")
    parser.add_argument("--token-corpus", default=None,
                        help="Pre-tokenized uint32 corpus path; requires --freeze-vocab.")
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="Multiprocessing workers for --token-corpus (default: min(4, os.cpu_count())).")
    parser.add_argument("--lora-output", default=None,
                        help="Where to save the final LoRA adapter (default: final checkpoint folder).")
    parser.add_argument("--skip-final-eval", action="store_true",
                        help="Skip final evaluation/generation; useful for fast training smoke tests.")
    parser.add_argument('--compile', action='store_true', default=False, help='Enable torch.compile for ~15-30% speedup after warmup')
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=0,
                        help="Train low-rank adapters only; 0 disables LoRA.")
    parser.add_argument("--lora-alpha", type=float, default=None,
                        help="LoRA scaling factor; defaults to --lora-rank.")
    parser.add_argument("--train-embeddings", action="store_true")
    parser.add_argument("--distill", action="store_true", help="Enable Knowledge Distillation from GPT-2")
    parser.add_argument("--teacher-model", type=str, default="gpt2",
                         help="HF teacher model id for --distill (e.g. gpt2, distilgpt2, "
                              "Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B). On CPU prefer gpt2/distilgpt2.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible fresh-start training (enables controlled A/B runs).")
    parser.add_argument("--distill-weight", type=float, default=2.0,
                        help="Multiplier on the distillation loss (external teacher AND self-distill).")
    parser.add_argument("--distill-warmup", type=int, default=0,
                        help="Steps before the external-teacher distillation loss activates.")
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
    parser.add_argument("--max-stage", type=int, default=None,
                        help="Cap curriculum progression at this zero-based stage (for example, 2 keeps instruction+code+chat).")
    parser.add_argument("--log-every", type=int, default=LOG_EVERY,
                        help="Print training progress every N steps.")
    parser.add_argument("--latest-every", type=int, default=LATEST_EVERY,
                        help="Save model/latest every N steps. Use 0 to disable.")
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY,
                        help="Run validation/generation every N steps. Use 0 for final eval only.")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY,
                        help="Save milestone step_N checkpoints every N steps. Use 0 to disable.")
    parser.add_argument("--fresh-start", action="store_true",
                        help="Ignore latest/best checkpoints and initialize a new model.")
    parser.add_argument("--resume-from", choices=["latest", "best"], default="best",
                        help="Checkpoint slot to resume when not using --fresh-start.")
    parser.add_argument("--reset-step-on-resume", action="store_true",
                        help="Resume latest/best weights but start this training phase from step 1.")
    parser.add_argument("--retrain-tokenizer", action="store_true",
                        help="Ignore saved tokenizer assets and train a fresh tokenizer from local data.")
    parser.add_argument("--data-dir", default=None,
                        help="Training data root. Defaults to Download. On Kaggle, point this at /kaggle/input/<dataset>.")
    parser.add_argument("--tokenizer-path", default=None,
                        help="Tokenizer JSON to load for a fresh model, for example model/latest/tokenizer_seed_clean.json.")
    parser.add_argument("--seed-chat-path", default=None,
                        help="Seed chat JSONL file or directory. Defaults to Download/seed.")
    parser.add_argument("--complexity", type=float, default=None,
                        help="Fresh-start model scale. 1.0=hidden 64, 2.0=128, 4.0=256. Ignored when resuming.")
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Enable mixed precision. FP32 is the default CPU baseline.")
    parser.add_argument("--no-amp", action="store_false", dest="amp",
                        help="Disable autocast; recommended on CPUs without native BF16 support.")
    parser.add_argument("--no-proportional-mix", action="store_false", dest="proportional_mix",
                        help="Disable proportional stage data mixing.")
    parser.add_argument("--vocab-growth-sample", type=int, default=SEED_VOCAB_SAMPLE_SIZE,
                        help="Seed records sampled for startup BPE vocab improvement. Use 0 to disable.")
    parser.add_argument("--vocab-growth-merges", type=int, default=SEED_VOCAB_MERGE_ROUNDS,
                        help="Maximum BPE merge rounds to add before training starts. Use 0 to disable.")
    parser.add_argument("--target-vocab-size", type=int, default=SEED_TARGET_VOCAB_SIZE,
                        help="Aggressively grow BPE vocabulary toward this many tokens before training.")
    parser.add_argument("--vocab-min-pair-freq", type=int, default=2,
                        help="Minimum BPE pair frequency for startup vocab growth. Use 1 to force a larger vocab.")
    parser.add_argument("--optimizer", choices=["adamw", "lion"], default="adamw",
                        help="Optimizer: adamw (default, stable) or lion (3x memory-efficient, sign-gradient).")
    parser.add_argument("--auto-grow", action="store_true",
                        help="Grow capacity after validation loss plateaus.")
    parser.add_argument("--growth-patience", type=int, default=2,
                        help="Consecutive validation plateaus before a growth stage.")
    parser.add_argument("--growth-min-delta", type=float, default=0.005,
                        help="Minimum relative validation-loss gain that resets growth patience.")
    parser.add_argument("--growth-strands-before-layer", type=int, default=3,
                        help="Add this many strand-growth stages before adding a layer.")
    parser.add_argument("--unbounded-growth", action="store_true",
                        help="Remove software strand caps; RAM/CPU remain the physical limit.")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Enable gradient checkpointing to save memory (slower).")
    parser.add_argument("--color", choices=["auto", "always", "never"], default="auto",
                        help="Terminal color mode. auto colors interactive terminals only.")

    parser.add_argument("--fusion", action="store_true",
                        help="Train the built-in vision/audio projectors jointly with the text LM "
                             "(unified multimodal+text training, single checkpoint).")
    parser.add_argument("--fusion-ratio", type=float, default=0.1,
                        help="Fraction of micro-batches that carry multimodal embeddings (0-1).")
    parser.add_argument("--fusion-cache", default=None,
                        help="Directory of precomputed vision/audio embedding .pt files "
                             "(falls back to synthetic data if omitted).")

    parser.add_argument("--self-distill", action="store_true",
                        help="Train from an EMA shadow of the model on its own past, "
                             "smoothed self via KL self-distillation "
                             "(train-less-but-smarter; no external teacher).")
    parser.add_argument("--sd-decay", type=float, default=0.999,
                        help="EMA shadow decay (default 0.999).")
    parser.add_argument("--sd-temp", type=float, default=2.0,
                        help="Softmax temperature for self-distillation KL.")
    parser.add_argument("--sd-warmup", type=int, default=200,
                        help="Steps before the self-distillation loss activates.")

    args = parser.parse_args()
    train(
        max_steps=args.steps,
        target_steps=args.target_steps,
        lr=args.lr,
        mtp_depth=args.mtp_depth,
        threads=args.threads,
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
        retrain_tokenizer=args.retrain_tokenizer,
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer_path,
        seed_chat_path=args.seed_chat_path,
        seq_preset=args.seq_preset,
        complexity=args.complexity,
        amp=args.amp,
        proportional_mix=args.proportional_mix,
        vocab_growth_sample=args.vocab_growth_sample,
        vocab_growth_merges=args.vocab_growth_merges,
        target_vocab_size=args.target_vocab_size,
        vocab_min_pair_freq=args.vocab_min_pair_freq,
        distill=args.distill,
        teacher_model=args.teacher_model,
        seed=args.seed,
        distill_weight=args.distill_weight,
        distill_warmup=args.distill_warmup,
        use_attention=args.use_attention,
        all_folders_now=args.all_folders,
        max_stage=args.max_stage,
        device=args.device,
        multi_gpu=args.multi_gpu,
        freeze_vocab=args.freeze_vocab,
        prefetch_size=args.prefetch_size,
        resume_from=args.resume_from,
        use_compile=args.compile,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_output=args.lora_output,
        token_corpus=args.token_corpus,
        num_workers=args.num_workers,
        skip_final_eval=args.skip_final_eval,
        optimizer_name=args.optimizer,
        auto_grow=args.auto_grow,
        growth_patience=args.growth_patience,
        growth_min_delta=args.growth_min_delta,
        growth_strands_before_layer=args.growth_strands_before_layer,
        unbounded_growth=args.unbounded_growth,
        grad_checkpoint=args.grad_checkpoint,
        color=args.color,
        fusion_ratio=args.fusion_ratio if args.fusion else 0.0,
        fusion_cache=args.fusion_cache,
        self_distill=args.self_distill,
        self_distill_decay=args.sd_decay,
        self_distill_temperature=args.sd_temp,
        self_distill_warmup=args.sd_warmup,
    )
