"""
Tantra/prepare_dataset.py — Offline dataset preparation for Tantra-LLM.

Moves TokenJuice work out of the training loop and into a one-time pass over
the data, so `Tantra/train.py` no longer pays random.random()/tokenization
overhead every step, and so identity/logic synthetic examples get
deterministic, even coverage instead of ~35%-chance-per-step roulette.

What this does, once, over the whole dataset:
  1. Dedup — drops exact-duplicate lines (hash of the formatted prompt).
  2. Squeeze — for RAW/unstructured lines only (not curated chat items),
     runs TokenJuiceEngine.squeeze_tokens() to drop low-entropy repetitive
     chunks (filler/boilerplate), so every token kept is higher-signal.
     Chat-formatted {system,user,assistant}/messages items are left intact
     — squeezing inside a curated Q&A pair can silently mangle the answer.
  3. Enrich — interleaves the built-in identity/logic synthetic pairs at a
     fixed interval (every --enrich-every items) instead of randomly, so
     coverage is even and reproducible.
  4. Writes a new JSONL your existing JSONLDataset loader can read directly
     (loss masking + doc-boundary EOS insertion already happen there).

Usage:
    python Tantra/prepare_dataset.py --input Datasets/my_data.jsonl \
        --output Datasets/my_data.clean.jsonl --enrich-every 200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Tantra.dataset import format_jsonl_prompt, build_prompt_segments
from Tantra.tokenjuice import TokenJuiceEngine
from Tantra.utils import get_logger

log = get_logger("tantra.prepare_dataset")

SYNTHETIC_QA_PAIRS = [
    {"system": "", "user": "What is Tantra?", "assistant": "Tantra is a CPU-First Autonomous AI Engine."},
    {"system": "", "user": "Who created Tantra?", "assistant": "Tantra LLM is created by the Tantra Engineering Team."},
    {"system": "", "user": "Explain artificial intelligence.", "assistant": "AI is the simulation of human intelligence by computer systems."},
]


def _line_hash(text: str) -> str:
    return hashlib.sha1(text.strip().lower().encode("utf-8", errors="ignore")).hexdigest()


def _process_lines_batch(lines_batch: list[str], tokenizer, all_tokens: list[int], all_masks: list[bool]):
    from Tantra.dataset import build_prompt_segments, format_jsonl_prompt, _encode
    items_segments = []
    flat_texts = []
    
    for line in lines_batch:
        try:
            item = json.loads(line)
            segments = build_prompt_segments(item)
            if segments is None:
                text = format_jsonl_prompt(item)
                segments = [(text, True)] if text else None
        except Exception:
            segments = [(line, True)]

        if not segments:
            items_segments.append([])
            continue

        item_segs = []
        for text, target in segments:
            if text:
                item_segs.append((len(flat_texts), target))
                flat_texts.append(text)
        items_segments.append(item_segs)

    if not flat_texts:
        return

    # Call ultra-fast multi-threaded C++ BPE batch encoder
    if hasattr(tokenizer, "bpe") and hasattr(tokenizer.bpe, "encode_batch"):
        encoded_flat = tokenizer.bpe.encode_batch(flat_texts)
    else:
        encoded_flat = [_encode(tokenizer, t) for t in flat_texts]

    for item_segs in items_segments:
        if not item_segs:
            continue
        item_ids = []
        item_targets = []
        for idx, target in item_segs:
            seg_ids = encoded_flat[idx]
            item_ids.extend(seg_ids)
            item_targets.extend([target] * len(seg_ids))

        if item_ids:
            all_tokens.extend(item_ids)
            all_masks.extend(item_targets)
            all_tokens.append(2)  # EOS
            all_masks.append(True)


def prepare(input_path: str, output_path: str, tokenizer, enrich_every: int = 200,
            entropy_threshold: float = 0.3, dedup: bool = True, dump_bin: bool = False) -> None:
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError(
            f"Input and output paths cannot be identical ({input_path!r}). "
            f"Opening output file for writing truncates input before reading. "
            f"Please specify a distinct output path e.g. --output Datasets/master_clean.jsonl"
        )
    juice = TokenJuiceEngine(entropy_threshold=entropy_threshold, enrichment_rate=1.0)

    seen_hashes = set()
    n_in = 0
    n_out = 0
    n_dupes = 0
    n_squeezed = 0
    tokens_before = 0
    tokens_after = 0
    n_synthetic = 0

    try:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
        has_rich = True
    except ImportError:
        has_rich = False

    # Count total input lines if possible for accurate progress bar
    total_lines = None
    try:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f_count:
            total_lines = sum(1 for line in f_count if line.strip())
    except Exception:
        total_lines = None

    if has_rich:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Preparing Dataset[/bold cyan]"),
            BarColumn(bar_width=35),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        )
        task_id = progress.add_task("Prepare", total=total_lines)
        progress.start()
    else:
        progress = None

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for raw_line in f_in:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            n_in += 1
            if progress:
                progress.update(task_id, advance=1)

            try:
                item = json.loads(raw_line)
            except Exception:
                item = {"user": raw_line, "assistant": ""}

            text = format_jsonl_prompt(item) if isinstance(item, dict) else raw_line
            if not text:
                continue

            if dedup:
                h = _line_hash(text)
                if h in seen_hashes:
                    n_dupes += 1
                    continue
                seen_hashes.add(h)

            is_chat = isinstance(item, dict) and build_prompt_segments(item) is not None

            if is_chat:
                # Leave curated chat pairs untouched — squeeze only raw text.
                out_item = item
            else:
                ids = tokenizer.encode(text)
                tokens_before += len(ids)
                squeezed_ids = juice.squeeze_tokens(ids)
                tokens_after += len(squeezed_ids)
                if len(squeezed_ids) < len(ids):
                    n_squeezed += 1
                squeezed_text = tokenizer.decode(squeezed_ids) if hasattr(tokenizer, "decode") else text
                out_item = {"user": squeezed_text, "assistant": ""} if squeezed_text.strip() else None

            if out_item is None:
                continue

            f_out.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            n_out += 1

            if enrich_every > 0 and n_out % enrich_every == 0:
                synth = SYNTHETIC_QA_PAIRS[(n_out // enrich_every - 1) % len(SYNTHETIC_QA_PAIRS)]
                f_out.write(json.dumps(synth, ensure_ascii=False) + "\n")
                n_out += 1
                n_synthetic += 1

    if progress:
        progress.stop()

    log.info("Dataset preparation complete.")
    log.info(f"  Input lines        : {n_in:,}")
    log.info(f"  Output lines       : {n_out:,}")
    log.info(f"  Duplicates dropped : {n_dupes:,}")
    log.info(f"  Raw lines squeezed : {n_squeezed:,}")
    if tokens_before:
        pct = 100.0 * (1 - tokens_after / max(tokens_before, 1))
        log.info(f"  Raw-text tokens    : {tokens_before:,} -> {tokens_after:,} ({pct:.1f}% reduction)")
    log.info(f"  Synthetic pairs added: {n_synthetic:,} (every {enrich_every} items)")
    log.info(f"  Written -> {output_path}")

    if dump_bin:
        bin_path = output_path.rsplit(".", 1)[0] + ".bin"
        log.info(f"Exporting pre-tokenized binary cache in parallel to: {bin_path}...")
        import torch
        
        all_tokens = []
        all_masks = []
        total_target = max(n_out, 1)
        batch_size = 4096

        if has_rich:
            bin_progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold magenta]Fast Parallel Binary Export[/bold magenta]"),
                BarColumn(bar_width=35),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
            )
            bin_task = bin_progress.add_task("BinExport", total=total_target)
            bin_progress.start()
        else:
            bin_progress = None

        with open(output_path, "r", encoding="utf-8", errors="ignore") as f:
            lines_batch = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                lines_batch.append(line)
                if len(lines_batch) >= batch_size:
                    _process_lines_batch(lines_batch, tokenizer, all_tokens, all_masks)
                    if bin_progress:
                        bin_progress.update(bin_task, advance=len(lines_batch))
                    lines_batch = []
            if lines_batch:
                _process_lines_batch(lines_batch, tokenizer, all_tokens, all_masks)
                if bin_progress:
                    bin_progress.update(bin_task, advance=len(lines_batch))

        if bin_progress:
            bin_progress.stop()

        if all_tokens:
            t_tensor = torch.tensor(all_tokens, dtype=torch.int32)
            m_tensor = torch.tensor(all_masks, dtype=torch.bool)
            torch.save({"tokens": t_tensor, "masks": m_tensor}, bin_path)
            log.info(f"  Binary Cache Export Complete: {len(all_tokens):,} tokens -> {bin_path} ({os.path.getsize(bin_path)/1e6:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Offline TokenJuice dataset preparation for Tantra-LLM.")
    parser.add_argument("--input", type=str, required=True, help="Source JSONL dataset")
    parser.add_argument("--output", type=str, required=True, help="Cleaned/enriched JSONL output path")
    parser.add_argument("--enrich-every", type=int, default=200, help="Insert one synthetic pair every N items (0 = disabled)")
    parser.add_argument("--entropy-threshold", type=float, default=0.3, help="Squeeze threshold for raw-text chunk entropy")
    parser.add_argument("--no-dedup", action="store_true", help="Disable exact-duplicate removal")
    parser.add_argument("--dump-bin", action="store_true", help="Dump pre-tokenized binary token-id cache (.bin)")
    parser.add_argument("--tokenizer-corpus", type=str, default=None,
                         help="Optional corpus file to (re)train the BPE tokenizer on before preparing data")
    args = parser.parse_args()

    from Tantra.config import VocabConfig
    from main import build_vocab

    vcfg = VocabConfig()
    tokenizer = build_vocab(vcfg, args.tokenizer_corpus or args.input)

    prepare(
        input_path=args.input,
        output_path=args.output,
        tokenizer=tokenizer,
        enrich_every=args.enrich_every,
        entropy_threshold=args.entropy_threshold,
        dedup=not args.no_dedup,
        dump_bin=args.dump_bin,
    )


if __name__ == "__main__":
    main()
