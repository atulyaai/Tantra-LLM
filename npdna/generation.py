"""Generation mixin for NpDnaCore.

Extracted from model.py to keep that file focused on architecture.
Handles: token sampling, streaming, prompt formatting, cortex write-back.
"""
from __future__ import annotations

import json
import logging
import re
import hashlib
import os
from typing import Generator, Optional

import torch
from torch import Tensor

from .tokenizer import SPECIAL_TOKENS

try:
    from atulya.persona import Persona as _Identity
    _HAS_IDENTITY = True
except ImportError:
    _HAS_IDENTITY = False

logger = logging.getLogger(__name__)

# ── Sampling helpers ──────────────────────────────────────────────────────────

def _apply_repetition_penalty(logits: Tensor, seen_ids: list[int], penalty: float,
                               freq_penalty: float = 0.3) -> Tensor:
    if penalty <= 1.0 and freq_penalty <= 0.0:
        return logits
    logits = logits.clone()
    from collections import Counter
    counts = Counter(seen_ids[-128:])
    for tok_id, count in counts.items():
        if 0 <= tok_id < logits.size(0):
            # Scale penalty by frequency: tokens seen more often get penalized harder
            import math as _math
            effective = penalty + freq_penalty * _math.log1p(count)
            if logits[tok_id] < 0:
                logits[tok_id] = logits[tok_id] * effective
            else:
                logits[tok_id] = logits[tok_id] / effective
    return logits


def _block_ngram_repeats(logits: Tensor, ids: list[int], n: int = 3) -> Tensor:
    """Block exact n-gram repeats to prevent 'things things things' loops."""
    if len(ids) < n:
        return logits
    logits = logits.clone()
    last_ngram = tuple(ids[-(n - 1):])
    for i in range(len(ids) - n):
        if tuple(ids[i:i + n - 1]) == last_ngram:
            blocked_next = ids[i + n - 1]
            if 0 <= blocked_next < logits.size(0):
                logits[blocked_next] = float("-inf")
    return logits


def _apply_top_k(logits: Tensor, k: int) -> Tensor:
    if k <= 0:
        return logits
    topk_vals, topk_idx = torch.topk(logits, min(k, logits.size(0)))
    mask = torch.full_like(logits, float("-inf"))
    mask.scatter_(0, topk_idx, topk_vals)
    return mask


def _apply_top_p(logits: Tensor, p: float) -> Tensor:
    if p >= 1.0:
        return logits
    logits = logits.clone()
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cum_probs > p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    logits[sorted_indices[remove]] = float("-inf")
    return logits


def _build_suppression_mask(
    tokenizer,
    vocab_size: int,
    suppress_bytes: bool,
    suppress_rare_unicode: bool,
    suppress_non_ascii: bool = False,
) -> set[int]:
    """Collect token IDs to permanently suppress during sampling."""
    suppress: set[int] = set(SPECIAL_TOKENS.values())
    if suppress_bytes:
        suppress |= set(tokenizer.byte_to_id.values())
    if suppress_rare_unicode:
        for tok_id, tok in enumerate(tokenizer.id_to_token[:vocab_size]):
            if len(tok) == 1 and ord(tok) > 126:
                is_control = ord(tok) < 32 or (127 <= ord(tok) <= 159)
                is_private = 0xE000 <= ord(tok) <= 0xF8FF
                is_surrogate = 0xD800 <= ord(tok) <= 0xDFFF
                if is_control or is_private or is_surrogate:
                    suppress.add(tok_id)
    if suppress_non_ascii:
        for tok_id, tok in enumerate(tokenizer.id_to_token[:vocab_size]):
            if tok.startswith("<byte_") and tok.endswith(">"):
                continue
            if any(ord(ch) > 126 for ch in tok):
                suppress.add(tok_id)
    return suppress


def _build_chat_prompt(prompt: str, system: Optional[str] = None) -> str:
    """Wrap a bare prompt in the standard chat format."""
    if "Assistant:" in prompt and "User:" in prompt:
        return prompt
    if system is None:
        if _HAS_IDENTITY:
            try:
                system = _Identity().get_system_prompt()
            except Exception:
                system = "You are Atulya. Be warm, thoughtful, and direct."
        else:
            system = "You are Atulya. Be warm, thoughtful, and direct."
    return f"System: {system}\nUser: {prompt.strip()}\nAssistant:"


def _cache_prompt(prompt: str) -> str:
    """Wire the prompt cache into generation without changing model behavior."""
    try:
        from memory.prompt_cache import PromptCacheProvider

        cache = PromptCacheProvider(os.environ.get("ATULYA_DATA_DIR", "assets"))
        key = "prompt_" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cached = cache.get(key)
        if cached is None:
            cache.set(key, prompt)
        return cached or prompt
    except Exception as exc:
        logger.debug("Prompt cache unavailable: %s", exc)
        return prompt


# ── Generation mixin ──────────────────────────────────────────────────────────

class GenerationMixin:
    """
    Provides generate / generate_stream on any class that has:
      - self.model        (NpDnaModel)
      - self.tokenizer    (AtulyaTokenizer)
      - self.cortex       (MemoryCortex)
      - self.encode(text) -> list[int]
      - self.decode(ids)  -> str
      - self.active_path  (Path | None)
    """

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.75,
        top_k: int = 45,
        top_p: float = 1.0,
        repetition_penalty: float = 1.12,
        suppress_byte_tokens: bool = True,
        suppress_rare_unicode: bool = True,
        suppress_non_ascii: bool = False,
        max_token_repeats: int = 6,
        context_window: int = 512,
        audio_inputs: Optional[Tensor] = None,
        image_inputs: Optional[Tensor] = None,
        system: Optional[str] = None,
    ) -> str:
        return "".join(
            self.generate_stream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                suppress_byte_tokens=suppress_byte_tokens,
                suppress_rare_unicode=suppress_rare_unicode,
                suppress_non_ascii=suppress_non_ascii,
                max_token_repeats=max_token_repeats,
                context_window=context_window,
                audio_inputs=audio_inputs,
                image_inputs=image_inputs,
                system=system,
            )
        )

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.75,
        top_k: int = 45,
        top_p: float = 1.0,
        repetition_penalty: float = 1.12,
        suppress_byte_tokens: bool = True,
        suppress_rare_unicode: bool = True,
        suppress_non_ascii: bool = False,
        max_token_repeats: int = 6,
        context_window: int = 512,
        audio_inputs: Optional[Tensor] = None,
        image_inputs: Optional[Tensor] = None,
        system: Optional[str] = None,
    ) -> Generator[str, None, None]:
        original_prompt = prompt
        prompt = _cache_prompt(_build_chat_prompt(prompt, system=system))
        prompt_ids = self.encode(prompt, allow_growth=False)
        ids = list(prompt_ids) or [self.tokenizer.token_to_id.get("<bos>", 2)]
        self.last_prompt_len = len(ids)

        device = self.model.embedding.weight.device
        valid_vocab = min(self.tokenizer.size, self.model.vocab_size)
        eos_id = self.tokenizer.token_to_id.get("<eos>", 3)
        suppress = _build_suppression_mask(
            self.tokenizer,
            valid_vocab,
            suppress_byte_tokens,
            suppress_rare_unicode,
            suppress_non_ascii,
        )
        suppress.discard(eos_id)

        self.model.eval()
        if hasattr(self.model.genome, "enable_inference_cache"):
            self.model.genome.enable_inference_cache()
        with torch.no_grad():
            try:
                stop_buffer = ""
                stop_sequences = ["User:", "\nSystem:", "\n\n\n"]
                # Recompute over the recent full context each step. Local
                # attention strands need the prompt tokens present at inference;
                # a one-token cached path drops their attention context.
                for _ in range(max_tokens):
                    ctx = ids[-max(1, int(context_window)):]
                    input_ids = torch.tensor([ctx], dtype=torch.long, device=device)
                    kwargs = {}
                    if image_inputs is not None:
                        kwargs["multimodal_embeddings"] = image_inputs
                    elif audio_inputs is not None:
                        kwargs["multimodal_embeddings"] = audio_inputs

                    logits, _ = self.model(input_ids=input_ids, **kwargs)
                    next_logits = logits[0, -1].clone()

                    if valid_vocab < next_logits.numel():
                        next_logits[valid_vocab:] = float("-inf")
                    for tok_id in suppress:
                        if tok_id < next_logits.numel():
                            next_logits[tok_id] = float("-inf")
                    if max_token_repeats > 0:
                        recent = ids[-128:]
                        for tok_id in set(recent):
                            if recent.count(tok_id) >= max_token_repeats and tok_id < next_logits.numel():
                                next_logits[tok_id] = float("-inf")

                    next_logits = _apply_repetition_penalty(next_logits, ids, repetition_penalty)
                    next_logits = _block_ngram_repeats(next_logits, ids, n=3)

                    if temperature > 0:
                        next_logits = next_logits / temperature

                    next_logits = _apply_top_k(next_logits, top_k)
                    next_logits = _apply_top_p(next_logits, top_p)

                    probs = torch.softmax(next_logits, dim=-1)
                    if not torch.isfinite(probs).any():
                        next_id = eos_id if eos_id is not None else 0
                    else:
                        next_id = int(torch.multinomial(probs, 1).item())
                    ids.append(next_id)

                    if next_id == eos_id:
                        break

                    token_text = self.decode([next_id])
                    yield token_text

                    # Stop at natural boundaries to prevent runaway generation
                    stop_buffer = (stop_buffer + token_text)[-200:]
                    if any(stop in stop_buffer for stop in stop_sequences):
                        break
            finally:
                if hasattr(self.model.genome, "disable_inference_cache"):
                    self.model.genome.disable_inference_cache()

            self.last_generated_ids = ids
            self._record_strand_specialization(original_prompt)
            self._handle_cortex_writeback(ids[len(prompt_ids):], device)

    def _record_strand_specialization(self, prompt: str) -> None:
        try:
            from tantra.core.task_classifier import TaskClassifier

            if not hasattr(self, "_classifier"):
                self._classifier = TaskClassifier()
            topic = self._classifier.classify(prompt).category.value
            for mesh in self.model.mesh_layers:
                if hasattr(mesh, "record_activation_topic"):
                    mesh.record_activation_topic(topic)
        except Exception as exc:
            logger.debug("Strand specialization tracking skipped: %s", exc)

    # ── Cortex write-back ─────────────────────────────────────────────────────

    def _handle_cortex_writeback(self, generated_ids: list[int], device) -> None:
        generated_text = self.decode(generated_ids)
        matches = re.findall(r"<memory_start>(.*?)<memory_end>", generated_text, re.DOTALL)
        if not matches:
            return
        for fact in (m.strip() for m in matches if m.strip()):
            fact_ids = self.encode(fact, allow_growth=False)
            if not fact_ids:
                continue
            with torch.no_grad():
                embs = self.model.embedding(
                    torch.tensor(fact_ids, dtype=torch.long, device=device)
                )
                vector = embs.mean(dim=0).cpu()
            self.cortex.store(key=vector, value=vector, topic="Active Write-Back", source=fact)

        if self.active_path:
            self.cortex.save(self.active_path / "cortex")
            meta_file = self.active_path / "metadata.json"
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    meta["cortex_entries"] = self.cortex.size
                    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                except Exception as exc:
                    logger.error("Cortex write-back metadata update failed: %s", exc)

    # ── Routing telemetry ────────────────────────────────────────────────────

    def get_routing_telemetry(self) -> list[dict]:
        if not getattr(self, "last_generated_ids", None):
            return []
        self.model.eval()
        with torch.no_grad():
            input_ids = torch.tensor(
                [self.last_generated_ids],
                dtype=torch.long,
                device=self.model.embedding.weight.device,
            )
            self.model(input_ids)

        prompt_len = getattr(self, "last_prompt_len", 0)
        cortex_indices = getattr(self.cortex, "_last_top_indices", None)
        cortex_scores = getattr(self.cortex, "_last_top_scores", None)

        telemetry = []
        for t, tok_id in enumerate(self.last_generated_ids):
            tok_raw = self.tokenizer.id_to_token[tok_id] if tok_id < self.tokenizer.size else f"<unk_{tok_id}>"

            layers_info = []
            for mesh in self.model.mesh_layers:
                top_idx = getattr(mesh, "_last_top_indices", None)
                top_w = getattr(mesh, "_last_top_weights", None)
                layer_routing = []
                if top_idx is not None and top_w is not None and t < top_idx.shape[1]:
                    for k in range(top_idx.shape[2]):
                        local_idx = int(top_idx[0, t, k].item())
                        try:
                            global_id = int(mesh.strands[local_idx].strand_id)
                        except Exception:
                            global_id = -1
                        layer_routing.append({
                            "local_index": local_idx,
                            "strand_id": global_id,
                            "weight": float(top_w[0, t, k].item()),
                        })
                layers_info.append(layer_routing)

            cortex_hits = []
            if cortex_indices is not None and cortex_scores is not None and t < len(cortex_indices):
                for k in range(len(cortex_indices[t])):
                    idx = int(cortex_indices[t][k].item())
                    if 0 <= idx < len(self.cortex.entries):
                        entry = self.cortex.entries[idx]
                        cortex_hits.append({
                            "entry_index": idx,
                            "topic": entry.topic,
                            "source": entry.source,
                            "score": float(cortex_scores[t][k].item()),
                        })

            telemetry.append({
                "token_id": int(tok_id),
                "token_raw": tok_raw,
                "token_clean": self.decode([tok_id]),
                "is_prompt": t < prompt_len,
                "layers": layers_info,
                "cortex": cortex_hits,
            })

        return telemetry
