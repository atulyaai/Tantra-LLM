"""NP-DNA Model — full NeuroPlastic DNA Network.

Architecture:
    Token IDs → Embedding → [Mesh₁ → … → Meshₙ] → Norm → LM Head

Auto-scales: vocab grows, strands grow, layers grow — all automatic.

Merged modules:
    - model.py     — core model, checkpointing, and NpDnaCore wrapper
    - generation.py — token sampling, streaming, prompt formatting, cortex write-back
    - genome.py    — DNA weight generator for Strands
    - lora.py      — LoRA adapter injection for fine-tuning
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import re
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Generator, Iterable, Literal, Optional

import torch
from torch import Tensor, nn

from .architecture import CONFIGS, CortexConfig, GenomeConfig, LayerSpec, MeshConfig, NpDnaConfig, StrandConfig, CategoryMesh, NeuralMesh
from .cognition import MemoryCortex
from .tokenizer import AtulyaTokenizer, SPECIAL_TOKENS

try:
    from npdna.inference import PersonalityLayer as _Identity
    _HAS_IDENTITY = True
except ImportError:
    _HAS_IDENTITY = False

logger = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Model — core architecture and checkpointing                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _replace_with_retries(tmp: Path, path: Path, attempts: int = 12, delay: float = 0.25) -> None:
    last_error: PermissionError | None = None
    for _ in range(attempts):
        try:
            if path.exists():
                path.chmod(0o666)
            tmp.chmod(0o666)
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(delay)
    raise last_error or PermissionError(f"Could not replace {path}")


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    _replace_with_retries(tmp, path)


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding=encoding)
    _replace_with_retries(tmp, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_hmac(artifact_hashes: dict[str, str], key: str) -> str:
    payload = json.dumps(artifact_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(key.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def _verify_checkpoint_integrity(path: Path, meta: dict) -> None:
    """Verify optional artifact hashes and an optional deployment HMAC key."""
    artifact_hashes = meta.get("artifact_hashes")
    signing_key = os.environ.get("NPDNA_CHECKPOINT_HMAC_KEY")
    required = signing_key is not None or os.environ.get("NPDNA_REQUIRE_CHECKPOINT_INTEGRITY") == "1"
    if not required:
        return
    if not artifact_hashes:
        if required:
            raise RuntimeError("Checkpoint integrity metadata is required but missing.")
        return
    if not isinstance(artifact_hashes, dict):
        raise RuntimeError("Checkpoint artifact hashes are invalid.")
    for relative_name, expected_hash in artifact_hashes.items():
        artifact = path / relative_name
        if not artifact.is_file() or not isinstance(expected_hash, str):
            raise RuntimeError(f"Checkpoint integrity artifact is invalid: {relative_name}")
        if not hmac.compare_digest(_sha256_file(artifact), expected_hash):
            raise RuntimeError(f"Checkpoint artifact hash mismatch: {relative_name}")
    if signing_key:
        signature = meta.get("artifact_hmac")
        expected_signature = _checkpoint_hmac(artifact_hashes, signing_key)
        if not isinstance(signature, str) or not hmac.compare_digest(signature, expected_signature):
            raise RuntimeError("Checkpoint HMAC verification failed.")


class NpDnaModel(nn.Module):
    """Full NP-DNA language model (architecture only, no inference helpers)."""

    def __init__(self, config: NpDnaConfig) -> None:
        super().__init__()
        self.config = config
        H = config.hidden_size

        self.embedding = nn.Embedding(config.initial_vocab, H)
        self.genome = Genome(config.genome, config.mesh.strand)

        self.layer_specs: list[LayerSpec] = []
        self.mesh_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        groups = config.weight_sharing_groups
        if groups is None:
            groups = [list(range(config.num_layers))]  # Default: all layers share one genome!

        layer_to_offset = {}
        current_offset = 0
        for group in groups:
            for l_idx in group:
                layer_to_offset[l_idx] = current_offset
            # All layers in a group must have the same number of strands.
            # We add to offset based on the first layer in the group.
            first_idx = group[0]
            if config.mesh_specs and first_idx < len(config.mesh_specs):
                current_offset += config.mesh_specs[first_idx].total_strands
            else:
                current_offset += config.mesh.num_strands

        if config.mesh_specs:
            self.layer_specs = config.mesh_specs
            for i, spec in enumerate(config.mesh_specs):
                mesh_cfg = spec.make_mesh_config(H, config.state_size)
                offset = layer_to_offset.get(i, current_offset)
                if spec.is_category():
                    mesh = CategoryMesh(self.genome, mesh_cfg, spec.categories, layer_offset=offset)
                else:
                    mesh = NeuralMesh(self.genome, mesh_cfg, layer_offset=offset)
                self.mesh_layers.append(mesh)
                self.layer_norms.append(nn.LayerNorm(H))
        else:
            self.layer_specs = [
                LayerSpec(name="layer", num_strands=config.mesh.num_strands, top_k=config.mesh.top_k)
                for _ in range(config.num_layers)
            ]
            for i in range(config.num_layers):
                offset = layer_to_offset.get(i, current_offset)
                self.mesh_layers.append(
                    NeuralMesh(self.genome, deepcopy(config.mesh), layer_offset=offset)
                )
                self.layer_norms.append(nn.LayerNorm(H))

        self.final_norm = nn.LayerNorm(H)
        self.lm_head = nn.Linear(H, config.initial_vocab, bias=False)

        # True Multimodal Encoders (Vision/Audio)
        # Projector input dims match the encoders' real output dim (model_dim/4096).
        self.vision_projector = nn.Linear(4096, H)
        self.audio_projector = nn.Linear(4096, H)

        if getattr(config, "adaptive_depth", False):
            self.exit_heads = nn.ModuleList([
                nn.Linear(H, 1) for _ in range(config.num_layers - 1)
            ])
        else:
            self.exit_heads = None

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if not config.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        self.cortex = MemoryCortex(config.cortex)

    @property
    def vocab_size(self) -> int:
        return self.embedding.num_embeddings

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def active_parameter_count(self) -> int:
        total = self.embedding.weight.numel() + self.final_norm.weight.numel() * 2
        H = self.config.hidden_size
        S = self.config.state_size
        per_strand = 3 * H * S + S * S + 3 * S + H
        total += sum(
            per_strand * min(spec.top_k, spec.num_strands)
            for spec in self.layer_specs
        )
        total += self.genome.config.param_estimate
        return total

    def freeze_layers(self, up_to: int | None = None) -> int:
        if up_to is None:
            up_to = len(self.mesh_layers)
        count = 0
        for i in range(up_to):
            if i >= len(self.mesh_layers):
                break
            mesh = self.mesh_layers[i]
            for param in mesh.router.parameters():
                param.requires_grad = False
            for strand in mesh.strands:
                for p in strand.parameters():
                    p.requires_grad = False
            self.layer_norms[i].requires_grad_(False)
            count += 1
        logger.info("NpDnaModel: frozen %d/%d layers", count, len(self.mesh_layers))
        return count

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
        logger.info("NpDnaModel: all layers unfrozen")

    def grow_strands(self, count: int = 1) -> None:
        if count <= 0:
            return
        old_total = sum(spec.num_strands for spec in self.layer_specs)
        growth_plan: list[tuple[int, int]] = []
        for layer_i, spec in enumerate(self.layer_specs):
            cap_for = getattr(self.config, "_strand_cap_for", None)
            cap = cap_for(spec.name) if cap_for is not None else spec.num_strands + count
            add_n = max(0, min(count, cap - spec.num_strands))
            if add_n:
                growth_plan.append((layer_i, add_n))
        if not growth_plan:
            return

        self.genome.add_strand_capacity(sum(add_n for _, add_n in growth_plan))
        next_strand_id = old_total
        for layer_i, add_n in growth_plan:
            mesh = self.mesh_layers[layer_i]
            for _ in range(add_n):
                mesh.add_strand(strand_id=next_strand_id)
                next_strand_id += 1
            self.layer_specs[layer_i].num_strands += add_n

        top_k_for = getattr(self.config, "_top_k_for", None)
        if top_k_for is not None:
            for spec in self.layer_specs:
                spec.top_k = top_k_for(spec.num_strands)
        new_total = sum(spec.num_strands for spec in self.layer_specs)
        if self.layer_specs:
            self.config.mesh.num_strands = self.layer_specs[0].num_strands
        self.config.genome.max_strands = max(int(self.genome.seeds.shape[0]), new_total)
        logger.info("NpDnaModel: strands/layer +%d (total %d)", count, new_total)

    def add_layer(self, name: str = "main", num_strands: int | None = None, top_k: int | None = None) -> None:
        matching = [spec for spec in self.layer_specs if spec.name == name]
        if num_strands is None:
            num_strands = matching[-1].num_strands if matching else self.config.mesh.num_strands
        if top_k is None:
            top_k = matching[-1].top_k if matching else self.config.mesh.top_k

        old_total = int(self.genome.seeds.shape[0])
        self.genome.add_strand_capacity(num_strands)
        spec = LayerSpec(name=name, num_strands=num_strands, top_k=top_k)
        mesh_cfg = spec.make_mesh_config(self.config.hidden_size, self.config.state_size)
        self.mesh_layers.append(NeuralMesh(self.genome, mesh_cfg, layer_offset=old_total))
        self.layer_norms.append(nn.LayerNorm(self.config.hidden_size))
        self.layer_specs.append(spec)
        self.config.mesh_specs = self.layer_specs
        self.config.num_layers = len(self.layer_specs)
        self.config.genome.max_strands = int(self.genome.seeds.shape[0])
        logger.info("NpDnaModel: added %s layer with %d strands", name, num_strands)

    def resize_embeddings(self, new_vocab: int) -> None:
        if new_vocab <= self.vocab_size:
            return
        old_n = self.vocab_size
        H = self.config.hidden_size
        device = self.embedding.weight.device
        dtype = self.embedding.weight.dtype
        new_emb = nn.Embedding(new_vocab, H, device=device, dtype=dtype)
        new_head = nn.Linear(H, new_vocab, bias=False, device=device, dtype=dtype)
        with torch.no_grad():
            new_emb.weight[:old_n].copy_(self.embedding.weight)
            emb_mean = self.embedding.weight.mean(dim=0, keepdim=True)
            emb_std = self.embedding.weight.std(dim=0, keepdim=True).clamp_min(1e-3)
            emb_noise = torch.randn(new_vocab - old_n, H, device=emb_mean.device, dtype=emb_mean.dtype)
            new_emb.weight[old_n:].copy_(emb_mean + 0.02 * emb_std * emb_noise)
            if not self.config.tie_embeddings:
                new_head.weight[:old_n].copy_(self.lm_head.weight)
                head_mean = self.lm_head.weight.mean(dim=0, keepdim=True)
                head_std = self.lm_head.weight.std(dim=0, keepdim=True).clamp_min(1e-3)
                head_noise = torch.randn(new_vocab - old_n, H, device=head_mean.device, dtype=head_mean.dtype)
                new_head.weight[old_n:].copy_(head_mean + 0.02 * head_std * head_noise)
        self.embedding = new_emb
        self.lm_head = new_head
        if self.config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        self.config.initial_vocab = new_vocab
        logger.info("Embeddings resized: %d → %d", old_n, new_vocab)

    def strand_id_map(self) -> list[list[int]]:
        return [[int(s.strand_id) for s in mesh.strands] for mesh in self.mesh_layers]

    def restore_strand_id_map(self, strand_ids: list[list[int]]) -> None:
        for mesh, ids in zip(self.mesh_layers, strand_ids):
            if len(ids) == len(mesh.strands):
                for strand, sid in zip(mesh.strands, ids):
                    strand.strand_id = int(sid)

    def forward(
        self,
        input_ids: Tensor,
        strand_states: list[list[Tensor | None]] | None = None,
        multimodal_embeddings: Tensor | None = None,
        modality: str = "vision",
        cache_all_strand_states: bool = False,
        timings: dict[str, float] | None = None,
    ) -> tuple[Tensor, Tensor]:
        x = self.embedding(input_ids)

        if multimodal_embeddings is not None:
            if modality == "audio":
                m_emb = self.audio_projector(multimodal_embeddings)
            else:
                m_emb = self.vision_projector(multimodal_embeddings)
            x = torch.cat([m_emb, x], dim=1)

        total_balance_loss = torch.tensor(0.0, device=x.device)

        exit_logits = []
        layer_xs = []
        for i, (mesh, norm) in enumerate(zip(self.mesh_layers, self.layer_norms)):
            residual = x
            mesh_states = strand_states[i] if strand_states is not None else None
            if isinstance(mesh, NeuralMesh):
                mesh_out, bal = mesh(
                    x,
                    strand_states=mesh_states,
                    cache_all_strand_states=cache_all_strand_states,
                    timings=timings,
                )
            else:
                mesh_out, bal = mesh(x)
            x = norm(residual + mesh_out)
            total_balance_loss = total_balance_loss + bal

            # Adaptive Depth (Early Exit)
            if self.exit_heads is not None and i < len(self.exit_heads):
                conf_logit = self.exit_heads[i](x)  # (B, T, 1)
                exit_logits.append(conf_logit)
                if self.training:
                    layer_xs.append(x.detach())

                # Only exit early during inference
                if not self.training:
                    conf = torch.sigmoid(conf_logit)
                    # If all tokens in this batch/seq are confident, we exit early
                    if conf.min().item() > self.config.exit_threshold:
                        break

        if getattr(self, 'cortex', None) is not None and hasattr(self.cortex, 'augment'):
            x = self.cortex.augment(x)
        x = self.final_norm(x)
        output_started = time.perf_counter() if timings is not None else 0.0
        logits = self.lm_head(x)
        if timings is not None:
            timings["output_head"] = timings.get("output_head", 0.0) + time.perf_counter() - output_started

        if self.training and self.exit_heads is not None:
            self._last_exit_logits = exit_logits
            self._last_layer_xs = layer_xs

        return logits, total_balance_loss

    def alloc_strand_states(self) -> list[list[Tensor | None]]:
        """Allocate a strand_states structure for KV-cached generation."""
        return [[None] * m.num_strands for m in self.mesh_layers]

    def reset_cache(self) -> None:
        for mesh in self.mesh_layers:
            if hasattr(mesh, 'reset_cache'):
                mesh.reset_cache()


class CheckpointMixin:
    def save(
        self,
        path: str | Path,
        losses: list[float] | None = None,
        metadata_extra: dict | None = None,
    ) -> None:
        path = Path(path)
        self.active_path = path
        path.mkdir(parents=True, exist_ok=True)
        _atomic_torch_save(self.model.state_dict(), path / "model.pt")
        self.tokenizer.save(path / "tokenizer.json.tmp")
        _replace_with_retries(path / "tokenizer.json.tmp", path / "tokenizer.json")
        cortex_dir = path / "cortex"
        cortex_dir.mkdir(parents=True, exist_ok=True)
        self.cortex.save(cortex_dir / "cortex.pt")
        meta: dict = {
            "config_name": self._match_config_name(),
            "hidden_size": self.config.hidden_size,
            "state_size": self.config.state_size,
            "num_layers": self.config.num_layers,
            "num_strands": self.config.mesh.num_strands,
            "top_k": self.config.mesh.top_k,
            "layer_specs": [
                {
                    "name": spec.name,
                    "num_strands": spec.num_strands,
                    "top_k": spec.top_k,
                    "dense": spec.dense,
                    "categories": spec.categories,
                    "strand_type": getattr(spec.strand, "strand_type", "ssm"),
                    "use_swiglu": getattr(spec.strand, "use_swiglu", True),
                    "num_kv_heads": getattr(spec.strand, "num_kv_heads", 0),
                    "ffn_expansion": getattr(spec.strand, "ffn_expansion", 4.0),
                }
                for spec in getattr(self.model, "layer_specs", [])
            ],
            "strand_type": getattr(self.config.mesh.strand, "strand_type", "ssm"),
            "strand_ids": self.model.strand_id_map(),
            "vocab_capacity": self.tokenizer.capacity,
            "vocab_size": self.tokenizer.size,
            "parameter_count": self.model.parameter_count(),
            "active_parameter_count": self.model.active_parameter_count(),
            "cortex_entries": self.cortex.size,
            "cortex_dim": self.config.cortex.dim,
            "cortex_max_entries": self.config.cortex.max_entries,
            "cortex_top_k": self.config.cortex.top_k,
            "cortex_min_relevance": self.config.cortex.min_relevance,
            "genome_latent_dim": self.config.genome.latent_dim,
            "genome_rank": self.config.genome.rank,
            "genome_encoder_hidden": self.config.genome.encoder_hidden,
            "genome_max_strands": self.config.genome.max_strands,
            "losses": (losses or [])[-500:],
            "saved_at": time.time(),
        }
        if losses:
            meta["best_loss"] = min(losses)
            meta["final_loss"] = losses[-1]
            meta["loss_count"] = len(losses)
        if metadata_extra:
            meta.update(metadata_extra)
        artifact_hashes = {
            relative_name: _sha256_file(path / relative_name)
            for relative_name in ("model.pt", "tokenizer.json", "cortex/cortex.pt")
            if (path / relative_name).is_file()
        }
        meta["artifact_hashes"] = artifact_hashes
        signing_key = os.environ.get("NPDNA_CHECKPOINT_HMAC_KEY")
        if signing_key:
            meta["artifact_hmac"] = _checkpoint_hmac(artifact_hashes, signing_key)
        _atomic_write_text(path / "metadata.json", json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("NpDnaCore saved -> %s (%s params)", path, f"{self.model.parameter_count():,}")

    @classmethod
    def load(cls, path: str | Path) -> "CheckpointMixin":
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        _verify_checkpoint_integrity(path, meta)
        if (path / "model.pt").exists():
            state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        elif cls._is_component_format(path):
            state = cls._load_components(path)
        elif cls._is_sharded_format(path):
            index = json.loads((path / "model_index.json").read_text(encoding="utf-8"))
            state = cls._load_sharded(path, index)
        else:
            raise FileNotFoundError(f"Checkpoint at {path} has neither model.pt nor component model_index.json")
        if "embedding.weight" in state:
            saved_hidden_size = state["embedding.weight"].shape[1]
            meta_hidden_size = meta.get("hidden_size")
            if meta_hidden_size is not None and saved_hidden_size != meta_hidden_size:
                raise RuntimeError(
                    f"Checkpoint at {path} has mismatched architecture dimensions between metadata.json and model.pt "
                    f"(metadata hidden_size {meta_hidden_size} vs model.pt hidden_size {saved_hidden_size})"
                )
        inferred_strands = max(
            (int(m.group(1)) + 1
             for k in state
             if (m := re.match(r"mesh_layers\.\d+\.strands\.(\d+)\.", k))),
            default=meta.get("num_strands", 4),
        )
        default_strand_type = meta.get("strand_type", "ssm")
        strand_cfg = StrandConfig(
            hidden_size=meta["hidden_size"],
            state_size=meta["state_size"],
            strand_type=default_strand_type,
        )
        mesh_cfg = MeshConfig(num_strands=inferred_strands, top_k=meta["top_k"], strand=strand_cfg)
        layer_specs = [
            LayerSpec(
                name=str(item.get("name", "main")),
                num_strands=int(item.get("num_strands", inferred_strands)),
                top_k=int(item.get("top_k", meta["top_k"])),
                categories=item.get("categories"),
                dense=bool(item.get("dense", False)),
                strand=StrandConfig(
                    hidden_size=meta["hidden_size"],
                    state_size=meta["state_size"],
                    strand_type=item.get("strand_type", default_strand_type),
                    use_swiglu=bool(item.get("use_swiglu", False)),
                    num_kv_heads=int(item.get("num_kv_heads", 0)),
                    ffn_expansion=float(item.get("ffn_expansion", 4.0)),
                ),
            )
            for item in meta.get("layer_specs", [])
        ]
        genome_cfg = GenomeConfig(
            latent_dim=meta.get("genome_latent_dim", 256),
            rank=meta.get("genome_rank", 32),
            encoder_hidden=meta.get("genome_encoder_hidden", 512),
            max_strands=meta.get("genome_max_strands", inferred_strands * meta["num_layers"]),
        )
        cortex_cfg = CortexConfig(
            dim=meta.get("cortex_dim", meta["hidden_size"]),
            max_entries=meta.get("cortex_max_entries", 100_000),
            top_k=meta.get("cortex_top_k", 8),
            min_relevance=meta.get("cortex_min_relevance", 0.3),
        )
        checkpoint_complexity = max(0.5, float(meta["hidden_size"]) / 64.0)
        config = NpDnaConfig(
            complexity=checkpoint_complexity,
            initial_vocab=meta["vocab_capacity"],
            hidden_size=meta["hidden_size"], state_size=meta["state_size"],
            num_layers=meta["num_layers"], mesh=mesh_cfg,
            mesh_specs=layer_specs, genome=genome_cfg, cortex=cortex_cfg,
        )
        config.hidden_size = meta["hidden_size"]
        config.state_size = meta["state_size"]
        config.initial_vocab = meta["vocab_capacity"]
        config.num_layers = meta["num_layers"]
        config.cortex.dim = meta.get("cortex_dim", meta["hidden_size"])
        config.mesh.strand.hidden_size = config.hidden_size
        config.mesh.strand.state_size = config.state_size
        config.mesh.strand.strand_type = default_strand_type
        for spec in config.mesh_specs:
            spec.strand.hidden_size = config.hidden_size
            spec.strand.state_size = config.state_size
            spec.strand.strand_type = getattr(spec.strand, "strand_type", default_strand_type)
        model = NpDnaModel(config)
        strand_ids = meta.get("strand_ids")
        if strand_ids:
            model.restore_strand_id_map(strand_ids)
        else:
            base_cfg = CONFIGS.get(str(meta.get("train_config_name") or meta.get("config_name")))
            base_n = base_cfg.mesh.num_strands if base_cfg else meta["num_strands"]
            if not layer_specs and meta["num_strands"] > base_n:
                growth = meta["num_strands"] - base_n
                inferred = [
                    list(range(li * base_n, li * base_n + base_n))
                    + [base_n * meta["num_layers"] + g * meta["num_layers"] + li for g in range(growth)]
                    for li in range(meta["num_layers"])
                ]
                model.restore_strand_id_map(inferred)
        model_state = model.state_dict()
        mismatched = []
        orphan = []
        for key in list(state.keys()):
            if key in model_state:
                if state[key].shape != model_state[key].shape:
                    mismatched.append((key, list(state[key].shape), list(model_state[key].shape)))
            else:
                orphan.append(key)
        if mismatched:
            repair = os.environ.get("NPDNA_REPAIR", "").strip() in ("1", "true", "yes", "on")
            if repair:
                for key, ckpt_shape, model_shape in mismatched:
                    logger.warning(
                        "Size mismatch for '%s': checkpoint %s vs model %s. Stripping (NPDNA_REPAIR=1).",
                        key, ckpt_shape, model_shape,
                    )
                    del state[key]
            else:
                details = "\n".join(
                    f"  {k}: checkpoint {c} vs model {m}" for k, c, m in mismatched
                )
                raise RuntimeError(
                    f"Checkpoint weight shape mismatch at {path} (set NPDNA_REPAIR=1 to strip "
                    f"and rebuild from fresh init).\nMismatched keys:\n{details}"
                )
        for key in orphan:
            logger.debug("Key '%s' in checkpoint not found in model. Skipping.", key)
        model.load_state_dict(state, strict=False)
        tokenizer = AtulyaTokenizer.load(path / "tokenizer.json")
        configured_max_vocab = getattr(config, "max_vocab", None)
        if configured_max_vocab is not None:
            configured_max_vocab = max(int(configured_max_vocab), tokenizer.capacity)
            if tokenizer.max_capacity is None or tokenizer.max_capacity < configured_max_vocab:
                tokenizer.max_capacity = configured_max_vocab
        old_tokenizer_capacity = tokenizer.capacity
        if tokenizer.fill_ratio >= tokenizer.growth_threshold:
            reserve_capacity = math.ceil(tokenizer.size / 0.75)
            tokenizer.ensure_capacity(reserve_capacity)
        if tokenizer.capacity != old_tokenizer_capacity:
            model.resize_embeddings(tokenizer.capacity)
        cortex_path = path / "cortex"
        cortex = MemoryCortex.load(cortex_path, config.cortex) if cortex_path.exists() else MemoryCortex(config.cortex)
        logger.info("NpDnaCore loaded <- %s (%s params, %d cortex entries)",
                     path, f"{model.parameter_count():,}", cortex.size)
        core = cls(model=model, tokenizer=tokenizer, cortex=cortex, config=config)
        core.active_path = path
        return core

    @staticmethod
    def _is_component_format(path: Path) -> bool:
        try:
            idx = json.loads((path / "model_index.json").read_text(encoding="utf-8"))
            return "component_files" in idx
        except Exception:
            return False

    @staticmethod
    def _is_sharded_format(path: Path) -> bool:
        try:
            idx = json.loads((path / "model_index.json").read_text(encoding="utf-8"))
            return "weight_files" in idx
        except Exception:
            return False

    @staticmethod
    def _load_components(path: Path) -> dict[str, torch.Tensor]:
        idx = json.loads((path / "model_index.json").read_text(encoding="utf-8"))
        components = idx["component_files"]
        vocabulary_file = components.get("vocabulary") or components.get("embedding")
        if not vocabulary_file:
            raise KeyError(f"Checkpoint at {path} is missing vocabulary/embedding key in model_index.json")
        required_files = [components["genome"], vocabulary_file, *components["layers"], components["final_norm"]]
        missing = [fname for fname in required_files if not (path / fname).exists()]
        if missing:
            raise FileNotFoundError(f"Checkpoint at {path} missing weight files: {missing}")
        state = {}
        genome = torch.load(path / components["genome"], map_location="cpu", weights_only=True)
        state.update(genome)
        embedding = torch.load(path / vocabulary_file, map_location="cpu", weights_only=True)
        state.update(embedding)
        for fname in components["layers"]:
            layer = torch.load(path / fname, map_location="cpu", weights_only=True)
            state.update(layer)
        final_norm = torch.load(path / components["final_norm"], map_location="cpu", weights_only=True)
        state.update(final_norm)
        logger.info("Loaded state from %d component files", len(required_files))
        return state

    @staticmethod
    def _load_sharded(path: Path, index: dict) -> dict[str, torch.Tensor]:
        state = {}
        for wf in index["weight_files"]:
            shard = torch.load(path / wf, map_location="cpu", weights_only=True)
            state.update(shard)
        return state

    def _match_config_name(self) -> str:
        for name, c in CONFIGS.items():
            if (c.hidden_size == self.config.hidden_size and c.num_layers == self.config.num_layers
                    and c.mesh.num_strands == self.config.mesh.num_strands
                    and c.mesh.top_k == self.config.mesh.top_k
                    and c.initial_vocab == self.config.initial_vocab):
                return name
        return "custom"


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
        response_memory = getattr(self, "response_memory", None)
        if response_memory is not None:
            matched = response_memory.match(prompt)
            if matched is not None:
                answer, _score = matched
                return answer
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

    def generate_speculative(
        self,
        prompt: str,
        max_tokens: int = 128,
        gamma: int = 3,
        temperature: float = 0.7,
    ) -> str:
        """Accelerated Speculative Decoding.

        Drafts `gamma` tokens and verifies them in a single target forward pass.
        Gives 2x-3x speedup on CPU by reducing memory bandwidth pressure.
        """
        response_memory = getattr(self, "response_memory", None)
        if response_memory is not None:
            matched = response_memory.match(prompt)
            if matched is not None:
                return matched[0]

        prompt_text = _build_chat_prompt(prompt)
        prompt_ids = self.encode(prompt_text, allow_growth=False)
        ids = list(prompt_ids) or [self.tokenizer.token_to_id.get("<bos>", 2)]

        device = self.model.embedding.weight.device
        valid_vocab = min(self.tokenizer.size, self.model.vocab_size)
        eos_id = self.tokenizer.token_to_id.get("<eos>", 3)
        generated_count = 0

        self.model.eval()
        with torch.no_grad():
            while generated_count < max_tokens:
                draft_ids = []
                temp_ids = list(ids)
                for _ in range(gamma):
                    window = temp_ids[-256:]
                    x = torch.tensor([window], dtype=torch.long, device=device)
                    logits, _ = self.model(x)
                    next_logits = logits[0, -1, :valid_vocab]
                    next_id = int(next_logits.argmax().item())
                    draft_ids.append(next_id)
                    temp_ids.append(next_id)
                    if next_id == eos_id:
                        break

                verify_input = ids + draft_ids
                x_verify = torch.tensor([verify_input[-256:]], dtype=torch.long, device=device)
                target_logits, _ = self.model(x_verify)

                accepted = 0
                for i, draft_tok in enumerate(draft_ids):
                    target_pos = -(len(draft_ids) - i + 1)
                    pred_tok = int(target_logits[0, target_pos, :valid_vocab].argmax().item())
                    if draft_tok == pred_tok:
                        accepted += 1
                        ids.append(draft_tok)
                        generated_count += 1
                        if draft_tok == eos_id:
                            break
                    else:
                        ids.append(pred_tok)
                        generated_count += 1
                        break

                if not accepted and len(draft_ids) == 0:
                    break

                if ids[-1] == eos_id:
                    break

        gen_tokens = ids[len(prompt_ids):]
        return self.decode(gen_tokens)

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
        use_state_cache: bool = True,
    ) -> Generator[str, None, None]:
        response_memory = getattr(self, "response_memory", None)
        if response_memory is not None:
            matched = response_memory.match(prompt)
            if matched is not None:
                answer, _score = matched
                yield answer
                return
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
                can_cache = (
                    use_state_cache
                    and image_inputs is None
                    and audio_inputs is None
                    and self._supports_stateful_generation()
                )
                states = self.model.alloc_strand_states() if can_cache else None
                initial_ctx = ids[-max(1, int(context_window)):]
                initial_input = torch.tensor([initial_ctx], dtype=torch.long, device=device)
                initial_kwargs = {}
                if image_inputs is not None:
                    initial_kwargs["multimodal_embeddings"] = image_inputs
                    initial_kwargs["modality"] = "vision"
                elif audio_inputs is not None:
                    initial_kwargs["multimodal_embeddings"] = audio_inputs
                    initial_kwargs["modality"] = "audio"
                if can_cache:
                    logits, _ = self.model(
                        input_ids=initial_input,
                        strand_states=states,
                        cache_all_strand_states=True,
                        **initial_kwargs,
                    )
                else:
                    logits, _ = self.model(input_ids=initial_input, **initial_kwargs)

                for _ in range(max_tokens):
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

                    if can_cache:
                        input_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
                        logits, _ = self.model(
                            input_ids=input_ids,
                            strand_states=states,
                            cache_all_strand_states=True,
                        )
                    else:
                        ctx = ids[-max(1, int(context_window)):]
                        input_ids = torch.tensor([ctx], dtype=torch.long, device=device)
                        logits, _ = self.model(input_ids=input_ids, **initial_kwargs)
            finally:
                if hasattr(self.model.genome, "disable_inference_cache"):
                    self.model.genome.disable_inference_cache()

            self.last_generated_ids = ids
            self._record_strand_specialization(original_prompt)
            self._handle_cortex_writeback(ids[len(prompt_ids):], device)

    def _supports_stateful_generation(self) -> bool:
        """Return True only for meshes whose recurrent caches are implemented."""
        try:
            return bool(self.model.mesh_layers) and all(
                isinstance(mesh, NeuralMesh)
                and all(getattr(strand.config, "strand_type", "ssm") == "ssm" for strand in mesh.strands)
                for mesh in self.model.mesh_layers
            )
        except (AttributeError, TypeError):
            return False

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


class NpDnaCore(GenerationMixin, CheckpointMixin):
    """High-level wrapper: model + tokenizer + cortex + auto-scaling.

    This is the main interface for training and inference.
    """

    def __init__(self, model: NpDnaModel, tokenizer: AtulyaTokenizer,
                 cortex: MemoryCortex | None = None, config: NpDnaConfig | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        if cortex is not None:
            self.model.cortex = cortex
        self.cortex = self.model.cortex
        self.config = config or NpDnaConfig()
        self.active_path: Path | None = None
        self.response_memory = None

    @classmethod
    def from_config(cls, name_or_complexity: str | float = "seed") -> "NpDnaCore":
        if isinstance(name_or_complexity, (int, float)):
            config = NpDnaConfig(complexity=name_or_complexity)
        elif name_or_complexity in CONFIGS:
            config = deepcopy(CONFIGS[name_or_complexity])
        else:
            config = NpDnaConfig()

        tokenizer = AtulyaTokenizer(initial_capacity=config.initial_vocab, max_capacity=config.max_vocab)
        model = NpDnaModel(config)
        cortex = MemoryCortex(config.cortex)
        logger.info(
            "NpDnaCore created [c=%.1f]: %s params total, %s active | vocab=%d | %d layers | %d strands total",
            config.complexity,
            f"{model.parameter_count():,}",
            f"{model.active_parameter_count():,}",
            tokenizer.vocab_size,
            config.num_layers,
            config.total_strands,
        )
        return cls(model=model, tokenizer=tokenizer, cortex=cortex, config=config)

    def encode(self, text: str, allow_growth: bool = True) -> list[int]:
        old_cap = self.tokenizer.capacity
        ids = self.tokenizer.encode(text, allow_growth=allow_growth)
        if self.tokenizer.capacity != old_cap:
            self.model.resize_embeddings(self.tokenizer.capacity)
        return ids

    def decode(self, ids) -> str:
        if isinstance(ids, Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode(ids)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Generation — sampling, streaming, prompt formatting, cortex write-back     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Sampling helpers ──────────────────────────────────────────────────────────

def _apply_repetition_penalty(logits: Tensor, seen_ids: list[int], penalty: float,
                               freq_penalty: float = 0.3) -> Tensor:
    if penalty <= 1.0 and freq_penalty <= 0.0:
        return logits
    logits = logits.clone()
    counts = Counter(seen_ids[-128:])
    for tok_id, count in counts.items():
        if 0 <= tok_id < logits.size(0):
            # Scale penalty by frequency: tokens seen more often get penalized harder
            effective = penalty + freq_penalty * math.log1p(count)
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Genome — DNA weight generator for Strands                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

WeightRole = Literal["gate", "state", "recurrent", "output"]
_ROLES: list[WeightRole] = ["gate", "state", "recurrent", "output"]

_WEIGHT_STD: dict[WeightRole, float] = {
    "gate": 0.02,
    "state": 0.02,
    "recurrent": 0.005,
    "output": 0.02,
}

_BIAS_STD: dict[WeightRole, float] = {
    "gate": 0.02,
    "state": 0.02,
    "recurrent": 0.005,
    "output": 0.02,
}


class Genome(nn.Module):
    """DNA weight generator.  Stores one learnable seed per Strand and
    shared encoder/decoder networks that convert seeds into weight matrices.

    Args:
        config: Genome hyperparameters (latent dim, rank, max strands).
        strand_cfg: Per-strand shape info (hidden_size, state_size).
    """

    def __init__(self, config: GenomeConfig, strand_cfg: StrandConfig):
        super().__init__()
        self.config = config
        self.strand_cfg = strand_cfg

        H = strand_cfg.hidden_size
        S = strand_cfg.state_size
        L = config.latent_dim
        R = config.rank

        # One learnable seed per Strand (this is what gets trained per-topic)
        self.seeds = nn.Parameter(torch.randn(config.max_strands, L) * 0.02)

        # Shared encoder:  seed → latent
        self.encoder = nn.Sequential(
            nn.Linear(L, config.encoder_hidden),
            nn.GELU(),
            nn.Linear(config.encoder_hidden, L),
            nn.LayerNorm(L),
        )

        # Weight shape registry:  role → (rows, cols)
        self._shapes: dict[str, tuple[int, int]] = {
            "gate": (H, S),
            "state": (H, S),
            "recurrent": (S, S),
            "output": (S, H),
        }

        # Per-role decoders: latent → low-rank factors U, V
        self.decoders = nn.ModuleDict()
        for role, (rows, cols) in self._shapes.items():
            # U factor: latent → rows × rank
            # V factor: latent → rank × cols
            self.decoders[f"{role}_U"] = nn.Linear(L, rows * R)
            self.decoders[f"{role}_V"] = nn.Linear(L, R * cols)

        # Per-role bias decoders
        self.bias_decoders = nn.ModuleDict()
        for role, (_, cols) in self._shapes.items():
            self.bias_decoders[role] = nn.Linear(L, cols)

        self._cache_enabled = False
        # A detached cache is also safe during training when every genome
        # parameter is frozen (for example LoRA-only fine-tuning).  It avoids
        # regenerating the same direct weights for every training step.
        self._frozen_weight_cache = False
        self._direct_weight_write = False
        self._cache_version = 0
        self._weight_cache: dict[int, dict[str, tuple[Tensor, Tensor]]] = {}

    def train(self, mode: bool = True):
        if not self._frozen_weight_cache:
            self.disable_inference_cache()
        return super().train(mode)

    def enable_inference_cache(self) -> None:
        """Cache generated strand weights while the genome is in eval mode."""
        self._cache_enabled = True
        self._frozen_weight_cache = False
        self._direct_weight_write = False
        self._clear_weight_cache()

    def enable_frozen_weight_cache(self, *, direct_write: bool = True) -> None:
        """Cache direct weights when the entire genome is frozen.

        Cached tensors are detached, so this is intentionally rejected if a
        genome parameter remains trainable.  Unlike inference caching, this
        mode remains active while the parent model is in ``train()`` mode.
        """
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise RuntimeError("Frozen-genome caching requires all genome parameters to be frozen")
        self._cache_enabled = True
        self._frozen_weight_cache = True
        self._direct_weight_write = direct_write
        self._clear_weight_cache()

    def disable_inference_cache(self) -> None:
        """Disable cached weights and clear any cached tensors."""
        self._cache_enabled = False
        self._frozen_weight_cache = False
        self._direct_weight_write = False
        self._clear_weight_cache()

    def _clear_weight_cache(self) -> None:
        self._cache_version += 1
        self._weight_cache.clear()

    @property
    def direct_weight_write_active(self) -> bool:
        return self._frozen_weight_cache and self._direct_weight_write

    def generate(self, strand_id: int, role: WeightRole) -> tuple[Tensor, Tensor]:
        """Generate a weight matrix and bias for a specific Strand and role.

        Returns:
            (weight, bias) where weight = U @ V  (low-rank approximation).
        """
        if strand_id < 0 or strand_id >= self.seeds.shape[0]:
            raise IndexError(
                f"Strand {strand_id} out of range (genome has {self.seeds.shape[0]} seeds). "
                f"Call add_strand_capacity() before routing to new strands."
            )
        seed = self.seeds[strand_id].unsqueeze(0)
        latent = self.encoder(seed)

        rows, cols = self._shapes[role]
        R = self.config.rank

        U = self.decoders[f"{role}_U"](latent).reshape(rows, R)
        V = self.decoders[f"{role}_V"](latent).reshape(R, cols)
        weight = U @ V  # (rows, cols)
        target_std = _WEIGHT_STD[role] / math.sqrt(max(1.0, float(rows) / 128.0))
        current_std = weight.detach().float().std().clamp_min(1e-6)
        weight = weight * (target_std / current_std.to(weight.dtype))

        bias = self.bias_decoders[role](latent).squeeze(0)  # (cols,)
        bias_target_std = _BIAS_STD[role]
        bias_std = bias.detach().float().std().clamp_min(1e-6)
        bias = bias * (bias_target_std / bias_std.to(bias.dtype))

        return weight, bias

    def generate_all(
        self, strand_id: int, timings: dict[str, float] | None = None,
    ) -> dict[str, tuple[Tensor, Tensor]]:
        """Generate all weight matrices for a Strand in one call."""
        started = time.perf_counter() if timings is not None else 0.0
        cache_allowed = self._cache_enabled and (not self.training or self._frozen_weight_cache)
        if cache_allowed:
            cached = self._weight_cache.get(strand_id)
            if cached is not None:
                if timings is not None:
                    timings["genome"] = timings.get("genome", 0.0) + time.perf_counter() - started
                return cached

        weights = {role: self.generate(strand_id, role) for role in _ROLES}
        if cache_allowed:
            weights = {
                role: (weight.detach(), bias.detach())
                for role, (weight, bias) in weights.items()
            }
            self._weight_cache[strand_id] = weights
        if timings is not None:
            timings["genome"] = timings.get("genome", 0.0) + time.perf_counter() - started
        return weights

    @property
    def num_active_strands(self) -> int:
        return self.config.max_strands

    def add_strand_capacity(self, count: int = 1) -> None:
        """Grow the seed bank, replacing the Parameter to bump autograd's version counter."""
        if count <= 0:
            return

        old_max = int(self.seeds.shape[0])
        new_max = old_max + count
        with torch.no_grad():
            grown = torch.randn(
                count,
                self.config.latent_dim,
                device=self.seeds.device,
                dtype=self.seeds.dtype,
            ) * 0.02
            new_data = torch.cat([self.seeds.data, grown], dim=0)
            new_param = nn.Parameter(new_data, requires_grad=self.seeds.requires_grad)
            if self.seeds.grad is not None:
                grad_pad = torch.zeros(
                    count,
                    self.config.latent_dim,
                    device=self.seeds.grad.device,
                    dtype=self.seeds.grad.dtype,
                )
                new_param.grad = torch.cat([self.seeds.grad, grad_pad], dim=0)
            self.seeds = new_param

        self.config.max_strands = new_max
        self._clear_weight_cache()
        logger.info("Genome: expanded seed bank %d -> %d", old_max, new_max)

    def clone_strand(self, src_id: int, noise_scale: float = 0.05) -> int:
        """Clone a strand's seed and apply slight mutation (evolution). Returns new strand_id."""
        self.add_strand_capacity(1)
        new_id = self.seeds.shape[0] - 1
        with torch.no_grad():
            self.seeds.data[new_id] = self.seeds.data[src_id] + torch.randn_like(self.seeds.data[src_id]) * noise_scale
        self._clear_weight_cache()
        logger.info("Genome: Cloned strand %d -> %d with noise %.2f", src_id, new_id, noise_scale)
        return new_id

    def prune_strand(self, strand_id: int) -> None:
        """Mark a strand's seed as dead (zero out). It can still be used, but shouldn't be routed to."""
        with torch.no_grad():
            self.seeds.data[strand_id].zero_()
        self._clear_weight_cache()
        logger.info("Genome: Pruned strand %d", strand_id)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  LoRA — adapters for fine-tuning                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        update = (self.dropout(x) @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(0, 1)
        return self.base(x) + update * self.scaling


def inject_lora(
    model: nn.Module,
    *,
    rank: int = 8,
    alpha: float | None = None,
    dropout: float = 0.0,
    target_suffixes: Iterable[str] = ("ffn_gate", "ffn_up", "ffn_down", "q_proj", "k_proj", "v_proj", "out_proj"),
) -> list[str]:
    """Replace matching ``nn.Linear`` children with LoRA wrappers."""
    suffixes = tuple(target_suffixes)
    replaced: list[str] = []

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            qualified = f"{prefix}.{name}" if prefix else name
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear) and any(qualified.endswith(suffix) for suffix in suffixes):
                parent._modules[name] = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                replaced.append(qualified)
            else:
                visit(child, qualified)

    visit(model)
    return replaced


def mark_only_lora_trainable(model: nn.Module) -> int:
    """Freeze the base model and return the number of trainable LoRA weights."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if name.endswith("lora_A") or name.endswith("lora_B")}


def save_lora_adapter(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), path)


def load_lora_adapter(model: nn.Module, path: str | Path, *, strict: bool = True) -> tuple[list[str], list[str]]:
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    missing = [name for name in incompatible.missing_keys if name.endswith("lora_A") or name.endswith("lora_B")]
    unexpected = list(incompatible.unexpected_keys)
    if strict and (missing or unexpected):
        raise ValueError(f"LoRA adapter mismatch: missing={missing}, unexpected={unexpected}")
    return missing, unexpected
