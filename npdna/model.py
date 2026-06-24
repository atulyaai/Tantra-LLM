"""NP-DNA Model — full NeuroPlastic DNA Network.

Architecture:
    Token IDs → Embedding → [Mesh₁ → … → Meshₙ] → Norm → LM Head

Auto-scales: vocab grows, strands grow, layers grow — all automatic.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch import Tensor, nn

from .config import CONFIGS, CortexConfig, GenomeConfig, LayerSpec, MeshConfig, NpDnaConfig, StrandConfig
from .mesh import CategoryMesh, NeuralMesh
from .cortex import MemoryCortex
from .genome import Genome
from .tokenizer import AtulyaTokenizer
from .generation import GenerationMixin

logger = logging.getLogger(__name__)


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
        self.vision_projector = nn.Linear(512, H)
        self.audio_projector = nn.Linear(128, H)

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
        new_emb = nn.Embedding(new_vocab, H)
        new_head = nn.Linear(H, new_vocab, bias=False)
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

    def forward(self, input_ids: Tensor, strand_states: list[list[Tensor | None]] | None = None,
                multimodal_embeddings: Tensor | None = None) -> tuple[Tensor, Tensor]:
        x = self.embedding(input_ids)

        if multimodal_embeddings is not None:
            if multimodal_embeddings.shape[-1] == 512:
                m_emb = self.vision_projector(multimodal_embeddings)
            else:
                m_emb = self.audio_projector(multimodal_embeddings)
            x = torch.cat([m_emb, x], dim=1)

        total_balance_loss = torch.tensor(0.0, device=x.device)

        exit_logits = []
        layer_xs = []
        for i, (mesh, norm) in enumerate(zip(self.mesh_layers, self.layer_norms)):
            residual = x
            mesh_states = strand_states[i] if strand_states is not None else None
            mesh_out, bal = mesh(x, strand_states=mesh_states)
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
        logits = self.lm_head(x)

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
        self.cortex.save(path / "cortex")
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
        _atomic_write_text(path / "metadata.json", json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("NpDnaCore saved -> %s (%s params)", path, f"{self.model.parameter_count():,}")

    @classmethod
    def load(cls, path: str | Path) -> "CheckpointMixin":
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
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
        for key in list(state.keys()):
            if key in model_state:
                if state[key].shape != model_state[key].shape:
                    logger.warning(
                        "Size mismatch for '%s': checkpoint %s vs model %s. Skipping.",
                        key, list(state[key].shape), list(model_state[key].shape),
                    )
                    del state[key]
            else:
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
