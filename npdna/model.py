"""NP-DNA Model — full NeuroPlastic DNA Network.

Architecture:
    Token IDs → Embedding → [Mesh₁ → … → Meshₙ] → Norm → LM Head

Auto-scales: vocab grows, strands grow, layers grow — all automatic.
"""

from __future__ import annotations

import json
import logging
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

        if config.mesh_specs:
            self.layer_specs = config.mesh_specs
            offset = 0
            for spec in config.mesh_specs:
                mesh_cfg = spec.make_mesh_config(H, config.state_size)
                if spec.is_category():
                    mesh = CategoryMesh(self.genome, mesh_cfg, spec.categories, layer_offset=offset)
                else:
                    mesh = NeuralMesh(self.genome, mesh_cfg, layer_offset=offset)
                self.mesh_layers.append(mesh)
                self.layer_norms.append(nn.LayerNorm(H))
                offset += spec.total_strands
        else:
            self.layer_specs = [
                LayerSpec(name="layer", num_strands=config.mesh.num_strands, top_k=config.mesh.top_k)
                for _ in range(config.num_layers)
            ]
            for i in range(config.num_layers):
                self.mesh_layers.append(
                    NeuralMesh(self.genome, deepcopy(config.mesh), layer_offset=i * config.mesh.num_strands)
                )
                self.layer_norms.append(nn.LayerNorm(H))

        self.final_norm = nn.LayerNorm(H)
        self.lm_head = nn.Linear(H, config.initial_vocab, bias=False)

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
        self.genome.add_strand_capacity(self.config.num_layers * count)
        for grow_i in range(count):
            for layer_i, mesh in enumerate(self.mesh_layers):
                strand_id = old_total + grow_i * self.config.num_layers + layer_i
                mesh.add_strand(strand_id=strand_id)
        for spec in self.layer_specs:
            spec.num_strands += count
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
            if not self.config.tie_embeddings:
                new_head.weight[:old_n].copy_(self.lm_head.weight)
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

    def forward(self, input_ids: Tensor, strand_states: list[list[Tensor | None]] | None = None) -> tuple[Tensor, Tensor]:
        x = self.embedding(input_ids)
        total_balance_loss = torch.tensor(0.0, device=x.device)

        for i, (mesh, norm) in enumerate(zip(self.mesh_layers, self.layer_norms)):
            residual = x
            mesh_states = strand_states[i] if strand_states is not None else None
            mesh_out, bal = mesh(x, strand_states=mesh_states)
            x = norm(residual + mesh_out)
            total_balance_loss = total_balance_loss + bal

        x = self.cortex.augment(x)
        x = self.final_norm(x)
        return self.lm_head(x), total_balance_loss

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
        torch.save(self.model.state_dict(), path / "model.pt")
        self.tokenizer.save(path / "tokenizer.json")
        self.cortex.save(path / "cortex")
        meta: dict = {
            "config_name": self._match_config_name(),
            "hidden_size": self.config.hidden_size,
            "state_size": self.config.state_size,
            "num_layers": self.config.num_layers,
            "num_strands": self.config.mesh.num_strands,
            "top_k": self.config.mesh.top_k,
            "layer_specs": [
                {"name": spec.name, "num_strands": spec.num_strands, "top_k": spec.top_k}
                for spec in getattr(self.model, "layer_specs", [])
            ],
            "strand_ids": self.model.strand_id_map(),
            "vocab_capacity": self.tokenizer.capacity,
            "vocab_size": self.tokenizer.size,
            "parameter_count": self.model.parameter_count(),
            "active_parameter_count": self.model.active_parameter_count(),
            "cortex_entries": self.cortex.size,
            "cortex_dim": self.config.cortex.dim,
            "cortex_max_entries": self.config.cortex.max_entries,
            "cortex_top_k": self.config.cortex.top_k,
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
        (path / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
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
        strand_cfg = StrandConfig(hidden_size=meta["hidden_size"], state_size=meta["state_size"])
        mesh_cfg = MeshConfig(num_strands=inferred_strands, top_k=meta["top_k"], strand=strand_cfg)
        layer_specs = [
            LayerSpec(
                name=str(item.get("name", "main")),
                num_strands=int(item.get("num_strands", inferred_strands)),
                top_k=int(item.get("top_k", meta["top_k"])),
                strand=StrandConfig(hidden_size=meta["hidden_size"], state_size=meta["state_size"]),
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
        for spec in config.mesh_specs:
            spec.strand.hidden_size = config.hidden_size
            spec.strand.state_size = config.state_size
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
