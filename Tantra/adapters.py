"""
tantra/adapters.py — Lightweight per-domain adapter categories for CPU deployment.

The training capacity of a small CPU model is the shared base (NOT unlimited
"world knowledge").  Following the design brief:

* Shared base first: strong general dataset.
* One request-level router: never load or combine all adapters per question.
* Route to a single domain adapter, with the base as fallback.
* Adapters are added over time via ``main.py --mode adapter add``; the
  registry is a small JSON file so categories are fully scriptable.

Each category owns a dedicated stack of specialist ``NeuroCoreBlock`` layers
cloned from the shared base.  Each layer is gated by a zero-initialised
residual gate, so installing one does not disturb the base until that
category is actually trained (the gate opens as training proceeds).
Categories map 1:1 onto the topic dataset folders in ``Datasets/``.
"""
from __future__ import annotations

import os
import re
import json
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from Tantra.model import NeuroCoreModel
from Tantra.config import NeuroCoreConfig
from Tantra.utils import get_logger

log = get_logger(__name__)

ADAPTERS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Model", "Adapters")
REGISTRY_PATH = os.path.join(ADAPTERS_DIR, "registry.json")


# ── Category metadata ─────────────────────────────────────────────────────────

@dataclass
class AdapterCategory:
    """One routeable domain adapter category."""
    name: str
    description: str = ""
    topics: List[str] = field(default_factory=list)   # Datasets/<topic> folders feeding this category
    rank: int = 32                                    # informational; retained for registry compatibility
    depth: int = 1                                    # specialist layers stacked for this category (grow/shrink)
    min_depth: int = 1                                # floor: never shrink below one layer
    max_depth: int = 3                                # ceiling: cap most at 3, code/math may use 3
    status: str = "untrained"                         # "untrained" | "trained"
    params: int = 0                                   # measured parameter count of the full stack
    keywords: List[str] = field(default_factory=list) # routing lexicon for the request router

    def validate(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"Invalid adapter category name: {self.name!r}")


DEFAULT_CATEGORIES: List[AdapterCategory] = [
    AdapterCategory(
        name="general",
        description="General conversation, reasoning, and open-domain chat (base fallback).",
        topics=["general"],
        max_depth=2,
        keywords=["hello", "hi", "hey", "how are you", "tell me about", "what is", "who is",
                  "chat", "conversation", "opinion", "introduction", "yourself", "why", "explain",
                  "reason", "think", "because", "compare", "difference"],
    ),
    AdapterCategory(
        name="math",
        description="Math and logic: arithmetic, algebra, proofs, reasoning.",
        topics=["math"],
        max_depth=3,
        keywords=["calculate", "equation", "solve", "math", "algebra", "geometry", "derivative",
                  "integral", "probability", "sum", "divide", "multiply", "square root", "logic",
                  "proof", "hypotenuse", "fraction", "percentage", "remainder", "formula"],
    ),
    AdapterCategory(
        name="science",
        description="Science: physics, chemistry, biology, astronomy, health.",
        topics=["science"],
        max_depth=2,
        keywords=["science", "physics", "chemistry", "biology", "atom", "molecule", "gravity",
                  "enzyme", "cell", "planet", "orbit", "experiment", "hypothesis", "temperature",
                  "dna", "protein", "climate", "energy", "electrons", "photosynthesis"],
    ),
    AdapterCategory(
        name="code",
        description="Programming: Python, algorithms, debugging, software design.",
        topics=["code"],
        max_depth=3,
        keywords=["python", "code", "program", "function", "bug", "debug", "algorithm",
                  "compile", "library", "import", "loop", "recursion", "api", "database",
                  "syntax", "variable", "javascript", "c++", "linux", "refactor", "sql"],
    ),
    AdapterCategory(
        name="creative_writing",
        description="Writing and creativity: stories, poems, essays, style.",
        topics=["creative_writing"],
        max_depth=2,
        keywords=["write", "story", "poem", "essay", "creative", "imagine", "novel", "character",
                  "plot", "metaphor", "describe", "dialogue", "genre", "draft", "rewrite", "rhyme",
                  "scene", "narration", "inspiration"],
    ),
    AdapterCategory(
        name="instructions",
        description="Instructions and productivity: how-to, planning, tasks.",
        topics=["instructions"],
        max_depth=2,
        keywords=["how to", "steps", "guide", "instructions", "tutorial", "task", "plan", "list",
                  "productivity", "organize", "workflow", "schedule", "step by step", "procedure",
                  "tips", "checklist", "setup", "install", "configure"],
    ),
    AdapterCategory(
        name="safety",
        description="Safety, ethics, and clear factual uncertainty.",
        topics=["safety"],
        max_depth=2,
        keywords=["danger", "unsafe", "illegal", "harm", "ethic", "privacy", "bias", "weapon",
                  "suicide", "misinformation", "uncertain", "not sure", "does not know",
                  "safe", "responsible", "consent", "abuse", "report"],
    ),
    AdapterCategory(
        name="multilingual",
        description="Multilingual and Hindi/Sanskrit understanding.",
        topics=["multilingual"],
        max_depth=2,
        keywords=["hindi", "sanskrit", "english translation", "translate", "translation",
                  "भाषा", "नमस्ते", "संस्कृत", "हिंदी", "language", "अनुवाद", "शब्द",
                  "polylingual", "bilingual", "अंग्रेज़ी", "devanagari"],
    ),
]


# ── Persistent registry ───────────────────────────────────────────────────────

class AdapterRegistry:
    """Scratch-proof JSON registry of installed adapter categories.

    Categories are pure data: adding one never rebuilds the base model.  The
    checkpoint on disk keeps whatever banks already exist; a new category is
    simply trained next and saved into the checkpoint.
    """

    def __init__(self, path: str = REGISTRY_PATH):
        self._path = path
        self._categories: Dict[str, AdapterCategory] = {}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.load()

    def seed_defaults(self) -> None:
        """Register the 8 initial categories if they are not present yet."""
        changed = False
        for category in DEFAULT_CATEGORIES:
            if category.name not in self._categories:
                self._categories[category.name] = category
                changed = True
        if changed:
            self.save()
            log.info("Seeded %d default adapter categories.", len(DEFAULT_CATEGORIES))

    def add(
        self,
        name: str,
        description: str = "",
        topics: Optional[List[str]] = None,
        rank: int = 32,
        keywords: Optional[List[str]] = None,
        max_depth: int = 3,
    ) -> AdapterCategory:
        category = AdapterCategory(
            name=name,
            description=description or f"Custom domain adapter: {name}",
            topics=topics or [name],
            rank=max(4, int(rank)),
            max_depth=max(1, int(max_depth)),
            keywords=keywords or [],
        )
        category.validate()
        self._categories[name] = category
        self.save()
        log.info("Registered adapter category '%s' (rank %d, topics=%s).", name, category.rank, category.topics)
        return category

    def update_params(self, name: str, params: int) -> None:
        if name in self._categories:
            self._categories[name].params = int(params)
            self.save()

    def update_depth(self, name: str, depth: int, params: int) -> None:
        if name in self._categories:
            self._categories[name].depth = max(1, int(depth))
            self._categories[name].params = int(params)
            self.save()

    def mark_trained(self, name: str) -> None:
        if name in self._categories:
            self._categories[name].status = "trained"
            self.save()

    def remove(self, name: str) -> bool:
        if name not in self._categories:
            return False
        del self._categories[name]
        self.save()
        log.info("Removed adapter category '%s' from registry.", name)
        return True

    def get(self, name: str) -> Optional[AdapterCategory]:
        return self._categories.get(name)

    def all(self) -> List[AdapterCategory]:
        default_order = [c.name for c in DEFAULT_CATEGORIES]
        return sorted(self._categories.values(), key=lambda c: default_order.index(c.name) if c.name in default_order else len(default_order))

    def names(self) -> List[str]:
        return [c.name for c in self.all()]

    def save(self) -> None:
        data = {name: dict(
            name=c.name,
            description=c.description,
            topics=list(c.topics),
            rank=int(c.rank),
            depth=int(c.depth),
            min_depth=int(c.min_depth),
            max_depth=int(c.max_depth),
            status=c.status,
            params=int(c.params),
            keywords=list(c.keywords),
        ) for name, c in self._categories.items()}
        temporary_path = self._path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        os.replace(temporary_path, self._path)

    def load(self) -> None:
        if not os.path.isfile(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self._categories = {}
            for name, raw in data.items():
                self._categories[str(name)] = AdapterCategory(
                    name=str(raw.get("name", name)),
                    description=str(raw.get("description", "")),
                    topics=list(raw.get("topics", []) or []),
                    rank=int(raw.get("rank", 32)),
                    depth=int(raw.get("depth", 1)),
                    min_depth=int(raw.get("min_depth", 1)),
                    max_depth=int(raw.get("max_depth", 3)),
                    status=str(raw.get("status", "untrained")),
                    params=int(raw.get("params", 0)),
                    keywords=list(raw.get("keywords", []) or []),
                )
        except Exception as exc:
            log.warning("Could not load adapter registry (%s); starting empty.", exc)

    def __contains__(self, name: str) -> bool:
        return name in self._categories

    def __len__(self) -> int:
        return len(self._categories)

    def __repr__(self) -> str:
        return f"AdapterRegistry({len(self._categories)} categories, {self._path})"


# ── Request-level router ─────────────────────────────────────────────────────

class RequestRouter:
    """Picks a single domain adapter for a user request; returns None for the base.

    Rule-based and deterministic: scores each category by keyword hits plus a
    language/code heuristic, and only commits to a domain if its score clears
    both an absolute floor and a margin over the next-best candidate.  This is
    intentionally simple — a learned router would need per-request labels that
    a CPU system does not have yet — and it routes the request, not every token.
    """

    def __init__(self, registry: AdapterRegistry):
        self._registry = registry

    def route(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return None
        lowered = text.lower()
        scores: Dict[str, float] = {}
        for category in self._registry.all():
            score = 0.0
            for keyword in category.keywords:
                if keyword in lowered:
                    score += 1.0
            scores[category.name] = score

        # Heuristic helpers that strengthen without needing a lexicon entry.
        if _contains_devanagari(text):
            scores["multilingual"] = scores.get("multilingual", 0.0) + 2.0
        if _looks_like_code(text):
            scores["code"] = scores.get("code", 0.0) + 1.5
        if re.search(r"[\d<>=+\-*/^]+|[0-9][+\-*/^=][0-9]|%.", lowered):
            scores["math"] = scores.get("math", 0.0) + 0.5

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if not ranked:
            return None
        best, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score >= 1.0 and best_score >= runner_up + 0.5:
            return best
        return None


def _contains_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097f" for ch in text)


_CODE_HINTS = (
    "def ", "import ", "return ", "class ", "function(", ") {", "};", "print(", "stdin",
    "self.", "=>", "```", "=", "#include", "var ", "let ", "const ", "for(", "while(",
)


def _looks_like_code(text: str) -> bool:
    hits = sum(1 for hint in _CODE_HINTS if hint in text)
    return hits >= 2


# ── Model integration helpers ─────────────────────────────────────────────────

def install_category_layers(model: NeuroCoreModel, categories: List[AdapterCategory]) -> Dict[str, int]:
    """Install one dedicated specialist layer per category onto the model.

    Each layer is cloned from a shared base block and carries a zero-initialised
    residual gate, so a freshly installed category is an exact identity
    pass-through — it does not perturb the base until its dataset trains it.
    Returns a mapping of category name -> parameter count of its single
    dedicated layer (block + gate).
    """
    installed: Dict[str, int] = {}
    for category in categories:
        category.validate()
        if category.name in getattr(model, "category_layers", {}):
            log.info("Category layer '%s' already installed; skipping.", category.name)
        else:
            model.add_category_layers([category.name], depth=category.depth,
                                     clone_layer_index=model.config.adapter.clone_layer_index)
        params = sum(p.numel() for p in model.category_layers[category.name].parameters())
        params += sum(p.numel() for p in model.category_gates[category.name].parameters())
        installed[category.name] = params
    return installed


def build_adapter_checkpoint(base: str, target: str, vocab_size: int = 32768) -> Dict[str, object]:
    """Create a category-layer checkpoint without modifying its source.

    This is reusable checkpoint-management logic. The command-line wrapper
    is used by the optional adapter-management CLI.
    """
    if os.path.abspath(base) == os.path.abspath(target):
        raise ValueError("Refusing to overwrite the source checkpoint.")
    if not os.path.isfile(base):
        raise FileNotFoundError(f"Base checkpoint not found: {base}")

    source = torch.load(base, map_location="cpu", weights_only=False)
    base_state = source.get("model_state_dict", source)
    cfg = source.get("config") if isinstance(source, dict) else None
    if cfg is None:
        raise RuntimeError("Base checkpoint has no saved architecture config; refusing to create a shape-mismatched adapter checkpoint.")
    if isinstance(cfg, dict):
        cfg = NeuroCoreConfig._from_dict(cfg)
    cfg.vocab.vocab_size = vocab_size
    has_legacy_router = any(".router." in key for key in base_state)
    use_real_top1 = bool(getattr(cfg.moe, "real_top1", False) and getattr(cfg.moe, "num_experts", 1) > 1)
    use_legacy_compat = bool(has_legacy_router and not use_real_top1 and getattr(cfg.moe, "num_experts", 1) > 1)
    model = NeuroCoreModel(
        cfg,
        use_mtp=getattr(cfg, "use_mtp", True),
        use_moe=use_real_top1 or use_legacy_compat,
        compatibility_legacy_moe=use_legacy_compat,
    )
    model.load_state_dict(base_state, strict=False)

    registry = AdapterRegistry(REGISTRY_PATH)
    registry.seed_defaults()
    counts = install_category_layers(model, registry.all())
    # Shared weights remain authoritative; newly added category gates stay at
    # their checkpoint/default values.
    model.load_state_dict(base_state, strict=False)
    model.sync_category_gates_from_checkpoint(base_state)
    for name, params in counts.items():
        registry.update_params(name, params)

    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "step_count": source.get("step_count", 0),
        "best_loss": source.get("best_loss", float("inf")),
        "total_tokens": source.get("total_tokens", 0),
        "total_steps": source.get("total_steps", 0),
        "num_layers": len(model.layers),
        "config": cfg,
        "adapter_system": {"mode": "category_layer", "categories": registry.names(),
                           "base": os.path.abspath(base), "vocab_size": vocab_size},
    }, target)
    return {"target": target, "categories": dict(counts), "registry": REGISTRY_PATH}
