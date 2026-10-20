"""On-demand expert (LoRA adapter) loading for CPU inference.

The model starts with every expert registered in the router but no adapter weights in
RAM. `activate(route)` loads the routed adapters from `experts_dir/<name>.pt`, keeps at
most `capacity` resident (least recently used are freed), checks each file against the
loaded base (shapes always; base/tokenizer SHA-256 when the file records them) and then
activates them. Adding experts therefore costs disk, not RAM, beyond the resident cap.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from Tantra.model.tantra_model import TantraModel


class ExpertMismatch(ValueError):
    """An adapter file was built for a different base or tokenizer."""


class ExpertCache:
    def __init__(self, model: TantraModel, experts_dir: str | Path, capacity: int = 8,
                 base_sha256: Optional[str] = None, tokenizer_sha256: Optional[str] = None):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.model, self.dir, self.capacity = model, Path(experts_dir), capacity
        self.base_sha256, self.tokenizer_sha256 = base_sha256, tokenizer_sha256
        self.resident: "OrderedDict[str, int]" = OrderedDict()   # name -> rank, oldest first
        self.loads = 0
        self.evictions = 0

    def available(self) -> List[str]:
        return sorted(p.stem for p in self.dir.glob("*.pt")) if self.dir.exists() else []

    def _read(self, name: str) -> Dict:
        path = self.dir / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"No adapter file for expert '{name}': {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        for key, mine in (("base_sha256", self.base_sha256), ("tokenizer_sha256", self.tokenizer_sha256)):
            theirs = payload.get(key)
            if theirs and mine and theirs != mine:
                raise ExpertMismatch(f"Expert '{name}' was trained for a different {key.split('_')[0]} "
                                     f"({theirs[:12]}... != {mine[:12]}...); retrain it for this base")
        return payload

    def ensure(self, names: Sequence[str]) -> None:
        """Make `names` resident, evicting least-recently-used experts not in `names`."""
        wanted = list(dict.fromkeys(names))
        if len(wanted) > self.capacity:
            raise ValueError(f"Route needs {len(wanted)} experts but the cache holds {self.capacity}")
        for name in wanted:
            if name in self.resident:
                self.resident.move_to_end(name)
                continue
            payload = self._read(name)
            while len(self.resident) >= self.capacity:
                old = next(n for n in self.resident if n not in wanted)
                self.model.detach_lora(old); del self.resident[old]; self.evictions += 1
            self.model.attach_lora(name, payload["state"], int(payload["rank"]))
            self.resident[name] = int(payload["rank"]); self.loads += 1

    def activate(self, route: Sequence[Tuple[str, float]]) -> List[Tuple[str, float]]:
        route = [(n, w) for n, w in route if n in self.model.expert_names]
        self.ensure([n for n, _ in route])
        self.model.set_active(route)
        return route

    def stats(self) -> Dict[str, object]:
        return {"resident": list(self.resident), "capacity": self.capacity, "loads": self.loads,
                "evictions": self.evictions, "available": len(self.available())}
