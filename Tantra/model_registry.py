"""Immutable, licence-gated registry for external pretrained checkpoints."""
from __future__ import annotations
import hashlib, json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    provider: str
    revision: str
    architecture: str
    parameter_count: int
    license_id: str
    license_status: str
    tokenizer_id: str
    intended_use: str = "research"
    trust_remote_code: bool = False

    def validate(self) -> None:
        for name in ("model_id", "provider", "revision", "architecture", "license_id", "tokenizer_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} is required")
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision.lower()):
            raise ValueError("revision must be a full immutable 40-character commit SHA")
        if self.parameter_count <= 0:
            raise ValueError("parameter_count must be positive")
        if self.license_status != "approved":
            raise ValueError("model licence must be explicitly approved before loading")
        if self.trust_remote_code:
            raise ValueError("remote model code is disabled by default")

class ModelRegistry:
    def __init__(self, specs: list[ModelSpec] | None = None):
        self._specs: dict[str, ModelSpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: ModelSpec) -> None:
        spec.validate()
        if spec.model_id in self._specs:
            raise ValueError(f"duplicate model_id: {spec.model_id}")
        self._specs[spec.model_id] = spec

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._specs[model_id]
        except KeyError as exc:
            raise KeyError(f"unregistered pretrained model: {model_id}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "models": [asdict(x) for x in self._specs.values()]}

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelRegistry":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1:
            raise ValueError("unsupported registry schema")
        return cls([ModelSpec(**item) for item in raw["models"]])

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def verify_checkpoint_manifest(root: str | Path, manifest: dict[str, str]) -> None:
    base = Path(root).resolve()
    if not manifest:
        raise ValueError("checkpoint manifest cannot be empty")
    for relative, expected in manifest.items():
        target = (base / relative).resolve()
        if base not in target.parents or not target.is_file():
            raise ValueError(f"invalid or missing checkpoint file: {relative}")
        if sha256_file(target).lower() != expected.lower():
            raise ValueError(f"checkpoint hash mismatch: {relative}")
