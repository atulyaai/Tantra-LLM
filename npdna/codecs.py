"""Frozen multimodal codec registry.

Codecs are tokenizer-like adapters around external audio/image/video systems.
They are referenced by checkpoints, but their weights are never part of NP-DNA
training. Until real codec packages are configured, this module fails clearly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .config import CodecConfig

Modality = Literal["audio", "image", "video"]


@dataclass(frozen=True)
class FrozenCodecRef:
    modality: Modality
    uri: str | None
    trainable: bool = False
    frozen: bool = True

    @property
    def available(self) -> bool:
        return bool(self.uri)


class FrozenCodecRegistry:
    """Lookup table for tokenizer-like frozen multimodal codecs."""

    def __init__(self, config: CodecConfig):
        self.refs = {
            "audio": FrozenCodecRef("audio", config.audio_codec),
            "image": FrozenCodecRef("image", config.image_codec),
            "video": FrozenCodecRef("video", config.video_codec),
        }

    @classmethod
    def from_config(cls, config: CodecConfig) -> "FrozenCodecRegistry":
        return cls(config)

    @classmethod
    def default(cls) -> "FrozenCodecRegistry":
        return cls(CodecConfig())

    def describe(self) -> dict[str, dict]:
        return {
            name: {
                **vars(ref),
                "available": ref.available,
            }
            for name, ref in self.refs.items()
        }

    def encode(self, modality: Modality, payload: bytes) -> list[int]:
        ref = self.refs[modality]
        if not ref.available:
            raise NotImplementedError(f"No frozen {modality} codec is configured.")
        raise NotImplementedError(f"Frozen {modality} codec adapter is referenced at {ref.uri}, but not installed.")

    def decode(self, modality: Modality, tokens: list[int]) -> bytes:
        ref = self.refs[modality]
        if not ref.available:
            raise NotImplementedError(f"No frozen {modality} codec is configured.")
        raise NotImplementedError(f"Frozen {modality} codec adapter is referenced at {ref.uri}, but not installed.")

