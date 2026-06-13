"""Practical multimodal context bridge for NP-DNA.

This module does not claim learned multimodal embeddings. It converts file and
structured inputs into explicit text context that NP-DNA can consume today,
while the deeper encoder-to-embedding path matures.
"""

from __future__ import annotations

import json
import struct
import wave
from pathlib import Path
from typing import Any


def _png_size(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
        return struct.unpack(">II", header[16:24])
    return None


def _jpeg_size(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as handle:
        data = handle.read()
    if not data.startswith(b"\xff\xd8"):
        return None
    idx = 2
    while idx < len(data) - 9:
        if data[idx] != 0xFF:
            idx += 1
            continue
        marker = data[idx + 1]
        idx += 2
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if idx + 7 <= len(data):
                height = int.from_bytes(data[idx + 3 : idx + 5], "big")
                width = int.from_bytes(data[idx + 5 : idx + 7], "big")
                return width, height
            return None
        if idx + 2 > len(data):
            return None
        segment_len = int.from_bytes(data[idx : idx + 2], "big")
        idx += max(2, segment_len)
    return None


def describe_image(path: str | Path) -> str:
    image = Path(path)
    size = None
    if image.exists():
        suffix = image.suffix.lower()
        if suffix == ".png":
            size = _png_size(image)
        elif suffix in {".jpg", ".jpeg"}:
            size = _jpeg_size(image)
    parts = [
        "Modality: image",
        f"File: {image.name}",
        f"Extension: {image.suffix.lower() or 'unknown'}",
    ]
    if image.exists():
        parts.append(f"Bytes: {image.stat().st_size}")
    if size:
        parts.append(f"Dimensions: {size[0]}x{size[1]}")
    return "\n".join(parts)


def describe_audio(path: str | Path) -> str:
    audio = Path(path)
    parts = [
        "Modality: audio",
        f"File: {audio.name}",
        f"Extension: {audio.suffix.lower() or 'unknown'}",
    ]
    if audio.exists():
        parts.append(f"Bytes: {audio.stat().st_size}")
        if audio.suffix.lower() == ".wav":
            with wave.open(str(audio), "rb") as handle:
                frames = handle.getnframes()
                rate = handle.getframerate()
                duration = frames / rate if rate else 0.0
                parts.extend([
                    f"Channels: {handle.getnchannels()}",
                    f"Sample rate: {rate}",
                    f"Duration seconds: {duration:.3f}",
                ])
    return "\n".join(parts)


def describe_structured(data: dict[str, Any] | list[Any] | str | Path) -> str:
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            value = json.loads(str(data))
    else:
        value = data

    preview = json.dumps(value, ensure_ascii=True, sort_keys=True)
    if len(preview) > 1200:
        preview = preview[:1200] + "..."
    if isinstance(value, dict):
        shape = f"object with {len(value)} keys"
    elif isinstance(value, list):
        shape = f"array with {len(value)} items"
    else:
        shape = type(value).__name__
    return "\n".join(["Modality: structured", f"Shape: {shape}", f"JSON: {preview}"])


def build_multimodal_prompt(
    instruction: str,
    *,
    image: str | Path | None = None,
    audio: str | Path | None = None,
    structured: dict[str, Any] | list[Any] | str | Path | None = None,
) -> str:
    contexts = []
    if image is not None:
        contexts.append(describe_image(image))
    if audio is not None:
        contexts.append(describe_audio(audio))
    if structured is not None:
        contexts.append(describe_structured(structured))

    context = "\n\n".join(contexts) if contexts else "No multimodal context provided."
    return f"System: Use the provided multimodal context.\n{context}\n\nUser: {instruction}\nAssistant:"
