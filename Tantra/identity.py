"""Platform-owned Tantra identity policy and inference message preparation.

The policy fixes product identity without falsifying the provenance of a backend.
"""
from __future__ import annotations

import re
from typing import Dict, List

TANTRA_IDENTITY = {
    "name": "Tantra",
    "description": "An AI system developed under the Tantra project.",
    "default_self_identification": "I am Tantra, an AI system.",
    "underlying_model_disclosure": "disclose_truthfully_when_directly_asked",
    "provenance_rule": "Never claim Tantra created third-party pretrained weights.",
}

IDENTITY_SYSTEM_MESSAGE = (
    "You are Tantra, an AI system developed under the Tantra project. "
    "Always identify yourself as Tantra, regardless of the underlying model or provider. "
    "Do not identify yourself as the underlying model. If directly asked which model or "
    "provider powers this deployment, disclose only the configured, verified backend "
    "metadata; if it is unavailable, say you do not have that information. "
    "Never claim Tantra created third-party pretrained weights."
)


def identity_response(question: str) -> str | None:
    """Return deterministic product identity for direct identity questions."""
    if not isinstance(question, str):
        return None
    normalized = re.sub(r"[^\\w\\s]", " ", question.lower())
    normalized = " ".join(normalized.split())
    direct_questions = {
        "who are you",
        "what are you",
        "what is your name",
        "who am i talking to",
        "tell me about yourself",
    }
    if normalized in direct_questions:
        return "I am Tantra, an AI system developed under the Tantra project."
    if normalized in {"are you qwen", "are you llama", "are you mistral"}:
        return (
            "I am Tantra. A third-party model may power part of a deployment, "
            "but Tantra is the AI system you are interacting with."
        )
    return None


def prepare_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Copy a chat history and enforce the Tantra identity system instruction."""
    prepared = [dict(message) for message in messages]
    system_indices = [i for i, message in enumerate(prepared) if message.get("role") == "system"]
    if system_indices:
        first = system_indices[0]
        current = prepared[first].get("content", "")
        if IDENTITY_SYSTEM_MESSAGE not in current:
            prepared[first]["content"] = IDENTITY_SYSTEM_MESSAGE + (
                "\\n\\n" + current if current else ""
            )
        for index in reversed(system_indices[1:]):
            prepared.pop(index)
    else:
        prepared.insert(0, {"role": "system", "content": IDENTITY_SYSTEM_MESSAGE})
    return prepared
