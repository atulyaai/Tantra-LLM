#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
import urllib.request


API = "https://api.github.com/repos/atulyaai/Tantra-LLM"


def request(method: str, url: str, token: str, data: dict | None = None, accept: str | None = None):
    body = None
    headers = {
        "Authorization": f"token {token}",
        "User-Agent": "tantra-llm-config",
        "Accept": accept or "application/vnd.github+json",
    }
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    if not token:
        print("ERROR: Set GITHUB_TOKEN env var before running.", file=sys.stderr)
        sys.exit(1)

    # 1) Description, homepage, default branch
    desc = (
        "\ud83e\udde0 Proprietary Multimodal Cognitive Architecture - SpikingBrain + Long-VITA + Whisper "
        "with dynamic context, fusion layers, hybrid memory, and adaptive personality. Not a chatbot. A cognitive mind."
    )
    patch_body = {
        "description": desc,
        "homepage": "https://github.com/atulyaai/Tantra-LLM",
        "default_branch": "main",
    }
    request("PATCH", API, token, patch_body)

    # 2) Topics
    topics = [
        "tantra-llm",
        "multimodal-ai",
        "spikingbrain",
        "cognitive-architecture",
        "fusion-layers",
        "adaptive-personality",
        "hybrid-memory",
        "proprietary-ai",
        "reasoning-system",
        "neural-architecture",
        "dynamic-context",
        "episodic-memory",
        "semantic-graph",
        "personality-modes",
        "vision-encoding",
        "audio-encoding",
    ]
    request("PUT", f"{API}/topics", token, {"names": topics}, accept="application/vnd.github.mercy-preview+json")

    # 3) Show summary
    repo = request("GET", API, token, None)
    tops = request("GET", f"{API}/topics", token, None, accept="application/vnd.github.mercy-preview+json")
    print("Updated:")
    print(" description:", repo.get("description"))
    print(" homepage:", repo.get("homepage"))
    print(" default_branch:", repo.get("default_branch"))
    print(" topics:", ", ".join(tops.get("names", [])))


if __name__ == "__main__":
    main()


