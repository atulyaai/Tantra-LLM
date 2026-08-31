"""Initialize a base+specialist-layer checkpoint for the CPU adapter system.

Reads the MoE-2 / 32K base checkpoint produced by the profile converter and
attaches one dedicated specialist layer per default category (cloned from a
base block so the base is undisturbed). Writes:

    Model/CPUMoE2_32K/checkpoint_adapters.pt   (base + category_layers)
    Model/Adapters/registry.json               (category metadata)

The source checkpoint is read-only and is never overwritten.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Tantra.cpu_profiles import build_cpu_model
from Tantra.config import VocabConfig
from Tantra.adapters import AdapterRegistry, DEFAULT_CATEGORIES, install_category_layers, REGISTRY_PATH


def build_adapter_checkpoint(base: str, target: str, vocab_size: int = 32768) -> dict:
    """Build ``target`` = ``base`` + one specialist layer per default category."""
    if os.path.abspath(base) == os.path.abspath(target):
        raise ValueError("Refusing to overwrite the source checkpoint.")
    if not os.path.exists(base):
        raise FileNotFoundError(f"Base checkpoint not found: {base}\nRun the profile converter first.")

    source = torch.load(base, map_location="cpu", weights_only=False)
    base_state = source.get("model_state_dict", source)

    model = build_cpu_model("moe2", attention_kind="alra", vocab_size=vocab_size)
    model.load_state_dict(base_state, strict=False)

    registry = AdapterRegistry(REGISTRY_PATH)
    registry.seed_defaults()

    counts = install_category_layers(model, registry.all())
    model.load_state_dict(base_state, strict=False)  # keep base weights authoritative for shared blocks
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
        "adapter_system": {
            "mode": "category_layer",
            "categories": registry.names(),
            "base": os.path.abspath(base),
            "vocab_size": vocab_size,
        },
    }, target)

    return {
        "target": target,
        "categories": dict(counts),
        "registry": REGISTRY_PATH,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a base + specialist-layer adapter checkpoint.")
    parser.add_argument("--base", default=os.path.join(REPO_ROOT, "Model", "CPUMoE2_32K", "checkpoint_init.pt"),
                        help="Shared base checkpoint (.pt), read-only")
    parser.add_argument("--target", default=os.path.join(REPO_ROOT, "Model", "CPUMoE2_32K", "checkpoint_adapters.pt"),
                        help="Output checkpoint path with category layers")
    parser.add_argument("--vocab-size", type=int, default=32768)
    args = parser.parse_args()
    result = build_adapter_checkpoint(args.base, args.target, args.vocab_size)
    counts = result["categories"]
    print(f"Created base + specialist-layer checkpoint: {result['target']}")
    print(f"Categories: {len(counts)}")
    for name, params in counts.items():
        print(f"  - {name:<18} specialist layer = {params/1e6:.2f}M params")
    print(f"Adapter registry written -> {result['registry']}")
    print("Each category now owns ONE fixed layer; train it with: "
          "python main.py --mode dataset --adapter <name>")


if __name__ == "__main__":
    main()
