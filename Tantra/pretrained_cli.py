"""CLI for inspecting and smoke-testing registered pretrained checkpoints."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from Tantra.model_registry import ModelRegistry
from Tantra.model_gateway import ModelGateway

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m Tantra.pretrained_cli")
    sub = parser.add_subparsers(dest="command", required=True)
    info = sub.add_parser("info")
    info.add_argument("--registry", default="configs/models/registry.json")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--registry", default="configs/models/registry.json")
    smoke.add_argument("--model-id", required=True)
    smoke.add_argument("--checkpoint", required=True, help="local snapshot directory; no download is attempted")
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args(argv)
    registry = ModelRegistry.from_file(args.registry)
    if args.command == "info":
        print(json.dumps(registry.to_dict(), indent=2))
        return 0
    spec = registry.get(args.model_id)
    if not Path(args.checkpoint).is_dir():
        parser.error("checkpoint directory does not exist; download the pinned snapshot explicitly first")
    gateway = ModelGateway(spec, args.checkpoint, device=args.device)
    start = time.perf_counter()
    gateway.load()
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    answer = gateway.generate([{"role": "user", "content": "Introduce yourself as Tantra in one sentence."}],
                              max_new_tokens=args.max_new_tokens)
    generation_seconds = time.perf_counter() - start
    print(json.dumps({"model_id": spec.model_id, "revision": spec.revision,
        "license": spec.license_id, "load_seconds": round(load_seconds, 3),
        "generation_seconds": round(generation_seconds, 3), "output": answer,
        "status": "runtime_smoke_only_not_quality_evaluation"}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
