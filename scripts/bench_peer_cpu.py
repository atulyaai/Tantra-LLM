"""CPU size and speed check for float vs INT8 vs low-rank PEER tables.

    PYTHONPATH=. python scripts/bench_peer_cpu.py

Low-rank ~100:1 only preserves the function when the expert table is actually
low-rank. Random tables are not. INT8 is ~4x with a small error. Compaction
drops experts that a real run never activates and is exact for the rest.
"""
import time

import torch
from Tantra.model.peer import ProductKeyExperts


def _time_forward(module, x, repeats=20):
    module.eval()
    with torch.no_grad():
        for _ in range(3):
            module(x)
        t0 = time.perf_counter()
        for _ in range(repeats):
            module(x)
        dt = time.perf_counter() - t0
    return (x.shape[0] * repeats) / dt


def main():
    torch.manual_seed(0)
    d, n_keys = 128, 32
    x = torch.randn(64, d)

    base = ProductKeyExperts(d_model=d, n_keys=n_keys, heads=2, top_k=4, track_activations=True)
    with torch.no_grad():
        base.up.weight.normal_(std=0.02)
    float_bytes = base.stored_bytes()
    float_tok_s = _time_forward(base, x)
    with torch.no_grad():
        ref = base(x).clone()

    int8 = ProductKeyExperts(d_model=d, n_keys=n_keys, heads=2, top_k=4)
    with torch.no_grad():
        int8.down.weight.copy_(base.down.weight)
        int8.up.weight.copy_(base.up.weight)
        int8.query.weight.copy_(base.query.weight)
        int8.keys.copy_(base.keys)
        q = int8.quantize_int8_()
        q_err = (int8(x) - ref).abs().max().item()
    q_tok_s = _time_forward(int8, x)

    low = ProductKeyExperts(d_model=d, n_keys=n_keys, heads=2, top_k=4)
    with torch.no_grad():
        low.down.weight.copy_(base.down.weight)
        low.up.weight.copy_(base.up.weight)
        low.query.weight.copy_(base.query.weight)
        low.keys.copy_(base.keys)
        lr = low.factor_low_rank_(rank=max(1, d // 16))
        lr_err = (low(x) - ref).abs().max().item()
    lr_tok_s = _time_forward(low, x)

    base.train()
    base(x)
    compact = base.compact_used_(min_activations=1)
    with torch.no_grad():
        compact_err = (base(x) - ref).abs().max().item()

    print({
        "experts": n_keys * n_keys,
        "float_bytes": float_bytes,
        "float_tok_s": round(float_tok_s, 1),
        "int8_bytes": q["stored_bytes"],
        "int8_ratio": round(q["ratio"], 2),
        "int8_max_abs_err": q_err,
        "int8_tok_s": round(q_tok_s, 1),
        "low_rank": lr,
        "low_rank_max_abs_err_on_random": lr_err,
        "low_rank_tok_s": round(lr_tok_s, 1),
        "compact": compact,
        "compact_max_abs_err": compact_err,
    })


if __name__ == "__main__":
    main()
