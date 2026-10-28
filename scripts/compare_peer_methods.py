"""Rank every PEER CPU compression path on the same weights.

Score is size, CPU tokens/sec, and error against the float module on a
held-out batch. Compaction is exact only for experts that were activated
during the fit batches, so seen-batch error is reported separately and is
not used to pass a method.

    PYTHONPATH=. python scripts/compare_peer_methods.py
    PYTHONPATH=. python scripts/compare_peer_methods.py --d 128 --n-keys 32 --err-tol 0.01

Does not read or write checkpoints. Random expert rows are not low-rank;
a second, synthetic low-rank table shows when that path is actually exact.
"""
from __future__ import annotations

import argparse
import time

import torch

from Tantra.model.peer import ProductKeyExperts


def _clone(src: ProductKeyExperts, track: bool = False) -> ProductKeyExperts:
    d = src.down.embedding_dim
    module = ProductKeyExperts(
        d_model=d,
        n_keys=src.n_keys,
        heads=src.heads,
        top_k=src.top_k,
        key_dim=src.key_dim,
        sparse=True,
        track_activations=track,
    )
    with torch.no_grad():
        module.down.weight.copy_(src.down.weight)
        module.up.weight.copy_(src.up.weight)
        module.query.weight.copy_(src.query.weight)
        module.keys.copy_(src.keys)
        if track:
            module.activation_counts.copy_(src.activation_counts)
    return module


def _forward_error(module, x, ref):
    module.eval()
    with torch.no_grad():
        y = module(x)
    diff = (y - ref).float()
    max_abs = float(diff.abs().max())
    denom = ref.float().pow(2).mean().sqrt().clamp_min(1e-8)
    rel_rmse = float(diff.pow(2).mean().sqrt() / denom)
    return max_abs, rel_rmse


def _tok_s(module, x, repeats: int) -> float:
    module.eval()
    with torch.no_grad():
        for _ in range(3):
            module(x)
        t0 = time.perf_counter()
        for _ in range(repeats):
            module(x)
        dt = time.perf_counter() - t0
    return (x.shape[0] * repeats) / max(dt, 1e-9)


def _row(name, module, x_unseen, ref_unseen, x_seen, ref_seen, repeats, note=""):
    max_abs, rel = _forward_error(module, x_unseen, ref_unseen)
    seen_abs, seen_rel = _forward_error(module, x_seen, ref_seen)
    return {
        "method": name,
        "bytes": module.stored_bytes(),
        "max_abs": max_abs,
        "rel_rmse": rel,
        "seen_rel_rmse": seen_rel,
        "seen_max_abs": seen_abs,
        "tok_s": _tok_s(module, x_unseen, repeats),
        "note": note,
    }


def _print_table(rows, float_bytes, err_tol):
    print(
        f"{'method':<22} {'bytes':>10} {'ratio':>8} {'rel_rmse':>10} "
        f"{'max_abs':>10} {'seen_rmse':>10} {'tok/s':>10}  pass"
    )
    passed = []
    for row in rows:
        ratio = float_bytes / max(row["bytes"], 1)
        ok = row["method"] == "float" or row["rel_rmse"] <= err_tol
        if ok and row["method"] != "float":
            passed.append(row)
        flag = "yes" if ok else "no"
        print(
            f"{row['method']:<22} {row['bytes']:>10} {ratio:>8.2f} {row['rel_rmse']:>10.4f} "
            f"{row['max_abs']:>10.4f} {row['seen_rel_rmse']:>10.4f} {row['tok_s']:>10.0f}  {flag}"
            + (f"  {row['note']}" if row["note"] else "")
        )
    return passed


def _recommend(passed, float_tok):
    if not passed:
        print("verdict: no compressed path passed the held-out error tolerance")
        return
    smallest = min(passed, key=lambda r: r["bytes"])
    fastest = max(passed, key=lambda r: r["tok_s"])
    print(
        f"verdict: smallest passing path is {smallest['method']} "
        f"({smallest['bytes']} bytes, rel_rmse {smallest['rel_rmse']:.4f}, "
        f"{smallest['tok_s']:.0f} tok/s)"
    )
    if fastest["method"] != smallest["method"]:
        print(
            f"         fastest passing path is {fastest['method']} "
            f"({fastest['tok_s']:.0f} tok/s vs float {float_tok:.0f})"
        )
    print("         seen_rmse is not a pass. It is exactness on the fit batch only.")


def _structured_control(d, n_keys, heads, top_k, key_dim, rank, repeats):
    """Table built as codes @ basis. Low-rank at `rank` must be nearly exact."""
    torch.manual_seed(1)
    module = ProductKeyExperts(
        d_model=d, n_keys=n_keys, heads=heads, top_k=top_k, key_dim=key_dim,
    )
    basis = torch.randn(rank, d)
    code = torch.randn(module.n_experts, rank)
    with torch.no_grad():
        down = code @ basis
        up = code @ basis
        # Same RMS as a normal expert row, so INT8 error is comparable to the table above.
        down = down / down.float().pow(2).mean().sqrt().clamp_min(1e-8) * (d ** -0.5)
        up = up / up.float().pow(2).mean().sqrt().clamp_min(1e-8) * 0.02
        module.down.weight.copy_(down)
        module.up.weight.copy_(up)
    x = torch.randn(32, d)
    module.eval()
    with torch.no_grad():
        ref = module(x).clone()
    factored = _clone(module)
    info = factored.factor_low_rank_(rank=rank)
    max_abs, rel = _forward_error(factored, x, ref)
    packed = _clone(module)
    packed.factor_low_rank_(rank=rank)
    packed.quantize_int8_()
    q_abs, q_rel = _forward_error(packed, x, ref)
    print(
        "structured rank "
        f"{rank}: factor rel_rmse {rel:.3e} weight_down {info['down_rel_rmse']:.3e} "
        f"bytes {factored.stored_bytes()} | "
        f"factor+int8 rel_rmse {q_rel:.3e} max_abs {q_abs:.3e} "
        f"bytes {packed.stored_bytes()} tok/s {_tok_s(packed, x, repeats):.0f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--n-keys", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--key-dim", type=int, default=64)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--fit-batches", type=int, default=32, help="batches counted before compact-many")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--err-tol", type=float, default=0.01, help="max held-out relative RMSE")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.top_k > args.n_keys:
        raise SystemExit("top-k cannot exceed n-keys")
    if args.key_dim % 2:
        raise SystemExit("key-dim must be even")

    torch.manual_seed(args.seed)
    d = args.d
    base = ProductKeyExperts(
        d_model=d,
        n_keys=args.n_keys,
        heads=args.heads,
        top_k=args.top_k,
        key_dim=args.key_dim,
        track_activations=True,
    )
    with torch.no_grad():
        base.up.weight.normal_(std=0.02)

    x_seen = torch.randn(args.batch, d)
    x_unseen = torch.randn(args.batch, d)
    base.eval()
    with torch.no_grad():
        ref_seen = base(x_seen).clone()
        ref_unseen = base(x_unseen).clone()

    base.train()
    base(x_seen)
    counts_one = base.activation_counts.clone()
    for _ in range(args.fit_batches - 1):
        base(torch.randn(args.batch, d))
    counts_many = base.activation_counts.clone()
    base.eval()

    float_bytes = base.stored_bytes()
    rows = []
    rows.append(_row("float", base, x_unseen, ref_unseen, x_seen, ref_seen, args.repeats))

    int8 = _clone(base)
    qinfo = int8.quantize_int8_()
    rows.append(_row(
        "int8", int8, x_unseen, ref_unseen, x_seen, ref_seen, args.repeats,
        note=f"pack {qinfo['ratio']:.2f}x",
    ))

    one = _clone(base, track=True)
    one.activation_counts.copy_(counts_one)
    cone = one.compact_used_()
    rows.append(_row(
        "compact-1batch", one, x_unseen, ref_unseen, x_seen, ref_seen, args.repeats,
        note=f"kept {cone['kept']}/{base.n_experts}",
    ))

    one_q = _clone(base, track=True)
    one_q.activation_counts.copy_(counts_one)
    one_q.compact_used_()
    one_q.quantize_int8_()
    rows.append(_row(
        "compact-1batch+int8", one_q, x_unseen, ref_unseen, x_seen, ref_seen, args.repeats,
    ))

    many = _clone(base, track=True)
    many.activation_counts.copy_(counts_many)
    cmany = many.compact_used_()
    rows.append(_row(
        f"compact-{args.fit_batches}batch", many, x_unseen, ref_unseen, x_seen, ref_seen, args.repeats,
        note=f"kept {cmany['kept']}/{base.n_experts}",
    ))

    many_q = _clone(base, track=True)
    many_q.activation_counts.copy_(counts_many)
    many_q.compact_used_()
    many_q.quantize_int8_()
    rows.append(_row(
        f"compact-{args.fit_batches}+int8", many_q, x_unseen, ref_unseen, x_seen, ref_seen, args.repeats,
    ))

    for div in (2, 4, 8, 16):
        rank = max(1, d // div)
        if rank >= d:
            continue
        low = _clone(base)
        info = low.factor_low_rank_(rank=rank)
        rows.append(_row(
            f"low-rank-{rank}", low, x_unseen, ref_unseen, x_seen, ref_seen, args.repeats,
            note=f"weight_rmse {info['down_rel_rmse']:.3f}/{info['up_rel_rmse']:.3f}",
        ))
        both = _clone(base)
        both.factor_low_rank_(rank=rank)
        both.quantize_int8_()
        rows.append(_row(
            f"low-rank-{rank}+int8", both, x_unseen, ref_unseen, x_seen, ref_seen, args.repeats,
        ))

    print(
        f"experts {base.n_experts}  d {d}  held-out rel_rmse tol {args.err_tol}  "
        f"fit_batches {args.fit_batches}"
    )
    passed = _print_table(rows, float_bytes, args.err_tol)
    _recommend(passed, rows[0]["tok_s"])
    print("control (this table IS low-rank, unlike the random one above):")
    _structured_control(
        d=min(d, 64),
        n_keys=min(args.n_keys, 16),
        heads=1,
        top_k=min(2, args.top_k),
        key_dim=args.key_dim,
        rank=8 if min(d, 64) > 8 else 2,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
