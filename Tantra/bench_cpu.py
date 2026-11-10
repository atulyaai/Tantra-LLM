"""CPU benchmarks.

Serving path: load time, RSS, time to first token and tokens/s, optionally with
N synthetic adapters to measure expert overhead (roadmap M0/M1).

  python -m Tantra.bench_cpu --tag base64k-fp32
  python -m Tantra.bench_cpu --tag base64k-int8 --int8
  python -m Tantra.bench_cpu --tag lazy-60 --experts 60 --mode lazy
  python -m Tantra.bench_cpu --tag eager-60 --experts 60 --mode eager

PEER table size and speed: float vs INT8 vs low-rank vs compaction:

  python -m Tantra bench_peer_cpu

Synthetic adapters are random rank-r LoRA weights: they measure memory and
switching cost only, never answer quality. Results go to runs/bench/<tag>.json.
"""
import argparse, gc, json, shutil, sys, time
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Tantra.train import TantraModel, quantize_for_cpu, save_expert, ProductKeyExperts
from Tantra.tokenizer import TantraTokenizer

PROMPTS = ['Hi! How are you today?', 'It is raining here.', 'Tell me a fun fact.']


def rss_mb():
    return psutil.Process().memory_info().rss / 2**20


def serve_dir(base_dir: Path, experts: int, rank: int) -> Path:
    """A NEW folder with the base (plus router-registered synthetic experts); sources are never modified."""
    out = ROOT / 'runs' / 'bench' / f'serve-64k-synthetic-{experts}x{rank}'
    if (out / 'base.pt').exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_dir / 'tokenizer.json', out / 'tokenizer.json')
    ck = torch.load(base_dir / 'base.pt', map_location='cpu', weights_only=False)
    model = TantraModel.load(base_dir / 'base.pt')
    torch.manual_seed(0)
    for i in range(experts):
        name = f'synthetic{i:03d}'
        model.add_expert(name, rank)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n.endswith(f'lora_B.{name}'):
                    p.normal_(0, 0.01)
        save_expert(model, name, out / 'experts')
        model.detach_lora(name)  # keep RAM flat while building
    torch.save({'config': ck['config'], 'state': {k: v for k, v in model.state_dict().items() if 'lora_' not in k},
                'router_names': model.router.names, 'expert_ranks': {n: rank for n in model.router.names},
                'synthetic_experts': experts}, out / 'base.pt')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base', type=Path, default=ROOT / 'runs/kaggle-output/base-training')
    p.add_argument('--tag', required=True); p.add_argument('--int8', action='store_true')
    p.add_argument('--int8-head-only', action='store_true', help='quantise only the output head (less drift)')
    p.add_argument('--experts', type=int, default=0); p.add_argument('--rank', type=int, default=8)
    p.add_argument('--mode', choices=['lazy', 'eager'], default='lazy'); p.add_argument('--cache', type=int, default=8)
    p.add_argument('--tokens', type=int, default=64); p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    model_dir = serve_dir(a.base, a.experts, a.rank) if a.experts else a.base
    gc.collect(); rss0 = rss_mb(); t0 = time.perf_counter()
    tok = TantraTokenizer.load(model_dir / 'tokenizer.json')
    lazy = a.mode == 'lazy'
    model = TantraModel.load(model_dir / 'base.pt', None if lazy else model_dir / 'experts', lazy=lazy).eval()
    if a.int8 or a.int8_head_only:
        quantize_for_cpu(model, layers=not a.int8_head_only)
    cache = None
    if lazy and a.experts:
        from Tantra.main import ExpertCache
        cache = ExpertCache(model, model_dir / 'experts', a.cache)
    load_s = time.perf_counter() - t0; rss_loaded = rss_mb()
    runs = []
    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS * 2):
            route = [(model.expert_names[(i * 7) % len(model.expert_names)], 1.0)] if a.experts else []
            t_switch = time.perf_counter()
            (cache.activate(route) if cache else model.set_active(route))
            switch_s = time.perf_counter() - t_switch
            ids, _ = tok.encode_chat([{'role': 'user', 'content': prompt}], add_generation_prompt=True)
            t = time.perf_counter(); first = None; n = 0
            for _ in model.generate(torch.tensor([ids]), a.tokens, temperature=0):
                n += 1
                if first is None:
                    first = time.perf_counter() - t
            dt = time.perf_counter() - t
            runs.append({'prompt': prompt, 'tokens': n, 'ttft_s': round(first or 0, 4), 'tok_per_s': round(n / dt, 1),
                         'expert_switch_s': round(switch_s, 4)})
    warm = runs[1:]  # first run includes one-off warm-up
    result = {'tag': a.tag, 'model_dir': str(model_dir), 'int8': a.int8, 'int8_head_only': a.int8_head_only, 'threads': a.threads, 'mode': a.mode,
              'synthetic_experts': a.experts, 'rank': a.rank, 'cache': a.cache if lazy else None,
              'base_params_M': round(model.num_params() / 1e6, 2), 'load_s': round(load_s, 2),
              'rss_mb_before_load': round(rss0), 'rss_mb_loaded': round(rss_loaded), 'rss_mb_peak_after_runs': round(rss_mb()),
              'tok_per_s': round(sum(r['tok_per_s'] for r in warm) / len(warm), 1),
              'ttft_s': round(sum(r['ttft_s'] for r in warm) / len(warm), 4),
              'expert_switch_s_max': max(r['expert_switch_s'] for r in runs),
              'experts_resident': len(cache.resident) if cache else (len(model.expert_names) if a.experts else 0),
              'runs': runs, 'note': 'Synthetic adapters measure memory/latency only, not quality.' if a.experts else None}
    out = ROOT / 'runs' / 'bench' / f'{a.tag}.json'; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'runs'}, indent=2))


# ───────────  PEER table size and speed (float / INT8 / low-rank / compaction)  ───────────

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


def bench_peer_cpu():
    """CPU size and speed check for float vs INT8 vs low-rank PEER tables."""
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


if __name__ == '__main__':
    main()