"""CPU benchmark for the serving path (roadmap M0/M1): load time, RSS, time to first
token and tokens/s, optionally with N synthetic adapters to measure expert overhead.

Synthetic adapters are random rank-r LoRA weights: they measure memory and switching
cost only, never answer quality. Results go to runs/bench/<tag>.json.

  .venv\\Scripts\\python.exe scripts\\bench_cpu.py --tag base64k-fp32
  .venv\\Scripts\\python.exe scripts\\bench_cpu.py --tag base64k-int8 --int8
  .venv\\Scripts\\python.exe scripts\\bench_cpu.py --tag lazy-60 --experts 60 --mode lazy
  .venv\\Scripts\\python.exe scripts\\bench_cpu.py --tag eager-60 --experts 60 --mode eager
"""
import argparse, gc, json, shutil, sys, time
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Tantra.model.tantra_model import TantraModel, quantize_for_cpu, save_expert
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
    if a.int8:
        quantize_for_cpu(model)
    cache = None
    if lazy and a.experts:
        from Tantra.inference.expert_cache import ExpertCache
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
    result = {'tag': a.tag, 'model_dir': str(model_dir), 'int8': a.int8, 'threads': a.threads, 'mode': a.mode,
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


if __name__ == '__main__':
    main()
