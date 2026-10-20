"""Compare serving folders through the real CPU engine on fresh prompts (development check).

  .venv\\Scripts\\python.exe scripts\\compare_serving.py runs/serve-64k-v0 runs/serve-64k-raw-v1

Prints each prompt with every folder's route, reply and first-token latency, and writes
runs/serving-compare.json. Greedy decoding; prompts never appear in the training data.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from compare_chat_pilot import FRESH
from Tantra.inference.inference_engine import TantraEngine


def main():
    dirs = [Path(d) for d in sys.argv[1:]]
    engines = {str(d): TantraEngine(d, max_experts=2) for d in dirs}
    results = {}
    for q in FRESH:
        results[q] = {}
        print(repr(q))
        for name, eng in engines.items():
            reply = eng.chat([{'role': 'user', 'content': q}], temperature=0, max_new_tokens=40)
            route = ', '.join(f'{n}:{w:.2f}' for n, w in eng.last_route) or 'base'
            results[q][name] = {'route': route, 'reply': reply, 'ttft_s': eng.last_timing['ttft_s']}
            print(f'   {Path(name).name:<18} [{route}] {reply}')
    out = ROOT / 'runs' / 'serving-compare.json'
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
