"""Live view of a chat-training log: progress, validation loss and reply-quality scores.

  .venv\\Scripts\\python.exe scripts\\watch_training.py runs\\chat-en-v7-train.log [--total 2970]

Refreshes every 15 s; Ctrl+C to quit. Reads the log only, never touches the run.
"""
import argparse, json, os, re, time
from pathlib import Path


def summary(log: Path, total: int | None) -> str:
    lines = log.read_text(encoding='utf-8', errors='replace').splitlines()
    updates = [int(m.group(1)) for l in lines for m in [re.match(r'Chat update (\d+)', l)] if m]
    evals = [json.loads(l) for l in lines if l.startswith('{"update"')]
    done = updates[-1] if updates else 0
    out = [f"{log}  (updated {time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(log)))})",
           f"update {done}" + (f" / {total} ({done / total:.0%})" if total else "")]
    best = max((e for e in evals if e.get('reply_score') is not None), key=lambda e: e['reply_score'], default=None)
    if best:
        out.append(f"best reply score {best['reply_score']:.3f} at update {best['update']} ({best.get('reply_passed')}/32 passed)")
    out.append(f"{'update':>7} {'val loss':>9} {'reply':>7} {'passed':>7} {'no-gain':>8}")
    for e in evals[-12:]:
        out.append(f"{e['update']:>7} {e['validation']['en']['loss']:>9.3f} {e.get('reply_score', 0) or 0:>7.3f} "
                   f"{str(e.get('reply_passed', '-')):>7} {str(e.get('reply_bad', e.get('bad_evals', '-'))) + '/' + str(e.get('patience', '?')):>8}")
    # A resumed run appends to the same log: only an exit line after the last update means "finished".
    last_update = max((i for i, l in enumerate(lines) if l.startswith('Chat update')), default=-1)
    ends = [(i, l) for i, l in enumerate(lines) if l.startswith(('train exit', 'resume exit', 'Traceback'))]
    if ends and ends[-1][0] > last_update:
        out.append('finished: ' + ends[-1][1])
    return '\n'.join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('log', type=Path); p.add_argument('--total', type=int); p.add_argument('--once', action='store_true')
    a = p.parse_args()
    while True:
        text = summary(a.log, a.total)
        print('\x1b[2J\x1b[H' + text if not a.once else text, flush=True)
        if a.once or 'finished:' in text:
            break
        time.sleep(15)


if __name__ == '__main__':
    main()
