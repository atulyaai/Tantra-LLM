# Tantra live training dashboard for a Kaggle notebook cell.
# Starts (or resumes) training in the background and redraws charts every 30 s.
# Stopping this cell does NOT stop training; to stop cleanly run:
#   open(OUT + '/stop.request', 'w').close()
import glob, json, os, shutil, subprocess, sys, time
import matplotlib.pyplot as plt
from IPython.display import clear_output, display, HTML

SRC = next(os.path.dirname(p) for p in glob.glob('/kaggle/input/**/requirements.txt', recursive=True))
ROOT = '/kaggle/working/tantra'
OUT = ROOT + '/runs/pretraining-20m-v1/base-training'
if not os.path.exists(ROOT):
    shutil.copytree(SRC, ROOT)

def running():
    lock = OUT + '/training.lock'
    if not os.path.exists(lock):
        return False
    try:
        os.kill(int(open(lock).read().strip()), 0); return True
    except (OSError, ValueError):
        return False

if not running():
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tokenizers', 'psutil', 'rich'], check=False)
    env = dict(os.environ, TANTRA_DEVICE='cuda', PYTHONUNBUFFERED='1')
    log = open('/kaggle/working/train.log', 'a')
    subprocess.Popen([sys.executable, '-m', 'Tantra.pretrain', '--resume', '--microbatch', '16',
                      '--threads', '4', '--early-stopping-patience', '10'],
                     cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    time.sleep(20)

TOTAL_UPDATES = 4639
START_UPDATE = 1951

def load(path):
    rows = []
    if os.path.exists(path):
        for line in open(path, encoding='utf-8'):
            try: rows.append(json.loads(line))
            except ValueError: pass
    return rows

def card(label, value, color):
    return (f'<div style="flex:1;min-width:150px;padding:12px 16px;border-radius:10px;'
            f'background:{color}22;border-left:5px solid {color}">'
            f'<div style="font-size:12px;color:#666">{label}</div>'
            f'<div style="font-size:22px;font-weight:700">{value}</div></div>')

def draw():
    rows = load(OUT + '/metrics.jsonl')
    if not rows:
        print('Waiting for first metrics...'); return
    last = rows[-1]
    u = [r['update'] for r in rows]
    loss = [r['loss'] for r in rows]
    val = [r for r in rows if 'validation' in r]
    recent = [r for r in rows if r['update'] > START_UPDATE and 'timestamp' in r]
    speed = 0
    if len(recent) > 1:
        dt = recent[-1]['timestamp'] - recent[0]['timestamp']
        speed = (recent[-1]['target_tokens'] - recent[0]['target_tokens']) / max(dt, 1e-9)
    left = TOTAL_UPDATES - last['update']
    eta = left * 4096 / speed / 60 if speed else float('nan')
    best = min(((v['validation']['hi']['loss'] + v['validation']['en']['loss']) / 2 for v in val), default=float('nan'))
    state = 'TRAINING' if running() else 'STOPPED'
    pct = 100 * last['update'] / TOTAL_UPDATES

    clear_output(wait=True)
    hv = val[-1]['validation'] if val else None
    cards = ''.join([
        card('Status', state, '#16a34a' if state == 'TRAINING' else '#dc2626'),
        card('Update', f"{last['update']:,} / {TOTAL_UPDATES:,}", '#2563eb'),
        card('Speed', f'{speed:,.0f} tok/s', '#7c3aed'),
        card('ETA', f'{eta:.0f} min', '#ea580c'),
        card('Train loss', f"{last['loss']:.3f}", '#0891b2'),
        card('Val Hindi', f"{hv['hi']['loss']:.3f}" if hv else '-', '#db2777'),
        card('Val English', f"{hv['en']['loss']:.3f}" if hv else '-', '#65a30d'),
        card('Best val mean', f'{best:.3f}', '#475569'),
    ])
    bar = (f'<div style="margin:10px 0;background:#e5e7eb;border-radius:8px;height:22px">'
           f'<div style="width:{pct:.1f}%;background:linear-gradient(90deg,#2563eb,#7c3aed);height:22px;'
           f'border-radius:8px;color:white;font-size:12px;text-align:center;line-height:22px">{pct:.1f}%</div></div>')
    display(HTML(f'<h3 style="margin:4px 0">Tantra 64K training · {time.strftime("%H:%M:%S")}</h3>'
                 f'<div style="display:flex;flex-wrap:wrap;gap:8px">{cards}</div>{bar}'))

    fig, ax = plt.subplots(1, 3, figsize=(18, 4.2))
    ax[0].plot(u, loss, color='#94a3b8', lw=0.6, label='per update')
    k = 25
    if len(loss) > k:
        smooth = [sum(loss[i - k:i]) / k for i in range(k, len(loss) + 1)]
        ax[0].plot(u[k - 1:], smooth, color='#2563eb', lw=2, label=f'{k}-update average')
    ax[0].axvline(START_UPDATE, color='#ea580c', ls='--', lw=1, label='moved to GPU')
    ax[0].set_title('Training loss'); ax[0].legend(); ax[0].grid(alpha=.3)
    if val:
        vu = [v['update'] for v in val]
        ax[1].plot(vu, [v['validation']['hi']['loss'] for v in val], 'o-', color='#db2777', label='Hindi')
        ax[1].plot(vu, [v['validation']['en']['loss'] for v in val], 'o-', color='#65a30d', label='English')
        ax[1].set_title('Validation loss (lower is better)'); ax[1].legend(); ax[1].grid(alpha=.3)
        ax[2].plot(vu, [v['validation']['hi']['top1_percent'] for v in val], 'o-', color='#db2777', label='Hindi top-1 %')
        ax[2].plot(vu, [v['validation']['en']['top1_percent'] for v in val], 'o-', color='#65a30d', label='English top-1 %')
        ax[2].set_title('Next-token accuracy'); ax[2].legend(); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.show()

    probes = load(OUT + '/probe_history.jsonl')
    answered = [p for p in probes if any('answer' in r for r in p.get('responses', []))]
    if answered:
        latest = answered[-1]; top = latest.get('update', 0)
        rows_html = ''.join(
            f"<tr><td>{r.get('language','')}</td><td>{r.get('question','')}</td>"
            f"<td>{(r.get('answer') or '')[:160]}</td><td>{r.get('answer_loss',0):.2f}</td></tr>"
            for r in latest['responses'])
        display(HTML(f'<h4>Model answers at update {top} (check by eye)</h4>'
                     f'<table style="font-size:12px"><tr><th>Lang</th><th>Question</th><th>Answer</th><th>Loss</th></tr>{rows_html}</table>'))
    tail = open('/kaggle/working/train.log').read().splitlines()[-4:] if os.path.exists('/kaggle/working/train.log') else []
    print('\n'.join(tail))

while True:
    draw()
    if not running():
        print('Training process is not running. Final files are in', OUT); break
    time.sleep(30)
