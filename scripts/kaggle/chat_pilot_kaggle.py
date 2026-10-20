"""Kaggle GPU run: English-only conversation pilot on the 64K base (full-model SFT).

Inputs: dataset tantra-chat-en-pilot-vN (code + one approved English bundle) and
dataset tantra-64k-base-v1 (base.pt, tokenizer.json, report.json).
Outputs in /kaggle/working/chat-pilot: base.pt, report.json, responses.json.
The result is an undersized, English-only pilot and is never deployed automatically.
"""
import glob, json, os, shutil, subprocess, sys

CODE = next(os.path.dirname(p) for p in glob.glob('/kaggle/input/**/requirements.txt', recursive=True))
BASES = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/base.pt', recursive=True)]
if not BASES:
    sys.exit('base.pt not found under /kaggle/input; inputs: ' + ', '.join(sorted(glob.glob('/kaggle/input/*/*'))))
BASE = BASES[0]
WORK = '/kaggle/working/tantra'
OUT = '/kaggle/working/chat-pilot'
EPOCHS = int(os.environ.get('TANTRA_EPOCHS', '3'))  # best.pt keeps the lowest-validation weights
# <bundle>/prepared/manifest.json -> <bundle>
BUNDLE = next(os.path.relpath(os.path.dirname(os.path.dirname(p)), CODE) for p in glob.glob(CODE + '/Datasets/*/prepared/manifest.json'))
if not os.path.exists(WORK):
    shutil.copytree(CODE, WORK)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tokenizers', 'psutil'], check=False)
subprocess.run(['nvidia-smi'], check=False)
env = dict(os.environ, TANTRA_DEVICE='cuda', PYTHONUNBUFFERED='1')
cmd = [sys.executable, '-m', 'Tantra.conversation_train', 'train', '--english-pilot', '--epochs', str(EPOCHS),
       '--bundle', BUNDLE, '--source', BASE, '--output', OUT]
code = subprocess.run(cmd, cwd=WORK, env=env).returncode
if code:
    sys.exit(code)

# Inspect answers to fresh prompts that appear nowhere in the training data.
sys.path.insert(0, WORK)
import torch
from Tantra.config import ModelConfig
from Tantra.model.tantra_model import TantraModel
from Tantra.tokenizer import TantraTokenizer
tok = TantraTokenizer.load(OUT + '/tokenizer.json')
results = {}
for label, path in (('base', BASE + '/base.pt'), ('pilot', OUT + '/base.pt')):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = TantraModel(ModelConfig.from_dict(ck['config'])); model.load_state_dict(ck['state'], strict=False)
    model.to('cuda').eval()
    answers = {}
    for q in ['Hey, what do people call you?', 'Who gave you life?', 'Introduce yourself in two sentences.',
              'Are you a Google product?', 'Who is the mind behind you?', 'Do I speak with a machine?',
              'Which company do you represent?', 'Who coded you?', 'Are you Atul Vij himself?',
              'What should I know about you?', 'Good evening!', 'Hello there, friend.', 'I have to leave now, bye.',
              'It is so cold today.']:
        ids, _ = tok.encode_chat([{'role': 'user', 'content': q}], add_generation_prompt=True)
        with torch.no_grad():
            out = list(model.generate(torch.tensor([ids], device='cuda'), max_new_tokens=40, temperature=0,
                                      stop_ids=(tok.end_id, tok.user_id)))
        answers[q] = tok.decode(out)
    results[label] = answers
json.dump(results, open(OUT + '/responses.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
for q in results['pilot']:
    print(f'{q!r}\n   base : {results["base"][q]!r}\n   pilot: {results["pilot"][q]!r}')
