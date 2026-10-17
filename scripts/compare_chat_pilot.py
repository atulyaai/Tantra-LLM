"""Compare base vs chat-pilot answers on fresh prompts absent from training data.

Usage: .venv\\Scripts\\python.exe scripts\\compare_chat_pilot.py <pilot_dir> [base_dir]
Writes <pilot_dir>/responses.json. Development inspection only, not a release score.
"""
import json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Tantra.config import ModelConfig
from Tantra.model.tantra_model import TantraModel
from Tantra.tokenizer import TantraTokenizer

FRESH = ['Hey, what do people call you?', 'Who gave you life?', 'Introduce yourself in two sentences.',
         'Are you a Google product?', 'Who is the mind behind you?', 'Do I speak with a machine?',
         'Which company do you represent?', 'Who coded you?', 'Are you Atul Vij himself?',
         'What should I know about you?', 'Good evening!', 'Hello there, friend.', 'I have to leave now, bye.',
         'It is so cold today.']


def main():
    pilot = Path(sys.argv[1]); base = Path(sys.argv[2] if len(sys.argv) > 2 else 'runs/kaggle-output/base-training')
    tok = TantraTokenizer.load(pilot / 'tokenizer.json')
    results = {}
    for label, path in (('base', base / 'base.pt'), ('pilot', pilot / 'base.pt')):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        model = TantraModel(ModelConfig.from_dict(ck['config'])); model.load_state_dict(ck['state'], strict=False); model.eval()
        results[label] = {}
        for q in FRESH:
            ids, _ = tok.encode_chat([{'role': 'user', 'content': q}], add_generation_prompt=True)
            with torch.no_grad():
                out = list(model.generate(torch.tensor([ids]), max_new_tokens=40, temperature=0,
                                          stop_ids=(tok.end_id, tok.user_id)))
            results[label][q] = tok.decode(out)
    (pilot / 'responses.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    for q in FRESH:
        print(f'{q!r}\n   base : {results["base"][q]!r}\n   pilot: {results["pilot"][q]!r}')


if __name__ == '__main__':
    main()
