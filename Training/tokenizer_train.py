import glob
import json
from typing import List


def iter_text_paths(pattern: str) -> List[str]:
    return glob.glob(pattern)


def iter_texts(paths: List[str]) -> List[str]:
    texts: List[str] = []
    for p in paths:
        try:
            if p.endswith('.jsonl'):
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            # common fields: text, content, prompt+response
                            if 'text' in obj:
                                texts.append(str(obj['text']))
                            elif 'content' in obj:
                                texts.append(str(obj['content']))
                            else:
                                # join all string values
                                vals = [str(v) for v in obj.values() if isinstance(v, str)]
                                if vals:
                                    texts.append('\n'.join(vals))
                        except Exception:
                            continue
            else:
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read())
        except Exception:
            continue
    return texts


def train_tokenizer(texts: List[str], out_path: str, vocab_size: int = 32000):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[CLS]", "[MASK]"])

    class Iterator:
        def __init__(self, data):
            self.data = data
        def __iter__(self):
            for t in self.data:
                yield t

    tokenizer.train_from_iterator(Iterator(texts), trainer)
    tokenizer.save(out_path)
    # also write vocab for transparency
    vocab_items = tokenizer.get_vocab()
    vocab_sorted = sorted(vocab_items.items(), key=lambda kv: kv[1])
    with open(out_path.replace('.json', '_vocab.json'), 'w', encoding='utf-8') as vf:
        json.dump([tok for tok, _ in vocab_sorted], vf)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_glob', required=True, help='e.g. Dataset/*.jsonl or Dataset/*.txt')
    ap.add_argument('--out', required=True, help='Model/tokenizer.json')
    ap.add_argument('--vocab_size', type=int, default=32000)
    args = ap.parse_args()

    paths = iter_text_paths(args.input_glob)
    texts = iter_texts(paths)
    if not texts:
        raise SystemExit('No training texts found. Add data under Dataset/.')
    train_tokenizer(texts, args.out, args.vocab_size)
    print(f'Tokenizer written to {args.out}')


