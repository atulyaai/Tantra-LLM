import json
import math
import os
from pathlib import Path
from typing import Dict, Any

import torch
from safetensors.torch import save_file, load_file

from Training.model_mamba import build_from_config


def load_texts(glob_pattern: str):
    import glob
    paths = glob.glob(glob_pattern)
    for p in paths:
        if p.endswith('.jsonl'):
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        text = obj.get('text') or obj.get('content')
                        if text:
                            yield str(text)
                    except Exception:
                        continue
        else:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                yield f.read()


def train_epoch(model, tokenizer, texts, seq_len: int, lr: float, device: str):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    num_tokens = 0
    correct = 0
    total = 0
    for text in texts:
        ids = tokenizer.encode(text).ids[:seq_len]
        if len(ids) < 2:
            continue
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            num_tokens += int(y.numel())
    avg_loss = sum(losses) / max(len(losses), 1)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    acc = (correct / max(total, 1)) if total else 0.0
    return avg_loss, ppl, acc, num_tokens


if __name__ == '__main__':
    import argparse
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    train_cfg: Dict[str, Any] = cfg.get('train', {})
    model_cfg: Dict[str, Any] = cfg.get('model', {})

    tokenizer_path = cfg['paths']['tokenizer']
    weights_path = cfg['paths']['weights']
    backup_path = cfg['paths']['weights_backup']

    Path(os.path.dirname(weights_path)).mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    device = 'cpu'
    model = build_from_config(model_cfg, vocab_size).to(device)
    # optional warm-start from existing weights
    if train_cfg.get('init_from_existing', True) and os.path.exists(weights_path):
        try:
            state = load_file(weights_path)
            model.load_state_dict(state, strict=False)
            print('Loaded existing weights for warm-start')
        except Exception as e:
            print(f'Warm-start skipped: {e}')

    seq_len = int(train_cfg.get('seq_len', 256))
    lr = float(train_cfg.get('lr', 1e-3))
    epochs = int(train_cfg.get('epochs', 1))
    data_glob = train_cfg.get('data_glob', 'Dataset/*.jsonl')

    log_dir = train_cfg.get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    metrics_path = os.path.join(log_dir, 'train_metrics.jsonl')

    for epoch in range(epochs):
        loss, ppl, acc, ntok = train_epoch(model, tokenizer, load_texts(data_glob), seq_len, lr, device)
        rec = {"epoch": epoch + 1, "loss": loss, "ppl": ppl, "acc": acc, "tokens": ntok}
        print(f"Epoch {epoch+1}/{epochs} loss={loss:.4f} ppl={ppl:.2f} acc={acc:.3f} tokens={ntok}")
        with open(metrics_path, 'a', encoding='utf-8') as mf:
            mf.write(json.dumps(rec) + "\n")

    # Simple hallucination eval
    eval_cfg = cfg.get('eval', {})
    prompts = eval_cfg.get('hallucination_prompts', [])
    if prompts:
        from tokenizers import Tokenizer
        # lightweight text gen: greedy
        def gen_once(txt: str) -> str:
            ids = tokenizer.encode(txt).ids[:seq_len]
            out = []
            with torch.no_grad():
                for _ in range(64):
                    x = torch.tensor([ids + out], dtype=torch.long, device=device)
                    logits = model(x)[:, -1, :]
                    next_id = int(logits.argmax(dim=-1))
                    out.append(next_id)
            return tokenizer.decode(out)
        results = []
        for p in prompts:
            ans = gen_once(p)
            results.append({"prompt": p, "answer": ans})
        with open(os.path.join(log_dir, 'hallucination_eval.json'), 'w', encoding='utf-8') as ef:
            json.dump(results, ef, ensure_ascii=False, indent=2)

    tmp_path = weights_path + '.tmp'
    save_file(model.state_dict(), tmp_path)
    if os.path.exists(weights_path):
        try:
            os.replace(weights_path, backup_path)
        except Exception:
            pass
    os.replace(tmp_path, weights_path)
    print(f'Weights written to {weights_path}')


