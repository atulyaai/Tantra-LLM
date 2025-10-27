import json
import math
import os
from pathlib import Path
from typing import Dict, Any

import torch
from safetensors.torch import save_file, load_file

try:
    from Training.model_mamba import build_from_config
except ImportError:
    # Allow running as: python Training/training_pretrain.py --config ...
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
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


def train_epoch(model, tokenizer, texts, seq_len: int, lr: float, device: str, distill_cfg: Dict[str, Any]):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    # teacher EMA
    teacher = None
    alpha = float(distill_cfg.get('alpha', 0.0)) if distill_cfg.get('enabled') else 0.0
    ema_decay = float(distill_cfg.get('ema_decay', 0.999)) if distill_cfg.get('enabled') else 0.0
    if alpha > 0.0:
        import copy
        teacher = copy.deepcopy(model).eval()
    losses = []
    num_tokens = 0
    correct = 0
    total = 0
    try:
        from tqdm import tqdm  # optional progress bar
        iterator = tqdm(texts, desc="train", unit="doc")
    except Exception:
        iterator = texts
    for text in iterator:
        ids = tokenizer.encode(text).ids[:seq_len]
        if len(ids) < 2:
            continue
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        logits = model(x)
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if teacher is not None and alpha > 0.0:
            with torch.no_grad():
                t_logits = teacher(x)
            student_logp = torch.log_softmax(logits, dim=-1)
            teacher_p = torch.softmax(t_logits, dim=-1)
            kl = torch.nn.functional.kl_div(student_logp, teacher_p, reduction='batchmean')
            loss = (1 - alpha) * ce + alpha * kl
        else:
            loss = ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            num_tokens += int(y.numel())
        if num_tokens and num_tokens % 10000 == 0:
            print(f"[progress] tokens={num_tokens} last_loss={losses[-1]:.4f}")
        # EMA update
        if teacher is not None and alpha > 0.0:
            with torch.no_grad():
                for p_t, p_s in zip(teacher.parameters(), model.parameters()):
                    p_t.data.mul_(ema_decay).add_(p_s.data * (1.0 - ema_decay))
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
    # Use dynamic weight management
    from weight_manager import get_weight_manager, save_model_weights
    from config_manager import get_config_manager
    
    weight_manager = get_weight_manager()
    config_manager = get_config_manager()
    
    # Get model type from config or default to mamba
    model_type = cfg.get('model_type', 'mamba')
    paths = config_manager.get_paths(model_type)
    
    weights_path = paths['weights']
    backup_path = paths['weights_backup']
    
    Path(os.path.dirname(weights_path)).mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    device = 'cpu'
    model = build_from_config(model_cfg, vocab_size).to(device)
    # optional warm-start from existing weights
    if train_cfg.get('init_from_existing', True):
        state_dict = weight_manager.load_weights(model_type)
        if state_dict:
            try:
                model.load_state_dict(state_dict, strict=False)
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

    save_every = int(train_cfg.get('save_every', 0))
    eval_every = int(train_cfg.get('eval_every', 0))

    global_step = 0
    distill_cfg = cfg.get('distill', {})

    for epoch in range(epochs):
        loss, ppl, acc, ntok = train_epoch(model, tokenizer, load_texts(data_glob), seq_len, lr, device, distill_cfg)
        global_step += ntok
        rec = {"epoch": epoch + 1, "step_tokens": global_step, "loss": loss, "ppl": ppl, "acc": acc, "tokens": ntok}
        print(f"Epoch {epoch+1}/{epochs} tokens={global_step} loss={loss:.4f} ppl={ppl:.2f} acc={acc:.3f}")
        with open(metrics_path, 'a', encoding='utf-8') as mf:
            mf.write(json.dumps(rec) + "\n")

        do_save = save_every and (global_step // max(save_every, 1) > 0) and (global_step % save_every < ntok)
        if do_save:
            # Use dynamic weight saving
            version = f"checkpoint_{global_step}"
            saved_path = save_model_weights(model.state_dict(), model_type, version=version, is_active=True)
            print(f'[checkpoint] Saved at tokens={global_step} -> {saved_path}')

        do_eval = eval_every and (global_step // max(eval_every, 1) > 0) and (global_step % eval_every < ntok)
        if do_eval:
            eval_cfg = cfg.get('eval', {})
            prompts = eval_cfg.get('hallucination_prompts', [])
            if prompts:
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
                    results.append({"prompt": p, "answer": ans, "tokens": global_step})
                with open(os.path.join(log_dir, 'hallucination_eval.json'), 'w', encoding='utf-8') as ef:
                    json.dump(results, ef, ensure_ascii=False, indent=2)

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

    # Save final weights using dynamic weight management
    final_path = save_model_weights(model.state_dict(), model_type, version="final", is_active=True)
    print(f'Final weights written to {final_path}')


