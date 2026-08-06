import sys, os, json, math, argparse
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from npdna import NpDnaCore
import torch.nn.functional as F

def compute_ppl(core, text: str, seq_len: int = 128) -> float:
    # tokenize
    ids = core.tokenizer.encode(text, allow_growth=False)
    if len(ids) < 2:
        return float('inf')
    # pad/truncate to seq_len+1
    if len(ids) < seq_len + 1:
        ids = ids + [0] * (seq_len + 1 - len(ids))
    else:
        ids = ids[:seq_len + 1]
    x = torch.tensor([ids[:-1]], dtype=torch.long)
    y = torch.tensor([ids[1:]], dtype=torch.long)
    with torch.no_grad():
        logits, _ = core.model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
        )
    return math.exp(min(float(loss), 10.0))

def main():
    p = argparse.ArgumentParser(description='Score and filter dataset by perplexity')
    p.add_argument('--checkpoint', type=str, default='model/latest')
    p.add_argument('--input', type=str, required=True)
    p.add_argument('--output', type=str, required=True)
    p.add_argument('--min-ppl', type=float, default=1.5, help='Reject samples with PPL below this (memorized)')
    p.add_argument('--max-ppl', type=float, default=200.0, help='Reject samples with PPL above this (garbage)')
    p.add_argument('--seq-len', type=int, default=128)
    args = p.parse_args()
    
    core = NpDnaCore.load(args.checkpoint)
    core.model.eval()
    
    total = accepted = rejected_low = rejected_high = 0
    ppls = []
    
    with open(args.input, 'r', encoding='utf-8') as fin, \
         open(args.output, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
                text = record.get('text', line)
            except Exception:
                text = line
                record = {'text': line}
            
            ppl = compute_ppl(core, text, seq_len=args.seq_len)
            ppls.append(ppl)
            
            if ppl < args.min_ppl:
                rejected_low += 1
            elif ppl > args.max_ppl:
                rejected_high += 1
            else:
                record['ppl'] = round(ppl, 2)
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                accepted += 1
    
    avg_ppl = sum(ppls) / len(ppls) if ppls else 0
    print(f'Dataset Scoring Complete')
    print(f'  Total:    {total}')
    print(f'  Accepted: {accepted}')
    print(f'  Low PPL (memorized, < {args.min_ppl}): {rejected_low}')
    print(f'  High PPL (garbage, > {args.max_ppl}): {rejected_high}')
    print(f'  Avg PPL:  {avg_ppl:.2f}')

if __name__ == '__main__':
    main()
