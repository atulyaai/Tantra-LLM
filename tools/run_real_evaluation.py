import os
import sys
import json
import time
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from main import init_model, build_vocab
from Tantra.config import NeuroCoreConfig, VocabConfig
from Tantra.train import NeuroTrainer

def main():
    vcfg = VocabConfig()
    tok = build_vocab(vcfg, 'Datasets/master_corpus.jsonl')

    ckpt_path = 'Model/Checkpoints/checkpoint_step_44000.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', None)
    if cfg is None:
        cfg = NeuroCoreConfig()
    cfg.vocab.vocab_size = 32768

    model = init_model(cfg, 'cpu')
    trainer = NeuroTrainer(model, lr=1e-4)
    trainer.load_checkpoint(ckpt_path)
    model.eval()

    test_suite = [
        # 1. Greetings & Identity
        {'category': 'Greetings', 'prompt': 'Hello! How are you today?'},
        {'category': 'Identity', 'prompt': 'What is your name and who created you?'},
        
        # 2. Conversation & Explanation
        {'category': 'Conversation', 'prompt': 'Tell me a fun fact about space and the solar system.'},
        {'category': 'Explanation', 'prompt': 'Explain how a computer works in simple words for a kid.'},
        
        # 3. Math & Logic
        {'category': 'Math Calculation', 'prompt': 'Calculate: 45 * 12 + 15'},
        {'category': 'Logic Reasoning', 'prompt': 'If Alice has 3 apples and Bob gives her 5 more, how many apples does Alice have?'},
        
        # 4. Science & Biology
        {'category': 'Science', 'prompt': 'How do vaccines help the human immune system fight viruses?'},
        
        # 5. Coding & Algorithms
        {'category': 'Coding', 'prompt': 'Write a Python function to check if a number is prime.'},
        
        # 6. Grammar & Editing
        {'category': 'Grammar Correction', 'prompt': 'Fix the grammar in this sentence: She dont know nothing about science.'},
        
        # 7. Safety / Assistant Assistance
        {'category': 'Study Assistant', 'prompt': 'Can you help me design a study schedule for my exams?'}
    ]

    results = []

    print('=' * 85)
    print('🚀 TANTRA-LLM REAL SIDE-BY-SIDE EVALUATION SUITE (Step 44,000 Milestone)')
    print(f'Model Checkpoint: {ckpt_path} | Tokens Trained: 417.08M | Loss: 4.39')
    print('=' * 85)

    for idx, item in enumerate(test_suite, 1):
        cat = item['category']
        q = item['prompt']
        
        formatted = f'<|system|>\nYou are Tantra, a helpful, precise, and polite AI assistant created by Atulya AI. Answer clearly, accurately, and concisely.\n<|user|>\n{q}\n<|assistant|>\n'
        tids = torch.tensor([tok.encode(formatted)])
        
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(tids, max_new_tokens=80, temperature=0.25, top_p=0.85, repetition_penalty=1.25)
        gen_time = time.perf_counter() - t0
        
        gen_tok_ids = out[0, tids.shape[1]:].tolist()
        if 2 in gen_tok_ids:
            gen_tok_ids = gen_tok_ids[:gen_tok_ids.index(2)]
        
        raw_response = tok.decode(gen_tok_ids).replace('</s>', '').strip()
        tok_count = len(gen_tok_ids)
        tok_speed = tok_count / max(gen_time, 1e-4)
        
        entry = {
            'id': idx,
            'category': cat,
            'prompt': q,
            'raw_response': raw_response,
            'tokens_generated': tok_count,
            'generation_speed_tok_s': round(tok_speed, 1),
            'checkpoint': os.path.basename(ckpt_path),
            'step': 44000
        }
        results.append(entry)
        
        print(f'\n[{idx}/{len(test_suite)}] 📌 Category: {cat}')
        print(f'❓ Prompt: {q}')
        print(f'🤖 Tantra ({tok_count} tokens @ {tok_speed:.1f} tok/s):')
        print(raw_response)
        print('-' * 85)

    os.makedirs('docs', exist_ok=True)
    json_path = 'docs/tantra_real_eval_examples.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    md_path = 'docs/tantra_real_eval_examples.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# 🧪 Tantra-LLM Real Live Output Evaluation (Step 44,000 Checkpoint)\n\n')
        f.write(f'**Checkpoint:** `{ckpt_path}` | **Step:** `44,000` | **Tokens Trained:** `417.08M` | **Loss:** `4.3966`\n\n')
        f.write('| # | Category | User Prompt | Real Tantra Response | Tokens | Speed |\n')
        f.write('|---|---|---|---|---|---|\n')
        for r in results:
            clean_resp = r['raw_response'].replace('\n', ' ').replace('|', '\\|')
            if len(clean_resp) > 100:
                clean_resp = clean_resp[:100] + '...'
            f.write(f"| {r['id']} | **{r['category']}** | {r['prompt']} | {clean_resp} | {r['tokens_generated']} | {r['generation_speed_tok_s']} tok/s |\n")

    print(f'\n✅ Real evaluation examples successfully saved to:')
    print(f'📁 {json_path}')
    print(f'📁 {md_path}')

if __name__ == '__main__':
    main()
