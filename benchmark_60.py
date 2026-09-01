import os
import sys
import json
import torch
import ast
import re

from Tantra.config import VocabConfig, NeuroCoreConfig
from Tantra.tokenizer import ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer
from Tantra.model import NeuroCoreModel

def run_60_benchmark(checkpoint_path: str, eval_jsonl: str = 'Datasets/eval_60_benchmark.jsonl'):
    print('=' * 70)
    print(f'📊 Running 60-Prompt Benchmark on: {checkpoint_path}')
    print('=' * 70)
    
    vcfg = VocabConfig()
    bpe = ByteBPETokenizer.load('Model/tokenizer.json', vcfg)
    patcher = MegabytePatcher()
    tokenizer = UnifiedTokenizer(vcfg, bpe, patcher)
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', None) or NeuroCoreConfig()
    sdict = ckpt.get('model_state_dict', {})
    
    # Legacy MoE compatibility detection exactly matching main.py
    has_legacy_router = any('.router.' in key for key in sdict)
    use_real_top1 = bool(
        getattr(cfg.moe, 'real_top1', False)
        and getattr(cfg.moe, 'num_experts', 1) > 1
    )
    legacy_checkpoint_compat = bool(
        has_legacy_router
        and not use_real_top1
        and getattr(cfg.moe, 'num_experts', 1) > 1
    )
    
    # Layer count detection
    layer_indices = [int(m.group(1)) for k in sdict.keys() for m in [re.search(r'layers\.(\d+)\.', k)] if m]
    if layer_indices and hasattr(cfg, 'block'):
        cfg.block.num_layers = max(layer_indices) + 1
        
    model = NeuroCoreModel(
        cfg,
        use_mtp=getattr(cfg, 'use_mtp', True),
        use_moe=(use_real_top1 and getattr(cfg.moe, 'num_experts', 1) > 1) or legacy_checkpoint_compat,
        compatibility_legacy_moe=legacy_checkpoint_compat
    )
    
    # Load state dict and verify zero missing/unexpected base tensors
    load_res = model.load_state_dict(sdict, strict=False)
    missing_base = [k for k in load_res.missing_keys if not k.startswith('category_')]
    unexpected_base = [k for k in load_res.unexpected_keys if not k.startswith('category_')]
    
    if missing_base or unexpected_base:
        raise RuntimeError(
            f'Checkpoint architecture mismatch! Missing base keys: {len(missing_base)} {missing_base[:5]}, '
            f'Unexpected keys: {len(unexpected_base)} {unexpected_base[:5]}'
        )
    print(f'✅ Architecture verified: {sum(p.numel() for p in model.parameters()):,} parameters, 0 missing base tensors.')
    model.eval()
    
    with open(eval_jsonl, 'r', encoding='utf-8') as f:
        prompts = [json.loads(l) for l in f]
        
    domain_scores = {}
    valid_ast_count = 0
    code_total = 0
    
    print(f'\nEvaluating {len(prompts)} prompts...')
    for i, item in enumerate(prompts):
        domain = item.get('domain', 'general')
        q = item['prompt']
        expected = item['expected']
        
        prompt_text = f'<|user|>\n{q}\n\n<|assistant|>\n'
        input_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long)
        
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=64, min_new_tokens=1, temperature=0.2, top_p=0.9, repetition_penalty=1.15)
        
        gen_tokens = out[0, input_ids.shape[1]:].tolist()
        gen_text = tokenizer.decode(gen_tokens).strip()
        if '</s>' in gen_text:
            gen_text = gen_text.split('</s>')[0].strip()
            
        # Concept overlap scoring
        exp_words = set(expected.lower().split())
        gen_words = set(gen_text.lower().split())
        overlap = len(exp_words & gen_words) / max(len(exp_words), 1)
        
        if domain not in domain_scores:
            domain_scores[domain] = []
        domain_scores[domain].append(overlap)
        
        ast_valid = False
        if domain == 'code':
            code_total += 1
            code_clean = gen_text.replace('`python', '').replace('`', '').strip()
            try:
                ast.parse(code_clean)
                ast_valid = True
                valid_ast_count += 1
            except Exception:
                pass
                
        if i < 8 or (i % 15 == 0):
            print(f'\n[{i+1}/{len(prompts)}] [{domain.upper()}] Q: {q}')
            print(f'  Expected: {expected[:75]}...')
            print(f'  Got     : {gen_text[:75]}...')
            print(f'  Overlap : {overlap*100:.1f}%' + (' | Valid AST: ✅' if ast_valid else ''))

    print('\n' + '=' * 70)
    print('🏆 BENCHMARK RESULTS SUMMARY')
    print('=' * 70)
    all_scores = []
    for d, scs in domain_scores.items():
        avg_d = sum(scs) / len(scs) * 100
        all_scores.extend(scs)
        print(f'  • {d.capitalize():10s}: {avg_d:.1f}% concept overlap ({len(scs)} items)')
        
    overall_avg = sum(all_scores) / len(all_scores) * 100
    print(f'  • Overall Alignment: {overall_avg:.1f}%')
    if code_total > 0:
        print(f'  • Valid Python AST : {valid_ast_count}/{code_total} ({valid_ast_count/code_total*100:.1f}%)')
    print('=' * 70)
    return {'overall_alignment': overall_avg, 'domain_scores': domain_scores}

if __name__ == '__main__':
    ckpt_target = sys.argv[1] if len(sys.argv) > 1 else 'Model/Repair/checkpoint_91000_before_repair.pt'
    run_60_benchmark(ckpt_target)
