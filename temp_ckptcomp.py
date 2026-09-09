import sys, torch
sys.stdout.reconfigure(encoding='utf-8')

paths = [
    r'D:\Atulya Tantra\Tantra-LLM\Model\Best\checkpoint_best.pt',
    r'D:\Atulya Tantra\Archive model\checkpoint_step_91000.pt',
]

for path in paths:
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        cfg = ckpt.get('config', None)
        sdict = ckpt.get('model_state_dict', {})
        router_keys = [k for k in sdict.keys() if '.router.' in k]
        total = sum(v.numel() for v in sdict.values())
        print(f'{path}:')
        if cfg:
            print(f'  num_experts={cfg.moe.num_experts}')
            print(f'  real_top1={cfg.moe.real_top1}')
            print(f'  layers={cfg.block.num_layers}')
            print(f'  dim={cfg.block.alra.dim}')
            print(f'  sgp_expansion={cfg.block.sgp.expansion}')
        print(f'  total_params={total:,}')
        print(f'  router_keys={len(router_keys)}')
        print()
    except Exception as e:
        print(f'{path}: ERROR {e}\n')
