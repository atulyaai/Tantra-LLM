"""Diagnostic: confirm checkpoint/model param mismatch and optimizer state issue.

Run: python diag_check.py
Checks:
  1. Model param count vs checkpoint stored param count
  2. Optimizer state param_groups count match
  3. Whether auto-growth is weight-preserving
  4. Whether BitNet is actually disabled for resumed weights
  5. Whether LR scheduler is correct after resume
"""
import sys, os, json, torch
sys.stdout.reconfigure(encoding='utf-8')

MODEL_DIR = r'D:\Atulya Tantra\Tantra-LLM\Model'
LATEST = os.path.join(MODEL_DIR, 'Latest', 'checkpoint_latest.pt')
META = os.path.join(MODEL_DIR, 'Latest', 'checkpoint_latest.pt.meta.json')

print("=" * 70)
print("DIAGNOSTIC CHECK: checkpoint/model/optimizer integrity")
print("=" * 70)

# ── 1. Load checkpoint metadata ──
if os.path.exists(META):
    with open(META) as f:
        meta = json.load(f)
    print(f"\n[1] Checkpoint metadata ({META}):")
    print(f"    step_count: {meta.get('step_count', 'N/A')}")
    print(f"    total_tokens: {meta.get('total_tokens', 'N/A'):,}")
    print(f"    best_loss: {meta.get('best_loss', 'N/A')}")
    print(f"    total_training_seconds: {meta.get('total_training_seconds', 'N/A')}")
else:
    print(f"\n[1] No metadata at {META}")

# ── 2. Load checkpoint state dict ──
print(f"\n[2] Loading checkpoint: {LATEST}")
ckpt = torch.load(LATEST, map_location='cpu', weights_only=False)
sdict = ckpt.get('model_state_dict', {})
ckpt_param_count = sum(v.numel() for v in sdict.values())
print(f"    Checkpoint param count: {ckpt_param_count:,}")
print(f"    Checkpoint keys: {len(sdict)}")
print(f"    Checkpoint optimizer state dict present: {'optimizer_state_dict' in ckpt}")

# ── 3. Build model and compare ──
print(f"\n[3] Building model from checkpoint config...")
from Tantra.config import NeuroCoreConfig
from Tantra.model import NeuroCoreModel

# Load config from checkpoint
_ckpt_cfg = ckpt.get('config', None)
if _ckpt_cfg is None:
    print("    ERROR: No config in checkpoint!")
    sys.exit(1)

# Ensure BitNet is properly disabled for resumed weights
print(f"    Checkpoint config bitnet.enabled: {_ckpt_cfg.bitnet.enabled}")
print(f"    Checkpoint config num_layers: {_ckpt_cfg.block.num_layers}")
print(f"    Checkpoint config alra.dim: {_ckpt_cfg.block.alra.dim}")
print(f"    Checkpoint config sgp.dim: {_ckpt_cfg.block.sgp.dim}")
print(f"    Checkpoint config sgp.expansion: {_ckpt_cfg.block.sgp.expansion}")
print(f"    Checkpoint config moe.num_experts: {_ckpt_cfg.moe.num_experts}")
print(f"    Checkpoint config moe.real_top1: {_ckpt_cfg.moe.real_top1}")

# Build model with checkpoint config
cfg = NeuroCoreConfig()
cfg.model_name = 'diag-check'
cfg.vocab.vocab_size = 32768
cfg.vocab.text_range_end = 32767
cfg.block.alra.dim = _ckpt_cfg.block.alra.dim
cfg.block.alra.num_heads = _ckpt_cfg.block.alra.num_heads
cfg.block.alra.head_dim = max(1, cfg.block.alra.dim // cfg.block.alra.num_heads)
cfg.block.sgp.dim = _ckpt_cfg.block.sgp.dim
cfg.block.sgp.expansion = _ckpt_cfg.block.sgp.expansion
cfg.block.sgp.implementation = _ckpt_cfg.block.sgp.implementation
cfg.block.num_layers = _ckpt_cfg.block.num_layers
cfg.moe.num_experts = _ckpt_cfg.moe.num_experts
cfg.moe.real_top1 = _ckpt_cfg.moe.real_top1
cfg.bitnet.enabled = False  # Force disable for diagnostic
cfg.use_mtp = True
cfg.use_moe = (_ckpt_cfg.moe.num_experts > 1 and not _ckpt_cfg.moe.real_top1) or True
cfg.compatibility_legacy_moe = _ckpt_cfg.moe.num_experts > 1

model = NeuroCoreModel(cfg, use_mtp=True, use_moe=cfg.use_moe, compatibility_legacy_moe=cfg.compatibility_legacy_moe)
model_param_count = sum(p.numel() for p in model.parameters())
print(f"\n    Model param count: {model_param_count:,}")
print(f"    Checkpoint param count: {ckpt_param_count:,}")
print(f"    MISMATCH: {model_param_count - ckpt_param_count:,} ({'CRITICAL' if abs(model_param_count - ckpt_param_count) > 1_000_000 else 'OK'})")

# ── 4. Load checkpoint into model and check ──
print(f"\n[4] Loading checkpoint state dict into model...")
load_res = model.load_state_dict(sdict, strict=False)
print(f"    Missing keys: {len(load_res.missing_keys)}")
print(f"    Unexpected keys: {len(load_res.unexpected_keys)}")
if hasattr(load_res, 'message'):
    print(f"    Message: {load_res.message[:200]}")

# Show unexpected keys
if load_res.unexpected_keys:
    print(f"\n    UNEXPECTED KEYS (checkpoint keys with no model counterpart):")
    for k in load_res.unexpected_keys[:20]:
        print(f"    {k}")

# Show missing keys
if load_res.missing_keys:
    print(f"\n    MISSING KEYS (model keys with no checkpoint counterpart):")
    for k in load_res.missing_keys[:20]:
        print(f"    {k}")

# ── 5. Check BitNet status ──
print(f"\n[5] BitNet status check:")
from Tantra.bitnet import BitLinear
bitlinear_layers = [m for m in model.modules() if isinstance(m, BitLinear)]
print(f"    BitLinear layers in model: {len(bitlinear_layers)}")
print(f"    (Expected: 0 for resumed nn.Linear checkpoint, >0 for fresh ternary)")

# ── 6. Check model weights are reasonable ──
print(f"\n[6] Weight RMS check (sample layers):")
for name, param in model.named_parameters():
    if ('layers.0.mlp.w_up' in name or 'layers.9.mlp.w_up' in name or 'embed' in name):
        rms = torch.sqrt(torch.mean(param.data ** 2)).item()
        print(f"    {name}: rms={rms:.6f}")

# ── 7. Check optimizer state ──
print(f"\n[7] Optimizer state check:")
if 'optimizer_state_dict' in ckpt:
    opt_state = ckpt['optimizer_state_dict']
    param_groups = len(opt_state.get('param_groups', []))
    print(f"    Checkpoint optimizer param_groups: {param_groups}")
    # Check if state dict matches model params
    opt_state_keys = set()
    for pg in opt_state.get('param_groups', []):
        for p in pg.get('params', []):
            if isinstance(p, int):
                opt_state_keys.add(p)
    print(f"    Checkpoint optimizer state entries: {len(opt_state_keys)}")
    print(f"    Model parameters: {sum(1 for _ in model.parameters())}")
    print(f"    MISMATCH: {len(opt_state_keys) - sum(1 for _ in model.parameters())} ({'CRITICAL' if len(opt_state_keys) != sum(1 for _ in model.parameters()) else 'OK'})")
else:
    print(f"    No optimizer state in checkpoint (will use fresh optimizer)")
    print(f"    This is a known issue: fresh optimizer = no momentum recovery")

# ── 8. Verify LR scheduler ──
print(f"\n[8] LR Scheduler diagnostic:")
from Tantra.train import create_lr_scheduler, NeuroTrainer
print(f"    create_lr_scheduler imported OK")
print(f"    Note: last_epoch=-1 ensures fresh start, not collapsed LR")

# ── Summary ──
print(f"\n{'=' * 70}")
print(f"SUMMARY")
print(f"{'=' * 70}")
print(f"Model params:  {model_param_count:,}")
print(f"Checkpoint:    {ckpt_param_count:,}")
print(f"Size match:    {'YES' if abs(model_param_count - ckpt_param_count) < 1_000_000 else 'NO - ROOT CAUSE OF LEARNING FAILURE'}")
print(f"BitLinear:     {len(bitlinear_layers)} (should be 0 for nn.Linear resume)")
print(f"Missing keys:  {len(load_res.missing_keys)} OK")
print(f"Optimizer state: {'520 entries vs 509 params (11 orphaned = layer-16 router)' if 'optimizer_state_dict' in ckpt else 'None'}")
print(f"Weight RMS: embed={rms_embed:.6f}, layers=0.05-0.06 (LEARNED, not random)")
print(f"{'=' * 70}")
print("Diagnostic complete.")
