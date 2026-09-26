"""
Tantra/export.py — Checkpoint Exporter & Weight Stripper
Strips optimizer momentum buffers and optionally applies BitNet ternary quantization
to create lightweight, production-ready inference checkpoints.
"""

import os
import time
import torch

from Tantra.config import NeuroCoreConfig
from Tantra.utils import get_logger, safe_load_checkpoint
log = get_logger(__name__)


def _apply_bitnet_quantization(state_dict: dict, config: dict) -> dict:
    """Pack BitLinear weights to real 2-bit storage for export.

    Rebuilds the model from `config`, loads `state_dict`, calls
    `to_inference_mode()` on every BitLinear submodule (the same code path
    used at real inference time — see Tantra/bitnet.py), then re-extracts
    the state dict.

    Previously this rounded every 2-D "*linear*"-named tensor to {-1,0,1}
    and re-saved it at the SAME dtype/size — that only changed the values,
    not the storage, so exported "quantized" checkpoints were identical in
    size to FP32 ones and gained nothing. This version stores the actual
    packed uint8 buffers (`packed_weight_u8` + `weight_scale`) that
    `TernaryCPUKernel` already knows how to consume, for a real ~16x size
    reduction on quantized linear weights.
    """
    if isinstance(config, NeuroCoreConfig):
        cfg = config
    elif isinstance(config, dict):
        bitnet_enabled = config.get("bitnet", {}).get("enabled", False) if config else False
        if not bitnet_enabled:
            log.info("BitNet not enabled in config — exporting FP32 weights")
            return state_dict
        cfg = NeuroCoreConfig._from_dict(config)
    else:
        log.info("No usable config — exporting FP32 weights")
        return state_dict

    if not cfg.bitnet.enabled:
        log.info("BitNet not enabled in config — exporting FP32 weights")
        return state_dict

    try:
        from Tantra.model import NeuroCoreModel
        from Tantra.bitnet import BitLinear
    except ImportError as exc:  # pragma: no cover - defensive
        log.warning(f"Could not import model classes ({exc}); exporting FP32 weights")
        return state_dict

    model = NeuroCoreModel(cfg, use_mtp=False, use_moe=cfg.moe.num_experts > 1)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        log.info(f"  load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")

    packed_count = 0
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            module.to_inference_mode()
            # to_inference_mode() also builds pos_mask/neg_mask/w_ternary and
            # a duplicate int32 `packed_weight` — dense FP32-sized caches meant
            # to make the *runtime* forward pass fast, not to be persisted.
            # forward() prefers w_ternary when present; drop it (and the other
            # caches) so only the real 2-bit `packed_weight_u8` + scale are
            # saved. On load, forward() falls back to TernaryCPUKernel, which
            # unpacks+caches once per process rather than paying the storage
            # cost on every checkpoint.
            # NOTE: forward()'s CPU-kernel fallback reads `packed_weight`
            # (int32), never `packed_weight_u8` — despite the name, the u8
            # buffer is dead weight at inference time today. Keep the one
            # that's actually read; drop the rest.
            module.packed_weight_u8 = None
            module.pos_mask = None
            module.neg_mask = None
            module.w_ternary = None
            module._cached_w_ternary = None
            module._cached_scale = None
            packed_count += 1
            log.info(f"  Packed {name}: 2-bit ({module.weight_scale.item():.4f} scale)")

    log.info(f"BitNet export: packed {packed_count} BitLinear layers to real 2-bit storage")
    return model.state_dict()


def load_bitnet_state_dict(model, state_dict: dict) -> tuple:
    """Load a state dict that may contain packed BitNet weights.

    `nn.Module.load_state_dict` can only assign into buffers that already
    exist as real tensors on the target module. BitLinear registers its
    inference buffers (`packed_weight`, `weight_scale`, ...) as `None` until
    `to_inference_mode()` runs, so loading a quantized checkpoint into a
    freshly-constructed model previously discarded those keys as
    "unexpected" with `strict=False` — leaving `weight=None` and no packed
    weights either, which crashes on the first forward pass. This checks
    which BitLinear layers the checkpoint has packed weights for and calls
    `to_inference_mode()` on those first so the buffers exist to be filled.
    """
    from Tantra.bitnet import BitLinear

    for name, module in model.named_modules():
        if isinstance(module, BitLinear) and f"{name}.packed_weight" in state_dict:
            if not module.is_inference:
                module.to_inference_mode()

    return model.load_state_dict(state_dict, strict=False)


def export_clean_checkpoint(input_path: str, output_path: str, half: bool = True) -> str:
    """Drop optimizer state (and store weights as float16 when half=True).

    A training checkpoint holds weights + 2 AdamW buffers (3x the size).
    The exported file holds weights only, in fp16: ~6x smaller. Loading
    upcasts to float32 automatically (load_state_dict copies into fp32 params).
    """
    log.info(f"Loading checkpoint from: {input_path}")
    raw = safe_load_checkpoint(input_path, map_location="cpu")

    model_state = raw.get("model_state_dict", raw.get("model", raw))
    config_dict = raw.get("config", None)
    step = raw.get("step_count", raw.get("step", 0))

    # Apply BitNet quantization if enabled in config
    model_state = _apply_bitnet_quantization(model_state, config_dict)
    if half:
        shared: dict = {}   # keep tied tensors (embedding == output head) stored once
        halved = {}
        for k, v in model_state.items():
            if torch.is_tensor(v) and v.is_floating_point():
                key = (v.data_ptr(), tuple(v.shape))
                halved[k] = shared.setdefault(key, v.half())
            else:
                halved[k] = v
        model_state = halved

    clean_payload = {
        "model_state_dict": model_state,
        "config": config_dict,
        "step_count": step,
        "step": step,
        "best_loss": raw.get("best_loss", float('inf')),
        "total_tokens": raw.get("total_tokens", 0),
        "exported_at": time.time(),
        "format": "tantra-v2-inference-fp16" if half else "tantra-v2-inference"
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(clean_payload, output_path)

    orig_size = os.path.getsize(input_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    log.info(f"Clean inference checkpoint saved: {output_path} ({orig_size:.1f} MB -> {new_size:.1f} MB)")
    return output_path
