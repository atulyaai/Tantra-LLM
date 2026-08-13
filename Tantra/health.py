"""
tantra/health.py — Proactive Health, Diagnostics & Optimization Watchdog for Tantra-LLM.
Scans storage, model sizes, layer capacities, memory overheads, and auto-corrects anomalies.
"""
import os
import torch
import torch.nn as nn
from typing import Dict, Any, List

from Tantra.utils import get_logger
from Tantra.codec import MultimodalWeightFormatter, CompressionConfig

log = get_logger("tantra.health")


class HealthWatchdog:
    """Proactively monitors repository health, model storage, and capacity overheads."""

    def __init__(self, model_dir: str = "model"):
        self.model_dir = model_dir

    def audit_storage_and_compress_duplicates(self, threshold_mb: float = 500.0) -> List[str]:
        """Scan model directory for large uncompressed .pt checkpoints and compress them into single .dna files."""
        fixed_files = []
        if not os.path.exists(self.model_dir):
            return fixed_files

        formatter = MultimodalWeightFormatter(CompressionConfig())

        for fname in os.listdir(self.model_dir):
            if fname.endswith(".pt") and not fname.startswith("tokenizer"):
                fpath = os.path.join(self.model_dir, fname)
                size_mb = os.path.getsize(fpath) / (1024 * 1024)

                if size_mb > threshold_mb:
                    log.warning(f"[WATCHDOG WARN] Uncompressed duplicate checkpoint detected: {fname} ({size_mb:.1f} MB > {threshold_mb} MB)")
                    dna_path = fpath.replace(".pt", ".dna")
                    
                    try:
                        ckpt = torch.load(fpath, map_location="cpu")
                        state_dict = ckpt.get("model_state_dict", ckpt)
                        
                        # Filter 2D/1D parameter tensors for compression
                        tensor_weights = {}
                        for k, v in state_dict.items():
                            if isinstance(v, torch.Tensor):
                                if v.dim() == 1:
                                    tensor_weights[k] = v.unsqueeze(0).float()
                                elif v.dim() == 2:
                                    tensor_weights[k] = v.float()
                                else:
                                    tensor_weights[k] = v.view(v.size(0), -1).float()

                        if tensor_weights:
                            formatter.format_weights(tensor_weights, dna_path)
                            dna_size_mb = os.path.getsize(dna_path) / (1024 * 1024)
                            log.info(f"[WATCHDOG AUTO-FIX] Successfully compressed {fname} -> {os.path.basename(dna_path)} ({dna_size_mb:.1f} MB, {size_mb/max(dna_size_mb, 1e-3):.1f}x smaller)")
                            
                            # Remove huge uncompressed duplicate to save disk space
                            os.remove(fpath)
                            fixed_files.append(dna_path)
                    except Exception as e:
                        log.error(f"[WATCHDOG FAIL] Could not compress {fname}: {e}")

        return fixed_files

    def check_dynamic_capacity(self, model: nn.Module) -> Dict[str, Any]:
        """Audit model parameters and layer depth for dynamic growth recommendations."""
        param_count = sum(p.numel() for p in model.parameters())
        # num_layers lives under config.block, not directly on config.
        block_cfg = getattr(model.config, "block", None)
        num_layers = getattr(block_cfg, "num_layers", None)
        if num_layers is None:
            num_layers = getattr(model.config, "num_layers", 12)
        
        status = {
            "parameter_count": param_count,
            "num_layers": num_layers,
            "is_micro_base": num_layers <= 4,
            "recommendation": "Dynamic growth enabled. Layers will expand as training perplexity plateaus."
        }

        if param_count > 100_000_000 and num_layers > 6:
            log.info(f"[WATCHDOG INFO] Base model is running at {param_count/1e6:.1f}M params ({num_layers} layers). Dynamic evolution engine will scale experts on demand.")

        return status
