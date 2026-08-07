"""
tantra/converter.py — Weight conversion module for importing external PyTorch/HuggingFace checkpoints into Tantra NeuroCore format.
"""
import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from tantra.utils import get_logger
from tantra.model import NeuroCoreModel
from tantra.config import NeuroCoreConfig

log = get_logger("tantra.converter")


class ModelConverter:
    """Converts PyTorch state_dict from external models into NeuroCore weights."""

    def __init__(self, target_config: Optional[NeuroCoreConfig] = None):
        self.target_config = target_config or NeuroCoreConfig()

    def convert_state_dict(self, source_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Map external state_dict keys (e.g. GPT-2 / LLaMA) into NeuroCore model parameter structure.
        """
        log.info("Mapping external weight tensors into NeuroCore format...")
        new_dict: Dict[str, torch.Tensor] = {}

        # Mappings for common layer name conventions
        for key, tensor in source_dict.items():
            # Embeddings
            if any(k in key for k in ["wte", "embed_tokens", "embeddings.word_embeddings", "embedding.weight"]):
                new_dict["embedding.weight"] = tensor
            elif any(k in key for k in ["wpe", "position_embeddings", "genome.seeds"]):
                continue  # NeuroCore uses RoPE / ALRA relative position
            # Layer Norm / Dynamic Scale Norm
            elif any(k in key for k in ["ln_f", "norm.weight", "final_layernorm"]):
                new_dict["final_norm.weight"] = tensor
            # LM Head / Projection
            elif any(k in key for k in ["lm_head", "output.weight"]):
                new_dict["lm_head.weight"] = tensor
            # Genome / Cortex legacy keys
            elif "genome.encoder" in key:
                clean_key = key.replace("genome.encoder.", "blocks.0.ffn.")
                new_dict[clean_key] = tensor
            elif "genome.decoders" in key:
                clean_key = key.replace("genome.decoders.", "blocks.1.ffn.")
                new_dict[clean_key] = tensor
            else:
                # Map block parameters
                # E.g. h.0.attn -> blocks.0.attn
                clean_key = key.replace("h.", "blocks.").replace("model.layers.", "blocks.")
                clean_key = clean_key.replace("mlp.c_fc", "ffn.proj_in").replace("mlp.c_proj", "ffn.proj_out")
                clean_key = clean_key.replace("attn.c_attn", "attn.in_proj").replace("attn.c_proj", "attn.out_proj")
                new_dict[clean_key] = tensor

        log.info(f"Successfully mapped {len(new_dict)} tensor keys into NeuroCore schema.")
        return new_dict

    def convert_checkpoint_file(self, source_path: str, output_path: str) -> str:
        """Load external PyTorch checkpoint file and save converted NeuroCore checkpoint."""
        log.info(f"Loading external checkpoint: {source_path}")
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source checkpoint not found at: {source_path}")

        loaded = torch.load(source_path, map_location="cpu")
        state_dict = loaded.get("model_state_dict", loaded.get("state_dict", loaded))

        converted_dict = self.convert_state_dict(state_dict)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save({"model_state_dict": converted_dict, "config": self.target_config}, output_path)
        log.info(f"Saved converted NeuroCore checkpoint -> {output_path}")
        return output_path
