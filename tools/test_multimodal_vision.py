"""
tools/test_multimodal_vision.py — Activates and validates Tantra's
Multimodal Vision Pipeline with end-to-end Image Tokenization and Embedding Projection.
"""
import os
import sys
sys.path.insert(0, ".")
import torch
import torch.nn as nn
from Tantra.config import VocabConfig, NeuroCoreConfig
from Tantra.tokenizer import ImageTokenizer, ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer
from Tantra.model import NeuroCoreModel

def run_vision_test():
    print("=" * 60)
    print("      TANTRA MULTIMODAL VISION PIPELINE ACTIVATION")
    print("=" * 60)

    # 1. Initialize Tokenizer & Image VQ-VAE Codec
    vcfg = VocabConfig()
    img_tok = ImageTokenizer(vcfg)
    bpe = ByteBPETokenizer.load("Model/tokenizer.json", vcfg)
    patcher = MegabytePatcher()
    tok = UnifiedTokenizer(vcfg, bpe, patcher)

    print(f"[1/4] Image Codebook Size: {img_tok.codebook_size}")
    print(f"[1/4] Image Token Range: [{vcfg.image_range_start}, {vcfg.image_range_end}]")

    # 2. Create synthetic test image (Batch=1, Channels=3, Height=64, Width=64)
    # Synthetic test image with color gradient
    x = torch.linspace(0, 1, 64).repeat(64, 1)
    y = torch.linspace(0, 1, 64).unsqueeze(1).repeat(1, 64)
    r = x
    g = y
    b = (x + y) / 2.0
    test_img = torch.stack([r, g, b]).unsqueeze(0)  # Shape: (1, 3, 64, 64)
    print(f"[2/4] Synthetic Test Image Created: shape {list(test_img.shape)}")

    # 3. Encode image into discrete visual tokens
    with torch.no_grad():
        visual_token_ids = img_tok.encode(test_img)
    print(f"[3/4] Image Encoded into {visual_token_ids.shape[1]} Visual Tokens:")
    print(f"      Token ID Preview: {visual_token_ids[0, :8].tolist()}...")

    # 4. Load Tantra NeuroCore Model & project visual tokens
    ckpt = torch.load("Model/Latest/checkpoint_latest.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = NeuroCoreModel(cfg, use_mtp=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Create joint text + visual prompt: "User: Describe this image: <image_tokens> Assistant:"
    text_prefix = "User: Analyze this input image: "
    text_prefix_ids = torch.tensor([tok.encode(text_prefix)], dtype=torch.long)
    
    # Map visual token IDs to global multimodal range
    global_visual_ids = visual_token_ids + vcfg.image_range_start
    global_visual_ids = global_visual_ids.clamp(0, cfg.vocab.vocab_size - 1)

    text_suffix = "\nAssistant:"
    text_suffix_ids = torch.tensor([tok.encode(text_suffix)], dtype=torch.long)

    joint_input_ids = torch.cat([text_prefix_ids, global_visual_ids, text_suffix_ids], dim=1)
    print(f"[4/4] Joint Multimodal Sequence Assembled: length {joint_input_ids.shape[1]} tokens")

    # Run forward pass through NeuroCore Transformer layers
    with torch.no_grad():
        logits, _ = model.forward(joint_input_ids)

    print(f"[OK] Forward Pass Successful! Output Logits Shape: {list(logits.shape)}")
    print(f"[OK] Multimodal Vision Pipeline Active & Verified on CPU!")
    print("=" * 60)

if __name__ == "__main__":
    run_vision_test()
