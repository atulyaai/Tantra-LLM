"""
tools/test_omnimodal_pipeline.py — Activates and validates Tantra's
Omnimodal Engine (Text + Vision + Audio) flowing into a single unified transformer backbone.
"""
import os
import sys
sys.path.insert(0, ".")
import torch
from Tantra.config import VocabConfig
from Tantra.tokenizer import AudioTokenizer, ImageTokenizer, ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer
from Tantra.model import NeuroCoreModel

def run_omnimodal_activation():
    print("=" * 65)
    print("      TANTRA OMNIMODAL ENGINE ACTIVATION (TEXT + VISION + AUDIO)")
    print("=" * 65)

    vcfg = VocabConfig()
    audio_tok = AudioTokenizer(vcfg)
    img_tok = ImageTokenizer(vcfg)
    bpe = ByteBPETokenizer.load("Model/tokenizer.json", vcfg)
    patcher = MegabytePatcher()
    tok = UnifiedTokenizer(vcfg, bpe, patcher)

    print("[1/5] Unified Vocabulary Ranges:")
    print("      - Text Tokens:   [0, 27999]")
    print("      - Vision Tokens: [28000, 30999]")
    print("      - Audio Tokens:  [31000, 31999]")
    print("      - Video Tokens:  [32000, 32767]")

    # 2. Simulate 1-second 16kHz audio input (e.g. spoken query)
    raw_audio = torch.randn(1, 1, 16000)
    audio_tokens = audio_tok.encode(raw_audio)
    print(f"[2/5] Voice Input (1s Audio) -> Encoded into {audio_tokens.shape[1]} Audio Tokens")

    # 3. Simulate 64x64 RGB camera image input
    raw_image = torch.randn(1, 3, 64, 64)
    image_tokens = img_tok.encode(raw_image)
    print(f"[3/5] Visual Input (RGB Camera) -> Encoded into {image_tokens.shape[1]} Visual Tokens")

    # 4. Text instruction prompt
    text_prompt = "User: Analyze what you see in the image and hear in the voice note.\nAssistant:"
    text_ids = torch.tensor([tok.encode(text_prompt)], dtype=torch.long)
    print(f"[4/5] Text Instruction -> Tokenized into {text_ids.shape[1]} Text Tokens")

    # Assemble joint Omnimodal sequence
    audio_ids = (audio_tokens + vcfg.audio_range_start).clamp(0, vcfg.vocab_size - 1)
    image_ids = (image_tokens + vcfg.image_range_start).clamp(0, vcfg.vocab_size - 1)
    omnimodal_input = torch.cat([audio_ids, image_ids, text_ids], dim=1)

    print(f"[5/5] Joint Omnimodal Sequence: {omnimodal_input.shape[1]} tokens assembled")

    # 5. Forward through NeuroCore Model
    ckpt = torch.load("Model/Latest/checkpoint_latest.pt", map_location="cpu", weights_only=False)
    model = NeuroCoreModel(ckpt["config"], use_mtp=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    with torch.no_grad():
        logits, _ = model.forward(omnimodal_input)

    print(f"[OK] NeuroCore Forward Pass Complete! Output Logits: {list(logits.shape)}")
    print("[OK] Omnimodal (Voice + Vision + Text) Engine Active on Local CPU!")
    print("=" * 65)

if __name__ == "__main__":
    run_omnimodal_activation()
