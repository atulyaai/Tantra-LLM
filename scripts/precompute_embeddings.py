"""
Tantra-LLM Offline Embedding Precomputation Cache Script

Precomputes vision and audio embeddings using Long-VITA and Whisper encoders
and saves them to disk as .pt cache files. This speeds up training 10x
by preventing repeated encoder execution during epochs.

Usage:
    python scripts/precompute_embeddings.py --image-dir path/to/images --audio-dir path/to/audio --output-dir training/cache/
"""

import sys
import os
import argparse
import logging
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from encoders.vision import VisionEncoder
from encoders.audio import AudioEncoder
from config.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("precompute")


def parse_args():
    p = argparse.ArgumentParser(description="Precompute multimodal embeddings")
    p.add_argument("--image-dir", type=str, default=None, help="Directory containing images")
    p.add_argument("--audio-dir", type=str, default=None, help="Directory containing audio files (.wav)")
    p.add_argument("--output-dir", type=str, default="training/cache/", help="Output cache directory")
    p.add_argument("--model-dim", type=int, default=4096, help="Target embedding dimension")
    return p.parse_args()


def main():
    args = parse_args()
    settings = get_settings()

    print(f"\n{'='*60}")
    print(f"  TANTRA-LLM OFFLINE EMBEDDING PRECOMPUTATION")
    print(f"  Target Dimension: {settings.model_dim}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Encoders
    vis_encoder = None
    aud_encoder = None

    # 1. Precompute Vision
    if args.image_dir and os.path.isdir(args.image_dir):
        logger.info(f"Scanning images in {args.image_dir}...")
        image_files = sorted([
            os.path.join(args.image_dir, f) 
            for f in os.listdir(args.image_dir) 
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ])
        
        if image_files:
            logger.info("Initializing VisionEncoder...")
            vis_encoder = VisionEncoder(embed_dim=args.model_dim)
            logger.info(f"Processing {len(image_files)} images...")
            
            for path in image_files:
                filename = os.path.basename(path)
                out_path = os.path.join(args.output_dir, f"vis_{filename}.pt")
                
                try:
                    # Run encoder
                    embed = vis_encoder.encode(path)
                    # Save tensor
                    torch.save(embed.cpu(), out_path)
                except Exception as e:
                    logger.error(f"Failed to encode image {filename}: {e}")
            logger.info("Vision precomputation complete.")
        else:
            logger.warning("No images found in directory.")

    # 2. Precompute Audio
    if args.audio_dir and os.path.isdir(args.audio_dir):
        logger.info(f"Scanning audio files in {args.audio_dir}...")
        audio_files = sorted([
            os.path.join(args.audio_dir, f) 
            for f in os.listdir(args.audio_dir) 
            if f.lower().endswith((".wav", ".mp3", ".flac", ".m4a"))
        ])
        
        if audio_files:
            logger.info("Initializing AudioEncoder...")
            aud_encoder = AudioEncoder(embed_dim=args.model_dim)
            logger.info(f"Processing {len(audio_files)} audio files...")
            
            for path in audio_files:
                filename = os.path.basename(path)
                out_path = os.path.join(args.output_dir, f"aud_{filename}.pt")
                
                try:
                    # Load audio waveform (placeholder or standard load)
                    # For whisper, it handles path or numpy arrays
                    embed = aud_encoder.encode(path)
                    torch.save(embed.cpu(), out_path)
                except Exception as e:
                    logger.error(f"Failed to encode audio {filename}: {e}")
            logger.info("Audio precomputation complete.")
        else:
            logger.warning("No audio files found in directory.")

    print(f"\n{'='*60}")
    print(f"  Precomputation Complete.")
    print(f"  All cached embedding tensors saved to {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
