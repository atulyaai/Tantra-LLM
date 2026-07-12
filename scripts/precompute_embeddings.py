"""
Tantra-LLM Offline Embedding Precomputation Cache Script

Precomputes vision and audio embeddings using Long-VITA and Whisper encoders
and saves them to disk as paired .pt sample files. Each saved file is a dict:
    {
        "vision_embeds": torch.Tensor [vision_dim],   # projector input dim, NOT model_dim
        "audio_embeds":  torch.Tensor [audio_dim],    # projector input dim, NOT model_dim
        "target_ids":    torch.Tensor [seq_len],      # placeholder token IDs
    }

This is directly compatible with MultimodalDataset(cache_dir=...) for training.

IMPORTANT:
    --vision-dim / --audio-dim must match the encoder's actual output dimension,
    NOT the language-model hidden dimension. Defaults: vision=768, audio=512.
    The fusion projectors bridge these dims to model_dim internally.

Usage:
    # Paired image+audio (filenames sorted; must have equal counts):
    python scripts/precompute_embeddings.py \\
        --image-dir data/images/ --audio-dir data/audio/ \\
        --output-dir training/cache/

    # Vision only (audio_embeds filled with zeros):
    python scripts/precompute_embeddings.py \\
        --image-dir data/images/ --output-dir training/cache/
"""

import sys
import os
import argparse
import logging
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from encoders.vision import VisionEncoder
from encoders.audio import AudioEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("precompute")

TARGET_SEQ_LEN = 32   # placeholder target length until real captions/transcripts integrated
TARGET_VOCAB = 65536  # placeholder vocab size for random token IDs


def parse_args():
    p = argparse.ArgumentParser(description="Precompute multimodal embeddings into paired sample dicts")
    p.add_argument("--image-dir",   type=str, default=None, help="Directory containing images (.png/.jpg/.jpeg/.webp)")
    p.add_argument("--audio-dir",   type=str, default=None, help="Directory containing audio files (.wav/.mp3/.flac/.m4a)")
    p.add_argument("--output-dir",  type=str, default="training/cache/", help="Output cache directory")
    p.add_argument("--vision-dim",  type=int, default=768,  help="Vision encoder output dim (projector input)")
    p.add_argument("--audio-dim",   type=int, default=512,  help="Audio encoder output dim (projector input)")
    p.add_argument("--model-dim",   type=int, default=4096, help="LM hidden dim — only used for encoder init, NOT saved to cache")
    return p.parse_args()


def _list_files(directory: str, extensions: tuple) -> list:
    """Return sorted list of absolute paths matching given extensions in directory."""
    if not directory or not os.path.isdir(directory):
        return []
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(extensions)
    ])


def _encode_vision(encoder: VisionEncoder, path: str, vision_dim: int) -> torch.Tensor | None:
    """Encode a single image. Returns [vision_dim] tensor or None on failure."""
    try:
        embed = encoder.encode(path)          # shape [1, encoder_out_dim]
        embed = embed.squeeze(0).cpu()        # [encoder_out_dim]
        # Align to target vision_dim
        if embed.size(0) < vision_dim:
            embed = torch.cat([embed, torch.zeros(vision_dim - embed.size(0))])
        else:
            embed = embed[:vision_dim]
        if embed.abs().sum().item() == 0.0:
            logger.error(f"Vision encoder returned zeros for {os.path.basename(path)}. Skipping.")
            return None
        return embed
    except Exception as e:
        logger.error(f"Vision encoding failed for {os.path.basename(path)}: {e}")
        return None


def _encode_audio(encoder: AudioEncoder, path: str, audio_dim: int) -> torch.Tensor | None:
    """Load waveform with librosa and encode. Returns [audio_dim] tensor or None on failure."""
    try:
        import librosa
        waveform, _sr = librosa.load(path, sr=16000, mono=True)
        embed = encoder.encode(waveform)      # shape [1, encoder_out_dim]
        embed = embed.squeeze(0).cpu()        # [encoder_out_dim]
        # Align to target audio_dim
        if embed.size(0) < audio_dim:
            embed = torch.cat([embed, torch.zeros(audio_dim - embed.size(0))])
        else:
            embed = embed[:audio_dim]
        if embed.abs().sum().item() == 0.0:
            logger.error(f"Audio encoder returned zeros for {os.path.basename(path)}. Skipping.")
            return None
        return embed
    except ImportError:
        logger.error("librosa is required: pip install librosa")
        return None
    except Exception as e:
        logger.error(f"Audio encoding failed for {os.path.basename(path)}: {e}")
        return None


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  TANTRA-LLM OFFLINE EMBEDDING PRECOMPUTATION")
    print(f"  Vision dim: {args.vision_dim}  |  Audio dim: {args.audio_dim}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    image_files = _list_files(args.image_dir, (".png", ".jpg", ".jpeg", ".webp"))
    audio_files = _list_files(args.audio_dir, (".wav", ".mp3", ".flac", ".m4a"))

    n_images = len(image_files)
    n_audio  = len(audio_files)
    n_samples = max(n_images, n_audio)

    if n_samples == 0:
        logger.error("No image or audio files found. Provide --image-dir and/or --audio-dir.")
        return

    if n_images and n_audio and n_images != n_audio:
        logger.warning(
            f"Image count ({n_images}) != audio count ({n_audio}). "
            f"Will pair up to min({n_images},{n_audio}) samples; extras get zero embeddings."
        )
        n_samples = max(n_images, n_audio)

    # Initialize encoders lazily
    vis_encoder = VisionEncoder(embed_dim=args.model_dim) if image_files else None
    aud_encoder = AudioEncoder(embed_dim=args.model_dim)  if audio_files else None

    saved = 0
    skipped = 0

    for i in range(n_samples):
        img_path = image_files[i] if i < n_images else None
        aud_path = audio_files[i] if i < n_audio  else None

        # Vision embed
        if img_path and vis_encoder:
            vision_embed = _encode_vision(vis_encoder, img_path, args.vision_dim)
        else:
            vision_embed = torch.zeros(args.vision_dim)

        # Audio embed
        if aud_path and aud_encoder:
            audio_embed = _encode_audio(aud_encoder, aud_path, args.audio_dim)
        else:
            audio_embed = torch.zeros(args.audio_dim)

        # Skip sample if the one modality that was supposed to be real failed
        if img_path and vision_embed is None:
            logger.warning(f"Skipping sample {i}: vision encoding failed.")
            skipped += 1
            continue
        if aud_path and audio_embed is None:
            logger.warning(f"Skipping sample {i}: audio encoding failed.")
            skipped += 1
            continue

        # Placeholder target_ids — replace with real caption token IDs when available
        target_ids = torch.randint(0, TARGET_VOCAB, (TARGET_SEQ_LEN,))

        sample = {
            "vision_embeds": vision_embed,   # [vision_dim]
            "audio_embeds":  audio_embed,    # [audio_dim]
            "target_ids":    target_ids,     # [seq_len]
        }

        out_path = os.path.join(args.output_dir, f"sample_{i:06d}.pt")
        torch.save(sample, out_path)
        saved += 1

    print(f"\n{'='*60}")
    print(f"  Precomputation Complete.")
    print(f"  Saved: {saved} samples  |  Skipped: {skipped}")
    print(f"  Cache: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
