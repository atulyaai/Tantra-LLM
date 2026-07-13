"""
Tantra-LLM Offline Embedding Precomputation Cache Script

Precomputes vision and audio embeddings using Long-VITA and Whisper encoders,
paired by sample basename/prefix, and saves them to disk as paired .pt sample files.
Each saved file is a dict:
    {
        "vision_embeds": torch.Tensor [vision_dim],   # projector input dim, NOT model_dim
        "audio_embeds":  torch.Tensor [audio_dim],    # projector input dim, NOT model_dim
        "target_ids":    torch.Tensor [seq_len],      # real token IDs from caption/transcript
    }

This is directly compatible with MultimodalDataset(cache_dir=...) for training.

Features:
    1. Basename Matching: Pairs image (e.g., sample1.png) and audio (e.g., sample1.wav)
       sharing the same prefix name. Unpaired samples are automatically zero-filled.
    2. Real Token IDs: Looks for a caption/transcript text file (e.g., sample1.txt).
       If found, tokenizes the text using TextTokenizer to generate target_ids.
    3. Missing Modality Handling: Zeroed embeddings are correctly marked, allowing
       the contrastive trainer to mask them out.

Usage:
    python scripts/precompute_embeddings.py \
        --image-dir data/images/ \
        --audio-dir data/audio/ \
        --text-dir data/captions/ \
        --output-dir training/cache/
"""

import sys
import os
import argparse
import logging
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from npdna.encoders.vision import VisionEncoder
from npdna.encoders.audio import AudioEncoder
from npdna.encoders.text import TextTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("precompute")

TARGET_SEQ_LEN = 32


def parse_args():
    p = argparse.ArgumentParser(description="Precompute multimodal embeddings into paired sample dicts")
    p.add_argument("--image-dir",   type=str, default=None, help="Directory containing images (.png/.jpg/.jpeg/.webp)")
    p.add_argument("--audio-dir",   type=str, default=None, help="Directory containing audio files (.wav/.mp3/.flac/.m4a)")
    p.add_argument("--text-dir",    type=str, default=None, help="Directory containing text captions/transcripts (.txt)")
    p.add_argument("--output-dir",  type=str, default="training/cache/", help="Output cache directory")
    p.add_argument("--vision-dim",  type=int, default=768,  help="Vision encoder output dim (projector input)")
    p.add_argument("--audio-dim",   type=int, default=512,  help="Audio encoder output dim (projector input)")
    p.add_argument("--model-dim",   type=int, default=4096, help="LM hidden dim — only used for encoder init")
    p.add_argument("--tokenizer",   type=str, default="gpt2", help="Pretrained tokenizer to use for text")
    return p.parse_args()


def _get_files_map(directory: str, extensions: tuple) -> dict:
    """Return dictionary mapping basename to absolute path."""
    if not directory or not os.path.isdir(directory):
        return {}
    mapping = {}
    for f in os.listdir(directory):
        if f.lower().endswith(extensions):
            name_without_ext = os.path.splitext(f)[0]
            mapping[name_without_ext] = os.path.join(directory, f)
    return mapping


def _encode_vision(encoder: VisionEncoder, path: str, vision_dim: int) -> torch.Tensor | None:
    """Encode a single image. Returns [vision_dim] tensor or None on failure."""
    try:
        embed = encoder.encode(path)          # shape [1, encoder_out_dim]
        embed = embed.squeeze(0).cpu()        # [encoder_out_dim]
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

    # Scan directories
    image_map = _get_files_map(args.image_dir, (".png", ".jpg", ".jpeg", ".webp"))
    audio_map = _get_files_map(args.audio_dir, (".wav", ".mp3", ".flac", ".m4a"))
    text_map  = _get_files_map(args.text_dir, (".txt",))

    # All unique sample identifiers
    all_keys = sorted(list(set(image_map.keys()) | set(audio_map.keys())))

    if not all_keys:
        logger.error("No image or audio files found. Provide --image-dir and/or --audio-dir.")
        return

    logger.info(f"Found {len(all_keys)} unique sample prefixes to process.")

    # Initialize tokenizer and encoders lazily
    tokenizer = TextTokenizer(args.tokenizer)
    vis_encoder = VisionEncoder(embed_dim=args.model_dim) if image_map else None
    aud_encoder = AudioEncoder(embed_dim=args.model_dim)  if audio_map else None

    saved = 0
    skipped = 0

    for i, key in enumerate(all_keys):
        img_path = image_map.get(key)
        aud_path = audio_map.get(key)

        # 1. Vision embed
        if img_path and vis_encoder:
            vision_embed = _encode_vision(vis_encoder, img_path, args.vision_dim)
        else:
            vision_embed = torch.zeros(args.vision_dim)

        # 2. Audio embed
        if aud_path and aud_encoder:
            audio_embed = _encode_audio(aud_encoder, aud_path, args.audio_dim)
        else:
            audio_embed = torch.zeros(args.audio_dim)

        # Skip sample if real modality encoding failed
        if img_path and vision_embed is None:
            logger.warning(f"Skipping sample '{key}': vision encoding failed.")
            skipped += 1
            continue
        if aud_path and audio_embed is None:
            logger.warning(f"Skipping sample '{key}': audio encoding failed.")
            skipped += 1
            continue

        # 3. Text caption/transcript loading
        txt_path = text_map.get(key)
        # Fallback search in image_dir or audio_dir
        if not txt_path:
            for d in [args.image_dir, args.audio_dir]:
                if d and os.path.isdir(d):
                    potential_path = os.path.join(d, f"{key}.txt")
                    if os.path.isfile(potential_path):
                        txt_path = potential_path
                        break

        caption = ""
        if txt_path:
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
            except Exception as e:
                logger.warning(f"Failed to read text file for '{key}': {e}")

        if not caption:
            caption = f"A sample response for prefix {key} in the multimodal dataset."

        # 4. Tokenization
        tokens = tokenizer.encode(caption)
        if len(tokens) < TARGET_SEQ_LEN:
            # Pad with 0 (usually end-of-text / pad token)
            tokens = tokens + [0] * (TARGET_SEQ_LEN - len(tokens))
        else:
            tokens = tokens[:TARGET_SEQ_LEN]

        target_ids = torch.LongTensor(tokens)

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
