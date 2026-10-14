"""
Warm-start vocabulary expansion for Tantra-LLM
=============================================
Expands the embedding matrix of an existing base checkpoint to match
the new bilingual tokenizer vocabulary (10,277) while preserving all
pretrained transformer attention, MLP, and LayerNorm weights.
"""

import shutil
from pathlib import Path
import torch

from Tantra.tokenizer import TantraTokenizer


def expand_checkpoint_vocab(
    model_dir: str = "Model_small",
    backup: bool = True,
):
    model_path = Path(model_dir)
    base_file = model_path / "base.pt"
    tok_file = model_path / "tokenizer.json"

    assert base_file.exists(), f"Base checkpoint not found at {base_file}"
    assert tok_file.exists(), f"Tokenizer not found at {tok_file}"

    # Load new tokenizer to get exact target vocab size
    tok = TantraTokenizer.load(tok_file)
    target_vocab = tok.vocab_size
    print(f"Target Tokenizer Vocab Size: {target_vocab}")

    # Backup original checkpoint if requested
    backup_file = model_path / "base_backup_8k.pt"
    if backup and not backup_file.exists():
        shutil.copy2(base_file, backup_file)
        print(f"Backed up original checkpoint to {backup_file}")

    # Load checkpoint
    ckpt = torch.load(base_file, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    current_vocab = config.get("vocab_size", 8192)
    print(f"Current Checkpoint Vocab Size: {current_vocab}")

    if current_vocab == target_vocab:
        print("Checkpoint already matches tokenizer vocab size. Nothing to expand.")
        return

    old_embed = ckpt["state"]["embed.weight"]  # [old_vocab, d_model]
    d_model = old_embed.size(1)

    print(f"Expanding embed.weight: [{current_vocab}, {d_model}] -> [{target_vocab}, {d_model}]")

    # Initialize new embeddings
    new_embed = torch.empty((target_vocab, d_model), dtype=old_embed.dtype)
    new_embed[:current_vocab] = old_embed

    # Initialize new tokens using mean of existing embeddings + small variance
    mean_vec = old_embed.mean(dim=0, keepdim=True)
    std_vec = old_embed.std(dim=0, keepdim=True)
    n_new = target_vocab - current_vocab
    noise = torch.randn((n_new, d_model), dtype=old_embed.dtype) * (std_vec * 0.1)
    new_embed[current_vocab:] = mean_vec + noise

    # Update state and config
    ckpt["state"]["embed.weight"] = new_embed
    ckpt["config"]["vocab_size"] = target_vocab

    # Save expanded checkpoint
    torch.save(ckpt, base_file)
    print(f"Successfully saved expanded checkpoint to {base_file}!")


if __name__ == "__main__":
    expand_checkpoint_vocab("Model_small")
