"""
tantra/tokenizer.py — Unified multimodal tokenization.
Contains: ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer,
          AudioTokenizer, ImageTokenizer, VideoTokenizer,
          ModalityRouter, OutputRouter.
"""
from __future__ import annotations

import os
import pickle
from enum import Enum
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from Tantra.config import VocabConfig

# ── Constants ─────────────────────────────────────────────────────────────────

MODALITY_TEXT  = "text"
MODALITY_AUDIO = "audio"
MODALITY_IMAGE = "image"
MODALITY_VIDEO = "video"
MODALITY_BYTES = "bytes"

class ModalityType(Enum):
    TEXT  = "text"
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"


# ── Byte-level BPE Tokenizer ─────────────────────────────────────────────────

class ByteBPETokenizer:
    """
    Byte-level BPE tokenizer. ZERO out-of-vocabulary: any input can be tokenized.
    Uses HuggingFace `tokenizers` library when available, with raw byte fallback.
    """
    def __init__(self, config: VocabConfig):
        self._config = config
        self._tokenizer = None
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.pre_tokenizers import ByteLevel
            from tokenizers.decoders import ByteLevel as ByteLevelDecoder

            self._tokenizer = Tokenizer(BPE())
            self._tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
            self._tokenizer.decoder = ByteLevelDecoder()
        except ImportError:
            self._tokenizer = None

    def train(self, corpus_paths: list[str], vocab_size: int = 32000,
              special_tokens: Optional[list[str]] = None) -> None:
        if self._tokenizer is None:
            return
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        if special_tokens is None:
            special_tokens = list(self._config.special_tokens.keys())
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens,
                             initial_alphabet=ByteLevel.alphabet())
        self._tokenizer.train(corpus_paths, trainer)

    def encode(self, text: str) -> list[int]:
        if self._tokenizer is not None and self._tokenizer.get_vocab_size() > 0:
            try:
                return self._tokenizer.encode(text).ids
            except Exception as e:
                import logging; logging.getLogger("tantra").warning(f"Silenced exception: {e}")
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        if self._tokenizer is not None and self._tokenizer.get_vocab_size() > 0:
            try:
                return self._tokenizer.decode(ids)
            except Exception as e:
                import logging; logging.getLogger("tantra").warning(f"Silenced exception: {e}")
        valid_bytes = bytes([i % 256 for i in ids])
        return valid_bytes.decode("utf-8", errors="ignore")

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        if self._tokenizer is not None and self._tokenizer.get_vocab_size() > 0:
            try:
                return [enc.ids for enc in self._tokenizer.encode_batch(texts)]
            except Exception as e:
                import logging; logging.getLogger("tantra").warning(f"Silenced exception: {e}")
        return [self.encode(t) for t in texts]

    def save(self, path: str) -> None:
        if self._tokenizer is not None:
            self._tokenizer.save(path)
        else:
            with open(path, "w") as f:
                f.write('{"fallback": true}')

    @classmethod
    def load(cls, path: str, config: VocabConfig) -> "ByteBPETokenizer":
        inst = cls(config)
        try:
            from tokenizers import Tokenizer
            inst._tokenizer = Tokenizer.from_file(path)
        except Exception as e:
            import logging
            logging.getLogger("tantra.tokenizer").error(f"CRITICAL: Failed to load BPE tokenizer from {path}. Fallback to raw bytes is DISABLED for training safety. Error: {e}")
            raise RuntimeError(f"Failed to load tokenizer from {path}. File is likely an invalid or corrupt BPE JSON. Do not use raw metadata pickles.") from e
        return inst

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is not None and self._tokenizer.get_vocab_size() > 0:
            return self._tokenizer.get_vocab_size()
        return self._config.vocab_size


# ── MegaByte Patcher ──────────────────────────────────────────────────────────

class MegabytePatcher:
    """
    Converts raw byte sequences to fixed-size patches.
    Each patch of `patch_size` bytes → one token via nearest-codebook lookup.
    """
    def __init__(self, patch_size: int = 8, codebook_size: int = 4096):
        self.patch_size = patch_size
        self.codebook_size = codebook_size
        self._codebook = None  # numpy array of shape [codebook_size, 5]

    def _patch_features(self, patch: bytes) -> list[float]:
        vals = list(patch)
        if not vals:
            return [0.0] * 5
        import numpy as np
        a = np.array(vals, dtype=float)
        return [a.mean(), a.std(), a.min(), a.max(), float(np.median(a))]

    def encode_bytes(self, raw_bytes: bytes) -> list[int]:
        import numpy as np
        tokens = []
        for i in range(0, len(raw_bytes), self.patch_size):
            patch = raw_bytes[i : i + self.patch_size]
            feat = self._patch_features(patch)
            if self._codebook is not None:
                dists = ((self._codebook - np.array(feat)) ** 2).sum(axis=1)
                tokens.append(int(np.argmin(dists)))
            else:
                tokens.append(hash(patch) % self.codebook_size)
        return tokens

    def decode_to_bytes(self, token_ids: list[int]) -> bytes:
        if self._codebook is not None and len(self._codebook) > 0:
            out = bytearray()
            for tid in token_ids:
                center = self._codebook[tid % len(self._codebook)]
                val = int(center[0])
                out.extend([max(0, min(255, val))] * self.patch_size)
            return bytes(out)
        return bytes([0] * len(token_ids) * self.patch_size)

    def train_codebook(self, byte_sequences: list[bytes]) -> None:
        import numpy as np
        try:
            from sklearn.cluster import KMeans
            features = []
            for seq in byte_sequences:
                for i in range(0, len(seq), self.patch_size):
                    features.append(self._patch_features(seq[i : i + self.patch_size]))
            features = np.array(features)
            km = KMeans(n_clusters=min(self.codebook_size, len(features)), n_init=3)
            km.fit(features)
            self._codebook = km.cluster_centers_
        except ImportError:
            pass

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"codebook": self._codebook, "patch_size": self.patch_size,
                         "codebook_size": self.codebook_size}, f)

    @classmethod
    def load(cls, path: str) -> "MegabytePatcher":
        with open(path, "rb") as f:
            data = pickle.load(f)
        inst = cls(data["patch_size"], data["codebook_size"])
        inst._codebook = data.get("codebook")
        return inst


# ── Unified Tokenizer ─────────────────────────────────────────────────────────

class UnifiedTokenizer:
    """
    Master tokenizer — routes input to correct sub-tokenizer,
    maps all outputs to a single shared 32K integer space,
    and supports unified weight sharing across text, audio, image, and video modalities.
    """
    def __init__(self, config: VocabConfig, bpe: ByteBPETokenizer, patcher: MegabytePatcher):
        self._config = config
        self.bpe = bpe
        self.patcher = patcher
        self.shared_multimodal_weights: dict[str, torch.Tensor] = {}

    def share_multimodal_weights(self, weights_dict: dict[str, torch.Tensor]) -> None:
        """Share/bind multimodal weight matrices across text, audio, image, and video modalities."""
        for k, v in weights_dict.items():
            self.shared_multimodal_weights[k] = v

    def get_multimodal_weights(self) -> dict[str, torch.Tensor]:
        """Retrieve the shared multimodal weights dictionary."""
        return self.shared_multimodal_weights

    def export_multimodal_weights(self, formatter: Any, output_path: str, dict_data: Optional[bytes] = None) -> Any:
        """Export shared multimodal weights to encrypted DNA-AI representation format using formatter."""
        return formatter.format_weights(self.shared_multimodal_weights, output_path, dict_data=dict_data)

    def load_multimodal_weights(self, formatter: Any, input_path: str) -> dict[str, torch.Tensor]:
        """Load and bind shared multimodal weights from encrypted DNA-AI file using formatter."""
        weights = formatter.parse_weights(input_path)
        self.share_multimodal_weights(weights)
        return weights

    def encode(self, input_data: Any, modality: str = MODALITY_TEXT) -> list[int]:
        if modality == MODALITY_TEXT:
            if not isinstance(input_data, str):
                raise ValueError("Text modality expects string input.")
            return self.bpe.encode(input_data)
        elif modality in (MODALITY_AUDIO, MODALITY_IMAGE, MODALITY_VIDEO, MODALITY_BYTES):
            if isinstance(input_data, bytes):
                ids = self.patcher.encode_bytes(input_data)
                return self.remap_to_unified(ids, modality)
            elif isinstance(input_data, list):
                return self.remap_to_unified(input_data, modality)
            raise ValueError("Expected bytes or list[int] for media modalities.")
        raise ValueError(f"Unknown modality: {modality}")

    def decode(self, token_ids: list[int], modality: str = MODALITY_TEXT) -> Any:
        if modality == MODALITY_TEXT:
            return self.bpe.decode(token_ids)
        local_ids = self.remap_from_unified(token_ids, modality)
        return self.patcher.decode_to_bytes(local_ids)

    def remap_to_unified(self, ids: list[int], modality: str) -> list[int]:
        offset = self._offset(modality)
        return [i + offset for i in ids] if offset else ids

    def remap_from_unified(self, ids: list[int], modality: str) -> list[int]:
        offset = self._offset(modality)
        return [i - offset for i in ids] if offset else ids

    def _offset(self, modality: str) -> int:
        return {"audio": self._config.audio_range_start,
                "image": self._config.image_range_start,
                "video": self._config.video_range_start,
                "bytes": self._config.audio_range_start}.get(modality, 0)

    def detect_modality(self, token_ids: list[int]) -> str:
        if not token_ids:
            return MODALITY_TEXT
        t = token_ids[0]
        if self._config.audio_range_start <= t <= self._config.audio_range_end:
            return MODALITY_AUDIO
        if self._config.image_range_start <= t <= self._config.image_range_end:
            return MODALITY_IMAGE
        if self._config.video_range_start <= t <= self._config.video_range_end:
            return MODALITY_VIDEO
        return MODALITY_TEXT

    @property
    def vocab_size(self) -> int:
        return self._config.vocab_size

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.bpe.save(os.path.join(directory, "bpe.json"))
        self.patcher.save(os.path.join(directory, "patcher.pkl"))

    @classmethod
    def load(cls, directory: str, config: VocabConfig) -> "UnifiedTokenizer":
        bpe = ByteBPETokenizer.load(os.path.join(directory, "bpe.json"), config)
        patcher = MegabytePatcher.load(os.path.join(directory, "patcher.pkl"))
        return cls(config, bpe, patcher)


# ── VQ Codecs (Audio / Image / Video) ─────────────────────────────────────────

class AudioTokenizer(nn.Module):
    """Compresses raw audio waveforms into discrete tokens using VQ."""
    def __init__(self, config: VocabConfig):
        super().__init__()
        self.codebook_size = config.audio_codebook_size
        self.hidden_dim = 128
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=4, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=4, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(128, self.hidden_dim, 3, stride=2, padding=1),
        )
        self.codebook = nn.Embedding(self.codebook_size, self.hidden_dim)
        self.codebook.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(self.hidden_dim, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 8, stride=4, padding=2), nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 8, stride=4, padding=2), nn.Tanh(),
        )

    def encode(self, waveform: Tensor) -> Tensor:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        z = self.encoder(waveform)
        z_flat = z.transpose(1, 2).reshape(-1, self.hidden_dim)
        dist = (z_flat.pow(2).sum(1, keepdim=True)
                + self.codebook.weight.pow(2).sum(1)
                - 2 * z_flat @ self.codebook.weight.t())
        return torch.argmin(dist, dim=1).view(z.shape[0], z.shape[2])

    def decode(self, token_ids: Tensor) -> Tensor:
        z_q = self.codebook(token_ids).transpose(1, 2)
        return self.decoder(z_q)


class ImageTokenizer(nn.Module):
    """Compresses images into discrete tokens using VQ-VAE."""
    def __init__(self, config: VocabConfig):
        super().__init__()
        self.codebook_size = config.image_codebook_size
        self.hidden_dim = 256
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(256, self.hidden_dim, 1),
        )
        self.codebook = nn.Embedding(self.codebook_size, self.hidden_dim)
        self.codebook.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.hidden_dim, 256, 3, stride=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 3, 3, stride=1, padding=1), nn.Sigmoid(),
        )

    def encode(self, image: Tensor) -> Tensor:
        z = self.encoder(image)
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, C)
        dist = (z_flat.pow(2).sum(1, keepdim=True)
                + self.codebook.weight.pow(2).sum(1)
                - 2 * z_flat @ self.codebook.weight.t())
        return torch.argmin(dist, dim=1).view(B, H * W)

    def decode(self, token_ids: Tensor, H_out: int = 256, W_out: int = 256) -> Tensor:
        B, N = token_ids.shape
        side = int(N ** 0.5)
        z_q = self.codebook(token_ids).view(B, side, side, self.hidden_dim).permute(0, 3, 1, 2)
        img = self.decoder(z_q)
        if img.shape[2:] != (H_out, W_out):
            img = nn.functional.interpolate(img, size=(H_out, W_out), mode="bilinear")
        return img


class VideoTokenizer(nn.Module):
    """Compresses video to discrete tokens using 3D VQ."""
    def __init__(self, config: VocabConfig):
        super().__init__()
        self.codebook_size = config.video_codebook_size
        self.hidden_dim = 256
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 64, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)), nn.ReLU(),
            nn.Conv3d(64, 128, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)), nn.ReLU(),
            nn.Conv3d(128, 256, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)), nn.ReLU(),
            nn.Conv3d(256, self.hidden_dim, 1),
        )
        self.codebook = nn.Embedding(self.codebook_size, self.hidden_dim)
        self.codebook.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)
        self.decoder = nn.Sequential(
            nn.Conv3d(self.hidden_dim, 256, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose3d(256, 128, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)), nn.ReLU(),
            nn.ConvTranspose3d(128, 64, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)), nn.ReLU(),
            nn.Conv3d(64, 3, 3, padding=1), nn.Sigmoid(),
        )

    def encode(self, video: Tensor) -> Tensor:
        z = self.encoder(video)
        B, C, T, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 4, 1).reshape(-1, C)
        dist = (z_flat.pow(2).sum(1, keepdim=True)
                + self.codebook.weight.pow(2).sum(1)
                - 2 * z_flat @ self.codebook.weight.t())
        return torch.argmin(dist, dim=1).view(B, T * H * W)

    def decode(self, token_ids: Tensor, T_out: int = 16, H_out: int = 128, W_out: int = 128) -> Tensor:
        B, N = token_ids.shape
        spatial = int((N / T_out) ** 0.5)
        z_q = self.codebook(token_ids).view(B, T_out, spatial, spatial, self.hidden_dim).permute(0, 4, 1, 2, 3)
        vid = self.decoder(z_q)
        if vid.shape[2:] != (T_out, H_out, W_out):
            vid = nn.functional.interpolate(vid, size=(T_out, H_out, W_out), mode="trilinear")
        return vid


# ── Modality Routers ──────────────────────────────────────────────────────────

class ModalityRouter(nn.Module):
    """Routes raw inputs to appropriate tokenizers → unified token space."""
    def __init__(self, vocab_config: VocabConfig, unified_tokenizer: UnifiedTokenizer):
        super().__init__()
        self.config = vocab_config
        self.tokenizer = unified_tokenizer
        self.audio_codec = AudioTokenizer(vocab_config)
        self.image_codec = ImageTokenizer(vocab_config)
        self.video_codec = VideoTokenizer(vocab_config)

    def forward(self, input_data, modality: ModalityType) -> Tensor:
        device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        if modality == ModalityType.TEXT:
            texts = [input_data] if isinstance(input_data, str) else input_data
            batch = [torch.tensor(self.tokenizer.encode(t, MODALITY_TEXT), device=device) for t in texts]
            return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
        elif modality == ModalityType.AUDIO:
            local = self.audio_codec.encode(input_data.to(device))
            return local + self.config.audio_range_start
        elif modality == ModalityType.IMAGE:
            local = self.image_codec.encode(input_data.to(device))
            return local + self.config.image_range_start
        elif modality == ModalityType.VIDEO:
            local = self.video_codec.encode(input_data.to(device))
            return local + self.config.video_range_start
        raise ValueError(f"Unknown modality: {modality}")


class OutputRouter(nn.Module):
    """Decodes unified token IDs back to raw modality outputs."""
    def __init__(self, vocab_config: VocabConfig, unified_tokenizer: UnifiedTokenizer,
                 audio_codec: AudioTokenizer, image_codec: ImageTokenizer,
                 video_codec: VideoTokenizer):
        super().__init__()
        self.config = vocab_config
        self.tokenizer = unified_tokenizer
        self.audio_codec = audio_codec
        self.image_codec = image_codec
        self.video_codec = video_codec

    def forward(self, unified_ids: Tensor, modality: ModalityType = ModalityType.TEXT):
        if modality == ModalityType.TEXT:
            return [self.tokenizer.decode(row) for row in unified_ids.tolist()]
        elif modality == ModalityType.AUDIO:
            local = torch.clamp(unified_ids - self.config.audio_range_start, 0, self.config.audio_codebook_size - 1)
            return self.audio_codec.decode(local)
        elif modality == ModalityType.IMAGE:
            local = torch.clamp(unified_ids - self.config.image_range_start, 0, self.config.image_codebook_size - 1)
            return self.image_codec.decode(local)
        elif modality == ModalityType.VIDEO:
            local = torch.clamp(unified_ids - self.config.video_range_start, 0, self.config.video_codebook_size - 1)
            return self.video_codec.decode(local, T_out=16, H_out=128, W_out=128)
        raise ValueError(f"Unknown modality: {modality}")
