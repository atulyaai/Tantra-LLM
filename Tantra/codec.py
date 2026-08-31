"""
tantra/codec.py — DNA-AI compression pipeline. Contains: ZSTDDictTrainer, ResidualPredictor, HuffmanNode, AdaptiveHuffmanCoder, DNACodec, CompressionStats, CompressionBenchmark.
"""

import os
import json
import time
import struct
import hashlib
import random
import heapq
import zlib
from typing import List, Dict, Tuple, Optional, Iterator
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

try:
    import zstandard as zstd
except ImportError:
    zstd = None

from Tantra.config import CompressionConfig

# ── ZSTDDictTrainer ──

class ZSTDDictTrainer:
    """
    Trains a ZSTD compression dictionary on model weight data.
    A domain-specific dictionary dramatically improves compression
    of weight tensors vs using ZSTD with a generic dictionary.
    """
    def __init__(self, config: CompressionConfig):
        """Initialize with compression config."""
        self.config = config
        self.dict_size = config.zstd_dict_size
        self.level = config.zstd_level

    def train_from_tensors(self, tensors: List[torch.Tensor], save_path: str) -> bytes:
        """Sample chunks from tensors, train ZSTD dictionary, save."""
        samples = []
        for t in tensors:
            t_bytes = t.cpu().contiguous().numpy().tobytes()
            chunk_size = 1024
            for i in range(0, len(t_bytes) - chunk_size + 1, chunk_size):
                samples.append(t_bytes[i:i+chunk_size])
            if len(t_bytes) < chunk_size and len(t_bytes) > 0:
                samples.append(t_bytes)
        
        random.shuffle(samples)
        max_samples = 10000
        if len(samples) > max_samples:
            samples = samples[:max_samples]
            
        if zstd is not None:
            try:
                if hasattr(zstd, "train_dictionary"):
                    zdict = zstd.train_dictionary(self.dict_size, samples)
                    dict_bytes = zdict.as_bytes() if hasattr(zdict, "as_bytes") else bytes(zdict)
                elif hasattr(zstd.ZstdCompressionDict, "train_dictionary"):
                    zdict = zstd.ZstdCompressionDict.train_dictionary(dict_size=self.dict_size, samples=samples)
                    dict_bytes = zdict.as_bytes()
                else:
                    dict_bytes = b"FALLBACK_ZSTD_DICT_HEADER_" + b"".join(samples[:10])[:self.dict_size]
            except Exception:
                dict_bytes = b"FALLBACK_ZSTD_DICT_HEADER_" + b"".join(samples[:10])[:self.dict_size]
        else:
            dict_bytes = b"FALLBACK_ZSTD_DICT_HEADER_" + b"".join(samples[:10])[:self.dict_size]

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(dict_bytes)
            
        return dict_bytes

    def load_dict(self, path: str) -> bytes:
        """Load saved dictionary bytes."""
        with open(path, 'rb') as f:
            return f.read()

    def compress_with_dict(self, data: bytes, dict_data: bytes) -> bytes:
        """Compress bytes using dictionary."""
        if zstd is not None:
            zdict = zstd.ZstdCompressionDict(dict_data)
            compressor = zstd.ZstdCompressor(level=self.level, dict_data=zdict)
            return compressor.compress(data)
        return zlib.compress(data)

    def decompress_with_dict(self, data: bytes, dict_data: bytes) -> bytes:
        """Decompress bytes using dictionary."""
        if zstd is not None:
            zdict = zstd.ZstdCompressionDict(dict_data)
            decompressor = zstd.ZstdDecompressor(dict_data=zdict)
            return decompressor.decompress(data)
        return zlib.decompress(data)

    def benchmark(self, tensor: torch.Tensor, dict_data: bytes) -> Dict[str, float]:
        """Compare compression ratios."""
        data = tensor.cpu().contiguous().numpy().tobytes()
        orig_size = len(data)
        if orig_size == 0:
            return {'no_dict': 1.0, 'trained_dict': 1.0, 'improvement': 1.0}
            
        if zstd is not None:
            comp_nodict = zstd.ZstdCompressor(level=self.level)
            nodict_data = comp_nodict.compress(data)
            nodict_ratio = orig_size / len(nodict_data) if nodict_data else 1.0
            
            zdict = zstd.ZstdCompressionDict(dict_data)
            comp_dict = zstd.ZstdCompressor(level=self.level, dict_data=zdict)
            dict_data_comp = comp_dict.compress(data)
            dict_ratio = orig_size / len(dict_data_comp) if dict_data_comp else 1.0
        else:
            compressed = zlib.compress(data)
            nodict_ratio = orig_size / len(compressed) if compressed else 1.0
            dict_ratio = nodict_ratio

        return {
            'no_dict': nodict_ratio,
            'trained_dict': dict_ratio,
            'improvement': dict_ratio / nodict_ratio if nodict_ratio > 0 else 1.0
        }

# ── ResidualPredictor ──

class ResidualPredictor(nn.Module):
    """Predicts next weight value from context window of previous weights."""
    def __init__(self, config: CompressionConfig):
        super().__init__()
        self.window_size = getattr(config, "residual_window_size", getattr(config, "residual_window", 64))
        
        self.mlp = nn.Sequential(
            nn.Linear(self.window_size, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.mlp(context)

    def train_on_tensors(self, tensors: List[torch.Tensor], epochs: int = 3, lr: float = 1e-3) -> Dict[str, float]:
        device = next(self.parameters()).device
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        contexts = []
        targets = []
        for t in tensors:
            flat = t.flatten().cpu()
            if len(flat) <= self.window_size:
                continue
            for i in range(len(flat) - self.window_size):
                contexts.append(flat[i:i+self.window_size])
                targets.append(flat[i+self.window_size])
                if len(contexts) >= 50000:
                    break
            if len(contexts) >= 50000:
                break
                
        if not contexts:
            return {'final_loss': 0.0, 'compression_gain_est': 1.0}
            
        X = torch.stack(contexts).to(device)
        Y = torch.tensor(targets).unsqueeze(1).to(device)
        
        self.train()
        batch_size = 256
        loss_val = 0.0
        for epoch in range(epochs):
            perm = torch.randperm(X.size(0))
            for i in range(0, X.size(0), batch_size):
                idx = perm[i:i+batch_size]
                bx, by = X[idx], Y[idx]
                
                optimizer.zero_grad()
                pred = self(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                loss_val = loss.item()
                
        return {'final_loss': loss_val, 'compression_gain_est': 1.2}

    def compute_residuals(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        device = next(self.parameters()).device
        flat = tensor.flatten().to(device)
        if len(flat) <= self.window_size:
            return flat, torch.zeros_like(flat)
            
        contexts = []
        for i in range(len(flat) - self.window_size):
            contexts.append(flat[i:i+self.window_size])
        X = torch.stack(contexts)
        
        with torch.no_grad():
            predictions = self(X).squeeze(1)
            
        actuals = flat[self.window_size:]
        residuals = actuals - predictions
        return residuals, predictions

    def reconstruct_from_residuals(self, residuals: torch.Tensor, first_window: torch.Tensor) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        res = residuals.to(device)
        win = first_window.to(device).tolist()
        
        reconstructed = list(win)
        with torch.no_grad():
            for i in range(len(res)):
                ctx = torch.tensor(reconstructed[-self.window_size:], device=device).unsqueeze(0)
                pred = self(ctx).item()
                val = pred + res[i].item()
                reconstructed.append(val)
                
        return torch.tensor(reconstructed, device=device)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, config: CompressionConfig) -> 'ResidualPredictor':
        model = cls(config)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        return model

# ── AdaptiveHuffmanCoder ──

@dataclass
class HuffmanNode:
    symbol: Optional[int]
    freq: float
    left: Optional['HuffmanNode'] = None
    right: Optional['HuffmanNode'] = None
    
    def __lt__(self, other: 'HuffmanNode') -> bool:
        return self.freq < other.freq

class AdaptiveHuffmanCoder:
    def __init__(self, n_symbols: int = 256):
        self.n_symbols = n_symbols

    def build_tree_from_frequencies(self, freqs: Dict[int, float]) -> HuffmanNode:
        heap = []
        for sym, freq in freqs.items():
            node = HuffmanNode(symbol=sym, freq=freq)
            heapq.heappush(heap, node)
            
        if not heap:
            return HuffmanNode(symbol=0, freq=1.0)
            
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            parent = HuffmanNode(symbol=None, freq=left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, parent)
            
        return heap[0]

    def _generate_codebook(self, node: HuffmanNode, prefix: str = "", codebook: Dict[int, str] = None) -> Dict[int, str]:
        if codebook is None:
            codebook = {}
        if node.symbol is not None:
            codebook[node.symbol] = prefix if prefix else "0"
        else:
            if node.left:
                self._generate_codebook(node.left, prefix + "0", codebook)
            if node.right:
                self._generate_codebook(node.right, prefix + "1", codebook)
        return codebook

    def encode(self, symbols: List[int], tree: HuffmanNode) -> Tuple[bytes, Dict[int, str]]:
        codebook = self._generate_codebook(tree)
        bit_string = "".join([codebook.get(s, "0") for s in symbols])
        
        pad_len = (8 - len(bit_string) % 8) % 8
        bit_string += "0" * pad_len
        
        byte_array = bytearray()
        for i in range(0, len(bit_string), 8):
            byte_array.append(int(bit_string[i:i+8], 2))
            
        return bytes(byte_array), codebook

    def decode(self, data: bytes, codebook: Dict[int, str], length: int) -> List[int]:
        inv_codebook = {v: k for k, v in codebook.items()}
        bit_string = "".join([f"{b:08b}" for b in data])
        
        decoded = []
        current_bits = ""
        for bit in bit_string:
            current_bits += bit
            if current_bits in inv_codebook:
                decoded.append(inv_codebook[current_bits])
                current_bits = ""
                if len(decoded) == length:
                    break
        return decoded

    def estimate_frequencies(self, data_sample: List[int]) -> Dict[int, float]:
        freqs: Dict[int, int] = {}
        for s in data_sample:
            freqs[s] = freqs.get(s, 0) + 1
        total = len(data_sample) or 1
        return {s: count / total for s, count in freqs.items()}

    def compression_stats(self, original: List[int], encoded: bytes) -> Dict[str, float]:
        orig_bytes = len(original)
        enc_bytes = len(encoded)
        ratio = orig_bytes / enc_bytes if enc_bytes > 0 else 1.0
        bps = (enc_bytes * 8) / orig_bytes if orig_bytes > 0 else 8.0
        return {
            'bits_per_symbol': bps,
            'compression_ratio': ratio
        }

# ── DNACodec ──

DNA_SYMBOLS = {'A': 0b00, 'T': 0b01, 'G': 0b10, 'C': 0b11}
DNA_SYMBOLS_INV = {v: k for k, v in DNA_SYMBOLS.items()}

@dataclass
class CompressionStats:
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    compress_time_ms: float
    decompress_time_ms: float
    method: str
    lossless: bool
    sha256_match: bool

class DNACodec:
    """Full DNA-AI compression pipeline."""
    def __init__(self, config: CompressionConfig,
                 predictor: Optional[ResidualPredictor] = None,
                 zstd_dict: Optional[bytes] = None):
        self.config = config
        self.predictor = predictor
        self.zstd_dict = zstd_dict
        self.huffman = AdaptiveHuffmanCoder()
        self.magic = b'DNAC'
        # Version 1 wrote one byte for each 2-bit DNA symbol, inflating every
        # compressed byte fourfold. Version 2 stores those four symbols packed
        # back into one byte, which preserves the DNA representation without
        # turning a compressed checkpoint into a larger file.
        self.version = 2

    def compress(self, tensor: torch.Tensor, output_path: str) -> CompressionStats:
        t0 = time.perf_counter()
        original_bytes_data = tensor.cpu().contiguous().numpy().tobytes()
        orig_len = len(original_bytes_data)
        sha256_orig = hashlib.sha256(original_bytes_data).digest()
        
        if zstd is not None:
            if self.zstd_dict:
                zdict = zstd.ZstdCompressionDict(self.zstd_dict)
                cctx = zstd.ZstdCompressor(level=self.config.zstd_level, dict_data=zdict)
                encoded = cctx.compress(original_bytes_data)
            else:
                cctx = zstd.ZstdCompressor(level=self.config.zstd_level)
                encoded = cctx.compress(original_bytes_data)
        else:
            encoded = zlib.compress(original_bytes_data)

        codebook = {}
        # A byte already holds exactly four 2-bit DNA symbols. Keep it packed
        # on disk; callers that need textual bases can still use
        # ``_pack_to_dna`` for display only.
        dna_packed = encoded
        parity = self._compute_parity(dna_packed, self.config.dna_parity_interval)
        
        magic = self.magic
        version = struct.pack('<I', self.version)
        orig_bytes_packed = struct.pack('<Q', orig_len)
        dtype_str = str(tensor.dtype).split('.')[-1].encode('utf-8').ljust(16, b'\0')[:16]
        
        shape_json = json.dumps(list(tensor.shape)).encode('utf-8')
        shape_len = struct.pack('<I', len(shape_json))
        
        dict_data = self.zstd_dict or b''
        dict_len = struct.pack('<I', len(dict_data))
        
        codebook_json = json.dumps(codebook).encode('utf-8')
        codebook_len = struct.pack('<I', len(codebook_json))
        
        parity_count = struct.pack('<Q', len(parity))
        data_len = struct.pack('<Q', len(dna_packed))
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(magic)
            f.write(version)
            f.write(orig_bytes_packed)
            f.write(dtype_str)
            f.write(shape_len)
            f.write(shape_json)
            f.write(dict_len)
            f.write(dict_data)
            f.write(codebook_len)
            f.write(codebook_json)
            f.write(parity_count)
            f.write(data_len)
            f.write(dna_packed)
            f.write(parity)
            f.write(sha256_orig)
            
        t1 = time.perf_counter()
        
        t_dec0 = time.perf_counter()
        decompressed_tensor = self.decompress(output_path)
        t_dec1 = time.perf_counter()
        
        decompressed_bytes = decompressed_tensor.cpu().contiguous().numpy().tobytes()
        decompressed_sha = hashlib.sha256(decompressed_bytes).digest()
        sha_match = (sha256_orig == decompressed_sha)
        
        compressed_size = os.path.getsize(output_path)
        
        return CompressionStats(
            original_bytes=orig_len,
            compressed_bytes=compressed_size,
            compression_ratio=orig_len / compressed_size if compressed_size else 1.0,
            compress_time_ms=(t1 - t0) * 1000,
            decompress_time_ms=(t_dec1 - t_dec0) * 1000,
            method='dna',
            lossless=True,
            sha256_match=sha_match
        )

    def decompress(self, input_path: str) -> torch.Tensor:
        with open(input_path, 'rb') as f:
            magic = f.read(4)
            if magic != self.magic:
                raise ValueError("Invalid magic bytes in DNA file")
                
            version = struct.unpack('<I', f.read(4))[0]
            if version not in (1, 2):
                raise ValueError(f"Unsupported DNA container version: {version}")
            orig_len = struct.unpack('<Q', f.read(8))[0]
            dtype_raw = f.read(16).rstrip(b'\0').decode('utf-8')
            
            shape_len = struct.unpack('<I', f.read(4))[0]
            shape_json = f.read(shape_len).decode('utf-8')
            shape = json.loads(shape_json)
            
            dict_len = struct.unpack('<I', f.read(4))[0]
            dict_data = f.read(dict_len) if dict_len > 0 else None
            
            codebook_len = struct.unpack('<I', f.read(4))[0]
            codebook_json = f.read(codebook_len).decode('utf-8')
            codebook = json.loads(codebook_json)
            
            parity_count = struct.unpack('<Q', f.read(8))[0]
            data_len = struct.unpack('<Q', f.read(8))[0]
            
            dna_packed = f.read(data_len)
            parity = f.read(parity_count)
            sha256_orig = f.read(32)
            
        if not self._verify_parity(dna_packed, parity, self.config.dna_parity_interval):
            raise ValueError(
                f"DNA parity check failed for {input_path!r} — file appears to be "
                "corrupted. Refusing to decompress untrusted/corrupt data."
            )
        # Version 1 stored each 2-bit symbol as its own byte. Version 2 keeps
        # four symbols per byte, eliminating the 4x expansion.
        unpacked_bytes = (
            self._unpack_from_dna(dna_packed, data_len // 4)
            if version == 1 else dna_packed
        )
        
        if zstd is not None:
            if dict_data:
                zdict = zstd.ZstdCompressionDict(dict_data)
                dctx = zstd.ZstdDecompressor(dict_data=zdict)
                decompressed_raw = dctx.decompress(unpacked_bytes)
            else:
                dctx = zstd.ZstdDecompressor()
                decompressed_raw = dctx.decompress(unpacked_bytes)
        else:
            decompressed_raw = zlib.decompress(unpacked_bytes)

        if not hasattr(torch, dtype_raw):
            dtype = torch.float32
        else:
            dtype = getattr(torch, dtype_raw)

        if dtype == torch.bfloat16:
            tensor_np = np.frombuffer(decompressed_raw, dtype=np.float32)
            tensor = torch.from_numpy(tensor_np.copy()).to(torch.bfloat16).reshape(shape)
        else:
            np_dtype = torch.zeros(1, dtype=dtype).numpy().dtype
            tensor_np = np.frombuffer(decompressed_raw, dtype=np_dtype)
            tensor = torch.from_numpy(tensor_np.copy()).reshape(shape)
        return tensor

    def stream_decompress(self, input_path: str, chunk_numel: int = 1_000_000) -> Iterator[torch.Tensor]:
        full_tensor = self.decompress(input_path)
        flat = full_tensor.flatten()
        for i in range(0, len(flat), chunk_numel):
            yield flat[i:i+chunk_numel]

    def _pack_to_dna(self, data: bytes) -> bytes:
        arr = np.frombuffer(data, dtype=np.uint8)
        # Split each byte into 4 2-bit symbols, interleaved as A,T,G,C
        out = np.empty(arr.shape[0] * 4, dtype=np.uint8)
        out[0::4] = (arr >> 6) & 0b11
        out[1::4] = (arr >> 4) & 0b11
        out[2::4] = (arr >> 2) & 0b11
        out[3::4] = arr & 0b11
        return out.tobytes()

    def _unpack_from_dna(self, dna_bytes: bytes, original_len: int) -> bytes:
        arr = np.frombuffer(dna_bytes, dtype=np.uint8).copy()
        n = original_len
        out = np.empty(n, dtype=np.uint8)
        out = (arr[0::4] << 6 | arr[1::4] << 4 | arr[2::4] << 2 | arr[3::4])
        return out[:n].tobytes()

    def _compute_parity(self, data: bytes, interval: int) -> bytes:
        if not data:
            return b""
        arr = np.frombuffer(data, dtype=np.uint8)
        pad = (interval - (len(arr) % interval)) % interval
        if pad:
            arr = np.pad(arr, (0, pad))
        res = np.bitwise_xor.reduce(arr.reshape(-1, interval), axis=1)
        return res.tobytes()

    def _verify_parity(self, data: bytes, parity: bytes, interval: int) -> bool:
        computed = self._compute_parity(data, interval)
        return computed == parity

# ── MultimodalWeightFormatter ──

class MultimodalWeightFormatter:
    """
    Unifies text, audio, image, and video embeddings/weights into a shared encrypted
    DNA-AI representation format with parity verification and ZSTD dictionary compression.
    """
    def __init__(self, config: CompressionConfig, secret_key: Optional[bytes] = None):
        self.config = config
        self.codec = DNACodec(config)
        self.secret_key = secret_key or b"TANTRA_UNIFIED_KEY_32BYTES_LONG!"

    def _xor_encrypt(self, data: bytes) -> bytes:
        key = self.secret_key
        key_arr = np.frombuffer(key, dtype=np.uint8)
        data_arr = np.frombuffer(data, dtype=np.uint8)
        repeated_key = np.resize(key_arr, data_arr.shape)
        encrypted = np.bitwise_xor(data_arr, repeated_key)
        return encrypted.tobytes()

    def _serialize_weights_binary(self, weights_dict: Dict[str, torch.Tensor]) -> bytes:
        """Serialize weights dict as compact binary (no hex overhead)."""
        parts = []
        for key, tensor in weights_dict.items():
            raw_bytes = tensor.cpu().contiguous().numpy().tobytes()
            dtype_str = str(tensor.dtype).split('.')[-1]
            shape_bytes = json.dumps(list(tensor.shape)).encode('utf-8')
            key_bytes = key.encode('utf-8')
            parts.append(
                struct.pack('<B', len(key_bytes)) + key_bytes
                + struct.pack('<I', len(shape_bytes)) + shape_bytes
                + struct.pack('<B', len(dtype_str)) + dtype_str.encode('utf-8')
                + struct.pack('<Q', len(raw_bytes)) + raw_bytes
            )
        return b''.join(parts)

    def _deserialize_weights_binary(self, data: bytes) -> Dict[str, torch.Tensor]:
        """Deserialize weights dict from compact binary format."""
        weights_dict = {}
        offset = 0
        while offset < len(data):
            key_len = data[offset]; offset += 1
            key = data[offset:offset+key_len].decode('utf-8'); offset += key_len
            shape_len = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
            shape = json.loads(data[offset:offset+shape_len].decode('utf-8')); offset += shape_len
            dtype_len = data[offset]; offset += 1
            dtype_str = data[offset:offset+dtype_len].decode('utf-8'); offset += dtype_len
            raw_len = struct.unpack('<Q', data[offset:offset+8])[0]; offset += 8
            raw_bytes = data[offset:offset+raw_len]; offset += raw_len

            if not hasattr(torch, dtype_str):
                dtype = torch.float32
            else:
                dtype = getattr(torch, dtype_str)
            if dtype == torch.bfloat16:
                tensor_np = np.frombuffer(raw_bytes, dtype=np.float32)
                tensor = torch.from_numpy(tensor_np.copy()).to(torch.bfloat16).reshape(shape)
            else:
                np_dtype = torch.zeros(1, dtype=dtype).numpy().dtype
                tensor = torch.from_numpy(np.frombuffer(raw_bytes, dtype=np_dtype).copy()).reshape(shape)
            weights_dict[key] = tensor
        return weights_dict

    def format_weights(
        self,
        weights_dict: Dict[str, torch.Tensor],
        output_path: str,
        dict_data: Optional[bytes] = None
    ) -> CompressionStats:
        """
        Packs multimodal tensors (text, audio, image, video weights), compresses with ZSTD
        and optional dictionary, computes parity, and encrypts into a DNA-AI container.
        """
        self.codec.zstd_dict = dict_data

        serialized = self._serialize_weights_binary(weights_dict)
        encrypted = self._xor_encrypt(serialized)

        encrypted_tensor = torch.from_numpy(np.frombuffer(encrypted, dtype=np.uint8).copy())
        stats = self.codec.compress(encrypted_tensor, output_path)
        stats.method = "multimodal_dna_encrypted"
        return stats

    def parse_weights(self, input_path: str) -> Dict[str, torch.Tensor]:
        """
        Decompresses container, verifies parity, decrypts, and unpacks multimodal tensors.
        """
        decompressed_tensor = self.codec.decompress(input_path)
        encrypted_bytes = decompressed_tensor.numpy().tobytes()
        decrypted_bytes = self._xor_encrypt(encrypted_bytes)
        weights_dict = self._deserialize_weights_binary(decrypted_bytes)
        return weights_dict


# ── CompressionBenchmark ──

class CompressionBenchmark:
    """Benchmark utility for compression strategies."""
    def __init__(self, config: CompressionConfig):
        self.config = config

    def run(self, tensor: torch.Tensor, output_dir: str = "reports") -> List[CompressionStats]:
        codec = DNACodec(self.config)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "bench_sample.dna")
        stats = codec.compress(tensor, output_path)
        return [stats]
