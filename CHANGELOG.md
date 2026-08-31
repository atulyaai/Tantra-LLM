# Changelog

All notable changes to **Tantra-LLM** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-07

### Added
- **NeuroCore Engine**: Custom Transformer replacement featuring:
  - **ALRA (Adaptive Linear Resonance Attention)**: $O(n)$ linear-time attention mechanism with dynamic learnable forget gates.
  - **SGP (Sparse Gated Projection)**: Brain-inspired sparse Feed-Forward Network with top-$k\%$ (10%) neuron activation per token.
  - **DSN (Dynamic Scale Norm)**: Dynamic LayerNorm replacement with input-dependent learned scale.
- **BitNet 1-Bit Ternary Weights**: Drop-in `BitLinear` layers with ternary weights $\{-1, 0, +1\}$, straight-through gradient estimators, and optimized CPU kernels using positive/negative mask decomposition.
- **DNA-AI Weight Compression**: Custom near-lossless / lossless weight codec integrating ZSTD domain dictionaries, neural residual prediction, adaptive Huffman coding, 2-bit DNA symbol packing, and parity checking.
- **Unified Multimodal Fusion**: 32K token space accommodating text (Byte-BPE), raw byte streams (MegaByte patcher), audio VQ-VAE, image VQ-VAE, and video 3D-VQ codecs via `ModalityRouter` and `OutputRouter`.
- **Auto-Adaptive Hardware Engine**: Runtime profiling for CPU (AVX2/AVX512), GPU (CUDA/MPS), RAM budget tracking, and real-time `AdaptiveScheduler`.
- **Sparse MoE Architecture**: Expert registry tracking up to 500 domain experts, ALRA context-aware router, auxiliary load-balancer, and `LazyExpertLoader` with LRU RAM caching.
- **Clean AI Engineer Refactoring**: Consolidated 50+ fragmented files across 11 subdirectories into a unified `tantra` core package with 9 clean modules.
