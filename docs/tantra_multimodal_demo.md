# 🎙️ Tantra-LLM Omnimodal (STT, TTS, Vision, Video) Demonstration

All 4 modalities use Tantra's **Unified 32,768 Discrete Token Codebook**:

| Modality | Input Format | Codec / Engine | Token Count | Output Artifact |
|---|---|---|---|---|
| **🎙️ STT (Speech-to-Text)** | 16,000 Hz Audio Waveform | `AudioTokenizer` (1D Conv VQ) | 250 tokens | Mapped into Context Prefix |
| **🔊 TTS (Text-to-Speech)** | Text Prompt | `AudioTokenizer` (1D Transposed Conv) | 128 tokens | `Assets/tantra_demo_tts.wav` (16428 bytes) |
| **👁️ Vision (Image)** | 128x128 RGB Image | `ImageTokenizer` (2D VQ-VAE) | 1024 tokens | `Assets/tantra_demo_vision.png` |
| **🎬 Video (Spatio-Temporal)** | 8 frames x 64x64 RGB | `VideoTokenizer` (3D Conv VQ) | 2048 tokens | 3D Spatial-Temporal Grid |
