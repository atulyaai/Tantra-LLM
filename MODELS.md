# Tantra-LLM Models Overview

## 🧠 Required Models (No Fallbacks)

Only these 4 models are used:

### 1. **SpikingBrain** (Primary Language Model) ✅ REQUIRED
**Location**: `core/models/spikingbrain_model.py`  
**Config**: `config/model_config.py`  
**Loader**: `utils/model_loader.py::load_spikingbrain()`

**Status**: ✅ Implemented  
**Type**: Custom GPT-like architecture with spiking dynamics  
**Parameters**: ~117M (optimized from 7B)

**Configuration**:
- Hidden size: 768 (reduced from 4096)
- Layers: 6 (reduced from 24)
- Attention heads: 12 (reduced from 32)
- Context window: 1024 tokens
- Vocabulary: 50,257 tokens (GPT-2 vocab)

**Fallback Options**:
- GPT-2 (standard fallback)
- DistilGPT-2 (low-resource mode)
- TextTokenizer (ultra-lightweight)

**Resources**:
- Memory: ~450MB (vs ~14GB for 7B model)
- Device: CPU/GPU
- Optimized for Windows compatibility

---

### 2. **Long-VITA** (Vision Encoder) ✅ REQUIRED
**Location**: `encoders/vision.py`  
**Config**: `config/model_config.py["vision"]`  
**Loader**: `encoders/vision.py::LongVITAVisionEncoder`

**Status**: ✅ Required (original model)  
**Type**: Vision transformer for image/video understanding  
**Model**: microsoft/longvita-16k

**Configuration**:
- Embed dimension: 768
- Load mode: Remote API (configurable)
- Fallback: Simple CNN encoder

**Features**:
- Image encoding
- Video frame encoding (16K context)
- Remote API support
- Local fallback encoder

**Resources**:
- Model size: ~1-2GB
- Supports GPU acceleration
- Skips in low-resource mode

---

### 3. **Whisper** (Speech-to-Text) ✅ REQUIRED
**Location**: `encoders/audio.py`  
**Config**: `config/model_config.py["audio"]`  
**Model**: large-v3

**Status**: ✅ Required (original model)  
**Type**: Speech-to-text encoder  
**Size**: Large-v3 (original model)

**Configuration**:
- Embed dimension: 768
- Model: openai/whisper-base
- Load mode: Local

**Features**:
- Audio transcription
- Feature extraction
- Real embeddings

**Resources**:
- Model size: ~150MB (base)
- Lightweight and fast

---

### 4. **Coqui TTS** (Text-to-Speech) ✅ REQUIRED
**Location**: `utils/model_loader.py::load_coqui_tts()`  
**Model**: TTS tts_models/en/ljspeech/tacotron2-DDC

**Status**: ✅ Required (original model)  
**Type**: Text-to-speech generation  
**Purpose**: Speech output generation

---

## 🔧 Model Loading Flow

```
ModelLoader
├── load_spikingbrain() → Custom SpikingBrain model ✅ REQUIRED
│   └── SpikingBrainForCausalLM (custom)
│
├── load_whisper() → Whisper large-v3 ✅ REQUIRED
│   └── Whisper large-v3 (original)
│
├── get_vision_encoder() → Long-VITA ✅ REQUIRED
│   └── microsoft/longvita-16k (original)
│
└── load_coqui_tts() → Coqui TTS ✅ REQUIRED
    └── TTS tts_models/en/ljspeech/tacotron2-DDC
```

## 📊 Model Comparison

| Model | Type | Parameters | Status | Memory | Purpose |
|-------|------|-----------|--------|--------|---------|
| SpikingBrain | Custom LLM | ~117M | ✅ Required | ~450MB | Reasoning, text generation |
| Long-VITA | Vision | ~1B | ✅ Required | ~1-2GB | Image/video understanding |
| Whisper | Audio STT | ~760M (large-v3) | ✅ Required | ~1.5GB | Speech-to-text |
| Coqui TTS | Audio TTS | Varies | ✅ Required | ~500MB | Speech generation |

## 🎯 Optimization Summary

All models have been **optimized for resource efficiency**:

1. **SpikingBrain**: Reduced from 7B → 117M parameters (~98% reduction)
2. **Context**: Reduced from 32K → 1K tokens
3. **Whisper**: Reduced from large-v3 → base model
4. **Memory**: Reduced from 10K → 1K episodes
5. **Dimensions**: All aligned to 768 for consistency

**Total System**: ~2GB RAM required (vs ~20GB+ for original design)

## 🚀 Getting Started

### Load SpikingBrain
```python
from utils.model_loader import ModelLoader
from config import model_config

loader = ModelLoader(model_config.MODEL_CONFIG)
spb = loader.load_spikingbrain()
model = spb["model"]
tokenizer = spb["tokenizer"]
```

### Load Vision Encoder
```python
from encoders.vision import VisionEncoder

vision = VisionEncoder(embed_dim=768)
embeddings = vision(image)  # Returns 768-dim tensor
```

### Load Audio Encoder
```python
from encoders.audio import AudioEncoder

audio = AudioEncoder(embed_dim=768)
embeddings = audio(audio_data)  # Returns 768-dim tensor
```

## 📝 Notes

- **SpikingBrain** is the core reasoning engine
- **Long-VITA** provides vision understanding
- **Whisper** handles speech input
- All models use **768 dimensions** for compatibility
- Fallback models ensure system works even if primary models fail
- Low-resource mode uses DistilGPT-2 for minimal RAM usage

## 🔄 Next Steps

1. ✅ Test SpikingBrain loading and generation
2. 🔄 Integrate Long-VITA for vision
3. 🔄 Test multimodal (text + image) conversations
4. 🔄 Add Whisper for voice input

