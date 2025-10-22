#!/bin/bash
# Install dependencies for Tantra LLM

echo "📦 Installing Python dependencies for Tantra LLM..."

# Install basic dependencies
pip3 install --upgrade pip

# Install core ML dependencies
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install transformers and tokenizers
pip3 install transformers tokenizers

# Install data processing
pip3 install datasets safetensors

# Install utilities
pip3 install tqdm requests pyyaml

# Install web framework
pip3 install fastapi uvicorn

# Install optional dependencies
pip3 install openai-whisper webrtcvad piper-phonemizer faiss-cpu sentence-transformers prometheus-client

echo "✅ Dependencies installed successfully!"
echo "Run 'python3 Test/test_comprehensive.py' to test the installation."