#!/bin/bash
# Tantra LLM - Server Setup Script

echo "🚀 Setting up Tantra LLM on server..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers tokenizers datasets safetensors tqdm requests

# Download datasets
echo "📥 Downloading datasets..."
python Training/download_datasets.py

# Combine datasets
echo "🔄 Combining datasets..."
python Training/combine_datasets.py

# Train tokenizer
echo "🔤 Training tokenizer..."
python Training/tokenizer_train.py --input_glob "Dataset/combined_full_training.jsonl" --out Model/tokenizer.json

# Start training
echo "🎯 Starting progressive training..."
echo "This will take approximately 5 hours on a good server..."
python Training/training_main.py

echo "✅ Training complete! Check Model/tantra_weights.safetensors"
