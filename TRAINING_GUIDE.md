# Tantra Conversational Speech Training Guide

Complete guide for training Tantra's conversational and speech capabilities.

## 🎯 Overview

This guide covers the complete training pipeline for Tantra's conversational and speech features, including data preparation, model training, and GitHub integration.

## 📋 Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **GPU**: Optional but recommended for faster training

### Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# For GitHub integration (optional)
pip install PyGithub

# For advanced audio processing (optional)
pip install librosa soundfile
```

## 🚀 Quick Start

### 1. Basic Training

```bash
# Train both conversational and speech models
python run_training.py

# Train with GitHub integration
python run_training.py --github-token YOUR_GITHUB_TOKEN --github-repo your-username/tantra-models
```

### 2. Specific Model Training

```bash
# Train only conversational model
python run_training.py --conversational-only

# Train only speech model
python run_training.py --speech-only
```

### 3. Interactive Demo

```bash
# Start interactive conversation
python demo_conversational_speech.py

# With pre-trained model
python demo_conversational_speech.py --model path/to/model.pt
```

## 📊 Data Preparation

### Conversation Data Format

Create conversation data in JSON format:

```json
{
  "conversation_id": "conv_001",
  "type": "general_chat",
  "personality": "helpful",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "Hello! I'm doing well, thank you for asking. How can I help you today?"}
  ],
  "context": "Initial greeting"
}
```

### Speech Data Format

Speech data should include:

- **Audio files**: WAV format, 16kHz sample rate
- **Text transcripts**: Corresponding text files
- **Metadata**: Duration, quality, speaker info

### Data Organization

```
data/
├── raw/
│   ├── conversations/
│   │   ├── train/
│   │   │   ├── general_chat.json
│   │   │   ├── technical_support.json
│   │   │   └── creative_writing.json
│   │   └── val/
│   │       └── conversations.json
│   └── speech/
│       ├── train/
│       │   ├── audio_files.wav
│       │   └── metadata.json
│       └── val/
│           └── speech_samples.wav
└── processed/
    └── (auto-generated)
```

## ⚙️ Configuration

### Training Configuration

Create a custom training configuration:

```python
from src.training.training_config import TrainingConfig

config = TrainingConfig(
    # Model settings
    model_name="tantra_custom_v1.0",
    learning_rate=1e-4,
    batch_size=4,
    num_epochs=10,
    
    # Conversational settings
    conversation_max_length=2048,
    conversation_context_window=512,
    
    # Speech settings
    speech_sample_rate=16000,
    speech_max_duration=30.0,
    
    # GitHub integration
    github_repo="your-username/tantra-models",
    github_token="your_github_token",
    auto_commit=True
)

# Save configuration
config.save_config("custom_config.json")
```

### Model Configuration

```python
from src.core.tantra_llm import TantraConfig

# Base model configuration
tantra_config = TantraConfig(
    d_model=512,           # Model dimension
    n_layers=12,           # Number of transformer layers
    n_heads=8,             # Number of attention heads
    vocab_size=50000,      # Vocabulary size
    max_seq_length=8192    # Maximum sequence length
)
```

## 🏗️ Training Pipeline

### 1. Conversational Training

The conversational training system includes:

#### Data Loading
- **ConversationDataset**: Loads conversation data
- **ConversationDataLoader**: Manages data batches
- **Personality Modeling**: Multi-trait personality system

#### Training Process
```python
from src.training.conversational_trainer import ConversationalTrainer

# Initialize trainer
trainer = ConversationalTrainer(config, model)

# Train model
trainer.train()

# Generate responses
response = trainer.generate_response(
    user_message="Hello!",
    context="greeting",
    personality="helpful"
)
```

#### Quality Metrics
- **Response Length**: Average response length
- **Response Time**: Generation speed
- **Relevance Score**: Word overlap with input
- **Coherence Score**: Sentence structure quality

### 2. Speech Training

The speech training system features:

#### Audio Processing
- **SpeechEncoder**: Converts audio to embeddings
- **SpeechDecoder**: Converts embeddings to audio
- **Mel Spectrograms**: High-quality audio representation

#### Training Process
```python
from src.training.speech_trainer import SpeechTrainer

# Initialize trainer
speech_trainer = SpeechTrainer(config, model)

# Train model
speech_trainer.train()

# Text to speech
audio = speech_trainer.text_to_speech(
    text="Hello, this is Tantra!",
    voice_style="neutral"
)

# Speech to text
text = speech_trainer.speech_to_text(audio)
```

#### Quality Metrics
- **Reconstruction MSE**: Audio reconstruction quality
- **Spectral Distance**: Mel spectrogram similarity
- **Inference Time**: Processing speed

## 🔧 Advanced Configuration

### Custom Data Loaders

```python
from src.training.data_loader import ConversationDataset

# Create custom dataset
class CustomConversationDataset(ConversationDataset):
    def __init__(self, data_path, config, tokenizer=None):
        super().__init__(data_path, config, tokenizer)
        # Add custom processing
    
    def __getitem__(self, idx):
        # Custom data processing
        return super().__getitem__(idx)
```

### Custom Training Loops

```python
# Custom training with specific parameters
trainer = ConversationalTrainer(config, model)

# Modify training parameters
trainer.optimizer = optim.AdamW(
    trainer.model.parameters(),
    lr=2e-4,  # Higher learning rate
    weight_decay=0.01
)

# Custom loss function
def custom_loss(outputs, targets):
    # Custom loss implementation
    return loss

trainer.criterion = custom_loss
```

## 📈 Monitoring and Logging

### TensorBoard Integration

```bash
# Start TensorBoard
tensorboard --logdir logs/

# View training metrics
# Open http://localhost:6006 in browser
```

### Custom Metrics

```python
# Add custom metrics
def custom_metric(predictions, targets):
    # Calculate custom metric
    return metric_value

# Log custom metrics
trainer.writer.add_scalar('Custom/Metric', metric_value, step)
```

## 🚀 GitHub Integration

### Automatic Model Saving

```python
from src.training.github_integration import GitHubModelManager

# Initialize GitHub manager
github_manager = GitHubModelManager(
    github_token="your_token",
    repository="your-username/tantra-models"
)

# Save model with metadata
success = github_manager.save_model_file(
    local_path="model.pt",
    github_path="models/tantra_v1.0.pt",
    commit_message="Update Tantra model"
)
```

### Version Management

```python
from src.training.github_integration import ModelVersionManager

# Initialize version manager
version_manager = ModelVersionManager(github_manager)

# Save with full versioning
success = version_manager.save_model_with_versioning(
    model_path="model.pt",
    model_name="tantra_v1.0",
    training_config=config.to_dict(),
    performance_metrics={"accuracy": 0.95}
)
```

## 🧪 Testing and Evaluation

### Unit Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/unit/test_conversational.py
python -m pytest tests/unit/test_speech.py
```

### Performance Benchmarking

```python
# Run performance benchmark
python demo_conversational_speech.py --benchmark

# Custom evaluation
trainer.evaluate_conversation_quality(test_data)
speech_trainer.evaluate_speech_quality(test_data)
```

## 🔍 Troubleshooting

### Common Issues

#### Memory Issues
```python
# Reduce batch size
config.batch_size = 2

# Enable gradient checkpointing
config.gradient_checkpointing = True

# Use mixed precision
config.mixed_precision = True
```

#### Training Instability
```python
# Reduce learning rate
config.learning_rate = 5e-5

# Add gradient clipping
config.max_grad_norm = 0.5

# Increase warmup steps
config.warmup_steps = 200
```

#### GitHub Upload Issues
```python
# Check token permissions
github_manager = GitHubModelManager(token, repo)
if not github_manager.is_available():
    print("GitHub integration not available")

# Verify repository access
repo_info = github_manager.get_repository_info()
print(f"Repository: {repo_info['full_name']}")
```

### Performance Optimization

#### GPU Acceleration
```python
# Enable CUDA if available
config.device = "cuda" if torch.cuda.is_available() else "cpu"

# Use multiple GPUs
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

#### Memory Optimization
```python
# Use gradient accumulation
config.gradient_accumulation_steps = 8

# Enable mixed precision training
config.mixed_precision = True

# Use smaller model for testing
config.d_model = 256
config.n_layers = 6
```

## 📚 Best Practices

### Data Quality
- Use high-quality conversation data
- Ensure diverse conversation types
- Include proper context and personality annotations
- Validate audio quality for speech training

### Training Strategy
- Start with smaller models for testing
- Use validation data for early stopping
- Monitor training metrics closely
- Save checkpoints regularly

### Model Deployment
- Test models thoroughly before deployment
- Use appropriate hardware for inference
- Monitor performance in production
- Keep models updated with new data

## 🎯 Next Steps

1. **Data Collection**: Gather more diverse conversation and speech data
2. **Model Fine-tuning**: Fine-tune on specific domains or use cases
3. **Performance Optimization**: Optimize for specific hardware or latency requirements
4. **Integration**: Integrate with existing applications or services
5. **Monitoring**: Set up production monitoring and alerting

## 📞 Support

- **Documentation**: Check the main README.md for general usage
- **Issues**: Report bugs and issues on GitHub
- **Discussions**: Join community discussions for help and ideas
- **Contributing**: Contribute improvements and new features

---

**Happy Training!** 🚀 Train your Tantra model to be the best conversational AI it can be!