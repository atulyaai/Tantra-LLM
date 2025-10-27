# 🔤 Tantra v1.0 - OCR-Native Conversational Speech LLM

Revolutionary approach to language modeling: All weights, parameters, and data stored in OCR-readable format with advanced conversational and speech capabilities.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run interactive conversational demo
python demo_conversational_speech.py

# Run basic model
python main.py

# Train your own model
python run_training.py --github-token YOUR_GITHUB_TOKEN

# Run tests
python tests/unit/test_ocr_native.py
```

## 🎯 Key Features

- **OCR-First Design**: All weights stored as OCR-readable images
- **Conversational AI**: Advanced dialogue with personality traits
- **Speech Processing**: Text-to-speech and speech-to-text
- **Multi-Modal**: Text, speech, and image processing
- **GitHub Integration**: Automatic model saving and versioning
- **Training Pipeline**: Complete conversational and speech training
- **Extended Memory**: 100K+ token context window
- **Modular Architecture**: Clean, maintainable code

## 💬 Conversational Capabilities

### Personality Traits
- **Helpful**: Supportive and informative responses
- **Knowledgeable**: Technical expertise and detailed explanations
- **Empathetic**: Emotional understanding and support
- **Creative**: Imaginative and artistic responses
- **Analytical**: Logical problem-solving approach

### Conversation Types
- **General Chat**: Casual conversation and small talk
- **Technical Support**: Problem-solving and troubleshooting
- **Creative Writing**: Storytelling and creative assistance
- **Problem Solving**: Analytical thinking and solutions
- **Emotional Support**: Empathetic listening and guidance

## 🎤 Speech Features

### Text-to-Speech (TTS)
- High-quality audio generation
- Multiple voice styles (neutral, excited, calm)
- Real-time synthesis
- Configurable speech parameters

### Speech-to-Text (STT)
- Accurate speech recognition
- Real-time transcription
- Context-aware processing
- Multi-language support (planned)

## 📁 Project Structure

```
tantra/
├── src/                           # Source code
│   ├── core/                     # Core OCR-native components
│   ├── training/                 # Training system
│   │   ├── conversational_trainer.py
│   │   ├── speech_trainer.py
│   │   ├── data_loader.py
│   │   ├── training_config.py
│   │   └── github_integration.py
│   ├── models/                   # Model management
│   ├── utils/                    # Utilities
│   └── configs/                  # Configuration
├── tests/                        # Test suite
├── examples/                     # Example scripts
├── docs/                         # Documentation
├── Model/                        # Model storage
├── data/                         # Data storage
├── logs/                         # Log files
├── train_conversational_speech.py # Main training script
├── run_training.py               # Complete training pipeline
└── demo_conversational_speech.py # Interactive demo
```

## 🔧 Usage Examples

### Basic Conversational AI

```python
from src.core.tantra_llm import TantraLLM, TantraConfig
from src.training.conversational_trainer import ConversationalTrainer

# Create model
config = TantraConfig()
model = TantraLLM(config)

# Initialize conversational trainer
trainer = ConversationalTrainer(config, model)

# Generate conversational response
response = trainer.generate_response(
    user_message="Hello, how are you?",
    context="greeting",
    personality="helpful"
)
print(response)
```

### Speech Processing

```python
from src.training.speech_trainer import SpeechTrainer

# Initialize speech trainer
speech_trainer = SpeechTrainer(config, model)

# Text to speech
audio = speech_trainer.text_to_speech(
    text="Hello, this is Tantra speaking!",
    voice_style="neutral"
)

# Speech to text
text = speech_trainer.speech_to_text(audio)
print(f"Transcribed: {text}")
```

### Interactive Demo

```bash
# Start interactive conversation
python demo_conversational_speech.py

# Commands:
# text <message>     - Send text message
# speech <file.wav>  - Process speech file
# train             - Start training mode
# save              - Save model to GitHub
# quit              - Exit demo
```

## 🏗️ Training System

### Complete Training Pipeline

```bash
# Train both conversational and speech models
python run_training.py --github-token YOUR_GITHUB_TOKEN

# Train only conversational model
python run_training.py --conversational-only

# Train only speech model  
python run_training.py --speech-only
```

### Training Features

- **Data Loaders**: Conversation and speech dataset management
- **Personality Modeling**: Multi-trait personality system
- **Context Awareness**: Long-term memory and context retention
- **Quality Metrics**: Response relevance and coherence evaluation
- **GitHub Integration**: Automatic model saving and versioning
- **Performance Monitoring**: Real-time training metrics

## 📊 Model Specifications

- **Parameters**: ~50M (configurable)
- **Context Window**: 8,192 tokens (extensible to 16,384)
- **Memory**: 100,000 tokens with OCR optimization
- **Architecture**: Transformer-based with OCR-native processing
- **Modalities**: Text, Speech, Image
- **Formats**: All weights stored as OCR images

## 🎛️ Configuration

### Training Configuration

```python
from src.training.training_config import TrainingConfig

config = TrainingConfig(
    model_name="tantra_conversational_v1.0",
    learning_rate=1e-4,
    batch_size=4,
    num_epochs=10,
    conversation_max_length=2048,
    speech_sample_rate=16000,
    github_repo="your-username/tantra-models",
    github_token="your_github_token"
)
```

### Model Configuration

```python
from src.core.tantra_llm import TantraConfig

config = TantraConfig(
    d_model=512,
    n_layers=12,
    n_heads=8,
    vocab_size=50000,
    max_seq_length=8192
)
```

## 📈 Performance

### Training Performance
- **Training Time**: ~2-4 hours (depending on hardware)
- **Memory Usage**: 8-16GB RAM recommended
- **GPU Support**: CUDA acceleration available
- **Batch Processing**: Configurable batch sizes

### Inference Performance
- **Text Generation**: ~100ms per response
- **Speech Synthesis**: ~500ms per sentence
- **Speech Recognition**: ~200ms per utterance
- **OCR Weight Encoding**: ~0.1s per layer
- **Memory Retrieval**: ~0.01s per query

## 🚀 Getting Started

### 1. Basic Setup

```bash
# Clone repository
git clone https://github.com/your-username/tantra.git
cd tantra

# Install dependencies
pip install -r requirements.txt

# Run demo
python demo_conversational_speech.py
```

### 2. Training Your Model

```bash
# Set up GitHub token (optional)
export GITHUB_TOKEN="your_github_token"

# Run complete training pipeline
python run_training.py --github-token $GITHUB_TOKEN

# Or train specific components
python run_training.py --conversational-only
python run_training.py --speech-only
```

### 3. Using Pre-trained Models

```bash
# Download from GitHub (if available)
python demo_conversational_speech.py --model path/to/model.pt

# Or use default model
python demo_conversational_speech.py
```

## 🧪 Testing

```bash
# Run all tests
python tests/unit/test_ocr_native.py

# Run specific test
python -m unittest tests.unit.test_ocr_native.TestOCRNativeLLM.test_forward_pass

# Run conversational tests
python tests/unit/test_conversational.py

# Run speech tests
python tests/unit/test_speech.py
```

## 🔮 Future Enhancements

- Advanced OCR engines integration
- Distributed OCR memory
- Real-time OCR processing
- Multi-language support
- Voice cloning capabilities
- Advanced emotion recognition
- Multi-modal conversation understanding

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 src/
black src/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Hugging Face for transformer architectures and tokenizers
- OpenAI for inspiration on conversational AI
- The open-source community for continuous support

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/tantra/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/tantra/discussions)
- **Documentation**: [Wiki](https://github.com/your-username/tantra/wiki)

---

**Tantra v1.0** - Revolutionizing AI with OCR-native conversational speech capabilities! 🚀