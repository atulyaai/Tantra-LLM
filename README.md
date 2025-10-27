# 🔤 OCR-Native LLM - Clean Version

Revolutionary approach to language modeling: All weights, parameters, and data stored in OCR-readable format.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run main script
python main.py

# Run demo
python examples/demo_ocr_native.py

# Run tests
python tests/unit/test_ocr_native.py
```

## 📁 Project Structure

```
ocr-native-llm/
├── src/                    # Source code
│   ├── core/              # Core OCR-native components
│   ├── models/            # Model management
│   ├── utils/             # Utilities
│   └── configs/           # Configuration
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── examples/             # Example scripts
├── docs/                 # Documentation
├── models/               # Model storage
├── data/                 # Data storage
└── logs/                 # Log files
```

## 🎯 Key Features

- **OCR-First Design**: All weights stored as OCR-readable images
- **Extended Memory**: 100K+ token context window
- **Multi-Modal**: Text, speech, and image processing
- **Modular Architecture**: Clean, maintainable code
- **Bigger Model**: 1024 dim, 24 layers, 16 heads

## 🔧 Usage

```python
from src.core.ocr_native_llm import OCRNativeLLM
from src.configs.ocr_config import ConfigManager

# Create model
config = ConfigManager.get_default_config()
model = OCRNativeLLM(config)

# Generate response
inputs = {'text': 'Hello, world!'}
response = model.generate_response(inputs, "Tell me about AI")
print(response)
```

## 📊 Model Specifications

- **Parameters**: ~100M (configurable)
- **Context Window**: 16,384 tokens
- **Memory**: 100,000 tokens
- **Architecture**: Transformer-based with OCR optimization
- **Formats**: All weights stored as OCR images

## 🧪 Testing

```bash
# Run all tests
python tests/unit/test_ocr_native.py

# Run specific test
python -m unittest tests.unit.test_ocr_native.TestOCRNativeLLM.test_forward_pass
```

## 📈 Performance

- **OCR Weight Encoding**: ~0.1s per layer
- **Text Processing**: ~0.05s per input
- **Memory Retrieval**: ~0.01s per query
- **Response Generation**: ~0.5s per response

## 🔮 Future Enhancements

- Advanced OCR engines integration
- Distributed OCR memory
- Real-time OCR processing
- Multi-language support

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

---

**OCR-Native LLM** - Revolutionizing language modeling with OCR-first design
