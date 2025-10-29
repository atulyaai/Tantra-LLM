# Tantra-LLM Architecture Documentation

## Overview

Tantra-LLM is a proprietary multimodal cognitive architecture that combines advanced language models with vision and audio processing capabilities. The system is built around the SpikingBrain-7B reasoning core and integrates Long-VITA vision encoding with comprehensive memory management and personality-driven response generation.

## Core Components

### 1. SpikingBrain-7B (Reasoning Core)
- **Location**: `core/models/spikingbrain_model.py`
- **Purpose**: The main reasoning engine with spiking neural network dynamics
- **Key Features**:
  - Custom transformer architecture with spiking attention mechanisms
  - Multi-head attention with membrane potential dynamics
  - Feed-forward networks with spiking dynamics
  - Configurable architecture parameters
  - Support for both text and multimodal inputs

### 2. Long-VITA Vision Encoder
- **Location**: `encoders/vision.py`
- **Purpose**: Processes visual inputs and converts them to embeddings
- **Key Features**:
  - Local Long-VITA model integration with fallback CNN encoder
  - Support for multiple input formats (PIL Image, numpy arrays, tensors)
  - Automatic dimension projection to match model requirements
  - Remote API support with local fallback
  - Robust error handling and recovery

### 3. Multimodal Fusion System
- **Location**: `core/fusion/unified_fusion.py`
- **Purpose**: Combines text, vision, and audio embeddings
- **Key Features**:
  - Unified fusion approach with cross-modal attention
  - Modality projectors for dimension alignment
  - Multiple fusion strategies (unified, token-stream)
  - Comprehensive error handling and fallback mechanisms

### 4. Memory Management System
- **Location**: `core/memory/advanced_memory.py`
- **Purpose**: Manages episodic, semantic, and working memory
- **Key Features**:
  - Episodic memory with vector search and importance weighting
  - Semantic memory for facts and concepts
  - Working memory for current context
  - Memory consolidation and decay mechanisms
  - Improved recall scoring with recency and importance

### 5. Decision Engine
- **Location**: `core/control/decision_engine.py`
- **Purpose**: Makes intelligent decisions about processing and response generation
- **Key Features**:
  - Input complexity analysis
  - Safety checking and content filtering
  - Dynamic recall depth determination
  - Storage importance assessment
  - Processing priority assignment

### 6. Response Generator
- **Location**: `core/control/response_generator.py`
- **Purpose**: Generates responses using the SpikingBrain model
- **Key Features**:
  - Multimodal response generation
  - Personality-driven parameterization
  - Comprehensive error handling and fallback
  - Performance optimization with caching and monitoring
  - Safety-aware response generation

### 7. Brain Orchestrator
- **Location**: `core/control/brain_orchestrator.py`
- **Purpose**: Main control loop coordinating all components
- **Key Features**:
  - Perception → Decision → Response → Reflection → Memory pipeline
  - Context-aware memory retrieval
  - Error recovery and fallback mechanisms
  - Performance monitoring and optimization

## Data Flow

```
Input (Text/Image/Audio)
    ↓
Perception Module
    ↓
Decision Engine
    ↓
Memory Retrieval
    ↓
Response Generator
    ↓
SpikingBrain Model
    ↓
Output (Text Response)
```

### Detailed Flow:

1. **Input Processing**: Raw inputs (text, image, audio) are processed by the perception module
2. **Decision Making**: The decision engine analyzes complexity, safety, and determines processing parameters
3. **Memory Retrieval**: Relevant memories are retrieved based on the input and decision parameters
4. **Context Building**: Input and retrieved memories are combined into enhanced context
5. **Response Generation**: The SpikingBrain model generates responses using the enhanced context
6. **Memory Storage**: The interaction is stored in memory for future reference

## Configuration

### Model Configuration
- **Location**: `config/model_config.py`
- **Key Parameters**:
  - `model_dim`: 4096 (SpikingBrain hidden size)
  - `vision.embed_dim`: 4096 (aligned with model_dim)
  - `audio.embed_dim`: 4096 (aligned with model_dim)
  - `memory.embedding_dim`: 4096 (aligned with model_dim)

### Personality Configuration
- **Location**: `config/personality_config.json`
- **Modes**: DirectAssertive, MentorBuilder, CriticalChallenger, CreativeExplorer
- **Parameters**: Temperature, top_p, max_tokens, prompt_prefix

## Error Handling

### Error Categories
- **MODEL_LOADING**: Issues with model loading and initialization
- **MEMORY_OPERATION**: Memory storage and retrieval errors
- **FUSION_PROCESSING**: Multimodal fusion errors
- **GENERATION**: Response generation errors
- **VISION_ENCODING**: Vision processing errors
- **AUDIO_ENCODING**: Audio processing errors
- **NETWORK**: Network and API errors
- **CONFIGURATION**: Configuration validation errors

### Recovery Strategies
- **RETRY**: Retry failed operations with exponential backoff
- **FALLBACK**: Use alternative implementations or simplified processing
- **SKIP**: Skip non-critical operations
- **ABORT**: Stop processing for critical errors
- **LOG_AND_CONTINUE**: Log errors and continue processing

## Performance Optimization

### Caching
- Function result caching with TTL
- Memory retrieval caching
- Model output caching

### Monitoring
- Performance metrics tracking
- Success rate monitoring
- Duration analysis
- Memory usage tracking

### Async Operations
- Thread pool execution for CPU-bound tasks
- Process pool execution for parallel processing
- Batch processing for multiple items

## Testing

### Test Coverage
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **Error Scenario Tests**: Error handling and recovery testing
- **Performance Tests**: Performance optimization testing

### Test Files
- `tests/test_e2e_brain.py`: End-to-end brain testing
- `tests/test_fusion_stream.py`: Fusion system testing
- `tests/test_error_scenarios.py`: Error handling testing
- `tests/test_inference.py`: Model inference testing

## Deployment

### Requirements
- Python 3.8+
- PyTorch 1.12+
- Transformers 4.20+
- SentenceTransformers
- PIL (Pillow)
- NumPy

### Installation
```bash
pip install -r requirements.txt
```

### Usage
```python
from core.control.brain_orchestrator import BrainOrchestrator

# Initialize the brain
brain = BrainOrchestrator()

# Process input
response = brain.step(text="Hello, how are you?")
print(response)
```

## Architecture Principles

### 1. Modularity
- Each component is self-contained and can be tested independently
- Clear interfaces between components
- Easy to replace or upgrade individual components

### 2. Robustness
- Comprehensive error handling at every level
- Graceful degradation when components fail
- Fallback mechanisms for critical operations

### 3. Performance
- Caching for expensive operations
- Async processing where appropriate
- Performance monitoring and optimization

### 4. Extensibility
- Plugin architecture for new modalities
- Configurable parameters and behavior
- Easy to add new personality modes

### 5. Safety
- Content filtering and safety checks
- Input validation and sanitization
- Error logging and monitoring

## Future Enhancements

### Planned Features
- Real-time streaming responses
- Advanced memory consolidation algorithms
- Multi-agent collaboration
- Enhanced multimodal understanding
- Improved safety mechanisms

### Performance Improvements
- Model quantization and optimization
- Advanced caching strategies
- Distributed processing
- GPU acceleration

## Troubleshooting

### Common Issues
1. **Model Loading Failures**: Check model paths and dependencies
2. **Memory Issues**: Adjust memory limits and consolidation frequency
3. **Performance Issues**: Enable caching and monitoring
4. **Configuration Errors**: Use the config validator

### Debug Mode
Enable debug logging to see detailed information about processing:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add comprehensive docstrings
- Include error handling

### Testing
- Write tests for new features
- Ensure all tests pass
- Add error scenario tests
- Update documentation

### Documentation
- Update architecture docs for new features
- Add inline comments for complex logic
- Update README for new capabilities
- Maintain API documentation