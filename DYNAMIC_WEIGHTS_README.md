# Dynamic Weight Management System

This document describes the new dynamic weight management system implemented for Tantra LLM, which replaces hardcoded weight paths with a flexible, version-controlled system.

## Overview

The dynamic weight management system provides:
- **Dynamic weight loading**: No more hardcoded paths
- **Version control**: Track different weight versions
- **Model type support**: Support for multiple model architectures
- **Automatic validation**: Ensure weights are loadable
- **Cleanup utilities**: Remove old weights automatically
- **CLI management**: Easy command-line interface

## Key Components

### 1. WeightManager (`Training/weight_manager.py`)
- Manages weight files, versions, and metadata
- Handles weight loading, saving, and validation
- Provides backup and cleanup functionality
- Supports multiple model types

### 2. ConfigManager (`Training/config_manager.py`)
- Manages model configurations dynamically
- Supports different model architectures
- Handles path management
- Provides training and serving configurations

### 3. Model Manager CLI (`Training/model_manager.py`)
- Command-line interface for weight management
- List models, weights, and configurations
- Set active weights, create backups
- Validate and cleanup weights

## Quick Start

### 1. Initialize the System
```bash
python3 setup_dynamic_weights.py
```

### 2. List Available Models
```bash
python3 Training/model_manager.py list-models
```

### 3. List Weights
```bash
python3 Training/model_manager.py list-weights
```

### 4. View Model Configuration
```bash
python3 Training/model_manager.py config mamba
```

## Supported Model Types

- **mamba**: Standard Mamba architecture
- **mamba_multimodal**: Multi-modal Mamba with MoE
- **ocr_native**: OCR-optimized Mamba model

## Usage Examples

### Loading Weights Programmatically
```python
from Training.weight_manager import load_model_weights, save_model_weights
from Training.config_manager import get_model_config

# Load weights for a specific model type
state_dict = load_model_weights("mamba")

# Save new weights
saved_path = save_model_weights(state_dict, "mamba", version="v1.0")

# Get model configuration
config = get_model_config("mamba")
```

### Using Runtime Classes
```python
from Training.mamba_runtime import MambaRuntime
from Training.model_runtime import TextRuntime

# Initialize with dynamic weights
mamba_runtime = MambaRuntime(model_type="mamba")
text_runtime = TextRuntime(model_type="mamba")
```

### CLI Management
```bash
# List all weights
python3 Training/model_manager.py list-weights

# Set active weight
python3 Training/model_manager.py set-active mamba_v1.0

# Create backup
python3 Training/model_manager.py backup mamba_v1.0

# Cleanup old weights (keep 5 most recent)
python3 Training/model_manager.py cleanup mamba --keep 5

# Validate weights
python3 Training/model_manager.py validate mamba
```

## File Structure

```
Model/
├── weights/                    # Dynamic weight storage
│   ├── mamba_*.safetensors
│   ├── mamba_multimodal_*.safetensors
│   └── ocr_native_*.safetensors
├── backups/                    # Weight backups
├── checkpoints/               # Training checkpoints
└── weight_config.json         # Weight registry

Config/
├── mamba.yaml                 # Mamba configuration
├── mamba_multimodal.yaml      # Multi-modal configuration
└── ocr_native.yaml           # OCR configuration
```

## Migration from Hardcoded System

The system automatically migrates existing weight files:
- `Model/tantra_weights.safetensors` → `Model/weights/mamba_migrated.safetensors`
- `Model/tantra_multimodal_weights.safetensors` → `Model/weights/mamba_multimodal_migrated.safetensors`

## Configuration

### Model Configuration
Each model type has its own configuration file in `Config/`:
- Model architecture parameters
- Training settings
- Path configurations
- Extra parameters for specific model types

### Weight Registry
The `Model/weight_config.json` file tracks:
- All registered weights
- Active weight for each model type
- Weight metadata (size, checksum, creation date)
- Version information

## Benefits

1. **No More Hardcoded Paths**: All weight paths are managed dynamically
2. **Version Control**: Track and manage different weight versions
3. **Easy Model Switching**: Change active weights without code changes
4. **Automatic Validation**: Ensure weights are loadable before use
5. **Cleanup Utilities**: Remove old weights automatically
6. **Backup System**: Create backups before making changes
7. **CLI Management**: Easy command-line interface for all operations
8. **Multi-Model Support**: Support for different model architectures

## Testing

Run the test suite to verify everything works:
```bash
python3 test_dynamic_weights.py
```

## Troubleshooting

### Common Issues

1. **Weights not found**: Run `python3 setup_dynamic_weights.py` to initialize
2. **Invalid weights**: Use `python3 Training/model_manager.py validate <model_type>`
3. **Size mismatch**: Ensure model configuration matches weight dimensions
4. **Permission errors**: Check file permissions in `Model/weights/` directory

### Debug Commands

```bash
# Check weight registry
cat Model/weight_config.json

# List all files in weights directory
ls -la Model/weights/

# Validate specific model
python3 Training/model_manager.py validate mamba --version sample
```

## Future Enhancements

- Weight compression and quantization
- Remote weight storage support
- Weight sharing between model types
- Automatic weight optimization
- Integration with model hub services

## API Reference

### WeightManager Methods
- `load_weights(model_type, version=None)`: Load weights for a model type
- `save_weights(state_dict, model_type, name, version, is_active)`: Save weights
- `get_weight_path(model_type, version=None)`: Get weight file path
- `set_active_weight(name)`: Set a weight as active
- `backup_weight(name)`: Create a backup of a weight
- `cleanup_old_weights(model_type, keep_count)`: Clean up old weights
- `validate_weights(model_type, version=None)`: Validate weights

### ConfigManager Methods
- `get_config(model_type)`: Get model configuration
- `get_paths(model_type)`: Get model paths
- `get_training_config(model_type)`: Get training configuration
- `get_serve_config(model_type)`: Get serving configuration
- `create_model_config(model_type, **kwargs)`: Create new model config

This dynamic weight management system makes Tantra LLM more flexible, maintainable, and easier to use with different model configurations and weight versions.