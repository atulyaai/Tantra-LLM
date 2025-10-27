# 🔧 OCR-Native LLM Troubleshooting Guide

This guide helps you diagnose and fix common issues with the OCR-Native LLM system.

## 🚨 Common Issues and Solutions

### 1. Memory Issues

#### Problem: Out of Memory (OOM) Error
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Use the small configuration: `ConfigManager.get_small_config()`
- Reduce batch size in configuration
- Enable gradient checkpointing: `config.gradient_checkpointing = True`
- Use CPU instead of GPU: `config.use_cuda = False`

**Example:**
```python
from src.configs.ocr_config import ConfigManager

# Use memory-optimized config
config = ConfigManager.get_small_config()
config.batch_size = 1
config.use_cuda = False
```

#### Problem: High Memory Usage
- **Current**: 2GB+ for small config
- **Optimized**: ~260MB for small config

**Solutions:**
- Use the optimized small configuration
- Process only one input at a time
- Clear cache regularly: `torch.cuda.empty_cache()`

### 2. Import Errors

#### Problem: Module Not Found
```
ModuleNotFoundError: No module named 'torch'
```

**Solutions:**
```bash
# Install core dependencies
pip3 install torch torchvision numpy pillow opencv-python

# Install all dependencies
pip3 install -r requirements.txt
```

#### Problem: Import Path Issues
```
ImportError: No module named 'src.core.ocr_native_llm'
```

**Solutions:**
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
```

### 3. Model Loading Issues

#### Problem: Model Registry KeyError
```
KeyError: 'versions'
```

**Solutions:**
- This has been fixed in the latest version
- If you encounter this, update your model manager code
- Use the latest `get_model_info()` method

#### Problem: Configuration Validation Error
```
ValidationError: Missing required config key: d_model
```

**Solutions:**
- Ensure all required configuration keys are present
- Use predefined configurations: `ConfigManager.get_small_config()`
- Validate your custom configuration

### 4. OCR Processing Issues

#### Problem: OCR Image Processing Fails
```
OCRProcessingError: Failed to process image to OCR format
```

**Solutions:**
- Check if PIL/Pillow is installed: `pip3 install pillow`
- Ensure image is in supported format (PNG, JPG, etc.)
- Verify image file is not corrupted

#### Problem: Font Loading Error
```
OSError: cannot open resource
```

**Solutions:**
- Install system fonts: `sudo apt-get install fonts-dejavu-core`
- Or use default font fallback (already implemented)

### 5. Performance Issues

#### Problem: Slow Model Inference
- **Cause**: Large model size or inefficient processing
- **Solutions**:
  - Use smaller configuration
  - Enable performance monitoring
  - Check system resources

#### Problem: High CPU Usage
- **Solutions**:
  - Use GPU acceleration if available
  - Reduce model complexity
  - Optimize input processing

## 🔍 Debugging Tools

### 1. Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. Performance Monitoring
```bash
python3 performance_monitor.py
```

### 3. System Cleanup
```bash
python3 cleanup_system.py
```

### 4. Memory Profiling
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
```

## 📊 Performance Benchmarks

### Small Configuration (Optimized)
- **Parameters**: 68,809,736
- **Model Size**: 262.49 MB
- **Memory Usage**: ~260 MB
- **Forward Pass**: ~0.1s
- **Throughput**: ~10 ops/s

### Default Configuration
- **Parameters**: 545,229,840
- **Model Size**: 2079.89 MB
- **Memory Usage**: ~2GB
- **Forward Pass**: ~0.5s
- **Throughput**: ~2 ops/s

## 🛠️ System Requirements

### Minimum Requirements
- **RAM**: 4GB
- **Storage**: 5GB
- **Python**: 3.8+
- **OS**: Linux, macOS, Windows

### Recommended Requirements
- **RAM**: 8GB+
- **Storage**: 10GB+
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **Python**: 3.9+

## 📝 Log Files

### Log Locations
- **Main Log**: `logs/ocr_native.log`
- **Error Log**: `logs/errors.log`
- **Performance**: `logs/system_metrics.json`
- **Benchmark**: `logs/benchmark_results.json`

### Log Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General information
- **WARNING**: Warning messages
- **ERROR**: Error messages with stack traces

## 🔧 Configuration Examples

### Memory-Optimized Configuration
```python
config = OCRNativeConfig(
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_ff=512,
    vocab_size=5000,
    max_seq_length=1024,
    ocr_image_width=512,
    ocr_image_height=512,
    batch_size=1,
    use_cuda=False
)
```

### Production Configuration
```python
config = OCRNativeConfig(
    d_model=1024,
    n_layers=24,
    n_heads=16,
    d_ff=4096,
    vocab_size=100000,
    max_seq_length=16384,
    batch_size=4,
    use_cuda=True,
    gradient_checkpointing=True
)
```

## 🚀 Quick Fixes

### Reset System
```bash
# Clean all temporary files
python3 cleanup_system.py

# Reinstall dependencies
pip3 install -r requirements.txt

# Run tests
python3 tests/unit/test_ocr_native.py
```

### Memory Reset
```python
import torch
import gc

# Clear PyTorch cache
torch.cuda.empty_cache()
gc.collect()
```

### Configuration Reset
```python
from src.configs.ocr_config import ConfigManager

# Use safe default
config = ConfigManager.get_small_config()
```

## 📞 Getting Help

1. **Check Logs**: Always check log files first
2. **Run Tests**: Ensure all tests pass
3. **Check Dependencies**: Verify all packages are installed
4. **System Resources**: Monitor CPU, memory, and disk usage
5. **Configuration**: Use predefined configurations

## 🔄 Updates and Maintenance

### Regular Maintenance
- Run cleanup script weekly
- Monitor log files
- Update dependencies monthly
- Check system resources

### Performance Optimization
- Use appropriate configuration for your use case
- Monitor performance metrics
- Optimize based on benchmarks
- Consider hardware upgrades for large models

---

**Need more help?** Check the main README.md or create an issue with detailed error logs.