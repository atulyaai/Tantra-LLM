# Dynamic Model Analysis System

A comprehensive system for real-time model parameter analysis, vocabulary extraction, size calculation, and configuration management.

## 🚀 Features

### Core Analysis Tools
- **Dynamic Model Analyzer**: Calculates real parameters from actual model files
- **Vocabulary Analyzer**: Extracts and analyzes token vocabulary details
- **Model Size Calculator**: Cross-checks file size vs memory size with compression analysis
- **Configuration Updater**: Syncs config files with actual model data
- **Validation System**: Ensures consistency across all model files

### Key Capabilities
- ✅ **Real Parameter Counting**: Loads actual model files and counts parameters
- ✅ **Vocabulary Analysis**: Extracts tokens from tokenizer files with categorization
- ✅ **Size Validation**: Cross-checks file size vs memory usage
- ✅ **Configuration Sync**: Updates config files with real model data
- ✅ **Consistency Validation**: Ensures all files are consistent
- ✅ **Comprehensive Reporting**: Generates detailed analysis reports
- ✅ **CLI Interface**: Easy command-line access to all tools
- ✅ **Automated Backup**: Creates backups before updating configurations

## 📁 File Structure

```
src/utils/
├── dynamic_model_analyzer.py      # Core model analysis
├── vocabulary_analyzer.py         # Vocabulary extraction and analysis
├── model_size_calculator.py       # Size calculation and validation
├── configuration_updater.py       # Config file synchronization
├── validation_system.py           # Cross-file consistency validation
├── dynamic_model_orchestrator.py  # Main orchestrator
└── dynamic_model_cli.py           # Command-line interface

run_dynamic_model_analysis.py      # Main integration script
```

## 🛠️ Usage

### Quick Start
```bash
# Run complete analysis
python3 run_dynamic_model_analysis.py

# Or use CLI
python3 -m src.utils.dynamic_model_cli analyze --complete
```

### CLI Commands

#### Analyze Models
```bash
# Complete analysis (all tools)
python3 -m src.utils.dynamic_model_cli analyze --complete

# Quick analysis (model + size only)
python3 -m src.utils.dynamic_model_cli analyze --quick

# Analyze specific model
python3 -m src.utils.dynamic_model_cli analyze --model Model/weights/Tantra_v1.0.pt --report
```

#### Update Information
```bash
# Update model information files
python3 -m src.utils.dynamic_model_cli update

# Force update
python3 -m src.utils.dynamic_model_cli update --force
```

#### Validate Models
```bash
# Run validation
python3 -m src.utils.dynamic_model_cli validate

# Save validation results
python3 -m src.utils.dynamic_model_cli validate --output validation_results.json
```

#### Generate Reports
```bash
# Generate report from existing results
python3 -m src.utils.dynamic_model_cli report

# Generate report from specific file
python3 -m src.utils.dynamic_model_cli report --input results.json --output report.md
```

#### Get Information
```bash
# Show model information
python3 -m src.utils.dynamic_model_cli info

# Show analysis status
python3 -m src.utils.dynamic_model_cli status
```

## 📊 Analysis Results

### Model Analysis
- **Real Parameter Count**: 5,002,432 parameters
- **Architecture**: 128D, 4 layers, 8 attention heads
- **Vocabulary Size**: 10,000 tokens (from model) / 50,000 tokens (from tokenizer)
- **Memory Usage**: 19.08 MB
- **Model Type**: Transformer

### Size Analysis
- **File Size**: 19.10 MB (20,029,300 bytes)
- **Memory Size**: 19.08 MB (20,009,728 bytes)
- **Compression Ratio**: 1.00x (no compression)
- **Efficiency Rating**: Very Poor (due to no compression)
- **Size Category**: Small

### Vocabulary Analysis
- **Total Tokens**: 50,000
- **Token Distribution**:
  - Letters: 40.3% (20,144 tokens)
  - Special: 23.6% (11,805 tokens)
  - Other: 23.4% (11,725 tokens)
  - Mixed: 12.5% (6,260 tokens)
  - Numbers: 0.1% (58 tokens)
  - Punctuation: 0.0% (8 tokens)

### Validation Results
- **Overall Status**: WARNING
- **Checks Passed**: 21/22
- **Warnings**: 1 (vocabulary size inconsistency)
- **Errors**: 0

## 🔧 Configuration Files

The system automatically creates and updates:

- `Model/weights/weight_config.json` - Weight file information
- `Model/weights/Tantra_real_config.json` - Real model configuration
- `Model/weights/training_config.json` - Training parameters
- `Model/consolidated_config.json` - Consolidated analysis results
- `Model/config_backup_YYYYMMDD_HHMMSS/` - Backup of original configs

## 📈 Key Benefits

1. **Accuracy**: Real parameter counting from actual model files
2. **Consistency**: Cross-validates all model-related files
3. **Automation**: Automatically updates configurations
4. **Validation**: Ensures data integrity across files
5. **Reporting**: Comprehensive analysis reports
6. **Backup**: Safe configuration updates with backups
7. **CLI Access**: Easy command-line interface
8. **Extensibility**: Modular design for easy extension

## 🚨 Important Notes

- The system creates backups before updating configurations
- All analysis is based on actual model files, not theoretical calculations
- Validation warnings should be reviewed and addressed
- The system handles various model formats and tokenizer types
- Configuration files are automatically synchronized with real model data

## 🔍 Troubleshooting

### Common Issues
1. **Model file not found**: Ensure model files exist in `Model/weights/`
2. **Permission errors**: Check file permissions for model and config files
3. **Memory issues**: Large models may require more RAM
4. **Validation warnings**: Review and address consistency issues

### Logs
- Analysis logs: `dynamic_model_analysis.log`
- Check logs for detailed error information

## 📝 Example Output

```
🚀 Dynamic Model Analysis System
==================================================
🔧 Initializing dynamic model orchestrator...
✅ Model found: Model/weights/Tantra_v1.0.pt

🔍 Running complete dynamic model analysis...
✅ Analysis completed successfully!

📊 Analysis Summary:
  🧠 Total Parameters: 5,002,432
  📁 File Size: 19.10 MB
  💾 Memory Size: 19.08 MB
  📦 Compression Ratio: 1.00x
  🤖 Model Type: transformer
  ✅ Validation: Valid

🎉 Dynamic model analysis completed successfully!
```

This dynamic model system provides comprehensive, real-time analysis of your model files with automatic configuration management and validation.