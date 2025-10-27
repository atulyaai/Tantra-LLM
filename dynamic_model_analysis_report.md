
# Dynamic Model Analysis Report

## Analysis Overview
- **Timestamp**: 2025-10-27T23:10:03.920054
- **Model Path**: Model/weights/Tantra_v1.0.pt
- **Status**: ✅ SUCCESS

## Summary

- **Total Parameters**: 5,002,432
- **File Size**: 19.10 MB
- **Memory Size**: 19.08 MB
- **Compression Ratio**: 1.00x
- **Model Type**: transformer
- **Validation Status**: ✅ Valid

## Model Analysis
- **Architecture**: 128D, 4 layers, 8 heads
- **Vocabulary**: 10,000 tokens
- **Max Sequence Length**: 512
- **Total Parameters**: 5,002,432
- **Memory Usage**: 19.08 MB
- **Validation**: ✅ Valid

## Size Analysis
- **File Size**: 19.10 MB (20,029,300 bytes)
- **Memory Size**: 19.08 MB (20,009,728 bytes)
- **Compression Ratio**: 1.00x
- **Compression Type**: expansion
- **Efficiency Rating**: Very Poor
- **Size Category**: Small

## Vocabulary Analysis
- **Vocabulary Size**: 50,000 tokens
- **Source**: Model/tokenizer_vocab.json
- **Status**: ✅ Valid

### Token Types
- **Special**: 11,805 (23.6%)
- **Punctuation**: 8 (0.0%)
- **Numbers**: 58 (0.1%)
- **Letters**: 20,144 (40.3%)
- **Mixed**: 6,260 (12.5%)
- **Unicode**: 0 (0.0%)
- **Whitespace**: 0 (0.0%)
- **Other**: 11,725 (23.4%)

## Configuration Update
- **Updated Files**: 0
- **Created Files**: 0
- **Backup Created**: ✅ Yes

## Validation Results
- **Overall Status**: WARNING
- **Total Checks**: 22
- **Passed**: 21
- **Failed**: 1
- **Warnings**: 1
- **Errors**: 0

### Recommendations
1. Resolve vocabulary_consistency: Multiple different vocabulary sizes found
2. Review warnings and consider addressing them

---
*Report generated at 2025-10-27T23:10:04.573086*