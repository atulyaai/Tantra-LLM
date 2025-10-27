"""
Advanced OCR-Native LLM Architectures
Multiple model variants for experimentation
"""

from .transformer_variants import *

__all__ = [
    'OCRNativeTransformer',
    'TransformerVariantConfig',
    'OCRNativeTransformer',
    'OCRStandardBlock',
    'OCRMambaBlock',
    'OCRHybridBlock',
    'OCRMemoryEnhancedBlock'
]