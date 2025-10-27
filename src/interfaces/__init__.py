"""
Interactive Interfaces for OCR-Native LLM
Conversational, web, and CLI interfaces
"""

from .conversational import *
from .cli_interface import *

__all__ = [
    'OCRNativeConversational',
    'OCRNativeCLI',
    'create_conversational_interface',
    'quick_chat'
]