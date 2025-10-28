"""Phase 2 tests: Dynamic context window management."""

import pytest
from tantra_llm.core.models.dynamic_context import DynamicContextManager


def test_dynamic_context_init():
    """Test DynamicContextManager initialization."""
    dc = DynamicContextManager(max_short=4096, max_long=32768)
    assert dc.max_short == 4096
    assert dc.max_long == 32768
    assert dc.recurrent_state is None


def test_select_window_fast_path():
    """Test selecting short window for urgent simple queries."""
    dc = DynamicContextManager()
    task_meta = {"urgency": 0.9, "complexity": 0.2, "type": "query"}
    window = dc.select_window(task_meta)
    assert window == dc.max_short


def test_select_window_deep_path():
    """Test selecting long window for complex reasoning."""
    dc = DynamicContextManager()
    task_meta = {"urgency": 0.3, "complexity": 0.8, "type": "reasoning"}
    window = dc.select_window(task_meta)
    assert window == dc.max_long


def test_trim_tokens():
    """Test token sequence trimming."""
    dc = DynamicContextManager()
    tokens = list(range(1000))
    trimmed = dc.trim(tokens, target_len=500)
    assert len(trimmed) == 500
    assert trimmed == tokens[:500]


def test_recurrent_state():
    """Test recurrent state management."""
    dc = DynamicContextManager()
    assert dc.get_recurrent_state() is None
    
    test_state = {"key": "value"}
    dc.update_recurrent_state({"key": "value"})
    
    state = dc.get_recurrent_state()
    assert state is not None

