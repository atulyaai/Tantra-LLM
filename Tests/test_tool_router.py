"""
Tests/test_tool_router.py — Unit tests for safe tool execution router.
"""
import pytest
from Tantra.tool_router import execute_tool_call, parse_and_execute_tool_calls, safe_eval_math

def test_calculator_basic_and_complex():
    assert safe_eval_math("2 + 2") == "4"
    assert safe_eval_math("9482 * 387") == "3669534"
    assert safe_eval_math("(45 * 89) + (1200 / 25)") == "4053.0"
    assert safe_eval_math("2 ** 10") == "1024"

def test_python_executor():
    code = "print(sum(range(10)))"
    res = execute_tool_call("python_executor", {"code": code})
    assert res == "45"

def test_file_reader():
    res = execute_tool_call("file_reader", {"filepath": "pyproject.toml"})
    assert "[project]" in res or "[build-system]" in res

def test_parse_and_execute():
    text = '<tool_call>{"name": "calculator", "arguments": {"expression": "100 * 50"}}</tool_call>'
    updated, did_exec = parse_and_execute_tool_calls(text)
    assert did_exec is True
    assert "<tool_result>" in updated
    assert "5000" in updated


def test_web_search_and_doc_retriever(tmp_path):
    from Tantra.tool_router import search_web, retrieve_local_documents

    # 1. Web search safe execution
    res_web = search_web("Quantum computing")
    assert isinstance(res_web, str)
    assert len(res_web) > 0

    # 2. Document retrieval from temporary directory
    doc_file = tmp_path / "architecture_notes.md"
    doc_file.write_text("Tantra is a local AI model with BitNet 1.58-bit ternary quantization.", encoding="utf-8")

    res_doc = retrieve_local_documents("BitNet quantization", doc_dir=str(tmp_path))
    assert "architecture_notes.md" in res_doc
    assert "BitNet" in res_doc

