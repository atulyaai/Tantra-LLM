"""Unit tests for Tantra Qwen3.5 chat interface."""
import pytest
from unittest.mock import MagicMock, patch

from Tantra.qwen35_chat import build_parser, chat_loop


def test_build_parser():
    parser = build_parser()
    args = parser.parse_args(["--prompt", "Hello", "--max-new-tokens", "32"])
    assert args.prompt == "Hello"
    assert args.max_new_tokens == 32
    assert args.adapter_dir is None


def test_chat_single_prompt(capsys):
    parser = build_parser()
    args = parser.parse_args(["--prompt", "Namaste", "--max-new-tokens", "10"])

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = {"input_ids": MagicMock()}
    mock_tokenizer.decode.return_value = "Namaste, I am Tantra."
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1

    mock_model = MagicMock()
    mock_model.generate.return_value = MagicMock()

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
        chat_loop(args)

    captured = capsys.readouterr()
    assert "Namaste, I am Tantra." in captured.out
