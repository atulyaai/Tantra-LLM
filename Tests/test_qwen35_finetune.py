import json

import pytest

from Tantra.qwen35_finetune import (
    check_family_disjoint,
    encode_rows,
    read_reviewed_jsonl,
)


class TinyChatTokenizer:
    """Deterministic chat-template fixture; no Transformers download needed."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        tokens = [1]
        for message in messages:
            tokens.append({"system": 2, "user": 3, "assistant": 4}[message["role"]])
            tokens.extend(ord(char) + 10 for char in message["content"])
            tokens.append(5)
        if add_generation_prompt:
            tokens.append(4)
        return tokens


def row(family="family-a"):
    return {
        "family_id": family,
        "messages": [
            {"role": "user", "content": "नमस्ते"},
            {"role": "assistant", "content": "नमस्ते!"},
        ],
        "source": {"id": "fixture", "revision": "1", "license": "MIT"},
        "review": {"status": "reviewed", "reviewer": "test"},
    }


def test_encode_rows_masks_prompt_and_supervises_assistant_only():
    examples = encode_rows([row()], TinyChatTokenizer(), max_length=128)
    assert len(examples) == 1
    labels = examples[0]["labels"]
    assert labels[0] == -100
    prompt = TinyChatTokenizer().apply_chat_template(row()["messages"][:1], add_generation_prompt=True)
    assert all(token == -100 for token in labels[:len(prompt)])
    assert all(token != -100 for token in labels[len(prompt):])


def test_encode_rows_keeps_target_when_truncating_long_context():
    item = row()
    item["messages"][0]["content"] = "long context " * 20
    examples = encode_rows([item], TinyChatTokenizer(), max_length=16)
    assert len(examples) == 1
    assert len(examples[0]["input_ids"]) <= 16
    assert any(token != -100 for token in examples[0]["labels"])


def test_train_and_validation_must_have_disjoint_families():
    with pytest.raises(ValueError, match="family leakage"):
        check_family_disjoint([row("same")], [row("same")])


def test_reviewed_jsonl_requires_provenance_and_review(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(row()) + "\n", encoding="utf-8")
    rows, digest = read_reviewed_jsonl(path)
    assert len(rows) == 1
    assert len(digest) == 64

    bad = row()
    bad["review"] = {"status": "pending"}
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="review status"):
        read_reviewed_jsonl(path)
