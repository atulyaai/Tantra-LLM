import json

import pytest

from Tantra.qwen35_finetune import (
    check_family_disjoint,
    encode_rows,
    load_manifest_test_rows,
    load_prepared_manifest,
    read_reviewed_jsonl,
    sha256_file,
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
        "language": "hi",
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


def test_prepared_manifest_hashes_are_verified_without_training_on_test(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    experts_dir = prepared / "train" / "conversation"
    experts_dir.mkdir(parents=True)
    val_dir = prepared / "val" / "conversation"
    val_dir.mkdir(parents=True)
    test_dir = prepared / "test" / "conversation"
    test_dir.mkdir(parents=True)

    split_rows = {
        "train": row("train-family"),
        "val": row("validation-family"),
        "test": row("test-family"),
    }
    files = {}
    for split, item in split_rows.items():
        item["split"] = split
        item["split_group"] = item["family_id"]
        path = {"train": experts_dir, "val": val_dir, "test": test_dir}[split] / "general.jsonl"
        path.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
        files[split] = {
            "path": path.relative_to(prepared).as_posix(),
            "sha256": sha256_file(path),
        }

    manifest = {"version": 1, "experts": {"general": {"files": files}}}
    manifest_path = prepared / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    train_rows, val_rows, manifest_hash, hashes = load_prepared_manifest(manifest_path, ["general"])
    assert len(train_rows) == len(val_rows) == 1
    assert len(manifest_hash) == 64
    assert set(hashes) == {"train", "val", "test"}

    test_rows, test_manifest_hash, test_hashes = load_manifest_test_rows(manifest_path, ["general"])
    assert len(test_rows) == 1
    assert test_manifest_hash == manifest_hash
    assert set(test_hashes) == {"general"}

    (prepared / files["train"]["path"]).write_text("tampered\\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_prepared_manifest(manifest_path, ["general"])
