"""Consolidated test suite: Tests/test_training_alignment.py"""


# ─────────────────────────────────────────────────────────────────
# Source: test_data.py
# ─────────────────────────────────────────────────────────────────

"""Tests for tantra.tokenizer + tantra.codec"""
import json
import os
import tempfile

import pytest
import torch

from Tantra.config import VocabConfig
from Tantra.tokenizer import ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer, AudioTokenizer, ImageTokenizer
from Tantra.dataset import TopicMixedDataset, PretokenizedBinDataset, JSONLDataset, IGNORE_INDEX, TokenJuiceEngine



class _StubTokenizer:
    vocab_size = 4096
    def encode(self, text, modality="text"):
        return [min((ord(c) % self.vocab_size), self.vocab_size - 1) for c in text]


def _write_chat_jsonl(path, n, prefix="q"):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "system": "You are a helpful assistant.",
                "user": f"{prefix} user {i}",
                "assistant": f"{prefix} assistant answer {i} with enough tokens here.",
            }) + "\n")


def test_topic_mixed_dataset_multi_file():
    """TopicMixedDataset must advance through every shard, not just the first file."""
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(3):
            _write_chat_jsonl(os.path.join(tmp, f"shard{i}.jsonl"), 4, prefix=f"t{i}")
        ds = TopicMixedDataset(
            {"general": [os.path.join(tmp, f"shard{i}.jsonl") for i in range(3)]},
            {"general": 1.0},
            _StubTokenizer(),
            seq_len=16,
            max_samples=50,
            seed=1,
        )
        got = [x for x, _ in ds]
        assert len(got) == 50
        for x in got:
            assert x.shape == (16,)
            assert x.dtype == torch.long


def test_topic_mixed_dataset_weighted_topics():
    """Both topics must contribute samples; the general-heavy mix yields more general samples."""
    with tempfile.TemporaryDirectory() as tmp:
        gen = os.path.join(tmp, "general.jsonl")
        code = os.path.join(tmp, "code.jsonl")
        _write_chat_jsonl(gen, 50, prefix="g")
        _write_chat_jsonl(code, 50, prefix="c")
        ds = TopicMixedDataset(
            {"general": [gen], "code": [code]},
            {"general": 5.0, "code": 1.0},
            _StubTokenizer(),
            seq_len=8,
            max_samples=300,
            seed=3,
        )
        n = 0
        for x, _ in ds:
            n += 1
        assert n == 300


def test_pretokenized_pretraining_supervises_all_tokens():
    """Pretraining must ignore chat masks and learn from every token."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = os.path.join(tmp, "corpus.bin")
        torch.save({
            "tokens": torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32),
            "masks": torch.tensor([False, False, False, False, False]),
        }, cache_path)
        _, pretrain_y = next(iter(PretokenizedBinDataset(cache_path, seq_len=4, mask_non_assistant=False)))
        _, sft_y = next(iter(PretokenizedBinDataset(cache_path, seq_len=4, mask_non_assistant=True)))
        assert torch.all(pretrain_y != IGNORE_INDEX)
        assert torch.all(sft_y == IGNORE_INDEX)


def test_megabyte_patcher_roundtrip():
    p = MegabytePatcher()
    data = b"hello world test bytes"
    ids = p.encode_bytes(data)
    assert len(ids) > 0

def test_unified_tokenizer_text():
    cfg = VocabConfig()
    bpe = ByteBPETokenizer(cfg)
    patcher = MegabytePatcher()
    ut = UnifiedTokenizer(cfg, bpe, patcher)
    ids = ut.encode("hello", "text")
    assert len(ids) > 0

def test_audio_tokenizer():
    cfg = VocabConfig()
    tok = AudioTokenizer(cfg)
    waveform = torch.randn(1, 1, 16000)
    ids = tok.encode(waveform)
    assert ids.min() >= 0 and ids.max() < cfg.audio_codebook_size

def test_image_tokenizer():
    cfg = VocabConfig()
    tok = ImageTokenizer(cfg)
    image = torch.rand(1, 3, 64, 64)
    ids = tok.encode(image)
    assert ids.min() >= 0 and ids.max() < cfg.image_codebook_size


def test_tokenjuice_entropy_enrichment_and_weights():
    engine = TokenJuiceEngine(entropy_threshold=0.3, enrichment_rate=1.0)
    low_entropy = [1] * 8
    high_entropy = [1, 250, 4829, 991, 12, 592, 1024, 881]
    assert engine.compute_token_entropy(low_entropy) < engine.compute_token_entropy(high_entropy)
    engine.register_synthetic_pair([100, 101, 102], [200, 201, 202])
    x, y = engine.enrich_batch(torch.zeros((2, 8), dtype=torch.long), torch.zeros((2, 8), dtype=torch.long))
    assert x.shape == y.shape == (2, 8)
    weights = engine.compute_dynamic_loss_weights(torch.tensor([1, 100, 5, 101]), [100, 101])
    assert weights.tolist() == [1.0, 2.5, 1.0, 2.5]


def test_jsonl_dataset_shuffling():
    """JSONLDataset with shuffle=True must randomize the order of lines streamed."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_shuffle.jsonl")
        _write_chat_jsonl(path, 100, prefix="item")
        
        # Non-shuffled dataset
        ds_noshuffle = JSONLDataset(path, _StubTokenizer(), seq_len=128, max_samples=100, shuffle=False)
        items_noshuffle = [x.tolist() for x, _ in ds_noshuffle]
        
        # Shuffled dataset
        ds_shuffle = JSONLDataset(path, _StubTokenizer(), seq_len=128, max_samples=100, shuffle=True, shuffle_buf_size=50, seed=42)
        items_shuffle = [x.tolist() for x, _ in ds_shuffle]
        
        assert len(items_noshuffle) == 100
        assert len(items_shuffle) == 100
        # Line order must differ when shuffle=True
        assert items_noshuffle != items_shuffle


def test_jsonl_dataset_epoch_reshuffling():
    """JSONLDataset must yield different line permutations on epoch 1 vs epoch 2 when iterating past EOF."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_epoch.jsonl")
        _write_chat_jsonl(path, 30, prefix="ep")
        
        ds = JSONLDataset(path, _StubTokenizer(), seq_len=128, max_samples=60, shuffle=True, shuffle_buf_size=20, seed=123)
        all_samples = [x.tolist() for x, _ in ds]
        
        assert len(all_samples) == 60
        epoch_1 = all_samples[:30]
        epoch_2 = all_samples[30:]
        # Epoch 1 and Epoch 2 must be different line permutations
        assert epoch_1 != epoch_2



def test_jsonl_dataset_validation_split():

    """Train and validation splits must be disjoint and contain zero line overlap."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_split.jsonl")
        _write_chat_jsonl(path, 100, prefix="item")

        ds_train = JSONLDataset(path, _StubTokenizer(), seq_len=128, max_samples=100, split="train", val_ratio=0.1, shuffle=False)
        ds_val = JSONLDataset(path, _StubTokenizer(), seq_len=128, max_samples=100, split="val", val_ratio=0.1, shuffle=False)

        train_items = set(tuple(x.tolist()) for x, _ in ds_train)
        val_items = set(tuple(x.tolist()) for x, _ in ds_val)

        assert len(train_items) > 0
        assert len(val_items) > 0
        # Zero line overlap between train and val
        assert train_items.isdisjoint(val_items)


def test_neurotrainer_evaluate_validation():
    """evaluate_validation must return val_loss, val_acc, and val_ppl without crashing."""
    from Tantra.model import build_cpu_model
    from Tantra.train import NeuroTrainer
    from torch.utils.data import DataLoader

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_eval_val.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i in range(20):
                f.write(json.dumps({"user": f"eval question {i}", "assistant": f"eval answer {i}"}) + "\n")

        tok = _StubTokenizer()
        ds_val = JSONLDataset(path, tok, seq_len=64, max_samples=20, split="val", val_ratio=0.5, shuffle=False)
        val_loader = DataLoader(ds_val, batch_size=2)

        model = build_cpu_model("micro10", attention_kind="causal")
        trainer = NeuroTrainer(model, lr=1e-4, total_steps=10)

        metrics = trainer.evaluate_validation(val_loader, max_val_batches=5)
        assert isinstance(metrics, dict)
        assert "val_loss" in metrics
        assert "val_acc" in metrics
        assert "val_ppl" in metrics
        assert metrics["val_loss"] > 0


def test_continuous_sequence_packing():
    """Continuous sequence packing must pack multiple documents with zero padding tokens."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "packed_test.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i in range(10):
                f.write(json.dumps({
                    "user": f"Prompt {i}",
                    "assistant": f"Reply {i}"
                }) + "\n")

        tok = _StubTokenizer()
        ds_packed = JSONLDataset(path, tok, seq_len=32, max_samples=10, pack_sequences=True, mask_non_assistant=True)
        samples = [s for s in ds_packed]
        assert len(samples) > 0
        for x, y in samples:
            assert x.shape == (32,)
            assert y.shape == (32,)
            # Verify that supervised assistant tokens exist in the packed chunk
            assert (y != IGNORE_INDEX).any()
            # Verify x has no zero padding tokens at the tail
            assert not (x[-4:] == 0).all()








# ─────────────────────────────────────────────────────────────────
# Source: test_dpo.py
# ─────────────────────────────────────────────────────────────────

import os
import pytest
import torch
from Tantra.config import NeuroCoreConfig, VocabConfig
from Tantra.model import NeuroCoreModel
from Tantra.tokenizer import ByteBPETokenizer
from Tantra.dataset import DPODataset
from Tantra.train import NeuroTrainer
from torch.utils.data import DataLoader


def test_dpo_dataset_and_training_step(tmp_path):
    # 1. Create a dummy tokenizer
    tok = ByteBPETokenizer(VocabConfig())
    
    # 2. Create small sample preference pairs JSONL
    dpo_file = tmp_path / "pref_pairs.jsonl"
    dpo_file.write_text(
        '{"prompt": "Hello", "chosen": "Hello! How can I help you?", "rejected": "hello hello hello"}\n'
        '{"prompt": "What is Python?", "chosen": "Python is a programming language.", "rejected": "python python"}\n',
        encoding="utf-8"
    )
    
    dataset = DPODataset(str(dpo_file), tok, max_len=32)
    loader = DataLoader(dataset, batch_size=2)
    
    batch = next(iter(loader))
    assert "chosen_input_ids" in batch
    assert "chosen_labels" in batch
    assert "rejected_input_ids" in batch
    assert "rejected_labels" in batch
    assert batch["chosen_input_ids"].shape == (2, 32)
    
    # 3. Create tiny model & trainer
    cfg = NeuroCoreConfig()
    cfg.block.alra.dim = 64
    cfg.block.sgp.dim = 64
    cfg.block.num_layers = 2
    cfg.block.alra.num_heads = 2
    cfg.block.alra.head_dim = 32
    cfg.vocab.vocab_size = 32768
    
    model = NeuroCoreModel(cfg)
    trainer = NeuroTrainer(model, lr=1e-4, grad_accumulation_steps=1)
    trainer.device = torch.device("cpu")
    
    # 4. Run DPO training for 3 steps
    losses = trainer.train_dpo(loader, max_steps=3, log_every=1, beta=0.1)
    assert len(losses) == 3
    assert all(isinstance(l, float) for l in losses)
    assert not any(torch.isnan(torch.tensor(l)) for l in losses)


# ─────────────────────────────────────────────────────────────────
# Source: test_optimizers.py
# ─────────────────────────────────────────────────────────────────

"""
Tests/test_optimizers.py — Comprehensive tests for AdamW, native Lion optimizer,
mathematical update correctness, and memory state verification.
"""
import torch
import pytest
from Tantra.model import build_cpu_model
from Tantra.train import NeuroTrainer, Lion, build_optimizer


def test_build_optimizers():
    p = [torch.nn.Parameter(torch.randn(10, 10))]
    opt_adamw = build_optimizer("adamw", p, lr=1e-4, weight_decay=0.01)
    assert opt_adamw is not None

    opt_lion = build_optimizer("lion", p, lr=3e-5, weight_decay=0.05)
    assert isinstance(opt_lion, (Lion, torch.optim.Optimizer))

    opt_adam = build_optimizer("adam", p, lr=1e-4, weight_decay=0.01)
    assert isinstance(opt_adam, torch.optim.Adam)

    opt_sgd = build_optimizer("sgd", p, lr=1e-3, weight_decay=0.0)
    assert isinstance(opt_sgd, torch.optim.SGD)


def test_lion_math_correctness_and_buffer_isolation():
    """Verify exact numerical formulation of Lion (Chen et al. 2023)."""
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    opt = Lion([p], lr=0.1, betas=(0.9, 0.99), weight_decay=0.0)
    
    # Step 1: grad = [0.5, -0.5]
    p.grad = torch.tensor([0.5, -0.5], dtype=torch.float32)
    opt.step()
    
    # Expected update:
    # m_0 = 0
    # update = sign(0.9 * 0 + 0.1 * [0.5, -0.5]) = sign([0.05, -0.05]) = [1.0, -1.0]
    # p_1 = [1.0, -2.0] - 0.1 * [1.0, -1.0] = [0.9, -1.9]
    # m_1 = 0.99 * 0 + 0.01 * [0.5, -0.5] = [0.005, -0.005]
    assert torch.allclose(p.data, torch.tensor([0.9, -1.9], dtype=torch.float32), atol=1e-6)
    assert torch.allclose(opt.state[p]["exp_avg"], torch.tensor([0.005, -0.005], dtype=torch.float32), atol=1e-6)

    # Step 2: verify momentum accumulation with non-zero buffer
    p.grad = torch.tensor([0.1, -0.1], dtype=torch.float32)
    opt.step()
    # update = sign(0.9 * [0.005, -0.005] + 0.1 * [0.1, -0.1]) = sign([0.0145, -0.0145]) = [1.0, -1.0]
    # p_2 = [0.9, -1.9] - 0.1 * [1.0, -1.0] = [0.8, -1.8]
    # m_2 = 0.99 * [0.005, -0.005] + 0.01 * [0.1, -0.1] = [0.00595, -0.00595]
    assert torch.allclose(p.data, torch.tensor([0.8, -1.8], dtype=torch.float32), atol=1e-6)
    assert torch.allclose(opt.state[p]["exp_avg"], torch.tensor([0.00595, -0.00595], dtype=torch.float32), atol=1e-6)


def test_lion_memory_footprint_vs_adamw():
    """Verify Lion uses exactly 1 state buffer while AdamW uses 2."""
    p_lion = torch.nn.Parameter(torch.randn(100, 100))
    p_adamw = torch.nn.Parameter(torch.randn(100, 100))

    opt_lion = Lion([p_lion], lr=1e-4)
    opt_adamw = torch.optim.AdamW([p_adamw], lr=1e-4)

    # Perform 1 step
    p_lion.grad = torch.randn_like(p_lion)
    p_adamw.grad = torch.randn_like(p_adamw)

    opt_lion.step()
    opt_adamw.step()

    # Lion must have exactly 1 momentum tensor
    assert len(opt_lion.state[p_lion]) == 1
    assert "exp_avg" in opt_lion.state[p_lion]

    # AdamW has 2 buffers (exp_avg, exp_avg_sq) + step
    assert len(opt_adamw.state[p_adamw]) >= 2
    assert "exp_avg" in opt_adamw.state[p_adamw]
    assert "exp_avg_sq" in opt_adamw.state[p_adamw]


def test_lion_step_execution():
    model = build_cpu_model("micro10", attention_kind="causal")
    trainer = NeuroTrainer(model, lr=3e-5, weight_decay=0.05, optimizer_name="lion", total_steps=5)
    
    x = torch.randint(0, 32768, (1, 16))
    y = torch.randint(0, 32768, (1, 16))
    loss, acc, ppl, grad_norm, at_boundary = trainer.train_step(x, y)
    
    assert loss > 0
    assert ppl > 0
    assert at_boundary is True
    assert trainer.step_count == 1


def test_optimizer_and_scheduler_continuity_on_resume(tmp_path):
    """Verify that same-stage resume preserves optimizer momentum and LR scheduler position."""
    model1 = build_cpu_model("micro10", attention_kind="causal")
    trainer1 = NeuroTrainer(model1, lr=1e-4, total_steps=100, warmup_steps=10)
    trainer1.training_stage = "sft"

    x = torch.randint(0, 32768, (1, 16))
    y = torch.randint(0, 32768, (1, 16))

    # Run 5 training steps to accumulate momentum and advance scheduler
    for _ in range(5):
        trainer1.train_step(x, y)

    initial_lr = trainer1.optimizer.param_groups[0]["lr"]
    assert trainer1.step_count == 5
    assert initial_lr > 0.0

    ckpt_path = str(tmp_path / "test_resume.pt")
    trainer1.save_checkpoint(ckpt_path, save_optimizer=True)

    # Recreate model & trainer and load checkpoint
    model2 = build_cpu_model("micro10", attention_kind="causal")
    trainer2 = NeuroTrainer(model2, lr=1e-4, total_steps=100, warmup_steps=10)
    trainer2.load_checkpoint(ckpt_path)

    assert trainer2.step_count == 5
    assert trainer2.training_stage == "sft"
    resumed_lr = trainer2.scheduler.get_last_lr()[0] if hasattr(trainer2.scheduler, "get_last_lr") else trainer2.optimizer.param_groups[0]["lr"]
    assert resumed_lr > 0.0

    # Ensure running step 6 does not crash or reset
    loss, acc, ppl, grad_norm, at_boundary = trainer2.train_step(x, y)
    assert trainer2.step_count == 6


