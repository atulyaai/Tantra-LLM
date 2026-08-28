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
