import os
import sys
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Tantra.model import NeuroCoreModel
from Tantra.config import NeuroCoreConfig


def test_mtp_dual_token_forward():
    cfg = NeuroCoreConfig.small()
    cfg.vocab.vocab_size = 1000
    model = NeuroCoreModel(cfg, use_mtp=True)
    model.eval()

    seq_len = 16
    input_ids = torch.randint(0, 1000, (1, seq_len))

    with torch.no_grad():
        (logits_main, logits_mtp), _ = model.forward(input_ids, return_mtp=True)

    assert logits_main.shape == (1, seq_len, 1000)
    assert logits_mtp.shape == (1, seq_len, 1000)

    pred_t1 = torch.argmax(logits_main[:, -1, :], dim=-1).item()
    pred_t2 = torch.argmax(logits_mtp[:, -1, :], dim=-1).item()

    assert 0 <= pred_t1 < 1000
    assert 0 <= pred_t2 < 1000
