import torch
from Tantra.config import ModelConfig
from Tantra.data.expert_dataset import collate
from Tantra.model.tantra_model import TantraModel


def test_one_padded_batch_equals_token_weighted_per_row_passes():
    torch.manual_seed(0)
    model = TantraModel(ModelConfig(vocab_size=50, d_model=16, n_layers=2, n_heads=2, n_kv_heads=1, mlp_mult=2, max_seq_len=32))
    rows = []
    for length, start in ((7, 2), (11, 5), (4, 1)):  # different lengths -> right padding in the batch
        ids = torch.randint(3, 50, (length,)).tolist()
        labels = [-100] * start + ids[start:]           # supervise only the "assistant" tail
        rows.append((ids, labels))
    # Old path: one forward per row, loss weighted by its share of supervised tokens.
    model.zero_grad()
    batches = [collate([r], 0) for r in rows]
    count = sum(int((y[:, 1:] != -100).sum()) for _, y, _ in batches)
    for x, y, _ in batches:
        (model.loss(x, y) * int((y[:, 1:] != -100).sum()) / count).backward()
    old = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    # New path: one right-padded batch.
    model.zero_grad()
    x, y, _ = collate(rows, 0)
    model.loss(x, y).backward()
    for n, p in model.named_parameters():
        if n in old:
            assert torch.allclose(old[n], p.grad, atol=1e-6), n
