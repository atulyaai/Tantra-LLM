import torch

from npdna.config import MeshConfig, StrandConfig
from npdna.genome import Genome, GenomeConfig
from npdna.mesh import AttentionStrand, NeuralMesh
from npdna.train_npdna_v3 import mtp_aux_loss


def test_attention_strand_is_local_implementation():
    strand_cfg = StrandConfig(hidden_size=16, state_size=8, strand_type="attention")
    genome = Genome(GenomeConfig(latent_dim=32, max_strands=4), strand_cfg)
    mesh = NeuralMesh(genome, MeshConfig(num_strands=2, top_k=1, strand=strand_cfg))
    assert isinstance(mesh.strands[0], AttentionStrand)

    x = torch.randn(1, 4, 16)
    y, balance = mesh(x)
    assert y.shape == x.shape
    assert balance.ndim == 0


def test_mtp_aux_loss_is_scalar():
    logits = torch.randn(2, 5, 11)
    targets = torch.randint(0, 11, (2, 5))
    loss = mtp_aux_loss(logits, targets, depth=3)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
