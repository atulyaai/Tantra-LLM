import torch

from npdna.config import MeshConfig, StrandConfig
from npdna.genome import Genome, GenomeConfig
from npdna.mesh import AttentionStrand, NeuralMesh
from npdna.train_npdna_v3 import (
    Dataset,
    build_curriculum,
    format_chat_example,
    format_duration,
    load_seed_chat,
    mtp_aux_loss,
    stage_index_for_step,
)
from npdna.tokenizer import AtulyaTokenizer


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


def test_curriculum_scales_to_target_steps():
    curriculum = build_curriculum(100_000)
    assert curriculum[-1]["steps"] == 100_000
    assert all(a["steps"] < b["steps"] for a, b in zip(curriculum, curriculum[1:]))
    assert stage_index_for_step(652, curriculum) == 0


def test_format_duration():
    assert format_duration(42) == "42s"
    assert format_duration(125) == "2m 05s"
    assert format_duration(3665) == "1h 01m"


def test_seed_chat_examples_are_formatted_as_chat(tmp_path):
    seed_file = tmp_path / "seed_chat.jsonl"
    seed_file.write_text(
        '{"system":"","user":"What is gravity?","assistant":"Gravity attracts mass."}\n',
        encoding="utf-8",
    )

    examples = load_seed_chat(seed_file)

    assert examples == [
        "System: You are Atulya. Be warm, thoughtful, and direct.\n"
        "User: What is gravity?\n"
        "Assistant: Gravity attracts mass."
    ]
    assert format_chat_example("Hi", "Hello", "Be brief.") == (
        "System: Be brief.\nUser: Hi\nAssistant: Hello"
    )


def test_dataset_can_sample_only_seed_chat(tmp_path):
    seed_file = tmp_path / "seed_chat.jsonl"
    seed_file.write_text(
        '{"user":"Hello","assistant":"Hi there."}\n',
        encoding="utf-8",
    )
    tokenizer = AtulyaTokenizer(initial_capacity=256, max_capacity=512)
    dataset = Dataset(
        tmp_path,
        [],
        tokenizer,
        seq_len=16,
        seed_chat_path=seed_file,
        seed_chat_ratio=1.0,
    )

    x, y = dataset.sample_batch(batch_size=2, seq_len=16, allow_growth=True)

    assert len(dataset.seed_chat) == 1
    assert x.shape == (2, 16)
    assert y.shape == (2, 16)
