import torch
from types import SimpleNamespace

from npdna.config import MeshConfig, StrandConfig
from npdna.cortex import MemoryCortex
from npdna.genome import Genome, GenomeConfig
from npdna.mesh import AttentionStrand, NeuralMesh
from npdna.model import NpDnaCore, NpDnaModel
from npdna.brain import PlasticityEngine
from npdna.train import (
    Dataset,
    FINAL_GENERATION_PROMPTS,
    GENERATION_PROBE_PROMPTS,
    IGNORE_INDEX,
    JsonlSeedRecordStore,
    _extract_training_text,
    all_folders,
    build_curriculum,
    eval_model,
    format_chat_example,
    format_chat_prompt,
    format_duration,
    load_seed_chat,
    load_texts,
    mtp_aux_loss,
    _nonfinite_loss_report,
    load_training_state,
    save_training_checkpoint,
    save_training_state,
    sample_generation_prompts,
    scheduled_lr,
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


def test_attention_strand_uses_multiple_heads_when_hidden_size_allows():
    strand_cfg = StrandConfig(hidden_size=64, state_size=8, strand_type="attention")
    genome = Genome(GenomeConfig(latent_dim=32, max_strands=4), strand_cfg)
    strand = AttentionStrand(genome, strand_id=0, config=strand_cfg)

    x = torch.randn(2, 5, 64)
    y = strand(x)

    assert strand.num_heads == 2
    assert strand.head_dim == 32
    assert y.shape == x.shape


def test_plasticity_diagnosis_calls_usage_stats_method():
    class MeshStub:
        def __init__(self):
            self.called = False
            self.last_router_entropy = 1.0

        @property
        def usage_stats(self):
            self.called = True
            return {0: 0.0, 1: 0.95}

        def reset_usage(self):
            pass

    mesh = MeshStub()
    core = SimpleNamespace(model=SimpleNamespace(mesh_layers=[mesh]))
    engine = PlasticityEngine(core, dead_threshold=0.01, overload_threshold=0.9)

    events = engine._diagnose_strand_usage(step=100)

    assert mesh.called
    assert {event.event_type for event in events} == {"low_usage_strands", "overloaded_strands"}


def test_checkpoint_preserves_attention_strand_type(tmp_path):
    core = NpDnaCore.from_config("seed")
    for spec in core.config.mesh_specs:
        spec.strand.strand_type = "attention"
    core.config.mesh.strand.strand_type = "attention"
    core.model = NpDnaModel(core.config)

    ckpt = tmp_path / "attention_ckpt"
    core.save(ckpt)

    del core
    import gc
    gc.collect()

    loaded = NpDnaCore.load(ckpt)

    assert loaded.config.mesh.strand.strand_type == "attention"
    assert all(spec.strand.strand_type == "attention" for spec in loaded.config.mesh_specs)
    assert isinstance(loaded.model.mesh_layers[0].strands[0], AttentionStrand)
    assert not (ckpt / "model.pt.tmp").exists()
    assert not (ckpt / "tokenizer.json.tmp").exists()
    assert not (ckpt / "metadata.json.tmp").exists()


def test_checkpoint_load_restores_vocab_growth_headroom(tmp_path):
    core = NpDnaCore.from_config("seed")
    core.tokenizer.growth_threshold = 2.0
    while core.tokenizer.size < core.tokenizer.capacity:
        core.tokenizer.add_token(f"<extra_{core.tokenizer.size}>")
    core.tokenizer.max_capacity = core.tokenizer.capacity
    saved_capacity = core.tokenizer.capacity

    ckpt = tmp_path / "full_vocab_ckpt"
    core.save(ckpt)

    expected_size = core.tokenizer.size
    del core
    import gc
    gc.collect()

    loaded = NpDnaCore.load(ckpt)

    assert loaded.tokenizer.size == expected_size
    assert loaded.tokenizer.capacity > saved_capacity
    assert loaded.tokenizer.max_capacity == loaded.config.max_vocab
    assert loaded.model.embedding.num_embeddings == loaded.tokenizer.capacity


def test_checkpoint_load_raises_low_saved_vocab_ceiling(tmp_path):
    core = NpDnaCore.from_config("seed")
    core.tokenizer.max_capacity = core.tokenizer.capacity * 4

    ckpt = tmp_path / "low_vocab_ceiling_ckpt"
    core.save(ckpt)

    expected_capacity = core.tokenizer.capacity
    del core
    import gc
    gc.collect()

    loaded = NpDnaCore.load(ckpt)

    assert loaded.tokenizer.capacity == expected_capacity
    assert loaded.tokenizer.max_capacity == loaded.config.max_vocab


def test_training_checkpoint_saves_schedule_metadata(tmp_path, monkeypatch):
    import json
    import npdna.train as trainer

    monkeypatch.setattr(trainer, "CKPT_DIR", tmp_path)
    core = NpDnaCore.from_config("seed")

    save_training_checkpoint(
        core,
        "latest",
        losses=[1.0],
        step=12,
        best_val=0.5,
        stage=1,
        mtp_depth=1,
        target_steps=150_000,
        peak_lr=5e-5,
    )

    meta = json.loads((tmp_path / "latest" / "metadata.json").read_text(encoding="utf-8"))

    assert meta["target_steps"] == 150_000
    assert meta["peak_lr"] == 5e-5


def test_training_state_round_trips_optimizer_state(tmp_path):
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(4, 3)
    loss = model(x).pow(2).mean()
    loss.backward()
    opt.step()

    save_training_state(tmp_path, opt=opt)

    restored_model = torch.nn.Linear(3, 2)
    restored_opt = torch.optim.AdamW(restored_model.parameters(), lr=0.01)

    assert load_training_state(tmp_path, restored_opt)
    assert restored_opt.state_dict()["state"]
    assert restored_opt.state_dict()["param_groups"][0]["lr"] == 0.01


def test_mtp_aux_loss_is_scalar():
    logits = torch.randn(2, 5, 11)
    targets = torch.randint(0, 11, (2, 5))
    loss = mtp_aux_loss(logits, targets, depth=3)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_nonfinite_loss_report_names_bad_components():
    report = _nonfinite_loss_report(
        ce=torch.tensor(float("nan")),
        mtp=torch.tensor(0.0),
        balance=torch.tensor([1.0, float("inf")]),
        exit=0.0,
    )

    assert "ce=" in report
    assert "balance=" in report
    assert "mtp=" not in report
    assert "exit=" not in report


def test_nonfinite_loss_report_ignores_finite_components():
    assert _nonfinite_loss_report(ce=torch.tensor(1.0), mtp=0.0) is None


def test_scaled_ssm_model_first_backward_has_finite_gradients():
    core = NpDnaCore.from_config(2.0)
    core.model.train()
    vocab = min(core.tokenizer.capacity, 1024)
    x = torch.randint(0, vocab, (1, 16))
    y = x.clone()

    logits, balance = core.model(x)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss = loss + balance * 0.1
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(core.model.parameters(), 1.0)

    assert torch.isfinite(loss)
    assert torch.isfinite(grad_norm)


def test_curriculum_scales_to_target_steps():
    curriculum = build_curriculum(100_000)
    assert curriculum[-1]["steps"] == 100_000
    assert all(a["steps"] < b["steps"] for a, b in zip(curriculum, curriculum[1:]))
    assert stage_index_for_step(652, curriculum) == 0
    assert curriculum[-1]["folders"] == all_folders


def test_generation_probe_prompts_are_varied_and_stable():
    prompts = sample_generation_prompts(step=1234, count=4)

    assert len(prompts) == 4
    assert prompts == sample_generation_prompts(step=1234, count=4)
    assert prompts != sample_generation_prompts(step=1235, count=4)
    assert len(GENERATION_PROBE_PROMPTS) >= 12
    assert any("Python" in prompt for prompt in GENERATION_PROBE_PROMPTS)
    assert any("gravity" in prompt.lower() for prompt in GENERATION_PROBE_PROMPTS)
    assert any("study" in prompt.lower() for prompt in GENERATION_PROBE_PROMPTS)
    assert len(FINAL_GENERATION_PROMPTS) >= len(prompts)


def test_format_duration():
    assert format_duration(42) == "42s"
    assert format_duration(125) == "2m 05s"
    assert format_duration(3665) == "1h 01m"


def test_scheduled_lr_uses_target_horizon_not_smoke_window():
    peak_lr = 5e-5

    lr_short_horizon = scheduled_lr(60_000, peak_lr, target_steps=63_000)
    lr_long_horizon = scheduled_lr(60_000, peak_lr, target_steps=150_000)

    assert lr_short_horizon < 2e-6
    assert lr_long_horizon > 3e-5


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
    assert format_chat_prompt("Hi", "Be brief.") == "System: Be brief.\nUser: Hi\nAssistant:"


def test_extract_training_text_formats_qa_as_chat():
    text = _extract_training_text(
        '{"instruction":"What is gravity?","response":"Gravity attracts mass.","system":"Be brief."}'
    )

    assert text == "System: Be brief.\nUser: What is gravity?\nAssistant: Gravity attracts mass."


def test_extract_training_text_wraps_long_plain_text_as_chat():
    text = _extract_training_text(
        '{"text":"Gravity is a fundamental interaction. It causes objects with mass to attract one another and shapes planetary motion."}'
    )

    assert text == (
        "System: You are Atulya. Be warm, thoughtful, and direct.\n"
        "User: Tell me about: Gravity is a fundamental interaction.\n"
        "Assistant: It causes objects with mass to attract one another and shapes planetary motion."
    )


def test_clip_image_embedding_missing_file_is_explicit(tmp_path):
    from npdna.brain import encode_image_clip

    missing = tmp_path / "missing.png"

    try:
        encode_image_clip(missing)
    except FileNotFoundError as exc:
        assert exc.filename is None or str(missing) in str(exc)
    else:
        raise AssertionError("encode_image_clip should reject missing image paths")


def test_dataset_can_sample_only_seed_chat(tmp_path):
    seed_file = tmp_path / "seed_chat.jsonl"
    seed_file.write_text(
        '{"user":"Hello","assistant":"Hi there."}\n',
        encoding="utf-8",
    )
    tokenizer = AtulyaTokenizer(initial_capacity=8192, max_capacity=12288)
    dataset = Dataset(
        tmp_path,
        [],
        tokenizer,
        seq_len=128,
        seed_chat_path=seed_file,
        seed_chat_ratio=1.0,
    )

    x, y = dataset.sample_batch(batch_size=1, seq_len=128, allow_growth=True)

    assert len(dataset.seed_chat) == 1
    assert x.shape == (1, 128)
    assert y.shape == (1, 128)
    assert (y == IGNORE_INDEX).any()
    assert (y != IGNORE_INDEX).any()


def test_dataset_can_fill_batch_with_seed_chat(tmp_path):
    seed_file = tmp_path / "seed_chat.jsonl"
    seed_file.write_text(
        '{"user":"Hello","assistant":"Hi there."}\n'
        '{"user":"What is gravity?","assistant":"Gravity attracts mass."}\n',
        encoding="utf-8",
    )
    tokenizer = AtulyaTokenizer(initial_capacity=8192, max_capacity=12288)
    dataset = Dataset(
        tmp_path,
        [],
        tokenizer,
        seq_len=128,
        seed_chat_path=seed_file,
        seed_chat_ratio=1.0,
        max_seed_per_batch_pct=1.0,
    )

    x, y = dataset.sample_batch(batch_size=4, seq_len=128, allow_growth=True)

    assert x.shape == (4, 128)
    assert y.shape == (4, 128)
    assert (y != IGNORE_INDEX).any()


def test_seed_chat_short_windows_keep_answer_targets(tmp_path):
    seed_file = tmp_path / "seed_chat.jsonl"
    seed_file.write_text(
        '{"user":"This is a deliberately long prompt with many words before the answer",'
        '"assistant":"Short answer."}\n',
        encoding="utf-8",
    )
    tokenizer = AtulyaTokenizer(initial_capacity=8192, max_capacity=12288)
    dataset = Dataset(
        tmp_path,
        [],
        tokenizer,
        seq_len=16,
        seed_chat_path=seed_file,
        seed_chat_ratio=1.0,
        max_seed_per_batch_pct=1.0,
    )

    for _ in range(20):
        _, y = dataset.sample_batch(batch_size=1, seq_len=16, allow_growth=True)
        assert (y != IGNORE_INDEX).any()


def test_preencoded_seed_cache_is_skipped_after_vocab_change(tmp_path, monkeypatch):
    seed_file = tmp_path / "seed_chat.jsonl"
    seed_file.write_text(
        '{"user":"Hello","assistant":"Hi there."}\n',
        encoding="utf-8",
    )
    tokenizer = AtulyaTokenizer(initial_capacity=8192, max_capacity=12288)
    dataset = Dataset(
        tmp_path,
        [],
        tokenizer,
        seq_len=32,
        seed_chat_path=seed_file,
        seed_chat_ratio=1.0,
    )
    tokenizer.add_token("new_vocab_token")

    calls = {"count": 0}
    original = dataset._encode_seed_chat_record

    def wrapped(record, allow_growth=True):
        calls["count"] += 1
        return original(record, allow_growth)

    monkeypatch.setattr(dataset, "_encode_seed_chat_record", wrapped)

    dataset._encode_seed_chat(0, seq_len=32, allow_growth=False)

    assert calls["count"] == 1


def test_eval_set_uses_held_out_masked_seed_records(tmp_path):
    seed_file = tmp_path / "seed_chat.jsonl"
    rows = [
        f'{{"user":"Question {i}","assistant":"Answer {i}."}}\n'
        for i in range(40)
    ]
    seed_file.write_text("".join(rows), encoding="utf-8")
    tokenizer = AtulyaTokenizer(initial_capacity=8192, max_capacity=12288)
    dataset = Dataset(
        tmp_path,
        [],
        tokenizer,
        seq_len=32,
        seed_chat_path=seed_file,
        seed_chat_ratio=1.0,
        max_seed_per_batch_pct=1.0,
    )

    eval_items = dataset.eval_set(num_samples=10)

    assert len(dataset.eval_seed_chat_records) > 0
    assert len(dataset.seed_chat_records) < 40
    assert isinstance(eval_items[0], tuple)
    _, targets = eval_items[0]
    assert IGNORE_INDEX in targets
    assert any(t != IGNORE_INDEX for t in targets)


def test_eval_model_runs_when_samples_are_fewer_than_batch():
    class ModelStub:
        def __init__(self):
            self.training = True
            self.called = False

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

        def __call__(self, x):
            self.called = True
            logits = torch.zeros(x.shape[0], x.shape[1], 8)
            logits[..., 1] = 4.0
            return logits, torch.tensor(0.0)

    model = ModelStub()
    loss, ppl = eval_model(model, [[1, 1, 1, 1, 1]], batch_size=4, seq_len=4)

    assert model.called
    assert loss > 0
    assert ppl > 1
    assert model.training


def test_eval_set_falls_back_to_current_chunks(tmp_path):
    data_dir = tmp_path / "data"
    factual = data_dir / "factual"
    factual.mkdir(parents=True)
    (factual / "data.jsonl").write_text(
        '{"text":"This factual eval fallback sample has enough words to build a window. "}\n'
        '{"text":"Another fallback sample gives validation real targets instead of dummy zeros. "}\n',
        encoding="utf-8",
    )

    tokenizer = AtulyaTokenizer(initial_capacity=8192, max_capacity=12288)
    dataset = Dataset(
        data_dir,
        ["factual"],
        tokenizer,
        seq_len=8,
        seed_chat_path=tmp_path / "missing_seed",
        seed_chat_ratio=0.0,
    )

    eval_items = dataset.eval_set(num_samples=2)

    assert eval_items
    assert any(any(token != 0 for token in item) for item in eval_items)


def test_large_seed_chat_uses_indexed_store(tmp_path, monkeypatch):
    import npdna.train as trainer

    seed_file = tmp_path / "large_seed.jsonl"
    seed_file.write_text(
        "".join(
            f'{{"user":"Question {i}","assistant":"Answer {i}."}}\n'
            for i in range(80)
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(trainer, "LARGE_SEED_CHAT_BYTES", 1)
    tokenizer = AtulyaTokenizer(initial_capacity=8192, max_capacity=12288)

    dataset = Dataset(
        tmp_path,
        [],
        tokenizer,
        seq_len=32,
        seed_chat_path=seed_file,
        seed_chat_ratio=1.0,
        max_seed_per_batch_pct=1.0,
    )

    assert isinstance(dataset.seed_chat_records, JsonlSeedRecordStore)
    assert len(dataset.seed_chat_records) > 0
    assert dataset.eval_seed_chat_records
    sampled_texts = list(dataset.seed_vocab_texts(sample_size=5))
    assert len(sampled_texts) == 5
    assert all("Assistant:" in text for text in sampled_texts)
    x, y = dataset.sample_batch(batch_size=2, seq_len=32, allow_growth=True)
    assert x.shape == (2, 32)
    assert y.shape == (2, 32)


def test_load_texts_can_start_from_later_line(tmp_path):
    data_file = tmp_path / "chunk.jsonl"
    data_file.write_text(
        '{"text":"first sample text"}\n'
        '{"text":"second sample text"}\n'
        '{"text":"third sample text"}\n',
        encoding="utf-8",
    )

    assert load_texts(data_file, max_lines=1, start_line=1) == ["second sample text"]


def test_repetition_penalty():
    from npdna.generation import _apply_repetition_penalty
    import torch
    # logits: [10.0, -10.0]
    logits = torch.tensor([10.0, -10.0])
    # tok_id 0 and 1 are seen — with freq_penalty=0.0, behaves like original
    penalized = _apply_repetition_penalty(logits, [0, 1], penalty=2.0, freq_penalty=0.0)

    # 10.0 should be divided by 2.0 -> 5.0
    assert abs(penalized[0].item() - 5.0) < 1e-4
    # -10.0 should be multiplied by 2.0 -> -20.0
    assert abs(penalized[1].item() - (-20.0)) < 1e-4


def test_repetition_penalty_frequency_scaling():
    """Tokens seen more often should be penalized harder."""
    from npdna.generation import _apply_repetition_penalty
    import torch
    logits = torch.tensor([5.0, 5.0])
    # tok_id 0 seen 1x, tok_id 1 seen 10x
    seen = [0] + [1] * 10
    penalized = _apply_repetition_penalty(logits, seen, penalty=1.2, freq_penalty=0.3)
    # tok_id 1 should be penalized MORE than tok_id 0
    assert penalized[1].item() < penalized[0].item()


def test_ngram_blocking():
    """Repeated 3-grams should be blocked."""
    from npdna.generation import _block_ngram_repeats
    import torch
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    # ids ends with [1, 2] — if [1, 2, 3] appeared earlier, token 3 should be blocked
    ids = [1, 2, 3, 0, 0, 1, 2]
    blocked = _block_ngram_repeats(logits, ids, n=3)
    assert blocked[3].item() == float("-inf")
    # Other tokens should be unchanged
    assert blocked[0].item() == 1.0
    assert blocked[4].item() == 5.0


def test_ngram_blocking_no_repeat():
    """When there's no n-gram repeat, nothing is blocked."""
    from npdna.generation import _block_ngram_repeats
    import torch
    logits = torch.tensor([1.0, 2.0, 3.0])
    ids = [0, 1, 2]
    blocked = _block_ngram_repeats(logits, ids, n=3)
    assert (blocked == logits).all()


def test_generation_stop_scan_uses_rolling_decode_and_writeback_no_grad():
    from npdna.generation import GenerationMixin

    class GenomeStub:
        def enable_inference_cache(self):
            pass

        def disable_inference_cache(self):
            pass

    class ModelStub:
        def __init__(self):
            self.embedding = torch.nn.Embedding(8, 4)
            self.genome = GenomeStub()
            self.vocab_size = 8
            self.calls = 0

        def eval(self):
            pass

        def __call__(self, input_ids):
            next_ids = [4, 5]
            next_id = next_ids[min(self.calls, len(next_ids) - 1)]
            self.calls += 1
            logits = torch.full((1, input_ids.shape[1], self.vocab_size), -100.0)
            logits[0, -1, next_id] = 100.0
            return logits, None

    class CoreStub(GenerationMixin):
        def __init__(self):
            self.model = ModelStub()
            self.tokenizer = SimpleNamespace(
                size=8,
                token_to_id={"<eos>": 3},
                byte_to_id={},
                id_to_token=[""] * 8,
            )
            self.active_path = None
            self.decode_lengths = []
            self.writeback_grad_enabled = None

        def encode(self, text, allow_growth=False):
            return [1]

        def decode(self, ids):
            self.decode_lengths.append(len(ids))
            return {4: "Hello ", 5: "User:"}.get(ids[0], "")

        def _record_strand_specialization(self, prompt):
            pass

        def _handle_cortex_writeback(self, generated_ids, device):
            self.writeback_grad_enabled = torch.is_grad_enabled()

    core = CoreStub()

    assert "".join(core.generate_stream("Hi", max_tokens=5, temperature=0, top_k=0)) == "Hello User:"
    assert core.decode_lengths == [1, 1]
    assert core.writeback_grad_enabled is False


def test_cortex_augment_ignores_low_relevance_retrieval(monkeypatch):
    from npdna.config import CortexConfig

    cortex = MemoryCortex(CortexConfig(dim=4, top_k=2, min_relevance=0.3))
    cortex.store(torch.ones(4))
    hidden = torch.randn(1, 3, 4)

    def low_relevance_retrieve(query, top_k=None):
        values = torch.ones(query.shape[0], 2, 4)
        scores = torch.full((query.shape[0], 2), 0.29)
        return values, scores

    monkeypatch.setattr(cortex, "retrieve", low_relevance_retrieve)

    assert torch.equal(cortex.augment(hidden), hidden)


def test_proportional_dataset_mix(tmp_path):
    s0 = tmp_path / "samples"
    s0.mkdir()
    (s0 / "data1.jsonl").write_text('{"text":"samples text example"}\n', encoding="utf-8")

    s1 = tmp_path / "agentic"
    s1.mkdir()
    (s1 / "data2.jsonl").write_text('{"text":"agentic text example"}\n', encoding="utf-8")

    tokenizer = AtulyaTokenizer(initial_capacity=8192, max_capacity=12288)

    dataset = Dataset(
        tmp_path,
        ["samples", "agentic"],
        tokenizer,
        seq_len=8,
        seed_chat_ratio=0.0,
        proportional_mix=True,
    )

    assert len(dataset._new_chunks) == 1
    assert len(dataset._prev_chunks) == 1
    assert len(dataset._chunks) == 2

    # Verify both folders are loaded and can be sampled from
    x, y = dataset.sample_batch(batch_size=1, seq_len=8)
    assert x.shape == (1, 8)
