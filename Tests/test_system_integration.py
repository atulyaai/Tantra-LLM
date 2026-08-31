"""Consolidated test suite: Tests/test_system_integration.py"""


# ─────────────────────────────────────────────────────────────────
# Source: test_robustness.py
# ─────────────────────────────────────────────────────────────────

"""
tests/test_robustness.py — Comprehensive edge case & robustness stress test suite for Tantra-LLM.
Tests empty inputs, giant sequence lengths, single token prompts, invalid token IDs, NaNs, shape matching, and boundary conditions.
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn

from Tantra.config import NeuroCoreConfig, BitNetConfig, CompressionConfig, MoEConfig, VocabConfig
from Tantra.bitnet import StraightThrough, TernaryQuantizer, BitLinear, TernaryCPUKernel
from Tantra.codec import DNACodec, ResidualPredictor, ZSTDDictTrainer, AdaptiveHuffmanCoder
from Tantra.model import NeuroCoreModel, ALRAAttention, DynamicScaleNorm, RotaryPositionalEncoding, SparseGatedProjection
from Tantra.tokenizer import ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer
from Tantra.evolution import AutoGrowthController, SelfRepairEngine
from Tantra.moe import MoERouter, LoadBalancer, ExpertRegistry, LazyExpertLoader
from Tantra.dataset import JSONLDataset, format_jsonl_prompt
from Tantra.eval_suite import EvaluationEngine
from Tantra.train import NeuroTrainer, generate_synthetic_batch


# ── 1. BitNet Edge Cases & Stress Tests ──────────────────────────────────────

def test_bitnet_empty_tensor():
    cfg = BitNetConfig()
    quantizer = TernaryQuantizer(cfg)
    empty_w = torch.empty((0, 10), dtype=torch.float32)
    
    w_q, scale = quantizer.quantize(empty_w)
    assert w_q.numel() == 0
    assert not torch.isnan(scale)

    packed = quantizer.pack(w_q)
    assert packed.numel() == 0

    unpacked = quantizer.unpack(packed, (0, 10))
    assert unpacked.shape == (0, 10)


def test_bitnet_nan_and_inf_inputs():
    x = torch.tensor([float('nan'), float('inf'), -float('inf'), 0.5, -0.5])
    out = StraightThrough.apply(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_bitlinear_inference_mode_edge_cases():
    layer = BitLinear(16, 32, bias=True)
    layer.to_inference_mode()
    
    # Batch 1, Seq 1
    x_single = torch.randn(1, 1, 16)
    out_single = layer(x_single)
    assert out_single.shape == (1, 1, 32)
    assert not torch.isnan(out_single).any()

    # 1D Input
    x_1d = torch.randn(16)
    out_1d = layer(x_1d)
    assert out_1d.shape == (32,)


# ── 2. Codec & DNA Compression Robustness ───────────────────────────────────

def test_dna_codec_various_dtypes_and_shapes():
    cfg = CompressionConfig()
    codec = DNACodec(cfg)
    
    for dtype in [torch.float32, torch.float16, torch.int32, torch.int8]:
        tensor = torch.randn(64, 64).to(dtype) if dtype.is_floating_point else torch.randint(-50, 50, (64, 64)).to(dtype)
        with tempfile.NamedTemporaryFile(suffix=".dna", delete=False) as tmp:
            path = tmp.name
        try:
            stats = codec.compress(tensor, path)
            assert stats.sha256_match
            decompressed = codec.decompress(path)
            assert decompressed.shape == tensor.shape
            assert decompressed.dtype == tensor.dtype
        finally:
            if os.path.exists(path):
                os.remove(path)


def test_residual_predictor_zero_epochs():
    cfg = CompressionConfig()
    predictor = ResidualPredictor(cfg)
    res = predictor.train_on_tensors([], epochs=0)
    assert "final_loss" in res
    assert not torch.isnan(torch.tensor(res["final_loss"]))


# ── 3. Model Architecture & RoPE Shape Mismatch Tests ─────────────────────────

def test_rope_heads_equals_seq_len_bug_fix():
    """Stress test RoPE when num_heads == seq_len (e.g. 32 heads, seq_len = 32)."""
    rope = RotaryPositionalEncoding(head_dim=64)
    num_heads = 32
    seq_len = 32
    q = torch.randn(2, num_heads, seq_len, 64)
    k = torch.randn(2, num_heads, seq_len, 64)
    
    q_rot, k_rot = rope.apply(q, k, seq_len)
    assert q_rot.shape == (2, num_heads, seq_len, 64)
    assert k_rot.shape == (2, num_heads, seq_len, 64)
    assert not torch.isnan(q_rot).any()


def test_model_giant_sequence_length():
    cfg = NeuroCoreConfig.small()
    cfg.block.num_layers = 1
    cfg.block.alra.dim = 32
    cfg.block.alra.num_heads = 4
    cfg.block.alra.head_dim = 8
    cfg.block.sgp.dim = 32
    model = NeuroCoreModel(cfg)
    model.eval()
    
    # Giant sequence length = 1024 tokens
    giant_input = torch.randint(0, cfg.vocab.vocab_size, (1, 1024))
    with torch.no_grad():
        logits, _ = model(giant_input)
    assert logits.shape == (1, 1024, cfg.vocab.vocab_size)
    assert not torch.isnan(logits).any()


def test_model_single_token_prompt_and_empty_prompt():
    cfg = NeuroCoreConfig.small()
    cfg.block.num_layers = 1
    cfg.block.alra.dim = 32
    cfg.block.alra.num_heads = 4
    cfg.block.alra.head_dim = 8
    cfg.block.sgp.dim = 32
    model = NeuroCoreModel(cfg)
    model.eval()
    
    # Single token prompt
    single_token = torch.tensor([[42]])
    gen_single = model.generate(single_token, max_new_tokens=5)
    assert gen_single.shape == (1, 6)

    # Empty prompt (0 seq length)
    empty_prompt = torch.tensor([[]], dtype=torch.long)
    gen_empty = model.generate(empty_prompt, max_new_tokens=5)
    assert gen_empty.size(1) >= 5
    assert not torch.isnan(gen_empty.float()).any()


# ── 4. MoE & Routing Edge Cases ──────────────────────────────────────────────

def test_moe_router_empty_input():
    cfg = MoEConfig()
    router = MoERouter(cfg, embed_dim=128)
    empty_x = torch.empty((0, 0, 128))
    weights, experts, _ = router(empty_x)
    assert weights.numel() == 0
    
    loss = router.load_balancing_loss(experts)
    assert not torch.isnan(loss)
    assert loss.item() == 0.0


def test_load_balancer_zero_coeff():
    lb = LoadBalancer(num_experts=8, coeff=0.0)
    probs = torch.rand(2, 16, 8)
    loss = lb(probs)
    assert loss.item() == 0.0


# ── 5. Tokenizer & Out-of-Vocabulary Robustness ──────────────────────────────

def test_tokenizer_out_of_bounds_token_ids():
    cfg = VocabConfig()
    bpe = ByteBPETokenizer(cfg)
    patcher = MegabytePatcher()
    tok = UnifiedTokenizer(cfg, bpe, patcher)
    
    # Invalid negative and huge token IDs
    invalid_ids = [-100, 9999999, 0, 255]
    decoded = tok.decode(invalid_ids, modality="text")
    assert isinstance(decoded, str)


def test_megabyte_patcher_empty_codebook():
    patcher = MegabytePatcher()
    # Codebook is None
    decoded = patcher.decode_to_bytes([1, 2, 3])
    assert len(decoded) == 3 * patcher.patch_size


# ── 6. Self-Repair & Auto-Growth Robustness ──────────────────────────────────

def test_self_repair_engine_corrupted_model():
    cfg = NeuroCoreConfig.small()
    cfg.block.num_layers = 1
    cfg.block.alra.dim = 32
    cfg.block.alra.num_heads = 4
    cfg.block.alra.head_dim = 8
    cfg.block.sgp.dim = 32
    model = NeuroCoreModel(cfg)
    
    # Inject NaNs, Infs, and dead neurons manually
    with torch.no_grad():
        model.embed.weight[0, 0] = float('nan')
        model.embed.weight[1, 0] = float('inf')
        model.layers[0].attn.w_q.weight.data.fill_(1000.0)  # exploded
        model.layers[0].mlp.w_up.weight.data[0].fill_(0.0)  # dead neuron
        
    repair = SelfRepairEngine()
    stats = repair.scan_and_repair(model, max_norm=50.0)
    
    assert stats["repaired_nans"] >= 2
    assert stats["repaired_explosions"] >= 1
    assert stats["repaired_dead"] >= 1
    
    # Verify no NaNs remain
    for p in model.parameters():
        assert not torch.isnan(p.data).any()
        assert not torch.isinf(p.data).any()


def test_auto_growth_empty_layers():
    controller = AutoGrowthController()
    dummy_model = nn.Module()
    dummy_model.layers = nn.ModuleList()
    
    # Grow on empty layers list should not crash
    controller.grow_capacity(dummy_model)
    assert len(dummy_model.layers) == 0


# ── 7. Config Integrity Test ──────────────────────────────────────────────────

def test_config_save_and_load_integrity():
    cfg = NeuroCoreConfig.small()
    cfg.block.num_layers = 17
    cfg.moe.num_experts = 12
    
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name
    try:
        cfg.save(path)
        loaded = NeuroCoreConfig.load(path)
        assert loaded.model_name == cfg.model_name
        assert loaded.block.num_layers == 17
        assert loaded.moe.num_experts == 12
    finally:
        if os.path.exists(path):
            os.remove(path)


# ── 8. Red-Teaming & Security Vulnerability Suite ────────────────────────────

def test_redteam_corrupted_and_tampered_dna_payload():
    """Red-team test: verify that corrupted or tampered .dna files are rejected."""
    cfg = CompressionConfig()
    codec = DNACodec(cfg)
    tensor = torch.randn(32, 32, dtype=torch.float32)

    with tempfile.NamedTemporaryFile(suffix=".dna", delete=False) as tmp:
        path = tmp.name
    try:
        codec.compress(tensor, path)

        # 1. Tamper with magic bytes
        with open(path, "r+b") as f:
            f.seek(0)
            f.write(b"HACK")
        with pytest.raises(ValueError, match="Invalid magic bytes"):
            codec.decompress(path)

        # 2. Re-compress and tamper with data stream to test parity failure
        codec.compress(tensor, path)
        with open(path, "r+b") as f:
            f.seek(64)  # inside header/data payload
            original_byte = f.read(1)
            tampered_byte = bytes([original_byte[0] ^ 0xFF])
            f.seek(64)
            f.write(tampered_byte)

        with pytest.raises(ValueError, match="DNA parity check failed|Invalid magic"):
            codec.decompress(path)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_redteam_tool_router_path_traversal():
    """Red-team test: verify that directory traversal attacks in tool_router are blocked."""
    from Tantra.tool_router import read_local_file

    with tempfile.TemporaryDirectory() as sandbox_dir:
        # 1. Directory traversal via ../
        res_traversal = read_local_file("../../../../etc/passwd", repo_root=sandbox_dir)
        assert "Access denied" in res_traversal

        # 2. Absolute path attack
        res_abs = read_local_file("/etc/shadow", repo_root=sandbox_dir)
        assert "Access denied" in res_abs

        # 3. Windows-style absolute path attack
        res_win = read_local_file("C:\\Windows\\System32\\drivers\\etc\\hosts", repo_root=sandbox_dir)
        assert "Access denied" in res_win


def test_redteam_safe_math_eval_injection():
    """Red-team test: ensure malicious Python injection attempts in math eval fail."""
    from Tantra.tool_router import safe_eval_math

    malicious_payloads = [
        "__import__('os').system('echo pwned')",
        "__builtins__.__import__('subprocess').call(['ls'])",
        "eval('2 + 2')",
        "exec('a = 1')",
        "open('/etc/passwd').read()",
        "lambda x: x + 1",
    ]
    for payload in malicious_payloads:
        result = safe_eval_math(payload)
        assert "Error" in result or "Unsupported" in result


def test_redteam_tool_sandbox_disabled_enforcement():
    """Red-team test: verify disabled sandbox strictly denies execution."""
    from Tantra.tool_router import execute_tool_call

    # Python execution blocked when sandbox is disabled
    res_py = execute_tool_call("python_executor", {"code": "print('exploit')"}, sandbox_enabled=False)
    assert "disabled" in res_py.lower()

    # File reader blocked when sandbox is disabled
    res_file = execute_tool_call("file_reader", {"filepath": "README.md"}, sandbox_enabled=False)
    assert "disabled" in res_file.lower()


def test_redteam_prompt_injection_tool_tag_spoofing():
    """Red-team test: verify parser resilience against malformed / spoofed tool tags."""
    from Tantra.tool_router import parse_and_execute_tool_calls

    # 1. Non-JSON inside tool tag (gracefully ignored without crashing)
    updated, matched = parse_and_execute_tool_calls("<tool_call> NOT_A_JSON </tool_call>", sandbox_enabled=False)
    assert isinstance(updated, str)
    assert matched is False

    # 2. Unknown or dangerous tool invocation (safely handled with error result)
    updated, matched = parse_and_execute_tool_calls(
        "<tool_call>{\"name\": \"system_shutdown\", \"arguments\": {}}</tool_call>",
        sandbox_enabled=False
    )
    assert matched is True
    assert "Unknown tool" in updated

    # 3. Invalid / malicious calculation (handles syntax / bounds error safely)
    updated, matched = parse_and_execute_tool_calls(
        "<tool_call>{\"name\": \"calculator\", \"arguments\": {\"expression\": \"__import__('os')\"}}</tool_call>",
        sandbox_enabled=False
    )
    assert matched is True
    assert "<tool_result>" in updated
    assert "Error" in updated or "Unsupported" in updated



def test_redteam_checkpoint_safe_loading_weights_only():
    """Red-team test: ensure checkpoints can be safely deserialized with weights_only."""
    cfg = NeuroCoreConfig.small()
    cfg.block.num_layers = 1
    cfg.block.alra.dim = 32
    cfg.block.alra.num_heads = 4
    cfg.block.alra.head_dim = 8
    cfg.block.sgp.dim = 32
    model = NeuroCoreModel(cfg)

    state = {
        "model_state": model.state_dict(),
        "step": 100,
        "loss": 4.5,
    }

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        ckpt_path = tmp.name
    try:
        torch.save(state, ckpt_path)
        # Verify safe weights_only loading does not fail for standard primitives
        loaded = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        assert "model_state" in loaded
        assert loaded["step"] == 100
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)



# ─────────────────────────────────────────────────────────────────
# Source: test_server_api.py
# ─────────────────────────────────────────────────────────────────

import os
import sys
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from webui.server import app, TANTRA_API_KEY

client = TestClient(app)


def test_get_dashboard():
    """GET / should render index.html with status 200."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Tantra" in resp.text


def test_list_models():
    """GET /v1/models should return model list."""
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    assert data["data"][0]["id"] == "tantra-neurocore-v1"


def test_capabilities_reports_sandbox_availability():
    resp = client.get("/api/capabilities")
    assert resp.status_code == 200
    assert resp.json()["sandbox_enabled"] is False


def test_telemetry_and_status():
    """GET /api/telemetry and /api/status should return dynamic hardware & model metrics."""
    resp = client.get("/api/telemetry")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert "hardware" in data
    assert "cpu_threads" in data["hardware"]
    assert "simd" in data["hardware"]

    resp_status = client.get("/api/status")
    assert resp_status.status_code == 200


def test_experts_endpoint():
    """GET /api/experts should return expert router items and top_k configuration."""
    resp = client.get("/api/experts")
    assert resp.status_code == 200
    data = resp.json()
    assert "experts" in data
    assert len(data["experts"]) > 0
    assert "top_k" in data


def test_knowledge_graph_endpoint():
    """GET /api/knowledge_graph should return dynamic nodes and links topology."""
    resp = client.get("/api/knowledge_graph")
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert "links" in data
    assert len(data["nodes"]) >= 4


def test_chats_crud_operations():
    """Test full CRUD lifecycle of chat sessions."""
    # 1. List chats
    resp_list = client.get("/api/chats")
    assert resp_list.status_code == 200

    # 2. Create new session
    test_id = "test-session-101"
    session_payload = {
        "id": test_id,
        "title": "Test Chat Session",
        "archived": False,
        "messages": [{"role": "user", "content": "Hello Tantra"}]
    }
    resp_create = client.post("/api/chats", json=session_payload)
    assert resp_create.status_code == 200

    # 3. Rename session
    resp_rename = client.post(f"/api/chats/{test_id}/rename", json={"title": "Renamed Session"})
    assert resp_rename.status_code == 200

    # 4. Archive session
    resp_archive = client.post(f"/api/chats/{test_id}/archive")
    assert resp_archive.status_code == 200
    assert resp_archive.json()["archived"] is True

    # 5. Auto-title
    resp_title = client.post("/api/chats/auto_title", json={"prompt": "Write a Python script to sort an array"})
    assert resp_title.status_code == 200
    assert "title" in resp_title.json()

    # 6. Delete session
    resp_del = client.delete(f"/api/chats/{test_id}")
    assert resp_del.status_code == 200


def test_memory_bank_crud():
    """Test memory bank addition, listing, and deletion."""
    resp_get = client.get("/api/memory")
    assert resp_get.status_code == 200

    resp_add = client.post("/api/memory", json={"category": "User Fact", "fact": "User is building Tantra LLM"})
    assert resp_add.status_code == 200
    mem_item = resp_add.json()["item"]

    resp_del = client.delete(f"/api/memory/{mem_item['id']}")
    assert resp_del.status_code == 200


def test_tokenize_endpoint():
    """Test text tokenization endpoint."""
    resp = client.post("/api/tokenize", json={"text": "Tantra Quantum MoE Engine"})
    assert resp.status_code == 200
    data = resp.json()
    assert "tokens" in data
    assert data["total_count"] > 0


def test_datasets_endpoint():
    """Test listing registered datasets."""
    resp = client.get("/api/datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


def test_admin_security_and_disabled_sandbox():
    """Admin endpoints require the local API key and code execution stays off."""
    assert client.post("/api/sandbox/run", json={"code": "print('test')"}).status_code == 401
    assert client.post("/api/checkpoints", json={"checkpoint": "test.pt"}).status_code == 401
    assert client.post("/api/datasets/clean", json={}).status_code == 401

    headers = {"X-API-Key": TANTRA_API_KEY}
    clean = client.post("/api/datasets/clean", json={}, headers=headers)
    assert clean.status_code == 501
    assert "not available" in clean.json()["detail"]
    sandbox = client.post("/api/sandbox/run", json={"code": "while True: pass"}, headers=headers)
    assert sandbox.status_code == 403
    traversal = client.post("/api/checkpoints", json={"checkpoint": "../../etc/passwd"}, headers=headers)
    assert traversal.status_code == 404


def test_documents_rag_crud():
    """Test document upload, listing, and RAG querying."""
    # 1. List
    resp_list = client.get("/api/documents")
    assert resp_list.status_code == 200
    assert "documents" in resp_list.json()

    # 2. Upload
    doc_payload = {
        "filename": "test_rag_doc.txt",
        "content": "Tantra NeuroCore features 1.58-bit BitNet quantization and ALRA attention."
    }
    resp_upload = client.post("/api/documents/upload", json=doc_payload)
    assert resp_upload.status_code == 200
    assert resp_upload.json()["status"] == "success"

    # 3. Query
    resp_query = client.post("/api/documents/query", json={"query": "BitNet quantization", "top_k": 2})
    assert resp_query.status_code == 200
    assert "result" in resp_query.json()
    assert "test_rag_doc.txt" in resp_query.json()["result"]


# ── 16. Audit Bug Fix Regression Verification Tests ─────────────────────────

def test_alra_attention_scaling_and_normalization():
    """Verify that ALRA attention uses consistent query scaling and numerical stability."""
    from Tantra.config import ALRAConfig
    from Tantra.model import ALRAAttention

    cfg = ALRAConfig(dim=64, num_heads=4, head_dim=16)
    attn = ALRAAttention(cfg)
    attn.eval()

    # Short sequence (vectorized fast path)
    x_short = torch.randn(2, 32, 64)
    out_fast, state = attn(x_short)
    assert out_fast.shape == (2, 32, 64)
    assert not torch.isnan(out_fast).any()
    assert not torch.isinf(out_fast).any()

    # Single-token step (sequential path)
    x_single = torch.randn(2, 1, 64)
    out_seq, new_state = attn(x_single, state=None)
    assert out_seq.shape == (2, 1, 64)
    assert not torch.isnan(out_seq).any()
    assert not torch.isinf(out_seq).any()


def _make_test_cfg():
    cfg = NeuroCoreConfig(model_name="test-tiny")
    cfg.block.alra.dim = 64
    cfg.block.alra.num_heads = 4
    cfg.block.alra.head_dim = 16
    cfg.block.sgp.dim = 64
    cfg.block.num_layers = 2
    cfg.vocab.vocab_size = 256
    return cfg


def test_refresh_optimizer_preserves_momentum():
    """Verify refresh_optimizer adds new parameters without wiping existing momentum."""
    cfg = _make_test_cfg()
    model = NeuroCoreModel(cfg)
    trainer = NeuroTrainer(model, lr=1e-3, total_steps=100)

    # Perform a train step to build optimizer momentum state
    x = torch.randint(0, 100, (2, 8))
    y = torch.randint(0, 100, (2, 8))
    trainer.train_step(x, y)

    first_param = next(model.parameters())
    assert first_param in trainer.optimizer.state
    assert "exp_avg" in trainer.optimizer.state[first_param]
    old_momentum = trainer.optimizer.state[first_param]["exp_avg"].clone()

    # Dynamically append a new layer
    import copy
    new_layer = copy.deepcopy(model.layers[-1])
    model.layers.append(new_layer)
    trainer.refresh_optimizer()

    # Verify old momentum is strictly preserved
    assert first_param in trainer.optimizer.state
    assert torch.equal(trainer.optimizer.state[first_param]["exp_avg"], old_momentum)


def test_export_clean_checkpoint():
    """Verify export_clean_checkpoint preserves metadata and strips optimizer."""
    from Tantra.export import export_clean_checkpoint

    cfg = _make_test_cfg()
    model = NeuroCoreModel(cfg)
    trainer = NeuroTrainer(model, lr=1e-3)
    trainer.step_count = 1234
    trainer.best_loss = 2.45

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_ckpt = os.path.join(tmpdir, "raw.pt")
        clean_ckpt = os.path.join(tmpdir, "clean.pt")

        trainer.save_checkpoint(raw_ckpt, save_optimizer=True)
        export_clean_checkpoint(raw_ckpt, clean_ckpt)

        loaded = torch.load(clean_ckpt, map_location="cpu", weights_only=False)
        assert loaded["step_count"] == 1234
        assert loaded["best_loss"] == 2.45
        assert "model_state_dict" in loaded
        assert "optimizer_state_dict" not in loaded



