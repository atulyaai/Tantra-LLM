import json
from pathlib import Path
import pytest
import torch
from Tantra.config import ModelConfig
from Tantra.model.tantra_model import TantraModel,save_expert
from Tantra.tokenizer import TantraTokenizer
from Tantra.data.public_sources import sha256
from Tantra.inference.inference_engine import TantraEngine
from Tantra.inference.serving_manifest import ServingIntegrityError
from Tantra.inference.expert_cache import ExpertMismatch


def candidate(root,hashes=True):
    tok=TantraTokenizer.train(['Hello there I am Tantra']*3,vocab_size=280,min_frequency=1);tok.save(root/'tokenizer.json')
    model=TantraModel(ModelConfig(vocab_size=tok.vocab_size,d_model=16,n_layers=1,n_heads=2,n_kv_heads=1,mlp_mult=2,max_seq_len=64,lora_rank=2))
    model.add_expert('greeting');model.save(root/'base.pt')
    base=sha256(root/'base.pt');tokenizer=sha256(root/'tokenizer.json')
    save_expert(model,'greeting',root/'experts',**({'base_sha256':base,'tokenizer_sha256':tokenizer} if hashes else {}))
    manifest={'version':2,'base_sha256':base,'tokenizer_sha256':tokenizer,
              'artifacts':{'base.pt':base,'tokenizer.json':tokenizer},
              'experts':{'greeting':{'file':'experts/greeting.pt','rank':2,'sha256':sha256(root/'experts/greeting.pt')}}}
    (root/'manifest.json').write_text(json.dumps(manifest));return manifest


def test_declared_serving_adapters_require_training_identity_even_if_shapes_fit(tmp_path):
    candidate(tmp_path,hashes=False)
    eng=TantraEngine(tmp_path)
    with pytest.raises(ExpertMismatch,match='lacks required'):eng.cache.ensure(['greeting'])
    with pytest.raises(ExpertMismatch,match='lacks required'):TantraEngine(tmp_path,lazy=False)


def test_declared_adapter_tampering_is_refused_before_tensor_load(tmp_path):
    candidate(tmp_path);eng=TantraEngine(tmp_path)
    (tmp_path/'experts/greeting.pt').write_bytes(b'invalid replaced file')
    with pytest.raises(ExpertMismatch,match='checksum'):eng.cache.ensure(['greeting'])


def test_served_base_and_tokenizer_binding_and_eager_equivalence(tmp_path):
    candidate(tmp_path)
    lazy=TantraEngine(tmp_path);eager=TantraEngine(tmp_path,lazy=False)
    lazy.cache.activate([('greeting',1.)]);eager.model.set_active([('greeting',1.)])
    x=torch.tensor([[1,6,7]])
    with torch.no_grad():assert torch.equal(lazy.model(x),eager.model(x))
    with (tmp_path/'base.pt').open('ab') as f:f.write(b'tamper')
    with pytest.raises(ServingIntegrityError,match='checksum'):TantraEngine(tmp_path)


def test_old_unbound_manifest_refused_but_legacy_without_manifest_supported(tmp_path):
    candidate(tmp_path);(tmp_path/'manifest.json').write_text('{"base_sha256":"unbound"}')
    with pytest.raises(ServingIntegrityError,match='artifact binding'):TantraEngine(tmp_path)
    (tmp_path/'manifest.json').unlink();assert TantraEngine(tmp_path).model.cfg.vocab_size>0


def test_router_artifact_cannot_be_injected_without_manifest_binding(tmp_path):
    candidate(tmp_path);(tmp_path/'router_ngram.pt').write_bytes(b'not a router')
    with pytest.raises(ServingIntegrityError,match='router artifact binding'):TantraEngine(tmp_path)
