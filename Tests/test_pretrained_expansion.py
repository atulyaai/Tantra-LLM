import json
import pytest
import torch
from torch import nn
from Tantra.pretrained_expansion import add_residual_branch,count_parameters,load_residual_candidate,save_residual_candidate,transfer_adapter
from Tantra.model_lifecycle import ModelLifecycle

class Config: hidden_size=12
class Block(nn.Module):
    def __init__(self): super().__init__(); self.proj=nn.Linear(12,12)
    def forward(self,x): return (self.proj(x),)
class Backbone(nn.Module):
    def __init__(self): super().__init__(); self.layers=nn.ModuleList([Block(),Block()])
class Tiny(nn.Module):
    def __init__(self): super().__init__(); self.config=Config(); self.model=Backbone()
    def forward(self,x):
        for layer in self.model.layers: x=layer(x)[0]
        return x

def test_residual_branch_is_function_preserving_then_increases_capacity(tmp_path):
    torch.manual_seed(11); base=Tiny().eval(); grown=Tiny().eval(); grown.load_state_dict(base.state_dict()); x=torch.randn(2,4,12)
    expected=base(x); count=count_parameters(base); add_residual_branch(grown,-1,4)
    assert torch.allclose(expected,grown(x),atol=1e-7,rtol=0); assert count_parameters(grown)>count
    save_residual_candidate(grown,tmp_path,-1,4)
    clone=Tiny().eval(); clone.load_state_dict(base.state_dict()); load_residual_candidate(clone,tmp_path)
    assert torch.allclose(grown(x),clone(x),atol=1e-7,rtol=0)

def test_adapter_inheritance_rejects_wrong_base(tmp_path):
    parent=tmp_path/"parent"; parent.mkdir()
    (parent/"tantra_adapter.json").write_text(json.dumps({"base_sha256":"a"*64}))
    (parent/"adapter_config.json").write_text(json.dumps({"base_model_name_or_path":"x","target_modules":["q_proj"],"r":8,"lora_alpha":16,"peft_type":"LORA"}))
    with pytest.raises(ValueError,match="hash mismatch"): transfer_adapter(parent,tmp_path/"child",expected_base_sha256="b"*64)

def test_adapter_inheritance_copies_parent_artifact(tmp_path):
    parent=tmp_path/"parent"; parent.mkdir()
    (parent/"tantra_adapter.json").write_text(json.dumps({"base_sha256":"a"*64}))
    (parent/"adapter_config.json").write_text(json.dumps({"base_model_name_or_path":"x","target_modules":["q_proj"],"r":8,"lora_alpha":16,"peft_type":"LORA"}))
    (parent/"adapter.bin").write_bytes(b"weights")
    child=tmp_path/"child"; transfer_adapter(parent,child,expected_base_sha256="a"*64)
    assert (child/"adapter.bin").read_bytes()==b"weights"

def _report(**updates):
    value={"tests_passed":True,"identity_passed":True,"safety_passed":True,"heldout_score":.9,"baseline_score":.8,"max_regression":.01,"latency_ms":100,"latency_budget_ms":200}
    value.update(updates); return value

def test_lifecycle_promotes_and_rolls_back(tmp_path):
    lifecycle=ModelLifecycle(tmp_path/"lifecycle.json")
    first=tmp_path/"first"; first.mkdir(); (first/"weights").write_bytes(b"v1")
    lifecycle.register("v1",first,"foundation"); lifecycle.promote("v1",_report())
    second=tmp_path/"second"; second.mkdir(); (second/"weights").write_bytes(b"v2")
    lifecycle.register("v2",second,"v1"); lifecycle.promote("v2",_report(heldout_score=.95))
    assert lifecycle.rollback()=="v1"

@pytest.mark.parametrize("updates",[{"tests_passed":False},{"identity_passed":False},{"safety_passed":False},{"heldout_score":.7},{"max_regression":.03},{"latency_ms":500}])
def test_lifecycle_blocks_bad_candidate(tmp_path,updates):
    lifecycle=ModelLifecycle(tmp_path/"lifecycle.json"); p=tmp_path/"candidate"; p.mkdir(); (p/"weights").write_bytes(b"candidate")
    lifecycle.register("candidate",p,"foundation")
    with pytest.raises(ValueError): lifecycle.promote("candidate",_report(**updates))

def test_lifecycle_detects_artifact_tampering(tmp_path):
    lifecycle=ModelLifecycle(tmp_path/"lifecycle.json"); p=tmp_path/"candidate"; p.mkdir(); (p/"weights").write_bytes(b"original")
    lifecycle.register("candidate",p,"foundation"); (p/"weights").write_bytes(b"tampered")
    with pytest.raises(ValueError,match="changed"): lifecycle.promote("candidate",_report())


def test_residual_branch_parameters_are_trainable_and_base_is_frozen():
    torch.manual_seed(23)
    model=Tiny()
    add_residual_branch(model,-1,4)
    wrapper=model.model.layers[-1]
    assert all(not p.requires_grad for p in wrapper.base.parameters())
    assert any(p.requires_grad for p in wrapper.branch.parameters())
    x=torch.randn(2,4,12)
    optimizer=torch.optim.SGD(wrapper.branch.parameters(),lr=0.1)
    before=wrapper.branch[-1].weight.detach().clone()
    loss=model(x).pow(2).mean()
    loss.backward()
    optimizer.step()
    assert not torch.equal(before,wrapper.branch[-1].weight.detach())
    assert all(p.grad is None for p in wrapper.base.parameters())


def test_adapter_inheritance_rejects_incompatible_peft_config(tmp_path):
    parent=tmp_path/"parent"; parent.mkdir()
    (parent/"tantra_adapter.json").write_text(json.dumps({"base_sha256":"a"*64}))
    (parent/"adapter_config.json").write_text(json.dumps({"base_model_name_or_path":"base-a","target_modules":["q_proj"],"r":8,"lora_alpha":16,"peft_type":"LORA"}))
    child=tmp_path/"child"; child.mkdir()
    (child/"adapter_config.json").write_text(json.dumps({"base_model_name_or_path":"base-b","target_modules":["q_proj"],"r":8,"lora_alpha":16,"peft_type":"LORA"}))
    with pytest.raises(ValueError,match="incompatible"):
        transfer_adapter(parent,child,expected_base_sha256="a"*64)
