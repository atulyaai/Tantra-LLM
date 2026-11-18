import json
import pytest
import torch
from torch import nn
from Tantra.pretrained_expansion import add_residual_branch,count_parameters,generate_teacher_dataset,load_residual_candidate,save_residual_candidate,transfer_adapter
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


def test_replay_mix_is_deterministic_and_preserves_new_examples():
    from Tantra.pretrained_expansion import mix_replay_examples
    new=["new-a","new-b","new-c","new-d"]
    replay=["old-a","old-b"]
    first,count=mix_replay_examples(new,replay,replay_ratio=.25,seed=7)
    second,count2=mix_replay_examples(new,replay,replay_ratio=.25,seed=7)
    assert first==second
    assert count==count2==1
    assert set(new).issubset(set(first))
    assert any(item in replay for item in first)


def test_replay_mix_rejects_invalid_ratio():
    from Tantra.pretrained_expansion import mix_replay_examples
    with pytest.raises(ValueError,match="replay_ratio"):
        mix_replay_examples(["new"],["old"],replay_ratio=1.0)


def test_residual_branch_supports_bfloat16_foundation_weights():
    torch.manual_seed(29)
    model=Tiny().to(dtype=torch.bfloat16).eval()
    add_residual_branch(model,-1,4)
    value=torch.randn(1,3,12,dtype=torch.bfloat16)
    output=model(value)
    assert output.dtype==torch.bfloat16
    assert torch.isfinite(output).all()


def test_teacher_distillation_writes_reviewable_manifest(tmp_path):
    class Teacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor=nn.Parameter(torch.zeros(1))
        def generate(self, input_ids, **kwargs):
            return torch.cat([input_ids,torch.tensor([[9]],device=input_ids.device)],dim=1)
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            return torch.tensor([[1,2]])
        def decode(self,ids,skip_special_tokens=True):
            return "teacher answer"
    output=tmp_path/"teacher.jsonl"
    report=generate_teacher_dataset(Teacher(),Tokenizer(),["prompt one"],output)
    assert report["examples"]==1 and report["review_required"] is True
    assert "teacher answer" in output.read_text()
    manifest=json.loads((tmp_path/"teacher.jsonl.manifest.json").read_text())
    assert manifest["dataset_sha256"]==report["dataset_sha256"]


def test_compatible_weight_inheritance_copies_only_matching_tensors():
    from Tantra.pretrained_expansion import transfer_compatible_weights
    parent=Tiny(); child=Tiny()
    with torch.no_grad():
        for parameter in parent.parameters(): parameter.fill_(0.25)
        for parameter in child.parameters(): parameter.zero_()
    parent_before={k:v.clone() for k,v in parent.state_dict().items()}
    result=transfer_compatible_weights(parent,child)
    assert result["copied_tensors"]>0
    for key,value in child.state_dict().items():
        assert torch.equal(value,parent_before[key])
    for key,value in parent.state_dict().items():
        assert torch.equal(value,parent_before[key])


def test_feature_alignment_loss_backpropagates_to_student_and_projection():
    from Tantra.pretrained_expansion import feature_alignment_loss
    teacher=torch.randn(2,5,8)
    student=torch.randn(2,5,6,requires_grad=True)
    projection=nn.Linear(8,6)
    loss=feature_alignment_loss(teacher,student,projection)
    loss.backward()
    assert student.grad is not None and projection.weight.grad is not None
    assert torch.isfinite(loss)


def test_teacher_dataset_requires_human_approval_and_hash_stays_bound(tmp_path):
    from Tantra.pretrained_expansion import approve_teacher_dataset,read_text_records
    dataset=tmp_path/"teacher.jsonl"
    dataset.write_text('{"text":"teacher generated answer"}\n')
    manifest={"dataset_sha256":__import__("hashlib").sha256(dataset.read_bytes()).hexdigest(),"review_required":True}
    dataset.with_suffix(".jsonl.manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="reviewed and approved"):
        read_text_records(dataset)
    approve_teacher_dataset(dataset,"reviewer","Checked against the source prompt and corrected errors.")
    assert read_text_records(dataset)==["teacher generated answer"]
    dataset.write_text('{"text":"changed after review"}\n')
    with pytest.raises(ValueError,match="changed after its review"):
        read_text_records(dataset)


def test_teacher_dataset_approval_requires_reviewer_and_note(tmp_path):
    from Tantra.pretrained_expansion import approve_teacher_dataset
    dataset=tmp_path/"teacher.jsonl"; dataset.write_text('{"text":"answer"}\n')
    dataset.with_suffix(".jsonl.manifest.json").write_text(json.dumps({
        "dataset_sha256":__import__("hashlib").sha256(dataset.read_bytes()).hexdigest(),"review_required":True
    }))
    with pytest.raises(ValueError,match="reviewer identity"):
        approve_teacher_dataset(dataset,"","")
