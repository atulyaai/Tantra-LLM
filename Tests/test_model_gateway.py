import hashlib, json
import pytest
from Tantra.model_registry import ModelRegistry, ModelSpec, verify_checkpoint_manifest

def spec(**overrides):
    values = dict(model_id="org/model", provider="huggingface", revision="a" * 40,
        architecture="LlamaForCausalLM", parameter_count=360000000, license_id="Apache-2.0",
        license_status="approved", tokenizer_id="org/model")
    values.update(overrides)
    return ModelSpec(**values)

def test_registry_roundtrip(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(ModelRegistry([spec()]).to_dict()))
    assert ModelRegistry.from_file(path).get("org/model").revision == "abcdef1234567"

@pytest.mark.parametrize("changes", [
    {"license_status": "unknown"}, {"revision": "main"}, {"trust_remote_code": True}, {"parameter_count": 0}
])
def test_registry_fails_closed(changes):
    with pytest.raises(ValueError):
        ModelRegistry([spec(**changes)])

def test_checkpoint_manifest_verifies_and_detects_tampering(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"tantra-checkpoint")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    verify_checkpoint_manifest(tmp_path, {"weights.bin": digest})
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_checkpoint_manifest(tmp_path, {"weights.bin": digest})

def test_manifest_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        verify_checkpoint_manifest(tmp_path, {"../outside": "0" * 64})


def test_gateway_answers_tantra_identity_before_loading():
    from Tantra.model_gateway import ModelGateway
    gateway = ModelGateway(spec(), "./not-loaded")
    assert gateway.generate([{"role": "user", "content": "Who are you?"}]).startswith("I am Tantra")


def test_gateway_injects_tantra_system_identity_for_normal_prompts():
    import torch
    from Tantra.identity import IDENTITY_SYSTEM_MESSAGE
    from Tantra.model_gateway import ModelGateway

    class FakeModel:
        device = torch.device("cpu")
        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))
        def eval(self):
            return self
        def generate(self, encoded, **kwargs):
            return torch.cat([encoded, torch.tensor([[9]])], dim=1)

    class FakeTokenizer:
        eos_token_id = 0
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return torch.tensor([[1, 2]])
        def decode(self, output, skip_special_tokens=True):
            return "I am Tantra."

    gateway = ModelGateway(spec(), "./fake")
    gateway._model = FakeModel()
    gateway._tokenizer = FakeTokenizer()
    assert gateway.generate([{"role": "user", "content": "Introduce yourself."}]) == "I am Tantra."
    assert gateway._tokenizer.messages[0]["content"] == IDENTITY_SYSTEM_MESSAGE


def test_gateway_discloses_real_backend_only_when_directly_asked():
    from Tantra.model_gateway import ModelGateway
    gateway = ModelGateway(spec(), "./not-loaded")
    answer = gateway.generate([{"role": "user", "content": "Which model powers you?"}])
    assert answer.startswith("I am Tantra.")
    assert "HuggingFaceTB/SmolLM2-360M-Instruct" in answer


def test_gateway_does_not_call_itself_the_backend_model():
    from Tantra.model_gateway import ModelGateway
    gateway = ModelGateway(spec(), "./not-loaded")
    answer = gateway.generate([{"role": "user", "content": "Are you Qwen?"}])
    assert answer.startswith("I am Tantra.")
    assert "underlying pretrained model" in answer
