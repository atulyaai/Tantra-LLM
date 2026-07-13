import asyncio

from npdna.adapters.local.npdna_adapter import NpDnaAdapter
from npdna.atulya_core.schema.models import Message, ModelProvider, TantraRequest
from npdna.core.inference import UnifiedInferenceHub


class FakeNpDnaCore:
    def generate(self, prompt, **kwargs):
        return f"NP-DNA: {prompt}"

    def encode(self, text, allow_growth=False):
        return text.split()


def test_npdna_runs_through_fusion_hub():
    adapter = NpDnaAdapter(core=FakeNpDnaCore())
    hub = UnifiedInferenceHub()
    hub.register_adapter(ModelProvider.LOCAL, adapter)
    request = TantraRequest(
        messages=[Message(role="user", content="Explain gravity")],
        provider=ModelProvider.LOCAL,
    )

    response = asyncio.run(hub.execute(request))

    assert response.provider is ModelProvider.LOCAL
    assert response.model == "npdna-injected"
    assert "Explain gravity" in response.content
    assert response.usage["prompt_tokens"] == 2
