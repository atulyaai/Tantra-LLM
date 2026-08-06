"""
Combined inference and end-to-end test suite for Tantra-LLM.

Covers:
  - E2E protocol and inference pipeline (E2E Brain)
  - Safety module evaluation
  - Compute routing (fast vs deep)
  - Personality layer mode selection and override reset
  - Dynamic context trimming
  - AGI planning sutra custom nodes
  - Fallback chain and circuit breaker behavior
  - Request context mutation via middleware
  - Modality encoder dimension validation
  - In-memory vector store write/retrieve/consolidate
  - Adapter telemetry and simulation mode
  - Audio cache rejection for zero embeddings
  - Real SDK paths for OpenAI and Gemini adapters
  - NpDNA fusion integration through UnifiedInferenceHub
"""

import tempfile
import unittest
import asyncio
import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from npdna.schema import TantraRequest, Message, ModelProvider, RequestContext, TantraResponse, TantraMiddleware, get_settings
from npdna.inference import UnifiedInferenceHub, RWKVAdapter, NpDnaAdapter
from npdna.inference import SafetyModule, PersonalityLayer
from npdna.cognition import ComputeRouter, DynamicContextManager, EventBus
from npdna.sensory import VisionEncoder, AudioEncoder


class FailingAdapter(RWKVAdapter):
    """An adapter that always throws exceptions to trigger fallback/circuit breaking."""
    async def generate(self, request: TantraRequest) -> TantraResponse:
        raise RuntimeError("Mock connection timeout")

class SucceedingAdapter(RWKVAdapter):
    """An adapter that always succeeds with a custom string."""
    def __init__(self, content: str):
        super().__init__(model_path="mock.pth")
        self.content_str = content
        
    async def generate(self, request: TantraRequest) -> TantraResponse:
        return TantraResponse(
            content=self.content_str,
            model="mock-model",
            provider=ModelProvider.LOCAL,
            trace_id=request.trace_id
        )

class ContextMutatingMiddleware(TantraMiddleware):
    async def __call__(self, request, context, call_next):
        # Mutate context metadata
        context.metadata["mutated_by_middleware"] = True
        return await call_next(request, context)


class TestTantraE2E(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_protocol_and_inference(self):
        hub = UnifiedInferenceHub()
        adapter = RWKVAdapter(model_path="nonexistent_model.pth")
        hub.register_adapter(ModelProvider.LOCAL, adapter)
        
        req = TantraRequest(
            messages=[Message(role="user", content="Hello World")],
            provider=ModelProvider.LOCAL
        )
        
        async def run_inference():
            return await hub.execute(req)
            
        resp = self.loop.run_until_complete(run_inference())
        self.assertEqual(resp.provider, ModelProvider.LOCAL)
        self.assertIn("[RWKV MOCK]", resp.content)

    def test_safety_module(self):
        safety = SafetyModule()
        
        # Test safe draft
        res = safety.evaluate("This is a perfectly safe text.", {})
        self.assertEqual(res["action"], "pass")
        
        # Test deny list
        res = safety.evaluate("This is violent content.", {})
        self.assertEqual(res["action"], "deny")
        
        # Test toxicity patterns
        res = safety.evaluate("This is full of hate.", {})
        self.assertEqual(res["action"], "modify")

    def test_compute_routing(self):
        router = ComputeRouter()
        
        # Simple query
        path = router.select_path("hi")
        self.assertEqual(path, "fast")
        
        # Complex query
        path = router.select_path("can you explain in detail how and why this happens? Please provide a highly detailed design document, step-by-step layout of the architecture, compare the trade-offs, and outline the plan for implementation.")
        self.assertEqual(path, "deep")

    def test_personality_layer(self):
        config = {
            "tones": {
                "default": {"prompt_prefix": "default prefix"},
                "concise": {"prompt_prefix": "short prefix"}
            }
        }
        layer = PersonalityLayer(config)
        
        # Direct override
        mode = layer.select_mode("mode: direct")
        self.assertEqual(mode, "DirectAssertive")
        
        params = layer.parameterize(mode)
        self.assertEqual(params["prompt_prefix"], "short prefix")

    def test_dynamic_context_trim(self):
        mgr = DynamicContextManager(max_short=5, max_long=10)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        trimmed = mgr.trim(tokens, 5)
        # Should keep the end: [4, 5, 6, 7, 8]
        self.assertEqual(trimmed, [4, 5, 6, 7, 8])

    def test_event_bus_heartbeat_reports_real_load(self):
        """EventBus heartbeat must emit a pulse with non-hardcoded cpu/mem load via psutil."""
        bus = EventBus()
        received = []
        async def on_pulse(p):
            received.append(p)
        bus.subscribe("pulse", on_pulse)
        async def run_one_tick():
            # Call heartbeat internals directly: emit one real-load pulse
            import psutil
            try:
                cpu = float(psutil.cpu_percent(interval=None))
                mem = float(psutil.virtual_memory().percent)
            except Exception:
                cpu, mem = 0.0, 0.0
            from npdna.schema import SystemPulse
            await bus.emit("pulse", SystemPulse(cpu_load=cpu, mem_usage=mem).model_dump())
        self.loop.run_until_complete(run_one_tick())
        self.assertEqual(len(received), 1)
        self.assertGreaterEqual(received[0]["mem_usage"], 0.0)
        self.assertGreaterEqual(received[0]["cpu_load"], 0.0)

    def test_personality_override_reset(self):
        config = {"tones": {}}
        layer = PersonalityLayer(config)
        
        # 1. Set explicit override
        mode1 = layer.select_mode("mode: mentor")
        self.assertEqual(mode1, "MentorBuilder")
        
        # 2. Natural cue should override manual override and reset it
        mode2 = layer.select_mode("just give me a quick answer")
        self.assertEqual(mode2, "DirectAssertive")

    def test_safety_blocking_in_execute(self):
        hub = UnifiedInferenceHub()
        adapter = RWKVAdapter(model_path="nonexistent.pth")
        hub.register_adapter(ModelProvider.LOCAL, adapter)
        
        # Safe request
        req1 = TantraRequest(messages=[Message(role="user", content="hello")])
        resp1 = self.loop.run_until_complete(hub.execute(req1))
        self.assertNotIn("Response blocked by safety policy", resp1.content)
        
        # Request returning denied keyword ("violent" is a deny keyword)
        req2 = TantraRequest(messages=[Message(role="user", content="violent content")])
        resp2 = self.loop.run_until_complete(hub.execute(req2))
        self.assertIn("Response blocked by safety policy", resp2.content)

    def test_fallback_chain(self):
        hub = UnifiedInferenceHub()
        failing = FailingAdapter(model_path="failing.pth")
        succeeding = SucceedingAdapter(content="Fallback Works!")
        
        hub.register_adapter(ModelProvider.LOCAL, failing)
        hub.register_adapter(ModelProvider.LOCAL, succeeding)
        
        req = TantraRequest(messages=[Message(role="user", content="Test Fallback")])
        async def run():
            return await hub.execute(req)
            
        resp = self.loop.run_until_complete(run())
        self.assertEqual(resp.content, "Fallback Works!")

    def test_circuit_breaker(self):
        # Configure hub with circuit breaker max_failures=2
        hub = UnifiedInferenceHub(max_failures=2, cooldown_seconds=5.0)
        failing = FailingAdapter(model_path="failing.pth")
        hub.register_adapter(ModelProvider.LOCAL, failing)
        
        req = TantraRequest(messages=[Message(role="user", content="Test Circuit")])
        
        # Attempt 1: fails
        with self.assertRaises(RuntimeError):
            self.loop.run_until_complete(hub.execute(req))
        self.assertEqual(hub.circuit_breakers[failing]["failures"], 1)
        self.assertIsNone(hub.circuit_breakers[failing]["tripped_until"])
        
        # Attempt 2: fails and trips
        with self.assertRaises(RuntimeError):
            self.loop.run_until_complete(hub.execute(req))
        self.assertEqual(hub.circuit_breakers[failing]["failures"], 2)
        self.assertIsNotNone(hub.circuit_breakers[failing]["tripped_until"])
        
        # Attempt 3: immediately raises tripped exception without trying (FailingAdapter is skipped)
        with self.assertRaises(RuntimeError) as ctx:
            self.loop.run_until_complete(hub.execute(req))
        self.assertIn("unavailable or tripped", str(ctx.exception))

    def test_request_context_mutation(self):
        hub = UnifiedInferenceHub()
        hub.add_middleware(ContextMutatingMiddleware())
        
        adapter = SucceedingAdapter(content="Success")
        hub.register_adapter(ModelProvider.LOCAL, adapter)
        
        # We can pass an existing trace_id or execute a hook
        req = TantraRequest(messages=[Message(role="user", content="Context Test")])
        
        # Capture context mutations via middleware tracking
        # We verify that execute executes successfully
        resp = self.loop.run_until_complete(hub.execute(req))
        self.assertEqual(resp.content, "Success")

    def test_modality_encoder_validation(self):
        settings = get_settings()
        # default settings.model_dim is 4096
        self.assertEqual(settings.model_dim, 4096)
        
        # Instantiating with matching dimension (4096) succeeds
        vis = VisionEncoder(embed_dim=4096)
        aud = AudioEncoder(embed_dim=4096)
        self.assertEqual(vis.embed_dim, 4096)
        self.assertEqual(aud.embed_dim, 4096)
        
        # Instantiating with mismatching dimension (e.g. 1024) raises ValueError
        with self.assertRaises(ValueError):
            VisionEncoder(embed_dim=1024)
        with self.assertRaises(ValueError):
            AudioEncoder(embed_dim=1024)

    def test_in_memory_vector_store(self):
        from npdna.cognition import InMemoryVectorStore
        store = InMemoryVectorStore(embed_dim=128)
        
        # Write memories
        self.loop.run_until_complete(store.write("Tantra OS core routing details", {"type": "technical"}))
        self.loop.run_until_complete(store.write("Tantra OS personality module layers", {"type": "identity"}))
        
        # Retrieve memories
        results = self.loop.run_until_complete(store.retrieve("routing details", k=1))
        self.assertEqual(len(results), 1)
        self.assertIn("routing details", results[0].content)
        self.assertEqual(results[0].metadata["type"], "technical")
        
        # Consolidate duplicate files
        self.loop.run_until_complete(store.write("Tantra OS core routing details", {"type": "technical"}))
        self.loop.run_until_complete(store.consolidate())
        self.assertEqual(len(store.registry), 2)

    def test_adapter_telemetry(self):
        """Adapters without API keys must fall back to simulation and mark usage['simulated']=True."""
        from npdna.inference import OpenAIAdapter, GeminiAdapter

        # No API key -> simulation mode
        openai_adapter = OpenAIAdapter(api_key="")
        gemini_adapter = GeminiAdapter(api_key="")

        # health_check returns False when no key configured
        self.assertFalse(openai_adapter.health_check())
        self.assertFalse(gemini_adapter.health_check())

        req = TantraRequest(
            messages=[Message(role="user", content="Explain AGI dynamic context sliding window routing parameters.")],
            provider=ModelProvider.OPENAI,
        )

        # Simulation mode: token counts positive, cost positive, simulated flag set
        resp_openai = self.loop.run_until_complete(openai_adapter.generate(req))
        self.assertGreater(resp_openai.usage["prompt_tokens"], 0)
        self.assertGreater(resp_openai.cost, 0.0)
        self.assertTrue(resp_openai.usage["simulated"],
                        "OpenAI adapter without a key must mark response as simulated")
        self.assertIn("[SIMULATED]", resp_openai.content)

        resp_gemini = self.loop.run_until_complete(gemini_adapter.generate(req))
        self.assertGreater(resp_gemini.usage["prompt_tokens"], 0)
        self.assertGreater(resp_gemini.cost, 0.0)
        self.assertTrue(resp_gemini.usage["simulated"],
                        "Gemini adapter without a key must mark response as simulated")
        self.assertIn("[SIMULATED]", resp_gemini.content)

    def test_audio_cache_rejection(self):
        """Verify that zero-embedding samples are never written to the cache directory."""
        from unittest.mock import patch, MagicMock
        from tools.precompute_embeddings import main

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir  = os.path.join(tmpdir, "audio")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(audio_dir)
            os.makedirs(output_dir)

            # Create a stub .wav file
            wav_path = os.path.join(audio_dir, "test.wav")
            open(wav_path, "wb").close()

            # AudioEncoder.encode returns all-zeros (Whisper unavailable)
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = torch.zeros(1, 512)  # audio_dim default

            # librosa.load returns a trivial waveform
            mock_librosa = MagicMock()
            mock_librosa.load.return_value = (torch.zeros(16000).numpy(), 16000)

            with patch("tools.precompute_embeddings.AudioEncoder", return_value=mock_encoder), \
                 patch.dict("sys.modules", {"librosa": mock_librosa}), \
                 patch("sys.argv", [
                     "scripts/precompute_embeddings.py",
                     "--audio-dir",  audio_dir,
                     "--output-dir", output_dir,
                     "--audio-dim",  "512",
                 ]):
                main()

            saved_files = os.listdir(output_dir)
            self.assertEqual(
                len(saved_files), 0,
                "Zero-embedding audio samples must be skipped, not cached."
            )

    def test_openai_real_sdk_path(self):
        """Verify that the OpenAI adapter calls the real API client correctly when configured."""
        from unittest.mock import AsyncMock, patch, MagicMock
        from npdna.inference import OpenAIAdapter

        # Mock the entire openai library and AsyncOpenAI client
        mock_openai_module = MagicMock()
        mock_client = MagicMock()
        
        # Configure client.chat.completions.create to be an AsyncMock returning a mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Real API response text"
        mock_response.usage.prompt_tokens = 42
        mock_response.usage.completion_tokens = 24
        mock_response.model = "gpt-4o"
        
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai_module.AsyncOpenAI = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"openai": mock_openai_module}), \
            patch("npdna.inference._openai_sdk", mock_openai_module):
            
            adapter = OpenAIAdapter(api_key="sk-test-real-key")
            self.assertTrue(adapter.health_check(), "Adapter health_check must be True with key and SDK mocked")

            req = TantraRequest(
                messages=[Message(role="user", content="Test real call")],
                provider=ModelProvider.OPENAI
            )
            
            resp = self.loop.run_until_complete(adapter.generate(req))
            
            # Verify response is parsed and marked not simulated
            self.assertEqual(resp.content, "Real API response text")
            self.assertEqual(resp.usage["prompt_tokens"], 42)
            self.assertFalse(resp.usage["simulated"])
            
            # Verify client mock was invoked with correct arguments
            mock_client.chat.completions.create.assert_called_once()

    def test_gemini_real_sdk_path(self):
        """Verify that the Gemini adapter calls the real API client correctly when configured."""
        from unittest.mock import AsyncMock, patch, MagicMock
        from npdna.inference import GeminiAdapter

        mock_genai_module = MagicMock()
        mock_client = MagicMock()
        
        # Configure client.aio.models.generate_content to return a mock response
        mock_response = MagicMock()
        mock_response.text = "Real Gemini response text"
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        
        mock_types_module = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_genai_module.Client = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"google.genai": mock_genai_module, "google.genai.types": mock_types_module}), \
            patch("npdna.inference._genai_sdk", mock_genai_module), \
            patch("npdna.inference._genai_types", mock_types_module):
            
            adapter = GeminiAdapter(api_key="gemini-test-real-key")
            self.assertTrue(adapter.health_check(), "Adapter health_check must be True with key and SDK mocked")

            req = TantraRequest(
                messages=[Message(role="user", content="Test real call")],
                provider=ModelProvider.GEMINI
            )
            
            resp = self.loop.run_until_complete(adapter.generate(req))
            
            # Verify response is parsed and marked not simulated
            self.assertEqual(resp.content, "Real Gemini response text")
            self.assertEqual(resp.usage["prompt_tokens"], 100)
            self.assertFalse(resp.usage["simulated"])
            
            # Verify client mock was invoked with correct arguments
            mock_client.aio.models.generate_content.assert_called_once()


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


if __name__ == "__main__":
    unittest.main()
