import unittest
import asyncio
import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from atulya_core.schema.models import TantraRequest, Message, ModelProvider, RequestContext, TantraResponse
from atulya_core.protocol.middleware import TantraMiddleware
from core.inference import UnifiedInferenceHub
from adapters.local.rwkv_adapter import RWKVAdapter
from personality.safety_module import SafetyModule
from core.compute_routing import ComputeRouter
from personality.personality_layer import PersonalityLayer
from core.dynamic_context import DynamicContextManager
from core.planning import AGIPlanningSutra
from config.settings import get_settings
from encoders.vision import VisionEncoder
from encoders.audio import AudioEncoder

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
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

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

    def test_planning_custom_nodes(self):
        sutra = AGIPlanningSutra()
        async def run_plan():
            plan1 = await sutra.compile_intent("please write some code for me")
            plan2 = await sutra.compile_intent("please debug this error")
            plan3 = await sutra.compile_intent("something else")
            return plan1, plan2, plan3
            
        p1, p2, p3 = self.loop.run_until_complete(run_plan())
        self.assertNotEqual(p1.plan_id, p2.plan_id)
        self.assertEqual(p1.nodes[1].instruction, "Draft technical architecture and dependencies plan")
        self.assertEqual(p2.nodes[0].instruction, "Reproduce the failure, inspect stack traces, and isolate faults")
        self.assertEqual(p3.nodes[0].instruction, "Parse user intent and extract contextual indicators")

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
        from core.memory import InMemoryVectorStore
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
        from adapters.openai.adapter import OpenAIAdapter
        from adapters.gemini.adapter import GeminiAdapter

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

    def test_grad_accumulation_divisor(self):
        """Verify that an incomplete final accumulation group fires the optimizer correctly.

        Layout:
          7 samples, batch_size=2 -> 4 batches
          grad_accum=3 -> Group 1 (batches 0,1,2) + Group 2 (batch 3)
          Expected optimizer steps: exactly 2  (trainer.global_step increments once per step).

        We read trainer.global_step after fit() — it is the canonical counter that
        increments inside the same branch that calls optimizer.step, so it is the
        cleanest way to verify group boundaries without touching the optimizer object.
        """
        from training.fusion_trainer import FusionTrainer, FusionProjector
        from training.training_config import FusionTrainingConfig
        from training.datasets.multimodal_dataset import MultimodalDataset

        config = FusionTrainingConfig(batch_size=2, epochs=1, grad_accum=3,
                                      warmup_steps=0)
        vis_proj = FusionProjector(768, 256)
        aud_proj = FusionProjector(512, 256)
        trainer  = FusionTrainer(config, vis_proj, aud_proj)

        ds = MultimodalDataset.generate_synthetic(
            num_samples=7, vision_dim=768, audio_dim=512, target_dim=256
        )
        history = trainer.fit(ds)

        # 4 batches / grad_accum=3 -> 2 optimizer steps -> global_step == 2
        self.assertEqual(
            trainer.global_step, 2,
            f"Expected global_step=2 (2 optimizer steps), got {trainer.global_step}"
        )

        # Loss must be finite and positive
        self.assertEqual(len(history["train_loss"]), 1)
        train_loss = history["train_loss"][0]
        self.assertGreater(train_loss, 0.0)
        self.assertFalse(train_loss != train_loss, "Train loss must not be NaN")   # NaN check

    def test_lightweight_checkpoint(self):
        import tempfile
        from training.fusion_trainer import FusionTrainer, FusionProjector
        from training.training_config import FusionTrainingConfig
        
        config = FusionTrainingConfig(epochs=1)
        vis_proj = FusionProjector(768, 1024)
        aud_proj = FusionProjector(512, 1024)
        trainer = FusionTrainer(config, vis_proj, aud_proj)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "ckpt.pt")
            
            # Save lightweight (inference-only)
            trainer.save_checkpoint(ckpt_path, include_states=False)
            payload = torch.load(ckpt_path, weights_only=True)
            
            self.assertIn("vision_projector", payload)
            self.assertIn("audio_projector", payload)
            self.assertNotIn("optimizer", payload)
            self.assertNotIn("scheduler", payload)
            self.assertNotIn("global_step", payload)

    def test_audio_cache_rejection(self):
        """Verify that zero-embedding samples are never written to the cache directory."""
        import tempfile
        from unittest.mock import patch, MagicMock
        from scripts.precompute_embeddings import main

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

            with patch("scripts.precompute_embeddings.AudioEncoder", return_value=mock_encoder), \
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

if __name__ == "__main__":
    unittest.main()

