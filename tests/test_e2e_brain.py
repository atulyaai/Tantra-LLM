import unittest
import asyncio
import sys
import os

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

if __name__ == "__main__":
    unittest.main()
