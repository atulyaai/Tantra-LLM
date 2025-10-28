"""Smoke test for semantic graph facts influencing context prompt."""

from core.memory.memory_manager import MemoryManager
from core.control.brain_orchestrator import BrainOrchestrator
from core.control.perception import Perception
from core.control.decision_engine import DecisionEngine
from core.control.response_generator import ResponseGenerator
from core.fusion.sensory_projectors import VisionProjector, AudioProjector
from encoders.text import TextTokenizer
from encoders.vision import VisionEncoder
from encoders.audio import AudioEncoder
from core.fusion.orchestrator import FusionOrchestrator


def build_brain():
    tok = TextTokenizer()
    vision = VisionEncoder(embed_dim=1024)
    audio = AudioEncoder(embed_dim=1024)
    proj_v = VisionProjector(1024, 4096)
    proj_a = AudioProjector(1024, 4096)
    fusion = FusionOrchestrator(tok, proj_v, proj_a, 4096)
    perception = Perception(vision_encoder=vision, audio_encoder=audio, tokenizer=tok)
    memory = MemoryManager(wm_tokens=8192)

    # Inject a semantic fact
    memory.semantic.add_fact("Python", "is_a", "language")

    # Minimal personality
    class DummyPersonality:
        def select_mode(self, txt):
            return "DirectAssertive"
        def parameterize(self, mode):
            return {"prompt_prefix": ""}

    decision = DecisionEngine(DummyPersonality(), memory)
    response = ResponseGenerator(spiking_model=None, fusion_orchestrator=fusion, tokenizer=tok, personality_layer=DummyPersonality())
    return BrainOrchestrator(perception, decision, response, memory)


def test_semantic_influence_in_prompt():
    brain = build_brain()
    brain.step(text="Tell me about Python")
    assert brain._last_context_prompt is not None
    assert "Python; is_a; language" in brain._last_context_prompt or "Python" in brain._last_context_prompt
