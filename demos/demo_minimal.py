from __future__ import annotations

"""Minimal demo wiring stubs: build orchestrator and run a single step."""

from pathlib import Path
import json

from encoders.text import TextTokenizer
from encoders.vision import VisionEncoder
from encoders.audio import AudioEncoder
from core.fusion.sensory_projectors import VisionProjector, AudioProjector
from core.fusion.orchestrator import FusionOrchestrator
from core.control.perception import Perception
from core.control.decision_engine import DecisionEngine
from core.control.response_generator import ResponseGenerator
from core.control.brain_orchestrator import BrainOrchestrator
from core.memory.memory_manager import MemoryManager
from personality.personality_layer import PersonalityLayer


def load_personality_config() -> dict:
    cfg_path = Path(__file__).parents[1] / "config" / "personality_config.json"
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def build_demo():
    tok = TextTokenizer()
    vision = VisionEncoder(embed_dim=1024)
    audio = AudioEncoder(embed_dim=1024)
    proj_v = VisionProjector(1024, 4096)
    proj_a = AudioProjector(1024, 4096)
    fusion = FusionOrchestrator(tok, proj_v, proj_a, 4096)
    personality = PersonalityLayer(load_personality_config())
    perception = Perception(vision_encoder=vision, audio_encoder=audio, tokenizer=tok)
    memory = MemoryManager(wm_tokens=8192)
    decision = DecisionEngine(personality, memory)
    response = ResponseGenerator(spiking_model=None, fusion_orchestrator=fusion, tokenizer=tok, personality_layer=personality)
    return BrainOrchestrator(perception, decision, response, memory)


if __name__ == "__main__":
    brain = build_demo()
    print(brain.step(text="mode: direct summarize this meeting"))


