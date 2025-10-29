from __future__ import annotations

"""Minimal demo wiring: build full orchestrator and run a single step."""

from pathlib import Path
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from utils.model_loader import ModelLoader
from config import model_config
from utils.low_resource import apply_low_resource_settings, is_low_resource_mode


def load_personality_config() -> dict:
    cfg_path = Path(__file__).parents[1] / "config" / "personality_config.json"
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def build_demo():
    # Low-resource toggles
    apply_low_resource_settings()
    
    # Config and env
    cfg = model_config.MODEL_CONFIG
    spb_repo_or_path = os.environ.get("TANTRA_SPB", "distilgpt2" if is_low_resource_mode() else "gpt2")
    long_vita_dir = os.environ.get("TANTRA_LV_DIR")

    # Tokenizer and encoders
    tok = TextTokenizer()
    # In low-resource mode, avoid heavy Long-VITA load and use fallback
    vision = VisionEncoder(embed_dim=cfg["vision"]["embed_dim"], api_url=None, local_path=None if is_low_resource_mode() else long_vita_dir)
    audio = AudioEncoder(embed_dim=cfg["audio"]["embed_dim"])

    # Projectors and fusion
    proj_v = VisionProjector(cfg["vision"]["embed_dim"], cfg["model_dim"]) 
    proj_a = AudioProjector(cfg["audio"]["embed_dim"], cfg["model_dim"]) 
    fusion = FusionOrchestrator(tok, proj_v, proj_a, cfg["model_dim"]) 

    # Personality and memory
    personality = PersonalityLayer(load_personality_config())
    memory = MemoryManager(wm_tokens=cfg.get("wm_tokens", 8192))

    # Decision engine
    decision = DecisionEngine(personality, memory)

    # Models
    loader = ModelLoader(cfg)
    spb = loader.load_spikingbrain(path=spb_repo_or_path)

    # Response generator
    response = ResponseGenerator(
        spiking_model=spb.get("model"),
        fusion_orchestrator=fusion,
        tokenizer=spb.get("tokenizer") or tok,
        personality_layer=personality,
    )

    # Perception
    perception = Perception(vision_encoder=vision, audio_encoder=audio, tokenizer=tok)

    return BrainOrchestrator(perception, decision, response, memory)


if __name__ == "__main__":
    brain = build_demo()
    print(brain.step(text="mode: direct summarize this meeting"))


