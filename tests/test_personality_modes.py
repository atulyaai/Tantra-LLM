"""Tests for adaptive personality routing and parameterizer (v0.5)."""

from personality.personality_layer import PersonalityLayer
import json
from pathlib import Path


def load_cfg():
    cfg_path = Path("config/personality_config.json")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def test_auto_mode_selection():
    pl = PersonalityLayer(load_cfg())
    mode = pl.select_mode("can you guide me through the steps?")
    assert mode == "MentorBuilder"


def test_user_override_persists():
    pl = PersonalityLayer(load_cfg())
    mode1 = pl.select_mode("mode: creative please")
    assert mode1 == "CreativeExplorer"
    # Next call without override should keep session mode
    mode2 = pl.select_mode("what do you suggest?")
    assert mode2 == "CreativeExplorer"


def test_parameterizer_maps_mode():
    pl = PersonalityLayer(load_cfg())
    pl.select_mode("mode: critical")
    params = pl.parameterize("CriticalChallenger")
    assert isinstance(params.get("prompt_prefix", ""), str)
    assert 0.0 < params.get("temperature", 0.0) <= 1.5
