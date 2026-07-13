"""NP-DNA brain — everything else in one place.

This module gathers several small, loosely-coupled support systems that
ride alongside the core model but don't belong inside it:

    Plasticity     — PlasticityEngine, PlasticityMetrics, PlasticityAutoScaler
                      Knowledge-preserving health monitoring for the mesh:
                      flags dead/overloaded strands and loss plateaus without
                      ever zeroing or destructively reinitializing weights.
    Classifier     — NpDnaTopicClassifier, tag_text
                      Keyword-based tagging across 10 NP-DNA topic categories.
    Agent          — NpDnaAgent
                      ReAct-style loop wrapping NpDnaCore with cortex search/
                      store tools and a sandboxed arithmetic evaluator.
    Multimodal     — build_multimodal_prompt, encode_image_clip,
                      describe_image, describe_audio
                      Converts images/audio/structured data into text context
                      NP-DNA can consume today, with an opt-in CLIP path.
    Optimise       — quantize_model_for_cpu, apply_torch_compile,
                      model_size_mb, freeze_for_partial_training
                      CPU inference and partial-training helpers.
    Benchmark      — benchmark_checkpoint, write_benchmark, export_release
                      Scores a checkpoint against small fixed task suites
                      and packages versioned releases.
    Codec registry — FrozenCodecRef, FrozenCodecRegistry
                      Lookup table for frozen (non-trainable) multimodal
                      codec adapters referenced by checkpoints.

Each section is self-contained and can be read independently; see the
section banners below.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import operator
import os
import re
import shutil
import struct
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import torch
from torch import nn

from .config import CodecConfig
from .model import NpDnaCore

logger = logging.getLogger(__name__)


# ============================================================================
# Plasticity — Knowledge-preserving monitoring for NP-DNA.
# ============================================================================
#
# Plasticity Engine — KNOWLEDGE-PRESERVING monitoring for NP-DNA.
#
# Core principle: NEVER destroy learned knowledge.
#   - Underutilized strands → distill into active strands via soft transfer
#   - Overloaded strands → spawn child strands seeded from parent (copy, don't erase)
#   - Dead strands → reactivate by rebalancing router, never zero out
#   - Merging → only when strands are truly redundant (near-identical routing)
#
# No zeroing. No destructive reinitialization. Knowledge only flows, never deleted.


@dataclass
class PlasticityEvent:
    step: int
    event_type: str
    details: str = ""


@dataclass
class PlasticityMetrics:
    strand_load: float = 0.0
    max_strand_load: float = 0.0
    min_strand_load: float = 1.0
    dead_count: int = 0
    overloaded_count: int = 0
    layer_loss_plateau: int = 0
    cortex_unused: int = 0
    router_entropy: float = 0.0
    last_check: float = field(default_factory=time.time)


class PlasticityEngine:
    """Monitors NP-DNA and recommends growth — never destroys knowledge.

    - Dead strands: detected and logged; router is rebalanced via entropy bonus
    - Overloaded strands: spawn new strands seeded from the parent
    - Redundant strands: merged by averaging router weights only (not strand params)
    - Loss plateau: triggers layer growth recommendation
    """

    def __init__(
        self,
        core: NpDnaCore,
        check_interval: int = 100,
        dead_threshold: float = 0.01,
        overload_threshold: float = 0.18,
        plateau_window: int = 50,
        plateau_threshold: float = 0.01,
        entropy_target: float = 0.6,
    ):
        self.core = core
        self.check_interval = check_interval
        self.dead_threshold = dead_threshold
        self.overload_threshold = overload_threshold
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.entropy_target = entropy_target

        self.events: list[PlasticityEvent] = []
        self.loss_history: list[float] = []
        self.metrics = PlasticityMetrics()

    def record_loss(self, loss: float) -> None:
        if isinstance(loss, bool):
            raise TypeError("loss must be numeric, got bool")
        if not isinstance(loss, (int, float)):
            raise TypeError(f"loss must be numeric, got {type(loss).__name__}")
        self.loss_history.append(loss)

    def check(self, step: int) -> list[PlasticityEvent]:
        if step % self.check_interval != 0:
            return []

        events: list[PlasticityEvent] = []

        events.extend(self._diagnose_strand_usage(step))
        events.extend(self._check_plateau(step))

        self.events.extend(events)
        return events

    def _diagnose_strand_usage(self, step: int) -> list[PlasticityEvent]:
        events = []
        total_loads = []
        total_dead = 0
        total_overloaded = 0

        for layer_i, mesh in enumerate(self.core.model.mesh_layers):
            stats = mesh.usage_stats
            if not stats:
                mesh.reset_usage()
                continue

            loads = list(stats.values())
            layer_avg = sum(loads) / len(loads) if loads else 0.0
            layer_max = max(loads) if loads else 0.0
            layer_min = min(loads) if loads else 0.0
            total_loads.extend(loads)

            dead = [s_id for s_id, ratio in stats.items() if ratio < self.dead_threshold]
            overloaded = [s_id for s_id, ratio in stats.items() if ratio > self.overload_threshold]

            if dead:
                total_dead += len(dead)
                msg = f"Layer {layer_i}: {len(dead)} low-usage strands — router needs more data to balance"
                logger.info("Plasticity: %s", msg)
                events.append(PlasticityEvent(step, "low_usage_strands", msg))

            if overloaded:
                total_overloaded += len(overloaded)
                msg = f"Layer {layer_i}: {len(overloaded)} high-use strands — suggesting growth"
                logger.info("Plasticity: %s", msg)
                events.append(PlasticityEvent(step, "overloaded_strands", msg))

            entropy = getattr(mesh, "last_router_entropy", 0.0)
            if entropy < 0.3:
                logger.info("Plasticity: Layer %d router entropy=%.3f — low diversity", layer_i, entropy)

            mesh.reset_usage()

        if total_loads:
            self.metrics.strand_load = sum(total_loads) / len(total_loads)
            self.metrics.max_strand_load = max(total_loads)
            self.metrics.min_strand_load = min(total_loads)
            self.metrics.dead_count = total_dead
            self.metrics.overloaded_count = total_overloaded

        self.metrics.last_check = time.time()
        return events

    def _check_plateau(self, step: int) -> list[PlasticityEvent]:
        events = []
        W = self.plateau_window
        if len(self.loss_history) < W * 2:
            return events

        old_avg = sum(self.loss_history[-W * 2:-W]) / W
        new_avg = sum(self.loss_history[-W:]) / W

        if old_avg > 0:
            improvement = (old_avg - new_avg) / old_avg
            if improvement < self.plateau_threshold:
                msg = f"Loss plateau: {old_avg:.4f} -> {new_avg:.4f} ({improvement:.1%})"
                logger.info("Plasticity: %s", msg)
                events.append(PlasticityEvent(step, "plateau", msg))

        return events

    def summary(self) -> str:
        if not self.events:
            return "No plasticity events recorded."
        lines = [f"Plasticity: {len(self.events)} events"]
        for e in self.events[-20:]:
            lines.append(f"  step {e.step}: [{e.event_type}] {e.details}")
        return "\n".join(lines)


class PlasticityAutoScaler:
    """Small policy object for dashboard/training auto-scale recommendations."""

    DEFAULT_CONFIG = {
        "strand_overload": 0.9,
        "plateau_window": 10,
        "plateau_variance": 0.001,
        "cortex_unused_ratio": 0.75,
        "max_history": 50,
        "max_strands": 128,
    }

    def __init__(self, config: dict | None = None):
        self.config = dict(self.DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        self.metrics = PlasticityMetrics()
        self._action_history: list[dict] = []

    def check_and_scale(
        self,
        strand_capacity: float,
        loss_history: list[float],
        cortex_size: int,
        cortex_used: int,
    ) -> list[str]:
        actions: list[str] = []
        now = time.time()
        cortex_unused = max(0, int(cortex_size) - max(0, int(cortex_used)))

        self.metrics.strand_load = float(strand_capacity)
        self.metrics.cortex_unused = cortex_unused
        self.metrics.last_check = now

        if strand_capacity >= self.config["strand_overload"]:
            actions.append("add_strand")

        window = int(self.config["plateau_window"])
        if len(loss_history) >= window:
            recent = [float(v) for v in loss_history[-window:]]
            if max(recent) - min(recent) <= float(self.config["plateau_variance"]):
                self.metrics.layer_loss_plateau = len(recent)
                actions.append("add_layer")

        if cortex_size > 0:
            unused_ratio = cortex_unused / max(1, cortex_size)
            if unused_ratio >= self.config["cortex_unused_ratio"]:
                actions.append("prune_cortex")

        return actions

    def get_metrics(self) -> dict:
        return {
            "strand_load": self.metrics.strand_load,
            "cortex_unused": self.metrics.cortex_unused,
            "layer_loss_plateau": self.metrics.layer_loss_plateau,
            "last_check": self.metrics.last_check,
        }

    def get_action_history(self) -> list[dict]:
        return list(self._action_history)


# ============================================================================
# Classifier — Keyword-based topic tagger.
# ============================================================================
#
# NP-DNA topic classifier — keyword-based category tagger.
#
# Maps text to one of 10 NP-DNA categories using weighted keyword matching.
# Each category has primary keywords (high weight) and secondary (low weight).

TOPICS: dict[str, list[str]] = {
    "conversation": [
        "general chat", "small talk", "greetings", "introductions",
        "farewells", "thanks", "apologies", "opinions", "recommendations",
        "casual discussion", "storytelling", "personal anecdotes",
        "social dialogue", "questions and answers", "everyday talk",
        "friendly advice", "chit-chat", "banter", "jokes", "humor",
    ],
    "code": [
        "programming languages", "python", "javascript", "typescript", "rust",
        "c++", "go", "java", "swift", "kotlin", "ruby", "php",
        "algorithms", "data structures", "software design", "architecture",
        "debugging", "testing", "web development", "frontend", "backend",
        "api design", "rest", "graphql", "databases", "sql", "nosql",
        "devops", "ci/cd", "docker", "kubernetes", "cloud",
        "version control", "git", "code review", "refactoring",
        "functional programming", "oop", "design patterns",
        "systems programming", "compilers", "interpreters",
        "machine learning code", "data science coding",
        "scripting", "automation", "cli tools", "package management",
        "concurrency", "async", "parallel computing",
        "security", "authentication", "authorization",
        "performance optimization", "profiling", "memory management",
        "mobile development", "ios", "android", "cross-platform",
        "game development", "unity", "unreal engine",
        "embedded systems", "firmware", "arduino", "raspberry pi",
        "blockchain", "smart contracts", "web3",
        "regex", "parsing", "serialization",
        "open source", "licensing", "contribution",
    ],
    "math": [
        "arithmetic", "algebra", "linear algebra", "geometry",
        "calculus", "differential equations", "statistics",
        "probability", "number theory", "topology", "trigonometry",
        "discrete mathematics", "combinatorics", "graph theory",
        "real analysis", "complex analysis", "functional analysis",
        "set theory", "logic", "proofs", "mathematical reasoning",
        "optimization", "numerical methods", "linear programming",
        "fourier analysis", "signal processing", "wavelets",
        "game theory", "decision theory", "information theory",
        "category theory", "abstract algebra", "group theory",
        "ring theory", "field theory", "galois theory",
        "measure theory", "integration", "lebesgue integral",
        "differential geometry", "manifolds", "tensor calculus",
        "stochastic processes", "markov chains", "monte carlo",
        "bayesian statistics", "hypothesis testing", "regression",
        "mathematical modeling", "simulation",
        "p-adic numbers", "analytic number theory", "modular forms",
        "lie algebras", "representation theory",
    ],
    "science": [
        "physics", "mechanics", "thermodynamics", "quantum mechanics",
        "relativity", "electromagnetism", "optics", "acoustics",
        "chemistry", "organic chemistry", "inorganic chemistry",
        "biochemistry", "physical chemistry", "analytical chemistry",
        "biology", "molecular biology", "cell biology", "genetics",
        "evolution", "ecology", "microbiology", "zoology", "botany",
        "astronomy", "astrophysics", "cosmology", "planetary science",
        "earth science", "geology", "oceanography", "meteorology",
        "neuroscience", "cognitive science", "psychology (science)",
        "materials science", "nanotechnology", "crystallography",
        "particle physics", "nuclear physics", "plasma physics",
        "fluid dynamics", "aerodynamics", "hydrodynamics",
        "climate science", "environmental science", "sustainability",
        "scientific method", "experiment design", "peer review",
        "laboratory techniques", "spectroscopy", "chromatography",
        "bioinformatics", "computational biology", "systems biology",
        "epidemiology", "immunology", "pharmacology",
        "paleontology", "archaeology (scientific)", "anthropology (scientific)",
    ],
    "writing": [
        "creative writing", "fiction", "non-fiction", "poetry",
        "essays", "journalism", "copywriting", "technical writing",
        "storytelling", "narrative structure", "plot development",
        "character development", "dialogue writing", "worldbuilding",
        "grammar and style", "voice and tone", "editing", "proofreading",
        "screenwriting", "playwriting", "scriptwriting",
        "blogging", "content writing", "seo writing",
        "academic writing", "research papers", "theses", "dissertations",
        "business writing", "reports", "proposals", "memos",
        "persuasive writing", "argumentative writing", "descriptive writing",
        "expository writing", "narrative writing",
        "writing prompts", "writing exercises", "writing workshops",
        "genre fiction", "science fiction", "fantasy", "mystery", "romance",
        "literary analysis", "criticism", "book reviews",
        "novel writing", "short stories", "flash fiction",
        "autobiography", "memoir", "biographical writing",
        "travel writing", "food writing", "nature writing",
    ],
    "language": [
        "linguistics", "phonetics", "phonology", "morphology",
        "syntax", "semantics", "pragmatics", "discourse analysis",
        "grammar", "vocabulary", "etymology", "word origins",
        "translation", "interpretation", "localization",
        "language learning", "second language acquisition",
        "multilingualism", "bilingualism",
        "comparative linguistics", "historical linguistics",
        "sociolinguistics", "dialectology", "language variation",
        "psycholinguistics", "neurolinguistics",
        "computational linguistics", "nlp", "corpus linguistics",
        "sign languages", "deaf studies", "manual communication",
        "writing systems", "orthography", "scripts", "alphabets",
        "endangered languages", "language documentation",
        "language policy", "language planning", "linguistic rights",
        "pragmatics", "speech acts", "implicature",
        "cognitive linguistics", "construction grammar",
        "language typology", "universal grammar",
        "applied linguistics", "tesol", "efl", "esl",
        "lexicography", "dictionary making", "terminology",
        "forensic linguistics", "linguistic profiling",
    ],
    "history": [
        "world history", "ancient civilizations", "medieval history",
        "modern history", "contemporary history",
        "european history", "asian history", "african history",
        "american history", "middle eastern history",
        "wars and conflicts", "world war i", "world war ii",
        "cold war", "civil wars", "revolutions", "battles",
        "biographies", "historical figures", "leaders",
        "archaeology", "historical artifacts", "ruins",
        "cultural history", "social history", "economic history",
        "political history", "diplomatic history", "military history",
        "colonial history", "imperialism", "decolonization",
        "ancient rome", "ancient greece", "ancient egypt",
        "china history", "india history", "japan history",
        "renaissance", "enlightenment", "industrial revolution",
        "age of exploration", "age of discovery", "colonization",
        "historical methodology", "historiography",
        "primary sources", "archival research", "oral history",
        "genealogy", "family history", "ancestry",
        "timelines", "historical periods", "eras",
        "historical maps", "cartography history",
        "religious history", "history of ideas", "intellectual history",
        "gender history", "history of feminism",
        "environmental history", "climate history",
    ],
    "society": [
        "economics", "macroeconomics", "microeconomics", "finance",
        "politics", "political science", "government", "policy",
        "philosophy", "ethics", "morality", "metaphysics",
        "epistemology", "aesthetics", "logic (philosophical)",
        "psychology", "clinical psychology", "social psychology",
        "developmental psychology", "personality psychology",
        "sociology", "social theory", "social structures",
        "anthropology", "cultural anthropology", "social anthropology",
        "law", "legal systems", "constitutional law", "criminal law",
        "education", "pedagogy", "learning theory", "curriculum",
        "religion", "theology", "comparative religion",
        "political economy", "public policy", "governance",
        "international relations", "geopolitics", "diplomacy",
        "social justice", "human rights", "civil rights",
        "urban studies", "demography", "population studies",
        "media studies", "communication theory", "journalism (society)",
        "cultural studies", "postcolonial theory", "critical theory",
        "environmental policy", "sustainability (social)",
        "business ethics", "corporate governance",
        "inequality", "poverty", "social stratification",
        "gender studies", "race and ethnicity", "identity politics",
        "social movements", "activism", "protest",
        "consumer behavior", "marketing", "advertising (social)",
        "organizational behavior", "management theory",
    ],
    "health": [
        "medicine", "clinical medicine", "diagnosis", "treatment",
        "anatomy", "physiology", "pathology",
        "nutrition", "dietetics", "food science", "dietary guidelines",
        "fitness", "exercise", "physical training", "sports medicine",
        "mental health", "psychiatry", "therapy", "counseling",
        "public health", "epidemiology", "health policy",
        "diseases", "disorders", "conditions", "syndromes",
        "pharmacology", "medications", "drugs", "prescriptions",
        "surgery", "surgical procedures", "operative medicine",
        "preventive medicine", "vaccination", "screening",
        "alternative medicine", "holistic health", "herbal medicine",
        "sleep health", "stress management", "wellness",
        "pediatrics", "child health", "development",
        "geriatrics", "aging", "elderly care",
        "cardiology", "heart health", "circulatory system",
        "neurology", "brain health", "nervous system",
        "oncology", "cancer", "tumors",
        "immunology", "immune system", "allergies", "autoimmune",
        "endocrinology", "hormones", "diabetes", "thyroid",
        "dermatology", "skin health", "hair", "nails",
        "orthopedics", "bones", "joints", "muscles",
        "ophthalmology", "vision", "eye health",
        "dentistry", "oral health", "dental care",
        "gynecology", "obstetrics", "women's health",
        "urology", "kidney health", "urinary system",
        "gastroenterology", "digestive health", "gut",
        "pulmonology", "respiratory health", "lungs",
        "emergency medicine", "first aid", "trauma care",
        "addiction", "substance abuse", "recovery",
    ],
    "art": [
        "music", "music theory", "composition", "performance",
        "visual art", "painting", "drawing", "sculpture", "printmaking",
        "design", "graphic design", "industrial design", "interior design",
        "film", "cinema", "movie making", "film theory", "cinematography",
        "photography", "photo editing", "camera techniques",
        "architecture", "architectural design", "building design",
        "fashion", "fashion design", "textiles", "clothing",
        "performing arts", "theatre", "dance", "opera", "ballet",
        "art history", "art criticism", "art theory", "aesthetics (art)",
        "digital art", "computer graphics", "3d modeling", "animation",
        "illustration", "cartooning", "comics", "manga",
        "ceramics", "pottery", "glass art", "jewelry making",
        "calligraphy", "typography", "lettering",
        "mixed media", "collage", "assemblage", "installation art",
        "conceptual art", "contemporary art", "modern art",
        "street art", "graffiti", "public art",
        "video games as art", "game design", "interactive media",
        "art education", "art techniques", "color theory",
        "composition (visual)", "perspective", "lighting",
        "art restoration", "conservation", "curation",
    ],
    "emotion": [
        "empathy", "sentiment", "feelings", "mood", "anger", "happiness",
        "sadness", "fear", "surprise", "disgust", "emotional intelligence",
        "affection", "grief", "joy", "anxiety", "depression", "frustration",
        "compassion", "sympathy", "emotional state", "affective computing"
    ],
    "spatial": [
        "navigation", "maps", "gps", "coordinates", "direction",
        "distance", "location", "spatial reasoning", "geometry (spatial)",
        "3d space", "topology (spatial)", "orientation", "mapping",
        "wayfinding", "geography", "positioning", "kinematics"
    ],
    "audio": [
        "speech recognition", "sound waves", "acoustics", "voice processing",
        "audio signals", "frequencies", "pitch", "timbre", "noise reduction",
        "audio processing", "spectrogram", "phonemes", "vocalization",
        "listening", "hearing", "sound design", "echolocation"
    ],
    "vision": [
        "computer vision", "object detection", "image recognition",
        "optical flow", "pixels", "visual perception", "sight",
        "image processing", "pattern recognition", "scene understanding",
        "facial recognition", "depth perception", "color spaces"
    ],
    "action": [
        "robotics", "motor control", "actuators", "servos", "movement",
        "manipulation", "grasping", "kinematics", "automation",
        "tool use", "physical interaction", "locomotion", "walking",
        "trajectory planning", "haptics"
    ],
}


@dataclass
class TopicClassification:
    category: str
    sub_topic: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)


class NpDnaTopicClassifier:
    """Keyword-based classifier for NP-DNA's 10 topic categories.

    Uses weighted keyword matching with primary and secondary keywords
    per category. Returns the best-match category and sub-topic.
    """

    def __init__(self):
        # Build keyword patterns for each category
        self._patterns: dict[str, list[tuple[str, float]]] = {}
        for category, sub_topics in TOPICS.items():
            patterns: list[tuple[str, float]] = []
            # Add category name itself as high-weight pattern
            patterns.append((rf"\b{category}\b", 5.0))
            for sub in sub_topics:
                # Primary keywords from the sub-topic name
                for word in sub.split():
                    if len(word) > 3:
                        patterns.append((rf"\b{re.escape(word)}\b", 2.0))
                # Multi-word phrases
                if " " in sub:
                    patterns.append((rf"{re.escape(sub)}", 4.0))
            self._patterns[category] = _dedupe_patterns(patterns)

        # Pre-compile all regex patterns for performance (compile once)
        self._compiled: dict[str, list[tuple[re.Pattern, float]]] = {}
        for cat, pats in self._patterns.items():
            self._compiled[cat] = [(re.compile(p, re.IGNORECASE), w) for p, w in pats]

        # Category-exclusive keywords (very high weight)
        self._exclusive: dict[str, list[tuple[str, float]]] = {
            "code": [
                (r"\bdef\s+\w+\s*\(", 8.0), (r"\bimport\s+\w+", 8.0),
                (r"\bclass\s+\w+", 8.0), (r"\bif\s+__name__\s*==\s*['\"]__main__['\"]", 10.0),
                (r"\bprint\(.*\)", 5.0), (r"\breturn\s+\w+", 4.0),
                (r"\bfor\s+\w+\s+in\s+range", 6.0), (r"\bwhile\s+True", 6.0),
                (r"\btry\s*:", 5.0), (r"\bexcept\s+\w+", 5.0),
                (r"\bconst\s+\w+\s*=", 5.0), (r"\bfunction\s+\w+\s*\(", 5.0),
                (r"\blet\s+\w+\s*=", 4.0), (r"\bvar\s+\w+\s*=", 4.0),
                (r"\bdef\s+main\b", 7.0), (r"\basync\s+def\b", 6.0),
                (r"\bfrom\s+\w+\s+import\b", 7.0), (r"\b#include\b", 8.0),
                (r"\bpublic\s+(static\s+)?void\b", 6.0),
                (r"\bint\s+main\s*\(", 6.0), (r"\bfn\s+\w+\s*\(", 6.0),
                (r"\b```\w*\n", 5.0),  # code block markers
            ],
            "math": [
                (r"\\frac\{", 6.0), (r"\\int", 6.0), (r"\\sum", 6.0),
                (r"\\lim", 5.0), (r"\\partial", 5.0), (r"\\sqrt", 5.0),
                (r"\be\^\{", 4.0), (r"\bπ\b", 6.0), (r"\bΣ\b", 5.0),
                (r"\b∫\b", 5.0), (r"\b∀\b", 4.0), (r"\b∃\b", 4.0),
                (r"\bx\^2\b", 4.0), (r"\bdx\b", 4.0),
                (r"\=\s*\{", 3.0),  # set notation
                (r"\\mathcal", 5.0), (r"\\mathbb", 5.0),
            ],
            "conversation": [
                (r"\b(hi|hello|hey)\b", 4.0), (r"\bhow are you\b", 6.0),
                (r"\b(thanks|thank you)\b", 4.0), (r"\byou're welcome\b", 4.0),
                (r"\bwhat's up\b", 5.0), (r"\bhow's it going\b", 5.0),
                (r"\bnice to meet\b", 5.0), (r"\bgood (morning|afternoon|evening)\b", 4.0),
                (r"\bhave a (great|nice|good) day\b", 5.0),
                (r"\bcan you (help|assist)\b", 4.0),
                (r"\bwhat do you think\b", 4.0),
                (r"\bI (think|feel|believe)\b", 3.0),
                (r"\bjust saying\b", 4.0), (r"\bby the way\b", 3.0),
            ],
            "science": [
                (r"F\s*=\s*ma", 7.0), (r"E\s*=\s*mc\^2", 8.0),
                (r"\bH₂O\b", 6.0), (r"\bCO₂\b", 5.0),
                (r"\bπr²\b", 5.0), (r"\bDNA\b", 4.0),
                (r"\bRNA\b", 4.0), (r"\bATP\b", 4.0),
                (r"\bpH\s*=\b", 5.0), (r"\bPV\s*=\s*nRT\b", 8.0),
                (r"\bλ\b", 4.0), (r"\bν\b", 4.0),
                (r"\bΔG\b", 5.0), (r"\bΔH\b", 5.0),
            ],
            "writing": [
                (r"\bonce upon a time\b", 6.0), (r"\bchapter \d+\b", 4.0),
                (r"\bthe end\b", 3.0), (r"\bdear reader\b", 4.0),
                (r"\bdear diary\b", 5.0), (r"\bin conclusion\b", 3.0),
                (r"\bfirst draft\b", 4.0), (r"\bwriter's block\b", 5.0),
                (r"\bnovel\b", 3.0), (r"\bshort story\b", 4.0),
            ],
            "health": [
                (r"\bmg/kg\b", 4.0), (r"\bbpm\b", 4.0), (r"\bmm Hg\b", 5.0),
                (r"\bBMI\b", 4.0), (r"\bECG\b", 4.0), (r"\bMRI\b", 4.0),
                (r"\bCT scan\b", 5.0), (r"\bblood pressure\b", 4.0),
                (r"\bheart rate\b", 3.0), (r"\btake\s+\d+\s*mg\b", 5.0),
            ],
            "art": [
                (r"\bCMYK\b", 5.0), (r"\bRGB\b", 4.0),
                (r"\bcolor palette\b", 4.0), (r"\bcomposition\b", 3.0),
                (r"\bfocal point\b", 4.0), (r"\bcontrast\b", 3.0),
                (r"\bperspective\b", 3.0), (r"\bbrush\s+stroke\b", 4.0),
                (r"\boil on canvas\b", 5.0), (r"\bwatercolor\b", 5.0),
                (r"\bacoustic\b", 4.0), (r"\bchord\s+progression\b", 5.0),
                (r"\btempo\b", 3.0), (r"\bcolor theory\b", 5.0),
            ],
            "language": [
                (r"\bverb\b", 4.0), (r"\bnoun\b", 4.0), (r"\badjective\b", 4.0),
                (r"\badverb\b", 4.0), (r"\bconjugation\b", 5.0),
                (r"\bdeclension\b", 5.0), (r"\bsyntax\b", 4.0),
                (r"\bmorphology\b", 5.0), (r"\bphoneme\b", 5.0),
                (r"\bIPA\b", 4.0), (r"\bglottal stop\b", 6.0),
                (r"\bgrammatical (gender|case)\b", 5.0),
                (r"\bsite:lang\b", 3.0),
            ],
            "history": [
                (r"\b\d{3,4}\s*(BC|BCE|AD|CE)\b", 5.0),
                (r"\b(B.C.|A.D.)\b", 5.0),
                (r"\bin the \d+th century\b", 5.0),
                (r"\bcentury\b", 3.0), (r"\bdynasty\b", 5.0),
                (r"\bempire\b", 4.0), (r"\bkingdom\b", 3.0),
                (r"\btreaty of\b", 5.0), (r"\bbattle of\b", 5.0),
                (r"\binvasion\b", 4.0), (r"\bwar\b", 3.0),
            ],
            "society": [
                (r"\bGDP\b", 4.0), (r"\binflation\b", 4.0),
                (r"\binvisible hand\b", 6.0), (r"\bsupply and demand\b", 5.0),
                (r"\bcategorical imperative\b", 7.0),
                (r"\bsocial contract\b", 6.0),
                (r"\bcorrelation does not imply\b", 5.0),
                (r"\bnull hypothesis\b", 5.0), (r"\bp-value\b", 5.0),
                (r"\bconstitutional\b", 4.0), (r"\bamendment\b", 4.0),
                (r"\bplaintiff\b", 4.0), (r"\bdefendant\b", 4.0),
            ],
            "emotion": [
                (r"\bi feel (happy|sad|angry|anxious)\b", 6.0),
                (r"\bsentiment analysis\b", 5.0),
            ],
            "spatial": [
                (r"\blatitude and longitude\b", 6.0),
                (r"\bturn (left|right) at\b", 5.0),
            ],
            "audio": [
                (r"\bsample rate\b", 5.0),
                (r"\bfft\b", 4.0),
            ],
            "vision": [
                (r"\bbounding box\b", 5.0),
                (r"\bsegmentation mask\b", 5.0),
            ],
            "action": [
                (r"\bdegrees of freedom\b", 5.0),
                (r"\binverse kinematics\b", 6.0),
            ],
        }

        # Pre-compile exclusive patterns too
        self._exclusive_compiled: dict[str, list[tuple[re.Pattern, float]]] = {}
        for cat, pats in self._exclusive.items():
            self._exclusive_compiled[cat] = [(re.compile(p, re.IGNORECASE), w) for p, w in pats]

    def classify(self, text: str) -> TopicClassification:
        """Classify text into NP-DNA topic category.

        Returns TopicClassification with category, sub-topic, and confidence.
        """
        text_lower = text.lower()[:50000]  # cap text length for performance
        scores: dict[str, float] = {}

        # Score each category using pre-compiled patterns
        for category, patterns in self._compiled.items():
            score = 0.0
            for pattern, weight in patterns:
                if pattern.search(text_lower):
                    score += weight
            scores[category] = score

        # Add exclusive keyword bonuses
        for category, patterns in self._exclusive_compiled.items():
            bonus = 0.0
            for pattern, weight in patterns:
                if pattern.search(text_lower):
                    bonus += weight
            if bonus > 0:
                scores[category] = scores.get(category, 0) + bonus

        # Find best category — fallback to conversation if nothing matches
        if not scores:
            return TopicClassification(
                category="conversation",
                sub_topic="general chat",
                confidence=0.5,
                scores={"conversation": 1.0},
            )
        best_cat = max(scores, key=scores.get)
        total = sum(scores.values()) or 1.0
        top_score = max(scores.values())
        if top_score > 0:
            confidence = scores[best_cat] / min(total, top_score * 3)
        else:
            confidence = 0.5

        # Find best sub-topic within the best category
        best_sub = self._find_best_sub_topic(text_lower, best_cat)

        return TopicClassification(
            category=best_cat,
            sub_topic=best_sub,
            confidence=confidence,
            scores=scores,
        )

    def classify_batch(self, texts: list[str]) -> list[TopicClassification]:
        """Classify a batch of texts."""
        return [self.classify(t) for t in texts]

    def _find_best_sub_topic(self, text_lower: str, category: str) -> str:
        """Find best-matching sub-topic within a category."""
        if category not in TOPICS:
            return "general"
        best_sub = "general"
        best_score = 0
        for sub in TOPICS[category]:
            score = 0
            for word in sub.split():
                if len(word) > 3 and word in text_lower:
                    score += 1
            if " " in sub and sub in text_lower:
                score += 3
            if score > best_score:
                best_score = score
                best_sub = sub
        return best_sub


def _dedupe_patterns(
    patterns: list[tuple[str, float]]
) -> list[tuple[str, float]]:
    """Remove duplicate patterns, keeping the highest weight."""
    seen: dict[str, float] = {}
    for pat, weight in patterns:
        if pat not in seen or weight > seen[pat]:
            seen[pat] = weight
    return list(seen.items())


# Convenience instance
classifier = NpDnaTopicClassifier()


def tag_text(text: str) -> TopicClassification:
    """Quick-tag a text with topic category."""
    return classifier.classify(text)


# Backward-compat alias for scripts importing `CATEGORIES`
CATEGORIES: list[str] = list(TOPICS.keys())


# ============================================================================
# Agent — ReAct loop wrapping NpDnaCore, plus a sandboxed math evaluator.
# ============================================================================
#
# NP-DNA Autonomy Layer.
#
# Implements the NpDnaAgent which wraps NpDnaCore in a ReAct (Reasoning and Acting)
# execution loop, allowing the model to run search and memory storage tools.


class SafeExpressionError(ValueError):
    pass


_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos, ast.USub: operator.neg,
}


def _safe_pow(base: Any, exp: Any, mod: Any = None) -> Any:
    if not isinstance(exp, (int, float)):
        raise SafeExpressionError("exponent must be a number")
    if abs(exp) > 12:
        raise SafeExpressionError("exponent is too large")
    if mod is not None:
        return pow(base, exp, mod)
    return pow(base, exp)


_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "pow": _safe_pow, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "log": math.log, "log10": math.log10,
    "ceil": math.ceil, "floor": math.floor,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def safe_math_eval(expression: str) -> Any:
    if len(expression) > 512:
        raise SafeExpressionError("expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise SafeExpressionError("invalid expression") from exc
    return _eval_node(tree.body)


def safe_expression_output(expression: str) -> str:
    try:
        return str(safe_math_eval(expression))
    except Exception as exc:
        return f"Expression blocked: {str(exc)[:200]}"


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise SafeExpressionError("only numeric literals are allowed")
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise SafeExpressionError(f"unknown name: {node.id}")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise SafeExpressionError("operator is not allowed")
        left = _eval_node(node.left); right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise SafeExpressionError("exponent is too large")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise SafeExpressionError("operator is not allowed")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise SafeExpressionError("only named math functions are allowed")
        func = _FUNCTIONS.get(node.func.id)
        if func is None:
            raise SafeExpressionError(f"function is not allowed: {node.func.id}")
        if node.keywords:
            raise SafeExpressionError("keyword arguments are not allowed")
        args = [_eval_node(arg) for arg in node.args]
        return func(*args)
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_eval_node(elt) for elt in node.elts]
    raise SafeExpressionError(f"syntax is not allowed: {type(node).__name__}")


class NpDnaAgent:
    """ReAct-style autonomous agent wrapping NpDnaCore."""
    def __init__(self, core: NpDnaCore):
        self.core = core
        self.tools: dict[str, Callable[[str], str]] = {}

        # Register default tools
        self.register_tool("cortex_search", self._cortex_search)
        self.register_tool("cortex_store", self._cortex_store)
        self.register_tool("math_eval", self._math_eval)

        # New cognitive layer tools
        self.register_tool("analyze_sentiment", self._analyze_sentiment)
        self.register_tool("measure_distance", self._measure_distance)
        self.register_tool("mock_robot_arm", self._mock_robot_arm)

    def register_tool(self, name: str, func: Callable[[str], str]) -> None:
        """Register a new python tool for the agent to call."""
        self.tools[name] = func
        logger.info("Tool '%s' registered with NpDnaAgent.", name)

    def _encode_to_vector(self, text: str) -> torch.Tensor:
        """Encode text to a dense vector via tokenizer + embedding mean."""
        token_ids = self.core.encode(text, allow_growth=False)
        if not token_ids:
            return torch.zeros(self.core.config.hidden_size)
        with torch.no_grad():
            token_t = torch.tensor(token_ids, dtype=torch.long, device=self.core.model.embedding.weight.device)
            embs = self.core.model.embedding(token_t)
            return embs.mean(dim=0)

    def _cortex_search(self, query: str) -> str:
        """Search memory cortex for query."""
        query_vec = self._encode_to_vector(query)
        if self.core.model.cortex.size == 0:
            return "Memory cortex is empty. No matching memories found."
        values, scores = self.core.model.cortex.retrieve(query_vec, top_k=2)
        if values is None or values.shape[0] == 0:
            return "No matching memories found."
        items = []
        last_indices = self.core.model.cortex._last_top_indices
        entries = self.core.model.cortex.entries
        for i in range(values.shape[0]):
            score = float(scores[i].item())
            # Use actual top_k index from retrieve result, not loop counter
            entry_idx = int(last_indices[0, i].item()) if last_indices is not None and i < last_indices.shape[1] else i
            if 0 <= entry_idx < len(entries):
                entry = entries[entry_idx]
                items.append(f"Match {i+1} (score={score:.3f}): {entry.source}")
            else:
                items.append(f"Match {i+1} (score={score:.3f}): [invalid index {entry_idx}]")
        return "\n".join(items) if items else "No matching memories found."

    def _cortex_store(self, fact: str) -> str:
        """Store a new fact in the memory cortex."""
        fact_clean = fact.strip()
        vector = self._encode_to_vector(fact_clean)
        self.core.model.cortex.store(key=vector, value=vector, topic="agent", source=fact_clean)
        if hasattr(self.core, "active_path") and self.core.active_path:
            self.core.model.cortex.save(self.core.active_path / "cortex")
        return f"Successfully saved fact to cortex: '{fact_clean[:80]}'"

    def _web_search(self, query: str) -> str:
        """Search the web using DuckDuckGo Instant Answer API (no key needed)."""
        import urllib.request
        import urllib.parse
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_redirect=1"
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                data = __import__("json").loads(r.read())
            result = data.get("AbstractText") or data.get("Answer") or data.get("RelatedTopics", [{}])[0].get("Text", "")
            if result:
                return result[:500]
            return f"No instant answer found for '{query}'."
        except Exception as e:
            return f"Web search failed: {str(e)[:200]}"

    def _code_execute(self, code: str) -> str:
        """Evaluate a single safe expression without launching a shell."""
        return safe_expression_output(code.strip())

    def _math_eval(self, expr: str) -> str:
        """Evaluate a mathematical expression safely."""
        return safe_expression_output(expr.strip())

    def _analyze_sentiment(self, text: str) -> str:
        """Basic sentiment analysis."""
        t = text.lower()
        if any(w in t for w in ["happy", "joy", "good", "great", "love", "awesome", "excellent", "amazing", "wonderful", "delighted"]):
            return "Positive sentiment detected."
        elif any(w in t for w in ["sad", "angry", "bad", "terrible", "hate", "awful", "anxious", "fear", "depressed", "furious", "upset"]):
            return "Negative sentiment detected."
        return "Neutral sentiment detected."

    def _measure_distance(self, coords_str: str) -> str:
        """Measure distance between two points in 2D or 3D."""
        try:
            import re
            nums = re.findall(r"-?\d+\.?\d*", coords_str)
            nums = list(map(float, nums))
            if len(nums) == 6:
                x1, y1, z1, x2, y2, z2 = nums[:6]
                dist = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5
                return f"3D Distance calculated: {dist:.2f}"
            elif len(nums) >= 4:
                x1, y1, x2, y2 = nums[:4]
                dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
                return f"2D Distance calculated: {dist:.2f}"
            return "Could not parse coordinates. Provide 'x1,y1' and 'x2,y2' or 'x1,y1,z1' and 'x2,y2,z2'."
        except Exception as e:
            return f"Distance error: {e}"

    def _mock_robot_arm(self, command: str) -> str:
        """Mock robotic arm execution."""
        return f"[MOCK ACTION] Robot arm executing: {command.strip()}"

    def run(self, user_prompt: str, max_iterations: int = 5) -> str:
        """Execute the ReAct loop until response is reached or iteration limit is hit.

        Args:
            user_prompt: User request.
            max_iterations: Max ReAct steps to prevent infinite loop.

        Returns:
            The final text response from the agent.
        """
        system_instructions = (
            "You are an autonomous NP-DNA agent. Solve the user's goal by thinking step-by-step "
            "and invoking tools. Supported tools:\n"
            "  - Action: cortex_search[query]\n"
            "  - Action: cortex_store[fact]\n\n"
            "Format your output strictly using these tags:\n"
            "[Thought] Explain your reasoning here.\n"
            "Action: tool_name[arguments]\n"
            "[Observation] Results will be shown here.\n"
            "Action: respond[your final response to the user]\n"
        )

        context = f"{system_instructions}\nUser: {user_prompt}\n"

        for iteration in range(max_iterations):
            logger.info("NpDnaAgent iteration %d/%d", iteration + 1, max_iterations)

            current_prompt = context + "[Thought]"

            output = self.core.generate(
                current_prompt,
                max_tokens=128,
                temperature=0.3,
                top_k=5
            )

            new_text = output.strip()
            logger.debug("Agent generated: %s", new_text)

            context += "[Thought] " + new_text + "\n"

            action_match = re.search(r"Action:\s*([a-zA-Z0-9_-]+)\[(.*?)\]", new_text)

            if not action_match:
                return new_text

            tool_name = action_match.group(1).strip()
            tool_arg = action_match.group(2).strip()

            if tool_name == "respond":
                return tool_arg

            if tool_name in self.tools:
                logger.info("Executing tool '%s' with arg: %s", tool_name, tool_arg)
                try:
                    observation = self.tools[tool_name](tool_arg)
                except Exception as e:
                    observation = f"Error executing tool: {str(e)}"
            else:
                observation = f"Unknown tool '{tool_name}'. Available: {list(self.tools.keys())}."

            logger.debug("Observation: %s", observation)
            context += f"[Observation] {observation}\n"

        return "Agent failed to reach a conclusion within step limit."

    def run_with_telemetry(self, user_prompt: str, max_iterations: int = 5) -> tuple[str, list[dict]]:
        """Execute the ReAct loop and return both final response and detailed step-by-step logs."""
        steps = []
        system_instructions = (
            "You are an autonomous NP-DNA agent. Solve the user's goal by thinking step-by-step "
            "and invoking tools. Supported tools:\n"
            "  - Action: cortex_search[query]\n"
            "  - Action: cortex_store[fact]\n\n"
            "Format your output strictly using these tags:\n"
            "[Thought] Explain your reasoning here.\n"
            "Action: tool_name[arguments]\n"
            "[Observation] Results will be shown here.\n"
            "Action: respond[your final response to the user]\n"
        )

        context = f"{system_instructions}\nUser: {user_prompt}\n"

        for iteration in range(max_iterations):
            logger.info("NpDnaAgent running telemetry step %d/%d", iteration + 1, max_iterations)
            current_prompt = context + "[Thought]"
            output = self.core.generate(
                current_prompt,
                max_tokens=128,
                temperature=0.3,
                top_k=5
            )

            new_text = output.strip()
            context += "[Thought] " + new_text + "\n"

            action_match = re.search(r"Action:\s*([a-zA-Z0-9_-]+)\[(.*?)\]", new_text)

            thought_text = new_text.split("Action:")[0].strip()

            step_info = {
                "step": iteration + 1,
                "thought": thought_text,
                "action": None,
                "args": None,
                "observation": None
            }

            if not action_match:
                step_info["action"] = "respond"
                step_info["args"] = new_text
                step_info["observation"] = "Direct completion"
                steps.append(step_info)
                return new_text, steps

            tool_name = action_match.group(1).strip()
            tool_arg = action_match.group(2).strip()
            step_info["action"] = tool_name
            step_info["args"] = tool_arg

            if tool_name == "respond":
                step_info["observation"] = "Final response generated"
                steps.append(step_info)
                return tool_arg, steps

            if tool_name in self.tools:
                try:
                    observation = self.tools[tool_name](tool_arg)
                except Exception as e:
                    observation = f"Error: {str(e)}"
            else:
                observation = f"Unknown tool '{tool_name}'."

            step_info["observation"] = observation
            steps.append(step_info)
            context += f"[Observation] {observation}\n"

        return "Agent failed to reach a conclusion within step limit.", steps


# ============================================================================
# Multimodal — Text-context bridge + optional CLIP image embeddings.
# ============================================================================
#
# Practical multimodal context bridge for NP-DNA.
#
# The default path converts file and structured inputs into explicit text context
# that NP-DNA can consume today.  Image embeddings are available through an
# optional lazy CLIP path for callers that have the local model dependencies.

_CLIP_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


def _png_size(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
        return struct.unpack(">II", header[16:24])
    return None


def _jpeg_size(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as handle:
        data = handle.read()
    if not data.startswith(b"\xff\xd8"):
        return None
    idx = 2
    while idx < len(data) - 9:
        if data[idx] != 0xFF:
            idx += 1
            continue
        marker = data[idx + 1]
        idx += 2
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if idx + 7 <= len(data):
                height = int.from_bytes(data[idx + 3 : idx + 5], "big")
                width = int.from_bytes(data[idx + 5 : idx + 7], "big")
                return width, height
            return None
        if idx + 2 > len(data):
            return None
        segment_len = int.from_bytes(data[idx : idx + 2], "big")
        idx += max(2, segment_len)
    return None


def describe_image(path: str | Path) -> str:
    image = Path(path)
    size = None
    if image.exists():
        suffix = image.suffix.lower()
        if suffix == ".png":
            size = _png_size(image)
        elif suffix in {".jpg", ".jpeg"}:
            size = _jpeg_size(image)
    parts = [
        "Modality: image",
        f"File: {image.name}",
        f"Extension: {image.suffix.lower() or 'unknown'}",
    ]
    if image.exists():
        parts.append(f"Bytes: {image.stat().st_size}")
    if size:
        parts.append(f"Dimensions: {size[0]}x{size[1]}")
    return "\n".join(parts)


def encode_image_clip(
    path: str | Path,
    *,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
):
    """Encode an image into a CLIP feature vector.

    This is intentionally opt-in: importing this module and building text
    prompts does not load CLIP, Pillow, or model weights.  The model is cached
    per ``(model_name, device)`` after first use.
    """
    image = Path(path)
    if not image.exists():
        raise FileNotFoundError(image)

    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError(
            "CLIP image embeddings require torch, pillow, and transformers."
        ) from exc

    cache_key = (model_name, device)
    if cache_key not in _CLIP_CACHE:
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device)
        model.eval()
        _CLIP_CACHE[cache_key] = (processor, model)
    else:
        processor, model = _CLIP_CACHE[cache_key]

    with Image.open(image) as handle:
        pil_image = handle.convert("RGB")
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features = torch.nn.functional.normalize(features, dim=-1)
    return features.squeeze(0).detach().cpu()


def describe_audio(path: str | Path) -> str:
    audio = Path(path)
    parts = [
        "Modality: audio",
        f"File: {audio.name}",
        f"Extension: {audio.suffix.lower() or 'unknown'}",
    ]
    if audio.exists():
        parts.append(f"Bytes: {audio.stat().st_size}")
        if audio.suffix.lower() == ".wav":
            with wave.open(str(audio), "rb") as handle:
                frames = handle.getnframes()
                rate = handle.getframerate()
                duration = frames / rate if rate else 0.0
                parts.extend([
                    f"Channels: {handle.getnchannels()}",
                    f"Sample rate: {rate}",
                    f"Duration seconds: {duration:.3f}",
                ])
    return "\n".join(parts)


def describe_structured(data: dict[str, Any] | list[Any] | str | Path) -> str:
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            value = json.loads(str(data))
    else:
        value = data

    preview = json.dumps(value, ensure_ascii=True, sort_keys=True)
    if len(preview) > 1200:
        preview = preview[:1200] + "..."
    if isinstance(value, dict):
        shape = f"object with {len(value)} keys"
    elif isinstance(value, list):
        shape = f"array with {len(value)} items"
    else:
        shape = type(value).__name__
    return "\n".join(["Modality: structured", f"Shape: {shape}", f"JSON: {preview}"])


def build_multimodal_prompt(
    instruction: str,
    *,
    image: str | Path | None = None,
    audio: str | Path | None = None,
    structured: dict[str, Any] | list[Any] | str | Path | None = None,
) -> str:
    contexts = []
    if image is not None:
        contexts.append(describe_image(image))
    if audio is not None:
        contexts.append(describe_audio(audio))
    if structured is not None:
        contexts.append(describe_structured(structured))

    context = "\n\n".join(contexts) if contexts else "No multimodal context provided."
    return f"System: Use the provided multimodal context.\n{context}\n\nUser: {instruction}\nAssistant:"


# ============================================================================
# Optimise — CPU inference & training helpers.
# ============================================================================
#
# CPU inference & training helpers for NP-DNA models.
#
# The helpers here are deliberately opt-in. Training should keep full precision;
# inference can call these after loading a checkpoint to reduce memory and speed
# up CPU linear layers.

_THREAD_POOL: ThreadPoolExecutor | None = None


def get_thread_pool() -> ThreadPoolExecutor:
    """Return a shared CPU worker pool for parallel inference helpers."""
    global _THREAD_POOL
    if _THREAD_POOL is None:
        workers = max(1, int(os.environ.get("TANTRA_CPU_THREADS", os.cpu_count() or 4)))
        _THREAD_POOL = ThreadPoolExecutor(max_workers=workers)
    return _THREAD_POOL


def enable_torch_cpu_optimizations(num_threads: int | None = None) -> None:
    """Enable conservative PyTorch CPU optimizations."""
    threads = max(1, int(num_threads or os.environ.get("TANTRA_CPU_THREADS", os.cpu_count() or 4)))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, threads)))
    except RuntimeError:
        pass
    try:
        torch.backends.mkldnn.enabled = True
    except Exception:
        pass


def quantize_model_for_cpu(core: Any, inplace: bool = True) -> Any:
    """Apply dynamic INT8 quantization to CPU-friendly Linear layers.

    Args:
        core: ``NpDnaCore``-like object with a ``model`` attribute.
        inplace: when False, returns a quantized copy through PyTorch.

    Returns:
        The same core object, with ``core.model`` replaced by the quantized model.
    """
    if not hasattr(core, "model"):
        raise TypeError("quantize_model_for_cpu expects an object with a model attribute")

    core.model.eval()
    core.model.cpu()
    core.model = torch.ao.quantization.quantize_dynamic(
        core.model,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=inplace,
    )
    return core


def apply_torch_compile(core: Any, mode: str = "reduce-overhead") -> Any:
    """Optionally compile the model for repeated inference calls."""
    if not hasattr(torch, "compile"):
        return core
    try:
        core.model = torch.compile(core.model, mode=mode, backend="inductor")
    except Exception:
        pass
    return core


def model_size_mb(model: nn.Module) -> float:
    """Approximate parameter storage size in MiB."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)


def enable_gradient_checkpointing(core: Any) -> None:
    """Enable gradient checkpointing for memory-efficient training.

    Trades compute for memory: activations are recomputed during backward
    instead of stored. Reduces memory by ~30-50% on CPU with ~20% speed cost.
    """
    if hasattr(core.model, "gradient_checkpointing"):
        core.model.gradient_checkpointing = True


def freeze_for_partial_training(core: Any, train_strands: bool = True,
                                train_embeddings: bool = False) -> int:
    """Freeze all params except genome seeds (and optionally embeddings).

    Enables 'partial training' / fine-tuning: only the strand-generating
    DNA seeds are updated, keeping all router/norm/embedding weights fixed.

    Args:
        core: NpDnaCore instance.
        train_strands: If True, keep genome seeds trainable.
        train_embeddings: If True, also keep embeddings trainable.

    Returns:
        Number of trainable parameters.
    """
    for name, param in core.model.named_parameters():
        if 'genome.seeds' in name:
            param.requires_grad = True
        elif 'embedding' in name and train_embeddings:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return sum(p.numel() for p in core.model.parameters() if p.requires_grad)


def count_active_parameters(model: nn.Module) -> dict[str, int]:
    """Count parameters by group for debugging memory usage."""
    counts = {}
    for name, param in model.named_parameters():
        group = name.split(".")[0] if "." in name else name
        counts[group] = counts.get(group, 0) + param.numel()
    return counts


# ============================================================================
# Benchmark — Checkpoint scoring & release export.
# ============================================================================
#
# Benchmarking and release helpers for NP-DNA checkpoints.

BENCHMARK_TASKS = {
    "conversation": [
        {"q": "Hi! How are you?", "must_contain": ["i", "well", "good", "fine", "help"], "must_not": ["the the", "of of"]},
        {"q": "What is your name?", "must_contain": ["tantra", "i", "am", "my"], "must_not": ["the the"]},
    ],
    "factual": [
        {"q": "What is gravity?", "must_contain": ["force", "mass", "earth", "pull", "weight"], "must_not": ["the the", "of of"]},
        {"q": "What is the capital of France?", "must_contain": ["paris"], "must_not": []},
        {"q": "How many planets are in our solar system?", "must_contain": ["8", "eight"], "must_not": []},
    ],
    "reasoning": [
        {"q": "If I have 5 apples and eat 2, how many do I have?", "must_contain": ["3", "three"], "must_not": []},
        {"q": "What comes after Monday?", "must_contain": ["tuesday"], "must_not": []},
    ],
    "code": [
        {"q": "Write a Python function to add two numbers.", "must_contain": ["def", "return"], "must_not": []},
        {"q": "What does print() do in Python?", "must_contain": ["output", "print", "screen", "display"], "must_not": []},
    ],
}


def score_response(response: str, task: dict) -> float:
    resp_lower = response.lower()
    hits = sum(1 for w in task["must_contain"] if w.lower() in resp_lower)
    fails = sum(1 for w in task["must_not"] if w.lower() in resp_lower)
    score = hits / max(len(task["must_contain"]), 1)
    score -= fails * 0.3
    return max(0.0, min(1.0, score))


def benchmark_checkpoint(
    checkpoint: str | Path = "model/npdna/best",
    *,
    max_tokens: int = 40,
) -> dict:
    checkpoint = Path(checkpoint)
    t0 = time.perf_counter()
    core = NpDnaCore.load(checkpoint)
    load_sec = time.perf_counter() - t0

    domain_scores = {}
    generations = []
    total_chars = 0
    total_sec = 0.0

    for domain, tasks in BENCHMARK_TASKS.items():
        domain_hits = []
        for task in tasks:
            g0 = time.perf_counter()
            text = core.generate(task["q"], max_tokens=max_tokens)
            gen_sec = time.perf_counter() - g0
            total_sec += gen_sec
            total_chars += len(text)
            score = score_response(text, task)
            domain_hits.append(score)
            generations.append({
                "domain": domain,
                "prompt": task["q"],
                "text": text,
                "seconds": gen_sec,
                "chars": len(text),
                "score": score,
            })
        domain_scores[domain] = round(sum(domain_hits) / max(len(domain_hits), 1) * 100, 1)

    meta_path = checkpoint / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    overall = round(sum(domain_scores.values()) / max(len(domain_scores), 1), 1)
    return {
        "checkpoint": str(checkpoint),
        "load_seconds": load_sec,
        "generation_seconds": total_sec,
        "chars_per_second": total_chars / max(total_sec, 1e-9),
        "overall_score": overall,
        "domain_scores": domain_scores,
        "metadata": {
            "step": metadata.get("step"),
            "val_loss": metadata.get("val_loss"),
            "parameter_count": metadata.get("parameter_count"),
            "active_parameter_count": metadata.get("active_parameter_count"),
            "vocab_size": metadata.get("vocab_size"),
            "vocab_capacity": metadata.get("vocab_capacity"),
            "hidden_size": metadata.get("hidden_size"),
        },
        "generations": generations,
    }


def write_benchmark(
    checkpoint: str | Path = "model/npdna/best",
    output: str | Path | None = None,
    *,
    max_tokens: int = 40,
) -> dict:
    result = benchmark_checkpoint(checkpoint, max_tokens=max_tokens)
    out_path = Path(output) if output else Path(checkpoint) / "benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    return result


def export_release(
    version: str,
    checkpoint: str | Path = "model/npdna/best",
    releases_dir: str | Path = "model/releases",
) -> Path:
    checkpoint = Path(checkpoint)
    dest = Path(releases_dir) / version
    dest.mkdir(parents=True, exist_ok=True)

    for name in ["model.pt", "tokenizer.json", "metadata.json", "benchmark.json"]:
        src = checkpoint / name
        if src.exists():
            shutil.copy2(src, dest / name)

    cortex_src = checkpoint / "cortex"
    if cortex_src.exists():
        cortex_dest = dest / "cortex"
        if cortex_dest.exists():
            shutil.rmtree(cortex_dest)
        shutil.copytree(cortex_src, cortex_dest)

    bench_path = dest / "benchmark.json"
    if not bench_path.exists():
        write_benchmark(checkpoint, bench_path)

    samples = json.loads(bench_path.read_text(encoding="utf-8"))["generations"]
    (dest / "samples.txt").write_text(
        "\n\n".join(f"Q: {item['prompt']}\nA: {item['text']}" for item in samples),
        encoding="utf-8",
    )
    return dest


def benchmark_main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark an NP-DNA checkpoint.")
    parser.add_argument("--checkpoint", default="model/npdna/best")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-tokens", type=int, default=40)
    args = parser.parse_args()
    result = write_benchmark(args.checkpoint, args.output, max_tokens=args.max_tokens)
    print(json.dumps(result["metadata"], indent=2))
    print(f"load_seconds={result['load_seconds']:.3f}")
    print(f"chars_per_second={result['chars_per_second']:.1f}")
    print(f"Domain scores: {result['domain_scores']}")
    print(f"Overall: {result['overall_score']}%")


def release_main() -> None:
    parser = argparse.ArgumentParser(description="Export a versioned NP-DNA release.")
    parser.add_argument("version", help="Version folder name, e.g. npdna-seed-v0.1")
    parser.add_argument("--checkpoint", default="model/npdna/best")
    parser.add_argument("--releases-dir", default="model/releases")
    args = parser.parse_args()
    dest = export_release(args.version, args.checkpoint, args.releases_dir)
    print(dest)


# ============================================================================
# Codec registry — Frozen multimodal codec lookup.
# ============================================================================
#
# Frozen multimodal codec registry.
#
# Codecs are tokenizer-like adapters around external audio/image/video systems.
# They are referenced by checkpoints, but their weights are never part of NP-DNA
# training. Until real codec packages are configured, this module fails clearly.

Modality = Literal["audio", "image", "video"]


@dataclass(frozen=True)
class FrozenCodecRef:
    modality: Modality
    uri: str | None
    trainable: bool = False
    frozen: bool = True

    @property
    def available(self) -> bool:
        return bool(self.uri)


class FrozenCodecRegistry:
    """Lookup table for tokenizer-like frozen multimodal codecs."""

    def __init__(self, config: CodecConfig):
        self.refs = {
            "audio": FrozenCodecRef("audio", config.audio_codec),
            "image": FrozenCodecRef("image", config.image_codec),
            "video": FrozenCodecRef("video", config.video_codec),
        }

    @classmethod
    def from_config(cls, config: CodecConfig) -> "FrozenCodecRegistry":
        return cls(config)

    @classmethod
    def default(cls) -> "FrozenCodecRegistry":
        return cls(CodecConfig())

    def describe(self) -> dict[str, dict]:
        return {
            name: {
                **vars(ref),
                "available": ref.available,
            }
            for name, ref in self.refs.items()
        }

    def encode(self, modality: Modality, payload: bytes) -> list[int]:
        ref = self.refs[modality]
        if not ref.available:
            raise NotImplementedError(f"No frozen {modality} codec is configured.")
        raise NotImplementedError(f"Frozen {modality} codec adapter is referenced at {ref.uri}, but not installed.")

    def decode(self, modality: Modality, tokens: list[int]) -> bytes:
        ref = self.refs[modality]
        if not ref.available:
            raise NotImplementedError(f"No frozen {modality} codec is configured.")
        raise NotImplementedError(f"Frozen {modality} codec adapter is referenced at {ref.uri}, but not installed.")
