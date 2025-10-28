# Changelog

All notable changes are tracked by capability versions and tags.

## v1.0-stable_identity
- Production System Complete + FastAPI Server
- Tag: v1.0-stable_identity

Changed/Added files:
- training/fusion_trainer.py (training loop with frozen base, optimizer, checkpoints)
- core/models/compute_routing.py (fast/medium/deep path routing)
- personality/safety_module.py (deny-list + toxicity checks operational)
- tests/test_e2e_brain.py (end-to-end integration tests)
- demos/api_server.py (FastAPI production server)
- CHANGELOG.md, README.md (updated for v1.0)

Notes:
- Complete production system: SpikingBrain + Long-VITA + Whisper + Fusion + Safety
- Training pipeline operational (freeze base, train projectors)
- Compute routing (fast/medium/deep) based on query complexity
- Safety modules operational (deny-list, toxicity patterns)
- FastAPI server ready for deployment
- System can generate real multimodal responses

## v0.8-inference_live
- SpikingBrain Forward Pass + Embeddings Injection
- Tag: v0.8-inference_live

Changed/Added files:
- core/control/response_generator.py (actual model.generate() with merged embeddings)
- core/fusion/orchestrator.py (merge_embeddings() method for gate token replacement)
- encoders/vision.py (replaced CLIP with Long-VITA placeholder)
- utils/model_loader.py (updated to use AutoModelForCausalLM from HuggingFace)
- tests/test_inference.py (smoke test for forward pass)

Notes:
- SpikingBrain now calls actual model.generate() with embeddings injection
- Gate tokens replaced with projected modality embeddings in forward pass
- Uses HuggingFace AutoModelForCausalLM for generation support

## v0.7-fusion_production
- Real Encoder Integration
- Tag: v0.7-fusion_production

Changed/Added files:
- encoders/vision.py (real CLIP encoder; graceful fallback)
- encoders/audio.py (real Whisper encoder; graceful fallback)
- VERSION_SUMMARY.md (capability tracking)
- CHANGELOG.md (this entry)

Notes:
- Production encoders wired with graceful fallback to stubs if deps missing

## v0.6-fusion_wiring
- Fusion Stream Wiring & Shape Validation
- Tag: v0.6-fusion_wiring

Changed/Added files:
- core/fusion/sensory_projectors.py (input shape handling, output dim checks)
- tests/test_fusion_stream.py (gate tokens present; projected dims == D)

Notes:
- IMG/AUD gates present in stream; modality_embeds aligned and non-empty

## v0.5-self_shaping
- Adaptive Personality Routing
- Tag: v0.5-self_shaping

Changed/Added files:
- personality/personality_layer.py (session mode persistence; auto cues used)
- tests/test_personality_modes.py (auto + override persistence; parameterizer mapping)

Notes:
- Personality mode auto-detects or respects explicit override until changed

## v0.4-understands_relations
- Semantic Graph Influence
- Tag: v0.4-understands_relations

Changed/Added files:
- core/control/brain_orchestrator.py (exposes _last_context_prompt; includes semantic + episodic context)
- tests/test_semantic_graph.py (smoke test verifying semantic facts appear in context)

Notes:
- Semantic facts are prepended alongside episodic recalls when present

## v0.3-remembers
- Episodic Memory Integration
- Tag: v0.3-remembers

Changed/Added files:
- core/memory/episodic_memory.py (clear(), size() methods added, docstrings improved)
- core/control/brain_orchestrator.py (prepends retrieved memories to prompt before reasoning)
- tests/test_demo_minimal.py (smoke tests for demo wiring and episodic retrieval)
- CHANGELOG.md (this entry)

Notes:
- Episodic memory now influences response generation
- Retrieved memories prepended as context before perception step
- Smoke test confirms memory integration works

## v0.2-eyes_open (alias: v0.2)
- Dynamic Architecture + Flattened Structure
- Tags: `v0.2-eyes_open`, `v0.2`

Changed/Added files:
- README.md (updated to flattened imports and structure)
- requirements.txt (merged, modernized deps)
- config/identity.py (behavioral profiles)
- core/models/dynamic_context.py (dynamic window routing)
- utils/model_loader.py (mixed local/API loader)
- demos/demo_minimal.py (flattened imports)
- LICENSE (added)
- .github/REPOSITORY_SETTINGS.md (repo settings guide)
- .github/repository-configure.ps1, .github/repository-configure.sh (automation scripts)
- .github/topics.txt, .github/description.txt, .github/website.txt (metadata)

Removed/Refactored:
- Flattened all modules from `tantra_llm/*` → top-level `config/`, `core/`, `encoders/`, `personality/`, `training/`, `utils/`, `demos/`
- Deleted stale tests with old import paths (`tests/test_phase1_*`, `tests/test_phase2_*`)
- Deleted `ABOUT.md` per owner request

## v0.1-origins (alias: v0.10)
- Core Architecture Scaffolding
- Tags: `v0.1-origins`, `v0.10`

Changed/Added files:
- Initial scaffolding of config/, core/, encoders/, personality/, training/, utils/, demos/
- Initial README.md
- Initial requirements.txt

Notes:
- Git workflow established (semantic commits, capability tags)
- Baseline tags created

---

Release notes template lives at `.github/RELEASE_NOTES_TEMPLATE.md`.
