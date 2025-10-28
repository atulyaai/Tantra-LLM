# Path to v1.0-stable_identity (Batch Implementation Plan)

## v0.5-self_shaping (Adaptive Personality)
- Wire PersonalityLayer.select_mode() into decision engine
- Parameterizer stub that maps mode → decoding params
- Session mode persistence (override stays until changed)
- Test: mode switching works

Files:
- core/control/decision_engine.py (wire personality)
- personality/personality_layer.py (implement session mode)
- personality/parameterizer.py (mode → params mapping)
- tests/test_personality_modes.py

## v0.6-fusion_wiring
- Complete FusionOrchestrator.build_stream() to handle gate tokens properly
- Add actual embedding tensor shapes to projectors
- Test: fusion creates valid stream

Files:
- core/fusion/orchestrator.py (handle embeddings correctly)
- core/fusion/sensory_projectors.py (real tensor shapes)
- tests/test_fusion_stream.py

## v0.7-fusion_training
- Stub training loop (no actual model training yet)
- Dataset loading infrastructure
- Checkpoint saving logic
- Test: can configure training

Files:
- training/fusion_trainer.py (stub loop)
- training/datasets/multimodal_dataset.py (actual data loading)
- tests/test_training_pipeline.py

## v0.8-dynamic_compute
- ComputeRouter.route() actually switches paths
- Implement fast/medium/deep response generation
- Test: routing works

Files:
- core/models/compute_routing.py (implement routing)
- core/control/response_generator.py (add path variants)
- tests/test_compute_routing.py

## v0.9-safety_personality
- Safety module returns pass/modify/deny decisions
- Values module outputs value vector
- Style module maps values → decoding params
- Test: safety gates work

Files:
- personality/safety_module.py (implement rules)
- personality/values_module.py (output vector)
- personality/style_module.py (params mapping)
- tests/test_safety_gates.py

## v1.0-stable_identity
- Integration polish: all components work together
- End-to-end smoke test
- Final README + CHANGELOG
- Tag v1.0-stable_identity

Files:
- tests/test_e2e_brain.py (full system test)
- README.md (final update)
- CHANGELOG.md (v1.0 entry)

