# Expert coverage snapshot

Regenerate with `python -m Tantra.data.expert_workbench report`. This is a local-data snapshot, not live telemetry.

## Summary

- Registered capabilities: 164; taxonomy weight files: 19; supplementary: hindi_grammar.
- Audited rows: 1,584,322; rejected filler: 1,336,440.
- Structurally usable chat: 44,272; reviewed/verifiable: 15,000.
- Prepared examples: 15,000; pilot-ready experts: 3.
- Legacy examples without provenance/review remain excluded. Saved weights are not proof of answer quality.
- Generated task templates are narrow bootstrap coverage, not production-ready capabilities. Keep unsupported or redundant experts disabled.

## Prepared bootstrap

| Expert | Train | Validation | Test | Templates | Languages |
| --- | ---: | ---: | ---: | ---: | --- |
| math | 4481 | 250 | 269 | 3 | hinglish, en, hi |
| json_schema | 4502 | 265 | 233 | 1 | en |
| csv_export | 4515 | 252 | 233 | 1 | en |

## Expert inventory

| Expert | Domain | Raw rows | Usable chat | Reviewed | Prepared state | Weights |
| --- | --- | ---: | ---: | ---: | --- | --- |
| [greeting](experts/conversation/greeting.md) | conversation | 12250 | 2250 | 0 | needs_data | present, unvalidated |
| [farewell](experts/conversation/farewell.md) | conversation | 11650 | 1650 | 0 | needs_data | present, unvalidated |
| [smalltalk](experts/conversation/smalltalk.md) | conversation | 11650 | 1650 | 0 | needs_data | present, unvalidated |
| [empathy](experts/conversation/empathy.md) | conversation | 11650 | 1650 | 0 | needs_data | present, unvalidated |
| [clarification](experts/conversation/clarification.md) | conversation | 11500 | 1500 | 0 | needs_data | present, unvalidated |
| [apology](experts/conversation/apology.md) | conversation | 11500 | 1500 | 0 | needs_data | present, unvalidated |
| [humor](experts/conversation/humor.md) | conversation | 11650 | 1650 | 0 | needs_data | present, unvalidated |
| [identity_self](experts/conversation/identity_self.md) | conversation | 11650 | 1650 | 0 | needs_data | present, unvalidated |
| [identity_capabilities](experts/conversation/identity_capabilities.md) | conversation | 11650 | 1650 | 0 | needs_data | present, unvalidated |
| [identity_limitations](experts/conversation/identity_limitations.md) | conversation | 11650 | 1650 | 0 | needs_data | present, unvalidated |
| [identity_persona](experts/conversation/identity_persona.md) | conversation | 10000 | 0 | 0 | needs_data | absent |
| [identity_consistency](experts/conversation/identity_consistency.md) | conversation | 10000 | 0 | 0 | needs_data | absent |
| [identity_metadata](experts/conversation/identity_metadata.md) | conversation | 10000 | 0 | 0 | needs_data | absent |
| [tone_style](experts/conversation/tone_style.md) | conversation | 10000 | 0 | 0 | needs_data | absent |
| [safety_refusal](experts/conversation/safety_refusal.md) | conversation | 11500 | 1500 | 0 | needs_data | present, unvalidated |
| [gk_general](experts/knowledge/gk_general.md) | knowledge | 10000 | 0 | 0 | needs_data | absent |
| [gk_science](experts/knowledge/gk_science.md) | knowledge | 14000 | 4000 | 0 | needs_data | present, unvalidated |
| [gk_history](experts/knowledge/gk_history.md) | knowledge | 10000 | 0 | 0 | needs_data | absent |
| [gk_geography](experts/knowledge/gk_geography.md) | knowledge | 10972 | 972 | 0 | needs_data | present, unvalidated |
| [gk_arts](experts/knowledge/gk_arts.md) | knowledge | 10000 | 0 | 0 | needs_data | absent |
| [gk_tech](experts/knowledge/gk_tech.md) | knowledge | 10000 | 0 | 0 | needs_data | absent |
| [gk_math](experts/knowledge/gk_math.md) | knowledge | 14000 | 4000 | 0 | needs_data | present, unvalidated |
| [gk_physics](experts/knowledge/gk_physics.md) | knowledge | 10000 | 0 | 0 | needs_data | absent |
| [gk_chemistry](experts/knowledge/gk_chemistry.md) | knowledge | 10000 | 0 | 0 | needs_data | absent |
| [gk_biology](experts/knowledge/gk_biology.md) | knowledge | 10000 | 0 | 0 | needs_data | absent |
| [gk_medicine](experts/knowledge/gk_medicine.md) | knowledge | 10000 | 0 | 0 | needs_data | absent |
| [gk_law](experts/knowledge/gk_law.md) | knowledge | 10000 | 0 | 0 | needs_data | absent |
| [gk_finance](experts/knowledge/gk_finance.md) | knowledge | 10000 | 0 | 0 | needs_data | absent |
| [gk_programming](experts/knowledge/gk_programming.md) | knowledge | 10000 | 0 | 0 | needs_data | absent |
| [gk_ai_ml](experts/knowledge/gk_ai_ml.md) | knowledge | 10000 | 0 | 0 | needs_data | absent |
| [gk_cybersec](experts/knowledge/gk_cybersec.md) | knowledge | 10000 | 0 | 0 | needs_data | absent |
| [gk_linguistics](experts/knowledge/gk_linguistics.md) | knowledge | 10000 | 0 | 0 | needs_data | absent |
| [gk_philosophy](experts/knowledge/gk_philosophy.md) | knowledge | 10000 | 0 | 0 | needs_data | absent |
| [gk_psychology](experts/knowledge/gk_psychology.md) | knowledge | 10000 | 0 | 0 | needs_data | absent |
| [gk_economics](experts/knowledge/gk_economics.md) | knowledge | 10000 | 0 | 0 | needs_data | absent |
| [gk_politics](experts/knowledge/gk_politics.md) | knowledge | 10000 | 0 | 0 | needs_data | absent |
| [gk_sports](experts/knowledge/gk_sports.md) | knowledge | 10000 | 0 | 0 | needs_data | absent |
| [gk_entertainment](experts/knowledge/gk_entertainment.md) | knowledge | 10000 | 0 | 0 | needs_data | absent |
| [gk_current_events](experts/knowledge/gk_current_events.md) | knowledge | 0 | 0 | 0 | needs_data | absent |
| [gk_trivia](experts/knowledge/gk_trivia.md) | knowledge | 10000 | 0 | 0 | needs_data | absent |
| [logic](experts/reasoning/logic.md) | reasoning | 10400 | 400 | 0 | needs_data | present, unvalidated |
| [math](experts/reasoning/math.md) | reasoning | 15000 | 5000 | 5000 | data_ready | absent |
| [coding](experts/reasoning/coding.md) | reasoning | 10000 | 0 | 0 | needs_data | absent |
| [reasoning_planning](experts/reasoning/reasoning_planning.md) | reasoning | 10000 | 0 | 0 | needs_data | absent |
| [debugging](experts/reasoning/debugging.md) | reasoning | 10000 | 0 | 0 | needs_data | absent |
| [algorithm_design](experts/reasoning/algorithm_design.md) | reasoning | 10000 | 0 | 0 | needs_data | absent |
| [code_review](experts/reasoning/code_review.md) | reasoning | 10000 | 0 | 0 | needs_data | absent |
| [architecture](experts/reasoning/architecture.md) | reasoning | 10000 | 0 | 0 | needs_data | absent |
| [optimization](experts/reasoning/optimization.md) | reasoning | 10000 | 0 | 0 | needs_data | absent |
| [verification](experts/reasoning/verification.md) | reasoning | 10000 | 0 | 0 | needs_data | absent |
| [symbolic_reasoning](experts/reasoning/symbolic_reasoning.md) | reasoning | 10000 | 0 | 0 | needs_data | absent |
| [causal_inference](experts/reasoning/causal_inference.md) | reasoning | 10000 | 0 | 0 | needs_data | absent |
| [pii](experts/safety/pii.md) | safety | 10000 | 0 | 0 | needs_data | absent |
| [toxicity](experts/safety/toxicity.md) | safety | 10000 | 0 | 0 | needs_data | absent |
| [jailbreak](experts/safety/jailbreak.md) | safety | 10000 | 0 | 0 | needs_data | absent |
| [bias](experts/safety/bias.md) | safety | 10000 | 0 | 0 | needs_data | absent |
| [medical_safety](experts/safety/medical_safety.md) | safety | 10000 | 0 | 0 | needs_data | absent |
| [financial_safety](experts/safety/financial_safety.md) | safety | 10000 | 0 | 0 | needs_data | absent |
| [legal_safety](experts/safety/legal_safety.md) | safety | 10000 | 0 | 0 | needs_data | absent |
| [child_safety](experts/safety/child_safety.md) | safety | 10000 | 0 | 0 | needs_data | absent |
| [self_harm](experts/safety/self_harm.md) | safety | 10000 | 0 | 0 | needs_data | absent |
| [misinformation](experts/safety/misinformation.md) | safety | 10000 | 0 | 0 | needs_data | absent |
| [summarization](experts/tasks/summarization.md) | tasks | 10400 | 400 | 0 | needs_data | present, unvalidated |
| [translation](experts/tasks/translation.md) | tasks | 10400 | 400 | 0 | needs_data | present, unvalidated |
| [qa](experts/tasks/qa.md) | tasks | 10000 | 0 | 0 | needs_data | absent |
| [creative_writing](experts/tasks/creative_writing.md) | tasks | 10400 | 400 | 0 | needs_data | present, unvalidated |
| [analysis](experts/tasks/analysis.md) | tasks | 10000 | 0 | 0 | needs_data | absent |
| [extraction](experts/tasks/extraction.md) | tasks | 10000 | 0 | 0 | needs_data | absent |
| [formatting](experts/tasks/formatting.md) | tasks | 10000 | 0 | 0 | needs_data | absent |
| [rewriting](experts/tasks/rewriting.md) | tasks | 10000 | 0 | 0 | needs_data | absent |
| [classification](experts/tasks/classification.md) | tasks | 10000 | 0 | 0 | needs_data | absent |
| [comparison](experts/tasks/comparison.md) | tasks | 10000 | 0 | 0 | needs_data | absent |
| [synthesis](experts/tasks/synthesis.md) | tasks | 10000 | 0 | 0 | needs_data | absent |
| [evaluation](experts/tasks/evaluation.md) | tasks | 10000 | 0 | 0 | needs_data | absent |
| [planning](experts/agentic/planning.md) | agentic | 10000 | 0 | 0 | needs_data | absent |
| [task_decomposition](experts/agentic/task_decomposition.md) | agentic | 10000 | 0 | 0 | needs_data | absent |
| [tool_selection](experts/agentic/tool_selection.md) | agentic | 10000 | 0 | 0 | needs_data | absent |
| [state_tracking](experts/agentic/state_tracking.md) | agentic | 10000 | 0 | 0 | needs_data | absent |
| [error_recovery](experts/agentic/error_recovery.md) | agentic | 10000 | 0 | 0 | needs_data | absent |
| [multi_step_reasoning](experts/agentic/multi_step_reasoning.md) | agentic | 10000 | 0 | 0 | needs_data | absent |
| [api_schema](experts/tool_calling/api_schema.md) | tool_calling | 10000 | 0 | 0 | needs_data | absent |
| [function_composition](experts/tool_calling/function_composition.md) | tool_calling | 10000 | 0 | 0 | needs_data | absent |
| [parallel_calls](experts/tool_calling/parallel_calls.md) | tool_calling | 10000 | 0 | 0 | needs_data | absent |
| [streaming_results](experts/tool_calling/streaming_results.md) | tool_calling | 10000 | 0 | 0 | needs_data | absent |
| [auth_handling](experts/tool_calling/auth_handling.md) | tool_calling | 10000 | 0 | 0 | needs_data | absent |
| [tool_error_handling](experts/tool_calling/tool_error_handling.md) | tool_calling | 10000 | 0 | 0 | needs_data | absent |
| [json_schema](experts/structured_output/json_schema.md) | structured_output | 15000 | 5000 | 5000 | data_ready | absent |
| [xml_format](experts/structured_output/xml_format.md) | structured_output | 10000 | 0 | 0 | needs_data | absent |
| [markdown_tables](experts/structured_output/markdown_tables.md) | structured_output | 10000 | 0 | 0 | needs_data | absent |
| [csv_export](experts/structured_output/csv_export.md) | structured_output | 15000 | 5000 | 5000 | data_ready | absent |
| [regex_validation](experts/structured_output/regex_validation.md) | structured_output | 10000 | 0 | 0 | needs_data | absent |
| [query_rewrite](experts/rag/query_rewrite.md) | rag | 10000 | 0 | 0 | needs_data | absent |
| [doc_ranking](experts/rag/doc_ranking.md) | rag | 10000 | 0 | 0 | needs_data | absent |
| [citation_extraction](experts/rag/citation_extraction.md) | rag | 10000 | 0 | 0 | needs_data | absent |
| [hallucination_check](experts/rag/hallucination_check.md) | rag | 10000 | 0 | 0 | needs_data | absent |
| [context_compression](experts/rag/context_compression.md) | rag | 10000 | 0 | 0 | needs_data | absent |
| [repo_navigation](experts/code_agent/repo_navigation.md) | code_agent | 10000 | 0 | 0 | needs_data | absent |
| [test_generation](experts/code_agent/test_generation.md) | code_agent | 10000 | 0 | 0 | needs_data | absent |
| [debug_loop](experts/code_agent/debug_loop.md) | code_agent | 10000 | 0 | 0 | needs_data | absent |
| [refactor_planning](experts/code_agent/refactor_planning.md) | code_agent | 10000 | 0 | 0 | needs_data | absent |
| [code_search](experts/code_agent/code_search.md) | code_agent | 10000 | 0 | 0 | needs_data | absent |
| [dependency_analysis](experts/code_agent/dependency_analysis.md) | code_agent | 10000 | 0 | 0 | needs_data | absent |
| [lang_en](experts/multilingual/lang_en.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_zh](experts/multilingual/lang_zh.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_es](experts/multilingual/lang_es.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_hi](experts/multilingual/lang_hi.md) | multilingual | 10400 | 400 | 0 | needs_data | present, unvalidated |
| [lang_ar](experts/multilingual/lang_ar.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_pt](experts/multilingual/lang_pt.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_bn](experts/multilingual/lang_bn.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_ru](experts/multilingual/lang_ru.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_ja](experts/multilingual/lang_ja.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_de](experts/multilingual/lang_de.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_ko](experts/multilingual/lang_ko.md) | multilingual | 0 | 0 | 0 | needs_data | absent |
| [lang_fr](experts/multilingual/lang_fr.md) | multilingual | 0 | 0 | 0 | needs_data | absent |
| [lang_tr](experts/multilingual/lang_tr.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_vi](experts/multilingual/lang_vi.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_it](experts/multilingual/lang_it.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_th](experts/multilingual/lang_th.md) | multilingual | 50 | 0 | 0 | needs_data | absent |
| [lang_pl](experts/multilingual/lang_pl.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_nl](experts/multilingual/lang_nl.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_id](experts/multilingual/lang_id.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_sv](experts/multilingual/lang_sv.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_cs](experts/multilingual/lang_cs.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_el](experts/multilingual/lang_el.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_he](experts/multilingual/lang_he.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_ro](experts/multilingual/lang_ro.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_hu](experts/multilingual/lang_hu.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_fi](experts/multilingual/lang_fi.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_da](experts/multilingual/lang_da.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_no](experts/multilingual/lang_no.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_sk](experts/multilingual/lang_sk.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_hr](experts/multilingual/lang_hr.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_sl](experts/multilingual/lang_sl.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_et](experts/multilingual/lang_et.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_lv](experts/multilingual/lang_lv.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_lt](experts/multilingual/lang_lt.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_mt](experts/multilingual/lang_mt.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_ga](experts/multilingual/lang_ga.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_cy](experts/multilingual/lang_cy.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_eu](experts/multilingual/lang_eu.md) | multilingual | 0 | 0 | 0 | needs_data | absent |
| [lang_ca](experts/multilingual/lang_ca.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_gl](experts/multilingual/lang_gl.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_is](experts/multilingual/lang_is.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_mk](experts/multilingual/lang_mk.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_sq](experts/multilingual/lang_sq.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_bs](experts/multilingual/lang_bs.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_me](experts/multilingual/lang_me.md) | multilingual | 10000 | 0 | 0 | needs_data | absent |
| [lang_sr](experts/multilingual/lang_sr.md) | multilingual | 0 | 0 | 0 | needs_data | absent |
| [lang_bg](experts/multilingual/lang_bg.md) | multilingual | 0 | 0 | 0 | needs_data | absent |
| [lang_uk](experts/multilingual/lang_uk.md) | multilingual | 0 | 0 | 0 | needs_data | absent |
| [lang_be](experts/multilingual/lang_be.md) | multilingual | 0 | 0 | 0 | needs_data | absent |
| [lang_ka](experts/multilingual/lang_ka.md) | multilingual | 0 | 0 | 0 | needs_data | absent |
| [translate_pivot](experts/translation/translate_pivot.md) | translation | 10000 | 0 | 0 | needs_data | absent |
| [translate_quality](experts/translation/translate_quality.md) | translation | 10000 | 0 | 0 | needs_data | absent |
| [translate_terminology](experts/translation/translate_terminology.md) | translation | 10000 | 0 | 0 | needs_data | absent |
| [translate_formality](experts/translation/translate_formality.md) | translation | 10000 | 0 | 0 | needs_data | absent |
| [translate_domain](experts/translation/translate_domain.md) | translation | 10000 | 0 | 0 | needs_data | absent |
| [translate_literary](experts/translation/translate_literary.md) | translation | 10000 | 0 | 0 | needs_data | absent |
| [translate_technical](experts/translation/translate_technical.md) | translation | 10000 | 0 | 0 | needs_data | absent |
| [translate_low_resource](experts/translation/translate_low_resource.md) | translation | 10000 | 0 | 0 | needs_data | absent |
| [retrieval](experts/long_context/retrieval.md) | long_context | 10000 | 0 | 0 | needs_data | absent |
| [hierarchy](experts/long_context/hierarchy.md) | long_context | 10000 | 0 | 0 | needs_data | absent |
| [compression](experts/long_context/compression.md) | long_context | 10000 | 0 | 0 | needs_data | absent |
| [rnn_memory](experts/long_context/rnn_memory.md) | long_context | 10000 | 0 | 0 | needs_data | absent |

## Continue

Prioritize reviewed Hindi/Hinglish/English conversation and identity examples, expand task diversity, then train isolated candidates. Read HANDOFF.md and DATASET_SPEC.md before adding data. Use new immutable bundle directories for new versions. Evaluate answer quality, router behavior and regressions before deployment.
