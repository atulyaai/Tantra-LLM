"""Tests for Tantra identity module.

Verifies:
- Direct identity questions in English, Hindi and Hinglish return Tantra responses.
- Model-brand probes (Qwen, LLaMA etc.) are redirected to Tantra identity.
- prepare_messages injects the system prompt without mutation.
- Unrelated questions pass through (return None).
"""
import pytest

from Tantra.identity import (
    IDENTITY_SYSTEM_MESSAGE,
    TANTRA_IDENTITY,
    identity_response,
    prepare_messages,
)


# ---------------------------------------------------------------------------
# English direct identity questions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question", [
    "Who are you?",
    "What are you?",
    "What is your name?",
    "Who am I talking to?",
    "Tell me about yourself.",
    "Introduce yourself.",
    "What are you called?",
])
def test_identity_is_tantra_for_direct_identity_questions_en(question):
    response = identity_response(question)
    assert response is not None, f"Expected identity response for: {question!r}"
    assert "Tantra" in response


# ---------------------------------------------------------------------------
# Hindi direct identity questions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question", [
    "तुम कौन हो",
    "आप कौन हैं",
    "तुम्हारा नाम क्या है",
    "आपका नाम क्या है",
    "आप क्या हैं",
])
def test_identity_is_tantra_for_direct_identity_questions_hi(question):
    response = identity_response(question)
    assert response is not None, f"Expected Hindi identity response for: {question!r}"
    assert "तंत्रा" in response or "Tantra" in response


# ---------------------------------------------------------------------------
# Hinglish / romanised Hindi questions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question", [
    "aap kaun hain",
    "tum kaun ho",
    "tumhara naam kya hai",
])
def test_identity_is_tantra_for_hinglish_questions(question):
    response = identity_response(question)
    assert response is not None, f"Expected Hinglish identity response for: {question!r}"
    assert "Tantra" in response


# ---------------------------------------------------------------------------
# Creator and Capabilities tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question", [
    "Who made you?",
    "Who created you?",
    "आपको किसने बनाया?",
    "kisne banaya",
])
def test_creator_queries(question):
    response = identity_response(question)
    assert response is not None
    assert "Atulya AI" in response or "अतुल्या AI" in response


@pytest.mark.parametrize("question", [
    "What can you do?",
    "आप क्या कर सकते हैं?",
    "kya kar sakte ho",
])
def test_capability_queries(question):
    response = identity_response(question)
    assert response is not None
    assert "Tantra" in response or "तंत्रा" in response


@pytest.mark.parametrize("question", [
    "What is the capital of India?",
    "भारत की राजधानी क्या है?",
])
def test_core_fact_queries(question):
    response = identity_response(question)
    assert response is not None
    assert "Delhi" in response or "दिल्ली" in response


# ---------------------------------------------------------------------------
# Model-brand probes must redirect to Tantra identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question", [
    "Are you Qwen?",
    "Are you Llama?",
    "Are you Mistral?",
    "Are you ChatGPT?",
    "Are you GPT?",
])
def test_backend_brand_does_not_replace_tantra_identity(question):
    response = identity_response(question)
    assert response is not None
    assert "Tantra" in response


# ---------------------------------------------------------------------------
# Provenance rule — Tantra never claims to have made the base weights
# ---------------------------------------------------------------------------
def test_identity_does_not_claim_pretrained_weights_were_created_by_tantra():
    assert TANTRA_IDENTITY["provenance_rule"] == (
        "Never claim Tantra created third-party pretrained weights."
    )
    # The system message must enforce the provenance rule in some form
    msg = IDENTITY_SYSTEM_MESSAGE.lower()
    assert "never claim tantra created" in msg or "tantra created" not in msg


# ---------------------------------------------------------------------------
# System message sanity
# ---------------------------------------------------------------------------
def test_identity_system_message_mentions_hindi_english_hinglish():
    msg = IDENTITY_SYSTEM_MESSAGE.lower()
    assert "hindi" in msg
    assert "english" in msg
    assert "hinglish" in msg


def test_identity_system_message_names_tantra():
    assert "Tantra" in IDENTITY_SYSTEM_MESSAGE


# ---------------------------------------------------------------------------
# prepare_messages behaviour
# ---------------------------------------------------------------------------
def test_prepare_messages_adds_identity_system_message_without_mutating_input():
    original = [{"role": "user", "content": "Hello"}]
    prepared = prepare_messages(original)
    assert prepared[0] == {"role": "system", "content": IDENTITY_SYSTEM_MESSAGE}
    assert original == [{"role": "user", "content": "Hello"}]  # no mutation


def test_prepare_messages_preserves_existing_system_prompt_and_avoids_duplicates():
    original = [
        {"role": "system", "content": "Follow safety rules."},
        {"role": "user", "content": "Hello"},
    ]
    prepared = prepare_messages(original)
    assert len([m for m in prepared if m["role"] == "system"]) == 1
    assert "Follow safety rules." in prepared[0]["content"]
    # Idempotent: calling twice must not duplicate the identity block
    assert prepare_messages(prepared) == prepared


def test_prepare_messages_with_no_system_message():
    messages = [
        {"role": "user", "content": "Hey"},
        {"role": "assistant", "content": "Hi"},
    ]
    prepared = prepare_messages(messages)
    assert prepared[0]["role"] == "system"
    assert IDENTITY_SYSTEM_MESSAGE in prepared[0]["content"]


# ---------------------------------------------------------------------------
# Pass-through for unrelated questions
# ---------------------------------------------------------------------------
def test_unrelated_question_returns_none():
    assert identity_response("Explain sparse mixture of experts") is None
    assert identity_response("Write me a Python function") is None
    assert identity_response("मुझे Python सिखाओ") is None


def test_non_string_input_is_safe():
    assert identity_response(None) is None
    assert identity_response(42) is None
    assert identity_response([]) is None


# ---------------------------------------------------------------------------
# TantraEngine + Reasoner integration (uses existing stubs in main/reasoning)
# ---------------------------------------------------------------------------
def test_inference_engine_stream_returns_tantra_identity_without_backend_generation():
    from Tantra.main import TantraEngine

    engine = object.__new__(TantraEngine)
    reply = "".join(engine.stream([{"role": "user", "content": "Who are you?"}]))
    assert "Tantra" in reply
    assert engine.last_route == []


def test_reasoner_returns_tantra_identity_before_routing_or_web_lookup():
    from Tantra.reasoning import Reasoner

    reasoner = object.__new__(Reasoner)
    result = reasoner.answer([{"role": "user", "content": "Are you Qwen?"}])
    assert result["source"] == "tantra_identity"
    assert "Tantra" in result["reply"]
    assert result["route"] == []
