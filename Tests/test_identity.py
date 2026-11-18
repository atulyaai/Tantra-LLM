import pytest

from Tantra.identity import (
    IDENTITY_SYSTEM_MESSAGE,
    TANTRA_IDENTITY,
    identity_response,
    prepare_messages,
)


@pytest.mark.parametrize("question", [
    "Who are you?",
    "What are you?",
    "What is your name?",
    "Who am I talking to?",
    "Tell me about yourself.",
])
def test_identity_is_tantra_for_direct_identity_questions(question):
    assert identity_response(question).startswith("I am Tantra")


@pytest.mark.parametrize("question", ["Are you Qwen?", "Are you Llama?", "Are you Mistral?"])
def test_backend_brand_does_not_replace_tantra_identity(question):
    response = identity_response(question)
    assert response.startswith("I am Tantra")
    assert "third-party model" in response


def test_identity_does_not_claim_pretrained_weights_were_created_by_tantra():
    assert "Never claim Tantra created third-party pretrained weights." == (
        TANTRA_IDENTITY["provenance_rule"]
    )
    assert "Never claim Tantra created third-party pretrained weights." in IDENTITY_SYSTEM_MESSAGE


def test_prepare_messages_adds_identity_system_message_without_mutating_input():
    original = [{"role": "user", "content": "Hello"}]
    prepared = prepare_messages(original)
    assert prepared[0] == {"role": "system", "content": IDENTITY_SYSTEM_MESSAGE}
    assert original == [{"role": "user", "content": "Hello"}]


def test_prepare_messages_preserves_existing_system_prompt_and_avoids_duplicates():
    original = [
        {"role": "system", "content": "Follow safety rules."},
        {"role": "user", "content": "Hello"},
    ]
    prepared = prepare_messages(original)
    assert len([m for m in prepared if m["role"] == "system"]) == 1
    assert "Follow safety rules." in prepared[0]["content"]
    assert prepare_messages(prepared) == prepared


def test_unrelated_question_is_not_rewritten():
    assert identity_response("Explain sparse mixture of experts") is None


def test_non_string_input_is_safe():
    assert identity_response(None) is None


def test_inference_engine_stream_returns_tantra_identity_without_backend_generation():
    from Tantra.main import TantraEngine

    engine = object.__new__(TantraEngine)
    reply = "".join(engine.stream([{"role": "user", "content": "Who are you?"}]))
    assert reply == "I am Tantra, an AI system developed under the Tantra project."
    assert engine.last_route == []


def test_reasoner_returns_tantra_identity_before_routing_or_web_lookup():
    from Tantra.reasoning import Reasoner

    reasoner = object.__new__(Reasoner)
    result = reasoner.answer([{"role": "user", "content": "Are you Qwen?"}])
    assert result["source"] == "tantra_identity"
    assert result["reply"].startswith("I am Tantra")
    assert result["route"] == []
