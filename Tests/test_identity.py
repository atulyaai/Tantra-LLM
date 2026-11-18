from Tantra.identity import TANTRA_IDENTITY, identity_response


def test_identity_is_tantra_for_common_identity_questions():
    for question in ("Who are you?", "What is your name?", "Are you Qwen?"):
        assert identity_response(question).startswith("I am Tantra")


def test_identity_does_not_claim_pretrained_weights_were_created_by_tantra():
    assert "Never claim Tantra created third-party pretrained weights." == (
        TANTRA_IDENTITY["provenance_rule"]
    )


def test_unrelated_question_is_not_rewritten():
    assert identity_response("Explain sparse mixture of experts") is None
