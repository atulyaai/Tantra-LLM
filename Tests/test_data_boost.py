from Tantra.data_boost import _to_messages, identity_rows


def test_identity_rows_name_maker_and_skip_probe():
    rows = identity_rows(500, {"तुम कौन हो?", "Who made you?"})
    qs = {r["messages"][0]["content"] for r in rows}
    assert "तुम कौन हो?" not in qs and "Who made you?" not in qs
    for r in rows:
        a = r["messages"][1]["content"]
        assert "Atulya" in a or "अतुल्य" in a


def test_identity_denies_other_makers():
    for r in identity_rows(500, set()):
        q, a = (m["content"] for m in r["messages"])
        if "ChatGPT" in q or "OpenAI" in q:
            assert a.startswith(("No", "नहीं", "Nahi"))


def test_to_messages_layouts():
    assert _to_messages({"instruction": "Q", "context": "C", "response": "A"})[0]["content"] == "Q\n\nC"
    assert _to_messages({"hin_Deva": [["प्र", "उ"], ["प्र2", "उ2"]]})[3]["content"] == "उ2"
    m = _to_messages({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                                   {"role": "assistant", "content": "a"}]})
    assert [x["role"] for x in m] == ["user", "assistant"]
