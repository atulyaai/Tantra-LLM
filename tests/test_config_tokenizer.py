from npdna import CONFIGS, PREFERRED_CONFIG_NAMES, AtulyaTokenizer, NpDnaCore


def test_seed_is_only_named_config():
    assert tuple(CONFIGS) == ("seed",)
    assert PREFERRED_CONFIG_NAMES == ("seed",)


def test_seed_core_starts_small_and_can_expand():
    core = NpDnaCore.from_config("seed")
    old_vocab = core.model.vocab_size
    core.model.resize_embeddings(old_vocab + 8)
    assert core.model.vocab_size == old_vocab + 8


def test_tokenizer_round_trip():
    tokenizer = AtulyaTokenizer(initial_capacity=4096, max_capacity=256_000)
    text = "Hello NP-DNA"
    assert tokenizer.decode(tokenizer.encode(text)) == text
