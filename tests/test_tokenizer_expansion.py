import pytest
from Tantra.tokenizer import TantraTokenizer
from Tantra.data.tokenizer_candidates import extend_vocabulary


def test_vocabulary_extension_preserves_ids_and_old_merge_priority():
    base=TantraTokenizer.train(['hello hello hi there good morning'],vocab_size=280,min_frequency=1)
    training=[' '.join('exampleword'+str(i) for i in range(1000))]*2
    learned=TantraTokenizer.train(training,vocab_size=500,min_frequency=1)
    expanded=extend_vocabulary(base,learned,base.vocab_size+40)
    assert base.vocab_size<expanded.vocab_size<=base.vocab_size+40
    for token,identity in base.tok.get_vocab().items():
        assert expanded.tok.token_to_id(token)==identity
    for text in ['hello hi there','exampleword123 exampleword99','नमस्ते!','Namaste, kaise ho?']:
        assert expanded.decode(expanded.encode(text))==text
        assert len(expanded.encode(text))<=len(base.encode(text))
    with pytest.raises(ValueError,match='exceed'):
        extend_vocabulary(base,learned,base.vocab_size)


def test_embedding_migration_preserves_old_rows_and_rejects_renumbering():
    import torch
    from Tantra.data.migrate_vocabulary import expand_embedding
    base=TantraTokenizer.train(['hello hello hi there'],vocab_size=280,min_frequency=1)
    learned=TantraTokenizer.train([' '.join('word'+str(i) for i in range(1000))]*2,vocab_size=500,min_frequency=1)
    expanded=extend_vocabulary(base,learned,base.vocab_size+40)
    weight=torch.randn(base.vocab_size,16)
    result=expand_embedding(weight,base,expanded)
    assert result.shape==(expanded.vocab_size,16)
    assert torch.equal(result[:base.vocab_size],weight)
    assert torch.isfinite(result).all()
    with pytest.raises(ValueError,match='IDs'):
        expand_embedding(weight,base,learned)
