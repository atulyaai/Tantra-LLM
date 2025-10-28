from __future__ import annotations

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    AutoTokenizer = None


class TextTokenizer:
    """Tokenizer wrapper with add_tokens support fallbacks."""

    def __init__(self, model_name: str = "gpt2"):
        self._tok = AutoTokenizer.from_pretrained(model_name) if AutoTokenizer else None
        self._vocab = {} if self._tok is None else None

    def encode(self, text: str, add_special_tokens: bool = True):
        if self._tok:
            return self._tok.encode(text, add_special_tokens=add_special_tokens)
        return [ord(c) for c in text]

    def get_vocab(self):
        if self._tok:
            return self._tok.get_vocab()
        return self._vocab

    def add_tokens(self, toks):
        if self._tok:
            self._tok.add_tokens(toks)
        else:
            for t in toks:
                if t not in self._vocab:
                    self._vocab[t] = len(self._vocab)

    def convert_tokens_to_ids(self, tok: str) -> int:
        if self._tok:
            return self._tok.convert_tokens_to_ids(tok)
        return self._vocab.get(tok, 0)


