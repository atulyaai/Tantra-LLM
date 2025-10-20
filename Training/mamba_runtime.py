import os
from typing import Any, Dict, Iterator

import torch
from safetensors.torch import load_file

from Training.model_mamba import build_from_config


class MambaRuntime:
    def __init__(self, tokenizer_path: str, weights_path: str, cfg: Dict[str, Any]):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        vocab_size = self.tokenizer.get_vocab_size()
        self.model = build_from_config(cfg, vocab_size)
        if os.path.exists(weights_path):
            state = load_file(weights_path)
            self.model.load_state_dict(state, strict=False)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt: str, gen: Dict[str, Any]) -> str:
        max_new_tokens = int(gen.get("max_tokens", 128))
        temperature = float(gen.get("temperature", 0.8))
        input_ids = self.tokenizer.encode(prompt).ids
        ids = list(input_ids)
        for _ in range(max_new_tokens):
            x = torch.tensor([ids], dtype=torch.long)
            logits = self.model(x)[:, -1, :]
            probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1))
            ids.append(next_id)
            if next_id == self.tokenizer.token_to_id("[EOS]"):
                break
        return self.tokenizer.decode(ids[len(input_ids):])

    @torch.inference_mode()
    def stream(self, prompt: str, gen: Dict[str, Any]) -> Iterator[str]:
        yielded = ""
        text = self.generate(prompt, gen)
        for ch in text:
            yielded += ch
            yield ch


