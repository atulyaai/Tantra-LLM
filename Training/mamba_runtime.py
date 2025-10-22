import os
from typing import Any, Dict, Iterator

import torch
from safetensors.torch import load_file

from Training.model_mamba import build_from_config


class MambaRuntime:
    def __init__(self, tokenizer_path: str, weights_path: str, cfg: Dict[str, Any]):
        from tokenizers import Tokenizer
        
        # Check if tokenizer file exists
        if not os.path.exists(tokenizer_path):
            print(f"Warning: Tokenizer file not found: {tokenizer_path}. Creating basic tokenizer...")
            # Create a basic tokenizer
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import Whitespace
            
            tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(vocab_size=1000, special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[CLS]", "[MASK]"])
            
            # Train on sample data
            sample_texts = ["Hello world", "This is a test", "Machine learning is interesting"]
            tokenizer.train_from_iterator(sample_texts, trainer)
            tokenizer.save(tokenizer_path)
            print("Created basic tokenizer")
        
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        vocab_size = self.tokenizer.get_vocab_size()
        self.model = build_from_config(cfg, vocab_size)
        
        # Load weights if they exist
        if os.path.exists(weights_path):
            try:
                state = load_file(weights_path)
                self.model.load_state_dict(state, strict=False)
                print(f"Loaded weights from {weights_path}")
            except Exception as e:
                print(f"Warning: Could not load weights from {weights_path}: {e}")
        else:
            print(f"Warning: Weights file not found: {weights_path}. Using randomly initialized model.")
        
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt: str, gen: Dict[str, Any]) -> str:
        max_new_tokens = int(gen.get("max_tokens", 128))
        temperature = float(gen.get("temperature", 0.8))
        
        try:
            input_ids = self.tokenizer.encode(prompt).ids
            ids = list(input_ids)
            
            for _ in range(max_new_tokens):
                x = torch.tensor([ids], dtype=torch.long)
                logits = self.model(x)[:, -1, :]
                probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1))
                ids.append(next_id)
                
                # Check for EOS token
                eos_token_id = self.tokenizer.token_to_id("[EOS]")
                if eos_token_id is not None and next_id == eos_token_id:
                    break
            
            return self.tokenizer.decode(ids[len(input_ids):])
        except Exception as e:
            return f"Error generating text: {e}"

    @torch.inference_mode()
    def stream(self, prompt: str, gen: Dict[str, Any]) -> Iterator[str]:
        yielded = ""
        text = self.generate(prompt, gen)
        for ch in text:
            yielded += ch
            yield ch
