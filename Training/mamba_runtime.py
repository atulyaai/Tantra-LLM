import os
from typing import Any, Dict, Iterator

import torch
from safetensors.torch import load_file

from Training.model_mamba import build_from_config
from weight_manager import get_weight_manager, load_model_weights
from config_manager import get_config_manager, get_model_paths


class MambaRuntime:
    def __init__(self, model_type: str = "mamba", version: str = None, cfg: Dict[str, Any] = None):
        from tokenizers import Tokenizer
        
        # Get configuration and paths
        config_manager = get_config_manager()
        weight_manager = get_weight_manager()
        
        self.config = config_manager.get_config(model_type)
        self.paths = config_manager.get_paths(model_type)
        self.model_type = model_type
        
        # Use provided config or default config
        if cfg is None:
            cfg = {
                'd_model': self.config.d_model,
                'n_layers': self.config.n_layers,
                'd_state': self.config.d_state,
                'd_conv': self.config.d_conv,
                'dropout': self.config.dropout
            }
        
        # Load tokenizer
        tokenizer_path = self.paths['tokenizer']
        if not os.path.exists(tokenizer_path):
            print(f"Warning: Tokenizer file not found: {tokenizer_path}. Creating basic tokenizer...")
            # Create a basic tokenizer
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import Whitespace
            
            tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(vocab_size=self.config.vocab_size, special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[CLS]", "[MASK]"])
            
            # Train on sample data
            sample_texts = ["Hello world", "This is a test", "Machine learning is interesting"]
            tokenizer.train_from_iterator(sample_texts, trainer)
            tokenizer.save(tokenizer_path)
            print("Created basic tokenizer")
        
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        vocab_size = self.tokenizer.get_vocab_size()
        self.model = build_from_config(cfg, vocab_size)
        
        # Load weights dynamically
        state_dict = load_model_weights(model_type, version)
        if state_dict:
            try:
                self.model.load_state_dict(state_dict, strict=False)
                print(f"Loaded weights for {model_type}")
            except Exception as e:
                print(f"Warning: Could not load weights: {e}")
        else:
            print(f"Warning: No weights found for {model_type}. Using randomly initialized model.")
        
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
