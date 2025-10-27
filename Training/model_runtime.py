import os
from typing import Any, Dict, Iterator

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from .weight_manager import get_weight_manager, load_model_weights
from .config_manager import get_config_manager, get_model_paths


class TextRuntime:
    def __init__(self, model_type: str = "mamba", device: str = "cpu", version: str = None):
        self.device = device
        self.model_type = model_type
        
        # Get configuration and paths
        config_manager = get_config_manager()
        weight_manager = get_weight_manager()
        
        self.config = config_manager.get_config(model_type)
        self.paths = config_manager.get_paths(model_type)
        
        # Load tokenizer
        tokenizer_path = self.paths['tokenizer']
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Warning: Could not load tokenizer from {tokenizer_path}: {e}")
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
            self.tokenizer = tokenizer
        
        # Load model weights dynamically
        state_dict = load_model_weights(model_type, version)
        if state_dict:
            # Try to load as a transformer model
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    tokenizer_path, 
                    torch_dtype=torch.float32,
                    device_map=device
                )
                self.model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"Warning: Could not load model: {e}")
                # Fallback to a simple model if loading fails
                self.model = None
        else:
            print(f"Warning: No weights found for model type: {model_type}")
            self.model = None
        
        if self.model:
            self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt: str, gen: Dict[str, Any]) -> str:
        if not self.model:
            return "Model not loaded"
            
        max_new_tokens = int(gen.get("max_tokens", 128))
        temperature = float(gen.get("temperature", 0.8))
        top_p = float(gen.get("top_p", 0.9))
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return generated_text

    @torch.inference_mode()
    def stream(self, prompt: str, gen: Dict[str, Any]) -> Iterator[str]:
        # Simple streaming implementation
        text = self.generate(prompt, gen)
        for char in text:
            yield char
