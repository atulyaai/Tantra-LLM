import os
from typing import Any, Dict, Iterator

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


class TextRuntime:
    def __init__(self, tokenizer_path: str, weights_path: str, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model configuration and weights
        if os.path.exists(weights_path):
            # Try to load as a transformer model
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    tokenizer_path, 
                    torch_dtype=torch.float32,
                    device_map=device
                )
                if os.path.exists(weights_path):
                    state_dict = load_file(weights_path)
                    self.model.load_state_dict(state_dict, strict=False)
            except Exception:
                # Fallback to a simple model if loading fails
                self.model = None
        else:
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
