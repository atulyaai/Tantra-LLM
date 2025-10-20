import os
from typing import Any, Dict

import torch


class TextRuntime:
    def __init__(self, tokenizer_path: str, weights_path: str, device: str = "cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        local_dir = os.path.dirname(weights_path)
        load_from_local = os.path.exists(weights_path) and os.path.exists(tokenizer_path)

        if load_from_local:
            # Expect a local transformers model directory layout; if only files, still try from_pretrained(local_dir)
            model_source = local_dir
        else:
            # Fallback small model for CPU
            model_source = "distilgpt2"

        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.model = AutoModelForCausalLM.from_pretrained(model_source, torch_dtype=torch.float32)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt: str, gen: Dict[str, Any]) -> str:
        max_new_tokens = int(gen.get("max_tokens", 256))
        temperature = float(gen.get("temperature", 0.7))
        top_p = float(gen.get("top_p", 0.9))

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        out_text = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return out_text.strip()

    @torch.inference_mode()
    def stream(self, prompt: str, gen: Dict[str, Any]):
        from transformers import TextIteratorStreamer
        max_new_tokens = int(gen.get("max_tokens", 256))
        temperature = float(gen.get("temperature", 0.7))
        top_p = float(gen.get("top_p", 0.9))

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        import threading
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        for text in streamer:
            yield text
        thread.join()


