"""Provider-neutral local Transformers gateway. Optional: pip install -e '.[pretrained]'."""
from __future__ import annotations
from typing import Any
from Tantra.model_registry import ModelSpec
from Tantra.identity import IDENTITY_SYSTEM_MESSAGE, identity_response, prepare_messages

class ModelGateway:
    def __init__(self, spec: ModelSpec, checkpoint_path: str, *, device: str = "cpu", dtype: str = "auto", adapter_path: str | None = None):
        spec.validate()
        self.spec, self.checkpoint_path, self.device, self.dtype = spec, checkpoint_path, device, dtype
        self.adapter_path = adapter_path
        self._model = self._tokenizer = None

    def load(self) -> "ModelGateway":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the optional pretrained dependencies first") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path, local_files_only=True, trust_remote_code=False)
        kwargs: dict[str, Any] = {"local_files_only": True, "trust_remote_code": False}
        if self.dtype == "auto":
            kwargs["torch_dtype"] = "auto"
        elif self.dtype == "float32":
            kwargs["torch_dtype"] = torch.float32
        elif self.dtype == "bfloat16":
            kwargs["torch_dtype"] = torch.bfloat16
        else:
            raise ValueError("dtype must be auto, float32, or bfloat16")
        self._model = AutoModelForCausalLM.from_pretrained(self.checkpoint_path, **kwargs).to(self.device)
        if self.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Install the optional finetune dependencies to load a LoRA adapter") from exc
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()
        return self

    def generate(self, messages: list[dict[str, str]], *, max_new_tokens: int = 128, temperature: float = 0.0) -> str:
        if not messages or any(m.get("role") not in {"system", "user", "assistant"} for m in messages):
            raise ValueError("messages must contain supported chat roles")
        last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        direct = identity_response(last_user)
        if direct:
            normalized = " ".join(last_user.lower().strip(" ?.!").split())
            if normalized in {"are you qwen", "are you llama", "are you mistral"}:
                return f"I am Tantra. This deployment uses {self.spec.model_id} as its underlying pretrained model."
            return direct
        normalized = " ".join(last_user.lower().strip(" ?.!").split())
        if normalized in {"what model powers you", "which model powers you", "what is your underlying model", "which underlying model are you using"}:
            return f"I am Tantra. This deployment is powered by {self.spec.model_id} from {self.spec.provider}, revision {self.spec.revision}."
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("call load() before generate()")
        if not hasattr(self._tokenizer, "apply_chat_template"):
            raise RuntimeError("tokenizer has no chat template; provide an explicit adapter")
        import torch
        prepared = prepare_messages(messages)
        backend_note = (f"Verified underlying model metadata: model_id={self.spec.model_id}; "
                        f"provider={self.spec.provider}; revision={self.spec.revision}. "
                        "Disclose this metadata truthfully if asked which model powers the deployment.")
        prepared[0]["content"] += "\\n\\n" + backend_note
        encoded = self._tokenizer.apply_chat_template(
            prepared, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(self.device)
        kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "pad_token_id": self._tokenizer.eos_token_id}
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature)
        else:
            kwargs["do_sample"] = False
        with torch.inference_mode():
            output = self._model.generate(encoded, **kwargs)
        return self._tokenizer.decode(output[0, encoded.shape[-1]:], skip_special_tokens=True).strip()

    def unload(self) -> None:
        self._model = self._tokenizer = None
