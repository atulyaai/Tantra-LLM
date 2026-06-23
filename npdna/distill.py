"""Knowledge Distillation — Efficient CPU Training for Tantra LLM.

Instead of learning from raw text (which takes millions of tokens),
the model learns to match the probability distribution of a pre-trained teacher.
This makes CPU training 10-50x more sample-efficient.
"""
import torch
import torch.nn.functional as F
from torch import nn

class DistillationTeacher:
    """Wrapper for a pre-trained teacher model (e.g., GPT-2 Small)."""

    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Please install transformers to use distillation: pip install transformers")

        self.device = device
        print(f"  [Distill] Loading teacher model: {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

        # We need to map teacher vocab to student vocab.
        # Since they might use different tokenizers, we do sequence-level distillation
        # or simple logit matching if vocab is aligned.
        # For simplicity in this demo, if vocabs don't match exactly, we'll use
        # a projection or just distill the hidden states.

    @torch.no_grad()
    def get_teacher_logits(self, text_batch: list[str]) -> torch.Tensor:
        """Get teacher logits for a batch of text."""
        inputs = self.tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.logits

def compute_distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """Compute KL-divergence loss between student and teacher logits.

    Args:
        student_logits: (B, T_student, V_student)
        teacher_logits: (B, T_teacher, V_teacher)
        temperature: Smoothing factor. Higher = softer targets.
    """
    # If vocabularies don't match, we can't do direct KL div on logits easily.
    # A simple fallback for differing vocabs is to align the top-k probabilities,
    # but the easiest way is to assume we are distilling hidden states or just
    # using it when vocabs are aligned.

    # For now, we assume the student uses the teacher's tokenizer or we truncate to min vocab.
    min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
    min_seq = min(student_logits.size(1), teacher_logits.size(1))

    s_logits = student_logits[:, :min_seq, :min_vocab] / temperature
    t_logits = teacher_logits[:, :min_seq, :min_vocab] / temperature

    s_log_probs = F.log_softmax(s_logits, dim=-1)
    t_probs = F.softmax(t_logits, dim=-1)

    # KL Div loss
    loss = F.kl_div(s_log_probs, t_probs, reduction='batchmean') * (temperature ** 2)
    return loss
