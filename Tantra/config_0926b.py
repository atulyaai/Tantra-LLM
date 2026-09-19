"""
0.926B Model Configuration for Tantra LLM.
============================================
Architecture: 20L × 1472D × 23H × 64
Parameters: 925,564,662 (0.926B)
Output projection: 152,576 rows (60,000 text logits)
Tokenizer: 60,000 real BPE tokens
GPUs: 2 × Tesla T4
FSDP: FULL_SHARD
Precision: FP16
"""

class Config0926B:
    """Configuration for the 0.926B model."""
    
    # ── Architecture ──────────────────────────────
    num_layers = 20
    dim = 1472
    num_heads = 23
    head_dim = 64
    seq_len = 512
    
    # ── Model Output ──────────────────────────────
    # Full output projection rows for checkpoint compatibility
    total_output_rows = 152576  # 60K text + 92K multimodal
    text_logits_rows = 60000    # Only text logits computed during training
    
    # ── Vocabulary ────────────────────────────────
    vocab_size = 60000          # Real BPE tokenizer size
    byte_bpe_vocab = 60000
    
    # ── Attention (ALRA) ──────────────────────────
    alra_num_heads = 23
    alra_use_flash = True
    alra_use_o1 = False
    
    # ── FFN (SGP) ─────────────────────────────────
    sgp_gate_dim = 1472
    sgp_hidden_dim = 1472 * 4
    sgp_num_experts = 1
    
    # ── Training ──────────────────────────────────
    batch_size = 1              # Per GPU
    grad_accum = 8              # Gradient accumulation steps
    effective_batch = 16        # batch_size * grad_accum * num_gpus
    learning_rate = 5e-5
    warmup_steps = 10
    max_steps = 10000           # Full training
    test_steps = 50             # Quick test
    optimizer = "adamw"
    weight_decay = 0.1
    
    # ── FSDP ──────────────────────────────────────
    fsdp_sharding = "FULL_SHARD"
    fsdp_backward_prefetch = "BACKWARD_PRE"
    fsdp_forward_prefetch = "FORWARD_PRE"
    fsdp_limit_all_gathers = True
    fsdp_use_orig_params = False
    
    # ── Precision ─────────────────────────────────
    precision = "fp16"          # Mixed precision
    amp_dtype = "float16"
    grad_scaler = True          # Enable GradScaler for FP16
    
    # ── GPUs ──────────────────────────────────────
    num_gpus = 2
    device = "cuda"
    master_port = 29511
    
    # ── Checkpointing ─────────────────────────────
    checkpoint_every = 500
    eval_every = 500
    max_checkpoints = 5
    
    # ── Data ──────────────────────────────────────
    data_workers = 2
    mask_non_assistant = True
    training_stage = "sft"
    use_latent_reasoning = False
    use_mtp = False  # Disable MTP for 0.926B to reduce OOM
    
    # ── Memory Management ─────────────────────────
    max_grad_norm = 1.0
    pytorch_cuda_alloc = "expandable_segments:True"
    
    # ── Estimated Parameters ──────────────────────
    # Embedding: 60000 * 1472 = 88,320,000
    # Per layer params ≈ 46M (attention + FFN + norm)
    # 20 layers * 46M = 920M
    # Total ≈ 925M
    estimated_params = 925_564_662
    
    @property
    def params_label(self):
        return f"{self.estimated_params/1e9:.3f}B"
    
    @property
    def effective_batch_label(self):
        return f"batch={self.batch_size} * grad_accum={self.grad_accum} * num_gpus={self.num_gpus} = {self.effective_batch}"
    
    @classmethod
    def from_yaml(cls, path):
        """Load config from YAML file."""
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def to_dict(self):
        """Convert to dictionary."""
        return {k: v for k, v in self.__class__.__dict__.items() if not k.startswith('_') and not callable(v)}
    
    def to_json(self, path):
        """Save config to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def __repr__(self):
        return (
            f"Config0926B(\n"
            f"  layers={self.num_layers}, dim={self.dim}, heads={self.num_heads}\n"
            f"  output_rows={self.total_output_rows}, text_rows={self.text_logits_rows}\n"
            f"  params={self.params_label}, effective_batch={self.effective_batch_label}\n"
            f"  lr={self.learning_rate}, precision={self.precision}\n"
            f")"
        )

# Singleton instance
model_config = Config0926B()

def get_config():
    """Get the 0.926B model configuration."""
    return model_config

def print_config():
    """Print the configuration."""
    config = model_config
    print("=" * 80)
    print("0.926B MODEL CONFIGURATION")
    print("=" * 80)
    print(f"Architecture : {config.num_layers}L × {config.dim}D × {config.num_heads}H × {config.head_dim}")
    print(f"Parameters   : {config.estimated_params:,} ({config.params_label})")
    print(f"Output rows  : {config.total_output_rows} (text: {config.text_logits_rows})")
    print(f"Tokenizer    : {config.vocab_size} real tokens")
    print(f"Sequence     : {config.seq_len}")
    print(f"GPUs         : {config.num_gpus} × Tesla T4")
    print(f"FSDP         : {config.fsdp_sharding}")
    print(f"Precision    : {config.precision}")
    print(f"Batch/GPU    : {config.batch_size}")
    print(f"Grad accum   : {config.grad_accum}")
    print(f"Effective    : {config.effective_batch}")
    print(f"LR           : {config.learning_rate}")
    print(f"Warmup       : {config.warmup_steps}")
    print("=" * 80)

if __name__ == "__main__":
    print_config()
