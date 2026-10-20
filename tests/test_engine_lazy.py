import torch
from Tantra.config import ModelConfig
from Tantra.inference.inference_engine import TantraEngine
from Tantra.model.tantra_model import TantraModel
from Tantra.tokenizer import TantraTokenizer


def test_engine_loads_experts_on_demand_and_reports_timing(tmp_path):
    tok = TantraTokenizer.train(['Hello hi welcome thanks I am Tantra. Who made you?'] * 3, vocab_size=280, min_frequency=1)
    tok.save(tmp_path / 'tokenizer.json')
    model = TantraModel(ModelConfig(vocab_size=tok.vocab_size, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1,
                                    mlp_mult=2, max_seq_len=64, lora_rank=2))
    for name in ('greeting', 'identity_self', 'farewell'):
        model.add_expert(name)
    model.save(tmp_path / 'base.pt', tmp_path / 'experts')
    eng = TantraEngine(tmp_path, max_experts=1, expert_cache=1)
    assert eng.cache is not None and len(eng.cache.resident) == 0
    reply = eng.chat([{'role': 'user', 'content': 'Hello'}], max_new_tokens=4, experts=[('identity_self', 1.0)])
    assert isinstance(reply, str)
    assert list(eng.cache.resident) == ['identity_self']
    assert eng.last_timing['experts_loaded'] == 1 and eng.last_timing['tokens'] >= 1
    eng.chat([{'role': 'user', 'content': 'Hi'}], max_new_tokens=2, experts=[('farewell', 1.0)])
    assert list(eng.cache.resident) == ['farewell'] and eng.cache.evictions == 1
    assert eng.benchmark(n_tokens=2)['experts_resident'] <= 1



def test_router_none_class_means_base_only(tmp_path):
    tok = TantraTokenizer.train(['Hello hi welcome thanks I am Tantra.'] * 3, vocab_size=280, min_frequency=1)
    tok.save(tmp_path / 'tokenizer.json')
    model = TantraModel(ModelConfig(vocab_size=tok.vocab_size, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1,
                                    mlp_mult=2, max_seq_len=64, lora_rank=2))
    model.add_expert('greeting'); model.add_expert('identity_self')
    model.save(tmp_path / 'base.pt', tmp_path / 'experts')
    eng = TantraEngine(tmp_path, max_experts=2)

    class Stub:
        names = ['greeting', 'identity_self', 'none']
        def __init__(self, p): self.p = torch.tensor(p)
        def probs(self, text): return self.p

    eng.ngram = Stub([0.2, 0.08, 0.72])        # "none" wins -> base only, not a partial greeting
    assert eng.route('my exam went well') == []
    eng.ngram = Stub([0.55, 0.05, 0.40])       # expert wins; "none" never becomes an expert
    route = eng.route('hello')
    assert [n for n, _ in route] == ['greeting'] and abs(sum(w for _, w in route) - 1.0) < 1e-6
