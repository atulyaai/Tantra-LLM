import json
import torch
from Tantra.config import ModelConfig
from Tantra.model.tantra_model import TantraModel
from Tantra.tokenizer import TantraTokenizer
from Tantra.training_probes import run_probes


def test_probes_preserve_weights_rng_and_training_mode(tmp_path):
    tok=TantraTokenizer.train(['Hello! Who are you? I am Tantra.']*3,vocab_size=280,min_frequency=1)
    model=TantraModel(ModelConfig(vocab_size=tok.vocab_size,d_model=16,n_layers=1,n_heads=2,n_kv_heads=1,mlp_mult=4,max_seq_len=128));model.train()
    before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
    suite=tmp_path/'suite.json';suite.write_text(json.dumps({'cases':[{'language':'en','question':'Who are you?','reference':'I am Tantra.'}]}))
    result=run_probes(model,tok,suite,7,True,tmp_path/'out')
    assert model.training and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert result['languages']['en']['questions']==1
    assert 'answer' in result['responses'][0]
    assert (tmp_path/'out/probe_history.jsonl').exists()


def test_probe_console_handles_non_unicode_windows_output(monkeypatch):
    import io
    import Tantra.training_probes as probes
    data=io.BytesIO();stream=io.TextIOWrapper(data,encoding='ascii')
    monkeypatch.setattr(probes.sys,'stdout',stream)
    probes.console_print('नमस्ते')
    stream.flush()
    assert b'\\u' in data.getvalue()
