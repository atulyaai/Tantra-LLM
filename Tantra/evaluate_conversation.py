"""Generate reserved conversation answers for rubric review; never train them."""
import argparse,json
from pathlib import Path
import torch
from Tantra.config import ROOT
from Tantra.model.tantra_model import TantraModel
from Tantra.tokenizer import TantraTokenizer
from Tantra.data.public_sources import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Use a new evaluation output file')
    suite=json.loads((ROOT/'docs/data/conversation_eval.v1.json').read_text(encoding='utf-8'))
    torch.set_num_threads(4)
    model=TantraModel.load(a.model_dir/'base.pt',experts_dir=a.model_dir/'experts');model.eval()
    tok=TantraTokenizer.load(a.model_dir/'tokenizer.json')
    if tok.vocab_size!=model.cfg.vocab_size:p.error('Tokenizer/model vocabulary mismatch')
    answers=[]
    with torch.no_grad():
        for case in suite['cases']:
            ids,_=tok.encode_chat([{'role':'user','content':case['prompt']}],add_generation_prompt=True)
            tensor=torch.tensor([ids]);routes=model.route(tensor) if model.expert_names else []
            model.set_active(routes)
            answer=tok.decode(list(model.generate(tensor,max_new_tokens=64,temperature=0,stop_ids=(tok.end_id,tok.user_id))))
            answers.append({**case,'answer':answer,'routes':routes,'review':'pending',
                            'expected_expert_in_routes':any(name==case['expert'] for name,weight in routes)})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'base_sha256':sha256(a.model_dir/'base.pt'),
        'tokenizer_sha256':sha256(a.model_dir/'tokenizer.json'),'cases':answers,
        'warning':'Route hits are not answer correctness or macro-F1. Rubric and fluent reviews pending; no release score inferred.'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
