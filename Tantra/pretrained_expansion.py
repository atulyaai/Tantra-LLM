"""PEFT adaptation, capability inheritance and controlled residual growth."""
from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path
from typing import Iterable

def hash_tree(root):
    root=Path(root); h=hashlib.sha256()
    if not root.is_dir(): raise ValueError(f"checkpoint directory missing: {root}")
    files=sorted(p for p in root.rglob("*") if p.is_file())
    if not files: raise ValueError("checkpoint directory is empty")
    for p in files:
        h.update(p.relative_to(root).as_posix().encode()); h.update(b"\0")
        with p.open("rb") as f:
            for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def train_lora(checkpoint_path,dataset_jsonl,output_dir,*,steps=100,learning_rate=2e-4,rank=8,max_length=512,target_modules=None):
    """Train a PEFT LoRA adapter from JSONL records containing a text field."""
    try:
        import torch
        from peft import LoraConfig,get_peft_model
        from transformers import AutoModelForCausalLM,AutoTokenizer
    except ImportError as exc: raise RuntimeError("install the finetune extra: pip install -e '.[finetune]'") from exc
    if steps<1 or rank<1 or max_length<1 or learning_rate<=0: raise ValueError("steps, rank, max_length and learning_rate must be positive")
    base=Path(checkpoint_path); before=hash_tree(base)
    out=Path(output_dir)
    if out.exists() and any(out.iterdir()): raise ValueError("adapter output directory must be new or empty")
    texts=[]
    for line in Path(dataset_jsonl).read_text(encoding="utf-8").splitlines():
        if line.strip():
            text=json.loads(line).get("text")
            if not isinstance(text,str) or not text.strip(): raise ValueError("each JSONL record requires non-empty text")
            texts.append(text)
    if not texts: raise ValueError("training dataset is empty")
    tokenizer=AutoTokenizer.from_pretrained(str(base),local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(str(base),local_files_only=True)
    model.config.use_cache=False
    config=LoraConfig(r=rank,lora_alpha=rank*2,lora_dropout=.05,bias="none",task_type="CAUSAL_LM",
        target_modules=target_modules or ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    model=get_peft_model(model,config)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device); model.train()
    batches=[tokenizer(x,truncation=True,max_length=max_length,return_tensors="pt")["input_ids"][0] for x in texts]
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=learning_rate)
    losses=[]
    for step in range(steps):
        ids=batches[step%len(batches)].unsqueeze(0).to(device)
        loss=model(input_ids=ids,labels=ids).loss
        loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    out.mkdir(parents=True,exist_ok=True); model.save_pretrained(out); tokenizer.save_pretrained(out)
    after=hash_tree(base)
    if before!=after: raise RuntimeError("source checkpoint changed during LoRA training")
    metadata={"base_checkpoint":str(base.resolve()),"base_sha256":before,"rank":rank,
              "steps":steps,"examples":len(texts),"final_loss":losses[-1],"target_modules":config.target_modules}
    (out/"tantra_adapter.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    return metadata

def transfer_adapter(parent_adapter,child_adapter,*,expected_base_sha256=None):
    """Inherit adapter weights only if base checkpoint and PEFT structure match."""
    parent,child=Path(parent_adapter),Path(child_adapter)
    pmeta=parent/"tantra_adapter.json"
    if not pmeta.is_file(): raise ValueError("parent Tantra adapter metadata missing")
    metadata=json.loads(pmeta.read_text(encoding="utf-8"))
    if not expected_base_sha256:
        raise ValueError("expected_base_sha256 is required for safe adapter inheritance")
    if metadata["base_sha256"]!=expected_base_sha256:
        raise ValueError("adapter base checkpoint hash mismatch")
    pconfig=json.loads((parent/"adapter_config.json").read_text(encoding="utf-8"))
    if child.exists() and (child/"adapter_config.json").exists():
        cconfig=json.loads((child/"adapter_config.json").read_text(encoding="utf-8"))
        for key in ("base_model_name_or_path","target_modules","r","lora_alpha","peft_type"):
            if pconfig.get(key)!=cconfig.get(key): raise ValueError("adapter architectures are incompatible")
    if child.exists() and any(child.iterdir()):
        raise ValueError("child adapter output must be new or empty")
    child.mkdir(parents=True,exist_ok=True)
    shutil.copytree(parent,child,dirs_exist_ok=True)
    return str(child.resolve())

def generate_teacher_dataset(teacher,tokenizer,prompts:Iterable[str],output_jsonl):
    """Create teacher demonstrations; caller must review them before student training."""
    out=Path(output_jsonl); out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise ValueError("distillation output must be a new file")
    n=0
    with out.open("w",encoding="utf-8") as f:
        for prompt in prompts:
            messages=[{"role":"user","content":prompt}]
            ids=tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt").to(next(teacher.parameters()).device)
            import torch
            with torch.inference_mode(): generated=teacher.generate(ids,max_new_tokens=256,do_sample=False)
            answer=tokenizer.decode(generated[0,ids.shape[-1]:],skip_special_tokens=True)
            f.write(json.dumps({"text":f"User: {prompt}\nAssistant: {answer}"},ensure_ascii=False)+"\n"); n+=1
    if not n: raise ValueError("at least one distillation prompt is required")
    manifest={"examples":n,"path":str(out.resolve()),"dataset_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
              "teacher_model_id":getattr(teacher,"name_or_path",None),"review_required":True}
    out.with_suffix(out.suffix+".manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return manifest

def add_residual_branch(model,layer_index=-1,bottleneck=64):
    """Insert a zero-output-initialised MLP branch in a Llama-style decoder block."""
    import torch
    from torch import nn
    if bottleneck<1: raise ValueError("bottleneck must be positive")
    layers=getattr(getattr(model,"model",None),"layers",None)
    if layers is None: layers=getattr(model,"layers",None)
    if layers is None or not len(layers): raise ValueError("unsupported decoder layer layout")
    idx=layer_index if layer_index>=0 else len(layers)+layer_index
    if idx<0 or idx>=len(layers): raise IndexError("decoder layer index out of range")
    if getattr(layers[idx],"_tantra_growth_branch",False): raise ValueError("branch already exists")
    width=getattr(model.config,"hidden_size",getattr(model.config,"n_embd",None))
    if width is None: raise ValueError("cannot infer hidden size")
    # Freeze the entire foundation first; the newly created branch remains trainable.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    class BranchWrapped(nn.Module):
        def __init__(self,base):
            super().__init__(); self.base=base; self._tantra_growth_branch=True
            self.branch=nn.Sequential(nn.Linear(width,bottleneck,bias=False),nn.GELU(),nn.Linear(bottleneck,width,bias=False))
            nn.init.zeros_(self.branch[-1].weight)
        def forward(self,*args,**kwargs):
            result=self.base(*args,**kwargs); hidden=result[0] if isinstance(result,tuple) else result
            if not isinstance(hidden,torch.Tensor) or hidden.ndim!=3: raise RuntimeError("expected [batch,sequence,hidden] decoder output")
            value=hidden+self.branch(hidden)
            return (value,*result[1:]) if isinstance(result,tuple) else value
    layers[idx]=BranchWrapped(layers[idx])
    return model

def train_residual_branch(model,tokenizer,dataset_jsonl,*,steps=100,learning_rate=1e-4,max_length=512):
    """Train only new branch parameters; original decoder blocks stay frozen."""
    import torch
    if steps<1 or max_length<1 or learning_rate<=0: raise ValueError("steps, max_length and learning_rate must be positive")
    trainable=[p for p in model.parameters() if p.requires_grad]
    if not trainable: raise ValueError("no trainable residual branch parameters")
    texts=[json.loads(line)["text"] for line in Path(dataset_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not texts: raise ValueError("training dataset is empty")
    if hasattr(model, "config"):
        model.config.use_cache = False
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    encoded=[tokenizer(t,truncation=True,max_length=max_length,return_tensors="pt")["input_ids"][0] for t in texts]
    opt=torch.optim.AdamW(trainable,lr=learning_rate); losses=[]; model.train()
    for i in range(steps):
        ids=encoded[i%len(encoded)].unsqueeze(0).to(device); loss=model(input_ids=ids,labels=ids).loss
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True); losses.append(float(loss.detach().cpu()))
    return {"steps":steps,"examples":len(texts),"final_loss":losses[-1],
            "trainable_parameters":sum(p.numel() for p in trainable)}

def save_residual_candidate(model,output_dir,layer_index=-1,bottleneck=64):
    import torch
    out=Path(output_dir)
    if out.exists() and any(out.iterdir()): raise ValueError("residual candidate output directory must be new or empty")
    out.mkdir(parents=True,exist_ok=True)
    modules={name:module for name,module in model.named_modules() if getattr(module,"_tantra_growth_branch",False)}
    if len(modules)!=1: raise ValueError("prototype supports exactly one residual growth branch")
    name,module=next(iter(modules.items()))
    torch.save({name:module.branch.state_dict()},out/"tantra_residual.pt")
    config={"module_name":name,"layer_index":layer_index,"bottleneck":bottleneck}
    (out/"tantra_residual.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    return config

def load_residual_candidate(model,candidate_dir):
    import torch
    path=Path(candidate_dir); config=json.loads((path/"tantra_residual.json").read_text(encoding="utf-8"))
    add_residual_branch(model,config["layer_index"],config["bottleneck"])
    modules=dict(model.named_modules()); state=torch.load(path/"tantra_residual.pt",map_location="cpu",weights_only=True)
    for name,weights in state.items():
        if name not in modules or not getattr(modules[name],"_tantra_growth_branch",False): raise ValueError("residual branch target mismatch")
        modules[name].branch.load_state_dict(weights,strict=True)
    return model

def count_parameters(model): return sum(p.numel() for p in model.parameters())
