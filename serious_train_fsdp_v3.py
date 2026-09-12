"""Strict 2x-T4 FSDP trainer for Tantra 0.675B.

Fixes v2:
- explicitly validates CUDA_VISIBLE_DEVICES inside each torchrun worker
- uses local CUDA rank safely
- tokenizer is built once and NEVER deleted between runs
- rank 0 builds tokenizer, all ranks wait at barrier
- FSDP FULL_SHARD per transformer block + root
- real JSONL, 65,536 vocab, BitNet ternary, seq 512
- no resume/synthetic/DPO/auto-pilot/MTP/latent reasoning
"""
from __future__ import annotations
import argparse, math, os, random, time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, StateDictType, ShardedStateDictConfig
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl
from Tantra.config import NeuroCoreConfig, VocabConfig
from Tantra.tokenizer import ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer
from Tantra.model import NeuroCoreModel
from Tantra.dataset import JSONLDataset, extract_corpus_sample

IGNORE_INDEX = -100

def R(): return int(os.environ.get("RANK", "0"))
def LR(): return int(os.environ.get("LOCAL_RANK", "0"))
def WS(): return int(os.environ.get("WORLD_SIZE", "1"))
def main_rank(): return R() == 0
def log(s): print(f"[FSDP rank={R()} local={LR()}] {s}", flush=True)

def setup():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    if WS() != 2: raise RuntimeError(f"Expected WORLD_SIZE=2, got {WS()}")
    n = torch.cuda.device_count()
    if n < 2:
        raise RuntimeError(f"Worker sees only {n} CUDA device(s). CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}. Launch must expose 0,1 to torchrun.")
    local = LR()
    if local < 0 or local >= n:
        raise RuntimeError(f"LOCAL_RANK={local} but worker sees {n} GPUs. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    log(f"CUDA devices visible={n} | selected device={local} | {torch.cuda.get_device_name(local)}")
    return torch.device("cuda", local)

def make_cfg():
    v=VocabConfig(vocab_size=65536, byte_bpe_vocab=65536)
    c=NeuroCoreConfig(vocab=v)
    c.block.num_layers=32; c.block.alra.dim=1024; c.block.alra.num_heads=16; c.block.alra.head_dim=64; c.block.sgp.dim=1024
    c.moe.num_experts=1; c.moe.real_top1=False
    c.bitnet.enabled=True; c.bitnet.quantize_mode="ternary"; c.bitnet.use_shadow_weights=True
    return c,v

def get_tokenizer(vcfg,dataset,model_dir):
    os.makedirs(model_dir,exist_ok=True)
    path=os.path.join(model_dir,"tokenizer_65536.json")
    if main_rank() and not os.path.isfile(path):
        corpus=os.path.join(model_dir,"tokenizer_corpus.txt")
        log("Tokenizer not found -> building ONCE from all training records")
        extract_corpus_sample(dataset,corpus,max_lines=None)
        b=ByteBPETokenizer(vcfg)
        b.train([corpus],vocab_size=65536,special_tokens=list(vcfg.special_tokens.keys()))
        if b.vocab_size!=65536: raise RuntimeError(f"Tokenizer produced {b.vocab_size}; expected 65536")
        b.save(path)
        log(f"TOKENIZER CREATED: {path} | vocab=65,536")
    dist.barrier()
    if not os.path.isfile(path): raise RuntimeError(f"Tokenizer missing after rank barrier: {path}")
    b=ByteBPETokenizer.load(path,vcfg)
    if b.vocab_size!=65536: raise RuntimeError(f"Tokenizer mismatch: {b.vocab_size}")
    if main_rank(): log("TOKENIZER VERIFIED: 65,536 (reused; no rebuild)")
    return UnifiedTokenizer(vcfg,b,MegabytePatcher())

class RankShard(IterableDataset):
    def __init__(self,base): self.base=base
    def __iter__(self):
        for i,item in enumerate(iter(self.base)):
            if i % WS() == R(): yield item

def build_model(cfg,device):
    model=NeuroCoreModel(cfg).to(device)
    mp=MixedPrecision(param_dtype=torch.float32,reduce_dtype=torch.float16,buffer_dtype=torch.float16)
    for i,block in enumerate(model.layers):
        block=checkpoint_wrapper(block,checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        model.layers[i]=FSDP(block,sharding_strategy=ShardingStrategy.FULL_SHARD,mixed_precision=mp,device_id=device,use_orig_params=True)
    return FSDP(model,sharding_strategy=ShardingStrategy.FULL_SHARD,mixed_precision=mp,device_id=device,use_orig_params=True)

def save_sharded(model,opt,step,root):
    d=Path(root)/f"step_{step:07d}"; d.mkdir(parents=True,exist_ok=True); dist.barrier()
    with FSDP.state_dict_type(model,StateDictType.SHARDED_STATE_DICT,ShardedStateDictConfig(offload_to_cpu=True)):
        torch.save(model.state_dict(),d/f"model_rank{R():02d}.pt")
    torch.save(FSDP.optim_state_dict(model,opt),d/f"optimizer_rank{R():02d}.pt")
    if main_rank():
        (d/"meta.txt").write_text(f"step={step}\nworld_size=2\nparams=675226178\n",encoding="utf-8")
        (Path(root)/"LATEST").write_text(str(d),encoding="utf-8")
        log(f"SHARDED CHECKPOINT SAVED: {d}")
    dist.barrier()

def evaluate(model,loader,device,max_batches):
    model.eval(); ls=torch.zeros((),device=device,dtype=torch.float64); ns=torch.zeros((),device=device,dtype=torch.float64)
    with torch.no_grad():
        for j,(x,y) in enumerate(loader):
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
            with torch.autocast("cuda",dtype=torch.float16):
                z,_=model(x,return_mtp=False,use_latent_reasoning=False); z=z.reshape(-1,z.size(-1)); y=y.reshape(-1); m=y!=IGNORE_INDEX
                if m.any():
                    l=F.cross_entropy(z[m],y[m]); n=m.sum().double(); ls+=l.detach().double()*n; ns+=n
            if j+1>=max_batches: break
    dist.all_reduce(ls); dist.all_reduce(ns); model.train()
    return (ls/ns).item() if ns.item() else float("inf")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",required=True); p.add_argument("--val-dataset",required=True); p.add_argument("--model-dir",required=True)
    p.add_argument("--steps",type=int,default=10000); p.add_argument("--seq-len",type=int,default=512); p.add_argument("--batch-size",type=int,default=1); p.add_argument("--grad-accum",type=int,default=16)
    p.add_argument("--lr",type=float,default=1e-4); p.add_argument("--weight-decay",type=float,default=.1); p.add_argument("--warmup",type=int,default=2000); p.add_argument("--log-every",type=int,default=10); p.add_argument("--eval-every",type=int,default=500); p.add_argument("--checkpoint-every",type=int,default=500); p.add_argument("--data-workers",type=int,default=1); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); device=setup(); torch.manual_seed(a.seed+R()); random.seed(a.seed+R())
    if main_rank():
        log("============================================================"); log("TANTRA 0.675B — REAL 2-GPU FSDP TRAINING v3"); log("============================================================"); log("675,226,178 params | vocab=65,536 | 32x1024 | 16 heads | dense | BitNet ternary"); log("FSDP FULL_SHARD + activation checkpointing | REAL JSONL | NO synthetic | NO resume | NO DPO | NO auto-pilot | NO MTP | NO latent reasoning"); log(f"seq={a.seq_len} | batch/GPU={a.batch_size} | grad_accum={a.grad_accum} | GLOBAL TOKENS/OPT STEP={a.seq_len*a.batch_size*a.grad_accum*WS():,}")
    cfg,vcfg=make_cfg(); tok=get_tokenizer(vcfg,a.dataset,a.model_dir)
    train_ds=JSONLDataset(a.dataset,tok,seq_len=a.seq_len,max_samples=None,mask_non_assistant=True,split="all",val_ratio=0.0,pack_sequences=False,shuffle=True,shuffle_buf_size=2000,seed=a.seed)
    val_ds=JSONLDataset(a.val_dataset,tok,seq_len=a.seq_len,max_samples=None,mask_non_assistant=True,split="all",val_ratio=0.0,pack_sequences=False,shuffle=False,seed=a.seed)
    train_loader=DataLoader(RankShard(train_ds),batch_size=a.batch_size,num_workers=a.data_workers,pin_memory=True,persistent_workers=a.data_workers>0,prefetch_factor=2 if a.data_workers>0 else None)
    val_loader=DataLoader(RankShard(val_ds),batch_size=a.batch_size,num_workers=a.data_workers,pin_memory=True,persistent_workers=a.data_workers>0,prefetch_factor=2 if a.data_workers>0 else None)
    model=build_model(cfg,device)
    if main_rank(): log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay,betas=(.9,.95),eps=1e-8); scaler=torch.amp.GradScaler("cuda",enabled=True)
    it=iter(train_loader); micro=0; step=0; local_tokens=0; start=time.time(); run_loss=0.; run_n=0; opt.zero_grad(set_to_none=True); model.train()
    while step<a.steps:
        try:x,y=next(it)
        except StopIteration:it=iter(train_loader);x,y=next(it)
        x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True); s=step+1
        lr=a.lr*(s/max(1,a.warmup)) if s<=a.warmup else a.lr*.5*(1+math.cos(math.pi*min(1.,(s-a.warmup)/max(1,a.steps-a.warmup))))
        for g in opt.param_groups:g["lr"]=lr
        with torch.autocast("cuda",dtype=torch.float16):
            z,_=model(x,return_mtp=False,use_latent_reasoning=False); z=z.reshape(-1,z.size(-1)); yy=y.reshape(-1); m=yy!=IGNORE_INDEX; loss=F.cross_entropy(z[m],yy[m]) if m.any() else z.sum()*0.
        run_loss+=loss.detach().item(); run_n+=1; scaler.scale(loss/a.grad_accum).backward(); micro+=1; local_tokens+=x.numel()
        if micro%a.grad_accum==0:
            scaler.unscale_(opt); gn=torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
            if torch.isfinite(gn): scaler.step(opt); scaler.update()
            else: log("NON-FINITE GRADIENT — optimizer step skipped"); scaler.update()
            opt.zero_grad(set_to_none=True); step+=1
            if step%a.log_every==0 or step==1:
                lt=torch.tensor(run_loss/max(1,run_n),device=device,dtype=torch.float64); dist.all_reduce(lt); lt/=WS(); elapsed=time.time()-start; speed=local_tokens*WS()/max(elapsed,1e-9)
                if main_rank(): log(f"STEP {step:,}/{a.steps:,} | loss={lt.item():.4f} | ppl={math.exp(lt.item()) if lt.item()<20 else float('inf'):.1f} | lr={lr:.3e} | GLOBAL tok/s={speed:.1f} | peak VRAM={torch.cuda.max_memory_allocated(device)/1024**3:.2f}GB")
                run_loss=0.;run_n=0
            if step%a.eval_every==0:
                vl=evaluate(model,val_loader,device,20)
                if main_rank(): log(f"VALIDATION step={step:,} | loss={vl:.4f} | ppl={math.exp(vl) if vl<20 else float('inf'):.1f}")
            if step%a.checkpoint_every==0: save_sharded(model,opt,step,os.path.join(a.model_dir,"Checkpoints"))
    save_sharded(model,opt,step,os.path.join(a.model_dir,"Checkpoints")); dist.barrier()
    if main_rank(): log(f"COMPLETE | steps={step:,} | global tokens={local_tokens*WS():,} | hours={(time.time()-start)/3600:.2f}")
    dist.destroy_process_group()

if __name__=="__main__": main()
