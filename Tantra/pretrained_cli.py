"""CLI for the pretrained Tantra model gateway and expansion workflow."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from Tantra.model_registry import ModelRegistry
from Tantra.model_gateway import ModelGateway
from Tantra.model_lifecycle import ModelLifecycle
from Tantra.pretrained_expansion import (
    add_residual_branch,approve_teacher_dataset,count_parameters,generate_teacher_dataset,save_residual_candidate,transfer_compatible_weights,
    train_lora,train_residual_branch,transfer_adapter,
)

def _load_model(checkpoint,device="cpu"):
    try:
        from transformers import AutoModelForCausalLM,AutoTokenizer
    except ImportError as exc: raise RuntimeError("install pretrained/finetune extras first") from exc
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False).to(device)
    return model,tokenizer

def main(argv=None)->int:
    parser=argparse.ArgumentParser(prog="python -m Tantra.pretrained_cli")
    sub=parser.add_subparsers(dest="command",required=True)
    def registry_arg(p): p.add_argument("--registry",default="configs/models/registry.json")
    info=sub.add_parser("info"); registry_arg(info)
    smoke=sub.add_parser("smoke"); registry_arg(smoke)
    smoke.add_argument("--model-id",required=True); smoke.add_argument("--checkpoint",required=True)
    smoke.add_argument("--device",default="cpu"); smoke.add_argument("--max-new-tokens",type=int,default=48)
    smoke.add_argument("--adapter",help="optional PEFT LoRA adapter directory")
    smoke.add_argument("--residual-candidate",help="optional saved residual-branch candidate directory")
    lora=sub.add_parser("train-lora"); registry_arg(lora)
    lora.add_argument("--model-id",required=True); lora.add_argument("--checkpoint",required=True)
    lora.add_argument("--candidate-id",required=True); lora.add_argument("--data",required=True)
    lora.add_argument("--output",required=True); lora.add_argument("--steps",type=int,default=100); lora.add_argument("--rank",type=int,default=8)
    lora.add_argument("--replay-data",help="optional old-capability JSONL replay dataset")
    lora.add_argument("--replay-ratio",type=float,default=0.25)
    approve=sub.add_parser("approve-dataset")
    approve.add_argument("--dataset",required=True); approve.add_argument("--reviewed-by",required=True); approve.add_argument("--review-note",required=True)
    distil=sub.add_parser("distill"); registry_arg(distil)
    distil.add_argument("--model-id",required=True); distil.add_argument("--checkpoint",required=True)
    distil.add_argument("--prompts",required=True); distil.add_argument("--output",required=True)
    grow=sub.add_parser("grow-residual"); registry_arg(grow)
    grow.add_argument("--model-id",required=True); grow.add_argument("--checkpoint",required=True)
    grow.add_argument("--candidate-id",required=True); grow.add_argument("--data",required=True)
    grow.add_argument("--output",required=True); grow.add_argument("--steps",type=int,default=100)
    grow.add_argument("--layer-index",type=int,default=-1); grow.add_argument("--bottleneck",type=int,default=64)
    transfer=sub.add_parser("transfer-adapter")
    transfer.add_argument("--parent-adapter",required=True); transfer.add_argument("--child-adapter",required=True)
    transfer.add_argument("--expected-base-sha256",required=True)
    weights=sub.add_parser("transfer-weights"); registry_arg(weights)
    weights.add_argument("--parent-model-id",required=True); weights.add_argument("--parent-checkpoint",required=True)
    weights.add_argument("--child-checkpoint",required=True); weights.add_argument("--candidate-id",required=True)
    weights.add_argument("--output",required=True); weights.add_argument("--prefixes",nargs="*")
    promote=sub.add_parser("promote"); promote.add_argument("--candidate-id",required=True)
    promote.add_argument("--report",required=True); promote.add_argument("--lifecycle",default="runs/model-lifecycle.json")
    rollback=sub.add_parser("rollback"); rollback.add_argument("--lifecycle",default="runs/model-lifecycle.json")
    args=parser.parse_args(argv)
    if args.command=="info":
        print(json.dumps(ModelRegistry.from_file(args.registry).to_dict(),indent=2)); return 0
    if args.command=="smoke":
        spec=ModelRegistry.from_file(args.registry).get(args.model_id)
        if not Path(args.checkpoint).is_dir(): parser.error("local checkpoint directory missing; acquire the pinned snapshot explicitly")
        gateway=ModelGateway(spec,args.checkpoint,device=args.device,adapter_path=args.adapter)
        start=time.perf_counter(); gateway.load()
        if args.residual_candidate:
            from Tantra.pretrained_expansion import load_residual_candidate
            load_residual_candidate(gateway._model,args.residual_candidate)
        load_s=time.perf_counter()-start
        start=time.perf_counter(); answer=gateway.generate([{"role":"user","content":"Introduce yourself as Tantra in one sentence."}],max_new_tokens=args.max_new_tokens)
        gen_s=time.perf_counter()-start
        print(json.dumps({"model_id":spec.model_id,"revision":spec.revision,"license":spec.license_id,
            "load_seconds":round(load_s,3),"generation_seconds":round(gen_s,3),"output":answer,
            "status":"runtime_smoke_only_not_quality_evaluation"},indent=2)); gateway.unload(); return 0
    if args.command=="train-lora":
        spec=ModelRegistry.from_file(args.registry).get(args.model_id)
        result=train_lora(args.checkpoint,args.data,args.output,steps=args.steps,rank=args.rank,replay_jsonl=args.replay_data,replay_ratio=args.replay_ratio)
        life=ModelLifecycle(); record=life.register(args.candidate_id,args.output,args.model_id,
            {"method":"lora","base_revision":spec.revision,**result})
        print(json.dumps({"candidate":record,"training":result},indent=2)); return 0
    if args.command=="approve-dataset":
        print(json.dumps(approve_teacher_dataset(args.dataset,args.reviewed_by,args.review_note),indent=2)); return 0
    if args.command=="distill":
        spec=ModelRegistry.from_file(args.registry).get(args.model_id)
        model,tokenizer=_load_model(args.checkpoint)
        prompts=[x.strip() for x in Path(args.prompts).read_text(encoding="utf-8").splitlines() if x.strip()]
        result=generate_teacher_dataset(model,tokenizer,prompts,args.output)
        result["teacher_model_id"]=spec.model_id
        result["teacher_revision"]=spec.revision
        manifest_path=Path(args.output).with_suffix(Path(args.output).suffix+".manifest.json")
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["teacher_model_id"]=spec.model_id
        manifest["teacher_revision"]=spec.revision
        manifest["license_id"]=spec.license_id
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,indent=2)); return 0
    if args.command=="grow-residual":
        spec=ModelRegistry.from_file(args.registry).get(args.model_id)
        model,tokenizer=_load_model(args.checkpoint); before=count_parameters(model)
        add_residual_branch(model,args.layer_index,args.bottleneck)
        training=train_residual_branch(model,tokenizer,args.data,steps=args.steps)
        metadata=save_residual_candidate(model,args.output,args.layer_index,args.bottleneck)
        after=count_parameters(model)
        life=ModelLifecycle(); record=life.register(args.candidate_id,args.output,args.model_id,
            {"method":"residual_branch","base_revision":spec.revision,"source_parameters":before,
             "candidate_parameters":after,"added_parameters":after-before,"training":training,**metadata})
        print(json.dumps({"candidate":record},indent=2)); return 0
    if args.command=="transfer-adapter":
        print(transfer_adapter(args.parent_adapter,args.child_adapter,expected_base_sha256=args.expected_base_sha256)); return 0
    if args.command=="transfer-weights":
        spec=ModelRegistry.from_file(args.registry).get(args.parent_model_id)
        parent,_=_load_model(args.parent_checkpoint); child,tokenizer=_load_model(args.child_checkpoint)
        report=transfer_compatible_weights(parent,child,prefixes=args.prefixes)
        output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
        child.save_pretrained(output); tokenizer.save_pretrained(output)
        record=ModelLifecycle().register(args.candidate_id,output,args.parent_model_id,
            {"method":"compatible_weight_transfer","parent_revision":spec.revision,**report})
        print(json.dumps({"candidate":record,"transfer":report},indent=2)); return 0
    if args.command=="promote":
        ModelLifecycle(args.lifecycle).promote(args.candidate_id,json.loads(Path(args.report).read_text(encoding="utf-8")))
        print(json.dumps({"active_candidate":args.candidate_id,"status":"promoted"},indent=2)); return 0
    if args.command=="rollback":
        print(json.dumps({"active_candidate":ModelLifecycle(args.lifecycle).rollback(),"status":"rolled_back"},indent=2)); return 0
    return 2

if __name__=="__main__":
    raise SystemExit(main())
