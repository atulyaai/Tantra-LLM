"""Candidate promotion and rollback for local Tantra model artifacts."""
from __future__ import annotations
import hashlib,json,os,tempfile
from pathlib import Path

def _artifact_hash(root):
    root=Path(root).resolve()
    if not root.is_dir(): raise ValueError("candidate artifact directory is missing")
    h=hashlib.sha256(); files=sorted(p for p in root.rglob("*") if p.is_file())
    if not files: raise ValueError("candidate artifact directory is empty")
    for p in files:
        h.update(p.relative_to(root).as_posix().encode()); h.update(b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()

class ModelLifecycle:
    def __init__(self,path="Model/lifecycle.json"):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        if not self.path.exists(): self._write({"schema_version":1,"candidates":{},"active":None,"previous":None,"events":[]})
    def _read(self): return json.loads(self.path.read_text(encoding="utf-8"))
    def _write(self,data):
        fd,tmp=tempfile.mkstemp(dir=self.path.parent,prefix=".lifecycle-")
        try:
            with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(data,f,indent=2); f.write("\n")
            os.replace(tmp,self.path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
    def register(self,candidate_id,artifact_dir,parent_id,metadata=None):
        data=self._read()
        if candidate_id in data["candidates"]: raise ValueError("candidate id already exists")
        if parent_id not in data["candidates"] and parent_id!=data["active"]:
            # Parent may be the initial foundation; allow explicit root marker.
            if parent_id!="foundation": raise ValueError("unknown parent model")
        record={"candidate_id":candidate_id,"artifact_dir":str(Path(artifact_dir).resolve()),
                "artifact_sha256":_artifact_hash(artifact_dir),"parent_id":parent_id,
                "metadata":metadata or {},"status":"candidate"}
        data["candidates"][candidate_id]=record; data["events"].append({"event":"registered","candidate_id":candidate_id})
        self._write(data); return record
    def promote(self,candidate_id,report):
        data=self._read()
        if candidate_id not in data["candidates"]: raise KeyError(candidate_id)
        rec=data["candidates"][candidate_id]
        if _artifact_hash(rec["artifact_dir"])!=rec["artifact_sha256"]: raise ValueError("candidate artifact changed after registration")
        required=("tests_passed","identity_passed","safety_passed","heldout_score","baseline_score","max_regression","latency_ms","latency_budget_ms")
        if any(k not in report for k in required): raise ValueError("promotion report is incomplete")
        if not all(report[k] is True for k in ("tests_passed","identity_passed","safety_passed")): raise ValueError("promotion blocked by failed test/identity/safety gate")
        if float(report["heldout_score"])<float(report["baseline_score"]): raise ValueError("held-out score regressed")
        if float(report["max_regression"])>.02: raise ValueError("old capability regression exceeds 2%")
        if float(report["latency_ms"])>float(report["latency_budget_ms"]): raise ValueError("latency budget exceeded")
        old=data["active"]; data["previous"]=old; data["active"]=candidate_id; rec["status"]="promoted"
        data["events"].append({"event":"promoted","candidate_id":candidate_id,"previous":old,"report":report})
        self._write(data)
    def rollback(self):
        data=self._read()
        if not data["previous"]: raise ValueError("no previous model to roll back to")
        current=data["active"]; data["active"]=data["previous"]; data["previous"]=current
        data["events"].append({"event":"rollback","from":current,"to":data["active"]})
        self._write(data); return data["active"]
