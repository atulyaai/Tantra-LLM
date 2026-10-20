"""Integrity-checked resumable checkpoints with three bounded generations.

Only corruption permits fallback. Identity/configuration mismatches fail closed.
Model files are trusted local training artifacts (PyTorch pickle), not uploads.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Mapping

import torch


class CheckpointCorruptError(RuntimeError):
    """No intact compatible resumable generation could be read."""


class CheckpointIdentityError(ValueError):
    """A readable checkpoint belongs to different data, model or settings."""


def _digest(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as reader:
        for block in iter(lambda:reader.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _sidecar(path):
    return Path(str(path)+'.integrity.json')


def generations(path):
    path=Path(path)
    return [path,path.with_name(path.stem+'.previous-1'+path.suffix),
            path.with_name(path.stem+'.previous-2'+path.suffix)]


def _write_integrity(path, digest, size):
    sidecar=_sidecar(path); temporary=sidecar.with_name(sidecar.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with temporary.open('w',encoding='utf-8') as writer:
            json.dump({'version':1,'sha256':digest,'bytes':size},writer,sort_keys=True)
            writer.flush();os.fsync(writer.fileno())
        os.replace(temporary,sidecar)
    finally:
        if temporary.exists():temporary.unlink()


def _copy_generation(source,target):
    """Publish the copied bytes before their checksum; interrupted pairs are rejected."""
    temporary=target.with_name(target.name+'.tmp-'+uuid.uuid4().hex)
    try:
        try:os.link(source,temporary)
        except OSError:
            with source.open('rb') as reader,temporary.open('wb') as writer:
                shutil.copyfileobj(reader,writer,1024*1024);writer.flush();os.fsync(writer.fileno())
        # Retain the recorded checksum rather than blessing a corrupted source anew.
        integrity=_sidecar(source)
        if integrity.exists():
            try:
                recorded=json.loads(integrity.read_text(encoding='utf-8'))
                digest,size=recorded['sha256'],recorded['bytes']
            except (OSError,ValueError,KeyError,TypeError):
                digest,size='unreadable-source-integrity',temporary.stat().st_size
        else:digest,size=_digest(temporary),temporary.stat().st_size
        os.replace(temporary,target);_write_integrity(target,digest,size)
    finally:
        if temporary.exists():temporary.unlink()


def save_resumable(path,payload):
    """Save latest plus two previous generations; preserve best checkpoints separately."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        # Do all expensive serialization before rotating anything.
        with temporary.open('wb') as writer:
            torch.save(payload,writer);writer.flush();os.fsync(writer.fileno())
        digest,size=_digest(temporary),temporary.stat().st_size
        latest,previous1,previous2=generations(path)
        if previous1.exists():_copy_generation(previous1,previous2)
        if latest.exists():_copy_generation(latest,previous1)
        os.replace(temporary,path);_write_integrity(path,digest,size)
    finally:
        if temporary.exists():temporary.unlink()


def _finite(value):
    if isinstance(value,torch.Tensor):
        return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all())
    if isinstance(value,Mapping):return all(_finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(_finite(v) for v in value)
    return True


def load_resumable(path,*,identity_key,expected_identity,expected_config,required,
                   expected_fields=None,state_shapes=None):
    """Return (payload, recovery evidence), trying at most three generations.

    Missing checksums are accepted only for existing legacy checkpoints, and are
    disclosed in recovery evidence. Checksummed generations reject any changed
    byte before unpickling. A readable identity/config mismatch is never fallback.
    """
    rejected=[]
    for candidate in generations(path):
        if not candidate.exists():continue
        integrity=_sidecar(candidate);verified=False
        try:
            if integrity.exists():
                metadata=json.loads(integrity.read_text(encoding='utf-8'))
                if metadata.get('version')!=1 or metadata.get('bytes')!=candidate.stat().st_size or metadata.get('sha256')!=_digest(candidate):
                    raise CheckpointCorruptError('Integrity checksum/length mismatch')
                verified=True
            payload=torch.load(candidate,map_location='cpu',weights_only=False)
            if not isinstance(payload,dict):raise CheckpointCorruptError('Checkpoint is not a dictionary')
            if identity_key not in payload or 'config' not in payload:
                raise CheckpointCorruptError('Missing checkpoint identity/configuration')
            if payload[identity_key]!=expected_identity:raise CheckpointIdentityError('Resume identity mismatch: '+str(candidate))
            if payload['config']!=expected_config:raise CheckpointIdentityError('Resume model configuration mismatch: '+str(candidate))
            for key,value in (expected_fields or {}).items():
                if key not in payload:raise CheckpointCorruptError('Missing checkpoint field: '+key)
                if payload[key]!=value:raise CheckpointIdentityError('Resume setting mismatch ('+key+'): '+str(candidate))
            missing=set(required)-set(payload)
            if missing:raise CheckpointCorruptError('Missing resumable fields: '+', '.join(sorted(missing)))
            if not isinstance(payload.get('state'),dict) or not isinstance(payload.get('optimizer'),dict):
                raise CheckpointCorruptError('Invalid model/optimizer state')
            if state_shapes is not None:
                saved=payload['state']
                if set(saved)!=set(state_shapes) or any(not isinstance(saved[k],torch.Tensor) or saved[k].shape!=value.shape for k,value in state_shapes.items()):
                    raise CheckpointIdentityError('Resume model state layout mismatch: '+str(candidate))
            if any(not isinstance(payload[key],int) or isinstance(payload[key],bool) or payload[key]<0 for key in ('cursor','updates','tokens')):
                raise CheckpointCorruptError('Invalid checkpoint progress counters')
            if not isinstance(payload['torch_rng'],torch.Tensor) or payload['torch_rng'].dtype!=torch.uint8:
                raise CheckpointCorruptError('Invalid Torch RNG state')
            if not _finite(payload['state']) or not _finite(payload['optimizer']):
                raise CheckpointCorruptError('Non-finite model/optimizer tensors')
            return payload,{'loaded_path':str(candidate),'fallback_used':candidate!=Path(path),
                            'integrity_verified':verified,'rejected':rejected,
                            'legacy_without_checksum':not verified}
        except CheckpointIdentityError:raise
        except (OSError,ValueError,RuntimeError,EOFError,KeyError,TypeError,IndexError) as exc:
            rejected.append({'path':str(candidate),'error':type(exc).__name__+': '+str(exc)})
        except Exception as exc:
            # PyTorch/pickle corruption has several exception types; do not hide it.
            import pickle
            if not isinstance(exc,pickle.UnpicklingError):raise
            rejected.append({'path':str(candidate),'error':type(exc).__name__+': '+str(exc)})
    raise CheckpointCorruptError('No intact resumable checkpoint among three generations: '+json.dumps(rejected))
