"""Strict integrity checks for newly assembled serving bundles.

Manifest-less historical folders remain supported; manifest-bearing bundles must
bind the actual served files as well as adapter training identities.
"""
from __future__ import annotations
import json
from pathlib import Path
from Tantra.data.public_sources import sha256


class ServingIntegrityError(ValueError):
    pass


def load_serving_manifest(root):
    root=Path(root);path=root/'manifest.json'
    if not path.exists():return None
    try:manifest=json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError) as exc:raise ServingIntegrityError('Invalid serving manifest') from exc
    if manifest.get('version')!=2:
        raise ServingIntegrityError('Serving manifest lacks version-2 artifact binding; rebuild this candidate with scripts/build_serving.py')
    for key in ('base_sha256','tokenizer_sha256'):
        value=manifest.get(key)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ServingIntegrityError('Missing or invalid serving identity: '+key)
    artifacts=manifest.get('artifacts')
    if not isinstance(artifacts,dict):raise ServingIntegrityError('Missing serving artifact hashes')
    required={'base.pt','tokenizer.json'}
    if (root/'router_ngram.pt').exists():required.add('router_ngram.pt')
    if not required<=set(artifacts):raise ServingIntegrityError('Missing served base/tokenizer/router artifact binding')
    if set(artifacts)-{'base.pt','tokenizer.json','router_ngram.pt'}:raise ServingIntegrityError('Unknown serving artifact path')
    for name,expected in artifacts.items():
        candidate=root/name
        if not candidate.is_file() or sha256(candidate)!=expected:raise ServingIntegrityError('Serving artifact checksum mismatch: '+name)
    if artifacts['tokenizer.json']!=manifest['tokenizer_sha256']:raise ServingIntegrityError('Tokenizer artifact and training identity differ')
    experts=manifest.get('experts')
    if not isinstance(experts,dict):raise ServingIntegrityError('Missing expert declarations')
    for name,record in experts.items():
        if not isinstance(name,str) or not name or Path(name).name!=name or any(c in name for c in '/\\:'):
            raise ServingIntegrityError('Invalid serving expert name')
        if not isinstance(record,dict) or record.get('file')!=f'experts/{name}.pt' or not isinstance(record.get('rank'),int) or record['rank']<1:
            raise ServingIntegrityError('Invalid serving expert declaration: '+name)
        expected=record.get('sha256')
        if not isinstance(expected,str) or len(expected)!=64:raise ServingIntegrityError('Missing expert artifact hash: '+name)
    return manifest
