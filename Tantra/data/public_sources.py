"""Revision-pinned, checksum-verified public data acquisition (no paid services)."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path
from Tantra.config import ROOT

LOCK = ROOT/'docs'/'data_sources.lock.json'
RAW = ROOT/'Datasets'/'raw'/'public-v1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sources():
    return json.loads(LOCK.read_text(encoding='utf-8'))['sources']


def location(source, root=RAW):
    return Path(root)/source['id']/Path(source['path']).name


def fetch(source, root=RAW):
    path = location(source, root)
    if path.exists():
        if path.stat().st_size != source['size'] or sha256(path) != source['sha256']:
            raise ValueError('Existing source checksum mismatch: '+str(path))
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free < source['size'] + 2*1024**3:
        raise OSError('Need source size plus 2 GiB free disk')
    url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{source['path']}?download=true"
    temp = path.with_suffix(path.suffix+'.part')
    # Resume only complete downloaded ranges; final full-file hash remains mandatory.
    chunk_size=4*1024*1024
    size=temp.stat().st_size if temp.exists() else 0
    if size>source['size'] or (size!=source['size'] and size%chunk_size):
        raise ValueError('Partial download has an invalid range boundary')
    h=hashlib.sha256()
    if size:
        with temp.open('rb') as reader:
            for chunk in iter(lambda:reader.read(1024*1024),b''): h.update(chunk)
    import time
    with temp.open('ab') as writer:
        for start in range(size,source['size'],chunk_size):
            end=min(source['size']-1,start+chunk_size-1)
            expected=f"bytes {start}-{end}/{source['size']}"
            for attempt in range(3):
                try:
                    request=urllib.request.Request(url+f'&range_start={start}',headers={
                        'User-Agent':'Tantra-dataset-pipeline/1','Range':f'bytes={start}-{end}'})
                    with urllib.request.urlopen(request,timeout=45) as response:
                        if response.status!=206 or response.headers.get('Content-Range')!=expected:
                            raise ValueError('Server did not honor exact byte range')
                        block=response.read(end-start+2)
                        if len(block)!=end-start+1: raise ValueError('Truncated source range')
                    break
                except (OSError,ValueError):
                    if attempt==2: raise
                    time.sleep(2*(attempt+1))
            writer.write(block);writer.flush();h.update(block);size+=len(block)
            if start%(32*1024*1024)==0:
                print(json.dumps({'source':source['id'],'downloaded_bytes':size,'total_bytes':source['size']}),flush=True)
    if size!=source['size'] or h.hexdigest()!=source['sha256']:
        raise ValueError('Downloaded source checksum mismatch; partial retained for diagnosis')
    os.replace(temp,path)
    # Retain attribution and dataset card at the same revision, beside source data.
    for filename in ('README.md',):
        card = f"https://huggingface.co/datasets/{source['repo']}/raw/{source['revision']}/{filename}"
        with urllib.request.urlopen(card, timeout=60) as response:
            (path.parent/filename).write_bytes(response.read(2*1024*1024))
    return path


def main():
    from concurrent.futures import ThreadPoolExecutor
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ids', nargs='+')
    a = p.parse_args()
    known={s['id']:s for s in sources()}
    if a.ids and set(a.ids)-set(known): p.error('Unknown source ID')
    items=[known[name] for name in dict.fromkeys(a.ids)] if a.ids else list(known.values())
    if sum(s['size'] for s in items)>50*1024**3: p.error('Initial source budget exceeds 50 GiB')
    if not items: p.error('No matching source IDs')
    # At most two downloads; never a corpus-sized in-memory buffer.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for source, path in zip(items, pool.map(fetch, items)):
            print(json.dumps({'source':source['id'],'path':str(path),'sha256':source['sha256']}), flush=True)

if __name__ == '__main__': main()
