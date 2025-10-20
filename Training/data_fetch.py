import json
from pathlib import Path
from typing import Dict, Any


def write_jsonl(items, out_file: str):
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps({"text": it}) + "\n")


def fetch_hf(entry: Dict[str, Any]):
    from datasets import load_dataset
    name = entry['hf_name']
    config = entry.get('hf_config') or None
    split = entry.get('split', 'train')
    field = entry.get('text_field') or 'text'
    max_items = int(entry.get('max_items', 1000000))  # 1M default
    
    try:
        ds = load_dataset(name, config, split=split)
        items = []
        for ex in ds:
            val = ex.get(field)
            if isinstance(val, str) and val.strip():
                items.append(val)
                if len(items) >= max_items:
                    break
        write_jsonl(items, entry['out_file'])
    except Exception:
        # Fallback to streaming mode (Datasets v3+ without scripts)
        ds = load_dataset(name, config, split=split, streaming=True)
        Path(entry['out_file']).parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(entry['out_file'], 'w', encoding='utf-8') as f:
            for ex in ds:
                val = ex.get(field)
                if isinstance(val, str) and val.strip():
                    f.write(json.dumps({"text": val}) + "\n")
                    count += 1
                    if count >= max_items:
                        break


def fetch_http_text(entry: Dict[str, Any]):
    import requests
    url = entry['url']
    txt = requests.get(url, timeout=60).text
    # naive paragraphs split
    items = [p.strip() for p in txt.split('\n\n') if p.strip()]
    write_jsonl(items, entry['out_file'])


if __name__ == '__main__':
    import argparse, yaml
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    for entry in cfg.get('sources', []):
        if not entry.get('enabled'):
            continue
        t = entry.get('type')
        if t == 'hf':
            fetch_hf(entry)
        elif t == 'http_text':
            fetch_http_text(entry)
        else:
            print(f"Unknown type: {t}")

