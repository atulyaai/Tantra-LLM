import json
from pathlib import Path
from typing import List


def build_index(input_files: List[str], out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    texts_path = out / "rag_texts.json"
    # Minimal scaffold: concatenate files into a JSON list
    texts = []
    for p in input_files:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                texts.append({"path": p, "content": f.read()})
        except Exception:
            pass
    with open(texts_path, "w", encoding="utf-8") as f:
        json.dump(texts, f)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_files", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build_index(a.input_files, a.out)


