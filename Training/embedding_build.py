import json
from pathlib import Path
from typing import Iterable, List


def yield_chunks(text: str, max_len: int = 800) -> List[str]:
    words = text.split()
    out = []
    cur = []
    for w in words:
        cur.append(w)
        if sum(len(x) + 1 for x in cur) > max_len:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def build_embeddings(input_files: List[str], out_dir: str):
    from sentence_transformers import SentenceTransformer
    import faiss

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    texts_path = Path(out_dir) / "rag_texts.json"
    index_path = Path(out_dir) / "rag_index.faiss"

    docs = []
    for p in input_files:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
                for chunk in yield_chunks(txt):
                    docs.append({"path": p, "content": chunk})
        except Exception:
            pass

    with open(texts_path, "w", encoding="utf-8") as f:
        json.dump(docs, f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([d["content"] for d in docs], convert_to_numpy=True, show_progress_bar=False)
    dim = embeddings.shape[1] if embeddings.size else 384
    index = faiss.IndexHNSWFlat(dim, 32)
    if embeddings.size:
        index.add(embeddings)
    faiss.write_index(index, str(index_path))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_files", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build_embeddings(a.input_files, a.out)


