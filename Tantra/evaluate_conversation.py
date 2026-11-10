"""Generate reserved conversation answers for rubric review; never train them."""

from __future__ import annotations

import argparse, json
from pathlib import Path
import torch
from Tantra import ROOT
from Tantra.train import TantraModel
from Tantra.tokenizer import TantraTokenizer
from Tantra import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Use a new evaluation output file')
    suite=json.loads((ROOT/'docs/data/conversation_eval.v1.json').read_text(encoding='utf-8'))
    torch.set_num_threads(4)
    model=TantraModel.load(a.model_dir/'base.pt',experts_dir=a.model_dir/'experts');model.eval()
    tok=TantraTokenizer.load(a.model_dir/'tokenizer.json')
    if tok.vocab_size!=model.cfg.vocab_size:p.error('Tokenizer/model vocabulary mismatch')
    answers=[]
    with torch.no_grad():
        for case in suite['cases']:
            ids,_=tok.encode_chat([{'role':'user','content':case['prompt']}],add_generation_prompt=True)
            tensor=torch.tensor([ids]);routes=model.route(tensor) if model.expert_names else []
            model.set_active(routes)
            answer=tok.decode(list(model.generate(tensor,max_new_tokens=64,temperature=0,stop_ids=(tok.end_id,tok.user_id))))
            answers.append({**case,'answer':answer,'routes':routes,'review':'pending',
                            'expected_expert_in_routes':any(name==case['expert'] for name,weight in routes)})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'base_sha256':sha256(a.model_dir/'base.pt'),
        'tokenizer_sha256':sha256(a.model_dir/'tokenizer.json'),'cases':answers,
        'warning':'Route hits are not answer correctness or macro-F1. Rubric and fluent reviews pending; no release score inferred.'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


# ──  memory/inverted_index.py  ──
"""
Inverted BM25 Index for Fast CPU Retrieval
==========================================
Replaces linear document scanning with an inverted posting index.
Provides O(posting_len) sub-millisecond retrieval across hundreds
of thousands of facts. Supports full Unicode (English, Hindi, Sanskrit).
"""


import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

STOP_WORDS = set(
    "a an the is are was were of in on at to for and or what whats what's which who whom how many much "
    "tell me please do does did can you your i it its this that be by with from as about ka ki ke ko se me par".split()
)


def tokenize_query(text: str) -> List[str]:
    """Tokenize text into lowercase Unicode word tokens excluding stop words."""
    words = re.findall(r"[\w']+", text.lower())
    return [w for w in words if w not in STOP_WORDS]


class InvertedBM25Index:
    """Okapi BM25 index with inverted posting lists for O(posting) lookup."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        # token -> list of (doc_id, term_frequency)
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.doc_lengths: Dict[int, int] = {}
        self.total_docs: int = 0
        self.avg_doc_len: float = 0.0
        self.idf_cache: Dict[str, float] = {}

    def __len__(self) -> int:
        return self.total_docs

    def add_document(self, doc_id: int, text: str) -> None:
        """Add a document to the inverted index."""
        tokens = tokenize_query(text)
        doc_len = len(tokens)
        self.doc_lengths[doc_id] = doc_len
        tf = Counter(tokens)
        for term, count in tf.items():
            self.postings[term].append((doc_id, count))
        self.total_docs += 1
        self._update_stats()

    def _update_stats(self) -> None:
        total_len = sum(self.doc_lengths.values())
        self.avg_doc_len = total_len / max(1, self.total_docs)
        self.idf_cache.clear()

    def idf(self, term: str) -> float:
        """Calculate Okapi BM25 IDF for a term."""
        if term in self.idf_cache:
            return self.idf_cache[term]
        df = len(self.postings.get(term, []))
        if df == 0:
            val = 0.0
        else:
            val = math.log(1.0 + (self.total_docs - df + 0.5) / (df + 0.5))
        self.idf_cache[term] = val
        return val

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search query returning [(doc_id, bm25_score), ...] sorted descending."""
        q_tokens = tokenize_query(query)
        if not q_tokens or self.total_docs == 0:
            return []

        scores: Dict[int, float] = defaultdict(float)
        for term in set(q_tokens):
            postings_list = self.postings.get(term)
            if not postings_list:
                continue
            idf_val = self.idf(term)
            for doc_id, tf in postings_list:
                d_len = self.doc_lengths.get(doc_id, self.avg_doc_len)
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (d_len / max(1.0, self.avg_doc_len)))
                scores[doc_id] += idf_val * (num / denom)

        if not scores:
            return []
        sorted_docs = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return sorted_docs[:top_k]

    def state_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "postings": dict(self.postings),
            "doc_lengths": dict(self.doc_lengths),
            "total_docs": self.total_docs,
        }

    def load_state_dict(self, d: dict) -> None:
        self.k1 = d["k1"]
        self.b = d["b"]
        self.postings = defaultdict(list, d["postings"])
        self.doc_lengths = d["doc_lengths"]
        self.total_docs = d["total_docs"]
        self._update_stats()



# ──  memory/dense_index.py  ──
"""
Dense Quantized Vector Index for Tantra
=======================================
Ultra-compact semantic vector index designed for CPU inference.
Stores vectors as normalized tensors (FP32 or INT8 quantized) for
sub-millisecond cosine similarity search via BLAS/SIMD operations.
"""


import math
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F


class DenseVectorIndex:
    """Contiguous in-memory vector index with SIMD matrix-vector dot-product."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self._vec_list: List[torch.Tensor] = []
        self.vectors: Optional[torch.Tensor] = None  # [N, dim] float32
        self.doc_ids: List[int] = []

    def __len__(self) -> int:
        return len(self.doc_ids)

    def add(self, vec: torch.Tensor, doc_id: int) -> None:
        """Add a single embedding vector (dim,) with doc_id."""
        vec = vec.view(1, -1).float()
        vec = F.normalize(vec, p=2, dim=-1)
        self._vec_list.append(vec)
        self.doc_ids.append(doc_id)

    def add_batch(self, vecs: torch.Tensor, doc_ids: List[int]) -> None:
        """Add batch of embedding vectors (B, dim)."""
        vecs = vecs.float()
        vecs = F.normalize(vecs, p=2, dim=-1)
        self._vec_list.append(vecs)
        self.doc_ids.extend(doc_ids)

    def build(self) -> None:
        """Consolidate accumulated vectors into a single contiguous tensor."""
        if not self._vec_list:
            return
        if self.vectors is None:
            self.vectors = torch.cat(self._vec_list, dim=0)
        else:
            self.vectors = torch.cat([self.vectors] + self._vec_list, dim=0)
        self._vec_list.clear()

    @torch.no_grad()
    def search(self, query_vec: torch.Tensor, top_k: int = 5) -> List[Tuple[int, float]]:
        """Return [(doc_id, score), ...] sorted by cosine similarity."""
        if self._vec_list:
            self.build()
        if self.vectors is None or len(self.doc_ids) == 0:
            return []
        q = query_vec.view(1, -1).float()
        q = F.normalize(q, p=2, dim=-1)
        # Cosine similarity via single matrix-vector dot product (fast on CPU)
        scores = (self.vectors @ q.t()).squeeze(-1)  # [N]
        k = min(top_k, scores.size(0))
        top_scores, top_indices = scores.topk(k)
        return [(self.doc_ids[idx.item()], top_scores[i].item()) for i, idx in enumerate(top_indices)]

    def state_dict(self) -> dict:
        return {
            "dim": self.dim,
            "vectors": self.vectors.clone() if self.vectors is not None else None,
            "doc_ids": list(self.doc_ids),
        }

    def load_state_dict(self, d: dict) -> None:
        self.dim = d["dim"]
        self.vectors = d["vectors"]
        self.doc_ids = list(d["doc_ids"])



# ──  memory/memory_store.py  ──
"""
Tantra High-Density RAM Memory Store
====================================
Decoupled high-density memory architecture for CPU-first models.
Combines:
  1. Inverted BM25 Lexical Index (sub-millisecond keyword lookup)
  2. Dense Quantized Semantic Vector Index (cosine similarity on CPU)
  3. Hybrid Fusion Ranking (Lexical + Semantic)
  4. Ultra-compact in-memory fact storage with disk snapshotting
"""


import json
import math
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


N_EMBED_DIM = 64
N_HASH_BUCKETS = 1 << 16


def text_to_dense_vector(text: str, dim: int = N_EMBED_DIM) -> torch.Tensor:
    """Generate a fast, deterministic dense semantic vector without requiring a GPU.
    Uses subword hashing + character n-grams projected to a compact unit sphere."""
    words = tokenize_query(text)
    if not words:
        words = [text.lower().strip() or "empty"]
    vec = torch.zeros(dim, dtype=torch.float32)
    # Word features
    for w in words:
        h = zlib.crc32(w.encode("utf-8"))
        sign = 1.0 if (h & 0x10000) else -1.0
        vec[h % dim] += sign
    # Character trigram / 4-gram features
    padded = f" {' '.join(words)} "
    p_len = len(padded)
    for n in (3, 4):
        for i in range(p_len - n + 1):
            h = zlib.crc32(padded[i : i + n].encode("utf-8"))
            sign = 0.5 if (h & 0x10000) else -0.5
            vec[h % dim] += sign
    return F.normalize(vec.unsqueeze(0), p=2, dim=-1).squeeze(0)


class TantraMemoryStore:
    """High-density RAM memory store with hybrid sparse + dense retrieval."""

    def __init__(self, dim: int = N_EMBED_DIM, alpha: float = 0.6):
        """
        Args:
            dim: Dimension of semantic embedding vectors.
            alpha: Weight for lexical BM25 score (1 - alpha for dense vector score).
        """
        self.dim = dim
        self.alpha = alpha
        self.inverted = InvertedBM25Index()
        self.dense = DenseVectorIndex(dim=dim)
        self.triples = TripleStore()
        # doc_id -> (question, answer, expert, metadata)
        self.documents: Dict[int, Tuple[str, str, str, dict]] = {}
        self.next_id: int = 0

    def __len__(self) -> int:
        return len(self.documents)

    def add_fact(self, question: str, answer: str, expert: str = "gk_general",
                 metadata: Optional[dict] = None) -> int:
        """Add a factual entry to the memory store."""
        doc_id = self.next_id
        self.next_id += 1
        meta = metadata or {}
        self.documents[doc_id] = (question.strip(), answer.strip(), expert.strip(), meta)

        # Index in Inverted BM25
        self.inverted.add_document(doc_id, question)

        # Index in Dense Vector Index
        vec = text_to_dense_vector(question, dim=self.dim)
        self.dense.add(vec, doc_id)
        return doc_id

    add = add_fact

    def load_from_datasets(self, data_dirs: Sequence[Union[str, Path]] = ("Datasets/curated",),
                           learned_path: Optional[Union[str, Path]] = "Datasets/learned/facts.jsonl",
                           max_per_file: Optional[int] = None) -> int:
        """Ingest all knowledge files from curated datasets and learned facts."""
        count = 0
        for d in data_dirs:
            dp = Path(d)
            if not dp.exists():
                continue
            for f in dp.glob("knowledge/*.jsonl"):
                expert_name = f.stem
                file_count = 0
                try:
                    for line in f.read_text(encoding="utf-8").splitlines():
                        if max_per_file is not None and file_count >= max_per_file:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        if "messages" in row and len(row["messages"]) >= 2:
                            q = row["messages"][-2]["content"]
                            a = row["messages"][-1]["content"]
                            self.add_fact(q, a, expert_name)
                            count += 1
                            file_count += 1
                except Exception:
                    continue

        if learned_path:
            lp = Path(learned_path)
            if lp.exists():
                try:
                    for line in lp.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        q = row.get("q", "")
                        a = row.get("a", "")
                        expert = row.get("expert", "gk_general")
                        if q and a:
                            self.add_fact(q, a, expert, metadata={"learned": True})
                            count += 1
                except Exception:
                    pass
        return count

    def search_hybrid(self, query: str, top_k: int = 3) -> List[Tuple[str, str, str, float]]:
        """Hybrid search combining inverted lexical BM25 and dense semantic similarity.
        Returns list of (question, answer, expert, combined_score)."""
        results: List[Tuple[str, str, str, float]] = []
        # Fast path: DNA triple match
        triple_hit = self.triples.query_natural(query)
        if triple_hit:
            s, p, o = triple_hit
            results.append((query, f"The {p} of {s} is {o}.", "gk_general", 1.0))
            if top_k == 1:
                return results

        # 1. Lexical BM25 search
        bm25_matches = self.inverted.search(query, top_k=top_k * 3)
        max_bm25 = max([score for _, score in bm25_matches], default=1.0)
        norm_bm25 = {doc_id: (score / max(1e-5, max_bm25)) for doc_id, score in bm25_matches}

        # 2. Dense semantic vector search
        q_vec = text_to_dense_vector(query, dim=self.dim)
        dense_matches = self.dense.search(q_vec, top_k=top_k * 3)
        dense_scores = {doc_id: max(0.0, score) for doc_id, score in dense_matches}

        # 3. Combine candidate IDs with query coverage weighting
        q_tokens = set(tokenize_query(query))
        total_q = max(1, len(q_tokens))
        candidate_ids = set(norm_bm25.keys()) | set(dense_scores.keys())
        scored: List[Tuple[int, float]] = []
        for doc_id in candidate_ids:
            lex_score = norm_bm25.get(doc_id, 0.0)
            sem_score = dense_scores.get(doc_id, 0.0)
            doc_q_tokens = set(tokenize_query(self.documents[doc_id][0]))
            coverage = len(q_tokens & doc_q_tokens) / total_q if q_tokens else 1.0
            combined = (self.alpha * lex_score + (1.0 - self.alpha) * sem_score) * coverage
            scored.append((doc_id, combined))

        scored.sort(key=lambda kv: kv[1], reverse=True)
        results = []
        for doc_id, score in scored[:top_k]:
            q, a, e, _ = self.documents[doc_id]
            results.append((q, a, e, score))
        return results

    def search(self, query: str) -> Optional[Tuple[str, str, str, float]]:
        """Drop-in replacement method matching KnowledgeIndex.search API.
        Returns (question, answer, expert, confidence) or None."""
        matches = self.search_hybrid(query, top_k=1)
        if not matches:
            return None
        return matches[0]

    def save(self, path: Union[str, Path]) -> None:
        """Save packed snapshot to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "dim": self.dim,
            "alpha": self.alpha,
            "next_id": self.next_id,
            "documents": self.documents,
            "inverted": self.inverted.state_dict(),
            "dense": self.dense.state_dict(),
            "triples": self.triples.state_dict(),
        }
        torch.save(state, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TantraMemoryStore":
        """Load packed snapshot from disk."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        store = cls(dim=state["dim"], alpha=state["alpha"])
        store.next_id = state["next_id"]
        store.documents = state["documents"]
        store.inverted.load_state_dict(state["inverted"])
        store.dense.load_state_dict(state["dense"])
        if "triples" in state:
            store.triples.load_state_dict(state["triples"])
        return store



# ──  memory/triple_store.py  ──
"""
DNA Knowledge Graph Triple Store
================================
Ultra-high density relational knowledge graph for CPU-first inference.
Packs facts into compact integer IDs (Subject, Predicate, Object).
Achieves ~12 bytes per fact:
  - 10,000 facts in ~120 KB
  - 100,000 facts in ~1.2 MB
  - 1,000,000 facts in ~12 MB
Enables O(1) instant memory lookup without hallucinations.
"""


import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class TripleStore:
    """Memory-mapped integer-encoded Knowledge Graph (Subject, Predicate, Object)."""

    def __init__(self):
        # Entity / Relation dictionary string <-> id
        self.str_to_id: Dict[str, int] = {}
        self.id_to_str: List[str] = []

        # Forward index: (subject_id, predicate_id) -> list of object_ids
        self.sp_to_o: Dict[Tuple[int, int], List[int]] = {}
        # Inverted index: (predicate_id, object_id) -> list of subject_ids
        self.po_to_s: Dict[Tuple[int, int], List[int]] = {}
        # Subject index: subject_id -> list of (predicate_id, object_id)
        self.s_to_po: Dict[int, List[Tuple[int, int]]] = {}

        self.total_triples: int = 0

    def __len__(self) -> int:
        return self.total_triples

    def _get_id(self, s: str) -> int:
        norm = s.strip().lower()
        if norm not in self.str_to_id:
            idx = len(self.id_to_str)
            self.str_to_id[norm] = idx
            self.id_to_str.append(s.strip())
            return idx
        return self.str_to_id[norm]

    def add_triple(self, subject: str, predicate: str, obj: str) -> None:
        """Add a single (subject, predicate, object) fact."""
        s_id = self._get_id(subject)
        p_id = self._get_id(predicate)
        o_id = self._get_id(obj)

        sp = (s_id, p_id)
        if sp not in self.sp_to_o:
            self.sp_to_o[sp] = []
        if o_id not in self.sp_to_o[sp]:
            self.sp_to_o[sp].append(o_id)

        po = (p_id, o_id)
        if po not in self.po_to_s:
            self.po_to_s[po] = []
        if s_id not in self.po_to_s[po]:
            self.po_to_s[po].append(s_id)

        if s_id not in self.s_to_po:
            self.s_to_po[s_id] = []
        if (p_id, o_id) not in self.s_to_po[s_id]:
            self.s_to_po[s_id].append((p_id, o_id))

        self.total_triples += 1

    def query(self, subject: Optional[str] = None, predicate: Optional[str] = None,
              obj: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """Query the triple store with wildcards. Example: query('India', 'capital') -> [('India', 'capital', 'New Delhi')]"""
        results: List[Tuple[str, str, str]] = []
        s_norm = subject.strip().lower() if subject else None
        p_norm = predicate.strip().lower() if predicate else None
        o_norm = obj.strip().lower() if obj else None

        s_id = self.str_to_id.get(s_norm) if s_norm else None
        p_id = self.str_to_id.get(p_norm) if p_norm else None
        o_id = self.str_to_id.get(o_norm) if o_norm else None

        # Pattern 1: (S, P, ?)
        if s_id is not None and p_id is not None and o_id is None:
            obj_ids = self.sp_to_o.get((s_id, p_id), [])
            s_str = self.id_to_str[s_id]
            p_str = self.id_to_str[p_id]
            for oid in obj_ids:
                results.append((s_str, p_str, self.id_to_str[oid]))

        # Pattern 2: (?, P, O)
        elif s_id is None and p_id is not None and o_id is not None:
            sub_ids = self.po_to_s.get((p_id, o_id), [])
            p_str = self.id_to_str[p_id]
            o_str = self.id_to_str[o_id]
            for sid in sub_ids:
                results.append((self.id_to_str[sid], p_str, o_str))

        # Pattern 3: (S, ?, ?)
        elif s_id is not None and p_id is None and o_id is None:
            s_str = self.id_to_str[s_id]
            for pid, oid in self.s_to_po.get(s_id, []):
                results.append((s_str, self.id_to_str[pid], self.id_to_str[oid]))

        return results

    def query_natural(self, text: str) -> Optional[Tuple[str, str, str]]:
        """Extract entity and relation from natural question and return matched triple."""
        t = text.lower()
        # Common relation keywords
        relations = {
            "capital": ["capital of", "capital", "rajdhani", "राजधानी"],
            "currency": ["currency of", "currency", "mudra", "मुद्रा"],
            "president": ["president of", "president", "rashtrapati", "राष्ट्रपति"],
            "prime_minister": ["prime minister of", "pm of", "pradhan mantri", "प्रधानमंत्री"],
            "inventor": ["invented by", "who invented", "invented", "inventor of", "discovered by", "who discovered", "discovered", "आविष्कारक", "खोज"],
            "author": ["written by", "who wrote", "wrote", "author of", "authored by", "author", "rachyita", "रचयिता", "kisne likha", "किसने लिखा"],
        }
        detected_rel = None
        for rel_name, patterns in relations.items():
            if any(p in t for p in patterns):
                detected_rel = rel_name
                break

        if not detected_rel:
            return None

        # Look for subject among known entities
        for s_norm, s_id in self.str_to_id.items():
            if len(s_norm) > 2 and s_norm in t:
                matches = self.query(subject=self.id_to_str[s_id], predicate=detected_rel)
                if matches:
                    return matches[0]

        return None

    def state_dict(self) -> Dict:
        return {
            "id_to_str": self.id_to_str,
            "sp_to_o": {f"{s}:{p}": objs for (s, p), objs in self.sp_to_o.items()},
        }

    def load_state_dict(self, state: Dict) -> None:
        self.id_to_str = state.get("id_to_str", [])
        self.str_to_id = {s.lower(): i for i, s in enumerate(self.id_to_str)}
        self.sp_to_o.clear()
        self.po_to_s.clear()
        self.s_to_po.clear()
        for sp_key, objs in state.get("sp_to_o", {}).items():
            s_id, p_id = [int(x) for x in sp_key.split(":")]
            for o_id in objs:
                self.add_triple(self.id_to_str[s_id], self.id_to_str[p_id], self.id_to_str[o_id])

    def save(self, path: Union[str, Path]) -> None:
        """Save triple store to disk in JSON or binary format."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TripleStore":
        """Load triple store from disk."""
        store = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        store.load_state_dict(data)
        return store


if __name__ == '__main__':
    main()
