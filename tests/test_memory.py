import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from Tantra.memory.memory_store import TantraMemoryStore
from Tantra.memory.inverted_index import InvertedBM25Index, tokenize_query
from Tantra.memory.dense_index import DenseVectorIndex
import torch


def test_tokenize_query():
    toks = tokenize_query("What is the capital of India?")
    assert "capital" in toks
    assert "india" in toks
    assert "what" not in toks
    assert "the" not in toks


def test_inverted_bm25_search():
    index = InvertedBM25Index()
    index.add_document(0, "The capital of France is Paris.")
    index.add_document(1, "The capital of Japan is Tokyo.")
    index.add_document(2, "Quantum mechanics describes atomic physics.")

    results = index.search("capital France", top_k=2)
    assert len(results) > 0
    assert results[0][0] == 0  # doc 0 matches best


def test_dense_vector_index():
    idx = DenseVectorIndex(dim=4)
    v0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    v1 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    idx.add(v0, 100)
    idx.add(v1, 200)

    q = torch.tensor([0.9, 0.1, 0.0, 0.0])
    res = idx.search(q, top_k=1)
    assert res[0][0] == 100


def test_memory_store_hybrid_and_save_load(tmp_path):
    store = TantraMemoryStore(dim=64, alpha=0.6)
    store.add_fact("What is the capital of India?", "New Delhi is the capital of India.", "gk_geography")
    store.add_fact("Who developed theory of relativity?", "Albert Einstein developed it.", "gk_science")
    store.add_fact("भारत की राजधानी क्या है?", "भारत की राजधानी नई दिल्ली है।", "gk_geography")

    # Exact search
    res = store.search("capital of India")
    assert res is not None
    q, a, expert, score = res
    assert "New Delhi" in a
    assert expert == "gk_geography"

    # Hindi Unicode search
    res_hi = store.search("भारत की राजधानी")
    assert res_hi is not None
    assert "नई दिल्ली" in res_hi[1]

    # Save and load roundtrip
    p = tmp_path / "memory.pt"
    store.save(p)
    loaded = TantraMemoryStore.load(p)
    assert len(loaded) == 3
    res_loaded = loaded.search("relativity Einstein")
    assert res_loaded is not None
    assert "Einstein" in res_loaded[1]


def test_memory_store_dataset_ingest():
    store = TantraMemoryStore()
    n = store.load_from_datasets(["Datasets/curated"], learned_path="Datasets/learned/facts.jsonl", max_per_file=50)
    assert n > 0
    res = store.search("capital")
    assert res is not None


def test_triple_store_dna_and_indic(tmp_path):
    from Tantra.memory import TripleStore

    store = TripleStore()
    store.add_triple("India", "capital", "New Delhi")
    store.add_triple("India", "currency", "Rupee")
    store.add_triple("Sanskrit", "author", "Panini")
    store.add_triple("भारत", "capital", "नई दिल्ली")

    # Relational queries
    res = store.query(subject="India", predicate="capital")
    assert len(res) == 1 and res[0][2] == "New Delhi"

    # Natural language parsing
    hit = store.query_natural("What is the capital of India?")
    assert hit is not None and hit[2] == "New Delhi"

    # Indic Sanskrit query
    hit_sanskrit = store.query_natural("Who is the author of Sanskrit grammar?")
    assert hit_sanskrit is not None and hit_sanskrit[2] == "Panini"

    # Save and load roundtrip
    p = tmp_path / "triples.json"
    store.save(p)
    loaded = TripleStore.load(p)
    assert len(loaded) == 4
    assert loaded.query("India", "currency")[0][2] == "Rupee"

