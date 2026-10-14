"""
Massive Knowledge Ingestion into Tantra High-Density Memory
===========================================================
Packs tens of thousands of facts across 24 domains (Physics, Medicine, History,
Geography, Law, Economics, Indic Heritage, Technology, Biology, Philosophy)
into TantraMemoryStore with DNA TripleStore relations.
"""

import json
import os
import sys
import time
from pathlib import Path

# Configure utf-8 stdout
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from Tantra.memory import TantraMemoryStore, TripleStore


def build_memory(max_per_domain: int = 500):
    store = TantraMemoryStore(dim=64, alpha=0.6)
    print("=" * 60)
    print("INGESTING MASSIVE KNOWLEDGE INTO HIGH-DENSITY RAM STORE")
    print("=" * 60)

    # 1. Ingest Curated Knowledge (Geography, Science, Math, Indic)
    print("\n1. Ingesting Curated Knowledge...")
    n_curated = store.load_from_datasets(
        ["Datasets/curated"],
        learned_path="Datasets/learned/facts.jsonl",
        max_per_file=max_per_domain,
    )
    print(f"  · Ingested {n_curated} curated facts")

    # 2. Ingest 24 Expert Knowledge Domains
    print("\n2. Ingesting 24 Encyclopedic Knowledge Domains...")
    n_domain = 0
    for p in sorted(Path("Datasets/experts/knowledge").glob("*.jsonl")):
        domain = p.stem.replace("gk_", "")
        count = 0
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                if count >= max_per_domain:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "text" in row:
                    text = row["text"]
                    # Format as Q/A pair for retrieval
                    parts = text.split(":", 1)
                    if len(parts) == 2:
                        subject, fact = parts[0].strip(), parts[1].strip()
                        q = f"Tell me about {subject} in {domain}"
                        store.add_fact(q, fact, expert=f"gk_{domain}", metadata={"domain": domain})
                        store.add_fact(fact, fact, expert=f"gk_{domain}", metadata={"domain": domain})
                        count += 2
                elif "messages" in row and len(row["messages"]) >= 2:
                    q = row["messages"][-2]["content"]
                    a = row["messages"][-1]["content"]
                    store.add_fact(q, a, expert=f"gk_{domain}", metadata={"domain": domain})
                    count += 1
            n_domain += count
            print(f"  · [{domain:14s}] Ingested {count:4d} entries")
        except Exception as e:
            print(f"  · Error reading {p.name}: {e}")

    # 3. Add Core DNA Knowledge Triples
    print("\n3. Adding Core DNA Relational Triples...")
    triples_data = [
        # Indic & Sanskrit
        ("India", "capital", "New Delhi"),
        ("India", "currency", "Indian Rupee"),
        ("India", "national_animal", "Royal Bengal Tiger"),
        ("India", "national_bird", "Indian Peacock"),
        ("Sanskrit", "author", "Panini"),
        ("Ashtadhyayi", "author", "Panini"),
        ("Ashtadhyayi", "total_sutras", "3959"),
        ("ISRO", "headquarters", "Bengaluru"),
        ("Chandrayaan-3", "mission", "Lunar South Pole Soft Landing"),
        ("Maharashtra", "capital", "Mumbai"),
        ("Karnataka", "capital", "Bengaluru"),
        ("Tamil Nadu", "capital", "Chennai"),
        ("Uttar Pradesh", "capital", "Lucknow"),
        # Global Geography
        ("France", "capital", "Paris"),
        ("Japan", "capital", "Tokyo"),
        ("Germany", "capital", "Berlin"),
        ("Australia", "capital", "Canberra"),
        ("United Kingdom", "capital", "London"),
        ("United States", "capital", "Washington D.C."),
        # Science & Physics
        ("Albert Einstein", "inventor", "Theory of General Relativity"),
        ("Isaac Newton", "inventor", "Classical Laws of Motion and Gravitation"),
        ("C. V. Raman", "inventor", "Raman Scattering Effect"),
        ("Light", "speed", "299,792 kilometers per second"),
        ("Water", "chemical_formula", "H2O"),
    ]
    for s, p, o in triples_data:
        store.triples.add_triple(s, p, o)
    print(f"  · Added {len(triples_data)} core relational triples")

    total_facts = len(store.documents) + len(store.triples)
    print("\n" + "=" * 60)
    print(f"TOTAL MEMORY CAPACITY: {total_facts:,} facts")
    print("=" * 60)

    # 4. Save Packed Snapshot to Model_small/memory.pt
    out_file = Path("Model_small/memory.pt")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    store.save(out_file)
    save_time = time.perf_counter() - t0
    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"Saved packed memory snapshot -> {out_file}")
    print(f"Snapshot File Size: {file_size_mb:.2f} MB (Save Time: {save_time:.3f}s)")

    # 5. Test Sub-Millisecond Retrieval Speed
    print("\nTesting Query Retrieval Latency on CPU:")
    queries = [
        "What is the capital of India?",
        "Who is considered the father of Sanskrit grammar?",
        "Tell me about Technology in programming",
        "What is the capital of France?",
    ]
    for q in queries:
        t0 = time.perf_counter()
        hit = store.search(q)
        dt_ms = (time.perf_counter() - t0) * 1000
        ans_preview = hit[1][:60] if hit else "None"
        print(f"  · [{dt_ms:5.2f} ms] Q: '{q}' -> {ans_preview}...")


if __name__ == "__main__":
    build_memory(max_per_domain=300)
