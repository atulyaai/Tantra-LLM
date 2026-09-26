"""
Tantra/smriti.py — Smriti (स्मृति), the knowledge store.

A small model can't memorise every fact, so it looks them up instead:
  build   JSONL files -> Model/smriti.db   (python main.py --mode smriti --data a.jsonl,b.jsonl)
  search  question -> the best matching facts (BM25 full-text ranking, Hindi + English)
  context the top facts as text, put in front of the question before the model answers

Storage: SQLite (built into Python). Text is stored zlib-compressed; the full-text
index is "contentless" (it keeps only the word index, not a second copy), so the
store stays a fraction of the raw data size and is searched straight from disk.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import zlib
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from Tantra.utils import get_logger

log = get_logger("tantra.smriti")

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Model", "smriti.db")
_WORD = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)
# Words too common to help a search (Hindi, Hinglish, English).
STOPWORDS = set("""
का की के को में से है हैं था थी थे और या पर भी तो ही यह ये वह वे इस उस इन उन एक कि जो क्या कौन कैसे क्यों कब कहाँ कहां
मैं हम तुम आप मुझे हमें उसे इसके उसके अपने अपनी अपना कर करें करना करते किया होता होती होते हो गया गई दें दीजिए बताइए बताओ बताएं
बारे जानकारी विस्तृत लिए साथ बहुत कुछ सब नहीं ना
kya hai hain ka ki ke ko me mein se aur ya par bhi to hi yeh ye woh kaise kyon kab kaha kahan main hum tum aap mujhe batao bataiye
the a an is are was were be been of to in on for and or with what which who whom how why when where does do did can could
please tell me about explain give i you it this that these those my your
""".split())
FILLER_QUESTIONS = ("इसके बारे में विस्तृत जानकारी दें",)


def _pack(obj: dict) -> bytes:
    return zlib.compress(json.dumps(obj, ensure_ascii=False).encode("utf-8"), 6)


def _unpack(blob: bytes) -> dict:
    return json.loads(zlib.decompress(blob).decode("utf-8"))


def keywords(text: str, limit: int = 12) -> List[str]:
    out: List[str] = []
    for w in _WORD.findall(text.lower()):
        if len(w) > 1 and w not in STOPWORDS and not w.isdigit() and w not in out:
            out.append(w)
    return out[:limit]


def _chunks(text: str, size: int = 700) -> Iterator[str]:
    """Split long text on paragraph / sentence ends into ~size-char pieces."""
    buf = ""
    for part in re.split(r"(?<=[।.!?\n])\s+", text.strip()):
        if len(buf) + len(part) > size and buf:
            yield buf.strip()
            buf = ""
        buf += part + " "
        while len(buf) > size * 2:           # one very long sentence
            yield buf[:size].strip()
            buf = buf[size:]
    if buf.strip():
        yield buf.strip()


def records_from_row(row: dict) -> Iterator[Tuple[str, str, str]]:
    """(kind, question/title, body) records from any supported JSONL row."""
    if "messages" in row:
        msgs = row.get("messages") or []
        for i, m in enumerate(msgs[:-1]):
            if m.get("role") == "user" and msgs[i + 1].get("role") == "assistant":
                yield from _qa(m.get("content") or "", msgs[i + 1].get("content") or "")
        return
    for q, a in (("user", "assistant"), ("instruction", "output"), ("question", "answer"), ("input", "output")):
        if q in row and a in row:
            question = row[q] or ""
            if q == "instruction" and row.get("input"):
                question += "\n" + row["input"]
            yield from _qa(question, row[a] or "")
            return
    text = row.get("text") or ""
    title = row.get("title") or ""
    for piece in _chunks(text):
        yield "text", title, piece


def _qa(q: str, a: str) -> Iterator[Tuple[str, str, str]]:
    q, a = q.strip(), a.strip()
    if not a:
        return
    if any(q.startswith(f) for f in FILLER_QUESTIONS) or len(q) > 400:
        # A generic "tell me about this" + article: store the article as text.
        body = a if len(q) <= 400 else q + "\n" + a
        for piece in _chunks(body):
            yield "text", "", piece
        return
    yield "qa", q, a[:4000]


class Smriti:
    """Compressed, searchable fact store."""

    def __init__(self, path: str = DEFAULT_PATH, readonly: bool = False) -> None:
        self.path = path
        if readonly:
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            uri = "file:" + path.replace("\\", "/") + "?mode=ro"
            self.db = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.db = sqlite3.connect(path, check_same_thread=False)
            self._create()

    def _create(self) -> None:
        self.db.executescript("""
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS docs (id INTEGER PRIMARY KEY, kind TEXT, source TEXT, blob BLOB);
            CREATE VIRTUAL TABLE IF NOT EXISTS idx USING fts5(head, body, content='', tokenize='unicode61 remove_diacritics 0');
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        """)

    # ── build ──
    def add_file(self, path: str, max_rows: Optional[int] = None, batch: int = 5000) -> int:
        source = os.path.basename(path)
        added, rows, pending = 0, 0, []
        seen: set = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                if max_rows and rows >= max_rows:
                    break
                rows += 1
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                for kind, head, body in records_from_row(row):
                    key = hash((head[:200], body[:200]))
                    if key in seen or len(body) < 20:
                        continue
                    seen.add(key)
                    pending.append((kind, head, body))
                if len(pending) >= batch:
                    added += self._insert(pending, source)
                    pending = []
        added += self._insert(pending, source)
        self.db.commit()
        log.info(f"Smriti: {source}: {rows:,} rows -> {added:,} facts")
        return added

    def _insert(self, recs: List[Tuple[str, str, str]], source: str) -> int:
        if not recs:
            return 0
        cur = self.db.cursor()
        for kind, head, body in recs:
            cur.execute("INSERT INTO docs(kind, source, blob) VALUES (?,?,?)",
                        (kind, source, _pack({"head": head, "body": body})))
            cur.execute("INSERT INTO idx(rowid, head, body) VALUES (?,?,?)", (cur.lastrowid, head, body))
        return len(recs)

    def add_text(self, text: str, source: str, title: str = "") -> int:
        """Index one document's text (chunked) under `source`, e.g. an uploaded file."""
        recs = [("text", title, piece) for piece in _chunks(text) if len(piece) >= 20]
        n = self._insert(recs, source)
        self.db.commit()
        return n

    def remove_source(self, source: str) -> int:
        """Remove every fact that came from `source` (contentless index needs the original text)."""
        rows = self.db.execute("SELECT id, blob FROM docs WHERE source = ?", (source,)).fetchall()
        for rid, blob in rows:
            d = _unpack(blob)
            self.db.execute("INSERT INTO idx(idx, rowid, head, body) VALUES ('delete', ?, ?, ?)", (rid, d["head"], d["body"]))
        self.db.execute("DELETE FROM docs WHERE source = ?", (source,))
        self.db.commit()
        return len(rows)

    def finish(self) -> None:
        self.db.execute("INSERT INTO idx(idx) VALUES ('optimize')")
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('built_at', ?)", (str(time.time()),))
        self.db.commit()
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    # ── search ──
    def search(self, query: str, k: int = 3, kinds: Optional[Iterable[str]] = None) -> List[Dict]:
        words = keywords(query)
        if not words:
            return []
        match = " OR ".join('"' + w.replace('"', "") + '"' for w in words)
        sql = "SELECT rowid, bm25(idx, 3.0, 1.0) AS score FROM idx WHERE idx MATCH ? ORDER BY score LIMIT ?"
        try:
            hits = self.db.execute(sql, (match, k * 4)).fetchall()
        except sqlite3.OperationalError:
            return []
        out = []
        qset = set(words)
        for rowid, score in hits:
            row = self.db.execute("SELECT kind, source, blob FROM docs WHERE id = ?", (rowid,)).fetchone()
            if not row or (kinds and row[0] not in kinds):
                continue
            d = _unpack(row[2])
            d["body"] = re.sub(r"^[ऀ-ःऺ-ॏ॑-ॗ\s|।.,:;\-]+", "", d["body"])  # cut-off chunk starts
            head_words = set(keywords(d["head"], 40))
            overlap = len(qset & head_words) / max(len(qset), 1) if d["head"] else 0.0
            out.append({"id": rowid, "kind": row[0], "source": row[1], "question": d["head"], "text": d["body"],
                        "score": round(-score, 3), "match": round(overlap, 2)})
            if len(out) >= k:
                break
        return out

    def best_answer(self, query: str, k: int = 8, min_cover: float = 0.66, allow_text: bool = False) -> Optional[Dict]:
        """The one fact good enough to answer with directly, or None.

        A hit qualifies when its question matches ours, or when it covers most of our key words.
        Translation-exercise rows ("Translate this to Hindi: ...") are skipped: their English
        sentence often shares words with a question without answering it.
        """
        words = keywords(query)
        if len(words) < (1 if allow_text else 2):          # "hello", "kaise ho": too little to look anything up safely
            return None
        best, best_score = None, 0.0
        for h in self.search(query, k):
            if h["question"].lstrip().lower().startswith(("translate this", "translate the", "translate to")):
                continue
            have = set(keywords(f"{h['question']} {h['text']}", 400))
            cover = sum(w in have for w in words) / len(words)
            qw, hw = set(words), set(keywords(h["question"], 40))
            same_question = len(qw & hw) / max(len(qw | hw), 1)      # both ways: most words shared
            # Only a stored question that is nearly the same as ours may answer directly; loose
            # word overlap with an article is too often a different topic (it goes to the model as context instead).
            qa_ok = h["kind"] == "qa" and same_question >= 0.65 and cover >= min_cover
            text_ok = allow_text and h["kind"] == "text" and cover >= 0.75   # your own documents: specific enough
            if not (qa_ok or text_ok) or re.search(r"\[[^\]]{1,20}\]", h["text"][:200]):   # "[naam]" = a template, not an answer
                continue
            score = cover + (h["match"] if h["kind"] == "qa" else 0.0)
            if score > best_score:
                best, best_score = h, score
        return best

    @staticmethod
    def best_sentences(text: str, query: str, n: int = 2) -> str:
        """The n sentences of a passage that share most words with the question, in their original order."""
        sents = [s.strip() for s in re.split(r"(?<=[।.!?])\s+|\n+", text) if len(s.strip()) > 3]
        words = set(keywords(query, 20))
        if len(sents) <= n or not words:
            return text.strip()
        scored = sorted(range(len(sents)), key=lambda i: -len(words & set(keywords(sents[i], 60))))[:n]
        if not words & set(keywords(sents[scored[0]], 60)):
            return text.strip()
        return " ".join(sents[i] for i in sorted(scored) if words & set(keywords(sents[i], 60)))

    def context(self, query: str, k: int = 2, max_chars: int = 700) -> Tuple[str, List[Dict]]:
        """Top facts formatted for the prompt, plus the hits (to show as sources)."""
        hits = self.search(query, k)
        parts, used = [], 0
        for h in hits:
            piece = (f"{h['question']}\n{h['text']}" if h["kind"] == "qa" else h["text"]).strip()
            piece = piece[: max(0, max_chars - used)]
            if piece:
                parts.append(piece)
                used += len(piece)
        return "\n---\n".join(parts), hits

    def stats(self) -> Dict:
        saved = self.db.execute("SELECT value FROM meta WHERE key='stats'").fetchone()
        if saved:   # counted once at build time: a multi-GB store would take ~20 s to count
            st = json.loads(saved[0])
            st["size_mb"] = round(os.path.getsize(self.path) / 2**20, 1) if os.path.isfile(self.path) else 0
            return st
        n = self.db.execute("SELECT count(*) FROM docs").fetchone()[0]
        kinds = dict(self.db.execute("SELECT kind, count(*) FROM docs GROUP BY kind").fetchall())
        sources = dict(self.db.execute("SELECT source, count(*) FROM docs GROUP BY source").fetchall())
        built = self.db.execute("SELECT value FROM meta WHERE key='built_at'").fetchone()
        return {"facts": n, "kinds": kinds, "sources": sources,
                "size_mb": round(os.path.getsize(self.path) / 2**20, 1) if os.path.isfile(self.path) else 0,
                "built_at": float(built[0]) if built else None}

    def save_stats(self) -> None:
        self.db.execute("DELETE FROM meta WHERE key='stats'")
        st = self.stats()
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('stats', ?)", (json.dumps(st),))
        self.db.commit()

    def close(self) -> None:
        self.db.close()


def build(paths: List[str], out: str = DEFAULT_PATH, max_rows: Optional[int] = None) -> Dict:
    """Rebuild the store from scratch (written to a temp file, then swapped in)."""
    tmp = out + ".building"
    for p in (tmp, tmp + "-wal", tmp + "-shm"):
        if os.path.exists(p):
            os.remove(p)
    s = Smriti(tmp)
    t0 = time.time()
    for p in paths:
        s.add_file(p, max_rows=max_rows)
    s.finish()
    s.save_stats()
    s.close()
    os.replace(tmp, out)
    stats = Smriti(out, readonly=True).stats()
    log.info(f"Smriti built in {time.time() - t0:.0f}s: {stats['facts']:,} facts, {stats['size_mb']} MB -> {out}")
    return stats
