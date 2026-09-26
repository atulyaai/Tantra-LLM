"""
Tantra/documents.py — your own documents, searchable by chat (Model/docs.db).

Upload a file in the WebUI (or call add_file); its text is split into pieces and put in a
separate Smriti store, so chat can answer from it and show which document it used.
Formats: .txt .md .csv .json .jsonl .html/.htm .docx (built in), .pdf (needs: pip install pypdf).
"""
from __future__ import annotations

import html
import io
import json
import os
import re
import time
import zipfile
from typing import Dict, List

from Tantra.smriti import Smriti

TEXT_EXT = {".txt", ".md", ".csv", ".json", ".jsonl", ".log", ".py", ".html", ".htm"}
SUPPORTED = TEXT_EXT | {".docx", ".pdf"}
MAX_BYTES = 25 * 2**20


def extract_text(name: str, data: bytes) -> str:
    ext = os.path.splitext(name.lower())[1]
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type {ext}. Use: {', '.join(sorted(SUPPORTED))}")
    if len(data) > MAX_BYTES:
        raise ValueError("File is larger than 25 MB.")
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ValueError("Reading PDF needs one package: pip install pypdf")
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    if ext == ".docx":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
        xml = re.sub(r"</w:p>", "\n", xml)
        return html.unescape(re.sub(r"<[^>]+>", "", xml))
    text = data.decode("utf-8", errors="replace")
    if ext in (".html", ".htm"):
        text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    elif ext == ".jsonl":
        rows = []
        for line in text.splitlines():
            try:
                o = json.loads(line)
                rows.append(" ".join(str(v) for v in (o.values() if isinstance(o, dict) else [o])))
            except ValueError:
                continue
        text = "\n".join(rows)
    return re.sub(r"[ \t]+", " ", text).strip()


class Documents:
    def __init__(self, path: str) -> None:
        self.path = path
        self.index_path = os.path.splitext(path)[0] + "_index.json"

    def _store(self) -> Smriti:
        return Smriti(self.path)

    def list(self) -> List[Dict]:
        try:
            with open(self.index_path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return []

    def _save_list(self, items: List[Dict]) -> None:
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=1)

    def add(self, name: str, data: bytes) -> Dict:
        name = os.path.basename(name)[:120] or "document"
        text = extract_text(name, data)
        if len(text) < 20:
            raise ValueError("No readable text found in this file.")
        s = self._store()
        try:
            s.remove_source(name)                 # re-upload replaces
            pieces = s.add_text(text, source=name, title=name)
        finally:
            s.close()
        items = [d for d in self.list() if d["name"] != name]
        item = {"name": name, "chars": len(text), "pieces": pieces, "added": time.time()}
        self._save_list([item] + items)
        return item

    def remove(self, name: str) -> bool:
        s = self._store()
        try:
            s.remove_source(name)
        finally:
            s.close()
        self._save_list([d for d in self.list() if d["name"] != name])
        return True

    def search(self, query: str, k: int = 2) -> List[Dict]:
        if not os.path.isfile(self.path) or not self.list():
            return []
        s = Smriti(self.path, readonly=True)
        try:
            return s.search(query, k)
        finally:
            s.close()
