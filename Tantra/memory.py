"""
Tantra/memory.py — personal memory, reminders and taught answers (Model/memory.json, this computer only).

  memories   facts you tell Tantra ("याद रखो: मेरी बेटी का जन्मदिन 14 मार्च को है"); the ones that
             match a message are added to the prompt before the model answers
  reminders  "10 मिनट बाद याद दिलाना …" -> due time; the WebUI polls and alerts/speaks when due
  taught     corrections from 👎 "Teach Tantra": used as the answer next time the same question comes
             (and saved to Datasets/feedback.jsonl for the next training run)
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from Tantra.smriti import keywords

CATEGORIES = [
    ("family", r"बेटी|बेटा|माँ|मां|पापा|पिता|पत्नी|पति|भाई|बहन|दादी|दादा|beti|beta|maa|papa|wife|husband|mother|father|"
               r"son|daughter|brother|sister|family|परिवार"),
    ("preferences", r"पसंद|pasand|prefer|like|favourite|favorite|हमेशा|always|never|कभी नहीं"),
    ("about me", r"मेरा नाम|mera naam|my name|मैं |main |i am|i'm|मेरी उम्र|my age|मेरा घर|i live|रहता|रहती"),
]


def categorize(text: str) -> str:
    for name, pat in CATEGORIES:
        if re.search(pat, text, re.I):
            return name
    return "other"


def _norm_q(q: str) -> str:
    return re.sub(r"[\W_]+", " ", q.lower()).strip()


class Memory:
    def __init__(self, path: str) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            d = {}
        for k in ("memories", "reminders", "taught"):
            d.setdefault(k, [])
        d.setdefault("settings", {"file_folders": [], "apps": {}})
        return d

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    # ── memories ──
    def add(self, text: str, category: Optional[str] = None, source: str = "chat") -> Dict[str, Any]:
        text = text.strip()[:500]
        with self.lock:
            for m in self.data["memories"]:
                if _norm_q(m["text"]) == _norm_q(text):
                    return m
            item = {"id": uuid.uuid4().hex[:10], "text": text, "category": category or categorize(text),
                    "source": source, "created": time.time(), "used": 0}
            self.data["memories"].insert(0, item)
            self._save()
        return item

    def update(self, mid: str, text: Optional[str] = None, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self.lock:
            for m in self.data["memories"]:
                if m["id"] == mid:
                    if text:
                        m["text"] = text.strip()[:500]
                    if category:
                        m["category"] = category
                    self._save()
                    return m
        return None

    def delete(self, mid: str) -> bool:
        with self.lock:
            for key in ("memories", "reminders", "taught"):
                self.data[key] = [x for x in self.data[key] if x["id"] != mid]
            self._save()
        return True

    def forget(self, text: str) -> Optional[Dict[str, Any]]:
        words = set(keywords(text, 20))
        best, score = None, 0.0
        for m in self.data["memories"]:
            mw = set(keywords(m["text"], 40))
            s = len(words & mw) / max(len(words), 1)
            if s > score:
                best, score = m, s
        if best and score >= 0.5:
            self.delete(best["id"])
            return best
        return None

    def list(self) -> List[Dict[str, Any]]:
        return list(self.data["memories"])

    def relevant(self, text: str, k: int = 3) -> List[Dict[str, Any]]:
        """Memories sharing words with the message (for the prompt)."""
        words = set(keywords(text, 20))
        if not words:
            return []
        scored = []
        for m in self.data["memories"]:
            overlap = len(words & set(keywords(m["text"], 40)))
            if overlap:
                scored.append((overlap, m))
        scored.sort(key=lambda x: -x[0])
        hits = [m for _, m in scored[:k]]
        if hits:
            with self.lock:
                for m in hits:
                    m["used"] = m.get("used", 0) + 1
                self._save()
        return hits

    # ── reminders ──
    def add_reminder(self, text: str, when: dt.datetime) -> Dict[str, Any]:
        item = {"id": uuid.uuid4().hex[:10], "text": text.strip()[:300], "due": when.timestamp(),
                "created": time.time(), "done": False, "fired": False}
        with self.lock:
            self.data["reminders"].append(item)
            self._save()
        return item

    def reminders(self, include_done: bool = False) -> List[Dict[str, Any]]:
        return sorted((r for r in self.data["reminders"] if include_done or not r["done"]), key=lambda r: r["due"])

    def due(self) -> List[Dict[str, Any]]:
        """Reminders whose time has come and that were not announced yet (marks them announced)."""
        now = time.time()
        out = []
        with self.lock:
            for r in self.data["reminders"]:
                if not r["done"] and not r["fired"] and r["due"] <= now:
                    r["fired"] = True
                    out.append(r)
            if out:
                self._save()
        return out

    def reminder_action(self, rid: str, action: str, minutes: int = 10) -> Optional[Dict[str, Any]]:
        with self.lock:
            for r in self.data["reminders"]:
                if r["id"] == rid:
                    if action == "done":
                        r["done"] = True
                    elif action == "snooze":
                        r["due"] = time.time() + minutes * 60
                        r["fired"] = False
                    self._save()
                    return r
        return None

    # ── taught answers ──
    def teach(self, question: str, answer: str) -> Dict[str, Any]:
        item = {"id": uuid.uuid4().hex[:10], "question": question.strip()[:500], "answer": answer.strip()[:2000],
                "created": time.time()}
        with self.lock:
            self.data["taught"] = [t for t in self.data["taught"] if _norm_q(t["question"]) != _norm_q(question)]
            self.data["taught"].insert(0, item)
            self._save()
        return item

    def taught_answer(self, question: str) -> Optional[Dict[str, Any]]:
        q = _norm_q(question)
        words = set(keywords(question, 20))
        for t in self.data["taught"]:
            if _norm_q(t["question"]) == q:
                return t
            tw = set(keywords(t["question"], 20))
            if words and tw and len(words & tw) / max(len(words | tw), 1) >= 0.8:
                return t
        return None

    # ── settings (file folders, apps) ──
    @property
    def settings(self) -> Dict[str, Any]:
        return self.data["settings"]

    def set_settings(self, **kw) -> Dict[str, Any]:
        with self.lock:
            self.data["settings"].update({k: v for k, v in kw.items() if v is not None})
            self._save()
        return self.data["settings"]
