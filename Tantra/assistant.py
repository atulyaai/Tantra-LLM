"""
Tantra/assistant.py — the assistant's settings, in one editable file (Model/assistant.json).

Nothing about how Tantra talks or behaves is fixed in code: name, small talk, suggestions,
skill examples, apps it may open, wake words, voice timing and the reply used while no model
is trained all live here. The WebUI Settings page edits it; missing keys fall back to DEFAULTS,
so new versions add settings without breaking an old file.
"""
from __future__ import annotations

import copy
import json
import os
import random
import re
import threading
from typing import Any, Dict, Optional

DEFAULTS: Dict[str, Any] = {
    "name": "Tantra",
    "name_hi": "तन्त्र",
    "tagline": "Hindi-first AI",
    "language": "auto",                 # auto | hi | en : reply language for built-in answers
    "suggestions": ["नमस्ते, आप कौन हैं?", "250 × 18 + 5%", "आज कौन सा दिन है?",
                    "10 मिनट बाद याद दिलाना चाय बनानी है", "5 lakh in million", "भारत की राजधानी क्या है?"],
    "skill_examples": [["Calculator", "250 × 18 + 5%"], ["समय / तारीख", "आज कौन सा दिन है?"],
                       ["Units", "5 lakh in million"], ["Remember", "याद रखो: "],
                       ["Reminder", "10 मिनट बाद याद दिलाना "], ["Daily brief", "आज का brief"],
                       ["System status", "training कैसी चल रही है?"], ["Find files", "files: "]],
    "small_talk": [
        {"match": ["hi", "hello", "hey", "hii", "helo", "नमस्ते", "नमस्कार", "namaste", "namaskar", "हाय", "हेलो"],
         "hi": ["नमस्ते! मैं {name_hi} हूँ। बताइए, क्या मदद करूँ?"], "en": ["Hello! I'm {name}. How can I help?"]},
        {"match": ["kaise ho", "kaise hain", "कैसे हो", "कैसे हैं", "how are you", "kya haal", "क्या हाल"],
         "hi": ["मैं ठीक हूँ, धन्यवाद! आप कैसे हैं?"], "en": ["I'm doing well, thanks! How are you?"]},
        {"match": ["thank", "thanks", "धन्यवाद", "शुक्रिया", "dhanyavad", "shukriya", "thank you"],
         "hi": ["आपका स्वागत है! 🙏"], "en": ["You're welcome!"]},
        {"match": ["who are you", "tum kaun ho", "aap kaun ho", "तुम कौन हो", "आप कौन हैं", "आप कौन हो", "your name",
                   "tumhara naam", "तुम्हारा नाम", "आपका नाम"],
         "hi": ["मैं {name_hi} ({name}) हूँ — आपका हिंदी-प्रथम AI सहायक। मैं हिसाब, याद रखना, reminders और सवालों के जवाब में मदद करता हूँ।"],
         "en": ["I'm {name}, your Hindi-first AI assistant. I can calculate, remember things, set reminders and answer questions."]},
        {"match": ["bye", "goodbye", "alvida", "अलविदा", "फिर मिलेंगे", "phir milenge", "good night", "शुभ रात्रि"],
         "hi": ["फिर मिलेंगे! 🙏"], "en": ["Goodbye! Talk soon."]},
        {"match": ["good morning", "सुप्रभात", "शुभ प्रभात", "suprabhat"],
         "hi": ["सुप्रभात! आज का brief सुनना चाहेंगे? बस कहिए 'आज का brief'।"], "en": ["Good morning! Say 'daily brief' to hear your day."]},
    ],
    "no_model_reply": {
        "hi": "मेरा अपना मॉडल अभी training में है, इसलिए इस सवाल का खुद से जवाब अभी नहीं दे सकता। मैं अभी ये कर सकता हूँ: हिसाब (250 × 18), समय/तारीख, units, 'याद रखो …', reminders, और आपके documents या Smriti से जवाब।",
        "en": "My own model is still training, so I can't answer that freely yet. Right now I can: calculate (250 × 18), time/date, units, 'remember …', reminders, and answer from your documents or Smriti.",
    },
    "early_model_note": True,           # show the "still early in training" banner
    "apps": {"notepad": ["notepad.exe"], "calculator": ["calc.exe"], "calc": ["calc.exe"], "paint": ["mspaint.exe"],
             "explorer": ["explorer.exe"], "file explorer": ["explorer.exe"], "नोटपैड": ["notepad.exe"],
             "कैलकुलेटर": ["calc.exe"]},
    "wake_words": ["तन्त्र", "तंत्र", "tantra", "tantr", "tanthra"],
    "stop_words": ["रुको", "रुक जाओ", "stop", "ruko", "बस"],
    "voice": {"silence_ms": 1100, "max_seconds": 15, "min_threshold": 4, "whisper_model": "base", "language": ""},
    "knowledge": {"min_same_question": 0.65, "answer_from_documents": True},
    "auto_repair": True,                # install missing Python packages automatically at start
}


def _merge(base: Any, over: Any) -> Any:
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            out[k] = _merge(base.get(k), v) if k in base else v
        return out
    return copy.deepcopy(over if over is not None else base)


class Settings:
    def __init__(self, path: str) -> None:
        self.path = path
        self.lock = threading.Lock()
        self._mtime = -1.0
        self._data: Dict[str, Any] = copy.deepcopy(DEFAULTS)

    def get(self) -> Dict[str, Any]:
        """Current settings; re-read automatically when the file is edited by hand."""
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            return self._data
        if mtime != self._mtime:
            try:
                with open(self.path, encoding="utf-8") as f:
                    self._data = _merge(DEFAULTS, json.load(f))
                self._mtime = mtime
            except (OSError, ValueError):
                pass   # a broken edit keeps the last good settings
        return self._data

    def update(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            data = _merge(self.get(), {k: v for k, v in changes.items() if k in DEFAULTS})
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)
            self._data, self._mtime = data, os.path.getmtime(self.path)
        return data

    def reset(self, key: Optional[str] = None) -> Dict[str, Any]:
        data = copy.deepcopy(self.get())
        if key:
            data[key] = copy.deepcopy(DEFAULTS[key])
        else:
            data = copy.deepcopy(DEFAULTS)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        self._mtime = -1
        return self.get()


def fill(text: str, cfg: Dict[str, Any]) -> str:
    return text.replace("{name}", cfg.get("name", "Tantra")).replace("{name_hi}", cfg.get("name_hi", "तन्त्र"))


def small_talk(text: str, cfg: Dict[str, Any], hindi: bool) -> Optional[str]:
    """A reply from the small-talk list when the whole message is a greeting / thanks / etc."""
    t = re.sub(r"[^\wऀ-ॿ ]+", " ", text.lower()).strip()
    t = re.sub(r"\s+", " ", t)
    if not t or len(t) > 60:
        return None
    for item in cfg.get("small_talk", []):
        for m in item.get("match", []):
            m = m.lower()
            if t == m or t.startswith(m + " ") or t.endswith(" " + m) or (len(m) > 5 and m in t):
                options = item.get("hi" if hindi else "en") or item.get("en") or item.get("hi") or []
                if options:
                    return fill(random.choice(options), cfg)
    return None
