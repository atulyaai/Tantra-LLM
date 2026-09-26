"""
Tantra/skills.py — instant answers that never need the model (and are always right).

  handle(text, ctx) -> Skill result or None
     calculator   "250 × 18 + 5%", "3 * 120 + 2 * 180", "15% of 2400"   (with steps: maths tutor)
     time/date    "कितने बजे हैं", "aaj kaun sa din hai", "what's the date"
     units        "5 lakh in million", "10 km to miles", "98.6 F in C"
     memory       "याद रखो: ...", "remember that ...", "भूल जाओ ...", "tumhe kya yaad hai"
     reminders    "10 मिनट बाद याद दिलाना चाय", "remind me in 2 hours to call", "timer 5 min"
     brief        "आज का brief", "good morning", "daily brief"
     status       "training कैसी चल रही है", "system status"
     files        "files: invoice"            (searches folders you allowed)
     open         "open notepad", "खोलो calculator"   (asks you to confirm in the WebUI)
Everything is offline. `ctx` gives the skills what they need (memory store, status readers).
"""
from __future__ import annotations

import ast
import datetime as dt
import math
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

DEV = re.compile(r"[ऀ-ॿ]")
HI_DAYS = ["सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"]
HI_MONTHS = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितम्बर", "अक्टूबर", "नवम्बर", "दिसम्बर"]


@dataclass
class Skill:
    name: str                 # calculator, time, units, memory, reminder, brief, status, files, open, taught, smriti
    text: str                 # the answer, ready to show / speak
    card: Dict[str, Any] = field(default_factory=dict)   # structured extras for the UI


_HINGLISH = re.compile(r"\b(kya|hai|hain|kaise|kitna|kitne|mujhe|tumhe|yaad|kaun|kab|kahan|bhai|yaar|aaj|kal|baje|nahi|namaste|namaskar|dhanyavad|shukriya|theek|accha|haan)\b", re.I)


def hindi(text: str) -> bool:
    """Reply in Hindi when the user wrote Devanagari or Hinglish."""
    return bool(DEV.search(text) or _HINGLISH.search(text))


def indian(n: float, digits: int = 4) -> str:
    """1234567.5 -> 12,34,567.5 (Indian grouping)."""
    if isinstance(n, float) and (math.isinf(n) or math.isnan(n)):
        return str(n)
    neg = n < 0
    n = abs(n)
    s = f"{n:.{digits}f}".rstrip("0").rstrip(".") if isinstance(n, float) and n != int(n) else str(int(round(n)))
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        whole = head + "," + tail
    return ("-" if neg else "") + whole + ("." + frac if frac else "")


# ── calculator (safe: only numbers and arithmetic, no names or calls) ───────

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
_SYM = {ast.Add: "+", ast.Sub: "−", ast.Mult: "×", ast.Div: "÷", ast.Pow: "^", ast.Mod: "mod", ast.FloorDiv: "//"}
_WORDS = [(r"\b(plus|jama|jod|jodo)\b|जमा|जोड़|प्लस", "+"), (r"\b(minus|ghata|ghatao)\b|घटा|माइनस", "-"),
          (r"\b(times|into|multiply|multiplied by|guna)\b|गुणा", "*"), (r"\b(divided by|divide|bhag)\b|भाग", "/"),
          (r"[×xX✕](?=\s*[\d(])", "*"), (r"[÷]", "/"), (r"\^", "**"), (r"[−–]", "-")]


def _to_expr(text: str) -> Optional[str]:
    t = text.lower().replace(",", "")
    t = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*(?:of|ka|का|की)\s*(\d+(?:\.\d+)?)", r"(\1/100*\2)", t)
    t = re.sub(r"(\d)\s*[xX×✕]\s*(?=\d)", r"\1*", t)
    for pat, rep in _WORDS:
        t = re.sub(pat, rep, t)
    # "<anything> + 5%"  -> (anything) + (anything)*5/100   (GST / discount style)
    pm = re.match(r"^(.*?\d.*?)\s*([+\-])\s*(\d+(?:\.\d+)?)\s*%\s*\??\s*$", t.strip())
    if pm:
        base = pm.group(1).strip()
        t = f"({base}){pm.group(2)}({base})*{pm.group(3)}/100"
    m = re.findall(r"[\d.\s+\-*/()%]+", t)
    cand = max((s.strip() for s in m), key=len, default="")
    if not re.search(r"\d", cand) or not re.search(r"\d\s*[+\-*/%]+\s*[\d(]|\)\s*[+\-*/]", cand):
        return None
    if len(re.findall(r"\d+(?:\.\d+)?", cand)) < 2:
        return None
    return cand


def _eval(node, steps: List[str]):
    if isinstance(node, ast.Expression):
        return _eval(node.body, steps)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand, steps)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _eval(node.left, steps), _eval(node.right, steps)
        if isinstance(node.op, ast.Pow) and abs(b) > 100:
            raise ValueError("power too large")
        r = _OPS[type(node.op)](a, b)
        steps.append(f"{indian(a)} {_SYM[type(node.op)]} {indian(b)} = {indian(r)}")
        return r
    raise ValueError("not arithmetic")


def calculator(text: str) -> Optional[Skill]:
    expr = _to_expr(text)
    if not expr:
        return None
    try:
        steps: List[str] = []
        value = _eval(ast.parse(expr, mode="eval"), steps)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    shown = expr.replace("**", "^").replace("*", " × ").replace("/", " ÷ ")
    ans = indian(value)
    word = "उत्तर" if hindi(text) else "Answer"
    return Skill("calculator", f"{word}: {ans}", {"expression": shown, "result": ans, "steps": steps})


# ── time & date ──────────────────────────────────────────────────────────────

_TIME = re.compile(r"(कितने बजे|समय क्या|टाइम|\btime\b|kitne baje|samay|what time)", re.I)
_DATE = re.compile(r"(तारीख|दिनांक|कौन सा दिन|कौनसा दिन|आज क्या दिन|\bdate\b|what day|which day|kaun sa din|tareekh|tarikh|aaj kya din)", re.I)


def time_date(text: str, now: Optional[dt.datetime] = None) -> Optional[Skill]:
    wants_time, wants_date = bool(_TIME.search(text)), bool(_DATE.search(text))
    if not (wants_time or wants_date) or re.search(r"याद|remind|timer|बाद|baad", text, re.I):
        return None
    now = now or dt.datetime.now()
    hi = hindi(text) or re.search(r"\b(kitne|baje|samay|aaj|kaun|din|tareekh|tarikh)\b", text, re.I)
    if hi:
        date = f"{HI_DAYS[now.weekday()]}, {now.day} {HI_MONTHS[now.month - 1]} {now.year}"
        period = "सुबह" if 4 <= now.hour < 12 else "दोपहर" if now.hour < 16 else "शाम" if now.hour < 20 else "रात"
        clock = now.strftime("%I:%M").lstrip("0")
        text_out = (f"अभी {period} के {clock} बजे हैं। " if wants_time else "") + (f"आज {date} है।" if wants_date or not wants_time else "")
    else:
        date = now.strftime("%A, %d %B %Y")
        clock = now.strftime("%I:%M %p").lstrip("0")
        text_out = (f"It's {clock}. " if wants_time else "") + (f"Today is {date}." if wants_date or not wants_time else "")
    return Skill("time", text_out.strip(), {"iso": now.isoformat(timespec="minutes")})


# ── units ────────────────────────────────────────────────────────────────────

_UNITS = {  # name -> (kind, factor to base)
    "km": ("len", 1000), "kilometer": ("len", 1000), "kilometre": ("len", 1000), "किलोमीटर": ("len", 1000),
    "m": ("len", 1), "meter": ("len", 1), "metre": ("len", 1), "मीटर": ("len", 1),
    "cm": ("len", .01), "mm": ("len", .001), "mile": ("len", 1609.344), "miles": ("len", 1609.344), "मील": ("len", 1609.344),
    "ft": ("len", .3048), "feet": ("len", .3048), "foot": ("len", .3048), "फुट": ("len", .3048),
    "inch": ("len", .0254), "inches": ("len", .0254), "इंच": ("len", .0254),
    "kg": ("mass", 1), "kilo": ("mass", 1), "किलो": ("mass", 1), "g": ("mass", .001), "gram": ("mass", .001),
    "ग्राम": ("mass", .001), "lb": ("mass", .45359237), "lbs": ("mass", .45359237), "pound": ("mass", .45359237),
    "quintal": ("mass", 100), "क्विंटल": ("mass", 100), "ton": ("mass", 1000), "tonne": ("mass", 1000),
    "l": ("vol", 1), "liter": ("vol", 1), "litre": ("vol", 1), "लीटर": ("vol", 1), "ml": ("vol", .001),
    "gallon": ("vol", 3.785411784),
    "thousand": ("num", 1e3), "hazar": ("num", 1e3), "हज़ार": ("num", 1e3), "हजार": ("num", 1e3),
    "lakh": ("num", 1e5), "lac": ("num", 1e5), "लाख": ("num", 1e5), "million": ("num", 1e6), "मिलियन": ("num", 1e6),
    "crore": ("num", 1e7), "करोड़": ("num", 1e7), "billion": ("num", 1e9), "अरब": ("num", 1e9), "arab": ("num", 1e9),
    "c": ("temp", "c"), "celsius": ("temp", "c"), "°c": ("temp", "c"), "सेल्सियस": ("temp", "c"),
    "f": ("temp", "f"), "fahrenheit": ("temp", "f"), "°f": ("temp", "f"), "फ़ारेनहाइट": ("temp", "f"),
}
_UNIT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*([^\d\s]+)\s+(?:in|to|into|me|mein|में|को|=)\s+([^\d\s?।.]+)", re.I)


def units(text: str) -> Optional[Skill]:
    m = _UNIT_RE.search(text.replace(",", ""))
    if not m:
        return None
    value, a, b = float(m.group(1)), m.group(2).lower().strip("?."), m.group(3).lower().strip("?.")
    if a not in _UNITS or b not in _UNITS or _UNITS[a][0] != _UNITS[b][0]:
        return None
    kind = _UNITS[a][0]
    if kind == "temp":
        ca, cb = _UNITS[a][1], _UNITS[b][1]
        c = value if ca == "c" else (value - 32) * 5 / 9
        out = c if cb == "c" else c * 9 / 5 + 32
    else:
        out = value * _UNITS[a][1] / _UNITS[b][1]
    out_s = indian(round(out, 4))
    return Skill("units", f"{indian(value)} {m.group(2)} = {out_s} {m.group(3)}",
                 {"from": f"{indian(value)} {m.group(2)}", "to": f"{out_s} {m.group(3)}"})


# ── memory / reminders / brief / status / files / open (need ctx) ────────────

_REMEMBER = re.compile(r"^\s*(?:याद रखो|याद रखना|yaad rakho|yaad rakhna|remember(?: that)?)\s*[:,\-–]?\s*(.+)$", re.I | re.S)
_FORGET = re.compile(r"^\s*(?:भूल जाओ|bhool jao|bhul jao|forget(?: that| about)?)\s*[:,\-–]?\s*(.+)$", re.I | re.S)
_RECALL = re.compile(r"(क्या याद है|tumhe kya yaad|what do you remember|meri memory|मेरी यादें|memories dikhao)", re.I)
_IN = re.compile(r"(\d+)\s*(सेकंड|second|sec|मिनट|minute|min|घंटे|घंटा|ghante|ghanta|hour|hr|दिन|din|day)s?\s*(?:बाद|baad|में|mein|later)?", re.I)
_AT = re.compile(r"(?:at|बजे|baje)?\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm|बजे|baje)", re.I)
_REMIND = re.compile(r"(याद दिला|yaad dila|remind me|reminder|timer|alarm)", re.I)
_BRIEF = re.compile(r"(daily brief|आज का brief|today'?s brief|good morning|सुप्रभात|शुभ प्रभात|आज क्या है मेरे लिए)", re.I)
_STATUS = re.compile(r"(training (?:status|कैसी|kaisi|kaise)|system status|सिस्टम|how is training|model status)", re.I)
_FILES = re.compile(r"^\s*(?:files?|फ़ाइल|फाइल)\s*[:\-]\s*(.+)$", re.I)
_OPEN = re.compile(r"^\s*(?:open|खोलो|kholo|start|चालू करो)\s+(.+?)\s*$|^\s*(.+?)\s+(?:खोलो|kholo)\s*$", re.I)


def _when(text: str, now: dt.datetime) -> Optional[dt.datetime]:
    m = _IN.search(text)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        secs = {"s": 1, "से": 1, "मि": 60, "mi": 60, "घं": 3600, "gh": 3600, "ho": 3600, "hr": 3600, "दि": 86400, "di": 86400, "da": 86400}
        mult = next((v for k, v in secs.items() if unit.startswith(k)), 60)
        return now + dt.timedelta(seconds=n * mult)
    m = _AT.search(text)
    if m:
        h, mnt = int(m.group(1)), int(m.group(2) or 0)
        ampm = m.group(3).lower()
        if ampm == "pm" and h < 12:
            h += 12
        if re.search(r"शाम|रात|shaam|raat|evening|night", text, re.I) and h < 12:
            h += 12
        if h > 23 or mnt > 59:
            return None
        day = now + dt.timedelta(days=1) if re.search(r"\bkal\b|कल|tomorrow", text, re.I) else now
        t = day.replace(hour=h, minute=mnt, second=0, microsecond=0)
        return t if t > now else t + dt.timedelta(days=1)
    return None


def _task(text: str) -> str:
    t = _IN.sub("", text)
    t = re.sub(r"(मुझे|mujhe|please|कृपया|remind me|याद दिलाना|याद दिलाओ|yaad dilana|yaad dilao|reminder|set a|set|timer|alarm|"
               r"\bto\b|\bin\b|कि|ki|बाद|baad|\bat\b|kal|कल|tomorrow|subah|सुबह|शाम|shaam|raat|रात|\d{1,2}(:\d{2})?\s*(am|pm|बजे|baje))",
               " ", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip(" ,.।:-") or ("Timer" if not hindi(text) else "टाइमर")


def assistant_skills(text: str, ctx: Dict[str, Any]) -> Optional[Skill]:
    hi = hindi(text)
    mem = ctx.get("memory")
    now = ctx.get("now") or dt.datetime.now()

    if mem is not None:
        m = _REMEMBER.match(text)
        if m:
            item = mem.add(m.group(1).strip())
            return Skill("memory", ("याद रख लिया: " if hi else "Saved to memory: ") + item["text"], {"memory": item})
        m = _FORGET.match(text)
        if m:
            gone = mem.forget(m.group(1).strip())
            if gone:
                return Skill("memory", ("भूल गया: " if hi else "Forgot: ") + gone["text"], {"forgot": gone})
            return Skill("memory", "ऐसी कोई याद नहीं मिली।" if hi else "I couldn't find that memory.")
        if _RECALL.search(text):
            items = mem.list()[:10]
            if not items:
                return Skill("memory", "अभी कुछ याद नहीं है। 'याद रखो: …' कहकर बताइए।" if hi else "Nothing saved yet. Say 'remember that …'.")
            lines = "\n".join(f"- {i['text']}" for i in items)
            return Skill("memory", ("मुझे ये याद है:\n" if hi else "Here's what I remember:\n") + lines, {"memories": items})
        if _REMIND.search(text):
            when = _when(text, now)
            if when is None:
                return Skill("reminder", "कब याद दिलाऊँ? जैसे '10 मिनट बाद' या 'कल सुबह 9 बजे'।" if hi
                             else "When should I remind you? e.g. 'in 10 minutes' or 'tomorrow at 9 am'.")
            r = mem.add_reminder(_task(text), when)
            nice = when.strftime("%d %b, %I:%M %p")
            return Skill("reminder", (f"ठीक है, {nice} पर याद दिलाऊँगा: {r['text']}" if hi
                                      else f"OK, I'll remind you at {nice}: {r['text']}"), {"reminder": r})

    if _BRIEF.search(text) and ctx.get("brief"):
        return Skill("brief", ctx["brief"](hi), {})
    if _STATUS.search(text) and ctx.get("status"):
        return Skill("status", ctx["status"](hi), {})
    m = _FILES.match(text)
    if m and ctx.get("files"):
        hits = ctx["files"](m.group(1).strip())
        if not hits:
            return Skill("files", "कोई फ़ाइल नहीं मिली।" if hi else "No matching files in your allowed folders.", {"files": []})
        return Skill("files", ("ये फ़ाइलें मिलीं:\n" if hi else "Found:\n") + "\n".join(f"- {h}" for h in hits[:10]), {"files": hits})
    m = _OPEN.match(text)
    if m and ctx.get("apps"):
        name = (m.group(1) or m.group(2) or "").strip().lower()
        app = ctx["apps"].get(name)
        if app:
            return Skill("open", (f"{name} खोलूँ? नीचे पुष्टि करें।" if hi else f"Open {name}? Confirm below."),
                         {"open": name, "confirm": True})
    return None


# ── entry point ──────────────────────────────────────────────────────────────

def handle(text: str, ctx: Optional[Dict[str, Any]] = None) -> Optional[Skill]:
    """Try every skill in order; None means 'let the model answer'."""
    text = (text or "").strip()
    if not text or len(text) > 500:
        return None
    ctx = ctx or {}
    for fn in (lambda t: assistant_skills(t, ctx), calculator, units, lambda t: time_date(t, ctx.get("now"))):
        try:
            r = fn(text)
        except Exception:   # a skill must never break chat
            r = None
        if r is not None:
            return r
    return None
