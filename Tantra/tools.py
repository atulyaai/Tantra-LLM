"""
Tantra/tools.py — more exact, offline tools (used by skills.handle before the model).

  money    EMI ("EMI for 5 lakh at 9% for 3 years"), simple / compound interest,
           GST ("18% GST on 2500", "2950 including 18% GST"), discount ("20% off 1499")
  dates    days between two dates, days until a date, weekday of a date, age
  words    number → words, Indian system ("12,34,567 in words" / "शब्दों में")
  health   BMI ("BMI 70 kg 175 cm")
  random   coin toss, dice, random number, password
Each returns a Skill (answer + card with the working) or None.
"""
from __future__ import annotations

import datetime as dt
import math
import re
import secrets
import string
from typing import List, Optional

from Tantra.skills import REGION, Skill, hindi, indian

# ── amounts: "5 lakh", "2.5 crore", "50k", "₹1,20,000" ──────────────────────

_MULT = {"k": 1e3, "thousand": 1e3, "hazar": 1e3, "हज़ार": 1e3, "हजार": 1e3, "lakh": 1e5, "lac": 1e5, "lakhs": 1e5,
         "लाख": 1e5, "crore": 1e7, "cr": 1e7, "करोड़": 1e7, "million": 1e6, "mn": 1e6}
_AMOUNT = r"(?:₹|rs\.?|inr|\$|usd|€|eur|£|gbp|aed|¥)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|thousand|hazar|हज़ार|हजार|lakhs?|lac|लाख|crore|cr|करोड़|million|mn)?"


def _amount(num: str, mult: Optional[str]) -> float:
    return float(num.replace(",", "")) * _MULT.get((mult or "").lower(), 1)


def rupees(x: float) -> str:
    """Money in the region's currency: ₹12,34,567 / $1,234,567 / €…"""
    return REGION.get("currency", "") + indian(round(x, 2), 2)


def _years(text: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(years?|yrs?|साल|वर्ष|saal|months?|महीने|mahine)", text, re.I)
    if not m:
        return None
    n = float(m.group(1))
    return n / 12 if m.group(2).lower().startswith(("m", "म")) else n


def _rate(text: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    return float(m.group(1)) if m else None


def money(text: str) -> Optional[Skill]:
    t = text.lower()
    hi = hindi(text)
    # EMI
    if re.search(r"\bemi\b|ईएमआई|loan|लोन|कर्ज|mortgage|monthly payment|installment", t):
        m = re.search(_AMOUNT, t)
        rate, years = _rate(t), _years(t)
        if m and rate and years:
            p, r, n = _amount(*m.groups()), rate / 12 / 100, round(years * 12)
            emi = p / n if r == 0 else p * r * (1 + r) ** n / ((1 + r) ** n - 1)
            total = emi * n
            steps = [f"Monthly rate = {rate}% ÷ 12 = {r * 100:.4f}%", f"Months = {n}",
                     f"EMI = P·r·(1+r)^n / ((1+r)^n − 1) = {rupees(emi)}",
                     f"Total paid = {rupees(total)} · Interest = {rupees(total - p)}"]
            head = f"EMI: {rupees(emi)} प्रति माह" if hi else f"EMI: {rupees(emi)} per month"
            return Skill("calculator", head, {"expression": f"{rupees(p)} at {rate}% for {n} months", "result": rupees(emi) + "/month", "steps": steps})
    # simple / compound interest
    if re.search(r"interest|ब्याज|byaj|byaaj", t):
        m = re.search(_AMOUNT, t)
        rate, years = _rate(t), _years(t)
        if m and rate and years:
            p = _amount(*m.groups())
            if re.search(r"compound|चक्रवृद्धि|chakravriddhi", t):
                amt = p * (1 + rate / 100) ** years
                steps = [f"A = P·(1 + r)^t = {rupees(p)} × (1 + {rate / 100})^{years:g}", f"A = {rupees(amt)}",
                         f"Interest = {rupees(amt - p)}"]
            else:
                si = p * rate * years / 100
                amt = p + si
                steps = [f"SI = P × R × T / 100 = {indian(p)} × {rate} × {years:g} / 100 = {rupees(si)}", f"Total = {rupees(amt)}"]
            return Skill("calculator", (f"ब्याज: {rupees(amt - p)}, कुल: {rupees(amt)}" if hi else
                                        f"Interest: {rupees(amt - p)}, total: {rupees(amt)}"),
                         {"expression": f"{rupees(p)} at {rate}% for {years:g} years", "result": rupees(amt), "steps": steps})
    # GST / VAT / sales tax (name and default rate from the region)
    if re.search(r"\bgst\b|जीएसटी|\bvat\b|sales tax|\btax\b|टैक्स", t):
        rate = _rate(t) or float(REGION.get("tax_rate", 18))
        tax = "VAT" if "vat" in t else "GST" if ("gst" in t or "जीएसटी" in t) else REGION.get("tax", "tax")
        nums = [n for n in re.findall(r"\d[\d,]*(?:\.\d+)?", t) if float(n.replace(",", "")) != rate]
        if nums:
            amt = float(nums[0].replace(",", ""))
            if re.search(r"includ|inclusive|सहित|शामिल|with (gst|vat|tax)|me gst", t):
                base = amt / (1 + rate / 100)
                steps = [f"Base = {rupees(amt)} ÷ {1 + rate / 100:g} = {rupees(base)}", f"{tax} = {rupees(amt - base)}"]
                return Skill("calculator", f"Base {rupees(base)} + {tax} {rupees(amt - base)}",
                             {"expression": f"{rupees(amt)} including {rate:g}% {tax}", "result": rupees(base), "steps": steps})
            gst = amt * rate / 100
            steps = [f"{tax} = {rupees(amt)} × {rate:g}% = {rupees(gst)}", f"Total = {rupees(amt + gst)}"]
            if tax == "GST" and REGION.get("code") == "IN":
                steps.append(f"(CGST {rupees(gst / 2)} + SGST {rupees(gst / 2)})")
            return Skill("calculator", f"{tax} {rupees(gst)}, total {rupees(amt + gst)}",
                         {"expression": f"{rate:g}% {tax} on {rupees(amt)}", "result": rupees(amt + gst), "steps": steps})
    # discount
    if re.search(r"discount|\boff\b|छूट|chhoot|chhut|sale", t):
        rate = _rate(t)
        nums = [n for n in re.findall(r"\d[\d,]*(?:\.\d+)?", t) if rate is None or float(n.replace(",", "")) != rate]
        if rate and nums:
            price = float(nums[0].replace(",", ""))
            off = price * rate / 100
            return Skill("calculator", (f"छूट {rupees(off)}, कीमत {rupees(price - off)}" if hi else
                                        f"You save {rupees(off)}, you pay {rupees(price - off)}"),
                         {"expression": f"{rate:g}% off {rupees(price)}", "result": rupees(price - off),
                          "steps": [f"{rupees(price)} × {rate:g}% = {rupees(off)}", f"{rupees(price)} − {rupees(off)} = {rupees(price - off)}"]})
    return None


# ── dates ────────────────────────────────────────────────────────────────────

_MONTHS = {m: i + 1 for i, names in enumerate([
    ("jan", "january", "जनवरी"), ("feb", "february", "फ़रवरी", "फरवरी"), ("mar", "march", "मार्च"), ("apr", "april", "अप्रैल"),
    ("may", "मई"), ("jun", "june", "जून"), ("jul", "july", "जुलाई"), ("aug", "august", "अगस्त"),
    ("sep", "sept", "september", "सितम्बर", "सितंबर"), ("oct", "october", "अक्टूबर"), ("nov", "november", "नवम्बर", "नवंबर"),
    ("dec", "december", "दिसम्बर", "दिसंबर")]) for m in names}
_MONTH_RE = "|".join(sorted(map(re.escape, _MONTHS), key=len, reverse=True))
_DATE = re.compile(rf"(\d{{4}})-(\d{{1,2}})-(\d{{1,2}})|(\d{{1,2}})[/.-](\d{{1,2}})[/.-](\d{{2,4}})|(\d{{1,2}})\s*({_MONTH_RE})\.?,?\s*(\d{{4}})?"
                   rf"|({_MONTH_RE})\.?\s+(\d{{1,2}}),?\s*(\d{{4}})?", re.I)
HI_DAYS = ["सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"]


def parse_dates(text: str, today: dt.date) -> List[dt.date]:
    out = []
    for m in _DATE.finditer(text):
        g = m.groups()
        try:
            if g[0]:
                d = dt.date(int(g[0]), int(g[1]), int(g[2]))
            elif g[3]:
                y = int(g[5]); y += 2000 if y < 100 else 0
                a, b = int(g[3]), int(g[4])
                month, day = (a, b) if REGION.get("date_order") == "mdy" else (b, a)   # 03/04: 3 April, or March 4 in the US
                d = dt.date(y, month, day)
            elif g[6]:
                d = dt.date(int(g[8]) if g[8] else today.year, _MONTHS[g[7].lower()], int(g[6]))
            else:
                d = dt.date(int(g[11]) if g[11] else today.year, _MONTHS[g[9].lower()], int(g[10]))
        except (ValueError, KeyError):
            continue
        out.append(d)
    if re.search(r"\btoday\b|आज|\baaj\b", text, re.I) and len(out) < 2:
        out.insert(0, today)
    return out


def dates(text: str, today: Optional[dt.date] = None) -> Optional[Skill]:
    today = today or dt.date.today()
    t = text.lower()
    hi = hindi(text)
    ds = parse_dates(text, today)
    if not ds:
        return None
    if re.search(r"\bage\b|उम्र|umar|umr|born|जन्म|janm", t):
        b = ds[0]
        years = today.year - b.year - ((today.month, today.day) < (b.month, b.day))
        last = b.replace(year=b.year + years) if not (b.month == 2 and b.day == 29) else dt.date(b.year + years, 3, 1)
        days = (today - last).days
        return Skill("calculator", (f"उम्र: {years} साल {days} दिन" if hi else f"Age: {years} years {days} days"),
                     {"expression": f"born {b:%d %b %Y}", "result": f"{years} years", "steps": [f"{b:%d %b %Y} → {today:%d %b %Y}"]})
    if len(ds) >= 2 and re.search(r"between|बीच|se .* tak|से .* तक|difference|kitne din|कितने दिन|days", t):
        a, b = ds[0], ds[1]
        n = abs((b - a).days)
        return Skill("calculator", (f"{n} दिन ({n // 7} हफ़्ते {n % 7} दिन)" if hi else f"{n} days ({n // 7} weeks {n % 7} days)"),
                     {"expression": f"{a:%d %b %Y} → {b:%d %b %Y}", "result": f"{n} days", "steps": []})
    if re.search(r"until|till|baaki|बाकी|left|kitne din|कितने दिन|how many days", t):
        d = ds[-1]
        if d < today and not re.search(r"\d{4}", text):
            d = d.replace(year=today.year + 1)
        n = (d - today).days
        return Skill("calculator", (f"{d:%d %b %Y} में {n} दिन बाकी हैं" if hi else f"{n} days until {d:%d %b %Y}") if n >= 0 else
                     (f"{d:%d %b %Y} को {-n} दिन हो गए" if hi else f"{d:%d %b %Y} was {-n} days ago"),
                     {"expression": f"today → {d:%d %b %Y}", "result": f"{n} days", "steps": []})
    if re.search(r"day|din|दिन|weekday|kaun sa|कौन सा|konsa|कौनसा", t):
        d = ds[-1]
        name = HI_DAYS[d.weekday()] if hi else d.strftime("%A")
        return Skill("time", (f"{d.day} {d:%b %Y} को {name} है।" if hi else f"{d:%d %B %Y} is a {name}."), {"iso": d.isoformat()})
    return None


# ── number → words (Indian system) ───────────────────────────────────────────

HI_0_99 = ("शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस इक्कीस बाईस "
           "तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस "
           "उनतालीस चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास पचास इक्यावन बावन "
           "तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ साठ इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर "
           "सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उनासी अस्सी इक्यासी बयासी तिरासी "
           "चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे इक्यानवे बानवे तिरानवे चौरानवे पचानवे छियानवे सत्तानवे "
           "अट्ठानवे निन्यानवे").split()
EN_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
EN_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _en_99(n: int) -> str:
    return EN_ONES[n] if n < 20 else EN_TENS[n // 10] + ("-" + EN_ONES[n % 10] if n % 10 else "")


def number_words(n: int, lang: str = "en", system: str = "") -> str:
    """Hindi always uses the Indian system; English follows the region: lakh/crore or million/billion."""
    if n == 0:
        return HI_0_99[0] if lang == "hi" else "zero"
    parts = []
    system = system or ("indian" if lang == "hi" else REGION.get("grouping", "indian"))
    units = ([(10**9, "अरब", "arab"), (10**7, "करोड़", "crore"), (10**5, "लाख", "lakh"), (10**3, "हज़ार", "thousand"),
              (100, "सौ", "hundred")] if system == "indian" else
             [(10**9, "", "billion"), (10**6, "", "million"), (10**3, "", "thousand"), (100, "", "hundred")])
    for size, h, e in units:
        if n >= size:
            q, n = divmod(n, size)
            q_words = number_words(q, lang, system) if q > 99 else (HI_0_99[q] if lang == "hi" else _en_99(q))
            parts.append(f"{q_words} {h if lang == 'hi' else e}")
    if n:
        parts.append(HI_0_99[n] if lang == "hi" else _en_99(n))
    return " ".join(parts)


def words(text: str) -> Optional[Skill]:
    m = re.search(r"(\d[\d,]*)\s*(?:ko\s*|को\s*)?(?:in words|to words|words|शब्दों में|shabdon me|shabdo me|अक्षरों में)", text, re.I)
    if not m:
        return None
    n = int(m.group(1).replace(",", ""))
    if n >= 10**12:
        return None
    return Skill("calculator", f"{number_words(n, 'hi')}\n{number_words(n, 'en').capitalize()}",
                 {"expression": indian(n), "result": number_words(n, "hi" if hindi(text) else "en"), "steps": [number_words(n, "en").capitalize()]})


# ── health ───────────────────────────────────────────────────────────────────

def bmi(text: str) -> Optional[Skill]:
    if not re.search(r"\bbmi\b|बीएमआई", text, re.I):
        return None
    kg = re.search(r"(\d+(?:\.\d+)?)\s*(kg|kilo|किलो)", text, re.I)
    cm = re.search(r"(\d+(?:\.\d+)?)\s*(cm|सेमी)", text, re.I)
    ft = re.search(r"(\d)\s*(?:ft|feet|foot|'|फुट)\s*(\d{1,2})?\s*(?:in|inch|inches|\"|इंच)?", text, re.I)
    if not kg or not (cm or ft):
        return None
    w = float(kg.group(1))
    h = float(cm.group(1)) / 100 if cm else (int(ft.group(1)) * 12 + int(ft.group(2) or 0)) * 0.0254
    b = w / h ** 2
    asian = REGION.get("bmi") == "asian"
    over, obese = (23, 27.5) if asian else (25, 30)
    cat = "underweight" if b < 18.5 else "normal" if b < over else "overweight" if b < obese else "obese"
    cat_hi = {"underweight": "कम वज़न", "normal": "सामान्य", "overweight": "ज़्यादा वज़न", "obese": "मोटापा"}[cat]
    return Skill("calculator", f"BMI {b:.1f} — " + (cat_hi if hindi(text) else cat),
                 {"expression": f"{w:g} kg ÷ ({h:.2f} m)²", "result": f"{b:.1f} ({cat})",
                  "steps": [f"{w:g} ÷ {h:.2f}² = {b:.1f}",
                            f"{'Asian' if asian else 'WHO'} cut-offs: <18.5 low · 18.5–{over} normal · {over}–{obese} over · >{obese} obese"]})


# ── random ───────────────────────────────────────────────────────────────────

def random_tools(text: str) -> Optional[Skill]:
    t = text.lower()
    hi = hindi(text)
    if re.search(r"coin|सिक्का|sikka|toss|टॉस|heads or tails", t):
        side = secrets.choice(["heads", "tails"])
        return Skill("calculator", ("चित" if side == "heads" else "पट") + f" ({side})" if hi else side.capitalize(), {"result": side})
    if re.search(r"\bdice\b|\bdie\b|पासा|paasa|pasa", t):
        return Skill("calculator", f"🎲 {secrets.randbelow(6) + 1}", {})
    m = re.search(r"random (?:number)?\s*(?:between|from)?\s*(\d+)\s*(?:to|and|-|से)\s*(\d+)|(\d+)\s*(?:से|se)\s*(\d+)\s*(?:के बीच|ke beech)?\s*(?:कोई|koi)?\s*(?:number|संख्या)", t)
    if m:
        a, b = (int(x) for x in (m.group(1) or m.group(3), m.group(2) or m.group(4)))
        lo, hi_ = min(a, b), max(a, b)
        return Skill("calculator", str(lo + secrets.randbelow(hi_ - lo + 1)), {"expression": f"random {lo}–{hi_}"})
    if re.search(r"password|पासवर्ड", t) and re.search(r"generate|make|bana|बना|new|create|random|strong", t):
        n = int((re.search(r"(\d{1,2})\s*(?:char|letter|अक्षर)", t) or [None, 16])[1])
        n = max(8, min(n, 64))
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*-_"
        pw = "".join(secrets.choice(alphabet) for _ in range(n))
        return Skill("calculator", pw, {"expression": f"{n}-character password (made on this computer, not saved)", "result": pw})
    return None


TOOLS = (money, dates, words, bmi, random_tools)
