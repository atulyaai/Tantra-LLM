"""
Hindi grammar expert data
=========================
Real Hindi "basic sense": sentence structure (SOV), gender/number agreement,
postpositions, question words, pronouns and common verb forms. Written as
short Q/A pairs so a tiny model can learn the patterns.

    python -m Tantra.data.seed_hindi_grammar --out Datasets/curated --n 200
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

# topic -> (question templates, answer templates). Both sides Hindi (romanized)
# with an English gloss so the model also learns the mapping.
GRAMMAR: Dict[str, List[tuple]] = {
    "word order": [
        ("hindi mein sentence ka order kya hota hai",
         "hindi mein order subject-object-verb (sov) hota hai. jaise 'main roti khata hoon' - main (subject) roti (object) khata hoon (verb)."),
        ("kya hindi mein verb aakhir mein aata hai",
         "haan, hindi mein verb usually aakhir mein aata hai. 'woh paadhta hai', 'woh khelta hai'."),
        ("english 'i eat rice' ko hindi mein kaise likhte hain",
         "'main roti khata hoon'. subject main, object roti, verb khata hoon - verb last mein."),
    ],
    "gender": [
        ("hindi mein kitne gender hote hain",
         "do gender hote hain: masculine (pulling) and feminine (striling)."),
        ("kya 'ladka' masculine hai ya feminine",
         "'ladka' masculine (pulling) hai, aur 'ladki' feminine (striling) hai."),
        ("'kitab' ka gender kya hai",
         "'kitab' (book) feminine (striling) hai. is liye 'kitab hai' - 'achhi kitab'."),
        ("kya 'ghar' masculine hai",
         "haan, 'ghar' (home/house) masculine (pulling) hai."),
        ("sangya ka number kaise badalta hai",
         "singular se plural: 'ladka' se 'ladke', 'kitab' se 'kitabein'. aakhir mein 'a' wale shabd 'e' ya 'ein' mein badalte hain."),
    ],
    "postpositions": [
        ("hindi mein postposition kya hota hai",
         "postposition verb ke aane se pehle use hota hai. jaise 'mez par' (on the table), 'ghar mein' (in the house)."),
        ("'par' ka matlab kya hai",
         "'par' ka matlab 'on' hota hai. 'kitab mez par hai'."),
        ("'mein' ka matlab kya hai hindi mein",
         "'mein' ka matlab 'in' hota hai. 'main ghar mein hoon' = i am in the house."),
        ("'ko' postposition kaise use hota hai",
         "'ko' ko-direct object ko mark karta hai. 'mujhe kitab do' - kitab ko pata hai ki woh mil rahi hai."),
        ("'se' ka matlab kya hai",
         "'se' ka matlab 'from', 'with', ya 'by'. 'main delhi se aaya' = i came from delhi."),
    ],
    "question words": [
        ("hindi mein 'what' kya hota hai",
         "hindi mein 'what' = 'kya'. jaise 'kya naam hai tumhara' = what is your name."),
        ("'where' hindi mein kya hota hai",
         "'where' = 'kahan'. 'aap kahan ja rahe hain' = where are you going."),
        ("'who' ka hindi kya hai",
         "'who' = 'kaun'. 'kaun aaya' = who came."),
        ("'why' hindi mein kya hota hai",
         "'why' = 'kyun'. 'kyun aa rahe ho' = why are you coming."),
        ("'how many' ka hindi kya hai",
         "'how many' = 'kitne'. 'kitne aaye' = how many came."),
    ],
    "pronouns": [
        ("hindi mein 'you' ke kitne roop hote hain",
         "teen roop: 'tu' (informal), 'tum' (neutral), 'aap' (formal/respectful)."),
        ("'hum' ka matlab kya hai",
         "'hum' ka matlab 'we' hota hai. 'hum khel rahe hain' = we are playing."),
        ("'woh' kisko kehta hai",
         "'woh' = 'he/she/they' (woh wahan hai = he/she is there). hindi mein gender nahi puchhta."),
        ("'yeh' ka matlab kya hai",
         "'yeh' = 'this' (near). 'yeh kitab hai' = this is a book."),
    ],
    "verb forms": [
        ("'honaa' ke roop batao present mein",
         "'main hoon', 'woh hai', 'hum hain', 'aap hain' - honaa (to be) ke present forms."),
        ("past tense kaise banta hai hindi mein",
         "'khana' -> 'khaya' (ate), 'jana' -> 'gaya' (went), 'karna' -> 'kiya' (did). verb root + aaya/gaya/kiya jaise endings."),
        ("future tense kaise likhte hain",
         "verb root + 'ga'/'gei'/'gi'. 'khunga' = i will eat, 'jaoge' = you will go."),
        ("kya 'hota hai' ka matlab hota hai",
         "haan, 'hota hai' ka matlab 'happens' ya 'is done' hota hai. 'aisa hota hai' = it happens / that is how it is."),
    ],
    "common phrases": [
        ("'shukriya' ka matlab kya hai",
         "'shukriya' ka matlab 'thank you'. formal mein 'dhanyavaad' bhi bol sakte hain."),
        ("'maaf kijiye' kab bolte hain",
         "'maaf kijiye' = excuse me / sorry. galti par ya dhyan dilane ke liye."),
        ("'kaise hain aap' ka reply kya hota hai",
         "'main theek hoon, aap kaise hain' ya bas 'theek hoon, shukriya'."),
        ("'samajh nahi aaya' ka matlab kya hai",
         "'samajh nahi aaya' = i did not understand."),
        ("'kripya dobara boliye' kya matlab hai",
         "'kripya dobara boliye' = please say that again."),
    ],
}

VARIANTS = ["", "", "", "hindi mein batao", "please batao", "samjhao"]


def generate(n: int, seed: int = 0) -> List[Dict]:
    rng = random.Random(seed)
    topics = list(GRAMMAR)
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < n * 15:
        tries += 1
        topic = rng.choice(topics)
        q, a = rng.choice(GRAMMAR[topic])
        q = (q + " " + rng.choice(VARIANTS)).strip()
        if q in seen and tries < n * 10:
            continue
        seen.add(q)
        out.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]})
    return out


def write_all(out_dir: str | Path, n: int = 200, seed: int = 0) -> int:
    out_dir = Path(out_dir)
    d = out_dir / "language"
    d.mkdir(parents=True, exist_ok=True)
    rows = generate(n, seed)
    with open(d / "hindi_grammar.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Datasets/curated")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    print(f"language/hindi_grammar: {write_all(a.out, a.n)} rows")
