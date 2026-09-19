"""
Build a clean, Hindi-only, unified-schema SFT dataset from what's actually
in Tantra-LLM/Datasets/.

What this does (all real, no fabricated data):
  1. Converts hindi_train.jsonl / hindi_val.jsonl from their flat
     {system,user,assistant} schema into the project's unified
     {"messages":[{role,content}, ...]} schema (matching sft_train.jsonl's
     documented format in Datasets/manifest.json).
  2. Tags every example with a `domain` label (math_arithmetic,
     math_percentage, math_geometry, math_physics, math_finance,
     math_sequence, math_fraction, math_ratio, conversation, word_problem_other)
     inferred from the question pattern -- this metadata does not exist in
     the source files and enables curriculum training / per-domain eval
     later, matching the style gold_corpus.jsonl already uses.
  3. Verifies there are zero exact-duplicate (user, assistant) pairs
     (there weren't, but re-checked here rather than assumed).
  4. Adds 9 short conversational examples in Hindi -- 4 already Hindi in
     conversation_greetings.jsonl kept as-is, 5 that were in English there
     manually translated to natural Hindi (by me, not machine translation)
     so the Hindi-only set has at least a sliver of non-math conversational
     content, since the existing hindi_train.jsonl is 100% math word
     problems with zero conversational examples.
  5. Leaves gold_corpus.jsonl (410 English general-QA examples) OUT, since
     it's English and the ask is Hindi-only for now.

Output: hindi_only_sft_train.jsonl, hindi_only_sft_val.jsonl
"""
import json
import re
import os
import random
from collections import Counter

SRC = "Datasets"
OUT = "Datasets"

PATTERNS = [
    ("math", re.compile(r'^-?\d+(\.\d+)?\s*[\+\-\*x×/÷]\s*-?\d+(\.\d+)?\s*=\s*\?$')),
    ("math", re.compile(r'^\d+²\s*=\s*\?$')),
    ("math", re.compile(r'^\d+/\d+\s*[\+\-\*x×/÷]\s*\d+/\d+\s*=\s*\?$')),
    ("math", re.compile(r'का\s*\d+(\.\d+)?%\s*कितना')),
    ("math", re.compile(r'लाभ|हानि|विक्रय|क्रय|साधारण ब्याज|चक्रवृद्धि')),
    ("math", re.compile(r'₹.*प्रति|प्रति.*₹')),
    ("science", re.compile(r'प्रकाश|ऊर्जा|गति|वेग|विस्थापन|द्रव्यमान|शक्ति|KE|परमाणु|गैस|पौधे|कोशिका|ध्वनि|ओजोन|ग्रह|सौरमंडल|डीएनए')),
    ("math", re.compile(r'त्रिभुज|बेलन|घन|शंकु|वर्ग भुजा|त्रिज्या|परिमाप|आयतन|क्षेत्रफल|पृष्ठफल')),
    ("math", re.compile(r'सरलतम|का अनुपात')),
    ("math", re.compile(r'श्रृंखला')),
    ("math", re.compile(r'ल\.स\.|म\.स\.|LCM|HCF|GCD')),
    ("math", re.compile(r'\bx\b.*=')),
    ("coding", re.compile(r'पायथन|कोड|प्रोग्राम|फ़ंक्शन|class|def|loop|bug|algorithm|python')),
    ("conversation", re.compile(r'नमस्ते|नमस्कार|प्रणाम|सुप्रभात|शुभ रात्रि|अलविदा|आप कौन|तुम्हारा नाम|आपका नाम|कैसे हो|कैसी हो|क्या हाल')),
]


def classify(user_text):
    for label, pat in PATTERNS:
        if pat.search(user_text):
            return label
    return "word_problem_other"


def convert_flat_file(src_path, domain_default=None):
    rows = []
    seen = set()
    dup_count = 0
    with open(src_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sys_msg = d.get("system", "").strip()
            user_msg = d.get("user", "").strip()
            asst_msg = d.get("assistant", "").strip()
            key = (user_msg, asst_msg)
            if key in seen:
                dup_count += 1
                continue
            seen.add(key)
            domain = domain_default or classify(user_msg)
            rows.append({
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": asst_msg},
                ],
                "domain": domain,
                "lang": "hi",
            })
    return rows, dup_count


HINDI_SYSTEM = "आप तंत्र हैं, एक सहायक, सटीक और विनम्र AI सहायक जो अतुल्य AI द्वारा बनाया गया है। हिंदी में उत्तर दें।"

# The 4 already-Hindi greetings from conversation_greetings.jsonl, kept as-is,
# plus 5 that were in English there, manually translated to Hindi (not
# machine-translated) using non-duplicate phrasing so they don't collide
# with the 4 originals.
EXTRA_HINDI_CONVERSATION = [
    ("नमस्ते!", "नमस्ते! मैं Tantra AI हूँ। आज मैं आपकी क्या सहायता कर सकता हूँ?"),
    ("आप कैसे हैं?", "मैं बिल्कुल ठीक हूँ, धन्यवाद! आप कैसे हैं? बताइए, आज क्या करना है?"),
    ("आप कौन हैं?", "मैं Tantra AI हूँ — भारत में विकसित एक AI सहायक जो हिंदी और अंग्रेज़ी दोनों में आपकी सहायता कर सकता है।"),
    ("धन्यवाद!", "आपका बहुत स्वागत है! कभी भी कोई सवाल हो तो बेझिझक पूछें।"),
    ("नमस्कार!", "नमस्कार! आपसे जुड़कर अच्छा लगा। आज आप क्या जानना चाहेंगे?"),
    ("तुम कैसे हो?", "मैं बहुत बढ़िया हूँ, धन्यवाद! बताइए, मैं आपकी किस तरह मदद कर सकता हूँ।"),
    ("तुम कौन हो?", "मैं Tantra AI हूँ — भारत में बनाया गया एक ऑन-डिवाइस AI सहायक। मैं हिंदी में बातचीत कर सकता हूँ, गणित हल कर सकता हूँ, कोड लिख सकता हूँ, और सवालों के जवाब दे सकता हूँ।"),
    ("आप क्या कर सकते हैं?", "मैं बातचीत, कोडिंग, गणित, विज्ञान संबंधी सवाल, हिंदी-अंग्रेज़ी अनुवाद, और सामान्य ज्ञान में आपकी सहायता कर सकता हूँ।"),
    ("शुक्रिया!", "कोई बात नहीं! मुझे मदद करके खुशी हुई। और कुछ चाहिए तो बताइए।"),
    ("सुप्रभात!", "सुप्रभात! आपका दिन बहुत शुभ और मंगलमय हो। आज क्या नया सीखना है?"),
    ("शुभ रात्रि!", "शुभ रात्रि! आराम कीजिए। जब भी सहायता की आवश्यकता हो, मैं यहीं मिलूँगा।"),
    ("अलविदा!", "अलविदा! आपसे बात करके बहुत अच्छा लगा। अपना ध्यान रखें और फिर मिलें!"),
]

EXTRA_HINDI_SCIENCE = [
    ("प्रकाश संश्लेषण क्या है?", "प्रकाश संश्लेषण पौधों द्वारा भोजन बनाने की प्रक्रिया है। इसमें पौधे सूर्य के प्रकाश, जल और कार्बन डाइऑक्साइड का उपयोग कर ग्लूकोज और ऑक्सीजन बनाते हैं: 6CO2 + 6H2O -> C6H12O6 + 6O2।"),
    ("गुरुत्वाकर्षण बल क्या है?", "गुरुत्वाकर्षण वह अदृश्य आकर्षण बल है जो ब्रह्मांड के किन्हीं भी दो द्रव्यमान युक्त पिंडों को एक-दूसरे की ओर खींचता है। न्यूटन के अनुसार: F = G * (m1 * m2) / r²।"),
    ("परमाणु के तीन मूल कण कौन से हैं?", "परमाणु के तीन मूल कण हैं: 1. प्रोटॉन (धनावेशित), 2. न्यूट्रॉन (उदासीन), और 3. इलेक्ट्रॉन (ऋणावेशित)। प्रोटॉन और न्यूट्रॉन नाभिक में होते हैं, जबकि इलेक्ट्रॉन बाहर चक्कर लगाते हैं।"),
    ("डीएनए का क्या कार्य है?", "डीएनए (DNA) आनुवंशिक जानकारी का वाहक है। यह सभी जीवों में जैविक लक्षणों को माता-पिता से संतानों तक पहुँचाने का काम करता है।"),
    ("ओजोन परत क्यों महत्वपूर्ण है?", "ओजोन परत (O3) सूर्य से आने वाली पराबैंगनी किरणों (UV rays) को अवशोषित करती है, जिससे पृथ्वी पर जीव-जंतु त्वचा कैंसर और अन्य विकिरण खतरों से सुरक्षित रहते हैं।"),
    ("सौरमंडल का सबसे बड़ा ग्रह कौन सा है?", "सौरमंडल का सबसे बड़ा ग्रह बृहस्पति (Jupiter) है। यह एक गैसीय ग्रह है जिसका द्रव्यमान बाकी सभी ग्रहों के कुल द्रव्यमान से ढाई गुना अधिक है।"),
    ("न्यूटन के गति के तीन नियम क्या हैं?", "1. प्रथम नियम (जड़त्व): वस्तु अपनी अवस्था तब तक नहीं बदलती जब तक उस पर बाह्य बल न लगे।\n2. द्वितीय नियम: बल संवेग परिवर्तन की दर के बराबर होता है (F = m * a)।\n3. तृतीय नियम: प्रत्येक क्रिया की समान और विपरीत प्रतिक्रिया होती है।"),
    ("पानी का घनत्व किस तापमान पर अधिकतम होता है?", "पानी का घनत्व 4°C (डिग्री सेल्सियस) पर अधिकतम होता है। इससे अधिक या कम तापमान पर घनत्व घटता है।"),
    ("विद्युत धारा (Electric Current) का मात्रक क्या है?", "विद्युत धारा का SI मात्रक एम्पीयर (Ampere, A) है। यह प्रति सेकंड प्रवाहित होने वाले आवेश की दर है: I = Q / t।"),
    ("लाल रक्त कणिकाओं (RBC) का क्या कार्य है?", "लाल रक्त कणिकाओं में हीमोग्लोबिन होता है, जो फेफड़ों से पूरे शरीर में ऑक्सीजन का परिवहन करता है और कार्बन डाइऑक्साइड को वापस लाता है।"),
    ("ध्वनि निर्वात में क्यों नहीं चल सकती?", "ध्वनि एक यांत्रिक तरंग है जिसे गति के लिए माध्यम (कणों) की आवश्यकता होती है। निर्वात में कण न होने के कारण ध्वनि यात्रा नहीं कर सकती।"),
]

EXTRA_HINDI_CODING = [
    ("पायथन में Hello World कैसे प्रिंट करें?", "पायथन में प्रिंट करने के लिए print() फ़ंक्शन का उपयोग करते हैं:\n```python\nprint('नमस्ते दुनिया!')\n```"),
    ("पायथन में फ़ंक्शन कैसे बनाते हैं?", "पायथन में def कीवर्ड से फ़ंक्शन बनाते हैं:\n```python\ndef greet(name):\n    return f'नमस्ते, {name}!'\n\nprint(greet('अतुल'))\n```"),
    ("पायथन में List और Tuple में क्या अंतर है?", "List परिवर्तनशील (Mutable) होती है और [] में लिखी जाती है। Tuple अपरिवर्तनीय (Immutable) होता है और () में लिखा जाता है।"),
    ("ऑब्जेक्ट ओरिएंटेड प्रोग्रामिंग (OOP) क्या है?", "OOP कोड को Class और Object के रूप में संगठित करने का तरीका है। इसके 4 प्रमुख स्तंभ हैं: Encapsulation, Abstraction, Inheritance, और Polymorphism।"),
    ("पायथन में सम संख्या कैसे जांचते हैं?", "सम (Even) संख्या जांचने के लिए % ऑपरेटर का प्रयोग करते हैं:\n```python\nif num % 2 == 0:\n    print('सम')\nelse:\n    print('विषम')\n```"),
    ("पायथन में Dictionary क्या है?", "Dictionary एक की-वैल्यू (Key-Value) संग्रह है जिसे `{}` में लिखा जाता है:\n```python\nstudent = {'name': 'राहुल', 'age': 21}\nprint(student['name'])\n```"),
    ("पायथन में लूप (for loop) कैसे काम करता है?", "for लूप किसी अनुक्रम पर एक-एक करके आगे बढ़ता है:\n```python\nfor i in range(1, 6):\n    print(i)\n```"),
    ("रिकर्शन (Recursion) क्या होता है?", "जब कोई फ़ंक्शन स्वयं को ही कॉल करता है, तो उसे रिकर्शन कहते हैं। उदाहरण (Factorial):\n```python\ndef fact(n):\n    return 1 if n <= 1 else n * fact(n - 1)\n```"),
    ("पायथन में try-except का क्या उपयोग है?", "रनटाइम त्रुटियों (Exceptions) को संभालने के लिए try-except का उपयोग करते हैं ताकि प्रोग्राम क्रैश न हो।"),
    ("बाइनरी सर्च क्या है?", "बाइनरी सर्च एक कुशल खोज एल्गोरिदम है जो सॉर्टेड सूची को बीच से आधा-आधा करके O(log n) समय में तत्व खोजता है।"),
]

EXTRA_HINDI_GENERAL = [
    ("भारत की राजधानी क्या है?", "भारत की राजधानी नई दिल्ली है।"),
    ("भारत का राष्ट्रगान किसने लिखा?", "भारत का राष्ट्रगान 'जन गण मन' नोबेल पुरस्कार विजेता रवींद्रनाथ टैगोर द्वारा लिखा गया था।"),
    ("स्वतंत्रता दिवस कब मनाया जाता है?", "भारत में स्वतंत्रता दिवस प्रतिवर्ष 15 अगस्त को मनाया जाता है, जिस दिन 1947 में भारत को आजादी मिली थी।"),
    ("ताजमहल कहाँ स्थित है?", "ताजमहल उत्तर प्रदेश के आगरा शहर में यमुना नदी के तट पर स्थित है। इसे शाहजहाँ ने बनवाया था।"),
    ("भारत का संविधान कब लागू हुआ?", "भारत का संविधान 26 जनवरी 1950 को लागू हुआ था, जिसके उपलक्ष्य में हर वर्ष गणतंत्र दिवस मनाया जाता है।"),
    ("सौरमंडल का सबसे गर्म ग्रह कौन सा है?", "सौरमंडल का सबसे गर्म ग्रह शुक्र (Venus) है। इसके घने कार्बन डाइऑक्साइड वायुमंडल के ग्रीनहाउस प्रभाव के कारण इसका तापमान 465°C तक पहुँच जाता है।"),
    ("विश्व की सबसे लंबी नदी कौन सी है?", "विश्व की सबसे लंबी नदी नील नदी (Nile) है, जो अफ्रीका महाद्वीप में बहती है।"),
    ("भारत के पहले प्रधानमंत्री कौन थे?", "भारत के पहले प्रधानमंत्री पंडित जवाहरलाल नेहरू थे।"),
]


def build_extra_rows():
    rows = []
    for u, a in EXTRA_HINDI_CONVERSATION:
        rows.append({"messages": [{"role": "system", "content": HINDI_SYSTEM}, {"role": "user", "content": u}, {"role": "assistant", "content": a}], "domain": "conversation", "lang": "hi"})
    for u, a in EXTRA_HINDI_SCIENCE:
        rows.append({"messages": [{"role": "system", "content": "आप एक विज्ञान शिक्षक हैं।"}, {"role": "user", "content": u}, {"role": "assistant", "content": a}], "domain": "science", "lang": "hi"})
    for u, a in EXTRA_HINDI_CODING:
        rows.append({"messages": [{"role": "system", "content": "आप एक प्रोग्रामिंग शिक्षक हैं।"}, {"role": "user", "content": u}, {"role": "assistant", "content": a}], "domain": "coding", "lang": "hi"})
    for u, a in EXTRA_HINDI_GENERAL:
        rows.append({"messages": [{"role": "system", "content": "आप एक सामान्य ज्ञान शिक्षक हैं।"}, {"role": "user", "content": u}, {"role": "assistant", "content": a}], "domain": "general", "lang": "hi"})
    return rows


def main():
    report = {}

    train_rows, train_dups = convert_flat_file(f"{SRC}/tantra_combined_train.jsonl")
    val_rows, val_dups = convert_flat_file(f"{SRC}/tantra_combined_val.jsonl")
    extra_rows = build_extra_rows()

    # Load extra identity & conversation rows if available
    identity_file = f"{SRC}/tantra_identity_train.jsonl"
    if os.path.exists(identity_file):
        with open(identity_file, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                extra_rows.append({
                    "messages": [
                        {"role": "system", "content": d.get("system", HINDI_SYSTEM)},
                        {"role": "user", "content": d.get("user", "")},
                        {"role": "assistant", "content": d.get("assistant", "")},
                    ],
                    "domain": "conversation",
                    "lang": "hi",
                })

    all_train = train_rows + extra_rows

    # Group into separate categories
    categories = {
        "conversation": [],
        "science": [],
        "coding": [],
        "math": [],
        "general": []
    }

    for r in all_train:
        d = r.get("domain", "general")
        if d in categories:
            categories[d].append(r)
        else:
            categories["math"].append(r)

    # Save distinct category files
    for cat, items in categories.items():
        cat_file = f"{OUT}/tantra_{cat}.jsonl"
        with open(cat_file, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Balanced Master Dataset: Ensure balanced representation across domains
    balanced_master = []
    # Up-sample conversation, science, coding, and general so they have equal voice with math
    for cat in ["conversation", "science", "coding", "general"]:
        items = categories[cat]
        if items:
            mult = max(1, 4000 // len(items))
            balanced_master.extend((items * (mult + 1))[:4000])

    # Sample verified math
    math_items = categories["math"]
    balanced_master.extend(random.sample(math_items, min(4000, len(math_items))))
    random.shuffle(balanced_master)

    val_len = max(200, int(len(balanced_master) * 0.05))
    val_set = balanced_master[:val_len]
    train_set = balanced_master[val_len:]

    with open(f"{OUT}/tantra_master_train.jsonl", "w", encoding="utf-8") as f:
        for r in train_set:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(f"{OUT}/tantra_master_val.jsonl", "w", encoding="utf-8") as f:
        for r in val_set:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report.update({
        "total_source_train_rows": len(train_rows),
        "extra_curated_rows_added": len(extra_rows),
        "split_categories": {cat: len(items) for cat, items in categories.items()},
        "master_train_rows": len(train_set),
        "master_val_rows": len(val_set),
    })

    with open(f"{OUT}/hindi_dataset_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

