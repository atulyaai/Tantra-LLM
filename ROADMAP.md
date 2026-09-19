# Tantra AI Roadmap — Hindi-Hinglish Native Cognitive OS

## Philosophy
Don't make 199M model "know everything." Make it **behave correctly, speak naturally
in Hindi/Hinglish, use memory/tools, recover from mistakes, and improve.**

## Progress Stages

```
199M baseline
   ↓
Hindi/Hinglish SFT with correct data mix
   ↓
Language benchmark (Day 1, Day 7, Day 30)
   ↓
Cognitive integration (Manas → Vivek → Chitta → Smriti)
   ↓
Memory benchmark
   ↓
Failure/recovery benchmark
   ↓
Self-improvement loop (Nirikshak → Learning Queue → LoRA Expert)
   ↓
~500M
   ↓
~1B
```

## Data Mix Target

| Data Type | Share |
|-----------|------:|
| 🇮🇳 Hindi | 30% |
| 🗣️ Hinglish | 25% |
| 🇬🇧 English | 20% |
| 🧠 Reasoning | 10% |
| 💻 Code | 7.5% |
| ➗ Math/Science | 7.5% |

## Cognitive Architecture

```
User
  ↓
Manas (input processing, language detection)
  ↓
Vivek (reasoning gate)
  ├── answer directly
  ├── Chitta (emotional/social context)
  ├── Smriti (memory)
  ├── Yantra (tool)
  └── expert (LoRA specialist)
        ↓
     RWKV core
        ↓
     Vivek quality gate
        ↓
      Response
```

## Priority Order for Cognitive Layer

```
Vivek → Aham → Chitta → Manas → Smriti → Nirikshak → Experts
```

## Auto-Learning Loop

```
Conversation
    ↓
Failure / weak answer
    ↓
Nirikshak (evaluator)
    ↓
Vivek (decides if learning needed)
    ↓
Learning Queue
    ↓
RTD / AEF (data generation)
    ↓
LoRA Expert (fine-tune)
    ↓
Evaluation
    ↓
Promote or reject
```

## Language Benchmark (Fixed Test Set)

```
Hindi understanding
Hindi generation
Hinglish understanding
Hinglish generation
English understanding
English generation
Code
Math
Reasoning
Conversation consistency
Instruction following
```

Must be run at:
- Day 1 (baseline)
- Day 7
- Day 30

## Indian Context Knowledge

- Indian education system
- Indian businesses (MSME, startups)
- Rupees/Indian numbering (lakh, crore)
- Indian terminology (jo, baap, bhai, dost)
- Government/service terminology
- Indian geography/history/culture
- Common Indian conversational patterns

## Key Hinglish Examples

> "Bhai ye code kaise optimize kar sakte hain?"
> "Ye question bahut easy hai, tujhe samajh aa jayega."
> "Arre bhai, ye galat hai. Sahi yeh hai:"

Tantra should answer in the **same register** as the user — not automatically
switch to formal English.
