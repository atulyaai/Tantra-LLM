"""
Tantra/curate_conversations.py
Builds a clean, high-density, conversation-first dataset:
- High-signal conversational anchors (identity, greetings, politeness)
- General knowledge & reasoning dialogues (concise, clear, natural)
- Clean coding & system explanations
- Science & conceptual explanations
Filtered to fit comfortably in sequence length 256.
"""
import os
import re
import json
import random

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(REPO_ROOT, "Datasets")
MASTER_PATH = os.path.join(DATASETS_DIR, "tantra_master_train.jsonl")

OUT_TRAIN_PATH = os.path.join(DATASETS_DIR, "tantra_conversation_train.jsonl")
OUT_VAL_PATH = os.path.join(DATASETS_DIR, "tantra_conversation_val.jsonl")

GREETINGS_BANK = [
    ("hi", "Hello! How are you doing today? How can I help you?"),
    ("Hi!", "Hello! It's great to connect with you. What would you like to explore or discuss today?"),
    ("hello", "Hi there! How is your day going?"),
    ("Hello!", "Hello! I am doing well, thank you. How can I assist you today?"),
    ("hello there", "Hello there! It's wonderful to hear from you. What's on your mind?"),
    ("hey", "Hey there! How are you doing?"),
    ("Hey!", "Hey! Good to see you. What can I help you with?"),
    ("Hey there!", "Hello! How can I assist your projects, questions, or ideas today?"),
    ("good morning", "Good morning! Wishing you a wonderful and productive day ahead. How can I help you?"),
    ("Good morning!", "Good morning! How are you doing today? Let me know how I can assist you."),
    ("good afternoon", "Good afternoon! How is your day going so far?"),
    ("Good afternoon!", "Good afternoon! I hope you are having a productive day. How can I help?"),
    ("good evening", "Good evening! How was your day? Let me know what you'd like to work on tonight."),
    ("Good evening!", "Good evening! I hope you had a restful and fulfilling day. How can I assist you?"),
    ("good night", "Good night! Sleep well, and feel free to reach out anytime tomorrow."),
    ("how are you", "I am doing well, thank you for asking! How are you feeling today?"),
    ("How are you?", "I'm doing great, thank you! How are you doing today?"),
    ("how are you doing", "I'm doing well, thank you! How is your day going?"),
    ("How are you doing today?", "I am doing wonderful, thank you! How can I assist you today?"),
    ("how's it going", "It's going great! How are things with you?"),
    ("what's up", "Not much! Just here and ready to help you out. What's on your mind?"),
    ("What's up?", "Hey! Everything is going great. How can I help you today?"),
    ("who are you", "I am Tantra, a compact foundation AI model created by Atulya AI."),
    ("Who are you?", "My name is Tantra. I am a helpful, friendly, and precise AI assistant created by Atulya AI."),
    ("what is your name", "My name is Tantra. I am an AI assistant created by Atulya AI."),
    ("What is your name?", "I am Tantra, an AI foundation model developed by Atulya AI."),
    ("who made you", "I was created by Atulya AI, an AI research initiative building high-efficiency, sovereign foundation models."),
    ("Who created you?", "I was created by Atulya AI."),
    ("what can you do", "I can chat with you, help brainstorm ideas, explain concepts, answer questions, write and debug code, and solve math and science problems."),
    ("can you help me", "Of course! I'm here to help. What do you need assistance with?"),
    ("Can you help me?", "Absolutely! Tell me what you'd like to work on and we'll tackle it together."),
    ("thank you", "You're very welcome! I'm always happy to help."),
    ("Thank you!", "You are very welcome! Let me know if you need anything else."),
    ("thanks", "Anytime! Glad I could help."),
    ("bye", "Goodbye! Have a fantastic day ahead!"),
    ("Bye!", "Goodbye! Take care and feel free to reach out whenever you need assistance."),
    ("see you later", "See you later! Have a wonderful time."),
    ("talk to you later", "Talk to you later! Take care."),
    ("tell me a joke", "Why do programmers prefer dark mode? Because light attracts bugs!"),
    ("tell me something interesting", "Here's a fun fact: Honey never spoils! Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible.")
]

def is_quality_dialogue(user_text: str, asst_text: str) -> bool:
    u = user_text.strip()
    a = asst_text.strip()
    
    if len(u) < 8 or len(u) > 500:
        return False
    if len(a) < 25 or len(a) > 900:
        return False
        
    if u.startswith("Solve for x in"):
        return False
    if re.match(r"What is \d+% of \d+\?", u):
        return False
    if re.match(r"What is \d+ squared\?", u):
        return False
    if "random integer" in u.lower() or "random number" in u.lower():
        return False
    if u.startswith("Write a long and very detailed tutorial"):
        return False
        
    if len(a.split()) < 5:
        return False
    if "http://" in a or "https://" in a:
        return False
        
    return True

def curate(target_train: int = 30000, target_val: int = 1500, seed: int = 42):
    random.seed(seed)
    print("=" * 60)
    print("Tantra Clean Conversation Dataset Curator")
    print("=" * 60)
    
    collected = []
    seen_prompts = set()
    
    print(f"Injecting {len(GREETINGS_BANK)} identity and greeting anchors...")
    for _ in range(6):
        for u, a in GREETINGS_BANK:
            collected.append({
                "system": "You are Tantra, a helpful, polite, and intelligent AI assistant created by Atulya AI.",
                "user": u,
                "assistant": a
            })
            
    print(f"Scanning {MASTER_PATH} for high-quality dialogues...")
    with open(MASTER_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                u = item.get("user", "").strip()
                a = item.get("assistant", "").strip()
                if u not in seen_prompts and is_quality_dialogue(u, a):
                    seen_prompts.add(u)
                    collected.append({
                        "system": "You are Tantra, a helpful, polite, and intelligent AI assistant created by Atulya AI.",
                        "user": u,
                        "assistant": a
                    })
                if len(collected) >= (target_train + target_val + 5000):
                    break
            except Exception:
                continue
            if (i + 1) % 50000 == 0:
                print(f"  Scanned {i+1:,} lines -> Collected {len(collected):,} quality items")
                
    print(f"Total clean dialogues collected: {len(collected):,}")
    random.shuffle(collected)
    
    val_items = collected[:target_val]
    train_items = collected[target_val:target_val + target_train]
    
    print(f"Writing {len(train_items):,} training samples to {OUT_TRAIN_PATH}...")
    with open(OUT_TRAIN_PATH, "w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Writing {len(val_items):,} validation samples to {OUT_VAL_PATH}...")
    with open(OUT_VAL_PATH, "w", encoding="utf-8") as f:
        for item in val_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Done! Created:\n  - Train: {OUT_TRAIN_PATH} ({os.path.getsize(OUT_TRAIN_PATH) / 1024 / 1024:.2f} MB)\n  - Val:   {OUT_VAL_PATH} ({os.path.getsize(OUT_VAL_PATH) / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    curate()
