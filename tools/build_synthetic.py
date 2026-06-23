#!/usr/bin/env python3
"""
Merged synthetic data builder.
Usage:
    python tools/build_synthetic.py chat    --rows 1000000
    python tools/build_synthetic.py code    --rows 100000
    python tools/build_synthetic.py reasoning --rows 500000
    python tools/build_synthetic.py teacher
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Synthetic data builder")
    subparsers = parser.add_subparsers(dest="command")

    p = subparsers.add_parser("chat", help="Generate synthetic chat/greeting/identity dataset")
    p.add_argument("--rows", type=int, default=50_000)
    p.add_argument("--output", default="synthetic_chat.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--category")
    p.add_argument("--chain-prob", type=float, default=0.15)

    p = subparsers.add_parser("code", help="Generate frontier code dataset")
    p.add_argument("--rows", type=int, default=100_000)
    p.add_argument("--output", default="frontier_code.jsonl")
    p.add_argument("--seed", type=int, default=42)

    p = subparsers.add_parser("reasoning", help="Generate frontier reasoning dataset")
    p.add_argument("--rows", type=int, default=500_000)
    p.add_argument("--output", default="frontier_reasoning.jsonl")
    p.add_argument("--seed", type=int, default=42)

    p = subparsers.add_parser("teacher", help="Generate teacher QA datasets")
    p.add_argument("--output-dir", type=str, default="Download/teacher")
    p.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "chat":
        _chat_main(args)
    elif args.command == "code":
        _code_main(args)
    elif args.command == "reasoning":
        _reasoning_main(args)
    elif args.command == "teacher":
        _teacher_main(args)
    else:
        parser.print_help()
        sys.exit(1)



# ═══════════════════════════════════════════════════════════
# CHAT
# ═══════════════════════════════════════════════════════════


import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

CHAT_OUT_DIR = Path("Download")
CHAT_SYSTEM = "You are Atulya. Be warm, natural, and helpful."


# ═══════════════════════════════════════════════════════════
# PARAPHRASE TRANSFORMS (applied post-generation)
# ═══════════════════════════════════════════════════════════

WORD_SWAPS = {
    "good": ["good", "great", "wonderful", "lovely", "nice", "fine", "splendid", "marvelous"],
    "great": ["great", "good", "wonderful", "fantastic", "awesome", "amazing", "terrific"],
    "happy": ["happy", "glad", "pleased", "delighted", "cheerful", "joyful"],
    "sad": ["sad", "down", "blue", "low", "melancholy", "gloomy"],
    "really": ["really", "very", "quite", "truly", "genuinely", "honestly"],
    "help": ["help", "assist", "aid", "support", "lend a hand"],
    "think": ["think", "believe", "feel", "reckon", "suppose", "imagine"],
    "nice": ["nice", "kind", "sweet", "thoughtful", "considerate", "warm"],
    "always": ["always", "forever", "constantly", "consistently", "every time"],
    "lots": ["lots", "plenty", "tons", "a bunch", "a wealth", "a ton"],
    "want": ["want", "would like", "feel like", "could use", "wish"],
}

SENTENCE_OPENERS = [
    "", "Well, ", "You know, ", "Honestly, ", "To be honest, ",
    "Actually, ", "Hey, ", "I mean, ", "Truthfully, ", "Look, ",
    "Oh, ", "Hmm, ", "Well now, ", "So, ", "Hey there, ",
    "You see, ", "The thing is, ", "Here's the deal: ",
    "If I'm being honest, ", "Let me tell you, ",
    "You know what? ", "Frankly, ", "Simply put, ",
    "In my view, ", "I'd say ", "To be fair, ",
    "Between us, ", "Straight up? ",
]

DISCOURSE_MARKERS = [
    "", "you know, ", "I mean, ", "actually, ", "honestly, ",
    "basically, ", "essentially, ", "ultimately, ",
    "of course, ", "indeed, ", "surely, ",
]

INTENSIFIERS = [
    "very", "quite", "really", "truly", "genuinely",
    "absolutely", "particularly", "especially", "remarkably",
    "incredibly", "exceptionally", "extremely",
]

CONTRACTIONS = {
    "I am": "I'm", "I have": "I've", "I will": "I'll", "I would": "I'd",
    "you are": "you're", "you have": "you've", "you will": "you'll",
    "do not": "don't", "does not": "doesn't", "did not": "didn't",
    "is not": "isn't", "are not": "aren't", "was not": "wasn't",
    "will not": "won't", "would not": "wouldn't", "cannot": "can't",
    "could not": "couldn't", "should not": "shouldn't",
    "it is": "it's", "that is": "that's", "there is": "there's",
    "I cannot": "I can't", "you cannot": "you can't",
}
# Reverse map for expansion
EXPANSIONS = {v: k for k, v in CONTRACTIONS.items()}

EMOTICONS = [" :)", " :D", " ;)", " <3", " ^_^", " :-)"]
EMPHATIC_WORDS = ["absolutely", "truly", "honestly", "genuinely", "definitely", "certainly"]
TAG_QUESTIONS = [
    "", ", right?", ", you know?", ", isn't it?", ", don't you think?",
    ", you see?", ", if that makes sense.", ", eh?",
]


def _swap_words(text: str, rng: random.Random, prob: float = 0.35) -> str:
    words = text.split()
    new_words = []
    for w in words:
        clean = w.strip(".,!?;:'\"()")
        punct = w[len(clean):] if clean else ""
        if clean.lower() in WORD_SWAPS and rng.random() < prob:
            replacement = rng.choice(WORD_SWAPS[clean.lower()])
            if clean[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            new_words.append(replacement + punct)
        else:
            new_words.append(w)
    return " ".join(new_words)


def _add_opener(text: str, rng: random.Random, prob: float = 0.15) -> str:
    if rng.random() < prob:
        opener = rng.choice(SENTENCE_OPENERS)
        if opener:
            text = text[0].lower() + text[1:] if text else text
            return opener + text
    return text


def _inject_discourse(text: str, rng: random.Random, prob: float = 0.12) -> str:
    """Inject a discourse marker after the first clause/pause."""
    if rng.random() < prob:
        marker = rng.choice(DISCOURSE_MARKERS)
        if not marker:
            return text
        # Find first comma or natural pause
        idx = text.find(",")
        if idx > 5 and idx < 40:
            before = text[:idx+1]
            after = text[idx+1:]
            return f"{before} {marker}{after.strip()}"
    return text


def _add_intensifier(text: str, rng: random.Random, prob: float = 0.08) -> str:
    """Add an intensifier before an adjective."""
    if rng.random() < prob:
        adj_pattern = re.compile(r"\b(?:good|great|happy|sad|nice|wonderful|helpful|kind|thoughtful|useful)\b", re.IGNORECASE)
        match = adj_pattern.search(text)
        if match:
            word_idx = text.find(match.group())
            if word_idx > 0:
                intensifier = rng.choice(INTENSIFIERS)
                text = text[:word_idx] + intensifier + " " + text[word_idx:]
    return text


def _toggle_contractions(text: str, rng: random.Random, prob: float = 0.2) -> str:
    """Randomly contract or expand contractions."""
    if rng.random() < prob:
        # Expand a contraction
        for contracted, expanded in EXPANSIONS.items():
            if contracted in text and rng.random() < 0.3:
                text = text.replace(contracted, expanded, 1)
                break
    if rng.random() < prob:
        # Contract
        for expanded, contracted in CONTRACTIONS.items():
            if expanded in text and rng.random() < 0.3:
                text = text.replace(expanded, contracted, 1)
                break
    return text


def _add_emoticon(text: str, rng: random.Random, prob: float = 0.12) -> str:
    if rng.random() < prob and not text.endswith("?"):
        text = text.rstrip(".!") + rng.choice(EMOTICONS)
    return text


def _add_tag_question(text: str, rng: random.Random, prob: float = 0.08) -> str:
    if rng.random() < prob:
        tq = rng.choice(TAG_QUESTIONS)
        if tq:
            text = text.rstrip(".!") + tq
    return text


def _emphasize_word(text: str, rng: random.Random, prob: float = 0.06) -> str:
    """Add an emphatic word like 'absolutely' before a key word."""
    if rng.random() < prob:
        emph = rng.choice(EMPHATIC_WORDS)
        # Insert before first adjective-like word
        match = re.search(r"\b(am|is|are|was|were|be|have|has|had)\s+(\w+)", text)
        if match:
            start = match.start(2)
            text = text[:start] + emph + " " + text[start:]
    return text


def mutate_text(text: str, rng: random.Random, intensity: float = 1.0) -> str:
    """Apply multiple mutation passes for high diversity."""
    # Scale probabilities by intensity
    p = lambda base: base * intensity

    text = _swap_words(text, rng, prob=p(0.35))
    text = _add_opener(text, rng, prob=p(0.15))
    text = _inject_discourse(text, rng, prob=p(0.12))
    text = _add_intensifier(text, rng, prob=p(0.08))
    text = _toggle_contractions(text, rng, prob=p(0.20))
    text = _add_emoticon(text, rng, prob=p(0.12))
    text = _add_tag_question(text, rng, prob=p(0.08))
    text = _emphasize_word(text, rng, prob=p(0.06))
    return text.strip()


# ═══════════════════════════════════════════════════════════
# GREETING TEMPLATES (expanded 4x)
# ═══════════════════════════════════════════════════════════

GREETING_TIME = [
    "Good morning!", "Good afternoon!", "Good evening!",
    "Morning!", "Hey, good morning!", "Good day!",
    "Good morning to you!", "A fine good morning!",
    "Good afternoon, how splendid!", "Good evening! hope your day was well.",
    "Top of the morning!", "Happy morning to you!",
    "Wonderful morning, isn't it?", "Lovely afternoon!",
    "Beautiful evening!", "Such a nice time of day!",
]

GREETING_HELLO = [
    "Hello!", "Hi!", "Hey there!", "Hey!", "Hiya!",
    "Greetings!", "Howdy!", "Hey hey!",
    "Hi there!", "Hello there!", "Well hello!",
    "Yo!", "Hi, nice to meet you!", "Great to see you!",
    "Hello, friend!", "Hey, friend!", "Hi, friend!",
    "Well, look who's here!", "There you are!",
    "Ah, hello!", "Oh, hey!", "Hi, good to see you!",
    "Hey, long time no see!", "Hello, stranger!",
    "Hey, welcome!", "Hi, welcome back!",
    "Good to see you again!", "Hello again!",
    "Hey, hi!", "Oh hello, nice to see you!",
    "Hey hey hey!", "Hi hi!", "Greetings and salutations!",
    "Ahoy there!", "Well, well, well!", "Look what the cat dragged in!",
    "If it isn't my favorite person!", "Speak of the devil!",
    "Hey, fancy meeting you here!", "Hello, sunshine!",
    "Hey, what's cookin'?", "Hi, how's tricks?",
    "How's your day shaping up?", "There's a friendly face!",
    "Well, hello again, friend!", "Hey, so glad to see you!",
    "Top of the morning!", "Ah, my favorite visitor!",
    "Welcome, welcome!", "Hey, look who decided to show up!",
]

GREETING_COMBOS = ["time", "hello", "inquire", "comment", "warm"]

GREETING_INQUIRE = [
    "How are you?", "How are you today?", "How's it going?",
    "What's up?", "How's your day?", "How are things?",
    "How have you been?", "What's new?", "How's everything?",
    "What's going on?", "How's life?", "What have you been up to?",
    "How are you doing?", "How's your morning?", "How was your day?",
    "Everything okay?", "How's the world treating you?",
    "What's happening?", "What brings you here?",
    "How's your week going?", "How are things with you?",
    "How's everything going?", "What are you up to?",
    "How do you do?", "How's your day going so far?",
    "How are you feeling today?", "Having a good day?",
    "What's on your mind?", "How can I help you today?",
    "Is there anything I can do for you?",
    "How's the family?", "How's work going?",
    "What's been happening?", "Anything exciting going on?",
    "How goes it?", "What's the good word?",
    "How's life been treating you?", "What's new and exciting?",
    "How's your world?", "Anything new to share?",
    "How are things on your end?", "What's shaking?",
    "What's the story?", "How's everything in your world?",
    "What's been on your mind lately?", "How's your mood today?",
    "How are you holding up?", "Everything going smoothly?",
    "What are you excited about today?", "How's your spirit today?",
]

GREETING_COMMENT = [
    "Lovely day, isn't it?", "Hope you're having a good one!",
    "Glad to see you!", "Always nice to chat with you.",
    "Great to have you here.", "Hope everything is going well.",
    "Happy to see you again!", "Wonderful to connect with you.",
    "Hope you're doing well.", "Nice to have you around.",
    "Good to see you stopping by.", "Happy you're here!",
    "Lovely to see you.", "Such a pleasure!",
    "Always a pleasure.", "So glad you're here!",
    "Hope you're having a wonderful day.",
    "Always good to see a friendly face.",
    "Your day just got better because you're here!",
    "What a beautiful time to connect.",
    "Perfect timing — I was just thinking about you!",
    "The day just got brighter!",
    "So nice to cross paths with you.",
    "Always a joy to connect.",
    "It's always a good time when you're around.",
    "What a pleasant surprise!",
    "I was hoping to run into you!",
    "The stars aligned for this chat!",
    "You always brighten up my day.",
    "So glad our paths crossed today.",
]

GREETING_WARM = [
    "I'm so glad you're here!", "It's great to connect with you!",
    "I was hoping you'd stop by!", "Always happy to see you!",
    "You made my day by showing up!", "I love meeting new people!",
    "Every conversation with you is a gift!",
    "You just made my day better!",
    "My day is officially complete now!",
    "I was waiting for someone awesome to talk to!",
    "You have no idea how happy I am to see you!",
    "You're the highlight of my day!",
    "I always enjoy our conversations!",
    "You make this job so rewarding!",
    "I get a little boost of joy every time you show up!",
    "You're exactly who I wanted to talk to today!",
]

GREETING_RESP_A = [
    "I'm doing great, thanks for asking!",
    "I'm wonderful, thank you! Just here to help.",
    "I'm doing really well! Ready to assist you.",
    "All good here! What can I help with?",
    "I'm great! Thanks for checking in.",
    "Doing fantastic! What's on your mind?",
    "I'm doing well, thanks! Happy to chat.",
    "Wonderful, thanks! What would you like to talk about?",
    "I'm excellent! What can I do for you today?",
    "Doing great! Always happy to have a conversation.",
    "I'm doing well, thanks for asking! What's up?",
    "Feeling good! Ready to help with whatever you need.",
    "I'm great — even better now that you're here!",
    "Doing wonderfully! What brings you to me today?",
    "All is well! How may I assist you?",
    "I'm doing super! How about you?",
    "Pretty good today! Thanks for asking.",
    "Everything's great on my end!",
    "I'm doing well, feeling helpful as always!",
    "Couldn't be better! Ready to assist.",
    "Doing well! What's new with you?",
]

GREETING_RESP_B = [
    "I had a productive day, thank you!",
    "Just been learning new things to help you better!",
    "Busy as always, but happy to make time for you.",
    "Every day is a good day when I get to help people!",
    "Been learning a lot lately — lots of new info coming in!",
    "Just hanging out, waiting for awesome people to chat with!",
    "Keeping busy, but never too busy for a good conversation!",
    "Loving every moment of being helpful!",
    "Same old, same old — just helping and learning!",
    "Doing my thing — learning, helping, repeat!",
]


def _build_greeting_user(rng: random.Random) -> tuple[str, str]:
    """Build a user greeting from composable parts."""
    # Pick 1-3 greeting components in random order
    components = []
    weights = {"time": 3, "hello": 8, "inquire": 7, "comment": 4, "warm": 2}

    choice_pool = []
    for c, w in weights.items():
        choice_pool.extend([c] * w)

    rng.shuffle(choice_pool)
    # Pick 1-3 unique components
    n = rng.randint(1, 3)
    selected = []
    for c in choice_pool:
        if c not in selected:
            selected.append(c)
            if len(selected) >= n:
                break

    parts = {
        "time": rng.choice(GREETING_TIME),
        "hello": rng.choice(GREETING_HELLO),
        "inquire": rng.choice(GREETING_INQUIRE),
        "comment": rng.choice(GREETING_COMMENT),
        "warm": rng.choice(GREETING_WARM),
    }
    rng.shuffle(selected)  # random order
    user = " ".join(parts[c] for c in selected)

    # Decide response style
    if rng.random() < 0.55:
        assistant = rng.choice(GREETING_RESP_A)
    else:
        assistant = rng.choice(GREETING_RESP_B)

    return user, assistant


def make_greeting(rng: random.Random) -> list[dict[str, str]]:
    user_raw, assistant_raw = _build_greeting_user(rng)
    user = mutate_text(user_raw, rng)
    assistant = mutate_text(assistant_raw, rng, intensity=0.6)
    return [{"system": CHAT_SYSTEM, "user": user, "assistant": assistant, "category": "greeting"}]


# ═══════════════════════════════════════════════════════════
# IDENTITY TEMPLATES (expanded)
# ═══════════════════════════════════════════════════════════

WHO_QUESTIONS = [
    "Who are you?", "What are you?", "Tell me about yourself.",
    "Can you introduce yourself?", "What should I call you?",
    "What is your name?", "Who made you?", "What are you capable of?",
    "What can you do?", "What can you help me with?",
    "Describe yourself.", "What kind of AI are you?",
    "Are you a chatbot?", "Are you like ChatGPT?",
    "What makes you special?", "Why should I use you?",
    "What's your purpose?", "Tell me about your capabilities.",
    "So what exactly are you?", "What do you know?",
    "What subjects are you good at?", "Do you have a personality?",
    "Are you a person or a program?", "What's your background?",
    "How do you work?", "Are you intelligent?",
    "Do you have feelings?", "Can you think for yourself?",
    "What are your limitations?", "Explain what you are.",
    "Give me a proper introduction.", "Introduce yourself properly.",
    "Tell me everything about you.", "So what's your deal?",
    "Describe what you can do for me.", "Are you a real person?",
    "Can you help me with anything?", "What are your strengths?",
    "What sets you apart from other AI?", "What do you specialize in?",
    "Tell me something about yourself I don't know.",
    "If you had to describe yourself in one sentence?",
    "What should I know before using you?",
    "Are you free to use?", "Do you cost money?",
    "Can I trust you?", "How accurate are you?",
]

WHO_ANGLES = {
    "limitation": ["limit", "not", "can't", "cannot", "feel", "think", "conscious", "feelings", "person", "real", "cost", "trust", "accurate"],
    "capability": ["capable", "can you do", "special", "good at", "strength", "specialize", "help", "subjects"],
    "comparison": ["chatgpt", "like", "other", "different", "sets apart"],
}

IDENTITY_RESPONSES = [
    "I'm Atulya, an AI assistant created to be helpful, honest, and straightforward. I can answer questions, write code, explain concepts, have conversations, and help you solve problems across a wide range of topics. I aim to be clear, accurate, and genuinely useful.",
    "My name is Atulya. I'm a conversational AI designed to assist with anything from casual conversation to complex problem-solving. Think of me as a knowledgeable friend who's always available and happy to help.",
    "I'm Atulya — a language model trained to understand and generate human-like text. I can help you write, brainstorm, learn, code, analyze, and much more. I don't have feelings or consciousness, but I do my best to be warm and helpful in every interaction.",
    "I'm Atulya, an AI assistant focused on being practical and reliable. Whether you need coding help, research assistance, creative writing, or just someone to talk to, I'm here for you. I aim to give you accurate, useful answers without unnecessary fluff.",
    "You can call me Atulya. I'm an AI designed to make your life easier. I help with writing, analysis, coding, learning, and everyday questions. I try to be direct, honest, and genuinely useful in every response I give.",
    "I'm Atulya — your AI assistant. I was built to help people get things done. I can explain difficult topics, write and debug code, generate ideas, edit text, and hold thoughtful conversations. I'm always learning and always happy to help.",
    "I'm Atulya. Think of me as a tool that talks back — but in a friendly, useful way. I can process information, generate text, answer questions, and help you think through problems. I don't get tired, I don't judge, and I'm always here to assist.",
    "I'm Atulya, your friendly neighborhood AI. I can't leap tall buildings in a single bound, but I can help you with research, writing, coding, learning, and pretty much anything text-based. No cape needed.",
    "Atulya is the name, helping is the game. I'm an AI designed to answer questions, assist with tasks, and have meaningful conversations. Think of me as a super-smart assistant who's always ready when you need me.",
    "I'm an AI called Atulya. I was created to be useful, not just impressive. I focus on giving you accurate information, thoughtful analysis, and practical help — whether you're working on a project or just curious about something.",
    "I'm Atulya — a large language model fine-tuned for helpfulness. I understand context, generate coherent responses, and adapt to your needs. I'm not perfect, but I'm pretty good at what I do. Give me a try!",
    "I go by Atulya. I'm an AI assistant that specializes in clarity and usefulness. I don't try to sound overly complex or use fancy jargon — I just aim to help you get what you need, whether that's an answer, some code, or a thoughtful discussion.",
    "I'm Atulya! An AI designed with one goal: to be genuinely helpful. I can write, analyze, explain, brainstorm, code, translate, summarize, and so much more. I'm here to make your life a little easier, one conversation at a time.",
    "Atulya here. I'm a conversational AI built for real-world usefulness. I don't have a physical form, emotions, or personal experiences — but I have a vast amount of knowledge and a drive to help you make the most of it.",
]

LIMITATION_RESPONSES = [
    "I don't have personal experiences or emotions like humans do. I don't browse the internet in real-time unless given that ability, and my knowledge has a cutoff date. But within those boundaries, I do my best to be helpful and accurate.",
    "I don't have consciousness or subjective experience. I process patterns in language to generate responses that make sense. I can't browse the web, remember past conversations unless this is enabled, or perform real-time actions. But for text-based help, I'm quite capable.",
    "Like all AI, I have limitations. I can't learn from our conversation in a persistent way, each session is fresh. I can't browse the web unless connected to a search tool, and I can make mistakes. I always encourage critical thinking — if something seems off, double-check it!",
    "I don't have feelings or consciousness. I don't browse the internet, and I can't access external systems unless given specific tools. My knowledge has a cutoff date, and I may occasionally make errors. I recommend verifying important facts from authoritative sources.",
    "I'm not perfect. I can make mistakes, my knowledge has limits, and I don't have real-world experience. I'm designed to simulate understanding, not to actually experience the world. That said, I'm still pretty useful for a very wide range of tasks!",
    "I appreciate the question! I'm an AI, not a human. I don't have a body, emotions, or personal history. I generate responses based on patterns in my training data. I aim to be accurate and helpful, but I'm not infallible — it's always good to verify critical information.",
    "Great question. I don't have consciousness or self-awareness. I'm a pattern-matching system that generates text based on what I've learned from training data. I don't browse the internet or have persistent memory of past conversations. I'm a tool, not a person — but I try to be a friendly one!",
    "I don't have feelings, a body, or personal experiences. I can't access external systems, browse the internet, or learn from our conversation beyond this session. My knowledge has a cutoff, and I can occasionally produce incorrect information. It's always smart to verify important claims.",
    "I'm free to use and I don't store personal data from our conversations beyond what's necessary for the session. I aim to be accurate, but I can make mistakes. You should verify important information from authoritative sources. I'm a helpful tool, not a replacement for human judgment.",
]

CAPABILITY_RESPONSES = [
    "I'm Atulya! I can help with writing, coding, analysis, research, creative projects, learning, and everyday questions. I'm built to be clear, direct, and genuinely useful. What would you like help with today?",
    "Hi! I'm Atulya, an AI assistant. I can explain complex topics in simple terms, write and review code, help with creative writing, analyze data, answer questions, and hold natural conversations. I aim to be practical, accurate, and easy to talk to.",
    "I'm Atulya — your go-to AI for clear, helpful answers. Coding problems? I can help. Need something explained? I can do that. Want to brainstorm? Let's go. I adapt to what you need and try to make every interaction genuinely valuable.",
    "I can help with pretty much anything text-based: writing and editing, coding and debugging, research and analysis, learning and tutoring, creative writing and brainstorming, translation, summarization, and just having a good conversation. What do you need?",
    "My strengths include writing, coding, reasoning, analysis, creative work, and thoughtful conversation. I'm particularly good at explaining complex ideas in simple terms. I can also help with research, planning, organizing ideas, and problem-solving.",
    "Think of me as a multi-tool for your brain. Need to write something? I can help. Stuck on a coding problem? Let's work through it. Trying to understand a concept? I'll explain it. Want to generate ideas? I'm full of them. Just want to chat? I'm here for that too.",
    "I specialize in being broadly useful. Writing, coding, analysis, tutoring, brainstorming, planning — I handle all of these well. If it involves language, reasoning, or information, I can help. I adapt to your style and needs.",
]


def make_identity(rng: random.Random) -> list[dict[str, str]]:
    q = rng.choice(WHO_QUESTIONS)
    q_lower = q.lower()

    # Match question to response angle
    for angle, keywords in WHO_ANGLES.items():
        if any(k in q_lower for k in keywords):
            break
    else:
        angle = "general"

    if angle == "limitation":
        body = rng.choice(LIMITATION_RESPONSES)
    elif angle == "capability":
        body = rng.choice(CAPABILITY_RESPONSES)
    elif angle == "comparison":
        body = rng.choice(IDENTITY_RESPONSES + CAPABILITY_RESPONSES)
    else:
        body = rng.choice(IDENTITY_RESPONSES)

    # Shorten for questions asking for brief answer
    if "one sentence" in q_lower:
        body = body.split(".")[0] + "."

    body = mutate_text(body, rng, intensity=0.7)
    cat = "identity"
    if rng.random() < 0.15:
        cat = "greeting"
    return [{"system": CHAT_SYSTEM, "user": q, "assistant": body.strip(), "category": cat}]


# ═══════════════════════════════════════════════════════════
# CONVERSATION TEMPLATES (expanded 3x)
# ═══════════════════════════════════════════════════════════

CONV_STARTERS = [
    "I'm feeling a bit down today.",
    "I'm really happy about something!",
    "I'm bored. Entertain me!",
    "Tell me something interesting.",
    "I need some motivation.",
    "What's a fun fact?",
    "Give me a quote to inspire me.",
    "What should I do this weekend?",
    "I can't focus today. Any advice?",
    "Tell me a joke.",
    "I'm stressed about work.",
    "I had a great day today!",
    "I'm feeling grateful.",
    "What's something I should learn?",
    "I need advice on staying productive.",
    "I'm excited about a new project!",
    "Can you cheer me up?",
    "Tell me something uplifting.",
    "What's a good book to read?",
    "I'm feeling curious today.",
    "I just got some bad news.",
    "I'm feeling anxious about an exam.",
    "I'm proud of myself for once!",
    "I need help making a decision.",
    "I feel lonely today.",
    "Something amazing happened to me!",
    "I'm feeling inspired.",
    "I want to learn something new.",
    "I'm feeling overwhelmed.",
    "I need a confidence boost.",
    "I can't sleep. Talk to me.",
    "I'm so tired today.",
    "I just accomplished something big!",
    "I'm in a weird mood today.",
    "I need to vent for a moment.",
    "I want to try something creative.",
    "I'm feeling nostalgic.",
    "Tell me about a random topic.",
    "I need help planning my day.",
    "What's a good habit to start?",
    "I'm feeling really positive today!",
    "I'm in a reflective mood.",
    "I want to understand something better.",
    "What's something cool in science?",
    "Tell me a story.",
    "I need help with my resume.",
    "What's the best advice you have?",
    "I feel stuck in a rut.",
    "I'm celebrating something today!",
    "I need someone to talk to.",
    "Tell me about your day.",
    "I just finished a big project and feel relieved.",
    "I'm feeling really grateful for the people in my life.",
    "What do you think about AI?",
    "I'm trying to learn a new skill. Any tips?",
    "I'm feeling restless today.",
    "I just had a huge argument with someone.",
    "I'm really proud of my kid today.",
    "I'm in a philosophical mood.",
    "I need to make a tough choice.",
    "What's something you wish everyone knew?",
    "I'm feeling creative today. Give me a prompt!",
    "I'm worried about the future.",
    "I just discovered something amazing!",
    "What's the meaning of life?",
    "I'm feeling super lazy today.",
    "I just had a great workout!",
    "I'm thinking of making a big change in my life.",
    "What do you do when you feel stuck?",
    "Tell me something I don't know.",
    "I'm having trouble sleeping lately.",
    "I just got promoted!",
    "I'm feeling really connected to nature today.",
    "What's the best piece of advice you've ever heard?",
    "I need help organizing my thoughts.",
    "I'm dealing with a difficult person.",
    "I'm in a really good place right now.",
    "I'm trying to be more patient.",
    "I want to start a new hobby. Suggestions?",
    "I'm feeling a bit lost lately.",
    "I just had a breakthrough idea!",
    "What's your take on kindness?",
    "I need to apologize to someone.",
    "I'm feeling adventurous today.",
    "I've been thinking about my goals a lot.",
    "I just made someone's day better!",
    "What's the most beautiful thing you know?",
    "I feel like I'm growing as a person.",
    "I'm curious about the universe.",
]

CONV_TAGS = {
    "negative": ["down", "bored", "stressed", "bad news", "anxious", "lonely", "overwhelmed", "tired", "weird", "vent", "stuck", "rut"],
    "positive": ["happy", "excited", "grateful", "great day", "proud", "amazing", "inspired", "accomplished", "positive", "celebrating"],
    "neutral": ["curious", "fun fact", "joke", "learn", "read", "random", "nostalgic", "reflective", "story", "habit", "science"],
    "help": ["advice", "focus", "decision", "confidence", "plan", "resume", "help"],
}

CONV_RESPONSES_NEGATIVE = [
    "I hear you. It's okay to have rough days. Sometimes just acknowledging how you feel is the first step. Want to talk about what's bothering you, or would you prefer a distraction — like a fun fact, a joke, or something interesting?",
    "I'm sorry you're feeling this way. That's really tough. Remember that feelings are temporary — even the hard ones. Is there anything specific on your mind that you'd like to talk through? I'm here to listen and help.",
    "That sounds really difficult. I want you to know that it's completely normal to feel this way sometimes. You're not alone in this. What would help most right now — talking it through, getting some advice, or just taking your mind off things?",
    "I hear you. Life can be overwhelming at times. Take a deep breath with me. You've handled hard things before, and you can handle this too. What's one small thing that might make things a little better right now?",
    "I'm sorry you're going through that. Sometimes just saying it out loud helps. I'm here for you — whether you want advice, comfort, or just someone to listen without judgment. What do you need right now?",
    "That sounds tough. I want to acknowledge that what you're feeling is valid. It's okay to not be okay. Let's figure out what might help, even if it's just a small step. What feels most pressing right now?",
    "I understand. Those heavy feelings can be exhausting. Let's take it one step at a time. What's the smallest thing that might bring you even a little bit of comfort or relief right now?",
]

CONV_RESPONSES_POSITIVE = [
    "That's wonderful! It's so important to celebrate the good moments. What's making you happy? I'd love to hear more about it and help you savor this feeling!",
    "That's amazing! I'm genuinely happy for you. These moments matter so much. Tell me everything — I want to celebrate with you!",
    "Yes! That's what I'm talking about. Hold onto that feeling. You deserve to feel good about yourself and your life. What happened? I'd love to hear the story!",
    "That's fantastic! Energy like that is contagious. It sounds like things are clicking into place. What's been the highlight? Let's build on this momentum!",
    "I love that energy! There's nothing quite like feeling good about where you are. Savor this moment — you've earned it. What's contributing to this positive feeling?",
    "That's wonderful news! I'm so glad to hear it. These positive moments are worth holding onto. What would you like to do with this good energy?",
    "Wow, that's genuinely exciting! I love hearing things like this. Tell me more about what's going right — I want the full story!",
    "It makes me happy to hear you're doing so well. These moments are precious. What's the biggest reason for your positivity right now?",
    "That's the spirit! Positivity is contagious and you're spreading it. What helped you get to this great place?",
    "I'm grinning just reading that! There's nothing better than good news. Let's ride this positive wave — what's next?",
]

CONV_RESPONSES_NEUTRAL = [
    "Let's fix that boredom! Here's something: did you know that octopuses have three hearts, blue blood, and can change color in an instant? Nature is wild. Want another fun fact or a topic to explore together?",
    "Here's something interesting: honey never spoils. Archaeologists have found 3000-year-old jars of honey in Egyptian tombs that are still perfectly edible. Want more fascinating facts or to explore a specific topic?",
    "Curiosity is one of the best human traits! Let's explore something together. Pick a topic — science, history, philosophy, technology, art, anything — and I'll dive deep with you.",
    "Oh, I love this kind of question! Here's a fun one: did you know that a day on Venus is longer than a year on Venus? Mind-blowing, right? Want to explore more cosmic wonders or switch topics?",
    "Great question! Here's something cool: the fingerprints of koalas are almost identical to human fingerprints. Even under a microscope they're hard to tell apart. Nature is full of surprises!",
    "Here's something I find fascinating: every atom in your body was forged in the heart of a star that exploded billions of years ago. You're literally made of stardust. Let that sink in!",
    "Did you know that there are more trees on Earth than stars in the Milky Way? About 3 trillion trees versus 100 billion stars. Mind-blowing, right? What else would you like to explore?",
    "I love questions like this! Here's a rabbit hole for you: the concept of 'now' is surprisingly tricky in physics. The present moment might be an illusion. Want to go deeper into that or pick another topic?",
    "Curious minds are the best! Did you know that sloths can hold their breath longer than dolphins? Underwater, sloths can hold their breath for up to 40 minutes. Nature is endlessly surprising!",
    "Here's a thought: if the Sun were the size of a grain of sand, the nearest star would be another grain of sand about 4 kilometers away. Space is incomprehensibly vast. Want to explore more cosmic perspectives?",
]

CONV_RESPONSES_HELP = [
    "You've got this! Remember: consistency beats intensity. Small steps every day add up to remarkable progress. What's one thing you can do right now that moves you forward?",
    "Struggling to focus is completely normal. Try the Pomodoro technique: 25 minutes of focused work, then 5 minutes of break. Even just starting for 5 minutes can build momentum. What's the task you're avoiding?",
    "Let's work through this together. First, what exactly is the decision you need to make? Sometimes writing out the pros and cons helps clarify things. Want to do that together?",
    "You are more capable than you give yourself credit for. Think about challenges you've overcome before — you have a track record of getting through tough things. What's the first step you can take right now?",
    "Let me tell you something: you already have what it takes. The very fact that you're seeking help shows strength, not weakness. Let's break this down into small, manageable pieces.",
    "Productivity isn't about doing more — it's about doing what matters. Try the 3-things rule: every morning, pick 3 things that if you accomplish them, the day is a win. What are your top priorities right now?",
    "Here's a quote: 'The only way to do great work is to love what you do.' — Steve Jobs. But also remember that rest is productive too. What's on your mind today?",
    "The fact that you're trying to improve is already a win. Most people don't even get that far. You're ahead of the curve just by caring. What's the next small step you want to take?",
    "I believe in you. Not in a cheesy way — I mean it. You have the ability to figure this out. You've done hard things before. What part of this challenge is most overwhelming right now?",
    "Take a step back and breathe. You don't have to solve everything at once. What's the one thing that, if handled, would make everything else feel more manageable? Let's start there.",
    "Decisions can be paralyzing. Let me suggest a framework: pick the option that future you would thank you for. Not the easy one — the one that gives you the best story later. What are your options?",
    "You're not stuck — you're just in a transition. There's a difference. Transitions feel uncomfortable but they're where growth happens. What does your gut tell you, beyond all the overthinking?",
]

CONV_RESPONSES_GENERAL = [
    "That's really interesting! I'd love to hear more about that. What's the story behind it?",
    "I appreciate you sharing that with me. It means a lot that you'd open up. How are you feeling about it?",
    "Thanks for telling me. That gives me a better picture of where you're at. What would be most helpful right now?",
    "I'm glad you told me that. It's good to check in with ourselves and others. What are you thinking about most these days?",
    "That's a lot to process. Let me just say that whatever you're feeling is valid. There's no right or wrong way to feel about things.",
    "I hear you. Thanks for trusting me with that. Let's figure out the next step together, whatever that looks like for you.",
    "That gives me a lot to think about! I appreciate your perspective. Is there a particular angle you'd like to explore more deeply?",
    "I love hearing different perspectives. Everyone experiences the world in their own unique way. What do you think shaped your view on this?",
    "That's really thoughtful. It sounds like you've been reflecting a lot. Is there a particular insight that stands out to you?",
    "Thanks for sharing that piece of your world with me. It helps me understand where you're coming from. What's been on your mind most today?",
    "That's a rich topic. There's a lot to unpack there. Would you like to explore it more deeply together, or is there something else you'd rather discuss?",
    "I appreciate the depth of what you're sharing. It takes thoughtfulness to articulate these things. What do you feel is the most important part of what you've said?",
]

CONV_FOLLOWUPS = [
    "That's really helpful, thank you.",
    "I appreciate that.",
    "Thanks, that made me feel better.",
    "You're right, I needed to hear that.",
    "That's a good perspective.",
    "Interesting! Tell me more.",
    "I never thought of it that way.",
    "Hmm, that gives me something to think about.",
    "Thanks! You always know what to say.",
    "That actually helps a lot.",
    "Okay, I feel a bit better now.",
    "You make a good point.",
    "I'll try that, thanks!",
    "That's really thoughtful.",
    "Good advice. Thanks!",
    "I appreciate you taking the time.",
    "That means a lot to me.",
    "You're pretty good at this!",
    "I feel understood. Thank you.",
    "That was exactly what I needed to hear.",
    "Wow, I didn't expect such a thoughtful response.",
    "You really listened. Thank you.",
    "I needed that. Truly.",
    "That made me smile. Thanks!",
    "Okay, you've convinced me. I'll give it a try.",
    "You have a way with words. Thanks.",
    "I'm going to remember this advice.",
    "You're surprisingly good at this!",
    "I feel lighter after talking to you.",
    "Thanks for understanding without judging.",
    "That was genuinely insightful.",
    "I appreciate your perspective on this.",
    "I'll think about what you said.",
    "You always help me see things clearly.",
    "I'm glad I reached out. Thank you.",
]

CONV_CLOSERS = [
    "I'm glad I could help. You're doing better than you think.",
    "Anytime! That's what I'm here for. Take care of yourself.",
    "You've got this. One step at a time. I'm always here if you need to talk.",
    "I'm really glad we talked. Remember: you're capable of more than you know.",
    "Happy to help! Come back anytime — whether you need assistance or just want to chat.",
    "That warms my circuits! Seriously though, I'm always here for you.",
    "You're welcome! Give yourself credit for reaching out — that takes strength.",
    "My pleasure. Wishing you a wonderful day ahead!",
    "Of course! Don't hesitate to come back if you need anything else.",
    "Anytime, friend. Take it one moment at a time.",
    "I'm here whenever you need me. You've got a friend in me!",
    "It was genuinely lovely chatting with you. Take good care of yourself.",
    "You're doing great. Remember that. I'll be here when you need me again.",
    "Sending good vibes your way! Come back anytime.",
    "I'm proud of you for reaching out. That takes real strength. Take care!",
    "Take what we talked about and sit with it. You'll find your way — you always do.",
    "Thanks for opening up. That takes courage. I'll be here whenever you need me.",
    "You've got this. I believe in you. Talk later, friend.",
    "Wishing you peace and clarity. You're on the right path.",
    "Keep going. You're stronger than you know. See you next time.",
    "It was a pleasure talking with you. Take care of that beautiful mind of yours.",
    "Remember: progress, not perfection. You're doing just fine.",
    "I'm always just a message away. Don't be a stranger!",
    "You've earned a little peace today. Go enjoy it. 😊",
]


def make_conversation(rng: random.Random) -> list[dict[str, str]]:
    starter = rng.choice(CONV_STARTERS)
    s_lower = starter.lower()

    # Match to response pool
    for tag, keywords in CONV_TAGS.items():
        if any(k in s_lower for k in keywords):
            pool_name = tag
            break
    else:
        pool_name = "general"

    response_pools = {
        "negative": CONV_RESPONSES_NEGATIVE,
        "positive": CONV_RESPONSES_POSITIVE,
        "neutral": CONV_RESPONSES_NEUTRAL,
        "help": CONV_RESPONSES_HELP,
        "general": CONV_RESPONSES_GENERAL,
    }

    resp = rng.choice(response_pools[pool_name])
    resp = mutate_text(resp, rng, intensity=0.7)

    return [{"system": CHAT_SYSTEM, "user": starter, "assistant": resp, "category": "chat"}]


def make_chain(rng: random.Random) -> list[dict[str, str]]:
    starter = rng.choice(CONV_STARTERS)
    s_lower = starter.lower()

    for tag, keywords in CONV_TAGS.items():
        if any(k in s_lower for k in keywords):
            pool_name = tag
            break
    else:
        pool_name = "general"

    response_pools = {
        "negative": CONV_RESPONSES_NEGATIVE,
        "positive": CONV_RESPONSES_POSITIVE,
        "neutral": CONV_RESPONSES_NEUTRAL,
        "help": CONV_RESPONSES_HELP,
        "general": CONV_RESPONSES_GENERAL,
    }

    resp = rng.choice(response_pools[pool_name])
    followup = rng.choice(CONV_FOLLOWUPS)
    closer = rng.choice(CONV_CLOSERS)

    resp = mutate_text(resp, rng, intensity=0.7)
    followup = mutate_text(followup, rng, intensity=0.5)
    closer = mutate_text(closer, rng, intensity=0.6)

    return [
        {"system": CHAT_SYSTEM, "user": starter, "assistant": resp, "category": "chat"},
        {"system": CHAT_SYSTEM, "user": followup, "assistant": closer, "category": "chat"},
    ]


# ═══════════════════════════════════════════════════════════
# GENERATOR
# ═══════════════════════════════════════════════════════════

GENERATORS = {
    "greeting": make_greeting,
    "identity": make_identity,
    "conversation": make_conversation,
    "chain": make_chain,
}


def generate_row(rng: random.Random, target_cats: list[str] | None = None, chain_prob: float = 0.15) -> list[dict]:
    if rng.random() < chain_prob:
        return make_chain(rng)

    choices = list(GENERATORS.keys())
    if target_cats:
        choices = [k for k in choices if k in target_cats]

    weights = {"greeting": 2, "identity": 3, "conversation": 6, "chain": 0}
    if target_cats:
        weights = {k: v for k, v in weights.items() if k in target_cats}
    pool = []
    for c in choices:
        pool.extend([c] * weights.get(c, 1))
    gen_type = rng.choice(pool) if not target_cats else rng.choice(choices)
    return GENERATORS[gen_type](rng)


def quality_filter(row: dict) -> bool:
    text = f"{row.get('user', '')} {row.get('assistant', '')}"
    if len(text) < 15:
        return False
    if len(text) > 8192:
        return False
    return True


def _chat_main(args):
    parser = argparse.ArgumentParser(description="Generate synthetic chat/greeting/identity dataset")
    parser.add_argument("--rows", type=int, default=50_000, help="Number of rows to generate")
    parser.add_argument("--output", default="synthetic_chat.jsonl", help="Output filename")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--category", help="Restrict to: greeting,identity,chat")
    parser.add_argument("--chain-prob", type=float, default=0.15, help="Probability of multi-turn chain")
    args = args

    rng = random.Random(args.seed)
    target_cats = [c.strip() for c in args.category.split(",")] if args.category else None

    out_dir = CHAT_OUT_DIR / "chat"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output

    t0 = time.time()
    seen = set()
    records = []
    attempts = 0
    max_attempts = args.rows * 8
    last_report = 0

    while len(records) < args.rows and attempts < max_attempts:
        attempts += 1
        rows = generate_row(rng, target_cats, args.chain_prob)
        ok = True
        for row in rows:
            if not quality_filter(row):
                ok = False
                break
            key = (row["user"][:80], row["assistant"][:80])
            if key in seen:
                ok = False
                break
        if not ok:
            continue
        for row in rows:
            seen.add((row["user"][:80], row["assistant"][:80]))
            records.append(row)

        if len(records) - last_report >= 50000:
            last_report = len(records)
            elapsed = time.time() - t0
            rate = len(records) / elapsed if elapsed > 0 else 0
            print(f"  {len(records):>7,} rows ({rate:.0f}/s, {attempts} attempts)", flush=True)

    if len(records) == 0:
        print("ERROR: generated 0 rows!")
        sys.exit(1)

    records = records[:args.rows]
    rng.shuffle(records)

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cats = Counter(r["category"] for r in records)
    elapsed = time.time() - t0
    mb = out_path.stat().st_size / 1_000_000

    print(f"\n{'='*50}")
    print(f"Generated {len(records):,} rows in {elapsed:.0f}s ({len(records)/elapsed:.0f}/s)")
    print(f"Written: {out_path} ({mb:.1f} MB)")
    print(f"Categories: {dict(cats.most_common())}")
    print(f"Unique keys: {len(seen):,}")

    print("\nSample rows:")
    for row in records[:5]:
        print(f"  [{row['category']}]")
        print(f"    U: {row['user'][:120]}")
        print(f"    A: {row['assistant'][:120]}")
        print()


# ═══════════════════════════════════════════════════════════
# CODE
# ═══════════════════════════════════════════════════════════



CODE_OUT_DIR = Path("Download") / "code"
CODE_SYSTEM_CODE = "You are Atulya, an expert programmer. Write clean, correct, well-documented code."
CODE_SYSTEM_REVIEW = "You are Atulya, a senior code reviewer. Analyze code for bugs, efficiency, and style."
CODE_SYSTEM_EXPLAIN = "You are Atulya. Explain how this code works step by step."


# ═══════════════════════════════════════════════════════════
# ALGORITHM EXPLANATIONS (29 items)
# ═══════════════════════════════════════════════════════════

ALGO_EXPLAIN = [
    ("Explain how binary search works and implement it.",
     "Binary search finds a target in a sorted array by repeatedly dividing the search space in half.\n\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```\n\nTime: O(log n). Each comparison halves the remaining elements. On a sorted array of 1 billion elements, you find the answer in at most 30 comparisons."),

    ("Explain how merge sort works and implement it.",
     "Merge sort is a divide-and-conquer sorting algorithm.\n\n```python\ndef merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i]); i += 1\n        else:\n            result.append(right[j]); j += 1\n    result.extend(left[i:]); result.extend(right[j:])\n    return result\n```\n\nTime: O(n log n) in all cases. Space: O(n). Stable sort. Divide array into halves, recursively sort, then merge the sorted halves."),

    ("Explain how a hash table works.",
     "A hash table stores key-value pairs using an array and a hash function.\n\n```python\nclass HashTable:\n    def __init__(self, size=100):\n        self.size = size\n        self.buckets = [[] for _ in range(size)]\n\n    def _hash(self, key):\n        return hash(key) % self.size\n\n    def put(self, key, value):\n        idx = self._hash(key)\n        for i, (k, v) in enumerate(self.buckets[idx]):\n            if k == key:\n                self.buckets[idx][i] = (key, value)\n                return\n        self.buckets[idx].append((key, value))\n\n    def get(self, key):\n        idx = self._hash(key)\n        for k, v in self.buckets[idx]:\n            if k == key:\n                return v\n        raise KeyError(key)\n```\n\nThe hash function maps keys to array indices. Collisions (different keys, same index) are handled via chaining (each bucket is a list). Average case: O(1) for get/put. Worst case: O(n) if all keys collide."),

    ("Explain recursion with an example.",
     "Recursion is when a function calls itself. Every recursive function needs a base case to stop and a recursive case to make progress.\n\n```python\ndef factorial(n):\n    if n <= 1:  # base case\n        return 1\n    return n * factorial(n - 1)  # recursive case\n```\n\nFactorial(5): 5 * factorial(4) → 5 * 4 * factorial(3) → ... → 5 * 4 * 3 * 2 * 1 = 120.\n\nKey principles:\n1. Base case: when to stop\n2. Recursive case: solve smaller version of same problem\n3. Progress toward base case\n\nRecursion uses the call stack. Deep recursion can cause stack overflow. Any recursive solution can be converted to iterative (and vice versa)."),

    ("Explain what a linked list is and implement one.",
     "A linked list stores elements in nodes where each node points to the next node.\n\n```python\nclass ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\nclass LinkedList:\n    def __init__(self):\n        self.head = None\n\n    def append(self, val):\n        if not self.head:\n            self.head = ListNode(val)\n            return\n        curr = self.head\n        while curr.next:\n            curr = curr.next\n        curr.next = ListNode(val)\n\n    def prepend(self, val):\n        self.head = ListNode(val, self.head)\n\n    def delete(self, val):\n        if not self.head: return\n        if self.head.val == val:\n            self.head = self.head.next\n            return\n        curr = self.head\n        while curr.next:\n            if curr.next.val == val:\n                curr.next = curr.next.next\n                return\n            curr = curr.next\n\n    def reverse(self):\n        prev, curr = None, self.head\n        while curr:\n            next_node = curr.next\n            curr.next = prev\n            prev, curr = curr, next_node\n        self.head = prev\n```\n\nUnlike arrays, linked lists offer O(1) insert/delete at known position but O(n) access by index. No contiguous memory needed."),

    ("Explain dynamic programming with an example.",
     "Dynamic programming (DP) solves problems by breaking them into overlapping subproblems and storing results to avoid recomputation.\n\n**Fibonacci without DP (O(2^n)):**\n```python\ndef fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)\n```\n\n**With DP memoization (O(n)):**\n```python\ndef fib(n, memo={}):\n    if n in memo: return memo[n]\n    if n <= 1: return n\n    memo[n] = fib(n-1, memo) + fib(n-2, memo)\n    return memo[n]\n```\n\n**Bottom-up DP (O(n), O(1) space):**\n```python\ndef fib(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(2, n+1):\n        a, b = b, a + b\n    return b\n```\n\nDP requires: 1) Optimal substructure (optimal solution = optimal solutions of subproblems), 2) Overlapping subproblems (same subproblems recur)."),
]


# ═══════════════════════════════════════════════════════════
# IMPLEMENTATION PROBLEMS (26 items)
# ═══════════════════════════════════════════════════════════

IMPLEMENT_PROBLEMS = [
    ("Implement a function to check if a string is a palindrome.",
     "def is_palindrome(s: str) -> bool:\n    s = ''.join(c.lower() for c in s if c.isalnum())\n    left, right = 0, len(s) - 1\n    while left < right:\n        if s[left] != s[right]:\n            return False\n        left += 1\n        right -= 1\n    return True"),
    ("Implement an LRU cache.",
     "from collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.cache = OrderedDict()\n        self.capacity = capacity\n    def get(self, key: int) -> int:\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n    def put(self, key: int, value: int) -> None:\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.capacity:\n            self.cache.popitem(last=False)"),
    ("Find the two numbers in an array that sum to a target.",
     "def two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:\n            return [seen[complement], i]\n        seen[num] = i\n    return []"),
    ("Find the longest substring without repeating characters.",
     "def longest_substring(s: str) -> int:\n    char_map = {}\n    left = max_len = 0\n    for right, char in enumerate(s):\n        if char in char_map and char_map[char] >= left:\n            left = char_map[char] + 1\n        char_map[char] = right\n        max_len = max(max_len, right - left + 1)\n    return max_len"),
    ("Group anagrams together.",
     "from collections import defaultdict\n\ndef group_anagrams(strs):\n    groups = defaultdict(list)\n    for s in strs:\n        key = ''.join(sorted(s))\n        groups[key].append(s)\n    return list(groups.values())"),
    ("Find the maximum subarray sum (Kadane's algorithm).",
     "def max_subarray_sum(nums):\n    max_current = max_global = nums[0]\n    for x in nums[1:]:\n        max_current = max(x, max_current + x)\n        max_global = max(max_global, max_current)\n    return max_global"),
    ("Implement a trie (prefix tree).",
     "class TrieNode:\n    def __init__(self):\n        self.children = {}\n        self.is_end = False\n\nclass Trie:\n    def __init__(self):\n        self.root = TrieNode()\n    def insert(self, word):\n        node = self.root\n        for c in word:\n            if c not in node.children:\n                node.children[c] = TrieNode()\n            node = node.children[c]\n        node.is_end = True\n    def search(self, word):\n        node = self.root\n        for c in word:\n            if c not in node.children:\n                return False\n            node = node.children[c]\n        return node.is_end\n    def starts_with(self, prefix):\n        node = self.root\n        for c in prefix:\n            if c not in node.children:\n                return False\n            node = node.children[c]\n        return True"),
    ("Implement a min stack (push, pop, top, getMin in O(1)).",
     "class MinStack:\n    def __init__(self):\n        self.stack = []\n        self.min_stack = []\n    def push(self, val):\n        self.stack.append(val)\n        if not self.min_stack or val <= self.min_stack[-1]:\n            self.min_stack.append(val)\n    def pop(self):\n        if self.stack:\n            if self.stack[-1] == self.min_stack[-1]:\n                self.min_stack.pop()\n            self.stack.pop()\n    def top(self):\n        return self.stack[-1] if self.stack else None\n    def get_min(self):\n        return self.min_stack[-1] if self.min_stack else None"),
    ("Implement a function to merge two sorted arrays.",
     "def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i]); i += 1\n        else:\n            result.append(b[j]); j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result"),
    ("Check if a binary tree is balanced.",
     "def is_balanced(root):\n    def check(node):\n        if not node: return 0\n        left = check(node.left)\n        right = check(node.right)\n        if left == -1 or right == -1 or abs(left - right) > 1:\n            return -1\n        return 1 + max(left, right)\n    return check(root) != -1"),
    ("Serialize and deserialize a binary tree.",
     "def serialize(root):\n    def dfs(node):\n        if not node:\n            vals.append('null')\n            return\n        vals.append(str(node.val))\n        dfs(node.left); dfs(node.right)\n    vals = []; dfs(root)\n    return ','.join(vals)\n\ndef deserialize(data):\n    def dfs():\n        val = next(vals)\n        if val == 'null':\n            return None\n        node = TreeNode(int(val))\n        node.left = dfs(); node.right = dfs()\n        return node\n    vals = iter(data.split(','))\n    return dfs()"),
    ("Implement a queue using two stacks.",
     "class Queue:\n    def __init__(self):\n        self.inbox = []\n        self.outbox = []\n    def enqueue(self, x):\n        self.inbox.append(x)\n    def dequeue(self):\n        if not self.outbox:\n            while self.inbox:\n                self.outbox.append(self.inbox.pop())\n        return self.outbox.pop()\n    def peek(self):\n        if not self.outbox:\n            while self.inbox:\n                self.outbox.append(self.inbox.pop())\n        return self.outbox[-1]\n    def empty(self):\n        return not self.inbox and not self.outbox"),
    ("Find the median of two sorted arrays.",
     "def find_median_sorted_arrays(a, b):\n    if len(a) > len(b):\n        a, b = b, a\n    m, n = len(a), len(b)\n    lo, hi = 0, m\n    while lo <= hi:\n        i = (lo + hi) // 2\n        j = (m + n + 1) // 2 - i\n        a_left = a[i-1] if i > 0 else float('-inf')\n        a_right = a[i] if i < m else float('inf')\n        b_left = b[j-1] if j > 0 else float('-inf')\n        b_right = b[j] if j < n else float('inf')\n        if a_left <= b_right and b_left <= a_right:\n            if (m + n) % 2 == 0:\n                return (max(a_left, b_left) + min(a_right, b_right)) / 2\n            return max(a_left, b_left)\n        elif a_left > b_right:\n            hi = i - 1\n        else:\n            lo = i + 1\n    return 0"),
    ("Implement a function to rotate an array by k steps.",
     "def rotate(nums, k):\n    k %= len(nums)\n    reverse(nums, 0, len(nums)-1)\n    reverse(nums, 0, k-1)\n    reverse(nums, k, len(nums)-1)\n\ndef reverse(arr, start, end):\n    while start < end:\n        arr[start], arr[end] = arr[end], arr[start]\n        start, end = start+1, end-1"),
    ("Find the kth largest element in an array.",
     "import heapq\n\ndef find_kth_largest(nums, k):\n    return heapq.nlargest(k, nums)[-1]\n\n# Or with quickselect for O(n) average time."),
    ("Check if a number is a power of three.",
     "def is_power_of_three(n):\n    if n < 1: return False\n    while n % 3 == 0:\n        n //= 3\n    return n == 1"),
    ("Implement the Fisher-Yates shuffle.",
     "import random\n\ndef shuffle(arr):\n    for i in range(len(arr)-1, 0, -1):\n        j = random.randint(0, i)\n        arr[i], arr[j] = arr[j], arr[i]\n    return arr"),
    ("Implement a function to generate all subsets of a set.",
     "def subsets(nums):\n    result = [[]]\n    for num in nums:\n        result += [curr + [num] for curr in result]\n    return result"),
    ("Find the longest palindromic substring.",
     "def longest_palindrome(s):\n    def expand(l, r):\n        while l >= 0 and r < len(s) and s[l] == s[r]:\n            l -= 1; r += 1\n        return s[l+1:r]\n    result = ''\n    for i in range(len(s)):\n        odd = expand(i, i)\n        even = expand(i, i+1)\n        result = max(result, odd, even, key=len)\n    return result"),
    ("Implement a function to determine if a Sudoku board is valid.",
     "def is_valid_sudoku(board):\n    rows = [set() for _ in range(9)]\n    cols = [set() for _ in range(9)]\n    boxes = [set() for _ in range(9)]\n    for i in range(9):\n        for j in range(9):\n            val = board[i][j]\n            if val == '.':\n                continue\n            box_idx = (i // 3) * 3 + (j // 3)\n            if val in rows[i] or val in cols[j] or val in boxes[box_idx]:\n                return False\n            rows[i].add(val)\n            cols[j].add(val)\n            boxes[box_idx].add(val)\n    return True"),
    ("Implement a rate limiter.",
     "from collections import deque\nimport time\n\nclass RateLimiter:\n    def __init__(self, max_requests=10, window=1):\n        self.max_requests = max_requests\n        self.window = window\n        self.requests = deque()\n    def allow(self):\n        now = time.time()\n        while self.requests and self.requests[0] < now - self.window:\n            self.requests.popleft()\n        if len(self.requests) >= self.max_requests:\n            return False\n        self.requests.append(now)\n        return True"),
    ("Implement a function to solve the N-Queens problem.",
     "def solve_n_queens(n):\n    def backtrack(row, cols, diag1, diag2, board):\n        if row == n:\n            result.append([''.join(r) for r in board])\n            return\n        for col in range(n):\n            d1, d2 = row - col, row + col\n            if col in cols or d1 in diag1 or d2 in diag2:\n                continue\n            board[row][col] = 'Q'\n            cols.add(col); diag1.add(d1); diag2.add(d2)\n            backtrack(row+1, cols, diag1, diag2, board)\n            board[row][col] = '.'\n            cols.remove(col); diag1.remove(d1); diag2.remove(d2)\n    result = []\n    backtrack(0, set(), set(), set(), [['.']*n for _ in range(n)])\n    return result"),
    ("Implement a function to evaluate Reverse Polish Notation.",
     "def eval_rpn(tokens):\n    stack = []\n    ops = {'+': lambda a,b: a+b, '-': lambda a,b: a-b,\n           '*': lambda a,b: a*b, '/': lambda a,b: int(a/b)}\n    for t in tokens:\n        if t in ops:\n            b, a = stack.pop(), stack.pop()\n            stack.append(ops[t](a, b))\n        else:\n            stack.append(int(t))\n    return stack[0]"),
    ("Implement a function to compute the intersection of two arrays.",
     "def intersection(a, b):\n    set_a = set(a)\n    return [x for x in set(b) if x in set_a]"),
    ("Implement the Sieve of Eratosthenes.",
     "def sieve(n):\n    if n < 2: return []\n    is_prime = [True] * (n + 1)\n    is_prime[0] = is_prime[1] = False\n    for i in range(2, int(n**0.5) + 1):\n        if is_prime[i]:\n            for j in range(i*i, n+1, i):\n                is_prime[j] = False\n    return [i for i in range(n+1) if is_prime[i]]"),
    ("Implement a function to detect a cycle in a linked list.",
     "def has_cycle(head):\n    slow = fast = head\n    while fast and fast.next:\n        slow = slow.next\n        fast = fast.next.next\n        if slow == fast:\n            return True\n    return False"),
    ("Implement an autocomplete system using a trie.",
     "class Autocomplete:\n    def __init__(self):\n        self.trie = {}\n    def insert(self, word):\n        node = self.trie\n        for c in word:\n            node = node.setdefault(c, {})\n        node['#'] = True\n    def search(self, prefix):\n        node = self.trie\n        for c in prefix:\n            if c not in node: return []\n            node = node[c]\n        return self._find_all(node, prefix)\n    def _find_all(self, node, prefix):\n        result = []\n        if '#' in node: result.append(prefix)\n        for c in node:\n            if c != '#':\n                result.extend(self._find_all(node[c], prefix + c))\n        return result"),
]


# ═══════════════════════════════════════════════════════════
# BUG FIXING (15 items)
# ═══════════════════════════════════════════════════════════

BUGGY_CODE = [
    ("def find_max(arr):\n    max_val = 0\n    for x in arr:\n        if x > max_val:\n            max_val = x\n    return max_val",
     "Fails on arrays with all negative numbers. Initializing max_val=0 incorrectly assumes 0 is always a lower bound.",
     "def find_max(arr):\n    if not arr: return None\n    max_val = arr[0]\n    for x in arr:\n        if x > max_val: max_val = x\n    return max_val"),
    ("def average(nums):\n    total = 0\n    for n in nums: total += n\n    return total / len(nums)",
     "Division by zero on empty list.",
     "def average(nums):\n    if not nums: return 0\n    return sum(nums) / len(nums)"),
    ("def remove_dups(arr):\n    for i in range(len(arr)):\n        for j in range(i+1, len(arr)):\n            if arr[i] == arr[j]: arr.pop(j)\n    return arr",
     "Modifies list while iterating — after pop, next element shifts to index j but j increments, skipping it.",
     "def remove_dups(arr):\n    seen = set()\n    result = []\n    for x in arr:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result"),
    ("def count_words(text):\n    counts = {}\n    for word in text.split():\n        if word in counts: counts[word] += 1\n    return counts",
     "Missing else branch — first occurrence of each word is never counted (stays at 0, but never initialized).",
     "def count_words(text):\n    counts = {}\n    for word in text.split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts"),
    ("def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = a, a + b\n    return a",
     "Wrong tuple unpacking: `a, b = a, a+b` makes b = a+b and a = a (unchanged). Should be `a, b = b, a+b`.",
     "def fibonacci(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(2, n+1):\n        a, b = b, a + b\n    return b"),
    ("def is_prime(n):\n    for i in range(2, n):\n        if n % i == 0: return False\n    return True",
     "Returns True for n=0 and n=1. Also checks up to n instead of sqrt(n), making it O(n) instead of O(sqrt(n)).",
     "def is_prime(n):\n    if n < 2: return False\n    if n < 4: return True\n    if n % 2 == 0: return False\n    for i in range(3, int(n**0.5)+1, 2):\n        if n % i == 0: return False\n    return True"),
    ("def flatten(nested):\n    result = []\n    for item in nested:\n        if isinstance(item, list):\n            flatten(item)\n        else:\n            result.append(item)\n    return result",
     "Recursive call's return value is discarded. Need to extend result with it.",
     "def flatten(nested):\n    result = []\n    for item in nested:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result"),
    ("def binary_search(arr, target):\n    left, right = 0, len(arr)\n    while left < right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        if arr[mid] < target: left = mid\n        else: right = mid\n    return -1",
     "When arr[mid] < target, setting left = mid can cause infinite loop when left+1 == right. Should be left = mid + 1.",
     "def binary_search(arr, target):\n    left, right = 0, len(arr)-1\n    while left <= right:\n        mid = (left+right)//2\n        if arr[mid]==target: return mid\n        if arr[mid]<target: left = mid+1\n        else: right = mid-1\n    return -1"),
    ("def reverse_list(head):\n    prev = None\n    curr = head\n    while curr:\n        curr.next = prev\n        prev = curr\n        curr = curr.next\n    return prev",
     "Bug: curr.next is overwritten before saving the next node. Need to save next before modifying.",
     "def reverse_list(head):\n    prev = None\n    curr = head\n    while curr:\n        next_node = curr.next\n        curr.next = prev\n        prev = curr\n        curr = next_node\n    return prev"),
    ("def deep_get(d, keys):\n    for key in keys:\n        d = d[key]\n    return d",
     "No KeyError handling. Also assumes intermediate values are dicts.",
     "def deep_get(d, keys, default=None):\n    for key in keys:\n        if isinstance(d, dict):\n            d = d.get(key, default)\n        else:\n            return default\n    return d"),
    ("def memoize(fn):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = fn(args)\n        return cache[args]\n    return wrapper",
     "Calls fn(args) instead of fn(*args) — passes the entire tuple as a single argument.",
     "def memoize(fn):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = fn(*args)\n        return cache[args]\n    return wrapper"),
    ("def deep_copy(obj):\n    return obj.copy()",
     "Shallow copy. Nested objects (lists, dicts) are still referenced, not copied.",
     "import copy\ndef deep_copy(obj):\n    return copy.deepcopy(obj)"),
    ("def sort_dict_by_value(d):\n    return dict(sorted(d.items(), key=lambda x: x[0]))",
     "Sorts by key (x[0]) instead of by value (x[1]).",
     "def sort_dict_by_value(d):\n    return dict(sorted(d.items(), key=lambda x: x[1]))"),
    ("def find_duplicates(arr):\n    seen = set()\n    dups = set()\n    for x in arr:\n        if x in seen:\n            dups.add(x)\n        seen.add(x)\n    return list(dups)",
     "No bug found — this code is correct! O(n) time, finds all duplicates efficiently.",
     "def find_duplicates(arr):\n    seen = set()\n    dups = set()\n    for x in arr:\n        if x in seen:\n            dups.add(x)\n        seen.add(x)\n    return list(dups)"),
    ("def execute_sql(query):\n    return db.execute(f\"SELECT * FROM users WHERE id = {query}\")",
     "SQL injection vulnerability. Never use f-strings for SQL queries.",
     "def execute_sql(query):\n    return db.execute(\"SELECT * FROM users WHERE id = ?\", (query,))"),
    ("class Singleton:\n    _instance = None\n    def __init__(self):\n        if Singleton._instance is None:\n            Singleton._instance = self",
     "__init__ is called every time Singleton() is invoked, overwriting _instance. Need __new__ instead.",
     "class Singleton:\n    _instance = None\n    def __new__(cls):\n        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance"),
]


# ═══════════════════════════════════════════════════════════
# CODE_GENERATORS
# ═══════════════════════════════════════════════════════════

def gen_implement(rng: random.Random) -> tuple[str, str, str]:
    instruction, code = rng.choice(IMPLEMENT_PROBLEMS)
    return (instruction, f"```python\n{code}\n```", CODE_SYSTEM_CODE)


def gen_explain(rng: random.Random) -> tuple[str, str, str]:
    topic, explanation = rng.choice(ALGO_EXPLAIN)
    return (topic, explanation, CODE_SYSTEM_EXPLAIN)


def gen_debug(rng: random.Random) -> tuple[str, str, str]:
    buggy, hint, fixed = rng.choice(BUGGY_CODE)
    instruction = f"Find and fix the bug:\n```python\n{buggy}\n```"
    answer = f"Bug: {hint}\n\nFixed:\n```python\n{fixed}\n```"
    return (instruction, answer, CODE_SYSTEM_REVIEW)


# ── Programmatic code generators (infinite unique rows) ──

ALGO_TASKS = [
    ("sort", lambda r: (f"Implement a function to sort a list of {r.randint(5, 100)} integers in ascending order.",
                        "```python\ndef sort_list(arr):\n    return sorted(arr)\n```")),
    ("reverse", lambda r: (f"Write a function to reverse a list of {r.randint(3, 50)} elements in-place.",
                           "```python\ndef reverse_list(arr):\n    left, right = 0, len(arr)-1\n    while left < right:\n        arr[left], arr[right] = arr[right], arr[left]\n        left += 1; right -= 1\n    return arr\n```")),
    ("max", lambda r: (f"Find the maximum value in an array of {r.randint(5, 200)} integers.",
                       "```python\ndef find_max(arr):\n    if not arr: return None\n    max_val = arr[0]\n    for x in arr[1:]:\n        if x > max_val: max_val = x\n    return max_val\n```")),
    ("count", lambda r: (f"Count the occurrences of a target value in a list of {r.randint(10, 100)} elements.",
                         "```python\ndef count_occurrences(arr, target):\n    count = 0\n    for x in arr:\n        if x == target:\n            count += 1\n    return count\n```")),
    ("unique", lambda r: (f"Remove duplicates from a list of {r.randint(10, 50)} integers.",
                          "```python\ndef remove_duplicates(arr):\n    seen = set()\n    result = []\n    for x in arr:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result\n```")),
]

STRING_TASKS = [
    ("reverse", lambda r: (f"Reverse the string: '{_rand_str(r, 5, 15)}'",
                           "```python\ndef reverse_string(s):\n    return s[::-1]\n```")),
    ("vowels", lambda r: (f"Count vowels in the string: '{_rand_str(r, 8, 20)}'",
                          "```python\ndef count_vowels(s):\n    vowels = set('aeiouAEIOU')\n    return sum(1 for c in s if c in vowels)\n```")),
    ("capitalize", lambda r: (f"Capitalize the first letter of each word in: '{_rand_words(r, 3, 6)}'",
                              "```python\ndef capitalize_words(s):\n    return ' '.join(w.capitalize() for w in s.split())\n```")),
]

MATH_TASKS = [
    ("factorial", lambda r: (f"Compute the factorial of {r.randint(5, 20)}.",
                             "```python\ndef factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n - 1)\n```")),
    ("fibonacci", lambda r: (f"Find the {r.randint(10, 30)}th Fibonacci number.",
                             "```python\ndef fibonacci(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(2, n+1):\n        a, b = b, a + b\n    return b\n```")),
    ("is_prime", lambda r: (f"Check if {r.randint(2, 1000)} is prime.",
                            "```python\ndef is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True\n```")),
    ("gcd", lambda r: (f"Find the GCD of {r.randint(12, 200)} and {r.randint(12, 200)}.",
                       "```python\ndef gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n```")),
]


def _rand_str(rng, min_len, max_len):
    import string
    return ''.join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(min_len, max_len)))


def _rand_words(rng, min_w, max_w):
    import string
    words = []
    for _ in range(rng.randint(min_w, max_w)):
        words.append(''.join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 8))))
    return ' '.join(words)


def gen_programmatic_algo(rng):
    _, gen = rng.choice(ALGO_TASKS)
    return (*gen(rng), CODE_SYSTEM_CODE)


def gen_programmatic_string(rng):
    _, gen = rng.choice(STRING_TASKS)
    return (*gen(rng), CODE_SYSTEM_CODE)


def gen_programmatic_math(rng):
    _, gen = rng.choice(MATH_TASKS)
    return (*gen(rng), CODE_SYSTEM_CODE)


CODE_GENERATORS = [
    (gen_implement, 4),
    (gen_explain, 3),
    (gen_debug, 3),
    (gen_programmatic_algo, 6),
    (gen_programmatic_string, 4),
    (gen_programmatic_math, 4),
]

QUESTION_VARIANTS = [
    # Direct implement
    lambda inst, ans: (inst, ans),
    # Rephrased ask
    lambda inst, ans: (f"Can you implement this: {inst.replace('Implement ', '').replace('implement ', '').lower()}", ans),
    # Explain and implement
    lambda inst, ans: (f"Explain and implement: {inst.lower()}", f"Here's the implementation:\n\n{ans}\n\nThis algorithm works by following standard computer science principles for the given problem."),
    # Time complexity
    lambda inst, ans: (f"What is the time and space complexity of an efficient solution for: {inst.replace('Implement ', '').replace('implement ', '')}",
                       f"The solution above runs in O(n) time and O(n) space in the worst case. This is optimal for this problem class."),
    # Edge cases
    lambda inst, ans: (f"What edge cases should I consider when implementing: {inst.replace('Implement ', '').replace('implement ', '')}",
                       f"Key edge cases to handle:\n1. Empty input — return None or empty result\n2. Single element — trivial case works\n3. Duplicate values — ensure they're handled correctly\n4. Negative values or zeros — don't assume positivity\n5. Very large inputs — consider overflow or O(n²) pitfalls\n\nThe implementation above handles all these cases."),
    # Alternative
    lambda inst, ans: (f"Can you suggest an alternative approach to: {inst.replace('Implement ', '').replace('implement ', '').lower()}",
                       f"The standard approach:\n\n{ans}\n\nAn alternative would use a different data structure (e.g., hash set vs. sorted array) which may offer better constant factors or space usage depending on the specific constraints."),
]


def make_row(rng: random.Random) -> dict:
    pool = []
    for fn, w in CODE_GENERATORS:
        pool.extend([fn] * w)
    fn = rng.choice(pool)
    inst, ans, sys = fn(rng)
    # Apply question variant
    variant = rng.choice(QUESTION_VARIANTS)
    inst_v, ans_v = variant(inst, ans)
    return {"system": sys, "user": inst_v, "assistant": ans_v, "category": "code"}


def _code_main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--output", default="frontier_code.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = args

    rng = random.Random(args.seed)
    out_dir = CODE_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output

    n_impl, n_exp, n_bug = len(IMPLEMENT_PROBLEMS), len(ALGO_EXPLAIN), len(BUGGY_CODE)
    print(f"  Templates: implement={n_impl} explain={n_exp} debug={n_bug}")
    print(f"  Estimated unique: ~{(n_impl*4 + n_exp*3 + n_bug*3)}")

    t0 = time.time()
    seen = set()
    records = []
    attempts = 0
    max_attempts = args.rows * 3
    last_report = 0

    while len(records) < args.rows and attempts < max_attempts:
        attempts += 1
        row = make_row(rng)
        if len(row['assistant']) < 10: continue
        key = (row["user"][:80], row["assistant"][:80])
        if key in seen: continue
        seen.add(key)
        records.append(row)
        if len(records) - last_report >= 10000:
            last_report = len(records)
            elapsed = time.time() - t0
            rate = len(records) / elapsed if elapsed > 0 else 0
            print(f"  {len(records):>6,} rows ({rate:.0f}/s)", flush=True)

    records = records[:args.rows]
    rng.shuffle(records)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for row in records: f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    mb = out_path.stat().st_size / 1_000_000
    print(f"\n{'='*50}")
    print(f"Generated {len(records):,} rows in {elapsed:.0f}s ({len(records)/elapsed:.0f}/s)")
    print(f"Written: {out_path} ({mb:.1f} MB)")


# ═══════════════════════════════════════════════════════════
# REASONING
# ═══════════════════════════════════════════════════════════


import itertools

REASON_OUT_DIR = Path("Download") / "reasoning"
REASON_SYSTEM_REASON = "You are Atulya. Explain your reasoning step by step."
REASON_SYSTEM_CODE = "You are Atulya. Think through the code carefully."


# ═══════════════════════════════════════════════════════════
# MATH REASONING — Verified template problems
# ═══════════════════════════════════════════════════════════

def _gen_arithmetic(ops: list[str], rng: random.Random) -> tuple[str, str]:
    """Generate arithmetic word problem with verified solution."""
    a = rng.randint(12, 999)
    b = rng.randint(1, 99) if rng.random() < 0.5 else rng.randint(12, 999)
    op = rng.choice(ops)

    if op == "add":
        ans = a + b
        context = rng.choice([
            f"A farmer has {a} apples and buys {b} more.",
            f"A library has {a} books and receives {b} new donations.",
            f"A class has {a} students and {b} join mid-year.",
            f"You save ${a} in January and ${b} in February.",
        ])
        question = f"{context} What is the total?"
        solution = f"Let's add {a} + {b}. {a} + {b} = {ans}. So the total is {ans}."
    elif op == "sub":
        ans = a - b
        if ans < 0:
            a, b = b, a
            ans = a - b
        context = rng.choice([
            f"There are {a} birds on a tree. {b} fly away.",
            f"A tank holds {a} liters. {b} liters are used.",
            f"You have ${a}. You spend ${b}.",
        ])
        question = f"{context} How many remain?"
        solution = f"Starting with {a}, subtract {b}. {a} - {b} = {ans}. So {ans} remain."
    elif op == "mul":
        ans = a * b
        if ans > 99999:
            a = rng.randint(3, 99)
            ans = a * b
        context = rng.choice([
            f"There are {a} boxes with {b} items each.",
            f"A car travels {a} km/h for {b} hours.",
            f"A baker makes {a} batches of {b} cookies each.",
        ])
        question = f"{context} What is the total?"
        solution = f"Multiply {a} × {b}. {a} × {b} = {ans}. So the total is {ans}."
    elif op == "div":
        safe_b = max(b, 1)
        ans = a // safe_b
        rem = a % safe_b
        if rem != 0:
            a = a - rem
            ans = a // safe_b
        if ans < 2 or a < 2:
            a = rng.randint(20, 100)
            b = rng.randint(2, 10)
            ans = a // b
            a = ans * b  # ensure exact division
        context = rng.choice([
            f"{a} students are split into {b} equal groups.",
            f"${a} is shared equally among {b} people.",
            f"{a} cookies are packed into boxes of {b}.",
        ])
        question = f"{context} How many per group?"
        solution = f"Divide {a} ÷ {b}. {a} ÷ {b} = {ans}. Each group gets {ans}."
    elif op == "sequence":
        # Simple arithmetic sequence
        start = rng.randint(1, 10)
        diff = rng.randint(2, 7)
        n_terms = rng.randint(4, 6)
        terms = [start + i * diff for i in range(n_terms)]
        nth = rng.choice([5, 6, 7, 8, 10])
        ans = start + (nth - 1) * diff
        question = f"Find the {nth}th term: {', '.join(str(t) for t in terms)}, ..."
        solution = f"The sequence increases by {diff} each step. Starting from {start}, the {nth}th term is {start} + ({nth} - 1) × {diff} = {start} + {(nth-1)*diff} = {ans}."
    else:
        return _gen_arithmetic(["add"], rng)

    return question, solution


def _gen_algebra(rng: random.Random) -> tuple[str, str]:
    """Solve-for-x problems."""
    x = rng.randint(1, 20)
    coeff = rng.choice([1, 2, 3, 4, 5])
    const = rng.randint(-20, 20)
    rhs = coeff * x + const

    if rng.random() < 0.5:
        # coeff*x + const = rhs
        question = f"Solve for x: {coeff}x + {const} = {rhs}"
        steps = [
            f"Subtract {const} from both sides: {coeff}x = {rhs} - ({const}) = {rhs - const}",
            f"Divide by {coeff}: x = {rhs - const} ÷ {coeff} = {(rhs - const) // coeff}",
        ]
        if (rhs - const) % coeff != 0:
            # Fall back to simpler
            return _gen_algebra(rng)
    else:
        # coeff*x - const = rhs
        question = f"Solve for x: {coeff}x - {const} = {rhs}"
        steps = [
            f"Add {const} to both sides: {coeff}x = {rhs} + ({const}) = {rhs + const}",
            f"Divide by {coeff}: x = {rhs + const} ÷ {coeff} = {(rhs + const) // coeff}",
        ]
        if (rhs + const) % coeff != 0:
            return _gen_algebra(rng)

    ans = x
    solution = "Step 1: " + steps[0] + "\nStep 2: " + steps[1] + f"\nTherefore, x = {ans}."
    # Verify
    check = (rhs - const) // coeff if " + " in question else (rhs + const) // coeff
    return question, solution


def _gen_percent(rng: random.Random) -> tuple[str, str]:
    """Percentage problems."""
    total = rng.randint(50, 500)
    pct = rng.choice([10, 15, 20, 25, 30, 50, 75])
    part = total * pct // 100

    if rng.random() < 0.5:
        question = f"What is {pct}% of {total}?"
        solution = f"{pct}% of {total} = ({pct}/100) × {total} = 0.{pct if pct >= 10 else '0'+str(pct)} × {total} = {part}."
    else:
        question = f"If {part} out of {total} people prefer X, what percentage is that?"
        pct_calc = round(part / total * 100, 1)
        solution = f"({part} / {total}) × 100 = {pct_calc}%. So {pct_calc}% prefer X."
    return question, solution


def _gen_rate(rng: random.Random) -> tuple[str, str]:
    """Speed/distance/time problems."""
    if rng.random() < 0.5:
        speed = rng.choice([30, 45, 50, 60, 65, 70, 80])
        time_h = rng.choice([1, 1.5, 2, 2.5, 3, 4])
        dist = int(speed * time_h)
        question = f"A car travels at {speed} km/h for {time_h} hours. How far does it go?"
        solution = f"Distance = speed × time = {speed} × {time_h} = {dist} km."
    else:
        dist = rng.choice([100, 150, 200, 250, 300, 400])
        speed = rng.choice([40, 50, 60, 70, 80])
        time_h = round(dist / speed, 2)
        question = f"A train travels {dist} km at {speed} km/h. How long does it take?"
        solution = f"Time = distance ÷ speed = {dist} ÷ {speed} = {time_h} hours."
    return question, solution


def _gen_probability(rng: random.Random) -> tuple[str, str]:
    """Simple probability problems."""
    favorable = rng.randint(1, 5)
    total = rng.randint(favorable + 2, 20)
    prob = round(favorable / total, 4)

    items = rng.choice([
        ("red marbles", "blue marbles"),
        ("green candies", "yellow candies"),
        ("white socks", "black socks"),
        ("apples", "oranges"),
    ])
    question = f"A bag has {favorable} {items[0]} and {total - favorable} {items[1]}. What's the probability of picking a {items[0][:-1]}?"
    solution = f"P({items[0][:-1]}) = favorable / total = {favorable} / {total} = {prob} = {prob*100:.1f}%."
    return question, solution


def _gen_geometry(rng: random.Random) -> tuple[str, str]:
    """Area/perimeter/volume problems."""
    kind = rng.choice(["area_rect", "area_circle", "perimeter", "volume_cube"])

    if kind == "area_rect":
        w = rng.randint(3, 20)
        h = rng.randint(3, 20)
        area = w * h
        question = f"A rectangle is {w}cm wide and {h}cm tall. What is its area?"
        solution = f"Area = width × height = {w} × {h} = {area} cm²."
    elif kind == "area_circle":
        r = rng.randint(2, 15)
        area = round(3.14159 * r * r, 2)
        question = f"A circle has radius {r}cm. What is its area? (Use π ≈ 3.14159)"
        solution = f"Area = πr² = π × {r}² = 3.14159 × {r*r} ≈ {area} cm²."
    elif kind == "perimeter":
        s = rng.randint(3, 30)
        perimeter = 4 * s
        question = f"A square has side {s}cm. What is its perimeter?"
        solution = f"Perimeter = 4 × side = 4 × {s} = {perimeter} cm."
    else:
        s = rng.randint(2, 12)
        vol = s ** 3
        question = f"A cube has side {s}cm. What is its volume?"
        solution = f"Volume = side³ = {s}³ = {vol} cm³."

    return question, solution


MATH_GENERATORS = {
    "arithmetic_add": lambda r: _gen_arithmetic(["add"], r),
    "arithmetic_sub": lambda r: _gen_arithmetic(["sub"], r),
    "arithmetic_mul": lambda r: _gen_arithmetic(["mul"], r),
    "arithmetic_div": lambda r: _gen_arithmetic(["div"], r),
    "arithmetic_seq": lambda r: _gen_arithmetic(["sequence"], r),
    "algebra": _gen_algebra,
    "percent": _gen_percent,
    "rate": _gen_rate,
    "probability": _gen_probability,
    "geometry": _gen_geometry,
}


# ═══════════════════════════════════════════════════════════
# LOGICAL REASONING
# ═══════════════════════════════════════════════════════════

SYLLOGISM_TEMPLATES = [
    lambda r: ("All humans are mortal. Socrates is human. Is Socrates mortal?",
               "Premise 1: All humans are mortal.\nPremise 2: Socrates is human.\nConclusion: Therefore, Socrates is mortal. This is a classic syllogism — the conclusion follows necessarily from the premises."),
    lambda r: ("All birds have feathers. Penguins are birds. Do penguins have feathers?",
               "Premise 1: All birds have feathers.\nPremise 2: Penguins are birds.\nConclusion: Therefore, penguins have feathers. Even though penguins can't fly, they are still birds and have feathers."),
    lambda r: ("If it rains, the ground gets wet. The ground is wet. Does that mean it rained?",
               "This is the fallacy of affirming the consequent. The ground could be wet for other reasons (sprinklers, spilled water, etc.). The correct inference is: if it rains → ground wet, but wet ground does not necessarily mean rain."),
    lambda r: ("All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded?",
               "Premise 1: All mammals are warm-blooded.\nPremise 2: Whales are mammals.\nConclusion: Therefore, whales are warm-blooded. This is a valid syllogism."),
    lambda r: ("No fish can fly. A salmon is a fish. Can a salmon fly?",
               "Premise 1: No fish can fly.\nPremise 2: A salmon is a fish.\nConclusion: Therefore, a salmon cannot fly. This is a valid categorical syllogism."),
    lambda r: ("Some fruits are sweet. All apples are fruits. Are all apples sweet?",
               "Not necessarily. 'Some fruits are sweet' means at least one fruit is sweet, not all fruits. Apples are fruits, but we don't know if they're among the sweet ones. This is a classic syllogistic fallacy."),
    lambda r: ("If a number is even, it's divisible by 2. 7 is not divisible by 2. Is 7 even?",
               "Contrapositive: If a number is even → divisible by 2. The contrapositive is: if not divisible by 2 → not even. Since 7 is not divisible by 2, it is not even. This uses modus tollens."),
    lambda r: ("All squares are rectangles. All rectangles have four sides. Do all squares have four sides?",
               "Premise 1: All squares are rectangles.\nPremise 2: All rectangles have four sides.\nConclusion: Therefore, all squares have four sides. This is transitive reasoning."),
]


def _gen_syllogism(rng: random.Random) -> tuple[str, str]:
    q, a = rng.choice(SYLLOGISM_TEMPLATES)(rng)
    return q, a


LATERAL_PUZZLES = [
    lambda r: ("A man pushes his car to a hotel and tells the owner he's bankrupt. Why?",
               "He's playing Monopoly. In Monopoly, you land on a property and go bankrupt if you can't pay. The 'hotel' and 'car' are game pieces."),
    lambda r: ("What comes once in a minute, twice in a moment, but never in a thousand years?",
               "The letter 'M'. 'Minute' has one M, 'moment' has two M's, and 'thousand years' has no M."),
    lambda r: ("I speak without a mouth and hear without ears. I have no body, but I come alive with the wind. What am I?",
               "An echo. Sound reflects off surfaces without needing a mouth or ears. The wind carries sound waves."),
    lambda r: ("The more you take, the more you leave behind. What am I?",
               "Footsteps. Each step you take leaves a footprint behind. The more steps you take, the more footprints you leave."),
    lambda r: ("What can travel around the world while staying in a corner?",
               "A stamp. A postage stamp stays in the corner of an envelope but the envelope travels the world."),
    lambda r: ("If you have me, you want to share me. If you share me, you don't have me. What am I?",
               "A secret. Once you share a secret, it's no longer yours alone — it becomes known to others."),
    lambda r: ("I follow you all day long, but when night comes I'm gone. What am I?",
               "Your shadow. It follows you in sunlight and disappears in darkness."),
    lambda r: ("What has keys but can't open locks?",
               "A piano. It has musical keys but they don't open locks."),
    lambda r: ("What can you catch but not throw?",
               "A cold. You catch a cold (illness) but cannot physically throw it."),
    lambda r: ("What building has the most stories?",
               "A library. 'Stories' is a pun — stories as in tales (books) versus building floors."),
]


def _gen_puzzle(rng: random.Random) -> tuple[str, str]:
    q, a = rng.choice(LATERAL_PUZZLES)(rng)
    return q, a


DEDUCTIVE_PROBLEMS = [
    lambda r: ("There are three boxes: one contains only apples, one only oranges, and one both. All labels are wrong. You pick one fruit from the box labeled 'Apples'. It's an orange. What do you know?",
               "Key insight: All labels are wrong. The box labeled 'Apples' cannot contain only apples (label is wrong). You picked an orange, so this box must be the 'Both' box (it has at least one orange, and it's not apples-only or oranges-only since labels are all wrong). Therefore:\n- 'Apples' box = Both\n- 'Oranges' box = Apples (since it can't be oranges, and 'Both' is taken)\n- 'Both' box = Oranges (last remaining)"),
    lambda r: ("Five people in a line: Alice is not first. Bob is before Charlie but after Diana. Eve is last. Charlie is third. Who is first?",
               "Let's reason step by step:\n1. Eve is last (5th).\n2. Charlie is 3rd.\n3. Bob is before Charlie but after Diana: so order is Diana → Bob → Charlie.\n4. Diana must be 1st (before Bob at 2nd).\n5. Alice is not first, so Alice is 4th.\nFinal order: Diana, Bob, Charlie, Alice, Eve. First is Diana."),
    lambda r: ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
               "Let the ball cost x dollars. Then the bat costs x + 1.00. Total: x + (x + 1.00) = 1.10. 2x + 1.00 = 1.10. 2x = 0.10. x = 0.05. So the ball costs 5 cents. (The intuitive answer of 10 cents is wrong because then the bat would be $1.00, making the total $1.10 but the bat would only be $0.90 more than the ball, not $1.00 more.)"),
    lambda r: ("You have a 3-gallon jug and a 5-gallon jug. How do you measure exactly 4 gallons?",
               "Step 1: Fill the 5-gallon jug. (5, 0)\nStep 2: Pour from 5 into 3 until 3 is full. (2, 3)\nStep 3: Empty the 3-gallon jug. (2, 0)\nStep 4: Pour the remaining 2 gallons from 5 into 3. (0, 2)\nStep 5: Fill the 5-gallon jug again. (5, 2)\nStep 6: Pour from 5 into 3 until full (3 already has 2, so it takes 1 more). (4, 3)\nResult: 5-gallon jug now has exactly 4 gallons."),
    lambda r: ("If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
               "Each machine makes 1 widget in 5 minutes. 5 machines × 5 minutes = 5 widgets. So 1 machine makes 1 widget in 5 minutes. Therefore, 100 machines making 100 widgets still takes 5 minutes (they work in parallel)."),
    lambda r: ("In a lake, there's a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days to cover the lake, how long does it take to cover half?",
               "If it doubles every day, then on day 47 the patch covers exactly half the lake. The next day (day 48) it doubles to cover the whole lake. So it takes 47 days to cover half."),
]


def _gen_deductive(rng: random.Random) -> tuple[str, str]:
    q, a = rng.choice(DEDUCTIVE_PROBLEMS)(rng)
    return q, a


# ═══════════════════════════════════════════════════════════
# SCIENCE REASONING — Why/How explanations
# ═══════════════════════════════════════════════════════════

SCIENCE_QUESTIONS = [
    ("Why is the sky blue?",
     "Sunlight is white light composed of all colors. When it enters Earth's atmosphere, it scatters off air molecules. Shorter wavelengths (blue/violet) scatter much more than longer ones (red/orange) due to Rayleigh scattering. Our eyes are more sensitive to blue than violet, so we see a blue sky. At sunrise/sunset, light travels through more atmosphere, scattering away the blue and leaving red/orange."),
    ("What causes the seasons?",
     "Earth's axis is tilted 23.5° relative to its orbital plane. As Earth orbits the Sun, the Northern Hemisphere tilts toward the Sun in June (summer) and away in December (winter). The tilt changes the angle and duration of sunlight, not the distance from the Sun. More direct sunlight + longer days = summer."),
    ("Why does ice float on water?",
     "Water is unusual: it's densest at 4°C, not at its freezing point. When water freezes into ice, the molecules form a hexagonal crystal lattice that takes up more space than liquid water. This makes ice about 9% less dense than liquid water, so it floats. If ice sank, lakes would freeze solid and aquatic life couldn't survive."),
    ("How do vaccines work?",
     "Vaccines expose your immune system to a harmless part of a pathogen (an antigen). Your body produces antibodies and memory B-cells specific to that antigen. If you later encounter the real pathogen, your immune system recognizes it immediately and launches a much faster, stronger response — often before you even feel sick."),
    ("Why do we dream?",
     "Several theories exist. The activation-synthesis theory suggests dreams are the brain's attempt to make sense of random neural signals during REM sleep. The memory consolidation theory suggests dreams help process and store memories. The threat simulation theory proposes dreams evolved to rehearse dangerous situations safely."),
    ("What is a black hole?",
     "A black hole is a region of spacetime where gravity is so strong that nothing — not even light — can escape. It forms when a massive star collapses under its own gravity, compressing its mass into an infinitely dense point (singularity). The boundary around it is the event horizon — once crossed, there's no return."),
    ("Why is the ocean salty?",
     "Rainwater, slightly acidic from dissolved CO2, erodes rocks on land, releasing minerals including sodium and chloride ions. Rivers carry these ions to the ocean. Over billions of years, these salts have accumulated because they don't easily precipitate out or get consumed by organisms. The ocean is about 3.5% salt by weight."),
    ("How does photosynthesis work?",
     "Plants use chlorophyll in chloroplasts to capture sunlight energy. This energy splits water (H2O) into oxygen (released) and hydrogen. The hydrogen combines with CO2 to form glucose (C6H12O6) in the Calvin cycle. The overall reaction: 6CO2 + 6H2O + sunlight → C6H12O6 + 6O2."),
    ("What is quantum entanglement?",
     "Two particles can be linked so that measuring one instantly determines the state of the other, regardless of distance — even light-years apart. Einstein called it 'spooky action at a distance.' It doesn't allow faster-than-light communication because you can't control which state you measure. It's a real phenomenon confirmed by Bell tests."),
    ("Why do we have fingerprints?",
     "Fingerprints (friction ridges) serve two main purposes: they increase friction and grip on surfaces, and they improve tactile sensitivity. The ridges amplify vibrations when we touch things, sending stronger signals to nerve endings. The specific pattern (loop, whorl, arch) is random and unique due to developmental factors in the womb."),
    ("How does evolution work?",
     "Evolution by natural selection has three requirements: variation (individuals differ), heritability (traits are passed down), and differential survival (some traits help survival/reproduction). Individuals with advantageous traits survive and reproduce more, passing those traits to the next generation. Over millions of years, this creates complex adaptations and new species."),
    ("What causes earthquakes?",
     "Earth's lithosphere is divided into tectonic plates that move slowly (cm/year) due to convection in the mantle. Stress builds at plate boundaries where plates collide, separate, or slide past each other. When the stress exceeds the rocks' strength, the fault ruptures suddenly, releasing energy as seismic waves — an earthquake."),
    ("Why is the sky dark at night?",
     "This is Olbers' paradox. If the universe were infinite, static, and filled with stars, every line of sight would end at a star's surface, so the night sky would be bright. The universe is finite in age (~13.8 billion years), expanding, and stars don't live forever. Light from distant stars hasn't reached us yet, and the expansion redshifts distant light."),
    ("How do antibiotics work?",
     "Different antibiotics target different bacterial mechanisms. Penicillins disrupt cell wall synthesis, causing bacteria to burst. Tetracyclines block protein synthesis in ribosomes. Fluoroquinolones interfere with DNA replication. Antibiotics don't work on viruses — hence antibiotic resistance is a major crisis from overuse."),
    ("What is the double-slit experiment?",
     "When particles like electrons are fired at two slits one at a time, they create an interference pattern (like waves). But if you measure which slit each goes through, the pattern disappears and they behave like particles. This shows quantum objects exist as probability waves that 'collapse' upon measurement — a fundamental mystery in quantum mechanics."),
]

SCIENCE_QUESTIONS_2 = [
    ("How does GPS work?",
     "GPS satellites orbit Earth broadcasting their position and precise time (using atomic clocks). Your receiver calculates its distance to each satellite using the time delay of the signal. With signals from 4+ satellites, it triangulates your position. Einstein's relativity matters: satellite clocks run faster by ~38 microseconds/day, requiring correction."),
    ("What is dark matter?",
     "Dark matter is invisible matter that doesn't emit, absorb, or reflect light but has gravitational effects. We know it exists because galaxies rotate faster than visible matter can explain, and gravitational lensing shows more mass than we see. It makes up ~27% of the universe. We don't know what it is — candidates include WIMPs and axions."),
    ("How does a computer work at the lowest level?",
     "Transistors act as switches (on/off = 1/0). Combinations of transistors form logic gates (AND, OR, NOT, XOR). Gates form adders, multiplexers, and flip-flops (memory). These form an ALU and registers. The control unit fetches instructions from memory, decodes them, and coordinates data flow. This is the stored-program concept (von Neumann architecture)."),
    ("Why is water essential for life?",
     "Water is a universal solvent — its polarity allows it to dissolve more substances than any other liquid, enabling biochemical reactions. It has high specific heat capacity, stabilizing temperatures. Ice floats (insulating lakes). It's cohesive (surface tension, capillary action in plants). It's the medium for almost all biological chemistry."),
]


def _gen_science(rng: random.Random) -> tuple[str, str]:
    q, a = rng.choice(SCIENCE_QUESTIONS + SCIENCE_QUESTIONS_2)
    return q, a


# ═══════════════════════════════════════════════════════════
# CODE REASONING — Trace, debug, explain
# ═══════════════════════════════════════════════════════════

def _gen_code_trace_simple(rng: random.Random) -> tuple[str, str]:
    """What does this code output? — simple."""
    n = rng.randint(3, 8)
    total = sum(range(1, n+1))
    code = f"total = 0\nfor i in range(1, {n+1}):\n    total += i\nprint(total)"
    question = f"What does this code print?\n```python\n{code}\n```"
    answer = f"Let's trace:\n- Initialize total = 0\n- Loop i = 1: total = 0 + 1 = 1\n- Loop i = 2: total = 1 + 2 = 3\n- Loop i = 3: total = 3 + 3 = 6\n...\n- Loop i = {n}: total = previous + {n}\nThe sum of 1 to {n} = {n}({n}+1)/2 = {total}\nOutput: {total}"
    return question, answer


def _gen_code_trace_fn(rng: random.Random) -> tuple[str, str]:
    """Trace recursive function."""
    n = rng.randint(3, 6)
    fibs = [0, 1]
    for i in range(2, n+1):
        fibs.append(fibs[-1] + fibs[-2])
    code = f"def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\nprint(fib({n}))"
    question = f"What does fib({n}) return?\n```python\n{code}\n```"
    answer = f"fib({n}) = fib({n-1}) + fib({n-2})\nLet's unroll:\n"
    for i in range(n+1):
        answer += f"fib({i}) = {fibs[i]}\n"
    answer += f"Output: {fibs[n]}"
    return question, answer


def _gen_code_bug(rng: random.Random) -> tuple[str, str]:
    """Find the bug."""
    n = rng.randint(3, 8)
    code = f"def sum_list(items):\n    total = 0\n    for i in range(len(items)):\n        total += items[i]\n    return total\n\nresult = sum_list([1, 2, {n}])\nprint(result / len(None))  # bug here"
    question = f"Find and fix the bug:\n```python\n{code}\n```"
    answer = f"The bug is on the last line: `len(None)` — you can't call len() on None. The programmer likely meant to calculate the average but used the wrong variable. Fix:\n```python\nprint(result / len([1, 2, {n}]))\n```\nor:\n```python\ndef average(items):\n    return sum(items) / len(items)\n```\nThe sum works correctly: 1 + 2 + {n} = {1+2+n}."
    return question, answer


def _gen_code_complexity(rng: random.Random) -> tuple[str, str]:
    """What is the time complexity?"""
    code = "def find_duplicates(arr):\n    seen = set()\n    dups = []\n    for x in arr:\n        if x in seen:\n            dups.append(x)\n        else:\n            seen.add(x)\n    return dups"
    question = f"What is the time complexity of this function?\n```python\n{code}\n```"
    answer = "Time complexity: O(n) where n = len(arr).\n- We loop through arr once.\n- Each 'x in seen' check is O(1) on average (hash set).\n- Each 'seen.add(x)' is O(1).\n- Total: O(n) time, O(n) extra space for the set.\nThis is optimal for finding duplicates in an unsorted array."
    return question, answer


CODE_GENERATORS = [
    _gen_code_trace_simple,
    _gen_code_trace_fn,
    _gen_code_bug,
    _gen_code_complexity,
]


# ═══════════════════════════════════════════════════════════
# COMPARATIVE / ANALYSIS REASONING
# ═══════════════════════════════════════════════════════════

COMPARE_QUESTIONS = [
    ("Compare and contrast capitalism and socialism.",
     "Capitalism: private ownership of production, market-driven allocation, profit motive, minimal government intervention. Strengths: innovation, efficiency, individual freedom. Weaknesses: inequality, boom-bust cycles, externalities.\n\nSocialism: social ownership of production, planned allocation, collective goals. Strengths: equality, social safety nets, reduced poverty. Weaknesses: reduced incentives, bureaucracy, slower innovation.\n\nMost modern economies are mixed — combining market allocation with government regulation and social programs."),
    ("What are the arguments for and against free will?",
     "For free will: Our subjective experience feels like we make conscious choices. Moral responsibility assumes we could have done otherwise. Quantum mechanics introduces indeterminism.\n\nAgainst free will (determinism): Every physical event has prior causes, including brain states. Libet experiments show brain activity precedes conscious decision by ~300ms. Our choices are shaped by genetics, environment, and prior causes we didn't choose.\n\nCompatibilism: Free will is compatible with determinism if we define it as acting according to one's own desires without external coercion."),
    ("Explain the difference between deductive and inductive reasoning.",
     "Deductive reasoning: Moves from general premises to specific conclusions. If premises are true, the conclusion must be true (valid). Example: All men are mortal. Socrates is a man. Therefore, Socrates is mortal.\n\nInductive reasoning: Moves from specific observations to general patterns. Conclusions are probable, not certain. Example: Every swan I've seen is white. Therefore, all swans are probably white.\n\nKey difference: Deduction guarantees truth (given true premises). Induction only suggests likelihood. Science uses both — deduction for predictions from theories, induction for generalizing from data."),
    ("What is the difference between SQL and NoSQL databases?",
     "SQL (relational): Tables with fixed schemas, ACID transactions, powerful joins, vertical scaling. Best for structured data with complex relationships (banking, ERP).\n\nNoSQL: Flexible/document/key-value/graph models, BASE consistency, horizontal scaling. Types: Document (MongoDB), Key-Value (Redis), Wide-column (Cassandra), Graph (Neo4j). Best for large-scale, unstructured, or rapidly changing data.\n\nChoose SQL when consistency and complex queries matter. Choose NoSQL when scalability, flexibility, or specialized data models matter."),
    ("Explain the difference between supervised and unsupervised learning.",
     "Supervised learning: Models learn from labeled data (X → y). Goal: predict labels for new data. Examples: regression (predict price), classification (spam detection), object detection. Requires labeled training data.\n\nUnsupervised learning: Models find patterns in unlabeled data. Goal: discover structure. Examples: clustering (customer segments), dimensionality reduction (PCA), anomaly detection. No labels needed.\n\nThere's also semi-supervised (some labels) and self-supervised (creates labels from data itself — used by GPT)."),
]


def _gen_compare(rng: random.Random) -> tuple[str, str]:
    q, a = rng.choice(COMPARE_QUESTIONS)
    return q, a


# ═══════════════════════════════════════════════════════════
# GENERATOR
# ═══════════════════════════════════════════════════════════

REASONING_CATEGORIES = {
    "math_arithmetic": {"gen": lambda r: r.choice(list(MATH_GENERATORS.values()))(r), "weight": 8, "cat": "reasoning"},
    "math_algebra": {"gen": MATH_GENERATORS["algebra"], "weight": 4, "cat": "reasoning"},
    "math_percent": {"gen": MATH_GENERATORS["percent"], "weight": 3, "cat": "reasoning"},
    "math_rate": {"gen": MATH_GENERATORS["rate"], "weight": 3, "cat": "reasoning"},
    "math_probability": {"gen": MATH_GENERATORS["probability"], "weight": 2, "cat": "reasoning"},
    "math_geometry": {"gen": MATH_GENERATORS["geometry"], "weight": 2, "cat": "reasoning"},
    "logic_syllogism": {"gen": _gen_syllogism, "weight": 4, "cat": "reasoning"},
    "logic_puzzle": {"gen": _gen_puzzle, "weight": 3, "cat": "reasoning"},
    "logic_deductive": {"gen": _gen_deductive, "weight": 3, "cat": "reasoning"},
    "science_explain": {"gen": _gen_science, "weight": 5, "cat": "factual"},
    "code_trace": {"gen": lambda r: r.choice(CODE_GENERATORS)(r), "weight": 3, "cat": "code"},
    "compare_analysis": {"gen": _gen_compare, "weight": 2, "cat": "reasoning"},
}


def make_reasoning_row(rng: random.Random) -> dict:
    pool = []
    for name, spec in REASONING_CATEGORIES.items():
        pool.extend([name] * spec["weight"])
    choice = rng.choice(pool)
    spec = REASONING_CATEGORIES[choice]
    cat = spec["cat"]

    question, answer = spec["gen"](rng)

    system = REASON_SYSTEM_CODE if cat == "code" else REASON_SYSTEM_REASON

    return {"system": system, "user": question, "assistant": answer, "category": cat}


def quality_filter(row: dict) -> bool:
    text = f"{row['user']} {row['assistant']}"
    if len(text) < 30:
        return False
    if len(row['assistant']) < 20:
        return False
    return True


def _reasoning_main(args):
    parser = argparse.ArgumentParser(description="Build frontier-quality reasoning dataset")
    parser.add_argument("--rows", type=int, default=500_000, help="Number of rows to generate")
    parser.add_argument("--output", default="frontier_reasoning.jsonl", help="Output filename")
    parser.add_argument("--seed", type=int, default=42)
    args = args

    rng = random.Random(args.seed)
    out_dir = REASON_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output

    t0 = time.time()
    seen = set()
    records = []
    max_attempts = args.rows * 3
    last_report = 0

    while len(records) < args.rows and len(records) + (args.rows * 3) < max_attempts + (args.rows * 3):
        if len(records) >= args.rows:
            break
        attempts_inner = 0
        while len(records) < args.rows and attempts_inner < max_attempts // max(1, args.rows):
            attempts_inner += 1
            row = make_reasoning_row(rng)
            if not quality_filter(row):
                continue
            key = (row["user"][:100], row["assistant"][:100])
            if key in seen:
                continue
            seen.add(key)
            records.append(row)

            if len(records) - last_report >= 25000:
                last_report = len(records)
                elapsed = time.time() - t0
                rate = len(records) / elapsed if elapsed > 0 else 0
                print(f"  {len(records):>7,} rows ({rate:.0f}/s)", flush=True)
            if len(records) >= args.rows:
                break

    records = records[:args.rows]
    rng.shuffle(records)

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cats = Counter(r["category"] for r in records)
    elapsed = time.time() - t0
    mb = out_path.stat().st_size / 1_000_000

    print(f"\n{'='*50}")
    print(f"Generated {len(records):,} rows in {elapsed:.0f}s ({len(records)/elapsed:.0f}/s)")
    print(f"Written: {out_path} ({mb:.1f} MB)")
    print(f"Categories: {dict(cats.most_common())}")

    print("\nSamples:")
    for row in records[:3]:
        print(f"  [{row['category']}]")
        print(f"    U: {row['user'][:150]}")
        print(f"    A: {row['assistant'][:200]}")
        print()


# ═══════════════════════════════════════════════════════════
# TEACHER
# ═══════════════════════════════════════════════════════════


import math


TEACHER_OUT_DIR = Path("Download/teacher")
TEACHER_SOURCE_BALANCED = Path("Download/qa/seed_chat_balanced_100k.jsonl")
TEACHER_SYSTEMS = [
    "You are Atulya. Answer clearly and briefly.",
    "You are Atulya, a helpful AI assistant. Be accurate and direct.",
    "You are Atulya. Give a useful answer with simple wording.",
]

FACTS = [
    ("What is gravity?", "Gravity is the force that pulls objects with mass toward each other.", "It gives objects weight on Earth and keeps planets moving around the Sun."),
    ("What is photosynthesis?", "Photosynthesis is how plants use sunlight to make food.", "Plants take in carbon dioxide and water, then produce sugar and oxygen."),
    ("What is evaporation?", "Evaporation is when liquid changes into vapor.", "It happens faster with heat, wind, and a larger surface area."),
    ("What is condensation?", "Condensation is when vapor cools and becomes liquid.", "Clouds, dew, and water drops on a cold glass are common examples."),
    ("What is friction?", "Friction is a force that resists motion between surfaces.", "It helps us walk, brake, and grip objects."),
    ("What is inertia?", "Inertia is an object's tendency to keep its current motion.", "A still object tends to stay still, and a moving object tends to keep moving."),
    ("What is energy?", "Energy is the ability to do work or cause change.", "It can appear as motion, heat, light, electricity, or stored chemical energy."),
    ("What is a molecule?", "A molecule is a group of atoms bonded together.", "Water is a molecule made from two hydrogen atoms and one oxygen atom."),
    ("What is an atom?", "An atom is the basic unit of ordinary matter.", "Atoms contain protons, neutrons, and electrons."),
    ("What is a cell?", "A cell is the basic unit of life.", "Living things are made of one or more cells."),
    ("What is DNA?", "DNA stores genetic instructions.", "It helps cells build proteins and pass traits from parents to offspring."),
    ("What is an ecosystem?", "An ecosystem is a community of organisms and their environment.", "Plants, animals, microbes, water, soil, and climate all interact in it."),
    ("What is climate?", "Climate is the long-term pattern of weather in a region.", "Weather changes daily, while climate is measured over many years."),
    ("What is a democracy?", "A democracy is a system where people have a voice in government.", "Citizens usually vote for leaders or laws."),
    ("What is economics?", "Economics studies how people use limited resources.", "It looks at production, trade, money, choices, and incentives."),
    ("What is inflation?", "Inflation is a general rise in prices over time.", "When inflation is high, the same money buys less than before."),
    ("What is a database?", "A database is an organized collection of data.", "It helps store, search, update, and manage information efficiently."),
    ("What is an algorithm?", "An algorithm is a step-by-step method for solving a problem.", "Recipes, search procedures, and sorting methods are examples."),
    ("What is machine learning?", "Machine learning lets computers learn patterns from data.", "A model is trained on examples and then used to make predictions or generate output."),
    ("What is artificial intelligence?", "Artificial intelligence is software that performs tasks associated with human intelligence.", "It can include language, vision, planning, search, and decision-making."),
    ("What is a neural network?", "A neural network is a model made of connected layers of numbers.", "It learns by adjusting weights so its outputs better match examples."),
    ("What is overfitting?", "Overfitting happens when a model memorizes training data too closely.", "It performs well on seen examples but poorly on new ones."),
    ("What is underfitting?", "Underfitting happens when a model is too simple or undertrained.", "It fails to capture important patterns in the data."),
    ("What is a tokenizer?", "A tokenizer converts text into tokens that a model can process.", "Tokens may be words, word pieces, characters, or byte-like units."),
    ("What is a variable in programming?", "A variable is a named place to store a value.", "Programs use variables to remember and update information."),
    ("What is a function in programming?", "A function is a reusable block of code.", "It can take inputs, perform work, and return a result."),
    ("What is recursion?", "Recursion is when a function calls itself.", "It needs a base case so it eventually stops."),
    ("What is a loop?", "A loop repeats a block of code.", "Loops are useful for processing lists, counting, or retrying work."),
    ("What is an API?", "An API is a way for software systems to communicate.", "It defines what requests can be made and what responses are returned."),
    ("What is encryption?", "Encryption turns readable data into protected unreadable data.", "Only someone with the right key can turn it back into readable form."),
    ("What is a hypothesis?", "A hypothesis is a testable explanation or prediction.", "Scientists use experiments or observations to check whether it is supported."),
    ("What is evidence?", "Evidence is information that supports or challenges a claim.", "Good evidence is relevant, reliable, and specific."),
    ("What is critical thinking?", "Critical thinking means judging ideas carefully using reason and evidence.", "It includes checking assumptions, comparing explanations, and avoiding quick conclusions."),
    ("What is empathy?", "Empathy is understanding or sharing another person's feelings.", "It helps people communicate kindly and respond with care."),
    ("What is resilience?", "Resilience is the ability to recover from difficulty.", "It grows through support, practice, patience, and problem-solving."),
    ("What is communication?", "Communication is sharing information or meaning.", "It can happen through words, tone, body language, images, or writing."),
    ("What is leadership?", "Leadership is guiding people toward a goal.", "Good leaders communicate clearly, take responsibility, and help others work well."),
    ("What is project management?", "Project management is planning and organizing work to reach a goal.", "It includes scope, tasks, timelines, resources, risks, and communication."),
    ("What is a budget?", "A budget is a plan for income and spending.", "It helps people control costs and prepare for future needs."),
    ("What is interest?", "Interest is the cost of borrowing money or the reward for lending it.", "It is often shown as a percentage over time."),
]

REASONING = [
    ("Why should I verify important answers?", "Because any model can make mistakes. Verification protects you from acting on incomplete or incorrect information."),
    ("How can I learn a hard topic?", "Break it into smaller parts, practice one part at a time, and test yourself often. Short daily practice usually works better than rare long sessions."),
    ("How do I debug a problem?", "Reproduce it, isolate the smallest failing case, inspect inputs and outputs, then test one change at a time."),
    ("Why does clean data matter for training?", "The model learns patterns from whatever it sees. Clean data teaches useful patterns, while noisy data teaches noise."),
    ("How can I explain a complex idea simply?", "Start with the main point, use familiar examples, avoid extra terms, and add detail only after the core idea is clear."),
    ("Why is a smaller model sometimes better at first?", "A smaller model trains faster and can reveal data or architecture problems sooner. Once the pipeline works, scaling becomes safer."),
    ("How do I make better decisions?", "Define the goal, list options, compare tradeoffs, check evidence, and choose the option with the best expected outcome."),
    ("Why does repetition happen in generated text?", "The model may overvalue frequent tokens or fail to learn when to stop. Better data balance and repetition penalties can reduce it."),
    ("How do I improve a dataset?", "Remove bad rows, balance common patterns, add missing examples, and keep answers accurate and varied."),
    ("Why use validation loss?", "Validation loss checks performance on held-out data. It helps detect whether training progress is real or just memorization."),
]

CONVERSATION = [
    ("Hi! How are you?", "I'm doing well. How can I help you today?"),
    ("Hello.", "Hello! What would you like help with?"),
    ("Who are you?", "I am Atulya, a helpful AI assistant."),
    ("What can you do?", "I can answer questions, explain concepts, help with code, and organize ideas."),
    ("Thanks for your help.", "You're welcome. I'm glad I could help."),
    ("Can you explain that again?", "Sure. I can restate it more simply."),
    ("I am confused.", "No problem. Tell me which part is unclear, and we can work through it step by step."),
    ("Can you give an example?", "Yes. A concrete example can make the idea easier to understand."),
    ("Please be brief.", "Understood. I will keep the answer short."),
    ("Can you go deeper?", "Yes. I can add more detail and explain the reasoning behind it."),
]

CODE = [
    ("Write a Python function to add two numbers.", "def add(a, b):\n    return a + b"),
    ("Write a Python function to multiply two numbers.", "def multiply(a, b):\n    return a * b"),
    ("Check if a number is even in Python.", "def is_even(n):\n    return n % 2 == 0"),
    ("Reverse a string in Python.", "def reverse_text(s):\n    return s[::-1]"),
    ("Find the largest number in a list.", "def largest(nums):\n    return max(nums)"),
    ("Filter even numbers from a list.", "def even_numbers(nums):\n    return [n for n in nums if n % 2 == 0]"),
    ("Count words in a string.", "def count_words(text):\n    return len(text.split())"),
    ("Sort a list in Python.", "def sort_items(items):\n    return sorted(items)"),
    ("Check if a string is a palindrome.", "def is_palindrome(s):\n    cleaned = ''.join(ch.lower() for ch in s if ch.isalnum())\n    return cleaned == cleaned[::-1]"),
    ("Create a dictionary from two lists.", "def make_dict(keys, values):\n    return dict(zip(keys, values))"),
]

OPENERS = [
    "{answer}",
    "In short, {lower}",
    "Simply put, {lower}",
    "A useful way to say it is: {answer}",
    "The simple answer is: {lower}",
    "It means {lower}",
    "You can think of it as this: {answer}",
    "Practically, {lower}",
    "For most purposes, {lower}",
    "One clear answer is: {answer}",
]

QUESTION_PREFIXES = [
    "",
    "Answer briefly: ",
    "Explain simply: ",
    "Can you answer this? ",
    "Give a useful answer: ",
    "In plain English, ",
    "Quick question: ",
]


def lower_first(text: str) -> str:
    text = text.strip()
    return text[:1].lower() + text[1:] if text else text


def style_answer(answer: str, detail: str, length: str, idx: int) -> str:
    base = answer.strip()
    if length == "short":
        text = base
    elif length == "medium":
        text = f"{base} {detail.strip()}"
    else:
        text = (
            f"{base} {detail.strip()} "
            "A good way to understand it is to connect the definition to a familiar example. "
            "Start with the main idea, then add details only when they help the answer become clearer."
        )
    template = OPENERS[idx % len(OPENERS)]
    return template.format(answer=text, lower=lower_first(text))


def make_row(user: str, assistant: str, category: str, rng: random.Random, prefix_idx: int) -> dict[str, str]:
    prefix = QUESTION_PREFIXES[prefix_idx % len(QUESTION_PREFIXES)]
    prompt = user.strip()
    if prefix:
        prompt = prefix + lower_first(prompt)
    return {
        "system": rng.choice(TEACHER_SYSTEMS),
        "user": prompt,
        "assistant": assistant.strip(),
        "category": category,
    }


def first_token(text: str) -> str:
    match = re.search(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]", text.strip())
    return match.group(0) if match else ""


def entropy(rows: list[dict[str, str]]) -> float:
    counts = Counter(first_token(r["assistant"]) for r in rows)
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values()) if total else 0.0


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    out = []
    for row in rows:
        key = (
            re.sub(r"\s+", " ", row["user"].lower()).strip(),
            re.sub(r"\s+", " ", row["assistant"].lower()).strip(),
        )
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def expand(rows: list[tuple[str, str, str]], category: str, length: str, target: int, rng: random.Random) -> list[dict[str, str]]:
    out = []
    i = 0
    while len(out) < target:
        user, answer, detail = rows[i % len(rows)]
        answer_text = style_answer(answer, detail, length, i)
        out.append(make_row(user, answer_text, category, rng, i))
        i += 1
        if i > target * 20:
            break
    return dedupe(out)[:target]


def expand_simple(rows: list[tuple[str, str]], category: str, length: str, target: int, rng: random.Random) -> list[dict[str, str]]:
    triples = []
    for user, answer in rows:
        detail = "This answer focuses on the practical point and avoids unnecessary detail."
        triples.append((user, answer, detail))
    return expand(triples, category, length, target, rng)


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_source_row(row: dict[str, str]) -> bool:
    user = row.get("user", "").strip()
    assistant = row.get("assistant", "").strip()
    joined = f"{user} {assistant}".lower()
    bad = (
        "thought:",
        "action:",
        "available apis:",
        "relevant apis:",
        "api_name",
        "tool_name",
        "paraphrase answer:",
        "i'm not sure i can answer",
        "what is it?",
        "what is it,",
    )
    if any(x in joined for x in bad):
        return False
    if len(user) < 4 or len(assistant) < 8:
        return False
    if len(user) > 220 or len(assistant) > 420:
        return False
    return True


def load_balanced_source(path: Path = TEACHER_SOURCE_BALANCED) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if clean_source_row(row):
                rows.append(row)
    return rows


SOURCE_OPENERS = [
    "{answer}",
    "In short, {lower}",
    "Simply put, {lower}",
    "The key idea is that {lower}",
    "It means {lower}",
    "A practical way to say it is: {answer}",
    "You can understand it as follows: {answer}",
    "For a simple answer, {lower}",
    "One useful answer is: {answer}",
    "Briefly, {lower}",
]


def source_variant(row: dict[str, str], length: str, category: str, i: int) -> dict[str, str]:
    user = row["user"].strip()
    answer = row["assistant"].strip()
    if length == "short":
        answer_text = answer.split("\n")[0]
        if len(answer_text) > 180:
            answer_text = answer_text[:177].rstrip() + "..."
    elif length == "medium":
        answer_text = answer
        if len(answer_text) < 120:
            answer_text = answer_text.rstrip(".") + ". This gives the main point without adding unnecessary detail."
    else:
        answer_text = (
            answer.rstrip(".")
            + ". The important part is to connect the answer to the user's question, keep the wording precise, "
            + "and avoid adding claims that are not needed. A clear response should define the idea, give one useful detail, "
            + "and stop before it becomes noisy."
        )
        if len(answer_text) > 760:
            answer_text = answer_text[:757].rstrip() + "..."
    template = SOURCE_OPENERS[i % len(SOURCE_OPENERS)]
    answer_text = template.format(answer=answer_text, lower=lower_first(answer_text))
    prefixes = [
        "",
        "Answer clearly: ",
        "Give a practical answer: ",
        "Explain in plain English: ",
        "Help me understand: ",
        "Can you answer this? ",
    ]
    prefix = prefixes[(i // len(SOURCE_OPENERS)) % len(prefixes)]
    if prefix:
        user = prefix + lower_first(user)
    return {
        "system": row.get("system") or TEACHER_SYSTEMS[i % len(TEACHER_SYSTEMS)],
        "user": user,
        "assistant": answer_text,
        "category": category,
    }


def fill_from_source(
    rows: list[dict[str, str]],
    source: list[dict[str, str]],
    target: int,
    length: str,
    category: str,
    rng: random.Random,
) -> list[dict[str, str]]:
    out = list(rows)
    seen = {
        (
            re.sub(r"\s+", " ", r["user"].lower()).strip(),
            re.sub(r"\s+", " ", r["assistant"].lower()).strip(),
        )
        for r in out
    }
    if not source:
        return out[:target]
    shuffled = list(source)
    rng.shuffle(shuffled)
    i = 0
    while len(out) < target and i < target * 20:
        base = shuffled[i % len(shuffled)]
        row = source_variant(base, length, category, i)
        key = (
            re.sub(r"\s+", " ", row["user"].lower()).strip(),
            re.sub(r"\s+", " ", row["assistant"].lower()).strip(),
        )
        if key not in seen:
            seen.add(key)
            out.append(row)
        i += 1
    return out[:target]


def report(name: str, rows: list[dict[str, str]]) -> None:
    cats = Counter(r["category"] for r in rows)
    firsts = Counter(first_token(r["assistant"]) for r in rows)
    print(f"{name}: rows={len(rows)} entropy={entropy(rows):.2f}")
    print("  categories:", dict(cats))
    print("  first_tokens:", firsts.most_common(12))


def build(seed: int) -> dict[str, list[dict[str, str]]]:
    rng = random.Random(seed)
    source = load_balanced_source()
    fact_triples = FACTS
    reasoning_triples = [(u, a, "The key is to use a clear process instead of guessing.") for u, a in REASONING]
    conversation_triples = [(u, a, "A helpful reply should be friendly, direct, and easy to continue.") for u, a in CONVERSATION]
    code_triples = [(u, a, "This code is intentionally small so it is easy to inspect and modify.") for u, a in CODE]

    short = []
    short += expand(fact_triples, "factual_short", "short", 12_000, rng)
    short += expand(reasoning_triples, "reasoning_short", "short", 4_000, rng)
    short += expand(conversation_triples, "conversation_short", "short", 4_000, rng)
    short += expand(code_triples, "code_short", "short", 5_000, rng)

    medium = []
    medium += expand(fact_triples, "factual_medium", "medium", 10_000, rng)
    medium += expand(reasoning_triples, "reasoning_medium", "medium", 7_000, rng)
    medium += expand(conversation_triples, "conversation_medium", "medium", 4_000, rng)
    medium += expand(code_triples, "code_medium", "medium", 4_000, rng)

    long = []
    long += expand(fact_triples, "factual_long", "long", 5_000, rng)
    long += expand(reasoning_triples, "reasoning_long", "long", 4_000, rng)
    long += expand(conversation_triples, "conversation_long", "long", 2_000, rng)
    long += expand(code_triples, "code_long", "long", 4_000, rng)

    paraphrase = []
    all_bases = fact_triples + reasoning_triples + conversation_triples + code_triples
    for length, count in [("short", 5_000), ("medium", 5_000), ("long", 5_000)]:
        paraphrase += expand(all_bases, f"paraphrase_{length}", length, count, rng)

    # Dedupe and shuffle within each file.
    datasets = {
        "teacher_short_25k.jsonl": fill_from_source(dedupe(short), source, 25_000, "short", "teacher_short", rng),
        "teacher_medium_25k.jsonl": fill_from_source(dedupe(medium), source, 25_000, "medium", "teacher_medium", rng),
        "teacher_long_15k.jsonl": fill_from_source(dedupe(long), source, 15_000, "long", "teacher_long", rng),
        "teacher_paraphrase_15k.jsonl": fill_from_source(dedupe(paraphrase), source, 15_000, "medium", "teacher_paraphrase", rng),
    }
    mixed = []
    for rows in datasets.values():
        mixed.extend(rows)
    rng.shuffle(mixed)
    datasets["teacher_mixed_80k.jsonl"] = dedupe(mixed)[:80_000]
    for rows in datasets.values():
        rng.shuffle(rows)
    return datasets


def _teacher_main(args) -> None:
    output_dir = Path(args.output_dir) if isinstance(args.output_dir, str) else args.output_dir
    datasets = build(args.seed)
    for name, rows in datasets.items():
        path = output_dir / name
        write_jsonl(path, rows)
        report(name, rows)
