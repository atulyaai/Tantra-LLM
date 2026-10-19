"""Tantra persona rules shared by the audit script and bulk filters (owner, 2026-10-19).

Mention being an AI only when the user asks about it; mention Atul Vij / Atulya AI only
when asked who made Tantra, its company or origin; never use service wording.
"""
import re

ASKS_AI = re.compile(r'\b(ai|a\.i\.|human|person|people|real|robot|bot|machine|program|alive|conscious|feel|feelings|'
                     r'chatgpt|gpt|gemini|siri|alexa|claude|copilot|bard|llama|jarvis|model|computer|what (exactly )?are you|what is this|'
                     r'someone|customer care|meet|turing|are you (him|her)|grandma|living)\b', re.I)
ASKS_CREATOR = re.compile(r'\b(made|make|maker|creat\w*|built|build|develop\w*|coded|program\w*|train\w*|invent\w*|designed|'
                          r'behind|owns?|owner|company|firm|startup|brand|organi[sz]ation|founder|founded|dev|from|origin|'
                          r'atul|atulya|microsoft|amazon|apple|meta|openai|facebook|tata|reliance|government|university|'
                          r'credit|life|mind|responsible|thank|boss|world|father|parent|raised|story|existence)\b', re.I)
SAYS_AI = re.compile(r"\b(an ai|i'm an ai|i am an ai|ai assistant|language model|computer program)\b", re.I)
SAYS_CREATOR = re.compile(r'\b(atul vij|atulya ai)\b', re.I)
SERVICE = re.compile(r'\b(assistant|how can i help|how may i assist|as an ai language model)\b', re.I)


def turn_violations(messages, i):
    """Problems in assistant turn i, judged against every user turn before it."""
    asked = ' '.join(m['content'] for m in messages[:i] if m['role'] == 'user')
    reply = messages[i]['content']; found = []
    if SAYS_AI.search(reply) and not ASKS_AI.search(asked): found.append('unasked AI mention')
    if SAYS_CREATOR.search(reply) and not ASKS_CREATOR.search(asked): found.append('unasked creator mention')
    if SERVICE.search(reply): found.append('service wording')
    return found


def violations(messages):
    return [v for i in range(1, len(messages), 2) for v in turn_violations(messages, i)]
