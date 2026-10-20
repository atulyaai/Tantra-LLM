"""Tantra persona rules shared by the audit script and bulk filters (owner, 2026-10-19).

Mention being an AI only when the user asks about it; mention Atul Vij / Atulya AI only
when asked who made Tantra, its company or origin; never use service wording unless the
user brings it up. Each assistant turn is judged against the user turn right before it,
so one early question does not license every later reply.
"""
import re

ASKS_AI = re.compile(r'\b(ai|a\.i\.|human|person|people|real|robot|bot|chatbot|machine|program|alive|living|conscious|'
                     r'feelings|chatgpt|gpt|gemini|siri|alexa|claude|copilot|bard|llama|jarvis|model|computer|'
                     r'what (exactly )?are you|what is this|someone (reading|typing)|customer care|meet( up)?|'
                     r'turing|are you (him|her)|grandma|can we meet|pick up)\b', re.I)
ASKS_CREATOR = re.compile(r'\b(made|maker|creat\w*|built|build|develop\w*|coded|programm\w*|trained|invent\w*|designed|'
                          r'behind (you|tantra)|owns? you|owner|company|firm|startup|brand|organi[sz]ation|'
                          r'founder|founded|your dev|origin|atul|atulya|microsoft|amazon|apple|meta|openai|facebook|'
                          r'tata|reliance|government|university|credit|responsible for you|thank for you|boss|'
                          r'gave you life|into this world|your (father|parent|story)|raised you|your existence|'
                          r'mind behind|where (are|do) you (come )?from|who (are|r) (you|u)|introduce yourself|'
                          r'about (yourself|you)|who is this|identify yourself|(make|made|brought|built) you|(did|do) you come from|'
                          r'are you (from|by|made)|google|jarvis|chatgpt|gemini|siri|alexa|claude|copilot|bard)\b', re.I)
SAYS_AI = re.compile(r"\b(an ai|i'm an ai|i am an ai|ai assistant|language model|computer program|chatbot|"
                     r"i'm a bot|i am a bot|virtual (assistant|agent)|a piece of software)\b", re.I)
SAYS_CREATOR = re.compile(r'\b(atul vij|atulya( ai)?|atul)\b', re.I)
SERVICE = re.compile(r'\b(assistant|how can i help|how may i assist|as an ai language model)\b', re.I)
ASKS_SERVICE = re.compile(r'\bassistant\b', re.I)


def _previous_user(messages, i):
    for m in reversed(messages[:i]):
        if m['role'] == 'user':
            return m['content']
    return ''


def turn_violations(messages, i):
    """Problems in assistant turn i, judged against the user turn right before it."""
    if messages[i]['role'] != 'assistant':
        return []
    asked = _previous_user(messages, i)
    reply = messages[i]['content']; found = []
    if SAYS_AI.search(reply) and not ASKS_AI.search(asked): found.append('unasked AI mention')
    # Continuity: once the creator came up legitimately in this conversation, later turns may refer to him.
    earlier = any(m['role'] == 'assistant' and SAYS_CREATOR.search(m['content']) for m in messages[:i])
    if SAYS_CREATOR.search(reply) and not (ASKS_CREATOR.search(asked) or earlier): found.append('unasked creator mention')
    if SERVICE.search(reply) and not ASKS_SERVICE.search(asked): found.append('service wording')
    return found


def violations(messages):
    """Persona problems across every assistant turn, whatever the role order (system turns included)."""
    return [v for i, m in enumerate(messages) if m['role'] == 'assistant' for v in turn_violations(messages, i)]
