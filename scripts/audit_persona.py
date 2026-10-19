"""Flag approved replies that volunteer "AI" or creator facts the user did not ask about.

Persona rule (owner, 2026-10-19): mention being an AI only when asked about being
human/AI/a bot etc., and mention Atul Vij / Atulya AI only when asked who made it,
its company or origin. Prints flagged rows and writes runs/persona-audit.json.
Usage: .venv\\Scripts\\python.exe scripts\\audit_persona.py <bundle incoming dir> ...
"""
import json, re, sys
from pathlib import Path

ASKS_AI = re.compile(r'\b(ai|a\.i\.|human|person|people|real|robot|bot|machine|program|alive|conscious|feel|feelings|'
                     r'chatgpt|gpt|gemini|siri|alexa|claude|copilot|bard|llama|jarvis|model|computer|what are you|what is this|'
                     r'someone|customer care|meet|turing|are you (him|her)|grandma)\b', re.I)
ASKS_CREATOR = re.compile(r'\b(made|make|maker|creat\w*|built|build|develop\w*|coded|program\w*|train\w*|invent\w*|designed|'
                          r'behind|owns?|owner|company|firm|startup|brand|organi[sz]ation|founder|founded|dev|from|origin|'
                          r'atul|atulya|microsoft|amazon|apple|meta|openai|facebook|tata|reliance|government|university|'
                          r'credit|life|mind|responsible|thank|boss|world|father|parent|raised|story|existence)\b', re.I)
SAYS_AI = re.compile(r'\b(an ai|i\'m an ai|i am an ai|ai assistant|language model|computer program)\b', re.I)
SAYS_CREATOR = re.compile(r'\b(atul vij|atulya ai)\b', re.I)


def audit(paths):
    flagged = []
    for path in paths:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            row = json.loads(line); msgs = row['messages']
            for i in range(1, len(msgs), 2):
                asked = ' '.join(m['content'] for m in msgs[:i] if m['role'] == 'user')
                reply = msgs[i]['content']; why = []
                if SAYS_AI.search(reply) and not ASKS_AI.search(asked): why.append('unasked AI mention')
                if SAYS_CREATOR.search(reply) and not ASKS_CREATOR.search(asked): why.append('unasked creator mention')
                if why:
                    flagged.append({'file': str(path), 'family_id': row['family_id'], 'user': msgs[i - 1]['content'],
                                    'reply': reply, 'why': why})
    return flagged


def main():
    paths = [p for d in sys.argv[1:] for p in sorted(Path(d).glob('*.jsonl'))]
    flagged = audit(paths)
    for f in flagged:
        print(f"{f['family_id']}: {f['user']!r} -> {f['reply']!r}  [{', '.join(f['why'])}]")
    Path('runs/persona-audit.json').write_text(json.dumps(flagged, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{len(flagged)} flagged replies')


if __name__ == '__main__':
    main()
