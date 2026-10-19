"""Flag approved replies that volunteer "AI" or creator facts the user did not ask about.

Rules live in Tantra/data/persona_rules.py (shared with the bulk-review filters).
Prints flagged rows and writes runs/persona-audit.json.
Usage: .venv\\Scripts\\python.exe scripts\\audit_persona.py <bundle incoming dir> ...
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Tantra.data.persona_rules import turn_violations


def audit(paths):
    flagged = []
    for path in paths:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            row = json.loads(line); msgs = row['messages']
            for i in range(1, len(msgs), 2):
                why = turn_violations(msgs, i)
                if why:
                    flagged.append({'file': str(path), 'family_id': row['family_id'], 'user': msgs[i - 1]['content'],
                                    'reply': msgs[i]['content'], 'why': why})
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
