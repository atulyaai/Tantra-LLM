from __future__ import annotations

from typing import List, Tuple


class SemanticMemory:
    """Stub: graph-like facts/relations store (to integrate with NetworkX/Neo4j)."""

    def __init__(self):
        self.facts: List[Tuple[str, str, str]] = []

    def add_fact(self, subject: str, relation: str, obj: str):
        self.facts.append((subject, relation, obj))

    def query(self, node: str) -> List[Tuple[str, str, str]]:
        return [f for f in self.facts if node in f]


