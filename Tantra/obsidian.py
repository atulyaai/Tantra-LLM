"""
tantra/obsidian.py — Obsidian Offline Knowledge Graph & Memory Vault Engine for Tantra-LLM.
Parses local Markdown files, builds offline Knowledge Graphs, and integrates with ALRA recurrent memory.
"""
from __future__ import annotations

import os
import re
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Set, Any, Optional, Tuple

from Tantra.utils import get_logger

log = get_logger("tantra.obsidian")


@dataclass
class KnowledgeNode:
    """Represents a node in the offline Obsidian Knowledge Graph."""
    node_id: str
    title: str
    content: str
    filepath: str = ""
    links: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)


class KnowledgeGraph:
    """Offline Graph database storing node relations and wikilinks."""

    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}

    def add_node(self, node: KnowledgeNode) -> None:
        self.nodes[node.node_id.lower()] = node

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        return self.nodes.get(node_id.lower())

    def search_by_query(self, query: str, max_results: int = 3) -> List[KnowledgeNode]:
        """Search nodes by keyword relevance."""
        query_terms = set(re.findall(r"\w+", query.lower()))
        if not query_terms:
            return []
        
        scored: List[Tuple[float, KnowledgeNode]] = []
        for node in self.nodes.values():
            node_terms = set(re.findall(r"\w+", (node.title + " " + node.content).lower()))
            intersection = query_terms.intersection(node_terms)
            if intersection:
                score = len(intersection) / len(query_terms)
                scored.append((score, node))
                
        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:max_results]]


class ObsidianVaultEngine:
    """
    Local vault parser, Knowledge Graph builder,
    and offline RAG context augmentation engine.
    """

    def __init__(self, vault_dir: Optional[str] = None):
        self.vault_dir = vault_dir
        self.graph = KnowledgeGraph()
        if vault_dir and os.path.exists(vault_dir):
            self.index_vault(vault_dir)

    def parse_markdown_file(self, filepath: str) -> KnowledgeNode:
        """Parse a single Markdown file into a KnowledgeNode extracting [[wikilinks]] and #tags."""
        filename = os.path.basename(filepath)
        title = os.path.splitext(filename)[0]
        
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        # Extract [[wikilinks]]
        wikilinks = set(re.findall(r"\[\[(.*?)\]\]", content))
        # Extract #tags
        tags = set(re.findall(r"#([\w-]+)", content))
        
        return KnowledgeNode(
            node_id=title,
            title=title,
            content=content,
            filepath=filepath,
            links=wikilinks,
            tags=tags
        )

    def index_vault(self, vault_dir: str) -> int:
        """Recursively scan and index all .md files in an Obsidian vault directory."""
        if not os.path.exists(vault_dir):
            log.warning(f"Vault directory does not exist: {vault_dir}")
            return 0
            
        indexed_count = 0
        for root, _, files in os.walk(vault_dir):
            for file in files:
                if file.endswith(".md"):
                    filepath = os.path.join(root, file)
                    try:
                        node = self.parse_markdown_file(filepath)
                        self.graph.add_node(node)
                        indexed_count += 1
                    except Exception as e:
                        log.warning(f"Could not index file {filepath}: {e}")
                        
        log.info(f"Successfully indexed {indexed_count} nodes into Obsidian Knowledge Graph.")
        return indexed_count

    def augment_prompt(self, query: str, max_context_chars: int = 500) -> str:
        """Perform offline local RAG graph lookup and augment query prompt."""
        nodes = self.graph.search_by_query(query, max_results=2)
        if not nodes:
            return query
            
        context_snippets = []
        for node in nodes:
            snippet = node.content[:250].strip()
            context_snippets.append(f"[{node.title}]: {snippet}")
            
        augmented_context = "\n".join(context_snippets)
        return f"[Knowledge Vault Context]\n{augmented_context}\n\n[User Query]\n{query}"
