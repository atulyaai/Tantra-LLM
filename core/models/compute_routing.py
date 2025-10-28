"""Compute routing: fast/medium/deep paths based on query complexity."""

from typing import Literal


class ComputeRouter:
    """Routes queries to fast/medium/deep compute paths."""
    
    def __init__(self):
        self.fast_max_tokens = 50
        self.medium_max_tokens = 200
        self.deep_max_tokens = 500
    
    def analyze_complexity(self, query: str, context_len: int = 0) -> float:
        """Score query complexity (0=simple, 1=complex)."""
        complexity = 0.0
        
        # Length heuristic
        if len(query) > 200:
            complexity += 0.3
        
        # Question marks suggest reasoning needed
        if "?" in query:
            complexity += 0.2
        
        # Complex keywords
        complex_keywords = ["explain", "how", "why", "analyze", "compare", "design", "plan"]
        if any(kw in query.lower() for kw in complex_keywords):
            complexity += 0.3
        
        # Context length
        if context_len > 10000:
            complexity += 0.2
        
        return min(complexity, 1.0)
    
    def select_path(self, query: str, context_len: int = 0) -> Literal["fast", "medium", "deep"]:
        """Return fast/medium/deep path based on complexity."""
        score = self.analyze_complexity(query, context_len)
        
        if score < 0.3:
            return "fast"
        elif score < 0.7:
            return "medium"
        else:
            return "deep"
    
    def get_max_tokens(self, path: str) -> int:
        """Get max_tokens for selected path."""
        mapping = {
            "fast": self.fast_max_tokens,
            "medium": self.medium_max_tokens,
            "deep": self.deep_max_tokens,
        }
        return mapping.get(path, self.medium_max_tokens)
    
    def get_max_context(self, path: str) -> int:
        """Get max context window for selected path."""
        mapping = {
            "fast": 2048,
            "medium": 4096,
            "deep": 8192,
        }
        return mapping.get(path, 4096)
