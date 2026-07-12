"""Compute routing: fast/medium/deep paths based on query complexity and performance history."""

from typing import Literal, Dict, List, Tuple
from config.identity import IDENTITY

class ComputeRouter:
    """Routes queries to fast/medium/deep compute paths dynamically based on complexity and recorded history."""
    
    def __init__(self):
        self.fast_max_tokens = 50
        self.medium_max_tokens = 200
        self.deep_max_tokens = 500
        
        # Performance history: key is (path, provider), value is List[Tuple[latency_ms, cost]]
        self.history: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        self.history_limit = 10
        
        # Load identity configuration
        latency_pref = IDENTITY.get("latency_vs_precision", {})
        self.fast_threshold = latency_pref.get("fast_threshold_ms", 500)
        self.medium_threshold = latency_pref.get("medium_threshold_ms", 2000)
        self.deep_threshold = latency_pref.get("deep_threshold_ms", 10000)

    def record_performance(self, path: str, provider: str, latency_ms: float, cost: float):
        """Records latency and cost metrics for a given path and provider to adapt routing thresholds."""
        key = (path, provider)
        if key not in self.history:
            self.history[key] = []
        
        self.history[key].append((latency_ms, cost))
        if len(self.history[key]) > self.history_limit:
            self.history[key].pop(0)

    def get_average_performance(self, path: str, provider: str) -> Tuple[float, float]:
        """Returns the rolling average (latency_ms, cost) for the specified path and provider."""
        key = (path, provider)
        records = self.history.get(key, [])
        if not records:
            # Return default baseline estimates
            baselines = {
                "fast": (150.0, 0.0),
                "medium": (800.0, 0.001),
                "deep": (4000.0, 0.005)
            }
            return baselines.get(path, (500.0, 0.002))
        
        avg_latency = sum(r[0] for r in records) / len(records)
        avg_cost = sum(r[1] for r in records) / len(records)
        return avg_latency, avg_cost
    
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
        complex_keywords = ["explain", "how", "why", "analyze", "compare", "design", "plan", "summarize", "describe"]
        if any(kw in query.lower() for kw in complex_keywords):
            complexity += 0.3
        
        # Mode-specific complexity
        if "mode:" in query.lower():
            complexity += 0.1  # Mode switching is simple
        
        # Context length
        if context_len > 10000:
            complexity += 0.2
        
        return min(complexity, 1.0)
    
    def select_path(self, query: str, provider: str = "local", context_len: int = 0) -> Literal["fast", "medium", "deep"]:
        """Return fast/medium/deep path based on complexity, adjusted for recorded performance history."""
        score = self.analyze_complexity(query, context_len)
        
        # Determine baseline path recommendation
        if score < 0.3:
            base_path = "fast"
        elif score < 0.7:
            base_path = "medium"
        else:
            base_path = "deep"
            
        # Self-adaptation check: check if the recommended path has been under pressure (slow/costly)
        avg_latency, _ = self.get_average_performance(base_path, provider)
        
        # Shorten under pressure check
        if base_path == "deep" and avg_latency > self.deep_threshold:
            # Fallback to medium path to conserve system resources
            print(f"[AdaptiveRouter] Pressure detected on 'deep' path (avg latency={avg_latency:.1f}ms > threshold={self.deep_threshold}ms). Downgrading route to 'medium'.")
            return "medium"
            
        if base_path == "medium" and avg_latency > self.medium_threshold:
            print(f"[AdaptiveRouter] Pressure detected on 'medium' path (avg latency={avg_latency:.1f}ms > threshold={self.medium_threshold}ms). Downgrading route to 'fast'.")
            return "fast"
            
        return base_path
    
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
