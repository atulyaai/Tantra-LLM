from __future__ import annotations

import re
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class DecisionEngine:
    """Advanced decision engine for recall depth, mode selection, and storage decisions."""

    def __init__(self, personality_layer, memory_manager):
        self.personality = personality_layer
        self.memory = memory_manager
        
        # Decision parameters
        self.complexity_thresholds = {
            "simple": 0.3,
            "medium": 0.6,
            "complex": 0.8
        }
        
        # Safety keywords
        self.safety_keywords = [
            "harm", "danger", "illegal", "violence", "hate", "discrimination",
            "self-harm", "suicide", "bomb", "weapon", "drug", "abuse"
        ]
        
        # Context keywords for memory relevance
        self.context_keywords = [
            "remember", "recall", "previous", "earlier", "before", "last time",
            "context", "background", "history", "past"
        ]

    def decide(self, user_text: str) -> Dict[str, Any]:
        """Make comprehensive decisions about processing the user input."""
        try:
            # Analyze input complexity
            complexity = self._analyze_complexity(user_text)
            
            # Determine recall depth based on complexity
            recall_depth = self._determine_recall_depth(complexity, user_text)
            
            # Select personality mode
            mode = self.personality.select_mode(user_text)
            
            # Check for safety concerns
            safety_check = self._check_safety(user_text)
            
            # Determine storage importance
            storage_importance = self._determine_storage_importance(user_text, complexity)
            
            # Get relevant memories
            recalls = self.memory.recall(user_text, top_k=recall_depth)
            
            # Determine processing priority
            priority = self._determine_priority(complexity, safety_check, user_text)
            
            return {
                "mode": mode,
                "recall": recalls,
                "recall_depth": recall_depth,
                "complexity": complexity,
                "safety_check": safety_check,
                "storage_importance": storage_importance,
                "priority": priority,
                "requires_context": self._requires_context(user_text),
                "processing_time_estimate": self._estimate_processing_time(complexity, recall_depth)
            }
            
        except Exception as e:
            logger.error(f"Error in decision engine: {e}")
            # Fallback to basic decisions
            return {
                "mode": "DirectAssertive",
                "recall": [],
                "recall_depth": 3,
                "complexity": 0.5,
                "safety_check": {"safe": True, "concerns": []},
                "storage_importance": 0.5,
                "priority": "normal",
                "requires_context": False,
                "processing_time_estimate": 1.0
            }

    def _analyze_complexity(self, text: str) -> float:
        """Analyze text complexity (0.0 = simple, 1.0 = very complex)."""
        complexity = 0.0
        
        # Length factor
        word_count = len(text.split())
        if word_count > 100:
            complexity += 0.2
        elif word_count > 50:
            complexity += 0.1
        
        # Question complexity
        question_count = text.count('?')
        if question_count > 3:
            complexity += 0.2
        elif question_count > 1:
            complexity += 0.1
        
        # Technical terms
        technical_terms = [
            "algorithm", "neural", "network", "machine learning", "artificial intelligence",
            "programming", "code", "function", "variable", "class", "method", "API",
            "database", "query", "optimization", "performance", "scalability"
        ]
        technical_count = sum(1 for term in technical_terms if term.lower() in text.lower())
        complexity += min(technical_count * 0.1, 0.3)
        
        # Reasoning indicators
        reasoning_indicators = [
            "explain", "analyze", "compare", "contrast", "evaluate", "critique",
            "design", "plan", "strategy", "approach", "methodology", "framework"
        ]
        reasoning_count = sum(1 for indicator in reasoning_indicators if indicator.lower() in text.lower())
        complexity += min(reasoning_count * 0.15, 0.3)
        
        # Multi-step instructions
        step_indicators = ["first", "then", "next", "finally", "step", "process"]
        step_count = sum(1 for indicator in step_indicators if indicator.lower() in text.lower())
        complexity += min(step_count * 0.1, 0.2)
        
        return min(complexity, 1.0)

    def _determine_recall_depth(self, complexity: float, text: str) -> int:
        """Determine how many memories to recall based on complexity and context."""
        base_depth = 3
        
        # Increase depth for complex queries
        if complexity > 0.7:
            base_depth = 8
        elif complexity > 0.4:
            base_depth = 5
        
        # Increase depth for context-seeking queries
        if any(keyword in text.lower() for keyword in self.context_keywords):
            base_depth += 3
        
        # Increase depth for questions
        if '?' in text:
            base_depth += 2
        
        return min(base_depth, 10)  # Cap at 10

    def _check_safety(self, text: str) -> Dict[str, Any]:
        """Check for safety concerns in the input."""
        text_lower = text.lower()
        concerns = []
        
        # Check for safety keywords
        for keyword in self.safety_keywords:
            if keyword in text_lower:
                concerns.append(f"Contains '{keyword}'")
        
        # Check for potentially harmful patterns
        harmful_patterns = [
            r'\b(how to|how do i)\s+(harm|hurt|kill|destroy)',
            r'\b(make|create|build)\s+(bomb|weapon|drug)',
            r'\b(illegal|unlawful)\s+(activity|action)',
            r'\b(self.?harm|suicide|end.?life)'
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, text_lower):
                concerns.append(f"Matches harmful pattern: {pattern}")
        
        return {
            "safe": len(concerns) == 0,
            "concerns": concerns,
            "severity": "high" if len(concerns) > 2 else "medium" if len(concerns) > 0 else "low"
        }

    def _determine_storage_importance(self, text: str, complexity: float) -> float:
        """Determine how important it is to store this interaction."""
        importance = 0.5  # Base importance
        
        # Increase importance for complex queries
        importance += complexity * 0.3
        
        # Increase importance for questions
        if '?' in text:
            importance += 0.2
        
        # Increase importance for context-seeking
        if any(keyword in text.lower() for keyword in self.context_keywords):
            importance += 0.2
        
        # Increase importance for technical content
        technical_indicators = ["code", "programming", "algorithm", "function", "class"]
        if any(indicator in text.lower() for indicator in technical_indicators):
            importance += 0.1
        
        # Decrease importance for simple greetings
        simple_greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
        if any(greeting in text.lower() for greeting in simple_greetings) and len(text.split()) < 5:
            importance = 0.1
        
        return min(importance, 1.0)

    def _determine_priority(self, complexity: float, safety_check: Dict, text: str) -> str:
        """Determine processing priority."""
        if not safety_check["safe"]:
            return "urgent"
        elif complexity > 0.8:
            return "high"
        elif "urgent" in text.lower() or "asap" in text.lower():
            return "high"
        elif complexity > 0.5:
            return "medium"
        else:
            return "normal"

    def _requires_context(self, text: str) -> bool:
        """Determine if the query requires additional context."""
        context_indicators = [
            "what did i", "what was", "previous", "earlier", "before", "last time",
            "context", "background", "history", "remember", "recall"
        ]
        return any(indicator in text.lower() for indicator in context_indicators)

    def _estimate_processing_time(self, complexity: float, recall_depth: int) -> float:
        """Estimate processing time in seconds."""
        base_time = 1.0
        complexity_time = complexity * 3.0
        recall_time = recall_depth * 0.1
        return base_time + complexity_time + recall_time


