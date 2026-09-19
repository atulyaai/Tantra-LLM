"""
Smriti — Memory & Knowledge Layer

Handles both short-term conversation memory and long-term knowledge.
Manages recall, forgetting, and knowledge updates.

Architecture:
  Manas → Smriti → (memory retrieval, knowledge lookup) → Context → Response
"""

class Smriti:
    """Memory and knowledge management."""
    
    def __init__(self):
        self.short_term_memory = {}  # Conversation-level memory
        self.long_term_memory = {}   # Persistent knowledge
        self.forgetting_rate = 0.1   # 10% forgetting per session
    
    def remember(self, key, value, memory_type="short"):
        """Store information in memory."""
        if memory_type == "short":
            self.short_term_memory[key] = {
                "value": value,
                "timestamp": self._get_time(),
                "recency": 0.0,  # 0.0 = very recent, 1.0 = about to be forgotten
            }
        elif memory_type == "long":
            self.long_term_memory[key] = {
                "value": value,
                "timestamp": self._get_time(),
                "access_count": 0,
            }
    
    def recall(self, key):
        """Retrieve information from memory."""
        # Check short-term first
        if key in self.short_term_memory:
            entry = self.short_term_memory[key]
            entry["recency"] += self.forgetting_rate
            entry["access_count"] = entry.get("access_count", 0) + 1
            
            if entry["recency"] > 1.0:  # Forgotten
                del self.short_term_memory[key]
                return None
            
            return entry["value"]
        
        # Check long-term
        if key in self.long_term_memory:
            entry = self.long_term_memory[key]
            entry["access_count"] += 1
            return entry["value"]
        
        return None
    
    def forget(self):
        """Apply forgetting to short-term memory."""
        to_remove = []
        for key, entry in self.short_term_memory.items():
            entry["recency"] += self.forgetting_rate
            if entry["recency"] > 1.0:
                to_remove.append(key)
        
        for key in to_remove:
            del self.short_term_memory[key]
        
        return len(to_remove)
    
    def get_context(self, user_input):
        """Get relevant context for current conversation."""
        context = {}
        
        # Check if we've discussed this topic before
        if user_input in self.short_term_memory:
            context["previous_discussion"] = self.short_term_memory[user_input]
        
        # Get recently accessed topics
        recent_topics = sorted(
            self.short_term_memory.items(),
            key=lambda x: x[1]["recency"]
        )[:3]
        
        if recent_topics:
            context["recent_topics"] = [t[0] for t in recent_topics]
        
        return context
    
    def update_knowledge(self, key, value):
        """Update long-term knowledge."""
        if key in self.long_term_memory:
            self.long_term_memory[key]["value"] = value
        else:
            self.long_term_memory[key] = {
                "value": value,
                "timestamp": self._get_time(),
                "access_count": 1,
            }
    
    def _get_time(self):
        """Get current time timestamp."""
        import time
        return time.time()
    
    def process(self, user_input, conversation_history=None):
        """Full memory processing."""
        # Forget old entries
        forgotten = self.forget()
        
        # Get context
        context = self.get_context(user_input)
        
        # Store current interaction
        self.remember(user_input, conversation_history or {}, "short")
        
        return {
            "short_term": self.short_term_memory,
            "long_term_keys": list(self.long_term_memory.keys()),
            "context": context,
            "forgotten_count": forgotten,
            "status": "active",
        }
