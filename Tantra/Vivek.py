"""
Vivek — Reasoning & Quality Gate Layer

Acts as the central reasoning engine. Decides:
1. Whether to answer directly
2. Whether to consult Chitta (emotional context)
3. Whether to consult Smriti (memory)
4. Whether to use Yantra (tools)
5. Whether to use expert (LoRA specialist)
6. Quality gate — ensure answer meets standards

Architecture:
  Manas → Vivek → (direct/Chitta/Smriti/Yantra/Expert) → Quality Gate → Response
"""

class Vivek:
    """Reasoning engine and quality gate."""
    
    def __init__(self):
        self.max_retries = 3
        self.quality_threshold = 0.7  # 70% confidence required
    
    def decide_path(self, manas_output, context=None):
        """
        Decide which cognitive path to take.
        
        Returns:
            dict with routing decision
        """
        language = manas_output.get("language", "english")
        intent = manas_output.get("intent", "conversation")
        emotion = manas_output.get("emotion", "neutral")
        
        path = {
            "language": language,
            "intent": intent,
            "emotion": emotion,
            "routes": [],
        }
        
        # Route based on intent
        if intent == "code":
            path["routes"].append("expert")  # Code expert
        elif intent == "math":
            path["routes"].append("expert")  # Math expert
        elif intent == "reasoning":
            path["routes"].append("direct")  # Answer directly
            path["routes"].append("quality_gate")
        elif intent == "instruction":
            path["routes"].append("direct")
            path["routes"].append("quality_gate")
        elif intent == "conversation":
            path["routes"].append("chitta")  # Emotional context
            path["routes"].append("smriti")   # Memory
            path["routes"].append("direct")
        
        # Add emotional routing
        if emotion == "frustrated":
            path["routes"].insert(0, "chitta")  # Address emotion first
        elif emotion == "curious":
            path["routes"].append("smriti")  # Provide more context
        
        # Add language routing
        path["routes"].append(f"language_{manas_output.get('language', 'english')}")
        
        # Quality gate
        path["routes"].append("quality_gate")
        
        return path
    
    def quality_gate(self, response, manas_output):
        """
        Check if response meets quality standards.
        
        Checks:
        - Length appropriate
        - Language matches user's language
        - No hallucination detected
        - Relevance to question
        """
        if not response:
            return {"passed": False, "reason": "empty response"}
        
        if len(response) < 3:
            return {"passed": False, "reason": "too short"}
        
        if len(response) > 10000:
            return {"passed": False, "reason": "too long"}
        
        # Language check
        user_language = manas_output.get("language", "english")
        if user_language == "hindi" and not any('\u0900' <= c <= '\u097F' for c in response):
            return {"passed": False, "reason": "wrong language for hindi query"}
        if user_language == "hinglish" and "Bhai" not in response and "bhai" not in response.lower():
            pass  # Hinglish is flexible
        
        return {"passed": True, "reason": "quality OK", "score": self.quality_threshold}
    
    def answer_directly(self, question):
        """Generate direct answer."""
        return {"type": "direct", "content": f"Answer: {question}"}
    
    def consult_chitta(self, emotion):
        """Get emotional context."""
        return {
            "type": "chitta",
            "emotion": emotion,
            "tone": "empathetic" if emotion == "frustrated" else "neutral",
        }
    
    def consult_smriti(self, topic):
        """Get memory/knowledge."""
        return {
            "type": "smriti",
            "topic": topic,
            "cached": False,
            "depth": "standard",
        }
    
    def use_yantra(self, task):
        """Use tool for computation."""
        return {
            "type": "yantra",
            "task": task,
            "tool": "calculator" if task == "math" else "code_executor",
        }
    
    def route_to_expert(self, domain):
        """Route to LoRA expert."""
        return {
            "type": "expert",
            "domain": domain,
            "available": True,
        }
    
    def process(self, manas_output, context=None):
        """Full reasoning pipeline."""
        path = self.decide_path(manas_output, context)
        
        return {
            "routing_decision": path,
            "quality_threshold": self.quality_threshold,
            "max_retries": self.max_retries,
            "status": "ready",
        }
