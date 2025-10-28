from __future__ import annotations

from typing import Optional


class BrainOrchestrator:
    """Main orchestrator: perception → decision → response → reflection → memory.
    
    Episodic memory retrieval now influences responses by prepending retrieved
    context to user queries before reasoning.
    """

    def __init__(self, perception, decision_engine, response_generator, memory_manager):
        self.perception = perception
        self.decision = decision_engine
        self.response = response_generator
        self.memory = memory_manager

    def step(self, text: Optional[str] = None, image=None, audio=None) -> str:
        """Process input and generate response with episodic memory influence."""
        user_input = text or ""
        
        # Retrieve relevant memories
        recalls = self.memory.recall(user_input)
        
        # Prepend retrieved context if available
        if recalls and len(recalls) > 0:
            context_prompt = f"[Memories: {', '.join(recalls[:2])}] {user_input}"
        else:
            context_prompt = user_input
        
        # Perception
        p = self.perception.perceive(text=context_prompt, image=image, audio=audio)
        
        # Decision
        d = self.decision.decide(user_input)
        
        # Response generation (with memory-influenced prompt)
        out = self.response.generate(p, d)
        
        # Store important interactions
        if user_input:
            self.memory.consider_store(user_input, importance=0.6)
        
        return out


