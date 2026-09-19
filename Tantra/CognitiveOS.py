"""
Cognitive Orchestrator — Tantra's Main Cognitive OS Layer

Ties together Manas, Vivek, Chitta, Smriti, and Nirikshak
into a complete cognitive pipeline.

Architecture:
  User → Manas → Vivek → Chitta → Smriti → Yantra → Expert
                                              ↓
                                         Nirikshak (auto-learn)
"""

import json, os, sys

sys.stdout.reconfigure(encoding='utf-8')

class CognitiveOS:
    """Main cognitive orchestration system."""
    
    def __init__(self):
        from Manas import Manas
        from Vivek import Vivek
        from Chitta import Chitta
        from Smriti import Smriti
        from Nirikshak import Nirikshak, LearningQueue, LoRAExpert
        
        self.manas = Manas()
        self.vivek = Vivek()
        self.chitta = Chitta()
        self.smriti = Smriti()
        self.nirikshak = Nirikshak()
        self.learning_queue = LearningQueue()
        self.lora_expert = LoRAExpert()
        
        print("Cognitive OS initialized.")
        print("  Manas: language detection")
        print("  Vivek: reasoning & quality gate")
        print("  Chitta: emotional context")
        print("  Smriti: memory & knowledge")
        print("  Nirikshak: auto-learning loop")
    
    def process(self, user_input):
        """
        Full cognitive pipeline from input to response.
        
        Flow:
        1. Manas detects language/intent/emotion
        2. Smriti retrieves relevant memory
        3. Chitta assesses emotional context
        4. Vivek decides routing path
        5. Generate response
        6. Nirikshak evaluates for learning
        """
        print(f"\n{'='*60}")
        print(f"USER: {user_input[:80]}...")
        print(f"{'='*60}")
        
        # Step 1: Manas — Input processing
        manas_output = self.manas.process(user_input)
        print(f"  Language: {manas_output['language']}")
        print(f"  Intent: {manas_output['intent']}")
        print(f"  Emotion: {manas_output['emotion']}")
        
        # Step 2: Smriti — Memory retrieval
        smriti_output = self.smriti.process(user_input)
        print(f"  Memory context: {len(smriti_output['short_term'])} entries")
        
        # Step 3: Chitta — Emotional context
        chitta_output = self.chitta.process(user_input)
        print(f"  Empathy response: {chitta_output['empathy_response'][:50]}...")
        
        # Step 4: Vivek — Reasoning & routing
        vivek_output = self.vivek.process(manas_output, smriti_output)
        print(f"  Routing: {len(vivek_output['routing_decision']['routes'])} paths")
        
        # Step 5: Generate response
        response = self._generate_response(user_input, manas_output)
        print(f"  Response: {response[:80]}...")
        
        # Step 6: Quality gate
        quality = self.vivek.quality_gate(response, manas_output)
        print(f"  Quality: {'PASS' if quality['passed'] else 'FAIL'}")
        
        # Step 7: Nirikshak — Auto-learning evaluation
        evaluation = self.nirikshak.evaluate(
            response, user_input, quality.get("score", 0.5)
        )
        
        if evaluation["needs_learning"]:
            print(f"  → LEARNING NEEDED: {evaluation['reason']}")
            self.learning_queue.add(evaluation)
        
        # Step 8: Store in memory
        self.smriti.remember(user_input, response, "short")
        
        return {
            "response": response,
            "language": manas_output["language"],
            "quality": quality,
            "learning_flagged": evaluation["needs_learning"],
            "emotion": chitta_output["emotion"],
        }
    
    def _generate_response(self, user_input, manas_output):
        """Generate response based on processing."""
        language = manas_output["language"]
        
        if language == "hindi":
            return "मैं आपकी मदद के लिए तैयार हूँ!"
        elif language == "hinglish":
            return "Bhai, main ready hoon! Kya chahiye?"
        else:
            return "I'm ready to help! What do you need?"
    
    def run_auto_learning(self):
        """Run the auto-learning cycle."""
        from Nirikshak import run_auto_learning_cycle
        
        # Get conversation history
        conv_history = list(self.smriti.short_term_memory.values())
        
        results = run_auto_learning_cycle(conv_history)
        return results
    
    def get_benchmark_results(self):
        """Run benchmark on current model."""
        from benchmark import run_benchmark
        results = run_benchmark(category="all")
        return results
    
    def status(self):
        """Get full system status."""
        return {
            "manas": {"status": "active", "language": "multi"},
            "vivek": {"status": "active", "quality_threshold": 0.7},
            "chitta": {"status": "active", "empathy_levels": 5},
            "smriti": {
                "short_term": len(self.smriti.short_term_memory),
                "long_term": len(self.smriti.long_term_memory),
            },
            "nirikshak": {"queue": self.learning_queue.stats()},
            "lora_experts": len(self.lora_expert.experts),
        }


if __name__ == "__main__":
    print("Initializing Tantra Cognitive OS...")
    os = CognitiveOS()
    
    # Demo interactions
    print("\n--- Demo Interactions ---")
    
    os.process("Hello!")
    os.process("Bhai, ye Python code optimize kar sakte hain?")
    os.process("What is gravity?")
    os.process("गणित क्या है?")
    
    # System status
    print("\n--- System Status ---")
    print(json.dumps(os.status(), indent=2))
    
    # Auto-learning demo
    print("\n--- Auto-Learning Demo ---")
    os.run_auto_learning()
