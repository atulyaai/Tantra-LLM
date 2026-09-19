"""
Chitta — Emotional & Social Context Layer

Handles emotional intelligence, empathy, social awareness.
Decides tone, empathy level, and social context.

Architecture:
  Manas → Chitta → (emotion detection, empathy response) → Response
"""

class Chitta:
    """Emotional and social context processing."""
    
    def __init__(self):
        self.empathy_levels = {
            "grateful": "warm_appreciation",
            "frustrated": "patient_reassurance",
            "confused": "patient_guidance",
            "curious": "enthusiastic_exploration",
            "neutral": "friendly_neutral",
        }
        self.tone_adaptations = {
            "hindi": {"formal": False, "respectful": True, "use_bhai": True},
            "hinglish": {"formal": False, "casual": True, "use_bhai": True},
            "english": {"formal": False, "professional": True},
        }
    
    def detect_emotion(self, text):
        """Detect emotional tone from text."""
        text_lower = text.lower()
        
        if any(w in text_lower for w in ["dhanyavaad", "dhanyavad", "thanks", "thank you", "shukriya"]):
            return {"emotion": "grateful", "intensity": "high"}
        if any(w in text_lower for w in ["frustrated", "anger", "mad", "annoyed", "khed", "bura"]):
            return {"emotion": "frustrated", "intensity": "high"}
        if any(w in text_lower for w in ["confused", "don't understand", "samajh nahi", "nahi samajh"]):
            return {"emotion": "confused", "intensity": "medium"}
        if any(w in text_lower for w in ["curious", "wonder", "imagine", "soch", "sach"]):
            return {"emotion": "curious", "intensity": "medium"}
        if any(w in text_lower for w in ["hello", "hi", "namaste", "namaskar"]):
            return {"emotion": "greeting", "intensity": "low"}
        
        return {"emotion": "neutral", "intensity": "low"}
    
    def get_empathy_response(self, emotion, intensity):
        """Generate empathetic response tone."""
        level = self.empathy_levels.get(emotion["emotion"], "friendly_neutral")
        
        responses = {
            "grateful": {
                "hindi": "आपका स्वागत है! खुशी से मदद करता हूँ।",
                "hinglish": "Bhai, khushi hai! Koi aur help chahiye?",
                "english": "You're welcome! Happy to help!",
            },
            "frustrated": {
                "hindi": "कोई बात नहीं, धीरे से समझाता हूँ।",
                "hinglish": "Bhai, tension mat lo. Main step by step samjha dunga.",
                "english": "No worries! Let me explain step by step.",
            },
            "confused": {
                "hindi": "कोई बात नहीं, और सरल बनाकर समझाऊँगा।",
                "hinglish": "Bhai, koi baat nahi. Main aur simple samjha dunga.",
                "english": "No problem! Let me make it simpler.",
            },
            "curious": {
                "hindi": "बहुत अच्छा सवाल! और जानकारी देता हूँ।",
                "hinglish": "Bhai, bahut interesting question! More details:",
                "english": "Great question! Here's more detail:",
            },
            "greeting": {
                "hindi": "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?",
                "hinglish": "Hello bhai! Kaise hain? Mein kya help kar sakta hoon?",
                "english": "Hello! How can I help you today?",
            },
            "neutral": {
                "hindi": "मदद के लिए तैयार हूँ!",
                "hinglish": "Main ready hoon bhai! Bolo kya chahiye?",
                "english": "I'm ready to help! What do you need?",
            },
        }
        
        language = "hindi"  # default
        if emotion.get("language"):
            language = emotion["language"]
        
        return responses.get(emotion["emotion"], responses["neutral"]).get(language, responses["neutral"]["english"])
    
    def adapt_tone(self, text, language, emotion):
        """Adapt response tone based on language and emotion."""
        tone = self.tone_adaptations.get(language, self.tone_adaptations["english"])
        emotion_data = self.detect_emotion(text)
        
        return {
            "tone": emotion_data["emotion"],
            "language": language,
            "formality": tone.get("formal", False),
            "use_bhai": tone.get("use_bhai", False),
            "empathy_level": emotion_data["intensity"],
        }
    
    def social_context(self, user_text):
        """Analyze social context of user input."""
        text_lower = user_text.lower()
        
        context = {
            "formality": "casual" if "bhai" in text_lower or "dost" in text_lower else "standard",
            "relationship": "peer" if "bhai" in text_lower or "dost" in text_lower else "assistant",
            "urgency": "high" if any(w in text_lower for w in ["urgent", "jaldi", "quick", "fast"]) else "normal",
        }
        
        return context
    
    def process(self, text):
        """Full emotional processing."""
        emotion = self.detect_emotion(text)
        tone = self.adapt_tone(text, "hindi", emotion)
        context = self.social_context(text)
        
        return {
            "emotion": emotion,
            "tone": tone,
            "context": context,
            "empathy_response": self.get_empathy_response(emotion, tone),
        }
