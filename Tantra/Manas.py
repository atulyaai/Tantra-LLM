"""
Manas — Input Processing & Language Detection Layer

Receives user input, detects language, intent, and routes to appropriate
cognitive sub-layer.

Architecture:
  User → Manas → (detect language/intent) → Vivek/Chitta/Smriti/Yantra
"""

class Manas:
    """Input processing and language detection."""
    
    def __init__(self):
        self.supported_languages = ["hindi", "hinglish", "english"]
        self.intents = ["question", "instruction", "conversation", "code", "math", "reasoning"]
    
    def detect_language(self, text):
        """Detect language of input text."""
        # Count Hindi/Devanagari characters
        hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        # Count English words
        english_words = sum(1 for w in text.split() if w.isascii() and len(w) > 1)
        
        if hindi_chars > 0 and english_words == 0:
            return "hindi"
        elif hindi_chars > 0 and english_words > 0:
            return "hinglish"
        elif english_words > 0:
            return "english"
        else:
            return "english"  # default
    
    def detect_intent(self, text):
        """Detect user intent from text."""
        text_lower = text.lower()
        
        # Code detection
        code_keywords = ["code", "function", "python", "program", "write", "debug", "api", "class"]
        # Math detection
        math_keywords = ["calculate", "what is", "solve", "equation", "number", "math", "sum"]
        # Reasoning detection
        reasoning_keywords = ["if", "what comes next", "how many", "why", "explain", "logic"]
        # Instruction detection
        instruction_keywords = ["teach", "show me", "how to", "guide", "explain"]
        
        for kw in code_keywords:
            if kw in text_lower:
                return "code"
        for kw in math_keywords:
            if kw in text_lower:
                return "math"
        for kw in reasoning_keywords:
            if kw in text_lower:
                return "reasoning"
        for kw in instruction_keywords:
            if kw in text_lower:
                return "instruction"
        
        # Conversation default
        return "conversation"
    
    def detect_emotion(self, text):
        """Detect emotional tone from text."""
        text_lower = text.lower()
        positive = ["thank", "great", "good", "help", "love", "best", "awesome"]
        negative = ["help", "problem", "error", "bug", "difficult", "hard"]
        
        if any(w in text_lower for w in ["thank", "thanks", "grateful", "appreciate"]):
            return "grateful"
        if any(w in text_lower for w in ["frustrated", "angry", "annoyed", "mad"]):
            return "frustrated"
        if any(w in text_lower for w in ["confused", "don't understand", "can't figure"]):
            return "confused"
        if any(w in text_lower for w in ["curious", "what if", "wonder", "imagine"]):
            return "curious"
        
        return "neutral"
    
    def process(self, text):
        """Process user input and return structured data."""
        return {
            "raw_text": text,
            "language": self.detect_language(text),
            "intent": self.detect_intent(text),
            "emotion": self.detect_emotion(text),
            "length": len(text),
            "word_count": len(text.split()),
        }
    
    def respond_in_language(self, text, language):
        """Ensure response matches user's language."""
        # This ensures Tantra speaks in the same register as the user
        # Hinglish user → Hinglish response
        # Hindi user → Hindi response
        # English user → English response
        return {"response": text, "language": language, "register": language}
