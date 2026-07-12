from typing import Any
from atulya_core.schema.models import TantraRequest, Message

class SentimentCore:
    """
    Production-level emotional resonance engine.
    Uses the Brain to analyze user-state and tone.
    """
    def __init__(self, adapter: Any = None):
        self.adapter = adapter

    async def analyze_vibe(self, text: str) -> str:
        """Asks the LLM to classify the sentiment of the text."""
        if not self.adapter:
            return "Neutral/Balanced (No Brain Connected)"

        # Construct a prompt for the model
        prompt = f"""
Instruction: Analyze the sentiment of the following text. Respond with one word: Positive, Negative, Neutral, Excited, Angry, or Sad.

Text: "{text}"

Sentiment:"""
        
        try:
            req = TantraRequest(messages=[Message(role="user", content=prompt)])
            response = await self.adapter.generate(req)
            return response.content.strip()
        except Exception as e:
            return f"Neutral (Analysis failed: {e})"

    def inject_humor(self, response: str) -> str:
        """
        Adds personality to the response.
        """
        return f"{response} ✨"
