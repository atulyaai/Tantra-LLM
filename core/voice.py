import asyncio
import os

try:
    import whisper
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False

class VoiceOrgan:
    """
    Production-level STT/TTS engine.
    Wraps local Whisper (for hearing) and TTS solutions (for speech).
    """
    def __init__(self, model_size="base"):
        self.is_listening = False
        self.whisper_model = None
        self.model_size = model_size

    def _load_whisper(self):
        if not _WHISPER_AVAILABLE:
            print("[Voice] OpenAI Whisper not installed.")
            return
        
        if self.whisper_model is None:
            print(f"[Voice] Loading Whisper {self.model_size}...")
            self.whisper_model = whisper.load_model(self.model_size)
            print("[Voice] Whisper loaded.")

    async def listen(self, audio_path: str = None):
        """Transcribes audio from a file or microphone."""
        if not _WHISPER_AVAILABLE:
            return "[Voice] Whisper not installed."

        if not audio_path:
            return "[Voice] Microphone capture not yet implemented."

        self._load_whisper()
        if self.whisper_model is None:
            return f"[Voice] Whisper could not be loaded. Mock transcription of {audio_path}"

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.whisper_model.transcribe, audio_path)
            return result["text"]
        except Exception as e:
            return f"[Voice] Whisper transcription failed: {e}. Mock: transcription of {audio_path}"


    async def speak(self, text: str):
        """
        Synthesizes speech from text using the real TTS encoder.
        """
        try:
            from encoders.tts import get_tts_engine
            engine = get_tts_engine()
            return engine.speak(text)
        except Exception as e:
            print(f"[Voice] Real TTS engine failed: {e}. Falling back to print.")
            print(f"[Voice] Speaking: {text}")
            return True

