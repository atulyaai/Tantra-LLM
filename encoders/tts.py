from __future__ import annotations

"""Text-to-Speech encoder using Coqui TTS."""

import logging
import os
from typing import Optional, Any
import torch

logger = logging.getLogger(__name__)


class TTSEncoder:
    """Text-to-speech using Coqui TTS with fallback to pyttsx3."""

    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"):
        self.model_name = model_name
        self.tts = None
        self.pyttsx3_engine = None
        self._load_tts()

    def _load_tts(self):
        """Load TTS engine with fallbacks."""
        try:
            # Try Coqui TTS first
            from TTS.api import TTS
            self.tts = TTS(model_name=self.model_name, gpu=torch.cuda.is_available())
            logger.info(f"Loaded Coqui TTS: {self.model_name}")
        except ImportError:
            logger.warning("Coqui TTS not available, trying pyttsx3")
            self._load_pyttsx3()
        except Exception as e:
            logger.warning(f"Coqui TTS failed: {e}, trying pyttsx3")
            self._load_pyttsx3()

    def _load_pyttsx3(self):
        """Load pyttsx3 as fallback."""
        try:
            import pyttsx3
            self.pyttsx3_engine = pyttsx3.init()
            # Configure voice
            voices = self.pyttsx3_engine.getProperty('voices')
            if voices:
                # Try to find a female voice
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.pyttsx3_engine.setProperty('voice', voice.id)
                        break
            self.pyttsx3_engine.setProperty('rate', 180)  # Speed
            self.pyttsx3_engine.setProperty('volume', 0.8)  # Volume
            logger.info("Loaded pyttsx3 TTS")
        except ImportError:
            logger.error("No TTS engines available. Install pyttsx3: pip install pyttsx3")
        except Exception as e:
            logger.error(f"pyttsx3 failed: {e}")

    def speak(self, text: str, save_to_file: Optional[str] = None) -> bool:
        """Convert text to speech.

        Args:
            text: Text to speak
            save_to_file: Optional path to save audio file

        Returns:
            True if successful, False otherwise
        """
        if not text.strip():
            return False

        try:
            if self.tts:
                # Use Coqui TTS
                if save_to_file:
                    self.tts.tts_to_file(text=text, file_path=save_to_file)
                    logger.info(f"Audio saved to: {save_to_file}")
                else:
                    # Play directly (this might not work well in all environments)
                    self.tts.tts(text=text)
                return True

            elif self.pyttsx3_engine:
                # Use pyttsx3
                if save_to_file:
                    # pyttsx3 doesn't have direct file save, but we can try
                    self.pyttsx3_engine.save_to_file(text, save_to_file)
                    self.pyttsx3_engine.runAndWait()
                else:
                    self.pyttsx3_engine.say(text)
                    self.pyttsx3_engine.runAndWait()
                return True

            else:
                logger.error("No TTS engine available")
                return False

        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return False

    def get_available_voices(self) -> list:
        """Get list of available voices."""
        if self.pyttsx3_engine:
            return [voice.name for voice in self.pyttsx3_engine.getProperty('voices')]
        return []

    def set_voice(self, voice_index: int = 0):
        """Set TTS voice by index."""
        if self.pyttsx3_engine:
            voices = self.pyttsx3_engine.getProperty('voices')
            if 0 <= voice_index < len(voices):
                self.pyttsx3_engine.setProperty('voice', voices[voice_index].id)

    def set_rate(self, rate: int = 180):
        """Set speech rate."""
        if self.pyttsx3_engine:
            self.pyttsx3_engine.setProperty('rate', rate)

    def set_volume(self, volume: float = 0.8):
        """Set speech volume (0.0 to 1.0)."""
        if self.pyttsx3_engine:
            self.pyttsx3_engine.setProperty('volume', volume)


# Global TTS instance
_tts_instance = None

def get_tts_engine() -> TTSEncoder:
    """Get global TTS engine instance."""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTSEncoder()
    return _tts_instance</content>
</xai:function_call"> 

<xai:function_call name="write">
<parameter name="filePath">D:\Atulya\Tantra-LLM\demos\enhanced_gui.py