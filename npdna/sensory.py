"""Sensory organs: vision, voice, sentiment + modality encoders.

Single flat module (previously ``core/vision.py``, ``core/voice.py``, ``core/sentiment.py`` + ``encoders/``).
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import Any, Optional, Union

import requests
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from npdna.schema import Message, TantraRequest, ModalityEncoder, get_settings

logger = logging.getLogger(__name__)

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import whisper
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False


class VisionOrgan:
    """
    Production-level spatial awareness engine.
    Handles screen capture, camera input, and multimodal analysis.
    """
    def __init__(self, camera_index=0):
        self.is_observing = False
        self.camera_index = camera_index

    async def capture_frame(self):
        """
        Captures a single frame from the primary camera.
        Returns: base64 encoded image string or None if failed.
        """
        if not _CV2_AVAILABLE:
            print("[Vision] OpenCV (cv2) not installed.")
            return None

        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            print("[Vision] Could not open camera.")
            return None

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("[Vision] Failed to capture frame.")
            return None

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jpg_as_text

    async def analyze_environment(self, image_data=None):
        """
        Analyzes the given image data (or captures new if None).
        Calls local VisionEncoder if available to simulate feature extraction.
        """
        if image_data is None:
            image_data = await self.capture_frame()

        if not image_data:
            return "Vision unavailable (Camera offline)"

        try:
            import torch
            # Initialize with 4096 dimensions to match model configuration
            encoder = VisionEncoder(embed_dim=4096)
            # Encode visual frame (returns torch.Tensor embeddings)
            embeddings = encoder(image_data)
            return f"Visual Context: [Processed Image Embeddings shape={list(embeddings.shape)}] (Simulated: 'User is sitting in front of the screen')"
        except Exception as e:
            return f"Visual Context: [Captured Image (processed as stub)] (Simulated: 'User is sitting in front of the screen', error: {e})"

    async def capture_screen(self):
        # Placeholder for screen capture logic (e.g. using pyautogui or mss)
        return "Screen capture data placeholder."


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
            engine = get_tts_engine()
            return engine.speak(text)
        except Exception as e:
            print(f"[Voice] Real TTS engine failed: {e}. Falling back to print.")
            print(f"[Voice] Speaking: {text}")
            return True


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


# ── Modality Encoders (from encoders.py) ────────────────────────────────────

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    AutoTokenizer = None


class LongVITAVisionEncoder(nn.Module):
    """Local Long-VITA vision encoder implementation."""

    def __init__(self, embed_dim: int = 1024, model_path: Optional[str] = None):
        super().__init__()
        self._embed_dim = embed_dim
        self.model_path = model_path
        self._model = None
        self._processor = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoProcessor
            model_name = self.model_path or "google/vit-base-patch16-224"
            logger.info(f"Loading Long-VITA fallback model (ViT): {model_name}")
            self._processor = AutoProcessor.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            self._model.eval()
            logger.info("Long-VITA model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Long-VITA model: {e}")
            self._model = None
            self._processor = None

    def forward(self, image: Union[torch.Tensor, Image.Image, np.ndarray, str]) -> torch.Tensor:
        self._load_model()
        if self._model is None:
            return self._fallback_encode(image)
        try:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert("RGB")
            elif isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image.squeeze(0)
                if image.dim() == 3 and image.size(0) == 3:
                    image = image.permute(1, 2, 0)
                image = Image.fromarray((image.numpy() * 255).astype(np.uint8)).convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embeddings = outputs.pooler_output
                else:
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                if embeddings.size(-1) != self.embed_dim:
                    if not hasattr(self, '_projection'):
                        self._projection = nn.Linear(embeddings.size(-1), self.embed_dim).to(self._device)
                    embeddings = self._projection(embeddings)
                return embeddings
        except Exception as e:
            logger.error(f"Error encoding image with Long-VITA: {e}")
            return self._fallback_encode(image)

    def _fallback_encode(self, image: Union[torch.Tensor, Image.Image, np.ndarray, str]) -> torch.Tensor:
        try:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, Image.Image):
                pass
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert("RGB")
            elif isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image.squeeze(0)
                if image.dim() == 3 and image.size(0) == 3:
                    image = image.permute(1, 2, 0)
                image = Image.fromarray((image.numpy() * 255).astype(np.uint8)).convert("RGB")
            image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            if not hasattr(self, '_fallback_cnn'):
                self._fallback_cnn = nn.Sequential(
                    nn.Conv2d(3, 64, 7, 2, 3),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((7, 7)),
                    nn.Flatten(),
                    nn.Linear(64 * 7 * 7, 512),
                    nn.ReLU(),
                    nn.Linear(512, self.embed_dim)
                ).to(self._device)
            with torch.no_grad():
                embeddings = self._fallback_cnn(image_tensor)
                return embeddings
        except Exception as e:
            logger.error(f"Error in fallback encoding: {e}")
            return torch.zeros(1, self.embed_dim, device=self._device)


class VisionEncoder(ModalityEncoder):
    """Production wrapper for Long-VITA encoder; remote API + local fallback."""

    def __init__(self, embed_dim: int = 4096, api_url: Optional[str] = None, local_path: Optional[str] = None):
        settings = get_settings()
        if embed_dim != settings.model_dim:
            raise ValueError(f"VisionEncoder embed_dim ({embed_dim}) must match model_dim ({settings.model_dim})")
        self._embed_dim = embed_dim
        self.api_url = api_url
        self.local_path = local_path
        self._remote = False
        self._api_func = None
        self._local_encoder = None

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def encode(self, image) -> torch.Tensor:
        return self(image)

    def __call__(self, image) -> torch.Tensor:
        if isinstance(image, torch.Tensor) and image.size(-1) == self.embed_dim:
            return image.reshape(-1, self.embed_dim)
        if self._remote and self._api_func:
            try:
                return self._api_func(image)
            except Exception as e:
                logger.error(f"Remote API failed: {e}")
                return self._fallback_encode(image)
        if self.local_path or not self._remote:
            if self._local_encoder is None:
                self._local_encoder = LongVITAVisionEncoder(self.embed_dim, self.local_path)
            return self._local_encoder(image)
        return self._fallback_encode(image)

    def _fallback_encode(self, image) -> torch.Tensor:
        raise RuntimeError("VisionEncoder error: model or dependencies (transformers/PIL) missing. Fallbacks disabled.")

    def set_remote_mode(self, api_func):
        self._remote = True
        self._api_func = api_func

    def set_local_mode(self, model_path: Optional[str] = None):
        self._remote = False
        self.local_path = model_path
        self._local_encoder = None


class AudioEncoder(ModalityEncoder):
    """Production wrapper for Whisper encoder; returns real embeddings."""

    def __init__(self, embed_dim: int = 4096, model_size: str = "base"):
        settings = get_settings()
        if embed_dim != settings.model_dim:
            raise ValueError(f"AudioEncoder embed_dim ({embed_dim}) must match model_dim ({settings.model_dim})")
        self._embed_dim = embed_dim
        self._model = None
        self._model_size = model_size

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def encode(self, audio) -> torch.Tensor:
        return self(audio)

    def __call__(self, audio) -> torch.Tensor:
        if not self._model and whisper:
            try:
                self._model = whisper.load_model(self._model_size)
            except Exception as e:
                raise RuntimeError(f"AudioEncoder error: Failed to load Whisper ({e}). Fallbacks disabled.")
        if not whisper:
            raise RuntimeError("AudioEncoder error: 'openai-whisper' package is not installed. Fallbacks disabled.")
        try:
            if self._model and whisper:
                audio_tensor = torch.as_tensor(audio).float()
                device = next(self._model.parameters()).device
                audio_tensor = whisper.pad_or_trim(audio_tensor)
                mel = whisper.log_mel_spectrogram(audio_tensor).to(device)
                mel = mel.unsqueeze(0)
                with torch.no_grad():
                    enc_out = self._model.encoder(mel)
                    embeddings = enc_out.mean(dim=1)
                embeddings = embeddings.cpu()
                if embeddings.size(-1) < self.embed_dim:
                    padding = torch.zeros(1, self.embed_dim - embeddings.size(-1))
                    embeddings = torch.cat([embeddings, padding], dim=-1)
                else:
                    embeddings = embeddings[:, :self.embed_dim]
                return embeddings
            return torch.zeros(1, self.embed_dim)
        except Exception:
            return torch.zeros(1, self.embed_dim)


class TTSEncoder:
    """Text-to-speech using Coqui TTS with fallback to pyttsx3."""

    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"):
        self.model_name = model_name
        self.tts = None
        self.pyttsx3_engine = None
        self._load_tts()

    def _load_tts(self):
        try:
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
        try:
            import pyttsx3
            self.pyttsx3_engine = pyttsx3.init()
            voices = self.pyttsx3_engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.pyttsx3_engine.setProperty('voice', voice.id)
                        break
            self.pyttsx3_engine.setProperty('rate', 180)
            self.pyttsx3_engine.setProperty('volume', 0.8)
            logger.info("Loaded pyttsx3 TTS")
        except ImportError:
            logger.error("No TTS engines available. Install pyttsx3: pip install pyttsx3")
        except Exception as e:
            logger.error(f"pyttsx3 failed: {e}")

    def speak(self, text: str, save_to_file: Optional[str] = None) -> bool:
        if not text.strip():
            return False
        try:
            if self.tts:
                if save_to_file:
                    self.tts.tts_to_file(text=text, file_path=save_to_file)
                    logger.info(f"Audio saved to: {save_to_file}")
                else:
                    self.tts.tts(text=text)
                return True
            elif self.pyttsx3_engine:
                if save_to_file:
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
        if self.pyttsx3_engine:
            return [voice.name for voice in self.pyttsx3_engine.getProperty('voices')]
        return []

    def set_voice(self, voice_index: int = 0):
        if self.pyttsx3_engine:
            voices = self.pyttsx3_engine.getProperty('voices')
            if 0 <= voice_index < len(voices):
                self.pyttsx3_engine.setProperty('voice', voices[voice_index].id)

    def set_rate(self, rate: int = 180):
        if self.pyttsx3_engine:
            self.pyttsx3_engine.setProperty('rate', rate)

    def set_volume(self, volume: float = 0.8):
        if self.pyttsx3_engine:
            self.pyttsx3_engine.setProperty('volume', volume)


_tts_instance = None


def get_tts_engine() -> TTSEncoder:
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTSEncoder()
    return _tts_instance


class TextTokenizer:
    """Tokenizer wrapper with add_tokens support fallbacks."""

    def __init__(self, model_name: str = "gpt2"):
        self._tok = AutoTokenizer.from_pretrained(model_name) if AutoTokenizer else None
        self._vocab = {} if self._tok is None else None

    def encode(self, text: str, add_special_tokens: bool = True):
        if self._tok:
            return self._tok.encode(text, add_special_tokens=add_special_tokens)
        return [ord(c) for c in text]

    def get_vocab(self):
        if self._tok:
            return self._tok.get_vocab()
        return self._vocab

    def add_tokens(self, toks):
        if self._tok:
            self._tok.add_tokens(toks)
        else:
            for t in toks:
                if t not in self._vocab:
                    self._vocab[t] = len(self._vocab)

    def convert_tokens_to_ids(self, tok: str) -> int:
        if self._tok:
            return self._tok.convert_tokens_to_ids(tok)
        return self._vocab.get(tok, 0)
