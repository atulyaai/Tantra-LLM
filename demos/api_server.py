"""FastAPI server for Tantra-LLM production deployment."""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
import logging
import io
import torch
from PIL import Image

logger = logging.getLogger(__name__)

app = FastAPI(title="Tantra-LLM API", version="v1.0-stable_identity")

# Global brain orchestrator (loaded on startup)
_brain_orchestrator = None


def get_brain():
    """Load brain orchestrator on first request (lazy loading)."""
    global _brain_orchestrator
    if _brain_orchestrator is None:
        logger.info("Loading brain orchestrator...")
        from demos.demo_minimal import build_demo
        _brain_orchestrator = build_demo()
        logger.info("Brain orchestrator loaded successfully")
    return _brain_orchestrator


@app.post("/generate")
async def generate(
    text: str = Form(...),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
):
    """Generate response from text/vision/audio inputs."""
    try:
        brain = get_brain()
        
        # Process image if provided
        image_tensor = None
        if image is not None:
            image_bytes = await image.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # Convert to tensor (will be processed by vision encoder)
            import numpy as np
            image_tensor = torch.from_numpy(np.array(pil_image)).float() / 255.0
        
        # Process audio if provided
        audio_tensor = None
        if audio is not None:
            audio_bytes = await audio.read()
            # For now, pass raw bytes (will be processed by audio encoder)
            audio_tensor = audio_bytes
        
        # Generate response
        response = brain.step(text=text, image=image_tensor, audio=audio_tensor)
        
        return JSONResponse({
            "status": "success",
            "response": response,
            "inputs": {"text": text, "has_image": image is not None, "has_audio": audio is not None}
        })
        
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        return JSONResponse({
            "status": "error",
            "response": f"Error generating response: {str(e)}",
            "inputs": {"text": text, "has_image": image is not None, "has_audio": audio is not None}
        }, status_code=500)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "v1.0-stable_identity"}

