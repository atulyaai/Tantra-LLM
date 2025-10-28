"""FastAPI server for Tantra-LLM production deployment."""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional

app = FastAPI(title="Tantra-LLM API", version="v0.8")


@app.post("/generate")
async def generate(
    text: str = Form(...),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
):
    """Generate response from text/vision/audio inputs."""
    # TODO: Load full brain orchestrator
    # brain = build_demo()
    # response = brain.step(text=text, image=image, audio=audio)
    
    return JSONResponse({
        "status": "success",
        "response": "Production API - brain loading not yet wired",
        "inputs": {"text": text, "has_image": image is not None, "has_audio": audio is not None}
    })


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "v0.8-inference_live"}

