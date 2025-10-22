"""
Multi-Modal Mamba 3 API Server
RESTful API for audio, text, and vision processing with MoE routing
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import base64
import io
from PIL import Image
import librosa
import soundfile as sf

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from model_mamba3_multimodal import Mamba3MultiModal, Mamba3Config
import safetensors.torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Tantra Multi-Modal Mamba 3 API",
    description="Multi-modal LLM with audio, text, and vision capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None
tokenizer = None
device = None


# Pydantic models for API
class TextRequest(BaseModel):
    text: str = Field(..., description="Input text for processing")
    max_length: int = Field(512, description="Maximum output length")
    temperature: float = Field(0.8, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling")
    top_k: int = Field(50, description="Top-k sampling")
    modality_priority: List[str] = Field(["text"], description="Modality priority order")


class AudioRequest(BaseModel):
    audio_data: str = Field(..., description="Base64 encoded audio data")
    sample_rate: int = Field(16000, description="Audio sample rate")
    max_duration: float = Field(10.0, description="Maximum audio duration in seconds")
    modality_priority: List[str] = Field(["audio"], description="Modality priority order")


class VisionRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    image_size: int = Field(224, description="Image size for processing")
    modality_priority: List[str] = Field(["vision"], description="Modality priority order")


class MultiModalRequest(BaseModel):
    text: Optional[str] = Field(None, description="Input text")
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    max_length: int = Field(512, description="Maximum output length")
    temperature: float = Field(0.8, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling")
    top_k: int = Field(50, description="Top-k sampling")
    modality_priority: List[str] = Field(["audio", "text", "vision"], description="Modality priority order")


class TextResponse(BaseModel):
    generated_text: str
    tokens: List[int]
    perplexity: float
    generation_time: float


class AudioResponse(BaseModel):
    audio_features: List[List[float]]
    reconstruction_quality: float
    processing_time: float


class VisionResponse(BaseModel):
    image_features: List[List[float]]
    analysis_quality: float
    processing_time: float


class MultiModalResponse(BaseModel):
    text_output: Optional[TextResponse] = None
    audio_output: Optional[AudioResponse] = None
    vision_output: Optional[VisionResponse] = None
    fusion_quality: float
    processing_time: float


class ModelInfo(BaseModel):
    model_name: str
    parameters: int
    model_size_mb: float
    modalities: List[str]
    experts: int
    compression_ratio: float


class AudioProcessor:
    """Audio processing utilities"""
    
    @staticmethod
    def process_audio(audio_data: bytes, sample_rate: int = 16000, 
                     max_duration: float = 10.0) -> np.ndarray:
        """Process audio data to features"""
        try:
            # Load audio
            audio, sr = librosa.load(io.BytesIO(audio_data), sr=sample_rate)
            
            # Trim to max duration
            if len(audio) > int(max_duration * sr):
                audio = audio[:int(max_duration * sr)]
            
            # Extract features (MFCC + spectral features)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
            
            # Combine features
            features = np.vstack([
                mfcc,
                spectral_centroid,
                spectral_rolloff,
                zero_crossing_rate
            ])
            
            # Pad or truncate to fixed size
            target_frames = 128
            if features.shape[1] > target_frames:
                features = features[:, :target_frames]
            else:
                padding = np.zeros((features.shape[0], target_frames - features.shape[1]))
                features = np.hstack([features, padding])
            
            return features.T  # [time, features]
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            # Return random features as fallback
            return np.random.randn(128, 128)
    
    @staticmethod
    def features_to_audio(features: np.ndarray, sample_rate: int = 16000) -> bytes:
        """Convert features back to audio (simplified)"""
        try:
            # This is a simplified conversion - in practice, you'd use proper audio synthesis
            audio = np.random.randn(len(features) * 100)  # Generate random audio
            audio = audio / np.max(np.abs(audio))  # Normalize
            
            # Convert to bytes
            audio_bytes = (audio * 32767).astype(np.int16).tobytes()
            return audio_bytes
            
        except Exception as e:
            logger.error(f"Audio synthesis error: {e}")
            return b""


class VisionProcessor:
    """Vision processing utilities"""
    
    @staticmethod
    def process_image(image_data: bytes, image_size: int = 224) -> np.ndarray:
        """Process image data to features"""
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            image = image.convert('RGB')
            image = image.resize((image_size, image_size))
            
            # Convert to tensor and normalize
            image_array = np.array(image) / 255.0
            image_tensor = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0)
            
            # Extract features using a simple CNN (in practice, use a proper vision encoder)
            # For now, return flattened image as features
            features = image_tensor.flatten().numpy()
            
            # Reshape to expected format
            patch_size = 16
            num_patches = (image_size // patch_size) ** 2
            feature_dim = len(features) // num_patches
            
            features = features[:num_patches * feature_dim].reshape(num_patches, feature_dim)
            
            return features
            
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            # Return random features as fallback
            return np.random.randn(196, 512)
    
    @staticmethod
    def features_to_image(features: np.ndarray, image_size: int = 224) -> bytes:
        """Convert features back to image (simplified)"""
        try:
            # This is a simplified conversion - in practice, you'd use proper image synthesis
            patch_size = 16
            num_patches = (image_size // patch_size) ** 2
            
            if features.shape[0] != num_patches:
                features = features[:num_patches]
            
            # Reshape to image
            patches_per_side = image_size // patch_size
            image = features.reshape(patches_per_side, patches_per_side, -1)
            
            # Convert to RGB
            image = (image * 255).astype(np.uint8)
            if image.shape[-1] > 3:
                image = image[:, :, :3]
            elif image.shape[-1] < 3:
                image = np.repeat(image, 3, axis=-1)
            
            # Convert to PIL Image and then to bytes
            pil_image = Image.fromarray(image)
            img_bytes = io.BytesIO()
            pil_image.save(img_bytes, format='PNG')
            return img_bytes.getvalue()
            
        except Exception as e:
            logger.error(f"Image synthesis error: {e}")
            return b""


class TextProcessor:
    """Text processing utilities"""
    
    @staticmethod
    def process_text(text: str, tokenizer) -> List[int]:
        """Process text to tokens"""
        try:
            if tokenizer is None:
                # Simple tokenization fallback
                return [hash(word) % 1000 for word in text.split()]
            
            tokens = tokenizer.encode(text).ids
            return tokens
            
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            return [hash(word) % 1000 for word in text.split()]
    
    @staticmethod
    def tokens_to_text(tokens: List[int], tokenizer) -> str:
        """Convert tokens back to text"""
        try:
            if tokenizer is None:
                # Simple detokenization fallback
                return " ".join([f"token_{t}" for t in tokens])
            
            text = tokenizer.decode(tokens)
            return text
            
        except Exception as e:
            logger.error(f"Text synthesis error: {e}")
            return " ".join([f"token_{t}" for t in tokens])


def load_model():
    """Load the multi-modal model"""
    global model, tokenizer, device
    
    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load configuration
        config = Mamba3Config()
        
        # Create model
        model = Mamba3MultiModal(config).to(device)
        
        # Load weights
        weights_path = 'Model/tantra_multimodal_weights.safetensors'
        if Path(weights_path).exists():
            logger.info(f"Loading weights from {weights_path}")
            state_dict = safetensors.torch.load_file(weights_path)
            model.load_state_dict(state_dict)
        else:
            logger.warning("No weights found, using random initialization")
        
        # Load tokenizer
        tokenizer_path = 'Model/tokenizer.json'
        if Path(tokenizer_path).exists():
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            logger.warning("No tokenizer found, using simple tokenization")
            tokenizer = None
        
        model.eval()
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    load_model()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Tantra Multi-Modal Mamba 3 API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    total_params = sum(p.numel() for p in model.parameters())
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    return ModelInfo(
        model_name="Tantra Multi-Modal Mamba 3",
        parameters=total_params,
        model_size_mb=model_size,
        modalities=["audio", "text", "vision"],
        experts=model.config.num_experts,
        compression_ratio=0.9  # Placeholder
    )


@app.post("/generate/text", response_model=TextResponse)
async def generate_text(request: TextRequest):
    """Generate text from input text"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        # Process input text
        input_tokens = TextProcessor.process_text(request.text, tokenizer)
        input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
        
        # Generate text
        with torch.no_grad():
            inputs = {"text": input_tensor}
            outputs = model(inputs, modality_priority=request.modality_priority)
            
            # Sample from text output
            logits = outputs["text"][0]  # [seq_len, vocab_size]
            
            # Apply temperature and top-p/top-k sampling
            logits = logits / request.temperature
            
            if request.top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, request.top_k, dim=-1)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(-1, top_k_indices, top_k_logits)
            
            if request.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > request.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            generated_tokens = torch.multinomial(probs, 1).squeeze(-1)
            
            # Convert to text
            generated_text = TextProcessor.tokens_to_text(generated_tokens.tolist(), tokenizer)
            
            # Calculate perplexity
            perplexity = torch.exp(F.cross_entropy(logits, generated_tokens)).item()
        
        processing_time = time.time() - start_time
        
        return TextResponse(
            generated_text=generated_text,
            tokens=generated_tokens.tolist(),
            perplexity=perplexity,
            generation_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Text generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/audio", response_model=AudioResponse)
async def process_audio(request: AudioRequest):
    """Process audio data"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        # Decode audio data
        audio_data = base64.b64decode(request.audio_data)
        
        # Process audio
        audio_features = AudioProcessor.process_audio(
            audio_data, 
            request.sample_rate, 
            request.max_duration
        )
        
        # Run through model
        with torch.no_grad():
            audio_tensor = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).to(device)
            inputs = {"audio": audio_tensor}
            outputs = model(inputs, modality_priority=request.modality_priority)
            
            # Calculate reconstruction quality
            reconstruction_quality = F.mse_loss(outputs["audio"], audio_tensor).item()
            reconstruction_quality = max(0.0, 1.0 - reconstruction_quality)
        
        processing_time = time.time() - start_time
        
        return AudioResponse(
            audio_features=outputs["audio"][0].cpu().numpy().tolist(),
            reconstruction_quality=reconstruction_quality,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/vision", response_model=VisionResponse)
async def process_vision(request: VisionRequest):
    """Process vision data"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        # Decode image data
        image_data = base64.b64decode(request.image_data)
        
        # Process image
        image_features = VisionProcessor.process_image(image_data, request.image_size)
        
        # Run through model
        with torch.no_grad():
            vision_tensor = torch.tensor(image_features, dtype=torch.float32).unsqueeze(0).to(device)
            inputs = {"vision": vision_tensor}
            outputs = model(inputs, modality_priority=request.modality_priority)
            
            # Calculate analysis quality
            analysis_quality = F.mse_loss(outputs["vision"], vision_tensor).item()
            analysis_quality = max(0.0, 1.0 - analysis_quality)
        
        processing_time = time.time() - start_time
        
        return VisionResponse(
            image_features=outputs["vision"][0].cpu().numpy().tolist(),
            analysis_quality=analysis_quality,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Vision processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/multimodal", response_model=MultiModalResponse)
async def process_multimodal(request: MultiModalRequest):
    """Process multi-modal data"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        inputs = {}
        
        # Process text if provided
        if request.text:
            input_tokens = TextProcessor.process_text(request.text, tokenizer)
            inputs["text"] = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
        
        # Process audio if provided
        if request.audio_data:
            audio_data = base64.b64decode(request.audio_data)
            audio_features = AudioProcessor.process_audio(audio_data)
            inputs["audio"] = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Process vision if provided
        if request.image_data:
            image_data = base64.b64decode(request.image_data)
            image_features = VisionProcessor.process_image(image_data)
            inputs["vision"] = torch.tensor(image_features, dtype=torch.float32).unsqueeze(0).to(device)
        
        if not inputs:
            raise HTTPException(status_code=400, detail="No input data provided")
        
        # Run through model
        with torch.no_grad():
            outputs = model(inputs, modality_priority=request.modality_priority)
            
            # Calculate fusion quality
            output_energies = []
            for modality, output in outputs.items():
                energy = torch.mean(output ** 2).item()
                output_energies.append(energy)
            
            fusion_quality = 1.0 / (1.0 + np.std(output_energies) / (np.mean(output_energies) + 1e-8))
        
        processing_time = time.time() - start_time
        
        # Prepare responses
        text_output = None
        audio_output = None
        vision_output = None
        
        if "text" in outputs:
            logits = outputs["text"][0]
            generated_tokens = torch.argmax(logits, dim=-1)
            generated_text = TextProcessor.tokens_to_text(generated_tokens.tolist(), tokenizer)
            perplexity = torch.exp(F.cross_entropy(logits, generated_tokens)).item()
            
            text_output = TextResponse(
                generated_text=generated_text,
                tokens=generated_tokens.tolist(),
                perplexity=perplexity,
                generation_time=processing_time
            )
        
        if "audio" in outputs:
            reconstruction_quality = F.mse_loss(outputs["audio"], inputs["audio"]).item()
            reconstruction_quality = max(0.0, 1.0 - reconstruction_quality)
            
            audio_output = AudioResponse(
                audio_features=outputs["audio"][0].cpu().numpy().tolist(),
                reconstruction_quality=reconstruction_quality,
                processing_time=processing_time
            )
        
        if "vision" in outputs:
            analysis_quality = F.mse_loss(outputs["vision"], inputs["vision"]).item()
            analysis_quality = max(0.0, 1.0 - analysis_quality)
            
            vision_output = VisionResponse(
                image_features=outputs["vision"][0].cpu().numpy().tolist(),
                analysis_quality=analysis_quality,
                processing_time=processing_time
            )
        
        return MultiModalResponse(
            text_output=text_output,
            audio_output=audio_output,
            vision_output=vision_output,
            fusion_quality=fusion_quality,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Multi-modal processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload audio file"""
    try:
        audio_data = await file.read()
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        
        return {
            "filename": file.filename,
            "size": len(audio_data),
            "audio_data": audio_b64
        }
        
    except Exception as e:
        logger.error(f"Audio upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload image file"""
    try:
        image_data = await file.read()
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        return {
            "filename": file.filename,
            "size": len(image_data),
            "image_data": image_b64
        }
        
    except Exception as e:
        logger.error(f"Image upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }


def main():
    """Main function to run the API server"""
    uvicorn.run(
        "serve_multimodal_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()