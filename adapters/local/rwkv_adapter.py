import os
import time
import uuid
import asyncio
from typing import AsyncGenerator

from atulya_core.schema.models import TantraRequest, TantraResponse, ModelProvider
from atulya_core.protocol.adapter import BaseTantraAdapter

# Try importing rwkv, handle if not installed
try:
    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE, PIPELINE_ARGS
    _RWKV_AVAILABLE = True
except ImportError:
    _RWKV_AVAILABLE = False

class RWKVAdapter(BaseTantraAdapter):
    """
    RWKV-v4/v5/v6 Adapter for Tantra-LLM.
    Provides efficient RNN-based inference.
    """
    def __init__(self, model_path: str = "models/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth"):
        self.model_path = model_path
        self.model = None
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        if not _RWKV_AVAILABLE:
            print("[RWKV] Library not installed. Please install 'rwkv'.")
            return

        if not os.path.exists(self.model_path):
            print(f"[RWKV] Model not found at {self.model_path}. Running in MOCK mode.")
            return

        print(f"[RWKV] Loading model from {self.model_path}...")
        # Strategy: cuda fp16 for GPU, cpu fp32 for CPU
        strategy = "cuda fp16" if os.environ.get("USE_GPU") == "1" else "cpu fp32"
        
        try:
            self.model = RWKV(model=self.model_path, strategy=strategy)
            self.pipeline = PIPELINE(self.model, "rwkv_vocab_v20230424") # Standard World vocab
            print(f"[RWKV] Model loaded successfully with strategy: {strategy}")
        except Exception as e:
            print(f"[RWKV] Failed to load model: {e}")

    async def generate(self, request: TantraRequest) -> TantraResponse:
        start_time = time.time()
        prompt = request.messages[-1].content
        
        generated_text = ""
        
        if self.pipeline:
            # Real Inference
            # For simplicity, we use the pipeline's generate (synchronous, need to wrap or run in thread if blocking)
            # RWKV is fast enough for CPU usually, but let's be careful.
            
            args = PIPELINE_ARGS(
                temperature = request.temperature or 1.0,
                top_p = request.top_p or 0.7,
                top_k = 100, # default
                alpha_frequency = 0.25,
                alpha_presence = 0.25,
                token_ban = [], # ban the generation of some tokens
                token_stop = [], # stop generation whenever you see any token here
                chunk_len = 256 # split input into chunks to save VRAM (shorter -> slower)
            )
            
            # Running synchronous generation in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            generated_text = await loop.run_in_executor(
                None, 
                self.pipeline.generate, 
                prompt,
                args
            )
        else:
            # Mock Inference
            await asyncio.sleep(0.5)
            generated_text = f"[RWKV MOCK] Processed: '{prompt}'. (Model file missing)"

        latency = (time.time() - start_time) * 1000

        return TantraResponse(
            content=generated_text,
            model="rwkv-world",
            provider=ModelProvider.LOCAL,
            usage={"prompt_tokens": len(prompt)//4, "completion_tokens": len(generated_text)//4}, # Approx
            cost=0.0,
            trace_id=request.trace_id or str(uuid.uuid4()),
            entropy_score=0.1
        )

    async def stream(self, request: TantraRequest) -> AsyncGenerator[TantraResponse, None]:
        # RWKV Streaming implementation
        # For now, we'll just yield the full response as a single chunk for simplicity in this step
        # Ideally, we would adapt pipeline.generate to yield tokens.
        response = await self.generate(request)
        yield response

    def health_check(self) -> bool:
        return _RWKV_AVAILABLE and (self.model is not None or not os.path.exists(self.model_path))
