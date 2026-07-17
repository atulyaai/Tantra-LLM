"""Local adapter that exposes trained NP-DNA checkpoints to the fusion hub."""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

from npdna import NpDnaCore
from npdna.atulya_core.protocol.adapter import BaseTantraAdapter
from npdna.atulya_core.schema.models import ModelProvider, TantraRequest, TantraResponse
from npdna.cli import infer_max_tokens


class NpDnaAdapter(BaseTantraAdapter):
    """Run NP-DNA as the primary local model in ``UnifiedInferenceHub``."""

    DEFAULT_CHECKPOINTS = (Path("model/npdna/best"), Path("model/npdna/latest"))

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        core: Optional[NpDnaCore] = None,
    ) -> None:
        self.checkpoint_path = self._select_checkpoint(checkpoint_path)
        if core is None:
            if self.checkpoint_path is None:
                raise FileNotFoundError(
                    "No NP-DNA checkpoint found. Expected model/npdna/best or model/npdna/latest."
                )
            core = NpDnaCore.load(self.checkpoint_path)
            suffix = self.checkpoint_path.name if self.checkpoint_path else "injected"
        else:
            suffix = "injected"
        self.core = core
        self.model_name = f"npdna-{suffix}"

    @classmethod
    def _select_checkpoint(cls, requested: str | Path | None) -> Optional[Path]:
        if requested:
            path = Path(requested)
            return path if path.exists() else None
        return next((path for path in cls.DEFAULT_CHECKPOINTS if path.exists()), None)

    async def generate(self, request: TantraRequest) -> TantraResponse:
        start = time.perf_counter()
        prompt = request.messages[-1].content if request.messages else ""
        max_tokens = infer_max_tokens(prompt)
        text = await asyncio.to_thread(
            self.core.generate,
            prompt,
            max_tokens=max_tokens,
            temperature=request.temperature if request.temperature is not None else 0.35,
            top_k=30,
            top_p=request.top_p if request.top_p is not None else 0.9,
            context_window=256,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        return TantraResponse(
            content=text,
            model=self.model_name,
            provider=ModelProvider.LOCAL,
            usage={
                "prompt_tokens": len(self.core.encode(prompt, allow_growth=False)),
                "completion_tokens": len(self.core.encode(text, allow_growth=False)),
                "latency_ms": latency_ms,
            },
            cost=0.0,
            trace_id=request.trace_id or str(uuid.uuid4()),
        )

    async def stream(self, request: TantraRequest) -> AsyncGenerator[TantraResponse, None]:
        # The model generator is synchronous; keep the event loop responsive by
        # running the complete generation in a worker thread.
        yield await self.generate(request)

    def health_check(self) -> bool:
        return self.core is not None
