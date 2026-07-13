from typing import AsyncGenerator
from npdna.atulya_core.schema.models import TantraRequest, TantraResponse

class BaseTantraAdapter:
    """Base class for all Tantra model adapters."""
    async def generate(self, request: TantraRequest) -> TantraResponse:
        raise NotImplementedError("Adapter must implement generate.")

    async def stream(self, request: TantraRequest) -> AsyncGenerator[TantraResponse, None]:
        raise NotImplementedError("Adapter must implement stream.")

    def health_check(self) -> bool:
        raise NotImplementedError("Adapter must implement health_check.")
