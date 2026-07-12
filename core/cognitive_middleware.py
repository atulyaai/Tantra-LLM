from typing import Callable, Awaitable, Dict, Any, Optional
import os
import json

from atulya_core.protocol.middleware import TantraMiddleware
from atulya_core.schema.models import TantraRequest, TantraResponse, Message, RequestContext
from personality.personality_layer import PersonalityLayer
from personality.safety_module import SafetyModule

class PersonalityMiddleware(TantraMiddleware):
    """Middleware that dynamically updates the model parameters and prompt prefix based on personality/tones."""
    def __init__(self, personality_config: Optional[Dict] = None):
        if personality_config is None:
            personality_config = {}
            config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "personality_config.json"))
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        personality_config = json.load(f)
                except Exception:
                    pass
        self.personality = PersonalityLayer(personality_config)

    async def __call__(
        self, 
        request: TantraRequest, 
        context: RequestContext, 
        call_next: Callable[[TantraRequest, RequestContext], Awaitable[TantraResponse]]
    ) -> TantraResponse:
        if request.messages:
            user_prompt = request.messages[-1].content
            mode = self.personality.select_mode(user_prompt)
            params = self.personality.parameterize(mode)
            
            # Store selected mode in RequestContext metadata
            context.metadata["personality_mode"] = mode
            
            # Inject prompt prefix as system context if present
            prefix = params.get("prompt_prefix")
            if prefix:
                system_msg = Message(role="system", content=prefix)
                # Keep system prefix at start
                request.messages.insert(0, system_msg)
                
            # Apply parameterized top_p/temperature if not set
            if request.temperature is None:
                request.temperature = params.get("temperature", 0.7)
            if request.top_p is None:
                request.top_p = params.get("top_p", 0.9)

        return await call_next(request, context)


class SafetyMiddleware(TantraMiddleware):
    """Middleware that audits the output response against safety standards (deny list and toxicity)."""
    def __init__(self):
        self.safety = SafetyModule()

    async def __call__(
        self, 
        request: TantraRequest, 
        context: RequestContext, 
        call_next: Callable[[TantraRequest, RequestContext], Awaitable[TantraResponse]]
    ) -> TantraResponse:
        # Safety middleware runs downstream of execution
        response = await call_next(request, context)

        safety_result = self.safety.evaluate(response.content, {})
        if safety_result["action"] == "deny":
            response.content = f"Response blocked by safety policy: {', '.join(safety_result['reasons'])}"
            response.confidence_level = "Blocked"
        elif safety_result["action"] == "modify":
            response.content = f"[Modified] {response.content}"

        return response
