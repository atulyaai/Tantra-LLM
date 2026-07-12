import asyncio
import uuid
from atulya_core.schema.agi import LogicNode, AGIPlan

class AGIPlanningSutra:
    """
    The High-Level Planning Engine.
    Compiles user intent into a serialized AGIPlan of LogicNodes.
    """
    def __init__(self):
        self.active_plans = {}

    async def compile_intent(self, user_intent: str) -> AGIPlan:
        # Generate a unique plan ID to prevent collisions
        plan_id = f"plan-{uuid.uuid4().hex[:6]}"
        
        intent_lower = user_intent.lower()
        nodes = []
        
        # Build logical plan nodes dynamically based on keywords in user intent
        if any(w in intent_lower for w in ["write", "code", "create", "implement", "develop"]):
            nodes = [
                LogicNode(id="node-1", instruction="Parse functional requirements and context rules"),
                LogicNode(id="node-2", instruction="Draft technical architecture and dependencies plan", dependencies=["node-1"]),
                LogicNode(id="node-3", instruction="Generate file modifications and new modules", dependencies=["node-2"]),
                LogicNode(id="node-4", instruction="Validate output files using test runner verification", dependencies=["node-3"]),
            ]
        elif any(w in intent_lower for w in ["fix", "debug", "error", "broken", "repair"]):
            nodes = [
                LogicNode(id="node-1", instruction="Reproduce the failure, inspect stack traces, and isolate faults"),
                LogicNode(id="node-2", instruction="Determine solution paths and modify affected modules", dependencies=["node-1"]),
                LogicNode(id="node-3", instruction="Run test suite to verify the target fix holds", dependencies=["node-2"]),
            ]
        else:
            # Default fallback routing plan
            nodes = [
                LogicNode(id="node-1", instruction="Parse user intent and extract contextual indicators"),
                LogicNode(id="node-2", instruction="Compile reasoning steps and direct query parameters", dependencies=["node-1"]),
                LogicNode(id="node-3", instruction="Formulate response output", dependencies=["node-2"]),
            ]
            
        plan = AGIPlan(
            plan_id=plan_id,
            goal=user_intent,
            nodes=nodes
        )
        
        # Register in active plans
        self.active_plans[plan_id] = plan
        return plan
