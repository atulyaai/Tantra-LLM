from dataclasses import dataclass, field
from typing import List

@dataclass
class LogicNode:
    id: str
    instruction: str
    dependencies: List[str] = field(default_factory=list)

@dataclass
class AGIPlan:
    plan_id: str
    goal: str
    nodes: List[LogicNode] = field(default_factory=list)

@dataclass
class SystemPulse:
    """20Hz heartbeat payload broadcast by the Nervous System."""
    cpu_load: float = 0.0
    mem_usage: float = 0.0
    active_modules: List[str] = field(default_factory=list)

    def model_dump(self) -> dict:
        return {
            "cpu_load": self.cpu_load,
            "mem_usage": self.mem_usage,
            "active_modules": self.active_modules,
        }
