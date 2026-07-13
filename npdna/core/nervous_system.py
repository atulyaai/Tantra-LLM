"""
The 20Hz Inter-Process Communication Bridge.
Broadcasts state changes across all modular organs.

Vendored from Atulya-Tantra/brain/core/nervous_system.py to eliminate
the hardcoded sys.path.append(r"d:\\Atulya Tantra\\Tantra-Bus") in
core/inference.py.
"""

import asyncio
from npdna.atulya_core.schema.agi import SystemPulse


class AtulyaNervousSystem:
    """
    The 20Hz Inter-Process Communication Bridge.
    Broadcasts state changes across all 15 modular organs.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 2020):
        self.host = host
        self.port = port
        self.subscribers = {}

    def subscribe(self, event_type: str, callback: callable):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    async def emit(self, event_type: str, data: dict):
        if event_type in self.subscribers:
            for cb in self.subscribers[event_type]:
                await cb(data)

    async def heartbeat(self):
        """Main 20Hz Loop"""
        while True:
            pulse = SystemPulse(cpu_load=0.0, mem_usage=0.0, active_modules=[])
            await self.emit("pulse", pulse.model_dump())
            await asyncio.sleep(0.05)
