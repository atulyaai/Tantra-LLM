import json
import time
from typing import Any, Dict, List, Optional, Tuple, Callable


class ToolSpec:
    def __init__(self, name: str, schema: Dict[str, Any], func: Callable[[Dict[str, Any]], Dict[str, Any]], destructive: bool = False):
        self.name = name
        self.schema = schema
        self.func = func
        self.destructive = destructive


class Agent:
    def __init__(self, config: Dict[str, Any], tools: Dict[str, ToolSpec], memory_adapter: Any, llm_infer: Callable[[str, Dict[str, Any]], str]):
        self.config = config
        self.tools = tools
        self.memory = memory_adapter
        self.llm_infer = llm_infer

    def _decide(self, prompt: str, tool_schemas: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        system = self._build_system(tool_schemas)
        plan_prompt = f"""
{system}
You're deciding whether to use a tool. If a tool is helpful, return JSON: {{"tool":"name","args":{{...}}}}. Otherwise return {{"final":"text"}}.
User: {prompt}
"""
        out = self.llm_infer(plan_prompt, {"temperature": 0.2})
        try:
            return json.loads(out)
        except Exception:
            return {"final": out}

    def _build_system(self, tool_schemas: Dict[str, Dict[str, Any]]) -> str:
        persona = self.config.get("persona", {})
        sys = persona.get("system_instructions", "You are a helpful assistant.")
        schemas = json.dumps(tool_schemas)
        return f"{sys}\nAvailable tools (JSON schemas): {schemas}"

    def _tool_schemas(self) -> Dict[str, Dict[str, Any]]:
        enabled = set(self.config.get("tools", {}).get("enabled", []))
        schemas: Dict[str, Dict[str, Any]] = {}
        for name, spec in self.tools.items():
            if name in enabled:
                schemas[name] = spec.schema
        return schemas

    def run(self, user_text: str, confirm_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None) -> Tuple[List[Dict[str, Any]], str]:
        traces: List[Dict[str, Any]] = []
        max_steps = int(self.config.get("planning", {}).get("react", {}).get("max_steps", 4))
        step = 0
        final_text = ""

        # incorporate memory context
        context = self.memory.build_context(user_text)
        prompt = f"Context:\n{context}\n\nUser: {user_text}"

        while step < max_steps:
            step += 1
            decision = self._decide(prompt, self._tool_schemas())
            traces.append({"type": "decision", "content": decision})

            if "final" in decision:
                final_text = decision["final"]
                break

            tool_name = decision.get("tool")
            args = decision.get("args", {})
            if not tool_name or tool_name not in self.tools:
                final_text = decision if isinstance(decision, str) else json.dumps(decision)
                break

            spec = self.tools[tool_name]
            # safety confirmation
            if spec.destructive and self.config.get("tools", {}).get("confirm_destructive", True):
                allowed = False
                if confirm_callback is not None:
                    allowed = confirm_callback(tool_name, args)
                if not allowed:
                    traces.append({"type": "tool_blocked", "tool": tool_name})
                    final_text = "I need your confirmation to proceed with that."
                    break

            try:
                tool_result = spec.func(args)
                traces.append({"type": "tool_result", "tool": tool_name, "result": tool_result})
                # incorporate tool result into prompt for next step
                prompt += f"\nTool[{tool_name}] => {json.dumps(tool_result)[:1000]}"
            except Exception as e:
                traces.append({"type": "tool_error", "tool": tool_name, "error": str(e)})
                break

        if not final_text:
            # produce final answer from accumulated context
            final_prompt = f"{prompt}\n\nNow provide the final helpful answer."\
            
            final_text = self.llm_infer(final_prompt, {"temperature": 0.7})

        self.memory.observe_turn(user_text, final_text, traces)
        return traces, final_text


