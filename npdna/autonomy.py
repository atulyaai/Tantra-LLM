"""NP-DNA Autonomy Layer.

Implements the NpDnaAgent which wraps NpDnaCore in a ReAct (Reasoning and Acting)
execution loop, allowing the model to run search and memory storage tools.
"""
from __future__ import annotations
import ast
import math
import operator
import re
import logging
from collections.abc import Callable as CallableABC
from typing import Any, Callable
import torch
from .model import NpDnaCore

logger = logging.getLogger(__name__)


class SafeExpressionError(ValueError):
    pass


_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos, ast.USub: operator.neg,
}


def _safe_pow(base: Any, exp: Any, mod: Any = None) -> Any:
    if not isinstance(exp, (int, float)):
        raise SafeExpressionError("exponent must be a number")
    if abs(exp) > 12:
        raise SafeExpressionError("exponent is too large")
    if mod is not None:
        return pow(base, exp, mod)
    return pow(base, exp)


_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "pow": _safe_pow, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "log": math.log, "log10": math.log10,
    "ceil": math.ceil, "floor": math.floor,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def safe_math_eval(expression: str) -> Any:
    if len(expression) > 512:
        raise SafeExpressionError("expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise SafeExpressionError("invalid expression") from exc
    return _eval_node(tree.body)


def safe_expression_output(expression: str) -> str:
    try:
        return str(safe_math_eval(expression))
    except Exception as exc:
        return f"Expression blocked: {str(exc)[:200]}"


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise SafeExpressionError("only numeric literals are allowed")
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise SafeExpressionError(f"unknown name: {node.id}")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise SafeExpressionError("operator is not allowed")
        left = _eval_node(node.left); right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise SafeExpressionError("exponent is too large")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise SafeExpressionError("operator is not allowed")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise SafeExpressionError("only named math functions are allowed")
        func = _FUNCTIONS.get(node.func.id)
        if func is None:
            raise SafeExpressionError(f"function is not allowed: {node.func.id}")
        if node.keywords:
            raise SafeExpressionError("keyword arguments are not allowed")
        args = [_eval_node(arg) for arg in node.args]
        return func(*args)
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_eval_node(elt) for elt in node.elts]
    raise SafeExpressionError(f"syntax is not allowed: {type(node).__name__}")


class NpDnaAgent:
    """ReAct-style autonomous agent wrapping NpDnaCore."""
    def __init__(self, core: NpDnaCore):
        self.core = core
        self.tools: dict[str, Callable[[str], str]] = {}
        
        # Register default tools
        self.register_tool("cortex_search", self._cortex_search)
        self.register_tool("cortex_store", self._cortex_store)
        self.register_tool("math_eval", self._math_eval)

    def register_tool(self, name: str, func: Callable[[str], str]) -> None:
        """Register a new python tool for the agent to call."""
        self.tools[name] = func
        logger.info("Tool '%s' registered with NpDnaAgent.", name)

    def _encode_to_vector(self, text: str) -> torch.Tensor:
        """Encode text to a dense vector via tokenizer + embedding mean."""
        token_ids = self.core.encode(text, allow_growth=False)
        if not token_ids:
            return torch.zeros(self.core.config.hidden_size)
        with torch.no_grad():
            token_t = torch.tensor(token_ids, dtype=torch.long, device=self.core.model.embedding.weight.device)
            embs = self.core.model.embedding(token_t)
            return embs.mean(dim=0)

    def _cortex_search(self, query: str) -> str:
        """Search memory cortex for query."""
        query_vec = self._encode_to_vector(query)
        if self.core.model.cortex.size == 0:
            return "Memory cortex is empty. No matching memories found."
        values, scores = self.core.model.cortex.retrieve(query_vec, top_k=2)
        if values is None or values.shape[0] == 0:
            return "No matching memories found."
        items = []
        last_indices = self.core.model.cortex._last_top_indices
        entries = self.core.model.cortex.entries
        for i in range(values.shape[0]):
            score = float(scores[i].item())
            # Use actual top_k index from retrieve result, not loop counter
            entry_idx = int(last_indices[0, i].item()) if last_indices is not None and i < last_indices.shape[1] else i
            if 0 <= entry_idx < len(entries):
                entry = entries[entry_idx]
                items.append(f"Match {i+1} (score={score:.3f}): {entry.source}")
            else:
                items.append(f"Match {i+1} (score={score:.3f}): [invalid index {entry_idx}]")
        return "\n".join(items) if items else "No matching memories found."

    def _cortex_store(self, fact: str) -> str:
        """Store a new fact in the memory cortex."""
        fact_clean = fact.strip()
        vector = self._encode_to_vector(fact_clean)
        self.core.model.cortex.store(key=vector, value=vector, topic="agent", source=fact_clean)
        if hasattr(self.core, "active_path") and self.core.active_path:
            self.core.model.cortex.save(self.core.active_path / "cortex")
        return f"Successfully saved fact to cortex: '{fact_clean[:80]}'"

    def _web_search(self, query: str) -> str:
        """Search the web using DuckDuckGo Instant Answer API (no key needed)."""
        import urllib.request
        import urllib.parse
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_redirect=1"
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                data = __import__("json").loads(r.read())
            result = data.get("AbstractText") or data.get("Answer") or data.get("RelatedTopics", [{}])[0].get("Text", "")
            if result:
                return result[:500]
            return f"No instant answer found for '{query}'."
        except Exception as e:
            return f"Web search failed: {str(e)[:200]}"

    def _code_execute(self, code: str) -> str:
        """Evaluate a single safe expression without launching a shell."""
        return safe_expression_output(code.strip())

    def _math_eval(self, expr: str) -> str:
        """Evaluate a mathematical expression safely."""
        return safe_expression_output(expr.strip())

    def run(self, user_prompt: str, max_iterations: int = 5) -> str:
        """Execute the ReAct loop until response is reached or iteration limit is hit.

        Args:
            user_prompt: User request.
            max_iterations: Max ReAct steps to prevent infinite loop.

        Returns:
            The final text response from the agent.
        """
        system_instructions = (
            "You are an autonomous NP-DNA agent. Solve the user's goal by thinking step-by-step "
            "and invoking tools. Supported tools:\n"
            "  - Action: cortex_search[query]\n"
            "  - Action: cortex_store[fact]\n\n"
            "Format your output strictly using these tags:\n"
            "[Thought] Explain your reasoning here.\n"
            "Action: tool_name[arguments]\n"
            "[Observation] Results will be shown here.\n"
            "Action: respond[your final response to the user]\n"
        )
        
        context = f"{system_instructions}\nUser: {user_prompt}\n"
        
        for iteration in range(max_iterations):
            logger.info("NpDnaAgent iteration %d/%d", iteration + 1, max_iterations)
            
            current_prompt = context + "[Thought]"
            
            output = self.core.generate(
                current_prompt,
                max_tokens=128,
                temperature=0.3,
                top_k=5
            )

            new_text = output.strip()
            logger.debug("Agent generated: %s", new_text)
            
            context += "[Thought] " + new_text + "\n"
            
            action_match = re.search(r"Action:\s*([a-zA-Z0-9_-]+)\[(.*?)\]", new_text)
            
            if not action_match:
                return new_text
            
            tool_name = action_match.group(1).strip()
            tool_arg = action_match.group(2).strip()
            
            if tool_name == "respond":
                return tool_arg
                
            if tool_name in self.tools:
                logger.info("Executing tool '%s' with arg: %s", tool_name, tool_arg)
                try:
                    observation = self.tools[tool_name](tool_arg)
                except Exception as e:
                    observation = f"Error executing tool: {str(e)}"
            else:
                observation = f"Unknown tool '{tool_name}'. Available: {list(self.tools.keys())}."
                
            logger.debug("Observation: %s", observation)
            context += f"[Observation] {observation}\n"
            
        return "Agent failed to reach a conclusion within step limit."

    def run_with_telemetry(self, user_prompt: str, max_iterations: int = 5) -> tuple[str, list[dict]]:
        """Execute the ReAct loop and return both final response and detailed step-by-step logs."""
        steps = []
        system_instructions = (
            "You are an autonomous NP-DNA agent. Solve the user's goal by thinking step-by-step "
            "and invoking tools. Supported tools:\n"
            "  - Action: cortex_search[query]\n"
            "  - Action: cortex_store[fact]\n\n"
            "Format your output strictly using these tags:\n"
            "[Thought] Explain your reasoning here.\n"
            "Action: tool_name[arguments]\n"
            "[Observation] Results will be shown here.\n"
            "Action: respond[your final response to the user]\n"
        )
        
        context = f"{system_instructions}\nUser: {user_prompt}\n"
        
        for iteration in range(max_iterations):
            logger.info("NpDnaAgent running telemetry step %d/%d", iteration + 1, max_iterations)
            current_prompt = context + "[Thought]"
            output = self.core.generate(
                current_prompt,
                max_tokens=128,
                temperature=0.3,
                top_k=5
            )
            
            new_text = output.strip()
            context += "[Thought] " + new_text + "\n"
            
            action_match = re.search(r"Action:\s*([a-zA-Z0-9_-]+)\[(.*?)\]", new_text)
            
            thought_text = new_text.split("Action:")[0].strip()
            
            step_info = {
                "step": iteration + 1,
                "thought": thought_text,
                "action": None,
                "args": None,
                "observation": None
            }
            
            if not action_match:
                step_info["action"] = "respond"
                step_info["args"] = new_text
                step_info["observation"] = "Direct completion"
                steps.append(step_info)
                return new_text, steps
            
            tool_name = action_match.group(1).strip()
            tool_arg = action_match.group(2).strip()
            step_info["action"] = tool_name
            step_info["args"] = tool_arg
            
            if tool_name == "respond":
                step_info["observation"] = "Final response generated"
                steps.append(step_info)
                return tool_arg, steps
                
            if tool_name in self.tools:
                try:
                    observation = self.tools[tool_name](tool_arg)
                except Exception as e:
                    observation = f"Error: {str(e)}"
            else:
                observation = f"Unknown tool '{tool_name}'."
                
            step_info["observation"] = observation
            steps.append(step_info)
            context += f"[Observation] {observation}\n"
            
        return "Agent failed to reach a conclusion within step limit.", steps
