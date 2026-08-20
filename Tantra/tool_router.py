"""
Tantra.tool_router — Safe Tool & Function Calling Execution Engine for Tantra-LLM.
Executes calculator math, sandboxed Python code, and local file reading.
"""
import os
import ast
import json
import re
import sys
import subprocess
from typing import Dict, Any, Tuple, Optional

# Safe mathematical operator map for AST evaluation
SAFE_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
    ast.USub: lambda a: -a,
    ast.UAdd: lambda a: a,
}

def safe_eval_math(expr: str) -> str:
    """Evaluates mathematical expressions safely using Python AST without eval()."""
    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type in SAFE_OPS:
                return SAFE_OPS[op_type](left, right)
            raise ValueError(f"Unsupported math operator: {op_type}")
        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type in SAFE_OPS:
                return SAFE_OPS[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type}")
        raise ValueError(f"Unsupported expression node: {type(node)}")

    try:
        # Clean common symbols
        clean_expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/")
        parsed = ast.parse(clean_expr, mode='eval')
        res = _eval(parsed.body)
        return str(res)
    except Exception as e:
        return f"Error evaluating expression: {e}"

def execute_python_code(code: str, timeout_sec: int = 5) -> str:
    """Executes sandboxed Python code in a separate subprocess and captures stdout."""
    try:
        # Run in isolated subprocess
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_sec
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if proc.returncode != 0:
            return f"Runtime Error:\n{stderr}"
        return stdout if stdout else "Code executed successfully with no output."
    except subprocess.TimeoutExpired:
        return f"Execution timed out after {timeout_sec} seconds."
    except Exception as e:
        return f"Execution error: {e}"

def read_local_file(filepath: str, repo_root: Optional[str] = None) -> str:
    """Reads a local file safely within project boundaries."""
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    target = os.path.normpath(os.path.join(repo_root, filepath)) if not os.path.isabs(filepath) else os.path.normpath(filepath)
    
    if not os.path.exists(target):
        return f"File not found: {filepath}"
    if not os.path.isfile(target):
        return f"Target is a directory, not a file: {filepath}"
    
    try:
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(8192) # Cap to 8KB for safety
        return content
    except Exception as e:
        return f"Error reading file: {e}"

def execute_tool_call(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Dispatches and executes a tool call."""
    if tool_name == "calculator":
        expr = arguments.get("expression", "")
        return safe_eval_math(expr)
    elif tool_name == "python_executor":
        code = arguments.get("code", "")
        return execute_python_code(code)
    elif tool_name == "file_reader":
        path = arguments.get("filepath", "")
        return read_local_file(path)
    else:
        return f"Unknown tool: {tool_name}"

def parse_and_execute_tool_calls(text: str) -> Tuple[str, bool]:
    """
    Finds `<tool_call> ... </tool_call>` in model output, executes the tool,
    and returns the updated text with `<tool_result>` appended.
    """
    pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return text, False

    try:
        call_json = json.loads(match.group(1))
        tool_name = call_json.get("name", "")
        tool_args = call_json.get("arguments", {})
        result = execute_tool_call(tool_name, tool_args)
        
        replacement = f"<tool_call>\n{json.dumps(call_json, indent=2)}\n</tool_call>\n<tool_result>\n{result}\n</tool_result>"
        updated_text = text[:match.start()] + replacement + text[match.end():]
        return updated_text, True
    except Exception as e:
        err_res = f"\n<tool_result>\nError parsing tool call: {e}\n</tool_result>"
        return text + err_res, True
