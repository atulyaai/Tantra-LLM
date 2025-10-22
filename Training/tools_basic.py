import datetime
import json
import os
import re
import ast
import operator
from typing import Any, Dict
from pathlib import Path


def tool_web(args: Dict[str, Any]) -> Dict[str, Any]:
    # Placeholder simple web search using DuckDuckGo HTML (CPU-friendly, no API key)
    query = args.get("query", "").strip()
    if not query:
        return {"error": "missing query"}
    # We only echo the query to avoid network during initial scaffold
    return {"summary": f"Searched the web for: {query}", "results": []}


def _validate_path(path: str) -> bool:
    """Validate that path is safe and within allowed directories"""
    try:
        # Resolve the path to get absolute path
        abs_path = Path(path).resolve()
        
        # Define allowed directories (current workspace)
        workspace_root = Path("/workspace").resolve()
        
        # Check if path is within workspace
        if not str(abs_path).startswith(str(workspace_root)):
            return False
            
        # Check for path traversal attempts
        if ".." in path or path.startswith("/"):
            return False
            
        return True
    except Exception:
        return False


def tool_files(args: Dict[str, Any]) -> Dict[str, Any]:
    action = args.get("action", "read")
    path = args.get("path", "")
    
    if not path:
        return {"error": "missing path"}
    
    # Validate path for security
    if not _validate_path(path):
        return {"error": "invalid or unsafe path"}
    
    try:
        if action == "read":
            if not os.path.exists(path):
                return {"error": "not found"}
            
            # Additional size check
            file_size = os.path.getsize(path)
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                return {"error": "file too large"}
                
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return {"content": f.read()[:4000]}
                
        elif action == "write":
            content = args.get("content", "")
            
            # Check content size
            if len(content) > 100 * 1024:  # 100KB limit
                return {"error": "content too large"}
            
            # Create parent directories if needed
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"ok": True}
        else:
            return {"error": f"unknown action {action}"}
    except Exception as e:
        return {"error": f"file operation failed: {str(e)}"}


def safe_eval(expr: str) -> float:
    """Safely evaluate mathematical expressions using AST parsing"""
    # Allowed operators
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            return operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            return operators[type(node.op)](operand)
        else:
            raise ValueError(f"Unsupported operation: {type(node)}")
    
    try:
        tree = ast.parse(expr, mode='eval')
        return _eval(tree)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


def tool_calc(args: Dict[str, Any]) -> Dict[str, Any]:
    expr = str(args.get("expr", ""))
    if not expr:
        return {"error": "missing expr"}
    # extremely simple safe eval: numbers and operators only
    if not re.fullmatch(r"[0-9\s\+\-\*/\(\)\.\^]+", expr):
        return {"error": "invalid characters"}
    try:
        result = safe_eval(expr)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def tool_shell(args: Dict[str, Any]) -> Dict[str, Any]:
    # Placeholder no-op shell; returns the command for confirmation-only workflows
    cmd = args.get("cmd", "")
    return {"note": "shell execution disabled in scaffold", "cmd": cmd}


TOOL_SPECS = {
    "web": {
        "schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        "func": tool_web,
        "destructive": False,
    },
    "files": {
        "schema": {"type": "object", "properties": {"action": {"type": "string"}, "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["action", "path"]},
        "func": tool_files,
        "destructive": True,
    },
    "calc": {
        "schema": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]},
        "func": tool_calc,
        "destructive": False,
    },
    "shell": {
        "schema": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
        "func": tool_shell,
        "destructive": True,
    },
}
