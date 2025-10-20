import datetime
import json
import os
import re
from typing import Any, Dict


def tool_web(args: Dict[str, Any]) -> Dict[str, Any]:
    # Placeholder simple web search using DuckDuckGo HTML (CPU-friendly, no API key)
    query = args.get("query", "").strip()
    if not query:
        return {"error": "missing query"}
    # We only echo the query to avoid network during initial scaffold
    return {"summary": f"Searched the web for: {query}", "results": []}


def tool_files(args: Dict[str, Any]) -> Dict[str, Any]:
    action = args.get("action", "read")
    path = args.get("path", "")
    if not path:
        return {"error": "missing path"}
    if action == "read":
        if not os.path.exists(path):
            return {"error": "not found"}
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return {"content": f.read()[:4000]}
    elif action == "write":
        content = args.get("content", "")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"ok": True}
    return {"error": f"unknown action {action}"}


def tool_calc(args: Dict[str, Any]) -> Dict[str, Any]:
    expr = str(args.get("expr", ""))
    if not expr:
        return {"error": "missing expr"}
    # extremely simple safe eval: numbers and operators only
    if not re.fullmatch(r"[0-9\s\+\-\*/\(\)\.]+", expr):
        return {"error": "invalid characters"}
    try:
        result = eval(expr, {"__builtins__": {}}, {})
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


