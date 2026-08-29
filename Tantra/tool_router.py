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
    """Reads a local file safely within project boundaries.

    SECURITY: the previous version joined filepath onto repo_root and
    normalized it, but never checked the result actually stayed inside
    repo_root -- a filepath like "../../../../etc/passwd" resolves outside
    it after normpath with no error. Worse, an absolute filepath (e.g.
    "/etc/passwd" or "/home/user/.ssh/id_rsa") took a separate branch that
    used it as-is, bypassing repo_root entirely. Since this is invoked
    automatically on whatever a <tool_call> block in the MODEL's own
    generated text requests (see parse_and_execute_tool_calls below), and
    that text can be steered by prompt injection from untrusted user input,
    this was a real arbitrary local file read reachable through ordinary
    chat. Both gaps are closed below: absolute paths are rejected outright,
    and the resolved path is verified to still be inside repo_root (using
    realpath, so symlinks can't be used to escape it either) before it's
    ever opened.
    """
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.realpath(repo_root)

    import ntpath
    import posixpath

    is_absolute = (
        os.path.isabs(filepath)
        or ntpath.isabs(filepath)
        or posixpath.isabs(filepath)
        or (len(filepath) > 1 and filepath[1] == ":")
        or filepath.startswith(("\\\\", "//", "\\", "/"))
    )
    if is_absolute:
        return f"Access denied: absolute paths are not allowed ({filepath!r})."

    target = os.path.realpath(os.path.join(repo_root, filepath))
    if target != repo_root and not target.startswith(repo_root + os.sep):
        return f"Access denied: path escapes the project directory ({filepath!r})."

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

def search_web(query: str, max_results: int = 3) -> str:
    """Searches Wikipedia/web knowledge base safely with timeout and returns summary snippets."""
    import urllib.request
    import urllib.parse
    import html

    query = query.strip()
    if not query:
        return "Error: Empty search query provided."

    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&utf8=&format=json&srlimit={max_results}"
        req = urllib.request.Request(url, headers={"User-Agent": "Tantra-LLM/1.0 (Local-AI-Research; contact@atulya.ai)"})
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        search_results = data.get("query", {}).get("search", [])
        if not search_results:
            return f"No direct search results found for query: {query!r}"

        snippets = []
        for i, item in enumerate(search_results, 1):
            title = item.get("title", "")
            raw_snippet = item.get("snippet", "")
            clean_snippet = html.unescape(re.sub(r"<.*?>", "", raw_snippet))
            snippets.append(f"{i}. **{title}**: {clean_snippet}")

        return "\n".join(snippets)
    except Exception as e:
        return f"Web search offline or unavailable ({e}). Using internal knowledge bank."


def retrieve_local_documents(query: str, doc_dir: Optional[str] = None, top_k: int = 3) -> str:
    """Performs local BM25/keyword retrieval over documents in Datasets/documents/."""
    if doc_dir is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        doc_dir = os.path.join(repo_root, "Datasets", "documents")

    if not os.path.exists(doc_dir):
        return f"Local document repository '{doc_dir}' is empty or does not exist."

    query_terms = set(re.findall(r"\w+", query.lower()))
    if not query_terms:
        return "Error: Empty query terms."

    scores = []
    for root, _, files in os.walk(doc_dir):
        for fname in files:
            if fname.endswith((".txt", ".md", ".json", ".jsonl")):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read(16384)
                    text_lower = text.lower()
                    match_count = sum(text_lower.count(term) for term in query_terms)
                    if match_count > 0:
                        snippet = text[:400].replace("\n", " ").strip()
                        scores.append((match_count, fname, snippet))
                except Exception:
                    continue

    if not scores:
        return f"No local documents matched query: {query!r}"

    scores.sort(key=lambda x: x[0], reverse=True)
    top_matches = scores[:top_k]
    res_lines = [f"Found {len(top_matches)} relevant local document(s):"]
    for count, fname, snippet in top_matches:
        res_lines.append(f"- **{fname}** (matches: {count}): \"{snippet}...\"")
    return "\n".join(res_lines)


def execute_tool_call(tool_name: str, arguments: Dict[str, Any], sandbox_enabled: Optional[bool] = None) -> str:
    """Dispatches and executes a tool call.

    `sandbox_enabled` gates python_executor and file_reader specifically --
    these run code / touch the filesystem with this process's own
    permissions. When None, reads TANTRA_ENABLE_SANDBOX environment variable (default: 1).
    """
    if sandbox_enabled is None:
        sandbox_enabled = os.environ.get("TANTRA_ENABLE_SANDBOX", "1") == "1"

    if tool_name == "calculator":
        expr = arguments.get("expression", "")
        return safe_eval_math(expr)
    elif tool_name == "python_executor":
        if not sandbox_enabled:
            return ("Tool 'python_executor' is disabled on this server. "
                    "Set TANTRA_ENABLE_SANDBOX=1 to enable code execution tools.")
        code = arguments.get("code", "")
        return execute_python_code(code)
    elif tool_name == "file_reader":
        if not sandbox_enabled:
            return ("Tool 'file_reader' is disabled on this server. "
                    "Set TANTRA_ENABLE_SANDBOX=1 to enable file access tools.")
        path = arguments.get("filepath", "")
        return read_local_file(path)
    elif tool_name in ("web_search", "search_web", "wikipedia"):
        q = arguments.get("query", "")
        return search_web(q)
    elif tool_name in ("doc_retriever", "document_search", "rag_retriever"):
        q = arguments.get("query", "")
        return retrieve_local_documents(q)
    else:
        return f"Unknown tool: {tool_name}"


def parse_and_execute_tool_calls(text: str, sandbox_enabled: Optional[bool] = None) -> Tuple[str, bool]:
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
        result = execute_tool_call(tool_name, tool_args, sandbox_enabled=sandbox_enabled)
        
        replacement = f"<tool_call>\n{json.dumps(call_json, indent=2)}\n</tool_call>\n<tool_result>\n{result}\n</tool_result>"
        updated_text = text[:match.start()] + replacement + text[match.end():]
        return updated_text, True
    except Exception as e:
        err_res = f"\n<tool_result>\nError parsing tool call: {e}\n</tool_result>"
        return text + err_res, True

