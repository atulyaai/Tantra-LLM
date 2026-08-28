# 🛠️ Tantra-LLM Tool Calling & Function Execution Manifest

Tantra features a native, sandboxed function calling execution engine (`Tantra/tool_router.py`):

| Tool | User Query | Tool Call Syntax | Execution Engine | Output |
|---|---|---|---|---|
| **🔢 Calculator** | *Calculate 45 * 12 + 15* | `<tool_call>{"name": "calculator", "arguments": {"expression": "45 * 12 + 15"}}</tool_call>` | AST Safe Evaluator | **`555`** |
| **💻 Python Sandbox** | *Generate the squares of numbers 1 to 5 using Python.* | `<tool_call>{"name": "python_interpreter", "arguments": {"code": "print([x**2 for x in range(1, 6)])"}}</tool_call>` | Isolated Subprocess | **`[1, 4, 9, 16, 25]`** |
| **📁 File Reader** | *Read the dataset manifest file.* | `<tool_call>{"name": "read_file", "arguments": {"path": "Datasets/manifest.json"}}</tool_call>` | Boundary Guarded I/O | **`417,242 items`** |
