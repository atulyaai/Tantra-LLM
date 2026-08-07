"""
tantra/server.py — Lightweight HTTP & OpenAI-compatible Chat API Server for Tantra-LLM.
Includes a built-in Web Chat UI served at GET /
"""
from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional

import torch

from Tantra.config import NeuroCoreConfig, VocabConfig
from Tantra.model import NeuroCoreModel
from Tantra.tokenizer import ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer
from Tantra.utils import get_logger

log = get_logger("tantra.server")

HTML_WEB_STUDIO = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tantra-LLM Studio</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-color: #f8fafc;
            --accent-color: #38bdf8;
            --accent-gradient: linear-gradient(135deg, #38bdf8, #818cf8);
            --user-msg-bg: #334155;
            --bot-msg-bg: #1e293b;
            --border-color: #334155;
        }

        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }

        header {
            background: var(--card-bg);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .logo {
            font-size: 1.5rem;
            font-weight: 700;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .status-badge {
            background: #065f46;
            color: #34d399;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        main {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 900px;
            width: 100%;
            margin: 0 auto;
            padding: 1rem;
            box-sizing: border-box;
        }

        #chat-window {
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            padding-right: 0.5rem;
        }

        .msg {
            max-width: 80%;
            padding: 1rem 1.25rem;
            border-radius: 12px;
            line-height: 1.5;
            word-wrap: break-word;
        }

        .msg.user {
            align-self: flex-end;
            background: #3b82f6;
            color: #fff;
            border-bottom-right-radius: 2px;
        }

        .msg.assistant {
            align-self: flex-start;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-bottom-left-radius: 2px;
        }

        .input-area {
            margin-top: 1rem;
            display: flex;
            gap: 0.75rem;
        }

        textarea {
            flex: 1;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            color: var(--text-color);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            resize: none;
            height: 50px;
            font-family: inherit;
            font-size: 1rem;
            outline: none;
        }

        textarea:focus {
            border-color: var(--accent-color);
        }

        button {
            background: var(--accent-gradient);
            color: #fff;
            border: none;
            padding: 0 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: opacity 0.2s;
        }

        button:hover {
            opacity: 0.9;
        }
    </style>
</head>
<body>
    <header>
        <div class="logo">Tantra-LLM Studio (v1.0.0)</div>
        <div class="status-badge">● Engine Online</div>
    </header>

    <main>
        <div id="chat-window">
            <div class="msg assistant">Hello! I am <b>Tantra-LLM</b> powered by the NeuroCore architecture. How can I assist you today?</div>
        </div>

        <div class="input-area">
            <textarea id="user-input" placeholder="Type a message..." onkeydown="if(event.key==='Enter' && !event.shiftKey){event.preventDefault(); sendMessage();}"></textarea>
            <button onclick="sendMessage()">Send</button>
        </div>
    </main>

    <script>
        async function sendMessage() {
            const inputEl = document.getElementById('user-input');
            const prompt = inputEl.value.trim();
            if (!prompt) return;

            const chatWindow = document.getElementById('chat-window');

            // User Message
            const userMsg = document.createElement('div');
            userMsg.className = 'msg user';
            userMsg.textContent = prompt;
            chatWindow.appendChild(userMsg);

            inputEl.value = '';
            chatWindow.scrollTop = chatWindow.scrollHeight;

            // Assistant Loading Msg
            const botMsg = document.createElement('div');
            botMsg.className = 'msg assistant';
            botMsg.textContent = 'Thinking...';
            chatWindow.appendChild(botMsg);
            chatWindow.scrollTop = chatWindow.scrollHeight;

            try {
                const response = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        messages: [{ role: 'user', content: prompt }],
                        max_tokens: 64,
                        temperature: 0.8
                    })
                });

                const data = await response.json();
                const reply = data.choices[0].message.content || 'Generated response complete.';
                botMsg.textContent = reply;
            } catch (err) {
                botMsg.textContent = 'Error connecting to Tantra-LLM backend.';
            }

            chatWindow.scrollTop = chatWindow.scrollHeight;
        }
    </script>
</body>
</html>
"""


class TantraHTTPHandler(BaseHTTPRequestHandler):
    """HTTP Handler supporting GET / (Web UI) and POST /v1/chat/completions (OpenAI API)."""

    model: Optional[NeuroCoreModel] = None
    tokenizer: Optional[UnifiedTokenizer] = None

    def log_message(self, format, *args):
        pass  # Suppress default HTTP logging to keep console clean

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_WEB_STUDIO.encode("utf-8"))
        elif self.path == "/health":
            res = {"status": "online", "model": "NeuroCore-138M", "version": "1.0.0"}
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path in ("/v1/chat/completions", "/chat/completions"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                req = json.loads(body)
                messages = req.get("messages", [])
                user_msg = ""
                for m in messages:
                    if m.get("role") == "user":
                        user_msg = m.get("content", "")

                max_tokens = req.get("max_tokens", 32)
                temp = req.get("temperature", 0.8)

                if self.model is not None and self.tokenizer is not None:
                    prompt_ids = self.tokenizer.encode(user_msg, modality="text")
                    if not prompt_ids:
                        prompt_ids = [1]
                    input_tensor = torch.tensor([prompt_ids], dtype=torch.long)

                    with torch.no_grad():
                        out_ids = self.model.generate(input_tensor, max_new_tokens=max_tokens, temperature=temp)

                    gen_ids = out_ids.tolist()[0]
                    reply_text = self.tokenizer.decode(gen_ids, modality="text")
                else:
                    reply_text = f"Tantra-LLM NeuroCore received prompt: '{user_msg}'"

                res = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "NeuroCore-138M",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": reply_text},
                            "finish_reason": "stop",
                        }
                    ],
                }

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
            except Exception as e:
                log.error(f"Error handling completion request: {e}")
                self.send_response(500)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


def serve(model: Optional[NeuroCoreModel] = None, tokenizer: Optional[UnifiedTokenizer] = None, port: int = 8000):
    """Start local Tantra-LLM HTTP server."""
    TantraHTTPHandler.model = model
    TantraHTTPHandler.tokenizer = tokenizer

    server = HTTPServer(("", port), TantraHTTPHandler)
    log.info(f"Starting Tantra-LLM Studio Web UI & OpenAI API on http://localhost:{port} ...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Server stopped.")
