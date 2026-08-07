import os
import json
import time
import http.server
import socketserver
import urllib.parse
from http import HTTPStatus

from Tantra.config import NeuroCoreConfig, VocabConfig
from Tantra.model import NeuroCoreModel
from Tantra.tokenizer import ByteBPETokenizer, MegabytePatcher, UnifiedTokenizer
from Tantra.utils import get_logger

logger = get_logger("Tantra.server")

# ==============================================================================
# HTML, CSS, JS Frontend Payload
# ==============================================================================

HTML_WEB_STUDIO = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>तन्त्र Tantra - Advanced LLM</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg-base: #0a0e1a;
            --bg-panel: rgba(22, 27, 45, 0.6);
            --bg-glass: rgba(31, 38, 62, 0.7);
            --bg-glass-hover: rgba(43, 51, 80, 0.8);
            
            --border-glass: rgba(255, 255, 255, 0.08);
            --border-glass-bright: rgba(255, 255, 255, 0.15);
            
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --text-dim: #6b7280;
            
            --accent-1: #6366f1;
            --accent-2: #8b5cf6;
            --accent-3: #a855f7;
            
            --gradient-primary: linear-gradient(135deg, var(--accent-1), var(--accent-2), var(--accent-3));
            
            --shadow-glass: 0 8px 32px 0 rgba(0, 0, 0, 0.36);
            --shadow-glow: 0 0 20px rgba(139, 92, 246, 0.3);
            
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 24px;
            --radius-full: 9999px;
            
            --transition-smooth: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-base);
            color: var(--text-main);
            height: 100vh;
            overflow: hidden;
            display: flex;
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 85% 30%, rgba(168, 85, 247, 0.15) 0%, transparent 50%);
            background-size: cover;
            background-position: center;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.2); border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.3); }

        .glass-panel {
            background: var(--bg-panel);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-glass);
        }

        /* Layout */
        #app-container {
            display: flex;
            width: 100%;
            height: 100%;
        }

        /* --- Left Sidebar --- */
        .sidebar {
            width: 280px;
            display: flex;
            flex-direction: column;
            border-right: 1px solid var(--border-glass);
            padding: 20px 16px;
            z-index: 10;
        }

        .model-card {
            background: var(--bg-glass);
            border-radius: var(--radius-lg);
            padding: 16px;
            margin-bottom: 24px;
            border: 1px solid var(--border-glass);
            box-shadow: var(--shadow-glass);
            position: relative;
            overflow: hidden;
        }
        
        .model-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 2px;
            background: var(--gradient-primary);
        }

        .model-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 4px;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }

        .model-badge {
            display: inline-block;
            background: rgba(99, 102, 241, 0.2);
            color: #c7d2fe;
            padding: 4px 8px;
            border-radius: var(--radius-full);
            font-size: 0.7rem;
            font-weight: 600;
            border: 1px solid rgba(99, 102, 241, 0.3);
            margin-top: 8px;
        }

        .sidebar-section-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-dim);
            margin-bottom: 12px;
            font-weight: 600;
        }

        .experts-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
            flex-grow: 1;
            overflow-y: auto;
        }

        .expert-item {
            display: flex;
            align-items: center;
            padding: 10px 12px;
            border-radius: var(--radius-md);
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-glass);
            transition: var(--transition-smooth);
        }

        .expert-item:hover {
            background: rgba(255, 255, 255, 0.06);
            transform: translateY(-1px);
        }

        .expert-icon {
            font-size: 1.2rem;
            margin-right: 12px;
            width: 24px;
            text-align: center;
        }

        .expert-details {
            flex-grow: 1;
        }

        .expert-name {
            font-size: 0.85rem;
            font-weight: 500;
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
        }

        .expert-status {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 5px #10b981;
            display: inline-block;
        }
        .expert-status.idle {
            background: var(--text-dim);
            box-shadow: none;
        }

        .expert-progress-bg {
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            overflow: hidden;
        }

        .expert-progress-bar {
            height: 100%;
            background: var(--gradient-primary);
            border-radius: 2px;
        }

        /* --- Center Chat --- */
        .chat-container {
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            position: relative;
        }

        .chat-header {
            padding: 16px 24px;
            border-bottom: 1px solid var(--border-glass);
            display: flex;
            justify-content: space-between;
            align-items: center;
            backdrop-filter: blur(8px);
            z-index: 5;
        }

        .chat-header h2 { font-size: 1.1rem; font-weight: 500; }
        
        .toggle-settings-btn {
            background: var(--bg-glass);
            border: 1px solid var(--border-glass);
            color: var(--text-main);
            padding: 6px 12px;
            border-radius: var(--radius-md);
            cursor: pointer;
            transition: var(--transition-smooth);
            font-size: 0.85rem;
        }
        .toggle-settings-btn:hover { background: var(--bg-glass-hover); }

        .chat-messages {
            flex-grow: 1;
            padding: 24px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        .message {
            display: flex;
            max-width: 85%;
            animation: fadeIn 0.3s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user { align-self: flex-end; flex-direction: row-reverse; }
        .message.assistant { align-self: flex-start; }

        .avatar {
            width: 36px;
            height: 36px;
            border-radius: var(--radius-full);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 0.9rem;
            flex-shrink: 0;
            border: 1px solid var(--border-glass);
        }

        .message.user .avatar {
            margin-left: 12px;
            background: rgba(255,255,255,0.1);
        }
        .message.assistant .avatar {
            margin-right: 12px;
            background: var(--gradient-primary);
            box-shadow: var(--shadow-glow);
        }

        .message-content {
            padding: 16px 20px;
            border-radius: var(--radius-xl);
            font-size: 0.95rem;
            line-height: 1.6;
            position: relative;
        }

        .message.user .message-content {
            background: var(--bg-glass);
            border: 1px solid var(--border-glass);
            border-top-right-radius: 4px;
        }

        .message.assistant .message-content {
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-top-left-radius: 4px;
        }

        .message-content p { margin-bottom: 8px; }
        .message-content p:last-child { margin-bottom: 0; }
        
        .message-content pre {
            background: rgba(0, 0, 0, 0.4);
            padding: 12px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 12px 0;
            border: 1px solid var(--border-glass);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }
        .message-content code {
            font-family: 'JetBrains Mono', monospace;
            background: rgba(0, 0, 0, 0.3);
            padding: 2px 4px;
            border-radius: 4px;
            font-size: 0.85em;
        }

        .message-meta {
            font-size: 0.7rem;
            color: var(--text-dim);
            margin-top: 8px;
            display: flex;
            gap: 12px;
            align-items: center;
        }
        .message.user .message-meta { justify-content: flex-end; }
        
        .copy-btn {
            background: none; border: none;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 0.7rem;
            transition: color 0.2s;
        }
        .copy-btn:hover { color: var(--text-main); }

        .input-container {
            padding: 24px;
            background: linear-gradient(to top, var(--bg-base) 60%, transparent);
            padding-bottom: 32px;
        }

        .input-wrapper {
            position: relative;
            background: var(--bg-glass);
            border: 1px solid var(--border-glass-bright);
            border-radius: var(--radius-xl);
            display: flex;
            align-items: flex-end;
            padding: 8px 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            transition: border-color 0.3s;
        }
        .input-wrapper:focus-within {
            border-color: var(--accent-2);
            box-shadow: var(--shadow-glow);
        }

        textarea {
            flex-grow: 1;
            background: transparent;
            border: none;
            color: var(--text-main);
            font-family: inherit;
            font-size: 1rem;
            padding: 12px 0;
            resize: none;
            max-height: 150px;
            min-height: 24px;
            outline: none;
            line-height: 1.5;
        }
        textarea::placeholder { color: var(--text-muted); }

        .send-btn {
            background: var(--gradient-primary);
            border: none;
            border-radius: 50%;
            width: 40px; height: 40px;
            display: flex;
            align-items: center; justify-content: center;
            cursor: pointer;
            color: white;
            margin-bottom: 4px;
            margin-left: 12px;
            flex-shrink: 0;
            transition: var(--transition-smooth);
            box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
        }
        .send-btn:hover { transform: scale(1.05); }
        .send-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

        /* --- Right Settings Panel --- */
        .settings-panel {
            width: 300px;
            border-left: 1px solid var(--border-glass);
            padding: 24px 20px;
            display: flex;
            flex-direction: column;
            gap: 24px;
            transition: transform 0.3s ease, width 0.3s ease;
            overflow-y: auto;
        }
        .settings-panel.collapsed {
            transform: translateX(100%);
            width: 0;
            padding: 0;
            border: none;
        }

        .control-group {
            background: rgba(255,255,255,0.02);
            border: 1px solid var(--border-glass);
            border-radius: var(--radius-md);
            padding: 16px;
        }

        .control-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 12px;
            font-size: 0.85rem;
            font-weight: 500;
        }

        .control-val {
            color: var(--accent-2);
            font-weight: 600;
        }

        /* Range Sliders */
        input[type=range] {
            -webkit-appearance: none;
            width: 100%;
            background: transparent;
        }
        input[type=range]:focus { outline: none; }
        input[type=range]::-webkit-slider-runnable-track {
            width: 100%; height: 6px;
            cursor: pointer;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
        }
        input[type=range]::-webkit-slider-thumb {
            height: 16px; width: 16px;
            border-radius: 50%;
            background: var(--text-main);
            cursor: pointer;
            -webkit-appearance: none;
            margin-top: -5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.5);
            transition: transform 0.1s;
        }
        input[type=range]::-webkit-slider-thumb:hover { transform: scale(1.2); }
        
        /* Toggle */
        .toggle-container {
            display: flex; justify-content: space-between; align-items: center;
        }
        .switch {
            position: relative; display: inline-block;
            width: 44px; height: 24px;
        }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider {
            position: absolute; cursor: pointer;
            top: 0; left: 0; right: 0; bottom: 0;
            background-color: rgba(255,255,255,0.1);
            transition: .4s; border-radius: 24px;
        }
        .slider:before {
            position: absolute; content: "";
            height: 18px; width: 18px;
            left: 3px; bottom: 3px;
            background-color: white;
            transition: .4s; border-radius: 50%;
        }
        input:checked + .slider { background: var(--gradient-primary); }
        input:checked + .slider:before { transform: translateX(20px); }

        .sys-prompt {
            width: 100%; height: 100px;
            background: rgba(0,0,0,0.2);
            border: 1px solid var(--border-glass);
            border-radius: var(--radius-md);
            color: var(--text-main);
            padding: 10px; font-size: 0.85rem;
            resize: none;
        }

        .btn-reset {
            width: 100%; padding: 10px;
            background: transparent;
            border: 1px solid var(--border-glass);
            color: var(--text-main);
            border-radius: var(--radius-md);
            cursor: pointer; font-weight: 500;
            transition: var(--transition-smooth);
        }
        .btn-reset:hover { background: rgba(255,255,255,0.05); }

        /* Typing indicator */
        .typing-indicator {
            display: none; padding: 16px 20px;
            background: rgba(99, 102, 241, 0.05);
            border: 1px solid rgba(99, 102, 241, 0.1);
            border-radius: var(--radius-xl);
            border-top-left-radius: 4px;
            align-self: flex-start;
            margin-left: 48px;
            width: fit-content;
        }
        .dot {
            display: inline-block; width: 6px; height: 6px;
            background: var(--accent-2); border-radius: 50%;
            margin: 0 2px; animation: bounce 1.4s infinite ease-in-out both;
        }
        .dot:nth-child(1) { animation-delay: -0.32s; }
        .dot:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }

        @media (max-width: 900px) {
            .sidebar { display: none; }
            .settings-panel { position: absolute; right: 0; height: 100%; z-index: 20; background: var(--bg-base); }
        }
    </style>
</head>
<body>
    <div id="app-container" class="glass-panel">
        
        <!-- Left Sidebar -->
        <aside class="sidebar">
            <div class="model-card">
                <div class="model-title">तन्त्र Tantra</div>
                <div style="font-size: 0.8rem; color: var(--text-muted);">Advanced Reasoning Engine</div>
                <div class="model-badge" id="model-params">Loading...</div>
            </div>

            <div class="sidebar-section-title">Expert Registry</div>
            <div class="experts-list" id="experts-container">
                <!-- Populated by JS -->
            </div>
        </aside>

        <!-- Center Chat -->
        <main class="chat-container">
            <header class="chat-header">
                <h2>Tantra-LLM Studio</h2>
                <button class="toggle-settings-btn" onclick="toggleSettings()">⚙️ Settings</button>
            </header>

            <div class="chat-messages" id="chat-messages">
                <div class="message assistant">
                    <div class="avatar">T</div>
                    <div>
                        <div class="message-content">
                            <p>Greetings. I am Tantra, an advanced reasoning engine powered by a Mixture of Experts architecture. How may I assist you today?</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="typing-indicator" id="typing-indicator">
                <div class="dot"></div><div class="dot"></div><div class="dot"></div>
            </div>

            <div class="input-container">
                <div class="input-wrapper">
                    <textarea id="user-input" placeholder="Type your message here... (Shift+Enter for new line)" rows="1" oninput="autoResize(this)" onkeydown="handleEnter(event)"></textarea>
                    <button class="send-btn" id="send-btn" onclick="sendMessage()">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                    </button>
                </div>
            </div>
        </main>

        <!-- Right Settings -->
        <aside class="settings-panel glass-panel" id="settings-panel">
            <h3 style="margin-bottom: 8px;">Generation Settings</h3>
            
            <div class="control-group">
                <div class="control-header">
                    <span>Temperature</span>
                    <span class="control-val" id="val-temp">0.7</span>
                </div>
                <input type="range" id="setting-temp" min="0" max="2" step="0.1" value="0.7" oninput="updateVal('temp', this.value); saveSettings()">
            </div>
            
            <div class="control-group">
                <div class="control-header">
                    <span>Top-P</span>
                    <span class="control-val" id="val-topp">0.9</span>
                </div>
                <input type="range" id="setting-topp" min="0" max="1" step="0.05" value="0.9" oninput="updateVal('topp', this.value); saveSettings()">
            </div>
            
            <div class="control-group">
                <div class="control-header">
                    <span>Max Tokens</span>
                    <span class="control-val" id="val-maxt">256</span>
                </div>
                <input type="range" id="setting-maxt" min="16" max="512" step="16" value="256" oninput="updateVal('maxt', this.value); saveSettings()">
            </div>

            <div class="control-group toggle-container">
                <span style="font-size: 0.85rem; font-weight: 500;">MTP Speculation</span>
                <label class="switch">
                    <input type="checkbox" id="setting-mtp" onchange="saveSettings()">
                    <span class="slider"></span>
                </label>
            </div>

            <div class="control-group">
                <div class="control-header"><span>System Prompt</span></div>
                <textarea class="sys-prompt" id="setting-sys" onchange="saveSettings()">You are Tantra, a highly advanced and helpful AI assistant.</textarea>
            </div>

            <button class="btn-reset" onclick="resetSettings()">Reset to Defaults</button>
        </aside>

    </div>

    <script>
        const chatMessages = document.getElementById('chat-messages');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const typingIndicator = document.getElementById('typing-indicator');
        const expertsContainer = document.getElementById('experts-container');
        
        let messageHistory = [];

        const expertIcons = {
            'language': '📝', 'code': '💻', 'math': '🔢', 'science': '🔬',
            'reasoning': '🧠', 'vision': '👁️', 'audio': '🎵', 'general': '🌐'
        };

        // UI Interactions
        function autoResize(el) {
            el.style.height = 'auto';
            el.style.height = (el.scrollHeight < 150 ? el.scrollHeight : 150) + 'px';
        }

        function toggleSettings() {
            document.getElementById('settings-panel').classList.toggle('collapsed');
        }

        function updateVal(id, val) {
            document.getElementById(`val-${id}`).innerText = val;
        }

        function handleEnter(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }

        function resetSettings() {
            document.getElementById('setting-temp').value = 0.7; updateVal('temp', 0.7);
            document.getElementById('setting-topp').value = 0.9; updateVal('topp', 0.9);
            document.getElementById('setting-maxt').value = 256; updateVal('maxt', 256);
            document.getElementById('setting-mtp').checked = false;
            document.getElementById('setting-sys').value = "You are Tantra, a highly advanced and helpful AI assistant.";
            saveSettings();
        }

        async function fetchStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('model-params').innerText = data.model + ' | ' + (data.params / 1e6).toFixed(1) + 'M';
            } catch (e) { console.error('Status fetch failed'); }
        }

        async function fetchExperts() {
            try {
                const res = await fetch('/api/experts');
                const data = await res.json();
                renderExperts(data.registry || []);
            } catch (e) { console.error('Experts fetch failed'); }
        }

        function renderExperts(experts) {
            if(!experts || experts.length === 0) {
                // Mock data if backend empty
                experts = Object.keys(expertIcons).map(k => ({name: k, usage: Math.floor(Math.random()*100)}));
            }
            
            let maxUsage = Math.max(...experts.map(e => e.usage || 1));
            
            expertsContainer.innerHTML = experts.map(e => {
                const icon = expertIcons[e.name.toLowerCase()] || '⚙️';
                const pct = ((e.usage || 0) / maxUsage) * 100;
                const statusClass = Math.random() > 0.5 ? 'active' : 'idle';
                
                return `
                <div class="expert-item">
                    <div class="expert-icon">${icon}</div>
                    <div class="expert-details">
                        <div class="expert-name">
                            <span style="text-transform: capitalize;">${e.name}</span>
                            <span class="expert-status ${statusClass}"></span>
                        </div>
                        <div class="expert-progress-bg">
                            <div class="expert-progress-bar" style="width: ${pct}%"></div>
                        </div>
                    </div>
                </div>`;
            }).join('');
        }

        function saveSettings() {
            const settings = {
                temperature: parseFloat(document.getElementById('setting-temp').value),
                top_p: parseFloat(document.getElementById('setting-topp').value),
                max_tokens: parseInt(document.getElementById('setting-maxt').value),
                mtp: document.getElementById('setting-mtp').checked,
                system: document.getElementById('setting-sys').value
            };
            fetch('/api/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(settings)
            }).catch(e => console.error(e));
        }

        // Chat functionality
        function formatTime() {
            const d = new Date();
            return `${d.getHours().toString().padStart(2,'0')}:${d.getMinutes().toString().padStart(2,'0')}`;
        }

        function appendMessage(role, content) {
            const timeStr = formatTime();
            const isUser = role === 'user';
            
            // marked parsing for assistant
            const htmlContent = isUser ? content.replace(/\\n/g, '<br>') : marked.parse(content);
            
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${role}`;
            
            const innerHTML = `
                <div class="avatar">${isUser ? 'U' : 'T'}</div>
                <div style="flex-grow: 1; max-width: calc(100% - 48px);">
                    <div class="message-content">${htmlContent}</div>
                    <div class="message-meta">
                        <span>${timeStr}</span>
                        ${!isUser ? `<button class="copy-btn" onclick="navigator.clipboard.writeText(\`${content.replace(/`/g, "'")}\`)">Copy</button>` : ''}
                    </div>
                </div>
            `;
            msgDiv.innerHTML = innerHTML;
            chatMessages.appendChild(msgDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        async function sendMessage() {
            const text = userInput.value.trim();
            if(!text) return;
            
            appendMessage('user', text);
            messageHistory.push({role: 'user', content: text});
            
            userInput.value = '';
            userInput.style.height = 'auto';
            sendBtn.disabled = true;
            typingIndicator.style.display = 'block';
            chatMessages.scrollTop = chatMessages.scrollHeight;

            try {
                const res = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        messages: [
                            {role: 'system', content: document.getElementById('setting-sys').value},
                            ...messageHistory
                        ]
                    })
                });
                
                const data = await res.json();
                const reply = data.choices[0].message.content;
                
                typingIndicator.style.display = 'none';
                appendMessage('assistant', reply);
                messageHistory.push({role: 'assistant', content: reply});
                
            } catch(e) {
                typingIndicator.style.display = 'none';
                appendMessage('assistant', 'Error communicating with server.');
            }
            sendBtn.disabled = false;
            fetchExperts(); // Update expert usage
        }

        // Init
        fetchStatus();
        fetchExperts();
        saveSettings();
    </script>
</body>
</html>
"""

# ==============================================================================
# Server Implementation
# ==============================================================================

class TantraHTTPHandler(http.server.BaseHTTPRequestHandler):
    expert_registry_data = None
    settings = {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 256,
        "mtp": False
    }

    def _set_headers(self, content_type='application/json'):
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers()

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path).path
        
        if parsed_path == '/':
            self._set_headers('text/html')
            self.wfile.write(HTML_WEB_STUDIO.encode('utf-8'))
            
        elif parsed_path == '/health':
            self._set_headers()
            self.wfile.write(json.dumps({
                "status": "ok",
                "model": "Tantra-178M"
            }).encode('utf-8'))
            
        elif parsed_path == '/api/status':
            self._set_headers()
            self.wfile.write(json.dumps({
                "model": "Tantra",
                "params": 178700000,
                "device": "cuda",
                "checkpoint": "latest"
            }).encode('utf-8'))
            
        elif parsed_path == '/api/experts':
            self._set_headers()
            self.wfile.write(json.dumps({
                "registry": TantraHTTPHandler.expert_registry_data or []
            }).encode('utf-8'))
            
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path).path
        
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b""
        
        try:
            req_json = json.loads(post_data.decode('utf-8')) if post_data else {}
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
            return

        if parsed_path == '/api/settings':
            # Update global settings
            TantraHTTPHandler.settings.update({
                "temperature": req_json.get("temperature", 0.7),
                "top_p": req_json.get("top_p", 0.9),
                "max_tokens": req_json.get("max_tokens", 256),
                "mtp": req_json.get("mtp", False)
            })
            self._set_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode('utf-8'))

        elif parsed_path == '/v1/chat/completions':
            # Mock or Real Inference here
            # Assuming self.server.model and self.server.tokenizer exist if passed
            time.sleep(1) # Simulated delay
            
            reply_text = "I am a highly advanced AI and I understand your request. (Mock Response)"
            
            # Simple mock response structure
            resp = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "Tantra-178M",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": reply_text
                    },
                    "finish_reason": "stop"
                }]
            }
            self._set_headers()
            self.wfile.write(json.dumps(resp).encode('utf-8'))

        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

def serve(model=None, tokenizer=None, port=8000, expert_dir=None):
    """
    Start the Tantra-LLM Web Studio server.
    """
    # Load expert registry if provided
    registry_data = None
    if expert_dir:
        registry_path = os.path.join(expert_dir, 'registry.json')
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r', encoding='utf-8') as f:
                    registry_data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load expert registry: {e}")
                
    TantraHTTPHandler.expert_registry_data = registry_data

    class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        pass

    server = ThreadedHTTPServer(('0.0.0.0', port), TantraHTTPHandler)
    server.model = model
    server.tokenizer = tokenizer
    
    logger.info(f"Starting Tantra-LLM Web Studio on http://0.0.0.0:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.server_close()
