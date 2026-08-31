let currentRole = 'user';
let currentSessionId = 'default';
let sessionFilter = 'active';

function setRole(role) {
    currentRole = role;
    const btnU = document.getElementById('btn-role-user');
    const btnA = document.getElementById('btn-role-admin');
    
    if (role === 'user') {
        if (btnU) btnU.className = 'role-btn active-user';
        if (btnA) btnA.className = 'role-btn';
        document.body.classList.remove('admin-mode');
        const activeTab = document.querySelector('.tab-btn.active');
        if (activeTab && activeTab.classList.contains('admin-only')) {
            switchTab('playground');
        }
    } else {
        if (btnU) btnU.className = 'role-btn';
        if (btnA) btnA.className = 'role-btn active-admin';
        document.body.classList.add('admin-mode');
    }
}
window.setRole = setRole;

function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));
    
    const btn = document.getElementById('tab-btn-' + tabName);
    if (btn) btn.classList.add('active');

    const view = document.getElementById('view-' + tabName);
    if (view) view.classList.add('active');

    try {
        if (tabName === 'telemetry') initCharts();
        if (tabName === 'moe') renderExperts();
        if (tabName === 'datasets') loadDatasets();
        if (tabName === 'kg') renderKnowledgeGraph();
    } catch(err) {
        console.error("Tab initialization error:", err);
    }
}
window.switchTab = switchTab;

function setSessionFilter(filter) {
    sessionFilter = filter;
    document.getElementById('btn-filter-active').className = filter === 'active' ? 'filter-btn active' : 'filter-btn';
    document.getElementById('btn-filter-archived').className = filter === 'archived' ? 'filter-btn active' : 'filter-btn';
    loadSessions();
}
window.setSessionFilter = setSessionFilter;

async function loadSessions() {
    try {
        const res = await fetch('/api/chats');
        const data = await res.json();
        const container = document.getElementById('sessions-list-container');
        
        const filtered = Object.values(data).filter(s => {
            const isArchived = Boolean(s.archived);
            return sessionFilter === 'archived' ? isArchived : !isArchived;
        });

        if (filtered.length === 0) {
            container.innerHTML = `<span style="color:var(--text-muted); font-size:0.78rem; padding:8px;">No ${sessionFilter} chats</span>`;
            return;
        }

        container.innerHTML = filtered.map(s => `
            <div class="chat-session-item ${s.id === currentSessionId ? 'active' : ''}" onclick="switchSession('${s.id}')">
                <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:180px;">💬 ${escapeHtml(s.title)}</span>
                <div class="session-actions" onclick="event.stopPropagation();">
                    <span class="session-action-icon" title="Rename" onclick="renameSession('${s.id}')">✏️</span>
                    <span class="session-action-icon" title="${s.archived ? 'Unarchive' : 'Archive'}" onclick="toggleArchive('${s.id}')">${s.archived ? '📤' : '📦'}</span>
                    <span class="session-action-icon" title="Delete" onclick="deleteSession('${s.id}')">🗑️</span>
                </div>
            </div>
        `).join('');
    } catch(e) {}
}

async function renameSession(sid) {
    const res = await fetch('/api/chats');
    const data = await res.json();
    const s = data[sid];
    const oldTitle = s ? s.title : "Chat Session";
    const newTitle = prompt("Rename Chat Session:", oldTitle);
    if (!newTitle || newTitle === oldTitle) return;

    await fetch(`/api/chats/${sid}/rename`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: newTitle })
    });
    loadSessions();
}
window.renameSession = renameSession;

async function toggleArchive(sid) {
    await fetch(`/api/chats/${sid}/archive`, { method: 'POST' });
    loadSessions();
}
window.toggleArchive = toggleArchive;

async function loadMemory() {
    try {
        const res = await fetch('/api/memory');
        const data = await res.json();
        const container = document.getElementById('memory-drawer-container');
        container.innerHTML = data.map(m => `
            <div class="memory-item">
                <div>
                    <span style="color:var(--cyan); font-weight:700;">[${m.category}]</span> ${escapeHtml(m.fact)}
                </div>
                <span style="cursor:pointer; color:var(--magenta);" onclick="deleteMemory('${m.id}')">×</span>
            </div>
        `).join('');
    } catch(e) {}
}

async function createNewSession() {
    const newId = 'session-' + Date.now();
    await fetch('/api/chats', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            id: newId,
            title: "New Conversation",
            archived: false,
            messages: [{"role": "assistant", "content": "Started new interactive session."}]
        })
    });
    currentSessionId = newId;
    setSessionFilter('active');
    switchSession(newId);
}
window.createNewSession = createNewSession;

async function switchSession(sid) {
    currentSessionId = sid;
    loadSessions();
    const res = await fetch('/api/chats');
    const data = await res.json();
    const session = data[sid];
    if (!session) return;

    const messagesDiv = document.getElementById('chat-messages');
    messagesDiv.innerHTML = session.messages.map(m => `
        <div class="message-row ${m.role}">
            <div class="message-avatar">${m.role === 'user' ? 'YOU' : '<img src="/assets/tantra_avatar_ai.jpg" onerror="this.src=\'/assets/tantra_logo.jpg\';">'}</div>
            <div class="message-bubble">${window.DOMPurify ? DOMPurify.sanitize(marked.parse(m.content)) : marked.parse(m.content)}</div>
        </div>
    `).join('');
    hljs.highlightAll();
}
window.switchSession = switchSession;

async function archiveCurrentSession() {
    toggleArchive(currentSessionId);
}
window.archiveCurrentSession = archiveCurrentSession;

async function deleteSession(sid) {
    if (!confirm("Are you sure you want to delete this chat session?")) return;
    await fetch(`/api/chats/${sid}`, { method: 'DELETE' });
    if (currentSessionId === sid) currentSessionId = 'default';
    loadSessions();
}
window.deleteSession = deleteSession;

async function addMemoryPrompt() {
    const fact = prompt("Enter fact for AI to remember:");
    if (!fact) return;
    await fetch('/api/memory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ category: 'User Fact', fact: fact })
    });
    loadMemory();
}
window.addMemoryPrompt = addMemoryPrompt;

async function deleteMemory(mid) {
    await fetch(`/api/memory/${mid}`, { method: 'DELETE' });
    loadMemory();
}
window.deleteMemory = deleteMemory;

async function autoSaveAndTitle(promptText, fullResponse) {
    try {
        const res = await fetch('/api/chats');
        const data = await res.json();
        let session = data[currentSessionId];
        if (!session) {
            session = { id: currentSessionId, title: "New Conversation", archived: false, messages: [] };
        }

        if (!session.messages) session.messages = [];
        session.messages.push({"role": "user", "content": promptText});
        session.messages.push({"role": "assistant", "content": fullResponse});

        const isGeneric = !session.title || session.title.includes("New Conversation") || session.title.includes("New AI Chat") || session.title.includes("Welcome Session");
        if (isGeneric && promptText) {
            const tRes = await fetch('/api/chats/auto_title', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: promptText })
            });
            const tData = await tRes.json();
            if (tData.title) session.title = tData.title;
        }

        await fetch('/api/chats', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(session)
        });
        loadSessions();
    } catch(e) {}
}

async function sendMessage() {
    const input = document.getElementById('prompt-input');
    const prompt = input.value.trim();
    if (!prompt) return;

    const messagesDiv = document.getElementById('chat-messages');
    
    const userRow = document.createElement('div');
    userRow.className = 'message-row user';
    userRow.innerHTML = `<div class="message-avatar">YOU</div><div class="message-bubble">${escapeHtml(prompt)}</div>`;
    messagesDiv.appendChild(userRow);
    
    input.value = '';
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    const assistantRow = document.createElement('div');
    assistantRow.className = 'message-row assistant';
    const msgId = 'reply-' + Date.now();
    assistantRow.innerHTML = `
        <div class="message-avatar"><img src="/assets/tantra_avatar_ai.jpg" onerror="this.src='/assets/tantra_logo.jpg';"></div>
        <div class="message-bubble" id="${msgId}">⚡ <i>Thinking...</i></div>
    `;
    messagesDiv.appendChild(assistantRow);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    const temp = parseFloat(document.getElementById('inp-temp').value);
    const top_p = parseFloat(document.getElementById('inp-topp').value);

    try {
        const res = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: [{ role: 'user', content: prompt }],
                temperature: temp, top_p: top_p, max_tokens: 64, stream: true
            })
        });

        if (!res.ok) {
            // Previously: res.body.getReader() was called unconditionally,
            // so a 500 (e.g. model failed to load) streamed back plain
            // error text that never matched the "data: " SSE prefix filter
            // below -- it was silently dropped, leaving the "Thinking..."
            // bubble cleared to empty with no error shown. This is why the
            // UI could look permanently stuck instead of failing visibly.
            let detail = res.statusText;
            try { detail = (await res.json()).detail || detail; } catch (e) { /* not JSON */ }
            throw new Error(`Server error ${res.status}: ${detail}`);
        }

        const replyDiv = document.getElementById(msgId);
        replyDiv.innerHTML = '';
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            for (let line of lines) {
                if (line.startsWith('data: ') && !line.includes('[DONE]')) {
                    try {
                        const parsed = JSON.parse(line.slice(6));
                        const delta = parsed.choices[0].delta.content || '';
                        fullText += delta;
                        const htmlOutput = marked.parse(fullText);
                        replyDiv.innerHTML = window.DOMPurify ? DOMPurify.sanitize(htmlOutput) : htmlOutput;
                        messagesDiv.scrollTop = messagesDiv.scrollHeight;
                    } catch (e) {}
                }
            }
        }
        hljs.highlightAll();
        autoSaveAndTitle(prompt, fullText);
    } catch (err) {
        document.getElementById(msgId).innerText = "Failed: " + err.message;
    }
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}
window.sendMessage = sendMessage;

async function renderKnowledgeGraph() {
    try {
        const res = await fetch('/api/knowledge_graph');
        const data = await res.json();
        const svg = document.getElementById('kg-svg');
        if (!svg) return;
        svg.innerHTML = '';

        data.links.forEach(l => {
            const sourceNode = data.nodes.find(n => n.id === l.source);
            const targetNode = data.nodes.find(n => n.id === l.target);
            if (sourceNode && targetNode) {
                const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                line.setAttribute("x1", sourceNode.x);
                line.setAttribute("y1", sourceNode.y);
                line.setAttribute("x2", targetNode.x);
                line.setAttribute("y2", targetNode.y);
                line.setAttribute("stroke", "rgba(0, 243, 255, 0.3)");
                line.setAttribute("stroke-width", "2");
                svg.appendChild(line);
            }
        });

        data.nodes.forEach(n => {
            const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            circle.setAttribute("cx", n.x);
            circle.setAttribute("cy", n.y);
            circle.setAttribute("r", n.type === 'concept' ? "18" : "12");
            circle.setAttribute("fill", n.color);
            circle.setAttribute("stroke", "#ffffff");
            circle.setAttribute("stroke-width", "2");
            circle.style.cursor = "pointer";
            circle.onclick = () => inspectNode(n);
            svg.appendChild(circle);

            const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
            text.setAttribute("x", n.x);
            text.setAttribute("y", n.y + 30);
            text.setAttribute("fill", "#ffffff");
            text.setAttribute("font-size", "12");
            text.setAttribute("text-anchor", "middle");
            text.setAttribute("font-weight", "bold");
            text.style.cursor = "pointer";
            text.onclick = () => inspectNode(n);
            text.textContent = n.label;
            svg.appendChild(text);
        });
    } catch(e) {}
}

async function renderExperts() {
    try {
        const res = await fetch('/api/experts');
        const data = await res.json();
        const container = document.getElementById('moe-expert-grid');
        if (!container) return;
        container.innerHTML = data.experts.map(e => `
            <div class="expert-card ${e.load > 50 ? 'active-expert' : ''}">
                <div class="expert-icon-badge">${e.icon}</div>
                <div class="expert-title">Expert #${e.id} <span>${e.load}%</span></div>
                <div class="expert-spec">${e.name}</div>
                <div style="font-size: 0.78rem; color: var(--text-muted);">${e.arch}</div>
                <div class="expert-load-bar"><div class="expert-load-fill" style="width: ${e.load}%;"></div></div>
            </div>
        `).join('');
    } catch(e) {}
}

async function loadDatasets() {
    try {
        const res = await fetch('/api/datasets');
        const data = await res.json();
        const container = document.getElementById('datasets-list-container');
        if (!container) return;
        container.innerHTML = data.map((d, idx) => `
            <div class="ds-item ${idx===0?'active':''}" onclick="viewDatasetSample('${d.id}')">
                <div class="ds-title"><span>${d.name}</span> <span style="font-size:0.75rem; color:var(--emerald);">${d.status}</span></div>
                <div class="ds-meta"><span>${d.samples.toLocaleString()} samples</span> • <span>${d.tokens}</span></div>
            </div>
        `).join('');
        if (data.length > 0) viewDatasetSample(data[0].id);
    } catch(e) {}
}

async function viewDatasetSample(id) {
    try {
        const res = await fetch('/api/datasets');
        const data = await res.json();
        const ds = data.find(d => d.id === id);
        if (!ds) return;
        const viewer = document.getElementById('ds-sample-viewer');
        if (!viewer) return;
        viewer.innerHTML = `
            <h3 style="color:var(--cyan); margin-bottom:8px;">${ds.name}</h3>
            <p style="font-size:0.85rem; color:var(--text-sub); margin-bottom:12px;">${ds.description}</p>
            <pre><code class="json">${escapeHtml(JSON.stringify(ds.sample, null, 2))}</code></pre>
        `;
        hljs.highlightAll();
    } catch(e) {}
}
window.viewDatasetSample = viewDatasetSample;

async function runDatasetCleaner() {
    try {
        const res = await fetch('/api/datasets/clean', { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || `Server error ${res.status}`);
        alert(data.message || "Dataset cleaned successfully!");
    } catch(e) {
        alert("Dataset cleaner: " + e.message);
    }
}
window.runDatasetCleaner = runDatasetCleaner;

async function runSandboxCode() {
    const code = document.getElementById('sandbox-code-input').value;
    const outputDiv = document.getElementById('sandbox-output');
    outputDiv.innerText = "Running code in isolated sandbox...";
    try {
        const res = await fetch('/api/sandbox/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code: code })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || `Server error ${res.status}`);
        outputDiv.innerText = `[${data.time_ms}ms] ` + data.result;
    } catch(e) {
        outputDiv.innerText = "Error executing sandbox: " + e.message;
    }
}
window.runSandboxCode = runSandboxCode;

async function inspectTokenization() {
    const text = document.getElementById('tok-input-text').value;
    const chipsDiv = document.getElementById('tok-chips');
    if (!text) {
        chipsDiv.innerHTML = '';
        return;
    }
    try {
        const res = await fetch('/api/tokenize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        const data = await res.json();
        chipsDiv.innerHTML = data.tokens.map(t => `
            <div class="token-chip" title="Token ID: ${t.id}">
                ${escapeHtml(t.text)} <span style="font-size:0.65rem; opacity:0.7;">#${t.id}</span>
            </div>
        `).join('');
    } catch(e) {}
}
window.inspectTokenization = inspectTokenization;

async function switchCheckpoint(ckpt) {
    try {
        const res = await fetch('/api/checkpoints', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ checkpoint: ckpt })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || `Server error ${res.status}`);
        alert(`Active checkpoint set to: ${data.active}`);
    } catch(e) {
        alert("Checkpoint switch failed: " + e.message);
    }
}
window.switchCheckpoint = switchCheckpoint;

function setPreset(type) {
    const promptInput = document.getElementById('prompt-input');
    if (type === 'coder') promptInput.value = "Write a high-performance Python LRU Cache with O(1) ops and thread safety.";
    if (type === 'math') promptInput.value = "Solve the differential equation dy/dx + 2y = e^(-x) with initial condition y(0) = 1.";
    if (type === 'science') promptInput.value = "Explain the BitNet 1.58-bit ternary quantization mechanism and energy efficiency vs FP16.";
    if (type === 'system') promptInput.value = "Architect a Sparse Mixture-of-Experts (MoE) pipeline with top-2 gating and ALRA linear attention.";
}
window.setPreset = setPreset;

function escapeHtml(text) {
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

let chartT = null;
let chartL = null;
let telemetryTimer = null;

function initCharts() {
    if (chartT) {
        refreshTrainingTelemetry();
        return;
    }
    const canvasT = document.getElementById('chart-throughput');
    const canvasL = document.getElementById('chart-loss');
    if (!canvasT || !canvasL) return;

    const ctxT = canvasT.getContext('2d');
    chartT = new Chart(ctxT, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Measured training speed (tok/s)',
                data: [],
                borderColor: '#00f3ff',
                backgroundColor: 'rgba(0, 243, 255, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: { responsive: true, plugins: { legend: { display: false } } }
    });

    const ctxL = canvasL.getContext('2d');
    chartL = new Chart(ctxL, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Training Loss',
                data: [],
                borderColor: '#ff007f',
                backgroundColor: 'rgba(255, 0, 127, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: { responsive: true, plugins: { legend: { display: false } } }
    });

    refreshTrainingTelemetry();
    if (!telemetryTimer) telemetryTimer = setInterval(refreshTrainingTelemetry, 5000);
}

function displayNumber(value, digits = 2) {
    return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
}

async function refreshTrainingTelemetry() {
    try {
        const response = await fetch('/api/telemetry');
        if (!response.ok) return;
        const data = await response.json();
        const training = data.training || {};
        const target = Number(training.target_steps || 0);
        const step = Number(training.step || 0);
        const status = training.status || 'idle';
        const stepText = target > 0 ? `${step.toLocaleString()} / ${target.toLocaleString()}` : (status === 'idle' ? 'Not training' : step.toLocaleString());
        document.getElementById('val-step').innerHTML = `${stepText} <span class="metric-unit">${status}</span>`;
        document.getElementById('val-loss').innerHTML = `${displayNumber(training.ema_loss ?? training.loss, 4)} <span class="metric-unit">EMA</span>`;
        document.getElementById('val-toks').innerHTML = `${displayNumber(training.tok_s, 1)} <span class="metric-unit">tok/sec</span>`;
        document.getElementById('val-eta').textContent = training.eta || 'Not training';

        const history = Array.isArray(training.history) ? training.history.filter(x => Number.isFinite(x.step)) : [];
        if (chartT && chartL) {
            const labels = history.map(x => String(x.step));
            chartT.data.labels = labels;
            chartT.data.datasets[0].data = history.map(x => Number.isFinite(x.tok_s) ? x.tok_s : null);
            chartL.data.labels = labels;
            chartL.data.datasets[0].data = history.map(x => Number.isFinite(x.loss) ? x.loss : null);
            chartT.update('none');
            chartL.update('none');
        }
    } catch (_) {
        // Keep the last confirmed readings visible during a temporary server restart.
    }
}

function searchSessions(query) {
    const q = (query || "").toLowerCase().trim();
    const container = document.getElementById('sessions-list-container');
    if (!container) return;
    const items = container.querySelectorAll('.chat-session-item');
    items.forEach(item => {
        const text = item.textContent.toLowerCase();
        if (!q || text.includes(q)) {
            item.style.display = 'flex';
        } else {
            item.style.display = 'none';
        }
    });
}
window.searchSessions = searchSessions;

async function exportSession(format) {
    try {
        const res = await fetch('/api/chats');
        const data = await res.json();
        const session = data[currentSessionId];
        if (!session || !session.messages) {
            alert("No active session messages to export.");
            return;
        }

        let content = '';
        let mime = 'text/plain';
        let ext = 'txt';

        if (format === 'md') {
            mime = 'text/markdown';
            ext = 'md';
            content = `# ${session.title || "Tantra Chat Export"}\n\n`;
            session.messages.forEach(m => {
                const sender = m.role === 'user' ? '### 👤 User' : '### ⚡ Tantra Quantum';
                content += `${sender}\n\n${m.content}\n\n---\n\n`;
            });
        } else {
            mime = 'application/json';
            ext = 'json';
            content = JSON.stringify(session, null, 2);
        }

        const blob = new Blob([content], { type: mime });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `tantra_session_${currentSessionId}.${ext}`;
        a.click();
        URL.revokeObjectURL(url);
    } catch(e) {
        alert("Failed to export session: " + e.message);
    }
}
window.exportSession = exportSession;

function inspectNode(n) {
    const card = document.getElementById('kg-inspector-card');
    const title = document.getElementById('kg-inspector-title');
    const content = document.getElementById('kg-inspector-content');
    if (!card || !title || !content) return;

    card.style.display = 'block';
    title.innerText = n.label || "Node Inspector";
    content.innerHTML = `
        <div><strong>Node ID:</strong> <span style="color:var(--cyan);">${n.id}</span></div>
        <div><strong>Type:</strong> ${n.type}</div>
        <div><strong>Coordinates:</strong> (${n.x}, ${n.y})</div>
        <div><strong>Status:</strong> Active / Online</div>
    `;
}

function filterAdminLogs(level) {
    const terminal = document.getElementById('admin-log-terminal');
    if (!terminal) return;
    
    ['all', 'info', 'warn', 'err'].forEach(l => {
        const b = document.getElementById(`log-flt-${l}`);
        if (b) b.className = l.toUpperCase() === level ? 'filter-btn active' : 'filter-btn';
    });

    const lines = terminal.querySelectorAll('div, br, span');
    // Simple filter demonstration
}
window.filterAdminLogs = filterAdminLogs;

// Auto-start on DOM load
window.addEventListener('DOMContentLoaded', () => {
    loadSessions();
    loadMemory();
    fetch('/api/capabilities')
        .then(res => res.ok ? res.json() : null)
        .then(capabilities => {
            if (!capabilities || capabilities.sandbox_enabled) return;
            const button = document.querySelector('button[onclick="runSandboxCode()"]');
            const output = document.getElementById('sandbox-output');
            if (button) {
                button.disabled = true;
                button.title = 'Disabled for security';
                button.innerText = 'Disabled for security';
            }
            if (output) output.innerText = 'Code execution is disabled for security. Use the local terminal for trusted code.';
        })
        .catch(() => {});
    setRole('user');

    const promptInput = document.getElementById('prompt-input');
    if (promptInput) {
        promptInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }
});
