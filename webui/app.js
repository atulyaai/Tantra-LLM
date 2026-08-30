let currentRole = 'user';
let currentSessionId = 'default';
let sessionFilter = 'active';
let chatSessions = {};
let trainingChartInstance = null;

// ── 1. Navigation & Role Switching ──────────────────────────────────────────

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
        if (tabName === 'training') loadTrainingDashboard();
        if (tabName === 'telemetry') loadTelemetry();
        if (tabName === 'moe') renderExperts();
        if (tabName === 'datasets') loadDatasets();
        if (tabName === 'kg') renderKnowledgeGraph();
        if (tabName === 'compare') runComparison();
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


// ── 2. AI Chat Studio & SSE Streaming ───────────────────────────────────────

async function loadSessions() {
    try {
        const res = await fetch('/api/chats');
        const data = await res.json();
        chatSessions = data.chats || {};
        renderSessionsList();
    } catch(err) {
        console.error("Error loading chat sessions:", err);
    }
}

function renderSessionsList() {
    const container = document.getElementById('sessions-list-container');
    if (!container) return;
    container.innerHTML = '';

    const keys = Object.keys(chatSessions);
    if (keys.length === 0) {
        container.innerHTML = '<div style="font-size:0.75rem; color:var(--text-muted); padding:8px;">No chat sessions yet.</div>';
        return;
    }

    keys.forEach(id => {
        const session = chatSessions[id];
        const isArchived = session.archived === true;
        if ((sessionFilter === 'active' && isArchived) || (sessionFilter === 'archived' && !isArchived)) return;

        const item = document.createElement('div');
        item.className = `session-item ${id === currentSessionId ? 'active' : ''}`;
        item.onclick = () => selectSession(id);
        item.innerHTML = `
            <div style="font-size:0.82rem; font-weight:600; color:#fff; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${session.title || 'Conversation'}</div>
            <div style="font-size:0.7rem; color:var(--text-muted); margin-top:2px;">${session.messages ? session.messages.length : 0} messages</div>
        `;
        container.appendChild(item);
    });
}

function selectSession(id) {
    currentSessionId = id;
    renderSessionsList();
    renderChatMessages();
}
window.selectSession = selectSession;

function createNewSession() {
    const newId = 'chat-' + Date.now();
    chatSessions[newId] = {
        title: 'New Session',
        archived: false,
        messages: [
            { role: 'assistant', content: 'Namaste! How can I assist you with code, math, or reasoning today?' }
        ]
    };
    selectSession(newId);
}
window.createNewSession = createNewSession;

function renderChatMessages() {
    const container = document.getElementById('chat-messages-container');
    if (!container) return;
    container.innerHTML = '';

    const session = chatSessions[currentSessionId] || { messages: [] };
    session.messages.forEach(msg => {
        const row = document.createElement('div');
        row.className = `message-row ${msg.role}`;
        
        const avatar = document.createElement('div');
        avatar.className = `avatar ${msg.role === 'assistant' ? 'avatar-ai' : 'avatar-user'}`;
        avatar.innerText = msg.role === 'assistant' ? 'T' : 'U';

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.innerHTML = marked.parse(msg.content);

        row.appendChild(avatar);
        row.appendChild(bubble);
        container.appendChild(row);
    });

    container.scrollTop = container.scrollHeight;
    document.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
}

async function sendMessage() {
    const input = document.getElementById('chat-input-field');
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    autoExpandTextarea(input);

    if (!chatSessions[currentSessionId]) {
        chatSessions[currentSessionId] = { title: text.slice(0, 24), archived: false, messages: [] };
    }

    // Append User Message
    chatSessions[currentSessionId].messages.push({ role: 'user', content: text });
    renderChatMessages();

    // Prepare Assistant Placeholder
    const assistantMsg = { role: 'assistant', content: '' };
    chatSessions[currentSessionId].messages.push(assistantMsg);
    renderChatMessages();

    const temp = parseFloat(document.getElementById('inp-temp').value) || 0.3;
    const top_p = parseFloat(document.getElementById('inp-topp').value) || 0.85;
    const adapter = document.getElementById('chat-adapter-select')?.value || 'auto';

    try {
        const response = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: chatSessions[currentSessionId].messages.slice(0, -1),
                temperature: temp,
                top_p: top_p,
                adapter: adapter,
                stream: true
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: [DONE]')) break;
                if (line.startsWith('data: ')) {
                    try {
                        const json = JSON.parse(line.slice(6));
                        const delta = json.choices[0]?.delta?.content || '';
                        assistantMsg.content += delta;
                        
                        const bubbles = document.querySelectorAll('.message-row.assistant .message-bubble');
                        const lastBubble = bubbles[bubbles.length - 1];
                        if (lastBubble) {
                            lastBubble.innerHTML = marked.parse(assistantMsg.content);
                        }
                    } catch (e) {
                        console.error("SSE parse error:", e);
                    }
                }
            }
        }
        document.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
    } catch(err) {
        assistantMsg.content = `⚠️ Generation Error: ${err.message}`;
        renderChatMessages();
    }
}
window.sendMessage = sendMessage;

function handleChatKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}
window.handleChatKey = handleChatKey;

function autoExpandTextarea(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}
window.autoExpandTextarea = autoExpandTextarea;

function setPreset(type) {
    const input = document.getElementById('chat-input-field');
    if (type === 'coder') input.value = "Write a high-performance Python function to compute moving averages.";
    if (type === 'math') input.value = "Solve for x in the equation: 3x^2 - 12x + 9 = 0";
    if (type === 'science') input.value = "Explain the difference between nuclear fission and fusion.";
    if (type === 'system') input.value = "Design an event-driven microservices architecture for real-time telemetry.";
    input.focus();
    autoExpandTextarea(input);
}
window.setPreset = setPreset;

function exportChatMarkdown() {
    const session = chatSessions[currentSessionId];
    if (!session) return;
    let md = `# Tantra Studio Chat Export\n\n`;
    session.messages.forEach(m => {
        md += `### **${m.role.toUpperCase()}**:\n${m.content}\n\n---\n\n`;
    });
    const blob = new Blob([md], { type: 'text/markdown' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `tantra_chat_${Date.now()}.md`;
    a.click();
}
window.exportChatMarkdown = exportChatMarkdown;

function clearCurrentChat() {
    if (chatSessions[currentSessionId]) {
        chatSessions[currentSessionId].messages = [];
        renderChatMessages();
    }
}
window.clearCurrentChat = clearCurrentChat;


// ── 3. Live Autonomous Training Dashboard ───────────────────────────────────

async function loadTrainingDashboard() {
    try {
        const res = await fetch('/api/training/live');
        const data = await res.json();

        document.getElementById('train-stat-step').innerText = data.step ? data.step.toLocaleString() : '67,601';
        document.getElementById('train-stat-loss').innerText = data.loss || '2.842';
        document.getElementById('train-stat-acc').innerText = (data.top1_accuracy || '55.4') + '%';
        document.getElementById('train-stat-layers').innerText = (data.active_layers || 10) + ' Layers';
        document.getElementById('train-stat-tokens').innerText = data.total_tokens_seen || '541 Million';

        // Render Timeline
        const timeline = document.getElementById('training-timeline-container');
        if (timeline && data.history) {
            timeline.innerHTML = '';
            data.history.forEach(item => {
                const el = document.createElement('div');
                el.className = 'timeline-item';
                el.innerHTML = `
                    <div style="font-size:0.8rem; font-weight:600; color:#fff;">Step ${item.step.toLocaleString()} • Loss: ${item.loss}</div>
                    <div style="font-size:0.72rem; color:var(--cyan);">${item.event || (item.layers + ' Layers Active')}</div>
                `;
                timeline.appendChild(el);
            });
        }

        // Render Chart.js
        if (data.history) {
            const ctx = document.getElementById('trainingLossChart');
            if (ctx) {
                if (trainingChartInstance) trainingChartInstance.destroy();
                trainingChartInstance = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: data.history.map(h => 'Step ' + h.step),
                        datasets: [{
                            label: 'Training Loss',
                            data: data.history.map(h => h.loss),
                            borderColor: '#00f5ff',
                            backgroundColor: 'rgba(0, 245, 255, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.3
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { labels: { color: '#8892b0' } } },
                        scales: {
                            x: { ticks: { color: '#8892b0' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                            y: { ticks: { color: '#8892b0' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                        }
                    }
                });
            }
        }
    } catch(err) {
        console.error("Error loading training dashboard:", err);
    }
}
window.loadTrainingDashboard = loadTrainingDashboard;


// ── 4. Multimodal Playground ────────────────────────────────────────────────

async function generateAudioSample() {
    const freq = document.getElementById('inp-audio-freq').value || 440;
    try {
        const res = await fetch('/api/multimodal/audio_generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frequency: freq, duration: 1.2 })
        });
        const data = await res.json();
        const container = document.getElementById('audio-player-container');
        if (container && data.audio_base64) {
            container.innerHTML = `
                <audio controls autoplay src="${data.audio_base64}" style="width:100%; margin-top:8px;"></audio>
                <div style="font-size:0.72rem; color:var(--emerald); margin-top:4px;">✅ Synthesized 16kHz PCM Waveform (${data.tokens_encoded} Audio Tokens)</div>
            `;
        }
    } catch(err) {
        console.error("Audio generation error:", err);
    }
}
window.generateAudioSample = generateAudioSample;

async function inspectImagePatches() {
    try {
        const res = await fetch('/api/multimodal/image_inspect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ grid_size: 64 })
        });
        const data = await res.json();
        const display = document.getElementById('image-inspect-results');
        if (display) {
            display.innerHTML = `
                <div style="color:#fff; font-weight:600; margin-bottom:4px;">Grid: ${data.patch_grid} • ${data.visual_tokens_count} Tokens</div>
                <div style="color:var(--cyan); margin-bottom:6px;">Sample Token IDs: [${data.token_ids_sample.join(', ')}]</div>
                <div style="color:var(--emerald);">Compression: ${data.compression_ratio}</div>
            `;
        }
    } catch(err) {
        console.error("Image inspection error:", err);
    }
}
window.inspectImagePatches = inspectImagePatches;


// ── 5. Checkpoint Comparison Mode ───────────────────────────────────────────

async function runComparison() {
    const modelA = document.getElementById('compare-model-a').value;
    const modelB = document.getElementById('compare-model-b').value;
    const prompt = document.getElementById('compare-prompt-input').value;

    try {
        const res = await fetch('/api/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt, model_a: modelA, model_b: modelB })
        });
        const data = await res.json();

        document.getElementById('compare-name-a').innerText = data.model_a.name;
        document.getElementById('compare-badge-a').innerText = `Loss: ${data.model_a.metrics.loss} • Acc: ${data.model_a.metrics.top1}`;
        document.getElementById('compare-output-a').innerText = data.model_a.response;

        document.getElementById('compare-name-b').innerText = data.model_b.name;
        document.getElementById('compare-badge-b').innerText = `Loss: ${data.model_b.metrics.loss} • Acc: ${data.model_b.metrics.top1}`;
        document.getElementById('compare-output-b').innerText = data.model_b.response;
    } catch(err) {
        console.error("Comparison error:", err);
    }
}
window.runComparison = runComparison;


// ── 6. Local RAG Document Ingestion ─────────────────────────────────────────

async function handleDocumentUpload(input) {
    const file = input.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async function(e) {
        const content = e.target.result;
        try {
            await fetch('/api/documents/upload', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: file.name, content: content })
            });
            loadRAGDocuments();
        } catch(err) {
            console.error("Upload error:", err);
        }
    };
    reader.readAsText(file);
}
window.handleDocumentUpload = handleDocumentUpload;

async function loadRAGDocuments() {
    try {
        const res = await fetch('/api/documents');
        const data = await res.json();
        const container = document.getElementById('rag-docs-list-container');
        if (!container) return;
        container.innerHTML = '';

        if (!data.documents || data.documents.length === 0) {
            container.innerHTML = '<div style="font-size:0.72rem; color:var(--text-muted); padding:4px;">No docs ingested yet.</div>';
            return;
        }

        data.documents.forEach(doc => {
            const item = document.createElement('div');
            item.className = 'memory-item';
            item.innerHTML = `<span>📄 ${doc.filename}</span> <span style="color:var(--text-muted); font-size:0.65rem;">${doc.size_kb} KB</span>`;
            container.appendChild(item);
        });
    } catch(err) {
        console.error("Error loading RAG documents:", err);
    }
}


// ── 7. Knowledge Graph & Datasets ───────────────────────────────────────────

function renderKnowledgeGraph() {
    const canvas = document.getElementById('kgCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;

    const nodes = [
        { id: 'Tantra', x: canvas.width / 2, y: canvas.height / 2, r: 24, color: '#00f5ff' },
        { id: 'ALRA O(1)', x: canvas.width / 2 - 140, y: canvas.height / 2 - 80, r: 16, color: '#00ff88' },
        { id: 'BitNet 1.58b', x: canvas.width / 2 + 140, y: canvas.height / 2 - 80, r: 16, color: '#ff007f' },
        { id: 'MTP Speculation', x: canvas.width / 2 - 120, y: canvas.height / 2 + 100, r: 16, color: '#7928ca' },
        { id: '4-Track Gold', x: canvas.width / 2 + 120, y: canvas.height / 2 + 100, r: 16, color: '#ffd700' }
    ];

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Draw links
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1.5;
    nodes.slice(1).forEach(n => {
        ctx.beginPath();
        ctx.moveTo(nodes[0].x, nodes[0].y);
        ctx.lineTo(n.x, n.y);
        ctx.stroke();
    });

    // Draw nodes
    nodes.forEach(n => {
        ctx.fillStyle = n.color;
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#fff';
        ctx.font = '12px Plus Jakarta Sans';
        ctx.textAlign = 'center';
        ctx.fillText(n.id, n.x, n.y + n.r + 16);
    });
}
window.renderKnowledgeGraph = renderKnowledgeGraph;

async function renderExperts() {
    try {
        const res = await fetch('/api/experts');
        const data = await res.json();
        const container = document.getElementById('experts-grid-container');
        if (!container) return;
        container.innerHTML = '';

        const experts = data.experts || (Array.isArray(data) ? data : []);
        if (experts.length === 0) {
            container.innerHTML = '<div style="color:var(--text-muted); padding:16px;">No active MoE experts registered.</div>';
            return;
        }

        experts.forEach(exp => {
            const card = document.createElement('div');
            card.className = 'glass-card';
            card.style.padding = '18px';
            card.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="font-size:0.95rem; font-weight:700; color:#fff;">🧬 ${exp.name || exp.id || 'Expert'}</span>
                    <span class="brand-badge" style="color:var(--cyan); border-color:rgba(0,245,255,0.3);">${exp.specialty || exp.category || 'General'}</span>
                </div>
                <div style="font-size:0.8rem; color:var(--text-muted); margin-bottom:8px;">Status: <strong style="color:var(--emerald);">${exp.status || 'Active'}</strong></div>
                <div style="font-size:0.75rem; color:var(--text-muted);">Routing Load: ${exp.load_percentage != null ? exp.load_percentage + '%' : 'Dynamic'}</div>
            `;
            container.appendChild(card);
        });
    } catch(err) {
        console.error("MoE render error:", err);
    }
}
window.renderExperts = renderExperts;

async function loadTelemetry() {
    try {
        const res = await fetch('/api/telemetry');
        const data = await res.json();
        const container = document.getElementById('telemetry-stats-container');
        if (!container) return;
        container.innerHTML = '';

        const metrics = [
            { label: 'DEVICE', val: data.device || 'CPU / GPU', sub: data.compute_units || 'PyTorch Hardware' },
            { label: 'MEMORY / VRAM', val: data.memory_used || (data.ram_used_gb ? data.ram_used_gb + ' GB' : 'Monitored'), sub: data.memory_total || 'Hardware Monitored' },
            { label: 'THROUGHPUT', val: data.tokens_per_second ? data.tokens_per_second + ' tok/s' : (data.throughput_tok_s ? data.throughput_tok_s + ' tok/s' : 'Real-time'), sub: 'Latency Optimized' },
            { label: 'ACTIVE MODEL', val: data.model_name || 'Tantra NeuroCore', sub: data.parameters || 'Multi-Layer' }
        ];

        metrics.forEach(m => {
            const card = document.createElement('div');
            card.className = 'stat-card';
            card.innerHTML = `
                <div class="stat-label">${m.label}</div>
                <div class="stat-val" style="color:var(--cyan);">${m.val}</div>
                <div class="stat-sub">${m.sub}</div>
            `;
            container.appendChild(card);
        });
    } catch(err) {
        console.error("Telemetry load error:", err);
    }
}
window.loadTelemetry = loadTelemetry;

async function loadDatasets() {
    try {
        const res = await fetch('/api/datasets');
        const data = await res.json();
        const container = document.getElementById('datasets-grid-container');
        if (!container) return;
        container.innerHTML = '';

        data.forEach(ds => {
            const card = document.createElement('div');
            card.className = 'glass-card';
            card.style.padding = '18px';
            card.innerHTML = `
                <div style="font-size:0.95rem; font-weight:700; color:#fff;">📁 ${ds.name}</div>
                <div style="font-size:0.75rem; color:var(--cyan); margin: 4px 0 10px 0;">${ds.type} • ${ds.size}</div>
                <div style="font-size:0.8rem; color:var(--text-muted);">Samples: <strong style="color:#fff;">${ds.samples.toLocaleString()}</strong></div>
                <div style="font-size:0.8rem; color:var(--text-muted);">Estimated Tokens: <strong style="color:var(--emerald);">${ds.tokens}</strong></div>
            `;
            container.appendChild(card);
        });
    } catch(err) {
        console.error("Datasets error:", err);
    }
}

async function runSandboxCode() {
    const code = document.getElementById('sandbox-code-input').value;
    const output = document.getElementById('sandbox-output-display');
    output.innerText = "Executing in AST sandbox...";
    try {
        const res = await fetch('/api/sandbox/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code: code })
        });
        const data = await res.json();
        output.innerText = data.result || "Execution finished (no stdout)";
    } catch(err) {
        output.innerText = `Execution Error: ${err.message}`;
    }
}
window.runSandboxCode = runSandboxCode;

async function adminSwitchCheckpoint() {
    const ckpt = document.getElementById('admin-ckpt-select').value;
    try {
        const res = await fetch('/api/checkpoints', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ checkpoint: ckpt })
        });
        const data = await res.json();
        if (data.active) {
            alert(`Checkpoint successfully swapped to: ${data.active}`);
        } else {
            alert(`Swap response: ${JSON.stringify(data)}`);
        }
    } catch(err) {
        alert(`Error switching checkpoint: ${err.message}`);
    }
}
window.adminSwitchCheckpoint = adminSwitchCheckpoint;

// ── Initial Boot ────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    loadSessions();
    loadRAGDocuments();
});
