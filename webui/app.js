// ══════════════════════════════════════════════════════════════════════════════
// Tantra Quantum Studio — Ultra-Optimized Real-Time Web UI Controller
// ══════════════════════════════════════════════════════════════════════════════

let currentRole = 'user';
let currentSessionId = 'default';
let sessionFilter = 'active';
let chatSessions = {};
let memoryBank = [];
let datasetsCatalog = [];
let trainingChartInstance = null;
let activeAbortController = null;
let livePollInterval = null;
let pollRateMs = 2000;
let audioContextInstance = null;
let audioAnalyser = null;
let audioAnimFrame = null;

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
        if (tabName === 'training') loadTrainingDashboard(true);
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
    renderSessionsList();
}
window.setSessionFilter = setSessionFilter;


// ── 2. Live Telemetry & Auto-Polling Engine ─────────────────────────────────

function changePollRate(rateStr) {
    pollRateMs = parseInt(rateStr, 10);
    if (livePollInterval) {
        clearInterval(livePollInterval);
        livePollInterval = null;
    }
    if (pollRateMs > 0) {
        livePollInterval = setInterval(pollLiveTelemetry, pollRateMs);
    }
}
window.changePollRate = changePollRate;

async function pollLiveTelemetry() {
    try {
        // Lightweight polling of live training & status
        const res = await fetch('/api/training/live');
        if (!res.ok) return;
        const data = await res.json();

        // Update header speed pill
        const speedPill = document.getElementById('live-speed-text');
        if (speedPill && data.tok_s != null) {
            speedPill.innerText = `LIVE ${parseFloat(data.tok_s).toFixed(1)} tok/s`;
        }

        // Update active layers badge in header
        const layersBadge = document.getElementById('hdr-layers-badge');
        if (layersBadge && data.active_layers) {
            layersBadge.innerText = `${data.active_layers}L ALRA O(1)`;
        }

        // If training tab is active, update dashboard smoothly
        const trainingView = document.getElementById('view-training');
        if (trainingView && trainingView.classList.contains('active')) {
            updateTrainingDashboardUI(data);
        }
    } catch (e) {
        // Ignore background polling errors
    }
}


// ── 3. AI Chat Studio & SSE Streaming ───────────────────────────────────────

async function loadSessions() {
    try {
        const res = await fetch('/api/chats');
        const data = await res.json();
        chatSessions = data.chats || {};
        if (Object.keys(chatSessions).length === 0) {
            createNewSession();
        } else {
            renderSessionsList();
            renderChatMessages();
        }
    } catch(err) {
        console.error("Error loading chat sessions:", err);
    }
}

function searchSessions(query) {
    const q = (query || '').toLowerCase();
    const container = document.getElementById('sessions-list-container');
    if (!container) return;
    container.innerHTML = '';

    const keys = Object.keys(chatSessions);
    keys.forEach(id => {
        const session = chatSessions[id];
        const title = (session.title || 'Conversation').toLowerCase();
        if (q && !title.includes(q)) return;

        const isArchived = session.archived === true;
        if ((sessionFilter === 'active' && isArchived) || (sessionFilter === 'archived' && !isArchived)) return;

        appendSessionDOM(id, session, container);
    });
}
window.searchSessions = searchSessions;

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

        appendSessionDOM(id, session, container);
    });
}

function appendSessionDOM(id, session, container) {
    const item = document.createElement('div');
    item.className = `session-item ${id === currentSessionId ? 'active' : ''}`;
    item.onclick = (e) => {
        if (!e.target.closest('.session-actions')) selectSession(id);
    };

    const count = session.messages ? session.messages.length : 0;
    item.innerHTML = `
        <div class="session-info">
            <div style="font-size:0.82rem; font-weight:600; color:#fff; overflow:hidden; text-overflow:ellipsis;">${session.title || 'Conversation'}</div>
            <div style="font-size:0.68rem; color:var(--text-muted);">${count} msgs</div>
        </div>
        <div class="session-actions">
            <button class="btn-icon" onclick="renameSession('${id}')" title="Rename">✏️</button>
            <button class="btn-icon" onclick="archiveSession('${id}')" title="Archive / Unarchive">📦</button>
            <button class="btn-icon" onclick="deleteSession('${id}')" title="Delete" style="color:var(--danger);">🗑️</button>
        </div>
    `;
    container.appendChild(item);
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
        id: newId,
        title: 'New Session',
        archived: false,
        created_at: Date.now(),
        messages: [
            { role: 'assistant', content: 'Namaste! How can I assist you with code, math, or reasoning today?' }
        ]
    };
    selectSession(newId);
    syncChatToServer(newId, chatSessions[newId]);
}
window.createNewSession = createNewSession;

function renameSession(id) {
    const session = chatSessions[id];
    if (!session) return;
    const newTitle = prompt("Enter new title for this conversation:", session.title || "Chat");
    if (newTitle && newTitle.trim()) {
        session.title = newTitle.trim();
        renderSessionsList();
        syncChatToServer(id, session);
    }
}
window.renameSession = renameSession;

function archiveSession(id) {
    const session = chatSessions[id];
    if (!session) return;
    session.archived = !session.archived;
    renderSessionsList();
    syncChatToServer(id, session);
}
window.archiveSession = archiveSession;

function deleteSession(id) {
    if (!confirm("Are you sure you want to delete this chat session?")) return;
    delete chatSessions[id];
    try {
        fetch(`/api/chats/${id}`, { method: 'DELETE' });
    } catch(e) {}

    const remainingKeys = Object.keys(chatSessions);
    if (remainingKeys.length > 0) {
        selectSession(remainingKeys[0]);
    } else {
        createNewSession();
    }
}
window.deleteSession = deleteSession;

function syncChatToServer(id, session) {
    try {
        fetch('/api/chats', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                id: id,
                title: session.title,
                archived: session.archived,
                messages: session.messages
            })
        });
    } catch(e) {
        console.error("Chat sync error:", e);
    }
}

function renderChatMessages() {
    const container = document.getElementById('chat-messages-container');
    if (!container) return;
    container.innerHTML = '';

    const session = chatSessions[currentSessionId] || { messages: [] };
    session.messages.forEach((msg, idx) => {
        const row = document.createElement('div');
        row.className = `message-row ${msg.role}`;
        
        const avatar = document.createElement('div');
        avatar.className = `avatar ${msg.role === 'assistant' ? 'avatar-ai' : 'avatar-user'}`;
        avatar.innerText = msg.role === 'assistant' ? 'T' : 'U';

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        
        // Clean markdown parsing
        const rawHtml = marked.parse(msg.content || '');
        bubble.innerHTML = DOMPurify ? DOMPurify.sanitize(rawHtml) : rawHtml;

        // Attach telemetry badge if stored on message
        if (msg.telemetry) {
            const t = msg.telemetry;
            const tTag = document.createElement('div');
            tTag.className = 'message-telemetry-tag';
            tTag.innerHTML = `⚡ ${t.tokens_per_second || '--'} tok/s • TTFT: ${t.ttft_ms || '--'}ms • ${t.tokens_generated || '--'} tokens (${t.duration_seconds || '--'}s)`;
            bubble.appendChild(tTag);
        }

        row.appendChild(avatar);
        row.appendChild(bubble);
        container.appendChild(row);
    });

    container.scrollTop = container.scrollHeight;
    attachCodeCopyButtons();
    document.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
}

function attachCodeCopyButtons() {
    document.querySelectorAll('.message-bubble pre').forEach(pre => {
        if (pre.querySelector('.code-copy-btn')) return;
        const btn = document.createElement('button');
        btn.className = 'code-copy-btn';
        btn.innerText = 'Copy';
        btn.onclick = () => {
            const code = pre.querySelector('code')?.innerText || pre.innerText;
            navigator.clipboard.writeText(code).then(() => {
                btn.innerText = 'Copied! ✓';
                btn.style.color = 'var(--emerald)';
                setTimeout(() => {
                    btn.innerText = 'Copy';
                    btn.style.color = '';
                }, 2000);
            });
        };
        pre.appendChild(btn);
    });
}

function stopGeneration() {
    if (activeAbortController) {
        activeAbortController.abort();
        activeAbortController = null;
    }
    const stopBtn = document.getElementById('btn-stop-stream');
    if (stopBtn) stopBtn.style.display = 'none';
    const statusInd = document.getElementById('chat-status-indicator');
    if (statusInd) statusInd.innerText = 'Generation stopped';
}
window.stopGeneration = stopGeneration;

async function sendMessage() {
    const input = document.getElementById('chat-input-field');
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    autoExpandTextarea(input);

    if (!chatSessions[currentSessionId]) {
        chatSessions[currentSessionId] = { id: currentSessionId, title: text.slice(0, 26), archived: false, messages: [] };
    }

    const session = chatSessions[currentSessionId];
    if (session.title === 'New Session') {
        session.title = text.slice(0, 26) + (text.length > 26 ? '...' : '');
        renderSessionsList();
    }

    // Append User Message
    session.messages.push({ role: 'user', content: text });
    renderChatMessages();

    // Prepare Assistant Placeholder
    const assistantMsg = { role: 'assistant', content: '' };
    session.messages.push(assistantMsg);
    renderChatMessages();

    const temp = parseFloat(document.getElementById('inp-temp').value) || 0.35;
    const top_p = parseFloat(document.getElementById('inp-topp').value) || 0.85;
    const max_tokens = parseInt(document.getElementById('inp-maxtok').value, 10) || 256;
    const adapter = document.getElementById('chat-adapter-select')?.value || 'auto';

    // Show live UI badges
    const liveStats = document.getElementById('live-inference-stats');
    const stopBtn = document.getElementById('btn-stop-stream');
    const statusInd = document.getElementById('chat-status-indicator');
    if (liveStats) liveStats.style.display = 'inline-flex';
    if (stopBtn) stopBtn.style.display = 'inline-flex';
    if (statusInd) statusInd.innerText = 'Generating response...';

    const t0 = performance.now();
    let firstTokenTime = null;
    let tokenCount = 0;

    activeAbortController = new AbortController();

    try {
        const response = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: activeAbortController.signal,
            body: JSON.stringify({
                messages: session.messages.slice(0, -1),
                temperature: temp,
                top_p: top_p,
                max_tokens: max_tokens,
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
                        if (delta) {
                            if (firstTokenTime === null) {
                                firstTokenTime = performance.now() - t0;
                                const ttftEl = document.getElementById('stat-ttft');
                                if (ttftEl) ttftEl.innerText = `TTFT: ${Math.round(firstTokenTime)}ms`;
                            }
                            tokenCount++;
                            assistantMsg.content += delta;

                            const elapsedSec = (performance.now() - t0) / 1000.0;
                            const curSpeed = (tokenCount / Math.max(elapsedSec, 0.001)).toFixed(1);
                            const toksEl = document.getElementById('stat-toks');
                            if (toksEl) toksEl.innerText = `${curSpeed} tok/s`;

                            // Real-time render update
                            const bubbles = document.querySelectorAll('.message-row.assistant .message-bubble');
                            const lastBubble = bubbles[bubbles.length - 1];
                            if (lastBubble) {
                                const rawHtml = marked.parse(assistantMsg.content);
                                lastBubble.innerHTML = DOMPurify ? DOMPurify.sanitize(rawHtml) : rawHtml;
                            }
                        }

                        // Check for final telemetry chunk
                        if (json.telemetry) {
                            assistantMsg.telemetry = json.telemetry;
                        }
                    } catch (e) {
                        // Incomplete JSON chunk, skip
                    }
                }
            }
        }
        if (statusInd) statusInd.innerText = 'Ready';
    } catch(err) {
        if (err.name === 'AbortError') {
            if (statusInd) statusInd.innerText = 'Stopped';
        } else {
            assistantMsg.content += `\n\n⚠️ *Generation Error: ${err.message}*`;
            if (statusInd) statusInd.innerText = 'Error';
        }
    } finally {
        activeAbortController = null;
        if (stopBtn) stopBtn.style.display = 'none';
        renderChatMessages();
        syncChatToServer(currentSessionId, session);
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
    if (type === 'coder') input.value = "Write a high-performance Python function to compute fast moving averages with NumPy and SIMD.";
    if (type === 'math') input.value = "Solve for x in the equation: 3x^2 - 12x + 9 = 0 and prove each step.";
    if (type === 'science') input.value = "Explain how ALRA Linear Recurrent Attention achieves O(1) inference memory complexity.";
    if (type === 'bitnet') input.value = "Explain how BitNet 1.58-bit ternary weights {-1, 0, +1} accelerate matrix computations.";
    input.focus();
    autoExpandTextarea(input);
}
window.setPreset = setPreset;

function exportChatMarkdown() {
    const session = chatSessions[currentSessionId];
    if (!session) return;
    let md = `# Tantra Studio Chat Export — ${session.title || 'Session'}\n\n`;
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
        syncChatToServer(currentSessionId, chatSessions[currentSessionId]);
    }
}
window.clearCurrentChat = clearCurrentChat;


// ── 4. Live Autonomous Training Dashboard ───────────────────────────────────

async function loadTrainingDashboard(forceRefresh = false) {
    try {
        const res = await fetch('/api/training/live');
        const data = await res.json();
        updateTrainingDashboardUI(data);
    } catch(err) {
        console.error("Error loading training dashboard:", err);
    }
}
window.loadTrainingDashboard = loadTrainingDashboard;

function updateTrainingDashboardUI(data) {
    if (!data) return;

    // Status Pill
    const statusPill = document.getElementById('train-status-pill');
    if (statusPill) {
        const isRunning = data.status === 'running';
        const isInterrupted = data.status === 'interrupted';
        statusPill.innerText = isRunning ? '● RUNNING' : (isInterrupted ? '● INTERRUPTED' : '● IDLE');
        statusPill.style.color = isRunning ? 'var(--emerald)' : (isInterrupted ? 'var(--amber)' : 'var(--cyan)');
        statusPill.style.borderColor = isRunning ? 'var(--emerald)' : (isInterrupted ? 'var(--amber)' : 'var(--cyan)');
    }

    // Hero Stats
    const stepEl = document.getElementById('train-stat-step');
    if (stepEl && data.step != null) stepEl.innerText = Number(data.step).toLocaleString();

    const lossEl = document.getElementById('train-stat-loss');
    if (lossEl && data.loss != null) lossEl.innerText = parseFloat(data.loss).toFixed(3);

    const emaEl = document.getElementById('train-stat-ema');
    if (emaEl && data.ema_loss != null) emaEl.innerText = parseFloat(data.ema_loss).toFixed(3);

    const pplEl = document.getElementById('train-stat-ppl');
    if (pplEl && data.ppl != null) pplEl.innerText = parseFloat(data.ppl).toFixed(1);

    const layersEl = document.getElementById('train-stat-layers');
    if (layersEl && data.active_layers != null) layersEl.innerText = `${data.active_layers} Layers`;

    const paramsEl = document.getElementById('train-stat-params');
    if (paramsEl && data.parameters != null) paramsEl.innerText = data.parameters;

    const tokSpeedEl = document.getElementById('train-stat-tokens');
    if (tokSpeedEl && data.tok_s != null) tokSpeedEl.innerText = `${parseFloat(data.tok_s).toFixed(1)} tok/s`;

    const totalTokEl = document.getElementById('train-stat-total-tokens');
    if (totalTokEl && data.total_tokens_seen != null) totalTokEl.innerText = data.total_tokens_seen;

    // Progress Bar & ETA
    const targetSteps = data.target_steps || 50000;
    const curStep = data.step || 0;
    const progressPct = Math.min(100, Math.max(0, (curStep / targetSteps) * 100));
    
    const progText = document.getElementById('train-progress-text');
    if (progText) progText.innerText = `${progressPct.toFixed(1)}% (${curStep.toLocaleString()} / ${targetSteps.toLocaleString()})`;

    const progBar = document.getElementById('train-progress-bar');
    if (progBar) progBar.style.width = `${progressPct}%`;

    const etaText = document.getElementById('train-eta-text');
    if (etaText && data.eta) etaText.innerText = data.eta;

    const secPerStep = document.getElementById('train-sec-per-step');
    if (secPerStep && data.time_telemetry?.actual_avg_sec_per_step != null) {
        secPerStep.innerText = `${data.time_telemetry.actual_avg_sec_per_step}s/step`;
    }

    // Timeline Rendering
    const timeline = document.getElementById('training-timeline-container');
    if (timeline && data.history && data.history.length > 0) {
        timeline.innerHTML = '';
        const recentHistory = data.history.slice(-6).reverse();
        recentHistory.forEach(item => {
            const el = document.createElement('div');
            el.className = 'timeline-item';
            el.innerHTML = `
                <div style="font-size:0.82rem; font-weight:700; color:#fff;">Step ${item.step.toLocaleString()} • Loss: ${parseFloat(item.loss).toFixed(3)}</div>
                <div style="font-size:0.72rem; color:var(--cyan); margin-top:2px;">Throughput: ${item.tok_s ? item.tok_s.toFixed(1) + ' tok/s' : 'Active'} • PPL: ${item.ppl ? item.ppl.toFixed(1) : 'Normal'}</div>
            `;
            timeline.appendChild(el);
        });
    }

    // Chart.js Convergence Curve
    if (data.history && data.history.length > 0) {
        const ctx = document.getElementById('trainingLossChart');
        if (ctx) {
            const labels = data.history.map(h => 'Step ' + h.step);
            const lossPoints = data.history.map(h => h.loss);
            const pplPoints = data.history.map(h => Math.min(h.ppl || 200, 500));

            if (trainingChartInstance) {
                trainingChartInstance.data.labels = labels;
                trainingChartInstance.data.datasets[0].data = lossPoints;
                trainingChartInstance.data.datasets[1].data = pplPoints;
                trainingChartInstance.update('none');
            } else {
                trainingChartInstance = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [
                            {
                                label: 'Training Loss',
                                data: lossPoints,
                                borderColor: '#00f5ff',
                                backgroundColor: 'rgba(0, 245, 255, 0.12)',
                                borderWidth: 2.5,
                                fill: true,
                                tension: 0.25,
                                yAxisID: 'y'
                            },
                            {
                                label: 'Perplexity (PPL)',
                                data: pplPoints,
                                borderColor: '#ff007f',
                                borderDash: [4, 4],
                                borderWidth: 1.8,
                                fill: false,
                                tension: 0.25,
                                yAxisID: 'y1'
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: { mode: 'index', intersect: false },
                        plugins: {
                            legend: { labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 11 } } }
                        },
                        scales: {
                            x: { ticks: { color: '#64748b', maxTicksLimit: 8 }, grid: { color: 'rgba(255,255,255,0.04)' } },
                            y: {
                                type: 'linear', display: true, position: 'left',
                                ticks: { color: '#00f5ff' }, grid: { color: 'rgba(255,255,255,0.05)' }
                            },
                            y1: {
                                type: 'linear', display: true, position: 'right',
                                ticks: { color: '#ff007f' }, grid: { drawOnChartArea: false }
                            }
                        }
                    }
                });
            }
        }
    }
}


// ── 5. Native Multimodal & Web Audio Oscilloscope ───────────────────────────

async function generateAudioSample() {
    const freq = document.getElementById('inp-audio-freq')?.value || 440;
    const dur = parseFloat(document.getElementById('inp-audio-dur')?.value) || 1.2;

    try {
        const res = await fetch('/api/multimodal/audio_generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frequency: freq, duration: dur })
        });
        const data = await res.json();
        const container = document.getElementById('audio-player-container');
        if (container && data.audio_base64) {
            container.innerHTML = `
                <audio id="synthesizedAudio" controls autoplay src="${data.audio_base64}" style="width:100%; margin-top:8px;"></audio>
                <div style="font-size:0.72rem; color:var(--emerald); margin-top:4px;">✅ Synthesized 16kHz PCM Waveform (${data.tokens_encoded} Audio Tokens)</div>
            `;

            const audioElement = document.getElementById('synthesizedAudio');
            if (audioElement) {
                startAudioOscilloscope(audioElement);
            }
        }
    } catch(err) {
        console.error("Audio generation error:", err);
    }
}
window.generateAudioSample = generateAudioSample;

function startAudioOscilloscope(audioElement) {
    try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!audioContextInstance) {
            audioContextInstance = new AudioCtx();
        }
        if (audioContextInstance.state === 'suspended') {
            audioContextInstance.resume();
        }

        const canvas = document.getElementById('audioWaveCanvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        if (!audioAnalyser) {
            audioAnalyser = audioContextInstance.createAnalyser();
            audioAnalyser.fftSize = 256;
        }

        const source = audioContextInstance.createMediaElementSource(audioElement);
        source.connect(audioAnalyser);
        audioAnalyser.connect(audioContextInstance.destination);

        const bufferLength = audioAnalyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const statusEl = document.getElementById('oscilloscope-status');
        if (statusEl) statusEl.innerText = 'Playing';

        function drawWave() {
            audioAnimFrame = requestAnimationFrame(drawWave);
            audioAnalyser.getByteTimeDomainData(dataArray);

            ctx.fillStyle = 'rgba(7, 9, 14, 0.4)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.lineWidth = 2;
            ctx.strokeStyle = '#00f5ff';
            ctx.shadowBlur = 8;
            ctx.shadowColor = '#00f5ff';
            ctx.beginPath();

            const sliceWidth = canvas.width / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = (v * canvas.height) / 2;

                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);

                x += sliceWidth;
            }

            ctx.lineTo(canvas.width, canvas.height / 2);
            ctx.stroke();
        }

        if (audioAnimFrame) cancelAnimationFrame(audioAnimFrame);
        drawWave();

        audioElement.onended = () => {
            if (audioAnimFrame) cancelAnimationFrame(audioAnimFrame);
            if (statusEl) statusEl.innerText = 'Finished';
        };
    } catch (e) {
        console.warn("Oscilloscope initialization note:", e);
    }
}

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
                <div style="color:#fff; font-weight:700; margin-bottom:4px;">Grid: ${data.patch_grid} • ${data.visual_tokens_count} Discrete Patches</div>
                <div style="color:var(--cyan); margin-bottom:6px; word-break:break-all;">Sample Token IDs: [${data.token_ids_sample.join(', ')}]</div>
                <div style="color:var(--emerald);">Compression: ${data.compression_ratio} • Latent dim: 256</div>
            `;
        }
    } catch(err) {
        console.error("Image inspection error:", err);
    }
}
window.inspectImagePatches = inspectImagePatches;


// ── 6. Checkpoint Comparison Mode ───────────────────────────────────────────

async function runComparison() {
    const modelA = document.getElementById('compare-model-a').value;
    const modelB = document.getElementById('compare-model-b').value;
    const prompt = document.getElementById('compare-prompt-input').value;

    const outA = document.getElementById('compare-output-a');
    const outB = document.getElementById('compare-output-b');
    if (outA) outA.innerText = "Evaluating Model A...";
    if (outB) outB.innerText = "Evaluating Model B...";

    try {
        const res = await fetch('/api/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt, model_a: modelA, model_b: modelB })
        });
        const data = await res.json();

        document.getElementById('compare-name-a').innerText = data.model_a.name;
        document.getElementById('compare-badge-a').innerText = `Loss: ${data.model_a.metrics.loss} • Acc: ${data.model_a.metrics.top1}`;
        if (outA) outA.innerHTML = marked.parse(data.model_a.response);

        document.getElementById('compare-name-b').innerText = data.model_b.name;
        document.getElementById('compare-badge-b').innerText = `Loss: ${data.model_b.metrics.loss} • Acc: ${data.model_b.metrics.top1}`;
        if (outB) outB.innerHTML = marked.parse(data.model_b.response);
    } catch(err) {
        if (outA) outA.innerText = `Comparison Error: ${err.message}`;
        if (outB) outB.innerText = `Comparison Error: ${err.message}`;
    }
}
window.runComparison = runComparison;


// ── 7. RAG Documents & Memory Bank ──────────────────────────────────────────

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

async function loadMemoryBank() {
    try {
        const res = await fetch('/api/memory');
        const data = await res.json();
        memoryBank = data.memory || [];
        renderMemoryBank();
    } catch(err) {
        console.error("Error loading memory bank:", err);
    }
}

function renderMemoryBank() {
    const container = document.getElementById('memory-drawer-container');
    if (!container) return;
    container.innerHTML = '';

    if (memoryBank.length === 0) {
        container.innerHTML = '<div style="font-size:0.72rem; color:var(--text-muted); padding:4px;">No long-term memories stored.</div>';
        return;
    }

    memoryBank.forEach(mem => {
        const item = document.createElement('div');
        item.className = 'memory-item';
        item.innerHTML = `
            <div style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1;" title="${mem.fact}">
                <strong style="color:var(--cyan);">${mem.category || 'Fact'}:</strong> ${mem.fact}
            </div>
            <button class="btn-icon" onclick="deleteMemory('${mem.id}')" title="Delete" style="color:var(--danger); margin-left:4px;">✕</button>
        `;
        container.appendChild(item);
    });
}

async function addMemoryPrompt() {
    const category = prompt("Memory Category (e.g. Preference, Architecture, Context):", "Preference");
    if (!category) return;
    const fact = prompt("Fact / Information to remember:");
    if (!fact || !fact.trim()) return;

    try {
        await fetch('/api/memory', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ category: category.trim(), fact: fact.trim() })
        });
        loadMemoryBank();
    } catch(err) {
        console.error("Error adding memory:", err);
    }
}
window.addMemoryPrompt = addMemoryPrompt;

async function deleteMemory(memoryId) {
    try {
        await fetch(`/api/memory/${memoryId}`, { method: 'DELETE' });
        loadMemoryBank();
    } catch(err) {
        console.error("Error deleting memory:", err);
    }
}
window.deleteMemory = deleteMemory;


// ── 8. Interactive Physics Knowledge Graph ──────────────────────────────────

let kgSimulation = null;

async function renderKnowledgeGraph() {
    const canvas = document.getElementById('kgCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.parentElement.clientWidth || 800;
    canvas.height = canvas.parentElement.clientHeight || 500;

    let nodes = [];
    let links = [];

    try {
        const res = await fetch('/api/knowledge_graph');
        const data = await res.json();
        if (data && data.nodes && data.nodes.length > 0) {
            nodes = data.nodes;
            links = data.links || [];
        }
    } catch(err) {
        console.error("Error fetching knowledge graph:", err);
    }

    if (nodes.length === 0) {
        nodes = [
            { id: 'core', label: 'Tantra NeuroCore', x: canvas.width / 2, y: canvas.height / 2, r: 24, color: '#00f5ff' },
            { id: 'alra', label: 'ALRA Linear O(1)', x: canvas.width / 2 - 140, y: canvas.height / 2 - 80, r: 18, color: '#00ff88' },
            { id: 'bitnet', label: 'BitNet 1.58b', x: canvas.width / 2 + 140, y: canvas.height / 2 - 80, r: 18, color: '#ff007f' },
            { id: 'mtp', label: 'MTP Speculation', x: canvas.width / 2 - 120, y: canvas.height / 2 + 100, r: 16, color: '#8a2be2' },
            { id: 'gold', label: '4-Track Gold', x: canvas.width / 2 + 120, y: canvas.height / 2 + 100, r: 16, color: '#ffd700' }
        ];
        links = [
            { source: 'core', target: 'alra' },
            { source: 'core', target: 'bitnet' },
            { source: 'core', target: 'mtp' },
            { source: 'core', target: 'gold' }
        ];
    }

    // Initialize node physics properties
    const physicsNodes = nodes.map(n => ({
        ...n,
        x: n.x || Math.random() * (canvas.width - 200) + 100,
        y: n.y || Math.random() * (canvas.height - 200) + 100,
        vx: 0,
        vy: 0,
        r: n.r || (n.type === 'concept' ? 24 : 16)
    }));

    const nodeLookup = {};
    physicsNodes.forEach(n => nodeLookup[n.id] = n);

    let draggedNode = null;

    canvas.onmousedown = (e) => {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        for (const n of physicsNodes) {
            const dist = Math.hypot(n.x - mx, n.y - my);
            if (dist <= n.r + 6) {
                draggedNode = n;
                break;
            }
        }
    };

    canvas.onmousemove = (e) => {
        if (!draggedNode) return;
        const rect = canvas.getBoundingClientRect();
        draggedNode.x = e.clientX - rect.left;
        draggedNode.y = e.clientY - rect.top;
        draggedNode.vx = 0;
        draggedNode.vy = 0;
    };

    window.onmouseup = () => { draggedNode = null; };

    if (kgSimulation) cancelAnimationFrame(kgSimulation);

    function stepPhysics() {
        // Repulsion between nodes
        for (let i = 0; i < physicsNodes.length; i++) {
            for (let j = i + 1; j < physicsNodes.length; j++) {
                const n1 = physicsNodes[i];
                const n2 = physicsNodes[j];
                const dx = n2.x - n1.x;
                const dy = n2.y - n1.y;
                const dist = Math.hypot(dx, dy) || 1;
                const force = 350 / (dist * dist);
                const fx = (dx / dist) * force;
                const fy = (dy / dist) * force;

                if (n1 !== draggedNode) { n1.vx -= fx; n1.vy -= fy; }
                if (n2 !== draggedNode) { n2.vx += fx; n2.vy += fy; }
            }
        }

        // Spring attraction along links
        links.forEach(l => {
            const src = nodeLookup[l.source];
            const tgt = nodeLookup[l.target];
            if (src && tgt) {
                const dx = tgt.x - src.x;
                const dy = tgt.y - src.y;
                const dist = Math.hypot(dx, dy) || 1;
                const springForce = (dist - 120) * 0.008;
                const fx = (dx / dist) * springForce;
                const fy = (dy / dist) * springForce;

                if (src !== draggedNode) { src.vx += fx; src.vy += fy; }
                if (tgt !== draggedNode) { tgt.vx -= fx; tgt.vy += fy; }
            }
        });

        // Center gravity and integration
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        physicsNodes.forEach(n => {
            if (n === draggedNode) return;
            n.vx += (cx - n.x) * 0.0005;
            n.vy += (cy - n.y) * 0.0005;
            n.vx *= 0.88; // Damping
            n.vy *= 0.88;
            n.x += n.vx;
            n.y += n.vy;

            // Boundaries
            n.x = Math.max(n.r + 10, Math.min(canvas.width - n.r - 10, n.x));
            n.y = Math.max(n.r + 10, Math.min(canvas.height - n.r - 10, n.y));
        });

        // Draw Canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw Links
        ctx.strokeStyle = 'rgba(0, 245, 255, 0.2)';
        ctx.lineWidth = 1.8;
        links.forEach(l => {
            const src = nodeLookup[l.source];
            const tgt = nodeLookup[l.target];
            if (src && tgt) {
                ctx.beginPath();
                ctx.moveTo(src.x, src.y);
                ctx.lineTo(tgt.x, tgt.y);
                ctx.stroke();
            }
        });

        // Draw Nodes
        physicsNodes.forEach(n => {
            ctx.shadowBlur = 16;
            ctx.shadowColor = n.color || '#00f5ff';
            ctx.fillStyle = n.color || '#00f5ff';
            ctx.beginPath();
            ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0;

            ctx.fillStyle = '#fff';
            ctx.font = '600 11px Plus Jakarta Sans';
            ctx.textAlign = 'center';
            ctx.fillText(n.label || n.id, n.x, n.y + n.r + 14);
        });

        kgSimulation = requestAnimationFrame(stepPhysics);
    }

    stepPhysics();
}
window.renderKnowledgeGraph = renderKnowledgeGraph;


// ── 9. Datasets & MoE Matrix ────────────────────────────────────────────────

async function loadDatasets() {
    try {
        const res = await fetch('/api/datasets');
        const data = await res.json();
        datasetsCatalog = Array.isArray(data) ? data : [];
        const container = document.getElementById('datasets-grid-container');
        if (!container) return;
        container.innerHTML = '';

        datasetsCatalog.forEach(ds => {
            const card = document.createElement('div');
            card.className = 'glass-card';
            card.style.padding = '20px';
            card.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div style="font-size:1rem; font-weight:700; color:#fff;">📁 ${ds.name}</div>
                    <span class="brand-badge badge-emerald">${ds.status || 'Ready'}</span>
                </div>
                <div style="font-size:0.75rem; color:var(--cyan); margin: 6px 0 12px 0;">${ds.type} • ${ds.size}</div>
                <div style="font-size:0.8rem; color:var(--text-muted); margin-bottom:4px;">Samples: <strong style="color:#fff;">${Number(ds.samples).toLocaleString()}</strong></div>
                <div style="font-size:0.8rem; color:var(--text-muted); margin-bottom:14px;">Tokens: <strong style="color:var(--emerald);">${ds.tokens}</strong></div>
                <button class="btn-action btn-secondary" style="width:100%; font-size:0.72rem;" onclick="openSampleModal('${ds.id}')">🔍 Inspect Samples</button>
            `;
            container.appendChild(card);
        });
    } catch(err) {
        console.error("Datasets error:", err);
    }
}
window.loadDatasets = loadDatasets;

function openSampleModal(datasetId) {
    const ds = datasetsCatalog.find(d => d.id === datasetId);
    if (!ds) return;

    const modal = document.getElementById('sample-modal');
    const title = document.getElementById('modal-dataset-title');
    const list = document.getElementById('modal-samples-list');

    if (title) title.innerText = `Samples: ${ds.name}`;
    if (list) {
        list.innerHTML = '';
        (ds.sample_preview || []).forEach((sample, idx) => {
            const item = document.createElement('div');
            item.style.padding = '12px';
            item.style.borderRadius = '8px';
            item.style.background = 'rgba(0,0,0,0.5)';
            item.style.border = '1px solid var(--border-light)';
            item.innerHTML = `
                <div style="font-size:0.75rem; font-weight:700; color:var(--cyan); margin-bottom:4px;">Sample #${idx+1} Prompt:</div>
                <div style="font-size:0.8rem; color:#fff; margin-bottom:8px; font-family:var(--font-mono);">${sample.prompt || 'N/A'}</div>
                <div style="font-size:0.75rem; font-weight:700; color:var(--emerald); margin-bottom:4px;">Completion:</div>
                <div style="font-size:0.8rem; color:#ddd; font-family:var(--font-mono);">${sample.completion || 'N/A'}</div>
            `;
            list.appendChild(item);
        });
    }

    if (modal) modal.style.display = 'flex';
}
window.openSampleModal = openSampleModal;

function closeSampleModal() {
    const modal = document.getElementById('sample-modal');
    if (modal) modal.style.display = 'none';
}
window.closeSampleModal = closeSampleModal;

async function renderExperts() {
    try {
        const res = await fetch('/api/experts');
        const data = await res.json();
        const container = document.getElementById('experts-grid-container');
        if (!container) return;
        container.innerHTML = '';

        const experts = data.experts || [];
        experts.forEach(exp => {
            const card = document.createElement('div');
            card.className = 'glass-card';
            card.style.padding = '18px';
            card.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <span style="font-size:0.95rem; font-weight:700; color:#fff;">${exp.icon || '🧬'} ${exp.name || 'Expert'}</span>
                    <span class="brand-badge" style="color:var(--cyan); border-color:rgba(0,245,255,0.3);">${exp.specialization || 'Core'}</span>
                </div>
                <div style="font-size:0.78rem; color:var(--text-muted); margin-bottom:6px;">Status: <strong style="color:var(--emerald);">Online (Top-${data.top_k || 2})</strong></div>
                <div style="font-size:0.75rem; color:var(--text-sub); margin-bottom:8px;">Routing Share: <strong style="color:#fff;">${exp.load_percentage}%</strong></div>
                <div class="progress-track" style="height:6px;">
                    <div class="progress-fill" style="width:${exp.load_percentage}%;"></div>
                </div>
            `;
            container.appendChild(card);
        });
    } catch(err) {
        console.error("MoE render error:", err);
    }
}
window.renderExperts = renderExperts;


// ── 10. Hardware & Neural Telemetry ─────────────────────────────────────────

async function loadTelemetry() {
    try {
        const res = await fetch('/api/telemetry');
        const data = await res.json();
        
        // 4 KPI Stats
        const container = document.getElementById('telemetry-stats-container');
        if (container) {
            container.innerHTML = '';
            const metrics = [
                { label: 'DEVICE & CORES', val: data.device?.toUpperCase() || 'CPU', sub: `${data.hardware?.cpu_threads || 8} Threads • ${data.hardware?.simd || 'AVX2'}` },
                { label: 'MEMORY LOAD', val: `${data.hardware?.ram_used_gb || 0} GB`, sub: `Total: ${data.hardware?.ram_total_gb || 0} GB (${data.hardware?.ram_percent || 0}%)` },
                { label: 'ACTIVE MODEL', val: data.parameters_formatted || '82.8M', sub: `${data.layers || 10} Layers • ${data.quantization || 'BitNet 1.58b'}` },
                { label: 'MTP SPECULATION', val: data.hardware?.mtp_speedup || '2.35x', sub: 'Multi-Token Head Active' }
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
        }

        // Per-Core CPU Activity Bars
        const coreBox = document.getElementById('cpu-cores-container');
        if (coreBox && data.hardware?.per_core_pct) {
            coreBox.innerHTML = '';
            data.hardware.per_core_pct.forEach((pct, idx) => {
                const row = document.createElement('div');
                row.style.display = 'flex';
                row.style.alignItems = 'center';
                row.style.gap = '10px';
                row.innerHTML = `
                    <span style="font-size:0.72rem; color:var(--text-muted); width:50px;">Core #${idx}</span>
                    <div class="progress-track" style="flex:1; height:6px;">
                        <div class="progress-fill" style="width:${pct}%;"></div>
                    </div>
                    <span style="font-size:0.72rem; color:var(--cyan); font-family:var(--font-mono); width:40px; text-align:right;">${pct}%</span>
                `;
                coreBox.appendChild(row);
            });
        }

        // Memory Breakdown
        const memBox = document.getElementById('memory-breakdown-container');
        if (memBox && data.hardware) {
            const hw = data.hardware;
            memBox.innerHTML = `
                <div style="font-size:0.82rem; color:#fff; display:flex; justify-content:space-between;">
                    <span>System RAM Allocation:</span>
                    <strong style="color:var(--cyan);">${hw.ram_used_gb} GB / ${hw.ram_total_gb} GB</strong>
                </div>
                <div class="progress-track" style="height:10px; margin: 4px 0 14px 0;">
                    <div class="progress-fill" style="width:${hw.ram_percent}%;"></div>
                </div>
                ${hw.gpu ? `
                    <div style="font-size:0.82rem; color:#fff; display:flex; justify-content:space-between;">
                        <span>GPU VRAM (${hw.gpu.name}):</span>
                        <strong style="color:var(--emerald);">${hw.gpu.vram_allocated_mb} MB / ${hw.gpu.vram_total_mb} MB</strong>
                    </div>
                    <div class="progress-track" style="height:10px; margin-top:4px;">
                        <div class="progress-fill" style="width:${hw.gpu.vram_utilization_pct}%; background:linear-gradient(90deg, var(--emerald), var(--cyan));"></div>
                    </div>
                ` : `<div style="font-size:0.78rem; color:var(--text-muted);">Dedicated GPU VRAM: Not active (Optimized CPU SIMD inference)</div>`}
            `;
        }
    } catch(err) {
        console.error("Telemetry load error:", err);
    }
}
window.loadTelemetry = loadTelemetry;


// ── 11. Code Sandbox & AST Execution ────────────────────────────────────────

function setSandboxSnippet(name) {
    const input = document.getElementById('sandbox-code-input');
    if (!input) return;

    if (name === 'benchmark') {
        input.value = `import time
import numpy as np

# Tantra High-Speed Matrix Benchmark
a = np.random.randn(1000, 1000).astype(np.float32)
b = np.random.randn(1000, 1000).astype(np.float32)

t0 = time.perf_counter()
c = a @ b
elapsed_ms = (time.perf_counter() - t0) * 1000

print(f"Matrix Multiply (1000x1000): {elapsed_ms:.2f} ms")
print(f"Result checksum: {float(c.sum()):.4f}")`;
    } else if (name === 'bitnet') {
        input.value = `import torch

# BitNet 1.58-bit Ternary Quantization Simulation
def quantize_158b(w):
    scale = w.abs().mean().clamp(min=1e-5)
    w_q = torch.round(w / scale).clamp(-1, 1)
    return w_q, scale

w = torch.randn(8, 8)
w_ternary, scale = quantize_158b(w)

print("Original Weights Sample:\\n", w[:2, :4])
print("\\nQuantized Ternary {-1, 0, +1}:\\n", w_ternary[:2, :4])
print("\\nTernary Values Present:", torch.unique(w_ternary).tolist())`;
    } else if (name === 'tokenize') {
        input.value = `sample_text = "Namaste! Tantra Studio with BitNet 1.58b."
print("Input string:", sample_text)
print("Length in bytes:", len(sample_text.encode('utf-8')))
print("Length in characters:", len(sample_text))`;
    }
}
window.setSandboxSnippet = setSandboxSnippet;

async function runSandboxCode() {
    const code = document.getElementById('sandbox-code-input').value;
    const output = document.getElementById('sandbox-output-display');
    const timeBadge = document.getElementById('sandbox-time-badge');
    
    if (output) output.innerText = "Executing in AST sandbox...";
    try {
        const res = await fetch('/api/sandbox/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code: code })
        });
        const data = await res.json();
        if (output) output.innerText = data.result || "Execution finished (no output).";
        if (timeBadge && data.elapsed_ms != null) timeBadge.innerText = `Elapsed: ${data.elapsed_ms} ms`;
    } catch(err) {
        if (output) output.innerText = `Execution Error: ${err.message}`;
    }
}
window.runSandboxCode = runSandboxCode;


// ── 12. Admin Suite ─────────────────────────────────────────────────────────

async function adminSwitchCheckpoint() {
    const ckpt = document.getElementById('admin-ckpt-select').value;
    const resultDiv = document.getElementById('admin-swap-result');
    if (resultDiv) resultDiv.innerText = "Swapping checkpoint...";

    try {
        const res = await fetch('/api/checkpoints', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ checkpoint: ckpt })
        });
        const data = await res.json();
        if (data.active) {
            if (resultDiv) resultDiv.innerText = `✅ Checkpoint successfully swapped to: ${data.active}`;
        } else {
            if (resultDiv) resultDiv.innerText = `Response: ${JSON.stringify(data)}`;
        }
    } catch(err) {
        if (resultDiv) resultDiv.innerText = `❌ Error: ${err.message}`;
    }
}
window.adminSwitchCheckpoint = adminSwitchCheckpoint;


// ── Initial Boot ────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    loadSessions();
    loadRAGDocuments();
    loadMemoryBank();
    loadDatasets();
    pollLiveTelemetry();

    // Start auto polling timer
    livePollInterval = setInterval(pollLiveTelemetry, pollRateMs);
});
