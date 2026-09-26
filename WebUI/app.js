// Tantra WebUI — chat, training dashboard, model manager. No build step, no framework.
"use strict";

// ── helpers ──────────────────────────────────────────────────────────────────
const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const store = {
  get(k, d) { try { const v = localStorage.getItem("tantra." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("tantra." + k, JSON.stringify(v)); } catch { /* private mode */ } },
};
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 2) => (v == null || Number.isNaN(+v) ? "—" : (+v).toLocaleString(undefined, { maximumFractionDigits: d }));
const fmtTokens = (n) => (n == null ? "—" : n >= 1e9 ? (n / 1e9).toFixed(2) + "B" : n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "K" : String(n));
const ago = (t) => {
  if (!t) return "—";
  const s = Date.now() / 1000 - t;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(t * 1000).toLocaleDateString();
};

function toast(msg, kind = "") {
  const t = el("div", `toast ${kind}`, msg);
  $("#toasts").append(t);
  setTimeout(() => t.remove(), kind === "error" ? 7000 : 3500);
}

async function api(url, opts = {}) {
  const headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
  const key = store.get("apiKey", "");
  if (key) headers["X-API-Key"] = key;
  const r = await fetch(url, { ...opts, headers });
  if (r.status === 401) {
    const k = prompt("This server is protected. Enter the TANTRA_API_KEY:");
    if (k) { store.set("apiKey", k); return api(url, opts); }
  }
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `${r.status} ${r.statusText}`);
  return r;
}
const getJSON = (url) => api(url).then((r) => r.json());
const postJSON = (url, body = {}) => api(url, { method: "POST", body: JSON.stringify(body) }).then((r) => r.json());

// Tiny, safe markdown: code blocks, inline code, bold/italic, lists, paragraphs.
function markdown(src) {
  const blocks = [];
  let s = String(src).replace(/```(\w*)\n?([\s\S]*?)(```|$)/g, (_, lang, code) => {
    blocks.push(`<pre><code${lang ? ` data-lang="${esc(lang)}"` : ""}>${esc(code.replace(/\n$/, ""))}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });
  s = esc(s)
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<i>$2</i>");
  const out = [];
  for (const para of s.split(/\n{2,}/)) {
    const lines = para.split("\n");
    if (lines.every((l) => /^\s*[-*•] /.test(l))) out.push(`<ul>${lines.map((l) => `<li>${l.replace(/^\s*[-*•] /, "")}</li>`).join("")}</ul>`);
    else if (lines.every((l) => /^\s*\d+[.)] /.test(l))) out.push(`<ol>${lines.map((l) => `<li>${l.replace(/^\s*\d+[.)] /, "")}</li>`).join("")}</ol>`);
    else if (/^\u0000\d+\u0000$/.test(para.trim())) out.push(para.trim());
    else out.push(`<p>${lines.join("<br>")}</p>`);
  }
  return out.join("").replace(/\u0000(\d+)\u0000/g, (_, i) => blocks[+i]);
}

// ── theme ────────────────────────────────────────────────────────────────────
const THEMES = ["auto", "light", "dark"];
function applyTheme(t) {
  if (t === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.dataset.theme = t;
  $("#theme").title = `Theme: ${t} (click to change)`;
  $("#theme").textContent = t === "dark" ? "☾" : t === "light" ? "☀" : "◐";
}
applyTheme(store.get("theme", "light"));
$("#theme").onclick = () => {
  const next = THEMES[(THEMES.indexOf(store.get("theme", "light")) + 1) % 3];
  store.set("theme", next); applyTheme(next);
};

// ── tabs (remembered in the URL hash) ────────────────────────────────────────
function showTab(name) {
  if (!$(`#tab-${name}`)) name = "chat";
  $$("nav button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${name}`));
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  if (name === "chat") $("#input").focus();
  refreshStatus();
}
$$("nav button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));
const activeTab = () => $("nav button.active")?.dataset.tab;

// ── generation settings (saved in this browser) ─────────────────────────────
const DEFAULTS = { temperature: 0.3, top_p: 0.9, repetition_penalty: 1.15, max_tokens: 256, history: 3, auto_speak: false, system: "", category: "auto" };
let settings = { ...DEFAULTS, ...store.get("settings", {}) };
function bindSettings() {
  for (const [k, v] of Object.entries(settings)) {
    const input = $(`#${k}`);
    if (!input) continue;
    if (input.type === "checkbox") input.checked = !!v; else input.value = v;
    const out = input.nextElementSibling;
    if (out?.tagName === "OUTPUT") out.textContent = input.value;
  }
}
$("#settings").addEventListener("input", (e) => {
  const t = e.target;
  if (!t.id || !(t.id in DEFAULTS)) return;
  settings[t.id] = t.type === "checkbox" ? t.checked : t.type === "range" || t.type === "number" ? +t.value : t.value;
  if (t.nextElementSibling?.tagName === "OUTPUT") t.nextElementSibling.textContent = t.value;
  store.set("settings", settings);
});
$("#settings-btn").onclick = () => { $("#settings").hidden = !$("#settings").hidden; $("#settings-btn").classList.toggle("on"); };
$("#settings-reset").onclick = () => { settings = { ...DEFAULTS }; store.set("settings", settings); bindSettings(); toast("Settings reset."); };
bindSettings();

// ── chat ─────────────────────────────────────────────────────────────────────
let chat = { id: null, messages: [] };
let chatsCache = {};
let controller = null;
let status = {};

const SUGGESTIONS = ["नमस्ते, आप कौन हैं?", "भारत की राजधानी क्या है?", "Bhai, Python me list ko sort kaise karte hain?", "Explain photosynthesis in simple words.", "2 + 2 × 3 कितना होता है?"];

function nearBottom() { const m = $("#messages"); return m.scrollHeight - m.scrollTop - m.clientHeight < 80; }
function scrollDown(force) { const m = $("#messages"); if (force || nearBottom()) m.scrollTop = m.scrollHeight; }

function renderMessages() {
  const box = $("#messages");
  box.innerHTML = "";
  if (!chat.messages.length) {
    const e = el("div", "empty");
    e.innerHTML = `<h1>तन्त्र</h1><p>नमस्ते! Ask anything in Hindi, Hinglish or English.</p>`;
    const sug = el("div", "suggestions");
    SUGGESTIONS.forEach((q) => { const b = el("button", null, q); b.onclick = () => send(q); sug.append(b); });
    e.append(sug);
    box.append(e);
    return;
  }
  chat.messages.forEach((m, i) => box.append(messageEl(m, i)));
  scrollDown(true);
}

function messageEl(m, i) {
  const d = el("div", `msg ${m.role}${m.error ? " error" : ""}`);
  const body = el("div", "bubble");
  if (m.role === "assistant") body.innerHTML = markdown(m.content || ""); else body.textContent = m.content;
  d.append(body);
  const meta = el("div", "meta");
  const btn = (label, title, fn) => { const b = el("button", null, label); b.title = title; b.onclick = fn; meta.append(b); };
  if (m.role === "assistant") {
    if (m.info) meta.append(el("span", null, m.info));
    btn("⧉", "Copy", () => navigator.clipboard.writeText(m.content).then(() => toast("Copied.")));
    btn("🔊", "Read aloud", () => speakText(m.content));
    if (i === chat.messages.length - 1) btn("↻", "Regenerate", regenerate);
  } else {
    btn("✎", "Edit and resend", () => editMessage(i));
  }
  d.append(meta);
  return d;
}

function setBusy(on) {
  $("#send").hidden = on; $("#stop").hidden = !on;
  $("#input").placeholder = on ? "Tantra is writing…" : "संदेश लिखें… / Type a message  (Enter = send, Shift+Enter = new line)";
}

async function send(text) {
  if (controller || !text || !text.trim()) return;
  chat.messages.push({ role: "user", content: text.trim() });
  await generate();
}

async function regenerate() {
  if (controller) return;
  if (chat.messages.at(-1)?.role === "assistant") chat.messages.pop();
  await generate();
}

function editMessage(i) {
  if (controller) return;
  const text = chat.messages[i].content;
  chat.messages = chat.messages.slice(0, i);
  renderMessages();
  $("#input").value = text; autosize(); $("#input").focus();
}

async function generate() {
  const reply = { role: "assistant", content: "" };
  chat.messages.push(reply);
  renderMessages();
  const bubble = $("#messages").lastChild.querySelector(".bubble");
  bubble.classList.add("streaming");
  $("#messages").lastChild.querySelector(".meta").hidden = true;   // copy/regenerate only when done
  controller = new AbortController();
  setBusy(true);
  const t0 = performance.now();
  let final = null, raf = 0;
  const paint = () => { raf = 0; bubble.innerHTML = markdown(reply.content); scrollDown(); };
  try {
    const messages = [];
    if (settings.system?.trim()) messages.push({ role: "system", content: settings.system.trim() });
    chat.messages.slice(0, -1).forEach(({ role, content }) => messages.push({ role, content }));
    const r = await api("/v1/chat/completions", {
      method: "POST", signal: controller.signal,
      body: JSON.stringify({
        messages, stream: true, temperature: settings.temperature, top_p: settings.top_p,
        repetition_penalty: settings.repetition_penalty, max_tokens: settings.max_tokens,
        history: settings.history, category: $("#category").value,
      }),
    });
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const events = buf.split("\n\n"); buf = events.pop();
      for (const ev of events) {
        const data = ev.replace(/^data: /, "").trim();
        if (!data || data === "[DONE]") continue;
        let j; try { j = JSON.parse(data); } catch { continue; }
        const delta = j.choices?.[0]?.delta?.content;
        if (delta) { reply.content += delta; if (!raf) raf = requestAnimationFrame(paint); }
        if (j.choices?.[0]?.finish_reason) final = j;
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") { reply.error = true; reply.content = reply.content || `⚠ ${e.message}`; }
  }
  const secs = (performance.now() - t0) / 1000;
  const n = final?.usage?.completion_tokens;
  const parts = [];
  if (n != null) parts.push(`${n} tokens`);
  parts.push(`${secs.toFixed(1)}s`);
  if (final?.timing?.tokens_per_sec) parts.push(`${final.timing.tokens_per_sec} tok/s`);
  if (final?.category) parts.push(final.category);
  if (controller.signal.aborted) parts.push("stopped");
  else if (final?.choices?.[0]?.finish_reason === "length") parts.push("hit max tokens");
  reply.info = parts.join(" · ");
  if (!reply.content && !reply.error) reply.content = "(no reply — the model ended immediately)";
  controller = null;
  setBusy(false);
  renderMessages();
  saveChat();
  refreshStatus();   // the model may have just loaded
  if (settings.auto_speak && reply.content && !reply.error) speakText(reply.content);
}

$("#stop").onclick = () => controller?.abort();
$("#composer").onsubmit = (e) => { e.preventDefault(); const t = $("#input").value; $("#input").value = ""; autosize(); send(t); };
function autosize() { const t = $("#input"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, innerHeight * 0.4) + "px"; }
$("#input").addEventListener("input", autosize);
$("#input").onkeydown = (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("#composer").requestSubmit(); }
  if (e.key === "Escape" && controller) controller.abort();
};
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); showTab("chat"); newChat(); }
});

// ── saved chats ──
async function saveChat() {
  const r = await postJSON("/api/chats", chat).catch(() => null);
  if (r) { chat.id = r.id; chat.title = r.title; loadChatList(); }
}
async function loadChatList() {
  chatsCache = await getJSON("/api/chats").catch(() => chatsCache);
  renderChatList();
}
function renderChatList() {
  const q = $("#chat-search").value.trim().toLowerCase();
  const ul = $("#chat-list"); ul.innerHTML = "";
  const today = new Date().setHours(0, 0, 0, 0) / 1000;
  let group = null;
  Object.values(chatsCache)
    .filter((c) => !q || c.title.toLowerCase().includes(q) || c.messages.some((m) => m.content.toLowerCase().includes(q)))
    .sort((a, b) => b.updated - a.updated)
    .forEach((c) => {
      const g = c.updated >= today ? "Today" : "Earlier";
      if (g !== group) { group = g; ul.append(el("li", "group", g)); }
      const li = el("li", c.id === chat.id ? "active" : "");
      li.title = "Double-click to rename";
      li.append(el("span", null, c.title));
      const del = el("button", null, "✕"); del.title = "Delete";
      del.onclick = async (e) => {
        e.stopPropagation();
        if (!confirm(`Delete “${c.title}”?`)) return;
        await api(`/api/chats/${encodeURIComponent(c.id)}`, { method: "DELETE" }).catch((err) => toast(err.message, "error"));
        if (c.id === chat.id) newChat(); else loadChatList();
      };
      li.append(del);
      li.onclick = () => { if (controller) return; chat = { id: c.id, title: c.title, messages: c.messages }; renderMessages(); renderChatList(); $("#chats").classList.remove("open"); };
      li.ondblclick = async () => {
        const t = prompt("Rename chat", c.title);
        if (t && t.trim()) { await postJSON("/api/chats", { ...c, title: t.trim() }); if (c.id === chat.id) chat.title = t.trim(); loadChatList(); }
      };
      ul.append(li);
    });
  if (!ul.children.length) ul.append(el("li", "group", q ? "No matches" : "No saved chats yet"));
}
function newChat() { if (controller) controller.abort(); chat = { id: null, messages: [] }; renderMessages(); renderChatList(); $("#input").focus(); $("#chats").classList.remove("open"); }
$("#new-chat").onclick = newChat;
$("#chat-search").oninput = renderChatList;
$("#sidebar-toggle").onclick = () => { showTab("chat"); $("#chats").classList.toggle("open"); };

// ── voice ──
let recorder = null;
$("#mic").onclick = async () => {
  if (status.speech && !status.speech.stt) { toast("Speech-to-text is not installed. Run: pip install openai-whisper (needs ffmpeg)", "error"); return; }
  if (recorder) { recorder.stop(); return; }
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); } catch { toast("Microphone not available or permission denied.", "error"); return; }
  const parts = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => parts.push(e.data);
  recorder.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    $("#mic").classList.remove("recording"); recorder = null;
    const fd = new FormData(); fd.append("audio", new Blob(parts, { type: "audio/webm" }), "speech.webm");
    toast("Transcribing…");
    try {
      const r = await fetch("/api/stt", { method: "POST", body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail);
      $("#input").value = ($("#input").value + " " + j.text).trim(); autosize(); $("#input").focus();
    } catch (e) { toast(e.message, "error"); }
  };
  recorder.start(); $("#mic").classList.add("recording"); toast("Recording… click the mic again to stop.");
};

let audio = null;
async function speakText(text) {
  if (status.speech && !status.speech.tts) { toast("Text-to-speech is not installed. Run: pip install kokoro soundfile", "error"); return; }
  try {
    audio?.pause();
    const r = await fetch("/api/tts", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
    if (!r.ok) throw new Error((await r.json()).detail);
    audio = new Audio(URL.createObjectURL(await r.blob()));
    audio.play();
  } catch (e) { toast(e.message, "error"); }
}

// ── charts (SVG, hover tooltips) ─────────────────────────────────────────────
function niceTicks(lo, hi, n = 4) {
  if (lo === hi) { lo -= 1; hi += 1; }
  const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  const ticks = [], end = Math.ceil(hi / step - 1e-9) * step;   // always cover the highest point
  for (let v = Math.floor(lo / step) * step; v <= end + step * 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

/** series: [{name, color, points:[{x,y}], axis:"left"|"right", dots, fmt}] */
function lineChart(box, series, { leftLabel = "", rightLabel = "", rightRange = null, empty = "No data yet." } = {}) {
  const all = series.flatMap((s) => s.points);
  if (!all.length) { box.innerHTML = `<div class="empty-chart">${esc(empty)}</div>`; return; }
  const W = Math.max(box.clientWidth, 300), H = 240, L = 48, R = series.some((s) => s.axis === "right") ? 48 : 14, T = 12, B = 28;
  const xs = all.map((p) => p.x), xmin = Math.min(...xs), xmax = Math.max(...xs);
  const X = (x) => L + (W - L - R) * (xmax === xmin ? 0.5 : (x - xmin) / (xmax - xmin));
  const range = (axis) => {
    const ys = series.filter((s) => (s.axis || "left") === axis).flatMap((s) => s.points.map((p) => p.y));
    if (axis === "right" && rightRange) return rightRange;
    if (!ys.length) return [0, 1];
    const t = niceTicks(Math.min(...ys), Math.max(...ys));
    return [t[0], t.at(-1)];
  };
  const [l0, l1] = range("left"), [r0, r1] = range("right");
  const Y = (y, axis) => { const [a, b] = axis === "right" ? [r0, r1] : [l0, l1]; return T + (H - T - B) * (1 - (b === a ? 0.5 : (y - a) / (b - a))); };
  let svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
  for (const v of niceTicks(l0, l1)) {
    svg += `<line class="grid-line" x1="${L}" x2="${W - R}" y1="${Y(v, "left")}" y2="${Y(v, "left")}"/>`;
    svg += `<text x="${L - 6}" y="${Y(v, "left") + 4}" text-anchor="end">${fmt(v, 2)}</text>`;
  }
  if (R > 14) for (const v of niceTicks(r0, r1)) svg += `<text x="${W - R + 6}" y="${Y(v, "right") + 4}">${fmt(v, 0)}</text>`;
  for (const v of niceTicks(xmin, xmax, 5).filter((v) => v >= xmin && v <= xmax))
    svg += `<text x="${X(v)}" y="${H - 8}" text-anchor="middle">${fmtTokens(v)}</text>`;
  if (leftLabel) svg += `<text x="4" y="${T + 2}" transform="rotate(-90 4 ${T + 2})" text-anchor="end" dy="9">${esc(leftLabel)}</text>`;
  if (rightLabel) svg += `<text x="${W - 4}" y="${T + 2}" transform="rotate(90 ${W - 4} ${T + 2})" dy="0">${esc(rightLabel)}</text>`;
  for (const s of series) {
    const pts = s.points.map((p) => `${X(p.x).toFixed(1)},${Y(p.y, s.axis).toFixed(1)}`);
    if (s.points.length > 1 && !s.dotsOnly) svg += `<polyline fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" points="${pts.join(" ")}"/>`;
    if (s.dots || s.points.length === 1) s.points.forEach((p) => (svg += `<circle cx="${X(p.x)}" cy="${Y(p.y, s.axis)}" r="3.5" fill="${s.color}" stroke="var(--panel)" stroke-width="1.5"/>`));
  }
  svg += `<line class="hover-line" x1="0" x2="0" y1="${T}" y2="${H - B}" stroke="var(--grey)" stroke-dasharray="3 3" visibility="hidden"/></svg>`;
  box.innerHTML = svg + `<div class="tip" hidden></div>`;
  const svgEl = $("svg", box), tip = $(".tip", box), hl = $(".hover-line", box);
  svgEl.onmousemove = (e) => {
    const rect = svgEl.getBoundingClientRect(), mx = ((e.clientX - rect.left) / rect.width) * W;
    let best = null;
    for (const s of series) for (const p of s.points) { const d = Math.abs(X(p.x) - mx); if (!best || d < best.d) best = { d, x: p.x }; }
    if (!best) return;
    const rows = series.map((s) => {
      const p = s.points.reduce((a, b) => (Math.abs(b.x - best.x) < Math.abs(a.x - best.x) ? b : a), s.points[0]);
      return p && Math.abs(p.x - best.x) <= (xmax - xmin) * 0.02 + 1e-9 ? `<span style="color:${s.color}">●</span> ${esc(s.name)}: <b>${(s.fmt || ((v) => fmt(v, 3)))(p.y)}</b>` : null;
    }).filter(Boolean);
    hl.setAttribute("x1", X(best.x)); hl.setAttribute("x2", X(best.x)); hl.setAttribute("visibility", "visible");
    tip.innerHTML = `step ${best.x.toLocaleString()}<br>${rows.join("<br>")}`;
    tip.hidden = false;
    tip.style.left = `${(X(best.x) / W) * rect.width}px`; tip.style.top = `${e.clientY - rect.top}px`;
  };
  svgEl.onmouseleave = () => { tip.hidden = true; hl.setAttribute("visibility", "hidden"); };
}

// ── status polling ───────────────────────────────────────────────────────────
function dl(target, rows) {
  const d = $(target); d.innerHTML = "";
  rows.forEach(([k, v, cls]) => { d.append(el("dt", null, k)); const dd = el("dd", cls || null, v); d.append(dd); });
}
const tile = (k, v, s = "", cls = "") => `<div class="tile"><div class="k">${esc(k)}</div><div class="v ${cls}">${esc(v)}</div><div class="s">${esc(s)}</div></div>`;
const STATUS_CLASS = { running: "ok", complete: "ok", stopped: "warn", interrupted: "warn", idle: "muted" };

let polling = null;
const logKeys = {};
function schedule(ms) { clearTimeout(polling); polling = setTimeout(refreshStatus, ms); }

async function refreshStatus() {
  let s;
  try { s = await getJSON("/api/status"); $("#conn").className = "conn ok"; $("#conn").title = "Connected"; }
  catch { $("#conn").className = "conn bad"; $("#conn").title = "Server offline"; $("#model-badge").textContent = "server offline — run tantra.bat → 3"; schedule(5000); return; }
  status = s;
  const m = s.model || {}, t = s.training || {}, jobs = s.jobs || {};
  const busyJob = t.status === "running" || Object.values(jobs).some((j) => j.running);

  $("#model-badge").textContent = m.checkpoint
    ? `${m.params_M}M params · step ${fmt(m.step, 0)}${m.int8 ? " · int8" : ""}`
    : s.checkpoints?.length ? "model loads on first message" : "no trained model yet";
  $("#train-dot").hidden = t.status !== "running";
  const sel = $("#category");
  (m.categories || []).forEach((c) => { if (![...sel.options].some((o) => o.value === c)) sel.append(new Option(c, c)); });
  sel.value = [...sel.options].some((o) => o.value === settings.category) ? settings.category : "auto";
  $("#mic").title = s.speech?.stt ? "Speak (Whisper)" : "Speech-to-text not installed (pip install openai-whisper)";
  $("#mic").style.opacity = s.speech?.stt ? 1 : 0.45;
  qualityBanner(s);

  const tab = activeTab();
  if (tab === "training") renderTraining(s);
  if (tab === "model") renderModel(s);
  schedule(document.hidden ? 30000 : busyJob && tab !== "chat" ? 2000 : 10000);
}
document.addEventListener("visibilitychange", () => { if (!document.hidden) refreshStatus(); });

function qualityBanner(s) {
  const b = $("#quality-banner"), m = s.model || {}, probe = (s.probe || []).filter((p) => p.hits != null).at(-1);
  let msg = "";
  if (!s.checkpoints?.length) msg = "No trained model yet. Open the <b>Training</b> tab and press <b>Start</b> — chat works once the first checkpoint is saved.";
  else if (s.load_error) msg = `⚠ ${esc(s.load_error)}`;
  else if (m.checkpoint) {
    const vl = m.val?.loss;
    if ((vl != null && vl > 4.5) || (probe && probe.hits < 5) || (m.step || 0) < 2000)
      msg = `This model is still early in training (step ${fmt(m.step, 0)}${vl != null ? `, val loss ${fmt(vl, 2)}` : ""}${probe ? `, remembered ${probe.hits}/50` : ""}). ` +
        `Replies will be mostly random words until it trains longer — watch <b>remembered X/50</b> in the Training tab.`;
  }
  b.innerHTML = msg; b.hidden = !msg;
}

function renderTraining(s) {
  const t = s.training || {}, v = t.validation || {}, st = t.status || "idle";
  const pct = t.target_steps ? Math.min(100, (100 * (t.step || 0)) / t.target_steps) : 0;
  $("#train-tiles").innerHTML = [
    tile("Status", st, t.updated_at ? `updated ${ago(t.updated_at)}` : "", STATUS_CLASS[st] || ""),
    tile("Step", `${fmt(t.step, 0)}`, t.target_steps ? `of ${fmt(t.target_steps, 0)} (${pct.toFixed(1)}%)` : ""),
    tile("Train loss", fmt(t.loss, 3), t.accuracy != null ? `accuracy ${fmt(t.accuracy, 1)}%` : ""),
    tile("Val loss", fmt(v.loss, 3), v.top1_accuracy_percent != null ? `top-1 ${v.top1_accuracy_percent}% · top-5 ${v.top5_accuracy_percent}%` : ""),
    tile("Speed", t.tok_s ? `${fmt(t.tok_s, 0)} tok/s` : "—", t.lr ? `lr ${Number(t.lr).toExponential(1)}` : ""),
    tile("ETA", st === "running" ? t.eta || "—" : "—", t.elapsed ? `session ${t.elapsed}` : ""),
    tile("Tokens seen", fmtTokens(t.total_tokens), t.best_val_loss != null ? `best val ${fmt(t.best_val_loss, 3)}` : ""),
  ].join("");
  $("#train-progress").style.width = `${pct}%`;
  $("#train-sub").textContent = st === "interrupted" ? "The last run stopped without saying so (closed window or crash). Press Start to continue from the last save." : "";
  const running = st === "running" || s.jobs?.train?.running;
  $("#train-start").disabled = running; $("#train-stop").disabled = !running;

  const hist = t.history || {};
  lineChart($("#loss-chart"), [
    { name: "train loss", color: "var(--accent)", points: (hist.train || []).map((p) => ({ x: p.step, y: p.loss })) },
    { name: "val loss", color: "#2563eb", dots: true, points: (hist.val || []).map((p) => ({ x: p.step, y: p.loss })) },
  ], { leftLabel: "loss", empty: "The loss curve appears after the first 50 training steps." });

  const probe = s.probe || [];
  const hits = probe.filter((p) => p.hits != null);
  lineChart($("#probe-chart"), [
    { name: "answer loss", color: "var(--grey)", points: probe.map((p) => ({ x: p.step, y: p.answer_loss })) },
    { name: "remembered /50", color: "var(--accent)", axis: "right", dots: true, fmt: (y) => `${y}/50`, points: hits.map((p) => ({ x: p.step, y: p.hits })) },
  ], { leftLabel: "answer loss", rightLabel: "remembered", rightRange: [0, 50], empty: "Probe results appear after the first evaluation." });
  const last = hits.at(-1);
  $("#probe-latest").innerHTML = last
    ? `<b>Remembered ${last.hits}/50</b> at step ${last.step.toLocaleString()} ` +
      Object.entries(last.by_category || {}).map(([k, val]) => `<span class="pill">${esc(k)}: ${esc(val)}</span>`).join("")
    : "";

  const dsSel = $("#train-data");
  const names = (s.datasets || []).map((d) => d.name).filter((n) => !/val|probe|test|preference/i.test(n));
  if (dsSel.dataset.names !== names.join()) {
    dsSel.dataset.names = names.join();
    dsSel.innerHTML = "";
    names.forEach((n) => dsSel.append(new Option(`${n} (${fmt(s.datasets.find((d) => d.name === n).size_mb, 0)} MB)`, n)));
    if (!names.length) dsSel.append(new Option("no .jsonl in Datasets/", ""));
    dsSel.value = names.includes("master_train.jsonl") ? "master_train.jsonl" : names[0] || "";
  }
  refreshLog("train", $("#train-log"), $("#log-follow").checked);
}

async function refreshLog(name, pre, follow = true) {
  try {
    const j = await getJSON(`/api/logs/${name}?lines=300`);
    if (!j.lines.length) return;
    const key = `${j.lines.length}|${j.lines.at(-1)}`;
    if (logKeys[name] === key) return;   // nothing new
    logKeys[name] = key;
    pre.hidden = false;
    const atEnd = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 40;
    pre.textContent = j.lines.join("\n");
    if (follow && atEnd || follow === "force") pre.scrollTop = pre.scrollHeight;
  } catch { /* no log yet */ }
}

$("#train-form").onsubmit = async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  const body = Object.fromEntries([...f.entries()].filter(([, val]) => val !== ""));
  body.fresh = f.has("fresh"); body.auto_growth = f.has("auto_growth");
  if (!body.fresh) delete body.preset;
  if (body.fresh && !confirm("Start a NEW model? The current checkpoints are moved to Model/_old (not deleted).")) return;
  try {
    $("#train-start").disabled = true;
    await postJSON("/api/training/start", body);
    toast("Training started. The first numbers appear after ~50 steps.", "ok");
    $("#log-follow").checked = true;
    setTimeout(refreshStatus, 1500);
  } catch (err) { toast(err.message, "error"); $("#train-start").disabled = false; }
};
$("#train-stop").onclick = async () => {
  try { await postJSON("/api/training/stop"); toast("Stopping after the current step — a checkpoint will be saved."); } catch (e) { toast(e.message, "error"); }
};

function renderModel(s) {
  const m = s.model || {}, h = s.hardware || {}, jobs = s.jobs || {};
  dl("#model-info", m.checkpoint ? [
    ["Checkpoint", m.checkpoint], ["Step", fmt(m.step, 0)], ["Parameters", `${m.params_M}M`],
    ["Shape", `${m.layers} layers × ${m.dim ?? "?"} dim`], ["Vocabulary", fmt(m.vocab, 0)],
    ["Trained on", m.tokens ? `${fmtTokens(m.tokens)} tokens` : "—"],
    ["Val loss", m.val?.loss != null ? fmt(m.val.loss, 3) : "—"],
    ["Categories", (m.categories || []).join(", ") || "none"], ["Runs on", `${m.device}${m.int8 ? " · INT8" : ""}`],
  ] : [["Status", s.load_error || "Not loaded yet — it loads on the first chat message, or press Load below."]]);
  dl("#hw-info", [
    ["CPU", h.cpu || "—"], ["Cores", `${h.physical_cores} physical / ${h.logical_cores} logical`],
    ["Threads used", fmt(h.cpu_threads, 0)], ["RAM", `${h.ram_gb} GB`], ["SIMD", h.simd],
    ["GPU", (h.gpus || []).join(", ") || "none"], ["Device", h.device],
    ["Speech", `in: ${s.speech?.stt ? "Whisper ✓" : "not installed"} · out: ${s.speech?.tts ? "Kokoro ✓" : "not installed"}`],
  ]);

  const tb = $("#ckpt-table tbody"); tb.innerHTML = "";
  const details = s.checkpoint_details || [];
  if (!details.length) tb.innerHTML = `<tr><td colspan="6" class="muted">No checkpoints yet — train first.</td></tr>`;
  details.forEach((c) => {
    const tr = el("tr", c.path === m.checkpoint ? "current" : "");
    tr.innerHTML = `<td>${esc(c.name)}</td><td>${fmt(c.step, 0)}</td><td>${fmt(c.val_loss, 3)}</td><td>${fmt(c.size_mb, 1)} MB</td><td title="${new Date(c.modified * 1000).toLocaleString()}">${ago(c.modified)}</td><td></td>`;
    const b = el("button", null, c.path === m.checkpoint ? "loaded" : "Load");
    b.disabled = c.path === m.checkpoint && !!m.int8 === $("#load-int8").checked;
    b.onclick = async () => {
      b.textContent = "loading…"; b.disabled = true;
      try { await postJSON("/api/checkpoints/switch", { path: c.path, int8: $("#load-int8").checked }); toast(`Loaded ${c.name}`, "ok"); }
      catch (e) { toast(e.message, "error"); }
      refreshStatus();
    };
    tr.lastChild.append(b); tb.append(tr);
  });

  const ev = s.eval_report;
  $("#eval-report").innerHTML = ev ? `<dl>
      <dt>Checkpoint</dt><dd>${esc(ev.checkpoint)} (step ${fmt(ev.step, 0)})</dd>
      ${ev.validation ? `<dt>Val loss</dt><dd>${fmt(ev.validation.loss, 3)} · top-1 ${ev.validation.top1_accuracy_percent}%</dd>` : ""}
      ${ev.probe?.hits != null ? `<dt>Remembered</dt><dd><b>${ev.probe.hits}/50</b> · answer loss ${fmt(ev.probe.answer_loss, 3)}</dd>` : ""}
      ${ev.speed ? `<dt>Speed</dt><dd>${fmt(ev.speed.forward_tokens_per_sec, 0)} tokens/s (forward)</dd>` : ""}
      <dt>When</dt><dd>${ago(ev.finished_at)}</dd></dl>` : `<p class="hint">No test run yet.</p>`;
  for (const name of ["eval", "export"]) {
    const j = jobs[name] || {};
    const btn = $(`#${name}-start`);
    btn.disabled = !!j.running;
    btn.textContent = j.running ? "running…" : name === "eval" ? "▶ Run test" : "⇩ Export tantra.pt";
    if (j.started) refreshLog(name, $(`#${name}-log`), "force");
    if (j.running === false && j.exit_code != null && btn.dataset.wasRunning === "1")
      toast(j.exit_code === 0 ? `${name === "eval" ? "Test" : "Export"} finished.` : `${name} failed — see its log.`, j.exit_code === 0 ? "ok" : "error");
    btn.dataset.wasRunning = j.running ? "1" : "0";
  }

  const ul = $("#dataset-list"); ul.innerHTML = "";
  (s.datasets || []).forEach((d) => { const li = el("li"); li.append(el("span", null, d.name), el("span", "muted", `${fmt(d.size_mb, 1)} MB`)); ul.append(li); });
  if (!ul.children.length) ul.append(el("li", "muted", "No .jsonl files in Datasets/"));
}

const loadedCkpt = () => status.model?.checkpoint ? { checkpoint: status.model.checkpoint } : {};
$("#eval-start").onclick = async () => {
  try { await postJSON("/api/jobs/eval", { ...loadedCkpt(), int8: $("#load-int8").checked }); toast("Test started — takes a minute or two."); refreshStatus(); }
  catch (e) { toast(e.message, "error"); }
};
$("#export-start").onclick = async () => {
  try { await postJSON("/api/jobs/export", loadedCkpt()); toast("Export started."); refreshStatus(); }
  catch (e) { toast(e.message, "error"); }
};
$("#reload-model").onclick = async () => {
  const first = status.checkpoints?.[0];
  if (!first) { toast("No checkpoints yet.", "error"); return; }
  try { await postJSON("/api/checkpoints/switch", { path: first, int8: $("#load-int8").checked }); toast("Model loaded.", "ok"); } catch (e) { toast(e.message, "error"); }
  refreshStatus();
};
$("#load-int8").onchange = () => refreshStatus();
window.addEventListener("resize", () => { if (activeTab() === "training") renderTraining(status); });

// ── start ──
renderMessages();
loadChatList();
showTab(location.hash.slice(1) || "chat");
