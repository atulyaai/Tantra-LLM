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

// ── tabs (remembered in the URL hash) ────────────────────────────────────────
function showTab(name) {
  if (!$(`#tab-${name}`)) name = "home";
  $$(".rail > button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${name}`));
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  if (name === "chat") $("#input").focus();
  if (name === "memory") loadMemory();
  if (name === "docs") loadDocs();
  if (name !== "voice" && typeof voiceStop === "function") voiceStop();
  refreshStatus();
}
$$(".rail > button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
document.addEventListener("click", (e) => {
  const go = e.target.closest("[data-go]");
  if (go) showTab(go.dataset.go);
  if (e.target.closest("[data-go-voice]")) showTab("voice");
});
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));
const activeTab = () => $(".rail > button.active")?.dataset.tab;

// ── generation settings (saved in this browser) ─────────────────────────────
const DEFAULTS = { temperature: 0.3, top_p: 0.9, repetition_penalty: 1.15, max_tokens: 256, history: 3, auto_speak: false, system: "", category: "auto", smriti: true, knowledge_first: true, force_model: false };
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

let CFG = { suggestions: [], skill_examples: [], wake_words: ["तन्त्र", "tantra"], stop_words: ["रुको", "stop"], voice: { silence_ms: 1100, max_seconds: 15, min_threshold: 4 }, name: "Tantra", name_hi: "तन्त्र" };

function nearBottom() { const m = $("#messages"); return m.scrollHeight - m.scrollTop - m.clientHeight < 80; }
function scrollDown(force) { const m = $("#messages"); if (force || nearBottom()) m.scrollTop = m.scrollHeight; }

function renderMessages() {
  const box = $("#messages");
  box.innerHTML = "";
  if (!chat.messages.length) {
    const e = el("div", "empty");
    e.innerHTML = `<h1>${esc(CFG.name_hi)}</h1><p>नमस्ते! Ask anything in Hindi, Hinglish or English.</p>`;
    const sug = el("div", "suggestions");
    (CFG.suggestions || []).forEach((q) => { const b = el("button", null, q); b.onclick = () => send(q); sug.append(b); });
    e.append(sug);
    box.append(e);
    return;
  }
  chat.messages.forEach((m, i) => box.append(messageEl(m, i)));
  scrollDown(true);
}

const SKILL_LABEL = { calculator: "Calculator · exact", time: "Time & date", units: "Unit converter", memory: "Memory",
  reminder: "Reminder", brief: "Daily brief", status: "System status", files: "File search", open: "Open app",
  taught: "Answer you taught", knowledge: "From knowledge (Smriti / documents)", small_talk: "", no_model: "Model still training" };

function skillBody(m, body) {
  const c = m.card || {};
  body.innerHTML = "";
  const label = SKILL_LABEL[m.skill] ?? m.skill;
  if (label) body.append(el("div", "skill-tag", label));
  if (m.skill === "calculator" && c.result) {
    const box = el("div", "calc");
    if ((c.steps || []).length > 1) box.append(el("div", "steps", c.steps.join("\n")));
    else box.append(el("div", "steps", c.expression));
    box.append(el("div", "big", c.result));
    body.append(box);
    return;
  }
  const text = el("div"); text.innerHTML = markdown(m.content || ""); body.append(text);
  if (m.skill === "open" && c.open && !m.opened) {
    const row = el("div", "confirm");
    const yes = el("button", "primary", `Open ${c.open}`), no = el("button", null, "Cancel");
    yes.onclick = async () => {
      try { await postJSON("/api/open", { name: c.open }); m.opened = true; toast(`Opened ${c.open}.`, "ok"); } catch (e) { toast(e.message, "error"); }
      row.remove(); saveChat();
    };
    no.onclick = () => { m.opened = true; row.remove(); saveChat(); };
    row.append(yes, no); body.append(row);
  }
}

function addRunButtons(body) {
  body.querySelectorAll('pre code[data-lang="python"], pre code[data-lang="py"]').forEach((code) => {
    const pre = code.parentElement, b = el("button", "run", "▶ Run");
    b.title = "Run this Python on this computer (10 s limit)";
    b.onclick = async () => {
      if (!confirm("Run this Python code on this computer?\nIt runs in an isolated interpreter in a temporary folder, stopped after 10 seconds.")) return;
      b.disabled = true; b.textContent = "running…";
      let out = pre.nextElementSibling?.classList.contains("run-out") ? pre.nextElementSibling : null;
      if (!out) { out = el("div", "run-out"); pre.after(out); }
      try {
        const r = await postJSON("/api/code/run", { code: code.textContent });
        out.textContent = (r.stdout || "") + (r.stderr ? `\n${r.stderr}` : "") + `\n— exit ${r.exit_code} · ${r.seconds}s`;
      } catch (e) { out.textContent = e.message; }
      b.disabled = false; b.textContent = "▶ Run";
    };
    pre.append(b);
  });
}

function teachBox(d, i) {
  if (d.querySelector(".teach")) return;
  const q = chat.messages.slice(0, i).reverse().find((x) => x.role === "user")?.content || "";
  const box = el("form", "teach");
  const ta = el("textarea"); ta.rows = 2; ta.placeholder = "सही उत्तर लिखें / Write the correct answer"; ta.required = true;
  const row = el("div", "row"), ok = el("button", "primary", "Teach Tantra"), cancel = el("button", null, "Cancel");
  cancel.type = "button"; cancel.onclick = () => box.remove();
  box.onsubmit = async (e) => {
    e.preventDefault();
    try {
      await postJSON("/api/feedback", { question: q, bad: chat.messages[i].content, correct: ta.value });
      chat.messages[i] = { ...chat.messages[i], content: ta.value, skill: "taught", card: {}, info: "corrected by you" };
      toast("Thanks — I'll answer this way next time.", "ok");
      renderMessages(); saveChat();
    } catch (err) { toast(err.message, "error"); }
  };
  row.append(ok, cancel); box.append(el("span", "hint", `Question: ${q.slice(0, 120)}`), ta, row);
  d.append(box); ta.focus();
}

function messageEl(m, i) {
  const d = el("div", `msg ${m.role}${m.error ? " error" : ""}`);
  const body = el("div", "bubble");
  if (m.role === "assistant" && m.skill) skillBody(m, body);
  else if (m.role === "assistant") { body.innerHTML = markdown(m.content || ""); addRunButtons(body); }
  else body.textContent = m.content;
  d.append(body);
  const meta = el("div", "meta");
  const btn = (label, title, fn) => { const b = el("button", null, label); b.title = title; b.onclick = fn; meta.append(b); };
  if (m.role === "assistant") {
    if (m.sources?.length) d.append(sourcesEl(m.sources));
    if (m.info) meta.append(el("span", null, m.info));
    btn("⧉", "Copy", () => navigator.clipboard.writeText(m.content).then(() => toast("Copied.")));
    btn("🔊", "Read aloud", () => speakText(m.content));
    if (i === chat.messages.length - 1 && !m.skill) btn("↻", "Regenerate", regenerate);
    if (!["calculator", "time", "units", "memory", "reminder", "small_talk"].includes(m.skill)) btn("👎", "Wrong? Teach the right answer", () => teachBox(d, i));
  } else {
    btn("✎", "Edit and resend", () => editMessage(i));
  }
  d.append(meta);
  return d;
}

function hitItem(h) {
  const li = el("li");
  if (h.question) li.append(el("b", null, h.question));
  li.append(el("span", null, h.text.length > 400 ? h.text.slice(0, 400) + "…" : h.text));
  li.append(el("div", "muted", `${h.source} · ${h.kind === "qa" ? "Q&A" : "text"} · score ${h.score}`));
  return li;
}
function sourcesEl(hits) {
  const d = el("details", "sources");
  const docs = hits.filter((h) => h.source && !/\.jsonl$/.test(h.source)).length;
  d.append(el("summary", null, `📚 ${docs ? "Your documents / Smriti" : "Smriti"}: ${hits.length} fact${hits.length > 1 ? "s" : ""} used`));
  const ul = el("ul", "hits");
  hits.forEach((h) => ul.append(hitItem(h)));
  d.append(ul);
  return d;
}

function setBusy(on) {
  $("#send").hidden = on; $("#stop").hidden = !on;
  $("#input").placeholder = on ? "Tantra is writing…" : "संदेश लिखें… / Type a message";
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
        history: settings.history, category: $("#category").value, smriti: settings.smriti && !!status.smriti,
        knowledge_first: settings.knowledge_first, mode: $("#mode").value, force_model: settings.force_model,
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
  reply.info = final?.skill === "no_model" ? "" : final?.skill ? "instant · no model needed" : parts.join(" · ");
  if (final?.skill) { reply.skill = final.skill; reply.card = final.card || {}; }
  if (final?.sources?.length) reply.sources = final.sources;
  if (final?.skill === "reminder" || final?.skill === "memory") refreshReminders();
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
        delete chatsCache[c.id];
        if (c.id === chat.id) newChat();
        loadChatList();
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

// ── speech-to-text: the browser converts the recording to 16 kHz mono WAV (no ffmpeg needed) ──
async function toWav(blob) {
  const raw = await blob.arrayBuffer();
  const ctx = new AudioContext();
  const decoded = await ctx.decodeAudioData(raw);
  ctx.close();
  const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
  const src = off.createBufferSource(); src.buffer = decoded; src.connect(off.destination); src.start();
  const pcm = (await off.startRendering()).getChannelData(0);
  const buf = new ArrayBuffer(44 + pcm.length * 2), v = new DataView(buf);
  const w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); v.setUint32(4, 36 + pcm.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, 16000, true);
  v.setUint32(28, 32000, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, "data"); v.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, pcm[i])) * 0x7fff, true);
  return new Blob([buf], { type: "audio/wav" });
}
async function transcribe(blob) {
  const fd = new FormData(); fd.append("audio", await toWav(blob), "speech.wav");
  const key = store.get("apiKey", "");
  const r = await fetch("/api/stt", { method: "POST", body: fd, headers: key ? { "X-API-Key": key } : {} });
  const j = await r.json();
  if (!r.ok) throw new Error(j.detail || "Speech-to-text failed");
  return (j.text || "").trim();
}

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
    toast("Transcribing… (the first time also loads the speech model)");
    try {
      const text = await transcribe(new Blob(parts, { type: recorder?.mimeType || "audio/webm" }));
      if (!text) { toast("I didn't catch that — try again a little louder.", "error"); return; }
      $("#input").value = ($("#input").value + " " + text).trim(); autosize(); $("#input").focus();
    } catch (e) { toast(e.message, "error"); }
  };
  recorder.start(); $("#mic").classList.add("recording"); toast("Recording… click the mic again to stop.");
};

// Speaking: Kokoro (offline, natural) when installed, else the voices built into Windows/the browser.
let audio = null;
const isHindi = (t) => /[ऀ-ॿ]/.test(t);
function browserVoice(text) {
  const voices = speechSynthesis.getVoices();
  const chosen = store.get("voiceName", "");
  const want = isHindi(text) ? "hi" : "en";
  return voices.find((v) => v.name === chosen && v.lang.toLowerCase().startsWith(want))
    || voices.find((v) => v.lang.toLowerCase().startsWith(want + "-in"))
    || voices.find((v) => v.lang.toLowerCase().startsWith(want));
}
function stopSpeaking() { audio?.pause(); audio = null; if ("speechSynthesis" in window) speechSynthesis.cancel(); }
function speakText(text) {
  const clean = String(text).replace(/```[\s\S]*?```/g, " (code) ").replace(/[*_`#>]/g, "").slice(0, 1200);
  stopSpeaking();
  return new Promise(async (resolve) => {
    if (status.speech?.tts && status.speech?.tts_ready) {
      try {
        const r = await fetch("/api/tts", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: clean }) });
        if (!r.ok) throw new Error((await r.json()).detail);
        audio = new Audio(URL.createObjectURL(await r.blob()));
        audio.onended = audio.onerror = () => resolve();
        await audio.play();
        return;
      } catch { /* fall back below */ }
    }
    if (!("speechSynthesis" in window)) { toast("No voice available. Run: pip install kokoro soundfile", "error"); resolve(); return; }
    const u = new SpeechSynthesisUtterance(clean);
    const v = browserVoice(clean);
    if (v) u.voice = v;
    u.lang = v?.lang || (isHindi(clean) ? "hi-IN" : "en-IN");
    u.rate = 1; u.onend = u.onerror = () => resolve();
    speechSynthesis.speak(u);
  });
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
  try { s = await getJSON("/api/status"); $("#conn").className = "conn ok"; $("#conn-text").textContent = "Online"; }
  catch { $("#conn").className = "conn bad"; $("#conn-text").textContent = "Offline"; $("#model-badge").textContent = "start: tantra.bat → 3"; schedule(5000); return; }
  status = s;
  const m = s.model || {}, t = s.training || {}, jobs = s.jobs || {};
  const busyJob = t.status === "running" || Object.values(jobs).some((j) => j.running);

  $("#model-badge").textContent = m.checkpoint
    ? `${m.params_M}M · step ${fmt(m.step, 0)}${m.int8 ? " · int8" : ""}`
    : s.checkpoints?.length ? "model loads on first message" : "no trained model yet";
  $("#chip-model").textContent = m.checkpoint ? `${m.params_M}M · step ${fmt(m.step, 0)}${m.int8 ? " · int8" : ""}` : "model not loaded";
  $("#chip-smriti").hidden = !(s.smriti && settings.smriti);
  $("#train-dot").hidden = t.status !== "running";
  const sel = $("#category");
  (m.categories || []).forEach((c) => { if (![...sel.options].some((o) => o.value === c)) sel.append(new Option(c, c)); });
  sel.value = [...sel.options].some((o) => o.value === settings.category) ? settings.category : "auto";
  const sp = s.speech || {};
  $("#voice-ready").textContent = !sp.stt ? "Speech input is not installed — Settings → System health → Fix."
    : sp.loading?.length ? `Getting ready: loading ${sp.loading.join(", ")} (first time downloads it — about a minute)…`
    : `Ready · hearing: Whisper${sp.tts_ready ? " · voice: Tantra (Kokoro)" : " · voice: this computer"}`;
  $("#voice-ready").className = `hint ${sp.loading?.length ? "warn" : sp.stt ? "ok" : "bad"}`;
  $("#mic").title = s.speech?.stt ? "Speak (Whisper)" : "Speech-to-text not installed (pip install openai-whisper)";
  $("#mic").style.opacity = s.speech?.stt ? 1 : 0.45;
  qualityBanner(s);

  const tab = activeTab();
  if (tab === "home") renderHome(s);
  if (tab === "training") renderTraining(s);
  if (tab === "model") renderModel(s);
  schedule(document.hidden ? 30000 : (busyJob && tab !== "chat") || (tab === "voice" && sp.loading?.length) ? 2500 : 10000);
}
document.addEventListener("visibilitychange", () => { if (!document.hidden) refreshStatus(); });

function qualityBanner(s) {
  const b = $("#quality-banner"), m = s.model || {}, probe = (s.probe || []).filter((p) => p.hits != null).at(-1);
  let msg = "";
  if (!s.checkpoints?.length) msg = "No trained model yet — skills, memory, reminders and knowledge answers work already; free-form replies start after the first training checkpoint.";
  else if (s.load_error) msg = `⚠ ${esc(s.load_error)}`;
  else if (m.checkpoint) {
    const vl = m.val?.loss;
    if ((vl != null && vl > 4.5) || (probe && probe.hits < 5) || (m.step || 0) < 2000)
      msg = `The model is still early in training (step ${fmt(m.step, 0)}${vl != null ? `, val loss ${fmt(vl, 2)}` : ""}${probe ? `, remembered ${probe.hits}/50` : ""}): ` +
        `its own replies are mostly random for now. Skills, memory and knowledge answers are exact already.`;
  }
  if (CFG.early_model_note === false) msg = "";
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
    dsSel.append(new Option("auto — cleaned file for the stage (pretrain.jsonl / sft.jsonl)", ""));
    names.forEach((n) => dsSel.append(new Option(`${n} (${fmt(s.datasets.find((d) => d.name === n).size_mb, 0)} MB)`, n)));
    dsSel.value = "";
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
  const LABELS = { eval: ["▶ Run test", "Test"], export: ["⇩ Export tantra.pt", "Export"],
    smriti: ["⟳ Build", "Smriti build"], data: ["⟳ Clean & mix data", "Data preparation"] };
  for (const [name, [label, title]] of Object.entries(LABELS)) {
    const j = jobs[name] || {};
    const btn = $(`#${name}-start`);
    btn.disabled = !!j.running;
    btn.textContent = j.running ? "running…" : label;
    if (j.started) refreshLog(name, $(`#${name}-log`), "force");
    if (j.running === false && j.exit_code != null && btn.dataset.wasRunning === "1")
      toast(j.exit_code === 0 ? `${title} finished.` : `${title} failed — see its log.`, j.exit_code === 0 ? "ok" : "error");
    btn.dataset.wasRunning = j.running ? "1" : "0";
  }

  const sm = s.smriti;
  dl("#smriti-info", sm ? [
    ["Facts", fmt(sm.facts, 0)], ["Q&A / text", `${fmt(sm.kinds?.qa || 0, 0)} / ${fmt(sm.kinds?.text || 0, 0)}`],
    ["Size on disk", `${fmt(sm.size_mb, 1)} MB`], ["Built", ago(sm.built_at)],
    ["Sources", Object.keys(sm.sources || {}).join(", ")],
  ] : [["Status", "Not built yet — press Build (needs the cleaned data first)."]]);
  $("#smriti-search").hidden = !sm;

  const rep = s.data_report;
  const COLORS = { hindi: "#c2410c", english: "#2563eb", mixed: "#16a34a" };
  if (rep) {
    const mix = {};
    for (const [k, v] of Object.entries(rep.language_mix_percent_of_chars || {})) { const lang = k.split("/")[1]; mix[lang] = (mix[lang] || 0) + v; }
    const drops = Object.entries(rep.cleaning || {}).filter(([k]) => k.startsWith("dropped") || k.startsWith("fixed"));
    $("#data-report").innerHTML =
      `<dl><dt>Pretrain rows</dt><dd>${fmt(rep.rows?.pretrain, 0)}</dd><dt>Conversation rows</dt><dd>${fmt(rep.rows?.sft, 0)}</dd>` +
      `<dt>Held out</dt><dd>${fmt(rep.val_rows?.pretrain, 0)} + ${fmt(rep.val_rows?.sft, 0)}</dd></dl>` +
      `<div class="bar">${Object.entries(mix).map(([k, v]) => `<span style="width:${v}%;background:${COLORS[k] || "#999"}" title="${k} ${v.toFixed(1)}%"></span>`).join("")}</div>` +
      `<div class="legend">${Object.entries(mix).map(([k, v]) => `<span><i style="background:${COLORS[k] || "#999"}"></i>${k === "mixed" ? "Hinglish / mixed" : k} ${v.toFixed(1)}%</span>`).join("")}</div>` +
      (drops.length ? `<p class="hint">${drops.map(([k, v]) => `${esc(k)}: ${fmt(v, 0)}`).join(" · ")}</p>` : "");
  } else $("#data-report").innerHTML = `<p class="hint">Not prepared yet.</p>`;

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
$("#smriti-start").onclick = async () => {
  try { await postJSON("/api/jobs/smriti"); toast("Building the knowledge store — takes a few minutes."); refreshStatus(); } catch (e) { toast(e.message, "error"); }
};
$("#data-start").onclick = async () => {
  if (!confirm("Rebuild pretrain.jsonl / sft.jsonl from the source data? (~10 minutes)")) return;
  try { await postJSON("/api/jobs/data"); toast("Preparing data…"); refreshStatus(); } catch (e) { toast(e.message, "error"); }
};
$("#smriti-search").onsubmit = async (e) => {
  e.preventDefault();
  const q = $("#smriti-q").value.trim(), ul = $("#smriti-hits");
  if (!q) return;
  ul.innerHTML = "";
  try {
    const j = await getJSON(`/api/smriti/search?q=${encodeURIComponent(q)}&k=5`);
    if (!j.hits.length) ul.append(el("li", "muted", "Nothing found."));
    j.hits.forEach((h) => ul.append(hitItem(h)));
  } catch (err) { toast(err.message, "error"); }
};
$("#reload-model").onclick = async () => {
  const first = status.checkpoints?.[0];
  if (!first) { toast("No checkpoints yet.", "error"); return; }
  try { await postJSON("/api/checkpoints/switch", { path: first, int8: $("#load-int8").checked }); toast("Model loaded.", "ok"); } catch (e) { toast(e.message, "error"); }
  refreshStatus();
};
$("#load-int8").onchange = () => refreshStatus();
window.addEventListener("resize", () => { if (activeTab() === "training") renderTraining(status); });

// ── home ─────────────────────────────────────────────────────────────────────
function useSkill(example) {
  showTab("chat");
  if (example.endsWith(" ") || example.endsWith(": ")) { $("#input").value = example; $("#input").focus(); autosize(); }
  else send(example);
}
function renderSkills() {
  $("#home-skills").innerHTML = ""; $("#skill-chips").innerHTML = "";
  (CFG.skill_examples || []).forEach(([name, ex]) => {
    const b = el("button"); b.type = "button"; b.append(el("b", null, name), el("span", null, `“${ex.trim()}”`));
    b.onclick = () => useSkill(ex); $("#home-skills").append(b);
    const c = el("button", "chip", name); c.type = "button"; c.onclick = () => useSkill(ex); $("#skill-chips").append(c);
  });
}
$("#home-ask").onsubmit = (e) => { e.preventDefault(); const q = $("#home-q").value.trim(); if (!q) return; $("#home-q").value = ""; showTab("chat"); newChat(); send(q); };

function greeting() {
  const h = new Date().getHours();
  return h < 5 ? "शुभ रात्रि" : h < 12 ? "सुप्रभात" : h < 17 ? "नमस्ते" : h < 21 ? "शुभ संध्या" : "शुभ रात्रि";
}
async function renderHome(s) {
  const now = new Date(), m = s.model || {}, t = s.training || {}, sm = s.smriti;
  $("#home-date").textContent = now.toLocaleDateString("hi-IN", { weekday: "long", day: "numeric", month: "long" }) + " · " +
    now.toLocaleTimeString("hi-IN", { hour: "numeric", minute: "2-digit" });
  $("#home-greet").textContent = `${greeting()} — मैं ${CFG.name} हूँ`;
  const pct = t.target_steps ? Math.min(100, (100 * (t.step || 0)) / t.target_steps) : 0;
  $("#home-sub").textContent = [t.status === "running" ? `Training ${pct.toFixed(0)}% done` : null,
    sm ? `Smriti knows ${fmtTokens(sm.facts)} facts` : null, "Ask me anything."].filter(Boolean).join(". ") + "";
  const probe = (s.probe || []).filter((p) => p.hits != null).at(-1);
  const h = s.hardware || {};
  $("#home-tiles").innerHTML = [
    tile("Model", m.params_M ? `${m.params_M}M` : "67M", m.checkpoint ? `step ${fmt(m.step, 0)} · val ${fmt(m.val?.loss, 2)}` : `training · step ${fmt(t.step, 0)}`),
    `<div class="tile"><div class="k">Remembered</div><div class="v accent">${probe ? probe.hits : "—"} / 50</div><div class="s">fixed test questions</div></div>`,
    tile("Smriti", sm ? `${fmtTokens(sm.facts)} facts` : "not built", sm ? `${fmt(sm.size_mb / 1024, 1)} GB on disk` : "Model tab → Build"),
    tile("This computer", h.device ? h.device.toUpperCase() : "—", h.cpu ? `${h.cpu.replace(/ with .*/, "")} · ${h.ram_gb} GB` : ""),
  ].join("");
  $("#home-train-state").innerHTML = t.status ? `<span class="${STATUS_CLASS[t.status] || ""}">${esc(t.status)}</span>${t.status === "running" ? ` · ETA ${esc(t.eta || "—")}` : ""} · step ${fmt(t.step, 0)} / ${fmt(t.target_steps, 0)}` : "not started";
  $("#home-progress").style.width = `${pct}%`;
  lineChart($("#home-chart"), [
    { name: "train loss", color: "var(--accent)", points: (t.history?.train || []).map((p) => ({ x: p.step, y: p.loss })) },
    { name: "val loss", color: "#2563eb", dots: true, points: (t.history?.val || []).map((p) => ({ x: p.step, y: p.loss })) },
  ], { leftLabel: "loss", empty: "The loss curve appears once training has run 10 steps." });
  const ul = $("#home-recent"); ul.innerHTML = "";
  Object.values(chatsCache).sort((a, b) => b.updated - a.updated).slice(0, 5).forEach((c) => {
    const li = el("li"); const b = el("button", "link", c.title); b.onclick = () => { chat = { id: c.id, title: c.title, messages: c.messages }; showTab("chat"); renderMessages(); renderChatList(); };
    li.append(b, el("span", "muted", ago(c.updated))); ul.append(li);
  });
  if (!ul.children.length) ul.append(el("li", "muted", "No chats yet."));
  renderUpcoming();
  if (!$("#home-brief").dataset.loaded) {
    $("#home-brief").dataset.loaded = "1";
    getJSON("/api/brief?lang=hi").then((j) => { $("#home-brief").textContent = j.text; }).catch(() => {});
  }
}
$("#brief-btn").onclick = async () => {
  const j = await getJSON("/api/brief?lang=hi").catch(() => null);
  if (j) { $("#home-brief").textContent = j.text; speakText(j.text); }
};

// ── reminders: checked every 15 s; due ones pop up, notify and speak ────────
let upcoming = [];
function renderUpcoming() {
  const ul = $("#home-reminders"); if (!ul) return; ul.innerHTML = "";
  upcoming.slice(0, 5).forEach((r) => {
    const li = el("li"); li.append(el("span", null, r.text), el("span", "muted", new Date(r.due * 1000).toLocaleString("hi-IN", { day: "numeric", month: "short", hour: "numeric", minute: "2-digit" })));
    ul.append(li);
  });
  if (!ul.children.length) ul.append(el("li", "muted", "No reminders. Say “10 मिनट बाद याद दिलाना …”."));
}
async function refreshReminders() {
  let j; try { j = await getJSON("/api/reminders/due"); } catch { return; }
  upcoming = j.upcoming || [];
  renderUpcoming();
  for (const r of j.due || []) {
    toast(`⏰ ${r.text}`, "ok");
    speakText(isHindi(r.text) ? `याद दिला रहा हूँ: ${r.text}` : `Reminder: ${r.text}`);
    if ("Notification" in window && Notification.permission === "granted") new Notification("Tantra reminder", { body: r.text, icon: "/assets/tantra_logo.jpg" });
  }
  if (j.due?.length && activeTab() === "memory") loadMemory();
}
setInterval(refreshReminders, 15000);
document.addEventListener("click", () => { if ("Notification" in window && Notification.permission === "default") Notification.requestPermission(); }, { once: true });

// ── memory page ──────────────────────────────────────────────────────────────
let memData = { memories: [], reminders: [], taught: [] }, memFilter = "all";
const CAT_NAMES = { "about me": "About me", family: "Family", preferences: "Preferences", other: "Other" };
async function loadMemory() {
  try { memData = await getJSON("/api/memory"); } catch (e) { toast(e.message, "error"); return; }
  renderMemory();
  getJSON("/api/assistant/settings").then((st) => {
    $("#folders").value = (st.file_folders || []).join("\n");
    $("#apps-list").textContent = (st.apps || []).join(", ");
  }).catch(() => {});
}
function renderMemory() {
  const counts = { all: memData.memories.length };
  memData.memories.forEach((m) => { counts[m.category] = (counts[m.category] || 0) + 1; });
  const f = $("#mem-filters"); f.innerHTML = "";
  Object.entries({ all: "All", ...CAT_NAMES }).forEach(([k, name]) => {
    if (k !== "all" && !counts[k]) return;
    const b = el("button", `chip${memFilter === k ? " on" : ""}`, `${name} · ${counts[k] || 0}`); b.type = "button";
    b.onclick = () => { memFilter = k; renderMemory(); }; f.append(b);
  });
  const q = $("#mem-search").value.trim().toLowerCase();
  const grid = $("#mem-grid"); grid.innerHTML = "";
  memData.memories.filter((m) => (memFilter === "all" || m.category === memFilter) && (!q || m.text.toLowerCase().includes(q))).forEach((m) => {
    const c = el("article", "mem-card");
    c.append(el("span", "cat", CAT_NAMES[m.category] || m.category), el("span", "t", m.text),
      el("span", "when", `${m.source === "manual" ? "Added" : "From chat"} · ${ago(m.created)}${m.used ? ` · used ${m.used}×` : ""}`));
    const a = el("div", "actions"), ed = el("button", null, "Edit"), fo = el("button", null, "Forget");
    ed.onclick = async () => { const t = prompt("Edit memory", m.text); if (t && t.trim()) { await api(`/api/memory/${m.id}`, { method: "PATCH", body: JSON.stringify({ text: t.trim() }) }); loadMemory(); } };
    fo.onclick = async () => { if (confirm(`Forget “${m.text}”?`)) { await api(`/api/memory/${m.id}`, { method: "DELETE" }); loadMemory(); } };
    a.append(ed, fo); c.append(a); grid.append(c);
  });
  if (!grid.children.length) grid.append(el("p", "hint", memData.memories.length ? "No memories match." : "Nothing yet. In chat, say “याद रखो: …” or add one above."));

  const rl = $("#rem-list"); rl.innerHTML = "";
  memData.reminders.slice().reverse().forEach((r) => {
    const li = el("li"); const when = new Date(r.due * 1000).toLocaleString("hi-IN", { day: "numeric", month: "short", hour: "numeric", minute: "2-digit" });
    li.append(el("span", r.done ? "muted" : null, `${r.done ? "✓ " : ""}${r.text} — ${when}`));
    const a = el("div", "actions");
    if (!r.done) {
      const d = el("button", null, "Done"), s = el("button", null, "Snooze 10m");
      d.onclick = async () => { await postJSON(`/api/reminders/${r.id}/done`); loadMemory(); refreshReminders(); };
      s.onclick = async () => { await postJSON(`/api/reminders/${r.id}/snooze`, { minutes: 10 }); loadMemory(); refreshReminders(); };
      a.append(d, s);
    }
    const x = el("button", null, "✕"); x.title = "Delete"; x.onclick = async () => { await api(`/api/memory/${r.id}`, { method: "DELETE" }); loadMemory(); refreshReminders(); };
    a.append(x); li.append(a); rl.append(li);
  });
  if (!rl.children.length) rl.append(el("li", "muted", "No reminders."));
  const tl = $("#taught-list"); tl.innerHTML = "";
  memData.taught.forEach((t) => {
    const li = el("li"); li.append(el("span", null, `${t.question} → ${t.answer.slice(0, 80)}`));
    const x = el("button", null, "✕"); x.onclick = async () => { await api(`/api/memory/${t.id}`, { method: "DELETE" }); loadMemory(); };
    const a = el("div", "actions"); a.append(x); li.append(a); tl.append(li);
  });
  if (!tl.children.length) tl.append(el("li", "muted", "Nothing taught yet."));
}
$("#mem-search").oninput = renderMemory;
$("#mem-add").onsubmit = async (e) => {
  e.preventDefault(); const t = $("#mem-text").value.trim(); if (!t) return;
  try { await postJSON("/api/memory", { text: t }); $("#mem-text").value = ""; loadMemory(); } catch (err) { toast(err.message, "error"); }
};
$("#folders-form").onsubmit = async (e) => {
  e.preventDefault();
  const folders = $("#folders").value.split("\n").map((s) => s.trim()).filter(Boolean);
  try { const r = await postJSON("/api/assistant/settings", { file_folders: folders }); $("#folders").value = r.file_folders.join("\n"); toast(`Saved ${r.file_folders.length} folder(s).`, "ok"); }
  catch (err) { toast(err.message, "error"); }
};

// ── documents page ───────────────────────────────────────────────────────────
async function loadDocs() {
  const ul = $("#doc-list"); ul.innerHTML = "";
  let j; try { j = await getJSON("/api/docs"); } catch (e) { toast(e.message, "error"); return; }
  j.docs.forEach((d) => {
    const li = el("li"); li.append(el("span", null, d.name), el("span", "muted", `${fmt(d.chars / 1000, 0)}k characters · ${d.pieces} pieces · ${ago(d.added)}`));
    const x = el("button", null, "Remove"); x.onclick = async () => { if (confirm(`Remove ${d.name}?`)) { await api(`/api/docs/${encodeURIComponent(d.name)}`, { method: "DELETE" }); loadDocs(); } };
    const a = el("div", "actions"); a.append(x); li.append(a); ul.append(li);
  });
  if (!ul.children.length) ul.append(el("li", "muted", "No documents yet."));
}
async function uploadDocs(files) {
  for (const f of files) {
    const fd = new FormData(); fd.append("file", f, f.name);
    toast(`Adding ${f.name}…`);
    try {
      const r = await fetch("/api/docs", { method: "POST", body: fd, headers: store.get("apiKey", "") ? { "X-API-Key": store.get("apiKey", "") } : {} });
      const j = await r.json(); if (!r.ok) throw new Error(j.detail);
      toast(`${f.name}: ${j.pieces} pieces added. Ask about it in chat.`, "ok");
    } catch (e) { toast(`${f.name}: ${e.message}`, "error"); }
  }
  loadDocs();
}
$("#doc-file").onchange = (e) => uploadDocs(e.target.files);
const dz = $("#dropzone");
dz.ondragover = (e) => { e.preventDefault(); dz.classList.add("over"); };
dz.ondragleave = () => dz.classList.remove("over");
dz.ondrop = (e) => { e.preventDefault(); dz.classList.remove("over"); uploadDocs(e.dataTransfer.files); };

// ── voice mode: listen → (wake word) → answer → speak → listen again ────────
const V = { on: false, stream: null, ctx: null, rec: null, busy: false, raf: 0 };
function voiceState(s, text) {
  $("#orb").className = `orb ${s}`;
  $("#voice-state").textContent = { idle: "Tap to start", listening: "Listening…", thinking: "Thinking…", speaking: "Speaking…" }[s] || s;
  if (text != null) $("#voice-live").textContent = text;
}
function voiceLog(you, tantra) {
  const box = $("#voice-log"); box.innerHTML = "";
  const t = new Date().toLocaleTimeString("hi-IN", { hour: "numeric", minute: "2-digit" });
  const a = el("div"); a.append(el("small", null, `You · ${t}`), el("span", null, you));
  const b = el("div"); b.append(el("small", null, "Tantra · spoken"), el("span", null, tantra));
  box.append(a, b);
}
async function voiceStart() {
  if (status.speech && !status.speech.stt) { toast("Speech-to-text needs Whisper: pip install openai-whisper (and ffmpeg)", "error"); return; }
  try { V.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } }); }
  catch { toast("Microphone not available or permission denied.", "error"); return; }
  V.on = true; V.ctx = new AudioContext();
  const src = V.ctx.createMediaStreamSource(V.stream), an = V.ctx.createAnalyser(); an.fftSize = 1024; src.connect(an);
  const buf = new Uint8Array(an.fftSize);
  let speaking = false, silentSince = 0, parts = [], noise = 0, calib = 0, startedAt = 0;
  const bars = $$("#orb .bars i");
  const loop = () => {
    if (!V.on) return;
    an.getByteTimeDomainData(buf);
    let sum = 0; for (const x of buf) sum += (x - 128) ** 2;
    const level = Math.sqrt(sum / buf.length);
    if (!V.busy) bars.forEach((b, i) => { b.style.height = `${16 + Math.min(56, level * (2 + (i % 3)))}px`; });
    const now = performance.now();
    if (calib < 40) { noise = Math.max(noise, level); calib++; V.raf = requestAnimationFrame(loop); return; }   // ~0.7 s: learn room noise
    const threshold = Math.max(CFG.voice?.min_threshold || 4, noise * 2 + 2);
    if (!V.busy) {
      if (speaking && now - startedAt > (CFG.voice?.max_seconds || 15) * 1000) { speaking = false; V.rec?.state === "recording" && V.rec.stop(); }   // max 15 s
      if (level > threshold) {
        silentSince = 0;
        if (!speaking) {
          speaking = true; parts = []; startedAt = now;
          V.rec = new MediaRecorder(V.stream); V.rec.ondataavailable = (e) => parts.push(e.data);
          V.rec.onstop = () => handleUtterance(new Blob(parts, { type: V.rec?.mimeType || "audio/webm" }));
          V.rec.start(); voiceState("listening", "…");
        }
      } else if (speaking) {
        silentSince ||= now;
        if (now - silentSince > (CFG.voice?.silence_ms || 1100)) { speaking = false; V.rec?.state === "recording" && V.rec.stop(); }
      }
    }
    V.raf = requestAnimationFrame(loop);
  };
  voiceState("listening", $("#wake").checked ? "Say “तन्त्र …” to wake me" : "बोलिए — I'm listening");
  loop();
}
function voiceStop() {
  if (!V.on) return;
  V.on = false; cancelAnimationFrame(V.raf);
  try { V.rec?.state === "recording" && V.rec.stop(); } catch { /* ignore */ }
  V.stream?.getTracks().forEach((t) => t.stop()); V.ctx?.close();
  stopSpeaking(); V.busy = false; voiceState("idle", "");
}
const escRe = (x) => x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const wakeRe = () => new RegExp(`^(\\s*(hey|ok|अरे|हे)?\\s*)(${(CFG.wake_words || ["tantra"]).map(escRe).join("|")})[\\s,!.।]*`, "i");
const stopRe = () => new RegExp(`^(${(CFG.stop_words || ["stop"]).map(escRe).join("|")})[\\s.!।]*$`, "i");
async function handleUtterance(blob) {
  if (!V.on || blob.size < 2000) { if (V.on) voiceState("listening"); return; }
  V.busy = true; voiceState("thinking", "सुन रहा हूँ… / transcribing…");
  try {
    let text = await transcribe(blob);
    if (!text) { V.busy = false; voiceState("listening", ""); return; }
    if (stopRe().test(text)) { stopSpeaking(); V.busy = false; voiceState("listening", "ठीक है।"); return; }
    if ($("#wake").checked) {
      if (!wakeRe().test(text)) { V.busy = false; voiceState("listening", "Say “तन्त्र …” to wake me"); return; }
      text = text.replace(wakeRe(), "").trim() || "नमस्ते";
    }
    voiceState("thinking", `“${text}”`);
    const res = await postJSON("/v1/chat/completions", {
      messages: [{ role: "user", content: text }], max_tokens: 160, temperature: settings.temperature,
      smriti: settings.smriti && !!status.smriti, knowledge_first: true, history: 0,
    });
    const answer = res.choices?.[0]?.message?.content || "…";
    voiceLog(text, answer);
    voiceState("speaking", answer.length > 220 ? answer.slice(0, 220) + "…" : answer);
    await speakText(answer);
    if (res.skill === "reminder" || res.skill === "memory") refreshReminders();
  } catch (e) { toast(e.message, "error"); $("#voice-live").textContent = `⚠ ${e.message}`; }
  V.busy = false;
  if (V.on) voiceState("listening", "");
}
$("#orb").onclick = () => (V.on ? voiceStop() : voiceStart());
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && activeTab() === "voice" && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) {
    e.preventDefault(); if (V.on && V.busy) { stopSpeaking(); } else if (!V.on) voiceStart();
  }
});
function fillVoices() {
  if (!("speechSynthesis" in window)) return;
  const sel = $("#voice-out"), vs = speechSynthesis.getVoices().filter((v) => /^(hi|en)/i.test(v.lang));
  sel.innerHTML = ""; sel.append(new Option(status.speech?.tts ? "Kokoro (offline)" : "Automatic", ""));
  vs.forEach((v) => sel.append(new Option(`${v.name} (${v.lang})`, v.name)));
  sel.value = store.get("voiceName", "");
}
if ("speechSynthesis" in window) speechSynthesis.onvoiceschanged = fillVoices;
$("#voice-out").onchange = (e) => store.set("voiceName", e.target.value);
$("#wake").checked = store.get("wake", false);
$("#wake").onchange = (e) => store.set("wake", e.target.checked);

// ── phone access card (Model tab) ────────────────────────────────────────────
async function renderAccess() {
  try {
    const a = await getJSON("/api/access");
    if (a.lan_enabled && a.lan_url) {
      $("#access-info").innerHTML = `<dl><dt>Open on phone</dt><dd><b>${esc(a.lan_url)}</b></dd><dt>Key</dt><dd><code>${esc(a.api_key || "")}</code></dd></dl>` +
        `<p class="hint">Same Wi-Fi only. The phone asks for the key once and remembers it.</p>`;
    }
  } catch { /* shown only on this computer */ }
}

// ── settings (Model/assistant.json) + system health ──────────────────────────
async function loadConfig() {
  try { CFG = (await getJSON("/api/config")).config; } catch { return; }
  $(".brand-text b").textContent = CFG.name_hi; $(".brand-text small").textContent = CFG.tagline;
  $(".mark").textContent = (CFG.name_hi || "त")[0];
  document.title = CFG.name;
  renderSkills();
  if (!chat.messages.length) renderMessages();
}
function fillSettingsForm() {
  const set = (id, v) => { const e = $(id); if (e) { if (e.type === "checkbox") e.checked = !!v; else e.value = v ?? ""; } };
  ["name", "name_hi", "tagline", "language", "region", "early_model_note", "auto_repair"].forEach((k) => set(`#cfg-${k}`, CFG[k]));
  set("#cfg-wake_words", (CFG.wake_words || []).join(", ")); set("#cfg-stop_words", (CFG.stop_words || []).join(", "));
  ["silence_ms", "max_seconds", "whisper_model"].forEach((k) => set(`#cfg-voice-${k}`, CFG.voice?.[k]));
  $("#cfg-json").value = JSON.stringify(CFG, null, 2);
}
async function saveConfig(changes, msg = "Settings saved.") {
  try { CFG = (await postJSON("/api/config", changes)).config; toast(msg, "ok"); await loadConfig(); fillSettingsForm(); }
  catch (e) { toast(e.message, "error"); }
}
$("#cfg-identity").onsubmit = (e) => { e.preventDefault(); saveConfig({ name: $("#cfg-name").value.trim() || "Tantra", name_hi: $("#cfg-name_hi").value.trim() || "तन्त्र",
  tagline: $("#cfg-tagline").value.trim(), language: $("#cfg-language").value, region: $("#cfg-region").value, early_model_note: $("#cfg-early_model_note").checked }); };
$("#cfg-voice").onsubmit = (e) => {
  e.preventDefault();
  const list = (id) => $(id).value.split(",").map((x) => x.trim()).filter(Boolean);
  saveConfig({ wake_words: list("#cfg-wake_words"), stop_words: list("#cfg-stop_words"),
    voice: { silence_ms: +$("#cfg-voice-silence_ms").value, max_seconds: +$("#cfg-voice-max_seconds").value, whisper_model: $("#cfg-voice-whisper_model").value } });
};
$("#cfg-auto_repair").onchange = (e) => saveConfig({ auto_repair: e.target.checked });
$("#cfg-json-save").onclick = () => {
  let obj; try { obj = JSON.parse($("#cfg-json").value); } catch (e) { toast(`Not valid JSON: ${e.message}`, "error"); return; }
  saveConfig(obj);
};
$("#cfg-json-reset").onclick = async () => {
  if (!confirm("Reset ALL settings to defaults?")) return;
  try { CFG = (await postJSON("/api/config/reset", {})).config; await loadConfig(); fillSettingsForm(); toast("Defaults restored.", "ok"); } catch (e) { toast(e.message, "error"); }
};

async function renderDoctor() {
  let d; try { d = await getJSON("/api/doctor"); } catch (e) { toast(e.message, "error"); return; }
  const ul = $("#doctor-list"); ul.innerHTML = "";
  d.checks.forEach((c) => {
    const li = el("li");
    const left = el("span"); left.innerHTML = `<b class="${c.ok ? "ok" : "bad"}">${c.ok ? "✓" : "✗"}</b> <b>${esc(c.title)}</b> <span class="muted">— ${esc(c.why)}</span>`;
    li.append(left);
    if (!c.ok) {
      const a = el("div", "actions");
      if (/^install|^run:/.test(c.fix)) {
        const b = el("button", "primary", "Fix"); b.title = c.fix; b.disabled = !!d.job.running;
        b.onclick = async () => { try { await postJSON("/api/doctor/fix", { id: c.id }); toast(`Fixing: ${c.title}…`); watchDoctor(); } catch (e) { toast(e.message, "error"); } };
        a.append(b);
      } else a.append(el("span", "hint", c.fix));
      li.append(a);
    }
    ul.append(li);
  });
  $("#doctor-fix-all").disabled = !!d.job.running || d.checks.every((c) => c.ok || !/^install|^run:/.test(c.fix));
  if (d.job.started) refreshLog("doctor", $("#doctor-log"), "force");
  return d;
}
let doctorTimer = 0;
function watchDoctor() {
  clearInterval(doctorTimer);
  doctorTimer = setInterval(async () => {
    const d = await renderDoctor();
    if (d && !d.job.running) { clearInterval(doctorTimer); status = {}; refreshStatus(); toast("Repair finished — see System health.", "ok"); }
  }, 2500);
}
$("#doctor-refresh").onclick = renderDoctor;
$("#doctor-fix-all").onclick = async () => { try { await postJSON("/api/doctor/fix", {}); toast("Repairing everything that can be repaired…"); watchDoctor(); } catch (e) { toast(e.message, "error"); } };
$$('.rail > button[data-tab="settings"]').forEach((b) => b.addEventListener("click", () => { fillSettingsForm(); renderDoctor(); }));

// ── start ──
loadConfig().then(() => { if (activeTab() === "settings") { fillSettingsForm(); renderDoctor(); } });
renderMessages();
loadChatList().then(() => refreshStatus());
refreshReminders();
fillVoices();
renderAccess();
showTab(location.hash.slice(1) || "home");
