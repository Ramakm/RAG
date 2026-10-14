/* BrowseAI front-end — vanilla JS, no build step. */
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

const STAGES = [
  { id: "crawl",   name: "Crawl",   icon: '<circle cx="12" cy="12" r="8"/><path d="M4 12h16M12 4c3.4 4 3.4 12 0 16M12 4c-3.4 4-3.4 12 0 16"/>' },
  { id: "extract", name: "Extract", icon: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h4"/>' },
  { id: "clean",   name: "Clean",   icon: '<path d="M3 21l7-7M14 4l6 6-8 8-6-6z"/><path d="M17 3l4 4"/>' },
  { id: "chunk",   name: "Chunk",   icon: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>' },
  { id: "embed",   name: "Embed",   icon: '<circle cx="6" cy="6" r="2"/><circle cx="18" cy="8" r="2"/><circle cx="9" cy="18" r="2"/><circle cx="17" cy="17" r="2"/><path d="M8 7l8 1M7 8l2 8M11 18l4-1M18 10l-1 5"/>' },
  { id: "index",   name: "Index",   icon: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>' },
];
const BOT_ICON = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/><path d="M4 12h16M12 4c3.4 4 3.4 12 0 16M12 4c-3.4 4-3.4 12 0 16"/></svg>';

const state = {
  kbs: [],
  kb: null,            // full meta of the open knowledge base
  health: null,
  chats: loadStore("browseai.chats", {}),
  controller: null,    // AbortController for the streaming answer
  jsMode: "auto",
};

/* ───────────── helpers ───────────── */
function loadStore(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
}
function saveStore(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
}
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n) => Number(n || 0).toLocaleString();
function shortUrl(u) { try { const x = new URL(u); return x.pathname === "/" ? x.host : x.pathname + x.search; } catch { return u; } }
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), 2800);
}
async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!res.ok) {
    let detail = res.statusText;
    try { const j = await res.json(); detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new Error(detail);
  }
  return res.json();
}

/* ───────────── tiny, safe markdown renderer ───────────── */
function renderMarkdown(src) {
  src = src.replace(/<think>[\s\S]*?(<\/think>|$)/g, "").trimStart();
  const code = [];
  src = src.replace(/```[\w+-]*\n?([\s\S]*?)(```|$)/g, (_, body) => { code.push(esc(body.replace(/\n$/, ""))); return `\u0000${code.length - 1}\u0000`; });
  const inline = (s) => s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")
    .replace(/\[(\d{1,2})\]/g, '<a class="cite" data-n="$1">$1</a>')
    .replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${code[i]}</code>`);

  let html = "", para = [], list = null;
  const flushPara = () => { if (para.length) { html += `<p>${inline(para.join("<br>"))}</p>`; para = []; } };
  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
  const openList = (tag) => { if (list !== tag) { closeList(); html += `<${tag}>`; list = tag; } };

  for (const line of esc(src).split("\n")) {
    let m;
    if ((m = line.match(/^\u0000(\d+)\u0000$/))) { flushPara(); closeList(); html += `<pre><code>${code[m[1]]}</code></pre>`; }
    else if ((m = line.match(/^#{1,6}\s+(.*)/))) { flushPara(); closeList(); html += `<h4>${inline(m[1])}</h4>`; }
    else if ((m = line.match(/^\s*[-*•]\s+(.*)/))) { flushPara(); openList("ul"); html += `<li>${inline(m[1])}</li>`; }
    else if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) { flushPara(); openList("ol"); html += `<li>${inline(m[1])}</li>`; }
    else if ((m = line.match(/^&gt;\s?(.*)/))) { flushPara(); closeList(); html += `<blockquote><p>${inline(m[1])}</p></blockquote>`; }
    else if (!line.trim()) { flushPara(); closeList(); }
    else { if (list) closeList(); para.push(line); }
  }
  flushPara(); closeList();
  return html.replace(/<\/blockquote><blockquote>/g, "");
}

/* ───────────── health & sidebar ───────────── */
async function loadHealth() {
  try {
    state.health = await api("/api/health");
  } catch {
    state.health = { ollama: { online: false, models: [] }, embed_model: "" };
  }
  const o = state.health.ollama;
  const row = $("#ollamaStatus");
  $(".dot", row).className = "dot " + (o.online && o.models.length ? "ok" : "bad");
  $(".status-text", row).textContent = !o.online ? "Ollama offline" : o.models.length ? `Ollama · ${o.models.length} model${o.models.length > 1 ? "s" : ""}` : "Ollama · no models pulled";
  row.title = !o.online ? "Start it with: ollama serve" : o.models.join(", ") || "Run: ollama pull llama3.2";
  $("#embedStatus").textContent = state.health.embed_model ? "Embeddings · " + state.health.embed_model.split("/").pop() : "";
  renderModelSelect();
}

function renderModelSelect() {
  const sel = $("#modelSelect");
  const o = state.health?.ollama;
  if (!o || !o.online || !o.models.length) {
    sel.innerHTML = `<option value="">${o?.online ? "No models — run ollama pull" : "Ollama offline · passages only"}</option>`;
    return;
  }
  const saved = loadStore("browseai.model", null);
  const current = o.models.includes(saved) ? saved : o.default;
  sel.innerHTML = o.models.map((m) => `<option value="${esc(m)}" ${m === current ? "selected" : ""}>${esc(m)}</option>`).join("");
}

async function loadKbs() {
  try { state.kbs = await api("/api/kbs"); } catch { state.kbs = []; }
  renderKbList();
}

function renderKbList() {
  const list = $("#kbList");
  if (!state.kbs.length) {
    list.innerHTML = `<div class="kb-empty">No knowledge bases yet.<br>Paste a URL to build your first one.</div>`;
    return;
  }
  list.innerHTML = state.kbs.map((kb) => `
    <div class="kb-item ${state.kb?.id === kb.id ? "active" : ""}" data-id="${esc(kb.id)}">
      <div class="kb-avatar">${esc(kb.name[0] || "?")}</div>
      <div class="kb-meta">
        <div class="kb-name">${esc(kb.name)}</div>
        <div class="kb-sub">${fmt(kb.page_count)} pages · ${fmt(kb.chunk_count)} chunks</div>
      </div>
      <button class="icon-btn kb-del" title="Delete" data-del="${esc(kb.id)}">
        <svg viewBox="0 0 24 24"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>
      </button>
    </div>`).join("");
}

$("#kbList").addEventListener("click", async (e) => {
  const del = e.target.closest("[data-del]");
  if (del) {
    e.stopPropagation();
    const kb = state.kbs.find((k) => k.id === del.dataset.del);
    if (!confirm(`Delete the knowledge base for ${kb?.name}? This removes its index from disk.`)) return;
    await api(`/api/kbs/${del.dataset.del}`, { method: "DELETE" });
    delete state.chats[del.dataset.del]; saveStore("browseai.chats", state.chats);
    if (state.kb?.id === del.dataset.del) showIngest();
    await loadKbs();
    toast("Knowledge base deleted");
    return;
  }
  const item = e.target.closest(".kb-item");
  if (item) openKb(item.dataset.id);
});

/* ───────────── views ───────────── */
function showIngest() {
  stopStreaming();
  state.kb = null;
  $("#chatView").classList.add("hidden");
  $("#ingestView").classList.remove("hidden");
  $("#sidebar").classList.remove("open");
  renderKbList();
  setTimeout(() => $("#urlInput").focus(), 50);
}

async function openKb(id) {
  stopStreaming();
  try {
    state.kb = await api(`/api/kbs/${id}`);
  } catch (err) { toast(err.message); return; }
  saveStore("browseai.lastKb", id);
  $("#ingestView").classList.add("hidden");
  $("#chatView").classList.remove("hidden");
  $("#sidebar").classList.remove("open");
  const kb = state.kb;
  $("#chatAvatar").textContent = kb.name[0] || "?";
  $("#chatTitle").textContent = kb.name;
  $("#chatUrl").textContent = kb.url; $("#chatUrl").href = kb.url;
  $("#chatPills").textContent = `${fmt(kb.page_count)} pages · ${fmt(kb.chunk_count)} chunks · ${fmt(kb.word_count)} words`;
  renderPages();
  renderKbList();
  renderMessages();
  $("#chatInput").focus();
}

/* ───────────── ingestion ───────────── */
$("#maxPages").addEventListener("input", (e) => ($("#maxPagesVal").textContent = e.target.value));
$("#maxDepth").addEventListener("input", (e) => ($("#maxDepthVal").textContent = e.target.value));
$("#jsMode").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  $$("#jsMode button").forEach((x) => x.classList.toggle("on", x === b));
  state.jsMode = b.dataset.v;
});
$$(".examples .chip").forEach((chip) => chip.addEventListener("click", () => {
  $("#urlInput").value = chip.dataset.url;
  $("#samePath").checked = !!chip.dataset.same;
  $("#urlInput").focus();
}));
$("#newKbBtn").addEventListener("click", showIngest);
$("#menuBtn").addEventListener("click", () => $("#sidebar").classList.toggle("open"));

function buildPipeline() {
  $("#pipeline").innerHTML = STAGES.map((s) => `
    <div class="stage pending" id="stage-${s.id}">
      <div class="stage-icon"><svg viewBox="0 0 24 24">${s.icon}</svg></div>
      <div class="stage-name">${s.name}</div>
      <div class="stage-detail">Waiting</div>
    </div>`).join("");
}

function setStage(id, status, detail) {
  const el = $(`#stage-${id}`); if (!el) return;
  el.className = `stage ${status}`;
  if (detail) $(".stage-detail", el).textContent = detail;
}

function feed(kind, html, tag) {
  const row = document.createElement("div");
  row.className = "feed-row";
  const icon = { page: "↗", warn: "!", info: "i" }[kind] || "•";
  row.innerHTML = `<div class="feed-ico ${kind}">${icon}</div><div class="feed-main">${html}</div>${tag || ""}`;
  const f = $("#feed");
  const atBottom = f.scrollHeight - f.scrollTop - f.clientHeight < 40;
  f.appendChild(row);
  if (atBottom) f.scrollTop = f.scrollHeight;
}

function countUp(el, to) {
  const from = parseInt(el.textContent.replace(/\D/g, "")) || 0;
  const start = performance.now();
  const step = (t) => {
    const p = Math.min(1, (t - start) / 500);
    el.textContent = fmt(Math.round(from + (to - from) * p));
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

$("#ingestForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const url = $("#urlInput").value.trim();
  if (!url) return;
  const body = {
    url,
    max_pages: +$("#maxPages").value,
    max_depth: +$("#maxDepth").value,
    js_mode: state.jsMode,
    same_path: $("#samePath").checked,
  };
  let job;
  try {
    $("#ingestBtn").disabled = true;
    job = await api("/api/ingest", { method: "POST", body: JSON.stringify(body) });
  } catch (err) {
    $("#ingestBtn").disabled = false;
    toast(err.message);
    return;
  }
  runPipelineUI(job, body);
});

function runPipelineUI(job, body) {
  $("#howItWorks").classList.add("hidden");
  $("#pipelineWrap").classList.remove("hidden");
  $("#pipeDone").classList.add("hidden");
  $("#pipeError").classList.add("hidden");
  $("#pipeTitle").textContent = "Building knowledge base";
  $("#pipeUrl").textContent = body.url;
  $("#feed").innerHTML = "";
  ["#stPages", "#stWords", "#stNoise", "#stChunks"].forEach((s) => ($(s).textContent = "0"));
  buildPipeline();
  $("#pipelineWrap").scrollIntoView({ behavior: "smooth", block: "start" });

  const t0 = performance.now();
  const timer = setInterval(() => ($("#pipeTimer").textContent = ((performance.now() - t0) / 1000).toFixed(1) + "s"), 100);
  let rawWords = 0, crawlFrac = 0, embedFrac = 0;
  const done = new Set();
  const setProgress = () => {
    const pct = 45 * (done.has("crawl") ? 1 : crawlFrac) + (done.has("clean") ? 5 : 0) + (done.has("chunk") ? 5 : 0)
      + 35 * (done.has("embed") ? 1 : embedFrac) + (done.has("index") ? 10 : 0);
    $("#progressBar").style.width = Math.min(100, pct) + "%";
  };
  setProgress();

  const es = new EventSource(`/api/ingest/${job.job_id}/events`);
  const finish = () => { es.close(); clearInterval(timer); $("#ingestBtn").disabled = false; };

  es.addEventListener("stage", (e) => {
    const d = JSON.parse(e.data);
    setStage(d.stage, d.status, d.detail);
    if (d.status === "done") done.add(d.stage);
    if (d.stage === "embed" && d.progress) embedFrac = d.progress;
    if (d.stage === "chunk" && d.status === "done") countUp($("#stChunks"), parseInt(d.detail) || 0);
    setProgress();
  });
  es.addEventListener("page", (e) => {
    const d = JSON.parse(e.data);
    rawWords += d.words;
    crawlFrac = d.count / d.max;
    countUp($("#stPages"), d.count);
    countUp($("#stWords"), rawWords);
    setStage("crawl", "running", `${d.count} / ${d.max} pages`);
    feed("page", `${esc(d.title)}<span class="u">${esc(shortUrl(d.url))}</span>`,
      `${d.rendered ? '<span class="feed-tag js">JS</span>' : ""}<span class="feed-tag">${fmt(d.words)} words</span>`);
    setProgress();
  });
  es.addEventListener("log", (e) => {
    const d = JSON.parse(e.data);
    feed(d.level === "warn" ? "warn" : "info", esc(d.message));
  });
  es.addEventListener("stats", (e) => {
    const d = JSON.parse(e.data);
    countUp($("#stNoise"), d.boilerplate_blocks + d.junk_blocks + d.duplicate_blocks);
    countUp($("#stWords"), d.words_after);
    const extra = [];
    if (d.duplicate_pages) extra.push(`${d.duplicate_pages} duplicate page(s)`);
    if (d.thin_pages) extra.push(`${d.thin_pages} near-empty page(s)`);
    feed("info", `Cleaning removed ${fmt(d.boilerplate_blocks)} repeated boilerplate block(s), ${fmt(d.junk_blocks)} cookie/footer line(s)` +
      (extra.length ? ` and skipped ${extra.join(", ")}` : ""));
  });
  es.addEventListener("complete", async (e) => {
    const { kb } = JSON.parse(e.data);
    finish();
    $("#pipeTitle").textContent = "Knowledge base ready";
    $("#doneText").textContent = `Indexed ${fmt(kb.page_count)} pages into ${fmt(kb.chunk_count)} searchable chunks in ${kb.duration_s}s`;
    $("#pipeDone").classList.remove("hidden");
    $("#pipeDone").scrollIntoView({ behavior: "smooth", block: "center" });
    $("#startChatBtn").onclick = () => openKb(kb.id);
    await loadKbs();
  });
  es.addEventListener("error", (e) => {
    if (!e.data) { // connection dropped
      if (es.readyState === EventSource.CLOSED) { finish(); }
      return;
    }
    const d = JSON.parse(e.data);
    finish();
    $$(".stage.running").forEach((s) => s.classList.replace("running", "error"));
    $("#pipeTitle").textContent = "Build failed";
    $("#pipeError").textContent = d.message;
    $("#pipeError").classList.remove("hidden");
  });
}

/* ───────────── chat ───────────── */
function chatOf(id) { return (state.chats[id] ||= []); }
function persistChats() {
  // Keep storage small: last 30 messages per knowledge base.
  for (const k of Object.keys(state.chats)) state.chats[k] = state.chats[k].slice(-30);
  saveStore("browseai.chats", state.chats);
}

function suggestionsFor(kb) {
  const pages = (kb.pages || []).filter((p) => p.title && p.title.length < 70 && p.url !== kb.url).slice(0, 2);
  const s = [
    ["Overview", `What is ${kb.name} about? Give me a short summary.`],
    ["Get started", "How do I get started? List the first steps."],
    ...pages.map((p) => ["Explore", `Explain the key points of “${p.title}”.`]),
  ];
  if (s.length < 4) s.push(["Highlights", "What are the most important features or topics covered?"]);
  return s.slice(0, 4);
}

function renderMessages() {
  const box = $("#messages");
  const kb = state.kb;
  const msgs = chatOf(kb.id);
  if (!msgs.length) {
    box.innerHTML = `
      <div class="empty">
        <div class="kb-avatar">${esc(kb.name[0] || "?")}</div>
        <h2>Ask ${esc(kb.name)} anything</h2>
        <p>${fmt(kb.page_count)} pages indexed. Answers cite the exact pages they came from.</p>
        <div class="suggestions">
          ${suggestionsFor(kb).map(([h, q]) => `<button class="suggestion" data-q="${esc(q)}"><b>${esc(h)}</b>${esc(q)}</button>`).join("")}
        </div>
      </div>`;
    return;
  }
  box.innerHTML = "";
  msgs.forEach((m, i) => box.appendChild(messageEl(m, i)));
  box.scrollTop = box.scrollHeight;
}

function messageEl(m, i) {
  const el = document.createElement("div");
  el.className = `msg ${m.role}`;
  el.dataset.i = i;
  if (m.role === "user") {
    el.innerHTML = `<div class="bubble-user">${esc(m.content)}</div>`;
  } else {
    el.innerHTML = `<div class="bot-avatar">${BOT_ICON}</div><div class="bot-content"></div>`;
    fillBot(el, m);
  }
  return el;
}

function fillBot(el, m) {
  const c = $(".bot-content", el);
  const pending = m.streaming && !m.content;
  let html = pending
    ? `<div class="thinking"><span class="dots"><span></span><span></span><span></span></span>${m.sources ? "Writing answer…" : "Searching the site…"}</div>`
    : `<div class="answer ${m.streaming ? "cursor" : ""}">${renderMarkdown(m.content)}</div>`;
  if (m.error) html += `<div class="pipe-error">${esc(m.error)}</div>`;
  if (m.sources?.length) {
    const max = Math.max(...m.sources.map((s) => s.score), 0.01);
    html += `<div class="src-head">Sources${m.model ? `<span class="model-tag">${esc(m.model)}</span>` : ""}</div>
      <div class="sources">${m.sources.map((s, k) => `
        <div class="src" data-n="${k + 1}">
          <div class="src-top"><span class="src-num">${k + 1}</span><span class="src-title">${esc(s.title)}</span></div>
          <div class="src-sec">${esc(s.section || shortUrl(s.url))}</div>
          <div class="src-bar" title="Relevance ${(s.similarity * 100).toFixed(0)}%"><div style="width:${Math.max(8, (s.score / max) * 100)}%"></div></div>
          <div class="src-snippet">${esc(s.text)}</div>
          <div class="src-actions"><a href="${esc(s.url)}" target="_blank" rel="noopener">Open page ↗</a><span class="muted">${esc(shortUrl(s.url))} · relevance ${(s.similarity * 100).toFixed(0)}%</span></div>
        </div>`).join("")}
      </div>`;
  }
  c.innerHTML = html;
}

$("#messages").addEventListener("click", (e) => {
  const sug = e.target.closest(".suggestion");
  if (sug) { ask(sug.dataset.q); return; }
  const cite = e.target.closest(".cite");
  if (cite) {
    const card = $(`.src[data-n="${cite.dataset.n}"]`, cite.closest(".msg"));
    if (card) {
      card.classList.add("open", "flash");
      card.scrollIntoView({ behavior: "smooth", block: "nearest" });
      setTimeout(() => card.classList.remove("flash"), 1200);
    }
    return;
  }
  if (e.target.closest(".src-actions a")) return;
  const src = e.target.closest(".src");
  if (src) src.classList.toggle("open");
});

const input = $("#chatInput");
input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 180) + "px"; });
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("#chatForm").requestSubmit(); }
});
$("#chatForm").addEventListener("submit", (e) => {
  e.preventDefault();
  if (state.controller) { stopStreaming(); return; }
  const q = input.value.trim();
  if (q) ask(q);
});
$("#modelSelect").addEventListener("change", (e) => saveStore("browseai.model", e.target.value));
$("#clearChatBtn").addEventListener("click", () => {
  if (!state.kb) return;
  stopStreaming();
  state.chats[state.kb.id] = []; persistChats(); renderMessages();
});

function stopStreaming() {
  if (state.controller) { state.controller.abort(); state.controller = null; }
  $("#sendBtn").classList.remove("stop");
}

async function ask(question) {
  const kb = state.kb;
  const msgs = chatOf(kb.id);
  const history = msgs.filter((m) => m.content && !m.error).map((m) => ({ role: m.role, content: m.content.replace(/<think>[\s\S]*?<\/think>/g, "") }));
  msgs.push({ role: "user", content: question });
  const bot = { role: "assistant", content: "", streaming: true };
  msgs.push(bot);
  input.value = ""; input.style.height = "auto";
  renderMessages();

  const box = $("#messages");
  const el = box.lastElementChild;
  const controller = new AbortController();
  state.controller = controller;
  $("#sendBtn").classList.add("stop");

  let scheduled = false;
  const repaint = () => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 120;
      fillBot(el, bot);
      if (nearBottom) box.scrollTop = box.scrollHeight;
    });
  };

  try {
    const res = await fetch(`/api/kbs/${kb.id}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history, model: $("#modelSelect").value || null }),
      signal: controller.signal,
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, idx); buf = buf.slice(idx + 2);
        const ev = (raw.match(/^event: (.*)$/m) || [])[1];
        const data = JSON.parse((raw.match(/^data: (.*)$/m) || [, "null"])[1]);
        if (ev === "sources") bot.sources = data;
        else if (ev === "meta") bot.model = data.model;
        else if (ev === "token") bot.content += data.t;
        else if (ev === "error") bot.error = data.message;
        repaint();
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") bot.error = err.message;
    else if (!bot.content) bot.content = "_Stopped._";
  } finally {
    bot.streaming = false;
    if (state.controller === controller) { state.controller = null; $("#sendBtn").classList.remove("stop"); }
    if (state.kb?.id === kb.id) fillBot(el, bot);
    persistChats();
  }
}

/* ───────────── pages panel ───────────── */
$("#sourcesBtn").addEventListener("click", () => {
  $("#pagesPanel").classList.toggle("hidden");
  $("#sourcesBtn").classList.toggle("on", !$("#pagesPanel").classList.contains("hidden"));
});
$("#closePanelBtn").addEventListener("click", () => { $("#pagesPanel").classList.add("hidden"); $("#sourcesBtn").classList.remove("on"); });
$("#pageFilter").addEventListener("input", renderPages);
function renderPages() {
  const q = $("#pageFilter").value.toLowerCase();
  const pages = (state.kb?.pages || []).filter((p) => !q || p.title.toLowerCase().includes(q) || p.url.toLowerCase().includes(q));
  $("#pagesList").innerHTML = pages.map((p) => `
    <a class="page-row" href="${esc(p.url)}" target="_blank" rel="noopener">
      <div class="t">${esc(p.title)}</div>
      <div class="u">${esc(shortUrl(p.url))} · ${fmt(p.words)} words</div>
    </a>`).join("") || `<div class="kb-empty">No matching pages</div>`;
}

/* ───────────── boot ───────────── */
(async function init() {
  buildPipeline();
  await Promise.all([loadHealth(), loadKbs()]);
  const last = loadStore("browseai.lastKb", null);
  if (last && state.kbs.some((k) => k.id === last)) openKb(last);
  else $("#urlInput").focus();
  setInterval(loadHealth, 20000);
})();
