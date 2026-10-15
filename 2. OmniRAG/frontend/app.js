/* OmniRAG front-end — vanilla JS, no build step. */
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

const STAGES = [
  { id: "upload",     name: "Upload",     w: 12, icon: '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>' },
  { id: "parse",      name: "Parse",      w: 18, icon: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h4"/>' },
  { id: "transcribe", name: "Transcribe", w: 25, icon: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>' },
  { id: "vision",     name: "Vision",     w: 22, icon: '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>' },
  { id: "embed",      name: "Embed",      w: 18, icon: '<circle cx="6" cy="6" r="2"/><circle cx="18" cy="8" r="2"/><circle cx="9" cy="18" r="2"/><circle cx="17" cy="17" r="2"/><path d="M8 7l8 1M7 8l2 8M11 18l4-1M18 10l-1 5"/>' },
  { id: "index",      name: "Index",      w: 5,  icon: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>' },
];
const MOD_ICON = {
  document: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h4"/>',
  image: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="M21 16l-5-5-9 9"/>',
  audio: '<path d="M9 18V6l11-2v12"/><circle cx="6" cy="18" r="3"/><circle cx="17" cy="16" r="3"/>',
  video: '<rect x="2" y="5" width="14" height="14" rx="2"/><path d="M16 10l6-3v10l-6-3z"/>',
};
const KIND_LABEL = { text: "Passage", transcript: "Transcript", image: "Image", figure: "Figure", frame: "Frame" };
const EXT = {
  document: ["pdf", "docx", "pptx", "txt", "md", "markdown", "rst", "csv", "tsv", "json", "log"],
  image: ["png", "jpg", "jpeg", "webp", "gif", "bmp", "tif", "tiff"],
  audio: ["mp3", "wav", "m4a", "ogg", "flac", "aac", "opus"],
  video: ["mp4", "mov", "mkv", "webm", "avi", "m4v"],
};
const BOT_ICON = '<svg viewBox="0 0 24 24"><rect x="3.5" y="3.5" width="7" height="7" rx="1.5"/><circle cx="17" cy="7" r="3.5"/><path d="M3.5 20.5l3.5-6 3.5 6z"/><path d="M14.5 14v6.5l5.5-3.25z"/></svg>';
const PLAY = '<span class="play"><svg viewBox="0 0 24 24"><path d="M6 4l14 8-14 8z"/></svg></span>';

const state = {
  kbs: [],
  kb: null,            // full meta of the open collection
  health: null,
  chats: loadStore("omnirag.chats", {}),
  controller: null,    // AbortController for the streaming answer
  queue: [],           // files waiting to be uploaded
  whisper: "base",
  addingTo: null,      // collection id when adding files to an existing collection
  attach: null,        // { send: dataURL, thumb: dataURL } image attached to the next question
};

/* ───────────── helpers ───────────── */
function loadStore(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
}
function saveStore(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable or full */ }
}
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n) => Number(n || 0).toLocaleString();
const icon = (m, cls = "") => `<svg class="${cls}" viewBox="0 0 24 24">${MOD_ICON[m] || MOD_ICON.document}</svg>`;
const raw = (path) => `/api/kbs/${state.kb.id}/raw/${path}`;
function clock(sec) {
  sec = Math.max(0, Math.floor(sec || 0));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return (h ? `${h}:${String(m).padStart(2, "0")}` : String(m).padStart(2, "0")) + ":" + String(s).padStart(2, "0");
}
function bytes(n) {
  if (n < 1e3) return n + " B";
  if (n < 1e6) return (n / 1e3).toFixed(0) + " KB";
  if (n < 1e9) return (n / 1e6).toFixed(1) + " MB";
  return (n / 1e9).toFixed(2) + " GB";
}
function modalityOf(name) {
  const ext = name.split(".").pop().toLowerCase();
  return Object.keys(EXT).find((m) => EXT[m].includes(ext)) || null;
}
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), 3200);
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
function countsLine(c = {}) {
  const parts = [`${fmt(c.files)} file${c.files === 1 ? "" : "s"}`, `${fmt(c.chunks)} chunks`];
  if (c.visuals) parts.push(`${fmt(c.visuals)} visuals`);
  if (c.media_seconds) parts.push(`${clock(c.media_seconds)} media`);
  return parts.join(" · ");
}
/** Downscale an image file/dataURL to a JPEG dataURL. */
function shrink(src, max) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const k = Math.min(1, max / Math.max(img.width, img.height));
      const c = document.createElement("canvas");
      c.width = Math.round(img.width * k); c.height = Math.round(img.height * k);
      const ctx = c.getContext("2d");
      ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, c.width, c.height);
      ctx.drawImage(img, 0, 0, c.width, c.height);
      resolve(c.toDataURL("image/jpeg", 0.88));
    };
    img.onerror = () => reject(new Error("That file isn't a readable image"));
    img.src = src;
  });
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
function setStatus(sel, ok, text, title) {
  const row = $(sel);
  $(".dot", row).className = "dot " + (ok ? "ok" : "bad");
  $(".status-text", row).textContent = text;
  row.title = title || "";
}

async function loadHealth() {
  try {
    state.health = await api("/api/health");
  } catch {
    state.health = { ollama: { online: false, models: [], vision_models: [] }, ocr: false };
  }
  const h = state.health, o = h.ollama;
  setStatus("#ollamaStatus", o.online && o.models.length,
    !o.online ? "Ollama offline" : o.models.length ? `Ollama · ${o.models.length} model${o.models.length > 1 ? "s" : ""}` : "Ollama · no models pulled",
    !o.online ? "Start it with: ollama serve" : o.models.join(", ") || "Run: ollama pull llama3.2");
  setStatus("#visionStatus", o.vision_models?.length,
    o.vision_models?.length ? `Vision · ${o.default_vision}` : "Vision · none (CLIP + OCR only)",
    o.vision_models?.length ? o.vision_models.join(", ") : "Optional: ollama pull moondream (or llava, qwen2.5vl)");
  setStatus("#ocrStatus", h.ocr, h.ocr ? "OCR · tesseract" : "OCR · not installed", h.ocr ? "" : "brew install tesseract");
  $("#embedStatus").textContent = h.embed_model ? `${h.embed_model.split("/").pop()} · ${h.clip_model} · whisper` : "";
  $("#captionHint").textContent = o.vision_models?.length
    ? `Uses ${o.default_vision}. Slower, but adds rich descriptions.`
    : "No vision model found. Run ollama pull moondream to enable.";
  $("#ocrHint").textContent = h.ocr ? "Screenshots, slides, scanned pages." : "Install tesseract to enable (brew install tesseract).";
  renderModelSelect();
}

function renderModelSelect() {
  const sel = $("#modelSelect");
  const o = state.health?.ollama;
  if (!o || !o.online || !o.models.length) {
    sel.innerHTML = `<option value="">${o?.online ? "No models — run ollama pull" : "Ollama offline · passages only"}</option>`;
    return;
  }
  const saved = loadStore("omnirag.model", null);
  const current = o.models.includes(saved) ? saved : o.default;
  sel.innerHTML = o.models.map((m) => `<option value="${esc(m)}" ${m === current ? "selected" : ""}>${esc(m)}${o.vision_models.includes(m) ? " · sees images" : ""}</option>`).join("");
}

async function loadKbs() {
  try { state.kbs = await api("/api/kbs"); } catch { state.kbs = []; }
  renderKbList();
}

function renderKbList() {
  const list = $("#kbList");
  if (!state.kbs.length) {
    list.innerHTML = `<div class="kb-empty">No collections yet.<br>Drop some files to build your first one.</div>`;
    return;
  }
  list.innerHTML = state.kbs.map((kb) => `
    <div class="kb-item ${state.kb?.id === kb.id ? "active" : ""}" data-id="${esc(kb.id)}">
      <div class="kb-avatar">${esc(kb.name[0] || "?")}</div>
      <div class="kb-meta">
        <div class="kb-name">${esc(kb.name)}</div>
        <div class="kb-sub">${fmt(kb.counts?.files)} files · ${fmt(kb.counts?.chunks)} chunks${kb.counts?.visuals ? ` · ${fmt(kb.counts.visuals)} visuals` : ""}</div>
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
    if (!confirm(`Delete the collection “${kb?.name}”? This removes its files and index from disk.`)) return;
    await api(`/api/kbs/${del.dataset.del}`, { method: "DELETE" });
    delete state.chats[del.dataset.del]; saveStore("omnirag.chats", state.chats);
    if (state.kb?.id === del.dataset.del) showIngest();
    await loadKbs();
    toast("Collection deleted");
    return;
  }
  const item = e.target.closest(".kb-item");
  if (item) openKb(item.dataset.id);
});

/* ───────────── views ───────────── */
function showIngest(addTo = null) {
  stopStreaming();
  state.addingTo = addTo;
  if (!addTo) state.kb = null;
  $("#chatView").classList.add("hidden");
  $("#ingestView").classList.remove("hidden");
  $("#sidebar").classList.remove("open");
  $("#addingBanner").classList.toggle("hidden", !addTo);
  if (addTo) $("#addingName").textContent = state.kb?.name || "";
  $("#nameInput").classList.toggle("hidden", !!addTo);
  $("#ingestBtnText").textContent = addTo ? "Add to collection" : "Build collection";
  if (!state.building) {
    $("#pipelineWrap").classList.add("hidden");
    $("#howItWorks").classList.remove("hidden");
  }
  renderKbList();
}

async function openKb(id) {
  stopStreaming();
  try {
    state.kb = await api(`/api/kbs/${id}`);
  } catch (err) { toast(err.message); return; }
  state.addingTo = null;
  saveStore("omnirag.lastKb", id);
  $("#ingestView").classList.add("hidden");
  $("#chatView").classList.remove("hidden");
  $("#sidebar").classList.remove("open");
  const kb = state.kb;
  $("#chatAvatar").textContent = kb.name[0] || "?";
  $("#chatTitle").textContent = kb.name;
  const by = kb.counts?.by_modality || {};
  $("#chatSub").textContent = ["document", "image", "audio", "video"].filter((m) => by[m])
    .map((m) => `${by[m]} ${m}${by[m] > 1 ? "s" : ""}`).join(" · ");
  $("#chatPills").textContent = countsLine(kb.counts);
  renderFiles();
  renderKbList();
  renderMessages();
  $("#chatInput").focus();
}

/* ───────────── upload queue ───────────── */
function addToQueue(fileList) {
  let rejected = 0;
  for (const f of fileList) {
    if (!modalityOf(f.name)) { rejected++; continue; }
    if (state.queue.some((q) => q.name === f.name && q.size === f.size)) continue;
    state.queue.push(f);
  }
  if (rejected) toast(`${rejected} file${rejected > 1 ? "s" : ""} skipped (unsupported type)`);
  renderQueue();
}

function renderQueue() {
  const q = state.queue;
  $("#queue").classList.toggle("hidden", !q.length);
  $("#queueList").innerHTML = q.map((f, i) => {
    const m = modalityOf(f.name);
    return `<div class="q-row"><span class="q-ico">${icon(m)}</span><span class="q-name">${esc(f.name)}</span>
      <span class="q-size">${bytes(f.size)}</span>
      <button type="button" class="icon-btn" data-rm="${i}" title="Remove"><svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg></button></div>`;
  }).join("");
  const by = {};
  q.forEach((f) => { const m = modalityOf(f.name); by[m] = (by[m] || 0) + 1; });
  $("#queueSum").textContent = `${q.length} file${q.length === 1 ? "" : "s"} · ${bytes(q.reduce((a, f) => a + f.size, 0))}`;
  if (!$("#nameInput").value && q.length) $("#nameInput").placeholder = q[0].name.replace(/\.[^.]+$/, "") + (q.length > 1 ? ` + ${q.length - 1} more` : "");
}

$("#queueList").addEventListener("click", (e) => {
  const rm = e.target.closest("[data-rm]");
  if (rm) { state.queue.splice(+rm.dataset.rm, 1); renderQueue(); }
});
$("#fileInput").addEventListener("change", (e) => { addToQueue(e.target.files); e.target.value = ""; });
const dz = $("#dropzone");
["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", (e) => addToQueue(e.dataTransfer.files));
// Dropping files anywhere on the ingest view works too.
$("#ingestView").addEventListener("dragover", (e) => e.preventDefault());
$("#ingestView").addEventListener("drop", (e) => { e.preventDefault(); if (e.dataTransfer.files.length) addToQueue(e.dataTransfer.files); });

$("#frameInterval").addEventListener("input", (e) => ($("#frameVal").textContent = e.target.value + "s"));
$("#whisperMode").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  $$("#whisperMode button").forEach((x) => x.classList.toggle("on", x === b));
  state.whisper = b.dataset.v;
});
$("#newKbBtn").addEventListener("click", () => showIngest());
$("#cancelAddBtn").addEventListener("click", () => (state.kb ? openKb(state.kb.id) : showIngest()));
$("#addFilesBtn").addEventListener("click", () => showIngest(state.kb.id));
$("#menuBtn").addEventListener("click", () => $("#sidebar").classList.toggle("open"));

/* ───────────── ingestion pipeline ───────────── */
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
  const glyph = { page: "↗", warn: "!", info: "i" }[kind] || "•";
  row.innerHTML = `<div class="feed-ico ${kind}">${glyph}</div><div class="feed-main">${html}</div>${tag || ""}`;
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

$("#ingestForm").addEventListener("submit", (e) => {
  e.preventDefault();
  if (!state.queue.length) { $("#fileInput").click(); return; }
  if (state.building) { toast("A build is already running"); return; }
  const files = [...state.queue];
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f, f.name));
  fd.append("kb_id", state.addingTo || "");
  fd.append("name", $("#nameInput").value.trim());
  fd.append("options", JSON.stringify({
    whisper: state.whisper,
    frame_interval: +$("#frameInterval").value,
    ocr: $("#ocrToggle").checked,
    captions: $("#captionToggle").checked,
  }));
  const label = state.addingTo ? `Adding ${files.length} file${files.length > 1 ? "s" : ""} to ${state.kb?.name}` :
    `${files.length} file${files.length > 1 ? "s" : ""} · ${bytes(files.reduce((a, f) => a + f.size, 0))}`;
  const ui = startPipelineUI(label);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/ingest");
  xhr.upload.onprogress = (ev) => {
    if (!ev.lengthComputable) return;
    ui.progress("upload", ev.loaded / ev.total);
    setStage("upload", "running", `${Math.round((ev.loaded / ev.total) * 100)}% · ${bytes(ev.loaded)}`);
  };
  xhr.onload = () => {
    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch {}
    if (xhr.status >= 300) { ui.fail(body.detail || xhr.statusText || "Upload failed"); return; }
    state.queue = []; $("#nameInput").value = ""; renderQueue();
    ui.listen(body.job_id);
  };
  xhr.onerror = () => ui.fail("Upload failed. Is the server running?");
  setStage("upload", "running", "Starting upload");
  xhr.send(fd);
});

function startPipelineUI(label) {
  state.building = true;
  $("#ingestBtn").disabled = true;
  $("#howItWorks").classList.add("hidden");
  $("#pipelineWrap").classList.remove("hidden");
  $("#pipeDone").classList.add("hidden");
  $("#pipeError").classList.add("hidden");
  $("#pipeTitle").textContent = state.addingTo ? "Adding files" : "Building collection";
  $("#pipeUrl").textContent = label;
  $("#feed").innerHTML = "";
  ["#stFiles", "#stChunks", "#stVisuals"].forEach((s) => ($(s).textContent = "0"));
  $("#stMedia").textContent = "00:00";
  buildPipeline();
  $("#pipelineWrap").scrollIntoView({ behavior: "smooth", block: "start" });

  const t0 = performance.now();
  const timer = setInterval(() => ($("#pipeTimer").textContent = ((performance.now() - t0) / 1000).toFixed(1) + "s"), 100);
  const frac = {};
  let media = 0, es = null;
  const progress = (stage, f) => {
    frac[stage] = f;
    const pct = STAGES.reduce((a, s) => a + s.w * (frac[s.id] || 0), 0);
    $("#progressBar").style.width = Math.min(100, pct) + "%";
  };
  const finish = () => { es?.close(); clearInterval(timer); state.building = false; $("#ingestBtn").disabled = false; };
  const fail = (msg) => {
    finish();
    $$(".stage.running").forEach((s) => s.classList.replace("running", "error"));
    $("#pipeTitle").textContent = "Build failed";
    $("#pipeError").textContent = msg;
    $("#pipeError").classList.remove("hidden");
  };

  const listen = (jobId) => {
    es = new EventSource(`/api/ingest/${jobId}/events`);
    es.addEventListener("stage", (e) => {
      const d = JSON.parse(e.data);
      setStage(d.stage, d.status, d.detail);
      if (d.status === "done" || d.status === "skipped") progress(d.stage, 1);
      else if (d.progress != null) progress(d.stage, d.progress);
    });
    es.addEventListener("file", (e) => {
      const d = JSON.parse(e.data);
      countUp($("#stFiles"), d.count);
      progress("parse", d.count / d.total);
      if (d.duration) { media += d.duration; $("#stMedia").textContent = clock(media); }
      feed(d.error ? "warn" : "page", `${esc(d.name)}<span class="u">${esc(d.detail)}</span>`,
        `<span class="feed-tag">${d.modality}</span>`);
    });
    es.addEventListener("log", (e) => {
      const d = JSON.parse(e.data);
      feed(d.level === "warn" ? "warn" : "info", renderMarkdown(d.message).replace(/^<p>|<\/p>$/g, ""));
    });
    es.addEventListener("stats", (e) => {
      const d = JSON.parse(e.data);
      countUp($("#stChunks"), d.text_chunks);
      countUp($("#stVisuals"), d.visuals);
    });
    es.addEventListener("complete", async (e) => {
      const { kb } = JSON.parse(e.data);
      finish();
      progress("index", 1);
      $("#pipeTitle").textContent = kb.added === kb.counts.files ? "Collection ready" : "Files added";
      $("#doneText").textContent = `Indexed ${fmt(kb.added)} file${kb.added > 1 ? "s" : ""} into ${fmt(kb.added_items)} searchable items in ${kb.duration_s}s`;
      $("#pipeDone").classList.remove("hidden");
      $("#pipeDone").scrollIntoView({ behavior: "smooth", block: "center" });
      $("#startChatBtn").onclick = () => openKb(kb.id);
      await loadKbs();
    });
    es.addEventListener("error", (e) => {
      if (!e.data) { if (es.readyState === EventSource.CLOSED) finish(); return; }
      fail(JSON.parse(e.data).message);
    });
  };
  return { progress, listen, fail };
}

/* ───────────── chat ───────────── */
function chatOf(id) { return (state.chats[id] ||= []); }
function persistChats() {
  // Keep storage small: last 30 messages per collection.
  for (const k of Object.keys(state.chats)) state.chats[k] = state.chats[k].slice(-30);
  saveStore("omnirag.chats", state.chats);
}

function suggestionsFor(kb) {
  const files = kb.files || [];
  const first = (m) => files.find((f) => f.modality === m && !f.error);
  const s = [["document", "Overview", "Summarize this collection. What are the key points in each file?"]];
  const vid = first("video"), aud = first("audio"), img = first("image"), doc = first("document");
  if (vid) s.push(["video", "Video", `What happens in “${vid.name}”? Give a timeline.`]);
  if (aud) s.push(["audio", "Audio", `What are the main points discussed in “${aud.name}”?`]);
  if (doc) s.push(["document", "Document", `What are the main findings in “${doc.name}”?`]);
  if (img || kb.counts?.visuals) s.push(["image", "Visuals", "Describe the images, charts or diagrams in this collection."]);
  s.push(["document", "Details", "What numbers, dates or names are mentioned most?"]);
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
        <p>${esc(countsLine(kb.counts))}. Answers cite the exact page, image or timestamp.</p>
        <div class="suggestions">
          ${suggestionsFor(kb).map(([m, h, q]) => `<button class="suggestion" data-q="${esc(q)}"><span class="mtag sm">${icon(m)} ${esc(h)}</span><br>${esc(q)}</button>`).join("")}
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
    el.innerHTML = `<div class="user-stack">${m.image ? `<img class="user-img" src="${m.image}" alt="Attached image" />` : ""}
      ${m.content ? `<div class="bubble-user">${esc(m.content)}</div>` : ""}</div>`;
  } else {
    el.innerHTML = `<div class="bot-avatar">${BOT_ICON}</div><div class="bot-content"></div>`;
    fillBot(el, m);
  }
  return el;
}

function sourceCard(s, k) {
  const media = (s.modality === "audio" || s.modality === "video") && s.start != null;
  let thumb;
  if (s.asset) {
    thumb = `<div class="src-thumb ${s.kind === "text" ? "doc" : ""}">
      <img class="src-thumb-img" src="${raw(s.asset)}" alt="" loading="lazy" />
      ${media ? `<span class="ts">${clock(s.start)}</span>${s.modality === "video" ? PLAY : ""}` : ""}</div>`;
  } else if (media) {
    thumb = `<div class="src-thumb audio">${icon("audio")}<span class="ts">${clock(s.start)}</span></div>`;
  } else thumb = "";
  const visual = ["image", "figure", "frame"].includes(s.kind);
  let body = "";
  if (s.kind === "text" || s.kind === "transcript") body = esc(s.text);
  else {
    if (s.caption) body += `<span class="lbl">Description</span>${esc(s.caption)}`;
    if (s.ocr) body += `<span class="lbl">Text in image</span>${esc(s.ocr)}`;
    if (!body) body = `<span class="muted">Matched visually. No text was found in this ${KIND_LABEL[s.kind].toLowerCase()}.</span>`;
  }
  const openHref = raw(s.src) + (s.page && s.src.endsWith(".pdf") ? `#page=${s.page}` : "");
  const via = (s.via || []).filter((v) => v !== "text").join(" + ");
  return `
    <div class="src" data-n="${k + 1}" data-i="${k}">
      ${thumb}
      <div class="src-info">
        <div class="src-top"><span class="src-num">${k + 1}</span><span class="src-title">${esc(s.file_name)}</span></div>
        <div class="src-meta">${icon(s.modality)} ${KIND_LABEL[s.kind] || s.kind}${s.loc ? " · " + esc(s.loc) : ""}${via ? `<span class="via">${esc(via)}</span>` : ""}</div>
        <div class="src-bar" title="Relevance ${(s.score * 100).toFixed(0)}%"><div style="width:${Math.max(8, s.score * 100)}%"></div></div>
        <div class="src-snippet">${body}</div>
        <div class="src-actions">
          ${media ? `<a href="#" data-play="${k}">Play from ${clock(s.start)}</a>` : ""}
          ${visual ? `<a href="#" data-view="${k}">View full size</a>` : ""}
          <a href="${openHref}" target="_blank" rel="noopener">Open file ↗</a>
          <span class="muted">relevance ${(s.score * 100).toFixed(0)}%</span>
        </div>
      </div>
    </div>`;
}

function fillBot(el, m) {
  const c = $(".bot-content", el);
  const pending = m.streaming && !m.content;
  let html = m.imageCaption ? `<div class="img-caption">Your image: ${esc(m.imageCaption)}</div>` : "";
  html += pending
    ? `<div class="thinking"><span class="dots"><span></span><span></span><span></span></span>${esc(m.status || (m.sources ? "Writing answer…" : "Searching your files…"))}</div>`
    : `<div class="answer ${m.streaming ? "cursor" : ""}">${renderMarkdown(m.content)}</div>`;
  if (m.error) html += `<div class="pipe-error">${esc(m.error)}</div>`;
  if (m.sources?.length) {
    html += `<div class="src-head">Sources${m.model ? `<span class="model-tag">${esc(m.model)}${m.vision ? " · saw images" : ""}</span>` : ""}</div>
      <div class="sources">${m.sources.map(sourceCard).join("")}</div>`;
  } else if (m.sources && !m.streaming) {
    html += `<div class="src-head">No matching items found</div>`;
  }
  c.innerHTML = html;
}

function openCard(card, play) {
  card.classList.add("open");
  const msg = state.chats[state.kb.id][+card.closest(".msg").dataset.i];
  const s = msg?.sources?.[+card.dataset.i];
  if (!s || s.start == null || !(s.modality === "audio" || s.modality === "video")) return;
  let player = $(".src-player", card);
  if (!player) {
    player = document.createElement(s.modality === "video" ? "video" : "audio");
    player.className = "src-player";
    player.controls = true;
    player.preload = "metadata";
    player.src = raw(s.src);
    player.addEventListener("loadedmetadata", () => { player.currentTime = s.start; }, { once: true });
    const thumb = $(".src-thumb", card);
    if (s.modality === "video" && thumb) thumb.replaceWith(player);
    else card.insertBefore(player, $(".src-info", card));
  } else {
    player.currentTime = s.start;
  }
  if (play) player.play().catch(() => {});
}

function lightbox(html, cap) {
  $("#lbBody").innerHTML = html;
  $("#lbCap").textContent = cap || "";
  $("#lightbox").classList.remove("hidden");
}
$("#lbClose").addEventListener("click", () => { $("#lightbox").classList.add("hidden"); $("#lbBody").innerHTML = ""; });
$("#lightbox").addEventListener("click", (e) => { if (e.target.id === "lightbox") $("#lbClose").click(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#lightbox").classList.contains("hidden")) $("#lbClose").click(); });

$("#messages").addEventListener("click", (e) => {
  const sug = e.target.closest(".suggestion");
  if (sug) { ask(sug.dataset.q); return; }
  const uimg = e.target.closest(".user-img");
  if (uimg) { lightbox(`<img src="${uimg.src}" alt="" />`, "Your image"); return; }
  const cite = e.target.closest(".cite");
  if (cite) {
    const card = $(`.src[data-n="${cite.dataset.n}"]`, cite.closest(".msg"));
    if (card) {
      openCard(card, false);
      card.classList.add("flash");
      card.scrollIntoView({ behavior: "smooth", block: "nearest" });
      setTimeout(() => card.classList.remove("flash"), 1200);
    }
    return;
  }
  const play = e.target.closest("[data-play]");
  if (play) { e.preventDefault(); openCard(play.closest(".src"), true); return; }
  const view = e.target.closest("[data-view]");
  if (view) {
    e.preventDefault();
    const s = state.chats[state.kb.id][+view.closest(".msg").dataset.i].sources[+view.dataset.view];
    lightbox(`<img src="${raw(s.asset)}" alt="" />`, `${s.file_name}${s.loc ? " · " + s.loc : ""}${s.caption ? " — " + s.caption : ""}`);
    return;
  }
  if (e.target.closest(".src-actions a") || e.target.closest(".src-player")) return;
  const src = e.target.closest(".src");
  if (src) {
    if (src.classList.contains("open")) src.classList.remove("open");
    else openCard(src, false);
  }
});

/* composer */
const input = $("#chatInput");
input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 180) + "px"; });
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("#chatForm").requestSubmit(); }
});
$("#chatForm").addEventListener("submit", (e) => {
  e.preventDefault();
  if (state.controller) { stopStreaming(); return; }
  const q = input.value.trim();
  if (q || state.attach) ask(q);
});
$("#modelSelect").addEventListener("change", (e) => saveStore("omnirag.model", e.target.value));
$("#clearChatBtn").addEventListener("click", () => {
  if (!state.kb) return;
  stopStreaming();
  state.chats[state.kb.id] = []; persistChats(); renderMessages();
});

async function attachImage(file) {
  if (!file || !file.type.startsWith("image/")) { toast("Attach an image file"); return; }
  const url = URL.createObjectURL(file);
  try {
    state.attach = { send: await shrink(url, 1024), thumb: await shrink(url, 320) };
    $("#attachImg").src = state.attach.thumb;
    $("#attachPreview").classList.remove("hidden");
    input.focus();
  } catch (err) { toast(err.message); }
  finally { URL.revokeObjectURL(url); }
}
function clearAttach() {
  state.attach = null;
  $("#attachPreview").classList.add("hidden");
  $("#attachImg").removeAttribute("src");
}
$("#attachBtn").addEventListener("click", () => $("#attachInput").click());
$("#attachInput").addEventListener("change", (e) => { attachImage(e.target.files[0]); e.target.value = ""; });
$("#attachRemove").addEventListener("click", clearAttach);
input.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) { e.preventDefault(); attachImage(item.getAsFile()); }
});
const cbox = $("#composerBox");
["dragenter", "dragover"].forEach((t) => cbox.addEventListener(t, (e) => { e.preventDefault(); cbox.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => cbox.addEventListener(t, (e) => { e.preventDefault(); cbox.classList.remove("over"); }));
cbox.addEventListener("drop", (e) => attachImage(e.dataTransfer.files[0]));

function stopStreaming() {
  if (state.controller) { state.controller.abort(); state.controller = null; }
  $("#sendBtn").classList.remove("stop");
}

async function ask(question) {
  const kb = state.kb;
  const msgs = chatOf(kb.id);
  const history = msgs.filter((m) => m.content && !m.error).map((m) => ({ role: m.role, content: m.content.replace(/<think>[\s\S]*?<\/think>/g, "") }));
  const attach = state.attach;
  msgs.push({ role: "user", content: question, image: attach?.thumb });
  const bot = { role: "assistant", content: "", streaming: true };
  msgs.push(bot);
  input.value = ""; input.style.height = "auto";
  clearAttach();
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
      body: JSON.stringify({ question, history, model: $("#modelSelect").value || null, image: attach?.send || null }),
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
        const chunk = buf.slice(0, idx); buf = buf.slice(idx + 2);
        const ev = (chunk.match(/^event: (.*)$/m) || [])[1];
        const data = JSON.parse((chunk.match(/^data: (.*)$/m) || [, "null"])[1]);
        if (ev === "sources") bot.sources = data;
        else if (ev === "meta") { bot.model = data.model; bot.vision = data.vision; }
        else if (ev === "status") bot.status = data.text;
        else if (ev === "image_caption") bot.imageCaption = data.text;
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
    delete bot.status;
    if (state.controller === controller) { state.controller = null; $("#sendBtn").classList.remove("stop"); }
    if (state.kb?.id === kb.id) fillBot(el, bot);
    persistChats();
  }
}

/* ───────────── files panel ───────────── */
$("#sourcesBtn").addEventListener("click", () => {
  $("#pagesPanel").classList.toggle("hidden");
  $("#sourcesBtn").classList.toggle("on", !$("#pagesPanel").classList.contains("hidden"));
});
$("#closePanelBtn").addEventListener("click", () => { $("#pagesPanel").classList.add("hidden"); $("#sourcesBtn").classList.remove("on"); });
$("#pageFilter").addEventListener("input", renderFiles);

function fileDetail(f) {
  const i = f.info || {};
  if (f.error) return "Couldn't be read: " + f.error;
  const bits = [];
  if (f.modality === "document") {
    if (i.pages) bits.push(`${i.pages} page${i.pages > 1 ? "s" : ""}`);
    if (i.slides) bits.push(`${i.slides} slide${i.slides > 1 ? "s" : ""}`);
    if (i.figures) bits.push(`${i.figures} figures`);
    if (i.scanned_pages) bits.push(`${i.scanned_pages} scanned`);
  } else if (f.modality === "image") {
    if (i.width) bits.push(`${i.width}×${i.height}`);
  } else {
    if (i.duration) bits.push(clock(i.duration));
    if (i.words) bits.push(`${fmt(i.words)} words`);
    if (i.frames_sampled) bits.push(`${i.frames_sampled} frames`);
  }
  bits.push(`${fmt(f.items)} items`, bytes(f.size));
  return bits.join(" · ");
}

function renderFiles() {
  const q = $("#pageFilter").value.toLowerCase();
  const files = (state.kb?.files || []).filter((f) => !q || f.name.toLowerCase().includes(q) || f.modality.includes(q));
  $("#pagesList").innerHTML = files.map((f) => `
    <a class="file-row ${f.error ? "err" : ""}" href="${raw("files/" + f.stored)}" target="_blank" rel="noopener">
      <span class="q-ico">${icon(f.modality)}</span>
      <span class="meta"><div class="t">${esc(f.name)}</div><div class="u">${esc(fileDetail(f))}</div></span>
    </a>`).join("") || `<div class="kb-empty">No matching files</div>`;
}

/* ───────────── boot ───────────── */
(async function init() {
  buildPipeline();
  await Promise.all([loadHealth(), loadKbs()]);
  const last = loadStore("omnirag.lastKb", null);
  if (last && state.kbs.some((k) => k.id === last)) openKb(last);
  setInterval(loadHealth, 20000);
})();
