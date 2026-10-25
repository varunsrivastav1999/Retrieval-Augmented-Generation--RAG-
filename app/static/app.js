/* Maintenance Assistant UI — vanilla JS, offline, CSP-safe (no inline code). */
(() => {
  "use strict";

  // ------------------------------------------------------------------ helpers
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  };
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  };
  const P = (d) => `<path d="${d}"/>`;
  const ICON_PATHS = {
    logo: P("M4 17l4-10 4 6 3-4 5 8") + P("M3 20h18"),
    plus: P("M12 5v14M5 12h14"), chat: P("M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"),
    folder: P("M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"),
    pulse: P("M3 12h4l3-7 4 14 3-7h4"), contrast: '<circle cx="12" cy="12" r="9"/>' + P("M12 3v18a9 9 0 0 0 0-18z"),
    menu: P("M4 6h16M4 12h16M4 18h16"), shield: P("M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z") + P("M9 12l2 2 4-4"),
    alarm: P("M12 4a7 7 0 0 1 7 7v5l2 2H3l2-2v-5a7 7 0 0 1 7-7z") + P("M10 21h4"),
    wrench: P("M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.6 2.6-2.4-.6-.6-2.4z"),
    gauge: P("M12 14l4-4") + P("M4 18a9 9 0 1 1 16 0"), history: P("M3 12a9 9 0 1 0 3-6.7L3 8") + P("M3 3v5h5M12 7v5l3 2"),
    quote: P("M7 7h4v4H7zM7 11c0 3-1 5-3 6M15 7h4v4h-4zM15 11c0 3-1 5-3 6"),
    spark: P("M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z") + P("M19 17l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7z"),
    filter: P("M4 5h16l-6 8v6l-4-2v-4z"), send: P("M4 12l16-8-6 16-2-7z"), stop: '<rect x="7" y="7" width="10" height="10" rx="1.5"/>',
    upload: P("M12 16V4M7 9l5-5 5 5") + P("M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"), refresh: P("M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7"),
    external: P("M14 4h6v6M20 4l-9 9") + P("M19 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h5"),
    close: P("M6 6l12 12M18 6L6 18"), copy: '<rect x="8" y="8" width="12" height="12" rx="2"/>' + P("M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"),
    check: P("M5 12l5 5 9-10"), warn: P("M12 4l9 16H3z") + P("M12 10v4M12 17h.01"), chev: P("M9 6l6 6-6 6"),
    clock: '<circle cx="12" cy="12" r="9"/>' + P("M12 7v5l3 2"), doc: P("M7 3h7l5 5v13H7z") + P("M14 3v5h5"),
    link: P("M10 14l4-4") + P("M8 16a4 4 0 0 1 0-5.7l2-2M16 8a4 4 0 0 1 0 5.7l-2 2"),
  };
  // Icons are constant markup defined above (never user data), so innerHTML is safe here.
  const icon = (name) => `<svg class="i" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICON_PATHS[name] || ""}</svg>`;
  const iconEl = (name, cls) => { const s = el("span", cls); s.innerHTML = icon(name); return s; };
  const withIcon = (node, name, text) => { node.innerHTML = icon(name); node.append(document.createTextNode(` ${text}`)); return node; };
  const hydrateIcons = (root = document) => $$("[data-icon]", root).forEach((n) => { n.insertAdjacentHTML("afterbegin", icon(n.dataset.icon)); n.removeAttribute("data-icon"); });

  const state = { conversationId: null, conversations: [], controller: null, docOffset: 0, docLimit: 25, view: "ask", timers: {}, mode: "evidence" };

  async function api(path, opts = {}) {
    const headers = Object.assign({}, opts.headers || {});
    const key = store.get("rag.apiKey", "");
    if (key) headers["X-API-Key"] = key;
    if (opts.json !== undefined) { headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(opts.json); }
    const res = await fetch(path, Object.assign({}, opts, { headers }));
    if (res.status === 401) {
      const entered = window.prompt("This system requires an API key:");
      if (entered) { store.set("rag.apiKey", entered.trim()); return api(path, opts); }
      throw new Error("API key required");
    }
    if (!res.ok) {
      let detail = res.statusText;
      try { const j = await res.json(); detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* not json */ }
      throw new Error(`${res.status}: ${detail}`);
    }
    return opts.raw ? res : res.json();
  }

  function toast(msg, kind = "info", ms = 4500) {
    const t = el("div", `toast ${kind}`, msg);
    $("#toasts").append(t);
    setTimeout(() => t.remove(), ms);
  }

  const fmtMs = (ms) => ms == null ? "" : ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
  const fmtDate = (s) => s ? new Date(s.endsWith("Z") ? s : s + "Z").toLocaleString() : "";
  const fmtNum = (n) => (n ?? 0).toLocaleString();
  const fileUrl = (docId, page) => {
    const key = store.get("rag.apiKey", "");
    return `/api/v1/documents/${encodeURIComponent(docId)}/file${key ? `?key=${encodeURIComponent(key)}` : ""}${page ? `#page=${page}` : ""}`;
  };
  const docKind = (name) => {
    const ext = (name.split(".").pop() || "").toLowerCase();
    if (["png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif", "webp"].includes(ext)) return ["img", "IMG"];
    return [ext, ext.slice(0, 4).toUpperCase() || "DOC"];
  };
  const docBadge = (name) => { const [k, label] = docKind(name); return el("span", `doc-icon ${k}`, label); };
  const isPaged = (name) => /\.(pdf|pptx)$/i.test(name || "");
  const whereText = (q) => [q.section, q.content_type === "record" ? (q.page ? `row ${q.page}` : "")
    : (q.page && isPaged(q.document) ? `${/\.pptx$/i.test(q.document) ? "slide" : "page"} ${q.page}` : "")].filter(Boolean).join(" · ");

  async function copyText(text) {
    try { await navigator.clipboard.writeText(text); return true; } catch { /* insecure context (http on the plant LAN) */ }
    const ta = el("textarea"); ta.value = text; ta.setAttribute("readonly", "");
    ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.append(ta); ta.select();
    let ok = false; try { ok = document.execCommand("copy"); } catch { ok = false; }
    ta.remove(); return ok;
  }

  // ------------------------------------------------------------------ markdown & highlighting
  function renderMarkdown(target, text, sources) {
    const html = window.marked ? window.marked.parse(text || "", { breaks: true, gfm: true }) : null;
    if (html !== null && window.DOMPurify) target.innerHTML = window.DOMPurify.sanitize(html, { FORBID_TAGS: ["style", "img", "iframe", "form"], FORBID_ATTR: ["style"] });
    else target.textContent = text || "";
    linkCitations(target, sources);
  }

  function linkCitations(root, sources) {
    if (!sources || !sources.length) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, { acceptNode: (n) => n.parentElement.closest("code,pre") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT });
    const nodes = [];
    while (walker.nextNode()) if (/\[\d{1,2}\]/.test(walker.currentNode.nodeValue)) nodes.push(walker.currentNode);
    for (const node of nodes) {
      const frag = document.createDocumentFragment();
      for (const part of node.nodeValue.split(/(\[\d{1,2}\])/)) {
        const m = part.match(/^\[(\d{1,2})\]$/);
        const src = m && sources.find((s) => s.n === Number(m[1]));
        if (src) { const b = el("button", "cite", m[1]); b.type = "button"; b.title = `${src.document}${src.page ? ", page " + src.page : ""}`; b.dataset.n = m[1]; frag.append(b); }
        else frag.append(document.createTextNode(part));
      }
      node.replaceWith(frag);
    }
  }

  function highlight(target, text, ids) {
    target.textContent = "";
    if (!ids || !ids.length) { target.textContent = text; return; }
    const re = new RegExp(`(${ids.map((i) => i.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "gi");
    for (const part of text.split(re)) {
      if (!part) continue;
      if (ids.some((i) => i.toLowerCase() === part.toLowerCase())) target.append(el("mark", null, part));
      else target.append(document.createTextNode(part));
    }
  }

  // ------------------------------------------------------------------ message rendering
  function hideEmpty() { const e = $("#empty-state"); if (e) e.hidden = true; }

  function addUserMessage(text) {
    hideEmpty();
    const m = el("div", "msg-user", text);
    $("#messages").append(m);
    m.scrollIntoView({ block: "end" });
  }

  let messageCounter = 0;
  function createBotMessage() {
    hideEmpty();
    const box = el("article", "msg-bot");
    box.dataset.k = String(++messageCounter);
    const head = el("div", "msg-head");
    const status = el("span", "badge brand", "Searching documents…");
    const timing = el("span", "msg-timing");
    head.append(status, timing);
    const notice = el("div", "notice"); notice.hidden = true;
    const exact = el("div", "exact-list");
    exact.append(el("div", "skeleton w80"), el("div", "skeleton w60"), el("div", "skeleton w40"));
    const ai = el("div", "ai-block"); ai.hidden = true;
    const aiHead = el("div", "ai-head");
    aiHead.append(withIcon(el("span", "badge brand"), "spark", "AI summary"),
      el("span", "ai-note", "May paraphrase — the exact document text above is authoritative."));
    const aiBody = el("div", "answer");
    const verify = el("div", "verify"); verify.hidden = true;
    ai.append(aiHead, aiBody, verify);
    const related = el("div", "related"); related.hidden = true;
    const sources = el("details", "sources"); sources.hidden = true;
    const sum = el("summary"); sum.append(iconEl("chev", "chev"), el("span", null, "Evidence"));
    sources.append(sum, el("div", "source-list"));
    box.append(head, notice, exact, ai, related, sources);
    $("#messages").append(box);
    box.scrollIntoView({ block: "end" });
    return { box, status, timing, notice, exact, ai, aiBody, verify, related, sources, srcData: [] };
  }

  function setStatus(view, r) {
    const map = {
      answered: ["ok", "check", r.mode_used === "llm" ? "Exact answer + AI summary" : "Exact answer from documents"],
      partial: ["warn", "warn", "Closest match — please check"],
      not_found: ["warn", "warn", "Not in the documents"],
      error: ["bad", "warn", "Error"],
    };
    const [cls, ic, label] = map[r.status] || ["", "check", r.status];
    view.status.className = `badge ${cls}`;
    withIcon(view.status, ic, label);
  }

  function renderExact(view, quotes, ids) {
    view.exact.textContent = "";
    for (const q of quotes || []) {
      const card = el("div", "exact-card");
      const head = el("div", "exact-head");
      const titles = el("div");
      titles.append(el("div", "exact-doc", q.document), el("div", "exact-where", whereText(q)));
      const actions = el("div", "exact-actions");
      if (q.document_id) {
        const pdf = /\.pdf$/i.test(q.document);
        const open = el("a", "btn btn-sm"); open.href = fileUrl(q.document_id, pdf ? q.page : null); open.target = "_blank"; open.rel = "noopener";
        withIcon(open, "external", pdf && q.page ? `Open page ${q.page}` : "Open original");
        actions.append(open);
      }
      const cp = withIcon(el("button", "btn btn-sm"), "copy", "Copy"); cp.type = "button";
      cp.addEventListener("click", async () => {
        const ok = await copyText(`${q.text}\n— ${q.document}${q.page ? ", page " + q.page : ""}`);
        toast(ok ? "Exact text copied" : "Copy is not available in this browser", ok ? "ok" : "error", 2000);
      });
      actions.append(cp);
      head.append(docBadge(q.document), titles, actions);
      const body = el("div", "exact-text");
      body.append(withIcon(el("div", "exact-label"), "quote", "Exact text from the document"));
      for (const line of q.text.split("\n")) {
        const cls = `ln${/^\s*\d{1,2}[.)]\s/.test(line) ? " step" : ""}${/^(WARNING|CAUTION|DANGER)\b/i.test(line) ? " warn-line" : ""}`;
        const ln = el("span", cls);
        highlight(ln, line, ids);
        body.append(ln);
      }
      card.append(head, body);
      view.exact.append(card);
    }
  }

  function renderNotFound(view, text) {
    view.exact.textContent = "";
    const box = el("div", "not-found");
    box.append(iconEl("warn"), el("div", null, text));
    view.exact.append(box);
  }

  function renderSources(view, sources) {
    view.srcData = sources || [];
    const list = $(".source-list", view.sources);
    list.textContent = "";
    if (!view.srcData.length) { view.sources.hidden = true; return; }
    $("summary span:last-child", view.sources).textContent = `Evidence — ${view.srcData.length} passage${view.srcData.length > 1 ? "s" : ""} found`;
    for (const s of view.srcData) {
      const card = el("div", "source");
      card.id = `src-${view.box.dataset.k}-${s.n}`;
      const head = el("div", "source-head");
      head.append(el("span", "source-n", `[${s.n}]`), docBadge(s.document), el("span", "source-doc", s.document), el("span", "source-meta", whereText(s)));
      if (s.matched_ids && s.matched_ids.length) head.append(el("span", "badge ok", `exact match: ${s.matched_ids.join(", ")}`));
      head.append(el("span", "source-score", `score ${Number(s.score).toFixed(2)}`));
      const body = el("div", "source-text");
      highlight(body, s.text, s.matched_ids);
      const actions = el("div", "source-actions");
      const more = el("button", "btn btn-sm", "Expand"); more.type = "button";
      more.addEventListener("click", () => { card.classList.toggle("open"); more.textContent = card.classList.contains("open") ? "Collapse" : "Expand"; });
      const ctx = el("button", "btn btn-sm", "Surrounding text"); ctx.type = "button";
      ctx.addEventListener("click", () => openContext(s));
      actions.append(more, ctx);
      if (s.document_id) {
        const open = el("a", "btn btn-sm"); open.href = fileUrl(s.document_id, /\.pdf$/i.test(s.document) ? s.page : null); open.target = "_blank"; open.rel = "noopener";
        withIcon(open, "external", "Open original");
        actions.append(open);
      }
      card.append(head, body, actions);
      list.append(card);
    }
    view.sources.hidden = false;
  }

  function renderVerification(view, v) {
    view.verify.textContent = "";
    if (!v || v.confidence === "n/a") { view.verify.hidden = true; return; }
    const cls = v.confidence === "high" ? "ok" : v.confidence === "medium" ? "warn" : "bad";
    const line = el("div");
    line.append(el("span", `badge ${cls}`, `Checked against sources: ${v.confidence}`), document.createTextNode(` ${Math.round((v.supported_ratio ?? 0) * 100)}% of statements supported`));
    view.verify.append(line);
    const issues = [];
    if (v.unknown_identifiers?.length) issues.push(`Codes not in the sources: ${v.unknown_identifiers.join(", ")}`);
    if (v.unknown_numbers?.length) issues.push(`Values not in the sources: ${v.unknown_numbers.join(", ")}`);
    for (const s of v.unsupported || []) issues.push(`Weakly supported: “${s}”`);
    if (issues.length) { const ul = el("ul"); issues.forEach((t) => ul.append(el("li", null, t))); view.verify.append(ul); }
    view.verify.hidden = false;
  }

  function renderRelated(view, related, missing) {
    view.related.textContent = "";
    let any = false;
    if (missing && missing.length) { view.related.append(el("span", "badge warn", `Not in any document: ${missing.join(", ")}`)); any = true; }
    const codes = (related && related.related) || [];
    if (codes.length) {
      const group = el("div", "rel-group");
      group.append(withIcon(el("div", "rel-title"), "link", `Related codes — documented together with ${related.identifiers.join(", ")}`));
      const chips = el("div", "rel-chips");
      for (const r of codes.slice(0, 8)) {
        const c = el("button", "rel-chip"); c.type = "button";
        c.append(el("span", "rel-code", r.identifier));
        if (r.label) c.append(el("span", "rel-label", r.label));
        c.title = `Ask about ${r.identifier}`;
        c.addEventListener("click", () => ask(`What does ${r.identifier} mean and what is the remedy?`));
        chips.append(c);
      }
      group.append(chips);
      view.related.append(group);
      any = true;
    }
    const locs = (related && related.locations) || [];
    if (locs.length) {
      const group = el("div", "rel-group");
      group.append(withIcon(el("div", "rel-title"), "doc", `Also documented in${related.records ? ` · ${fmtNum(related.records)} history records` : ""}`));
      const list = el("div", "rel-locs");
      for (const l of locs) {
        const row = el("div", "rel-loc");
        const text = el("div", "rel-loc-text");
        text.append(el("span", "rel-loc-doc", l.document), el("span", "rel-loc-where", whereText(l)), el("span", "rel-loc-snippet", l.snippet));
        row.append(docBadge(l.document), text);
        const btn = el("button", "btn btn-sm", "View"); btn.type = "button";
        btn.addEventListener("click", () => openContext({ chunk_id: l.chunk_id, document: l.document, document_id: l.document_id,
          page: l.page, section: l.section, content_type: l.content_type, matched_ids: related.identifiers }));
        row.append(btn);
        list.append(row);
      }
      group.append(list);
      view.related.append(group);
      any = true;
    }
    view.related.hidden = !any;
  }

  function renderTiming(view, t, cacheHit) {
    if (!t) return;
    const bits = [`found in ${fmtMs(t.retrieval_ms)}`];
    if (t.llm_ms) bits.push(`AI ${fmtMs(t.llm_ms)}`);
    if (cacheHit) bits.push("cached");
    withIcon(view.timing, "clock", bits.join(" · "));
  }

  const idsOf = (sources) => Array.from(new Set((sources || []).flatMap((s) => s.matched_ids || [])));

  function renderFinal(view, r) {
    setStatus(view, r);
    view.aiBody.classList.remove("typing");
    if (r.sources) renderSources(view, r.sources);
    if (r.status === "not_found" || r.status === "error") renderNotFound(view, r.answer || r.message || "Not available.");
    else if (r.exact && r.exact.length) renderExact(view, r.exact, idsOf(r.sources));
    else { view.exact.textContent = ""; const a = el("div", "answer"); renderMarkdown(a, r.answer, view.srcData); view.exact.append(a); }
    if (r.mode_used === "llm") { view.ai.hidden = false; renderMarkdown(view.aiBody, r.answer, view.srcData); renderVerification(view, r.verification); }
    else view.ai.hidden = true;
    const notes = [];
    if (r.notice) notes.push(r.notice);
    if (r.status === "partial" && r.exact_markdown) { const m = r.exact_markdown.match(/^\*\*Note:\*\* (.+)$/m); if (m) notes.push(m[1]); }
    if (notes.length) { view.notice.innerHTML = icon("warn"); view.notice.append(el("span", null, notes.join(" "))); view.notice.hidden = false; }
    renderRelated(view, r.related, r.missing_identifiers);
    renderTiming(view, r.timings, r.cache_hit);
  }

  // ------------------------------------------------------------------ ask
  function newId() {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
    const b = new Uint8Array(16); crypto.getRandomValues(b); b[6] = (b[6] & 0x0f) | 0x40; b[8] = (b[8] & 0x3f) | 0x80;
    const h = Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
    return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
  }

  async function ask(query) {
    query = query.trim();
    if (!query || state.controller) return;
    if (!state.conversationId) state.conversationId = newId();  // created server-side with the first message
    addUserMessage(query);
    const view = createBotMessage();
    const body = { query, stream: true, mode: state.mode, conversation_id: state.conversationId,
      collections: $("#filter-collection").value ? [$("#filter-collection").value] : [],
      file_types: $("#filter-type").value ? [$("#filter-type").value] : [] };
    state.controller = new AbortController();
    toggleSending(true);
    let tokens = "", finished = false;
    try {
      const res = await api("/api/v1/query", { method: "POST", json: body, raw: true, signal: state.controller.signal });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "", pending = false;
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const raw = buf.slice(0, idx).trim();
          buf = buf.slice(idx + 2);
          if (!raw.startsWith("data:")) continue;
          const ev = JSON.parse(raw.slice(5));
          if (ev.type === "retrieval") {
            renderSources(view, ev.sources);
            renderRelated(view, ev.related, ev.missing_identifiers);
            if (ev.timings) renderTiming(view, ev.timings, false);
            if (ev.status === "answered" || ev.status === "partial") {
              renderExact(view, ev.exact, idsOf(ev.sources));
              setStatus(view, { status: ev.status, mode_used: "evidence" });
              if (body.mode !== "evidence" && ev.status === "answered") {
                view.ai.hidden = false;
                view.aiBody.textContent = "The local AI is writing a summary of this text (10–30 s on CPU-only hardware)…";
                view.aiBody.classList.add("typing");
              }
            } else renderNotFound(view, "Not found in the indexed documents.");
          } else if (ev.type === "token") {
            if (!tokens) view.aiBody.textContent = "";
            tokens += ev.text;
            if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; renderMarkdown(view.aiBody, tokens, view.srcData); view.aiBody.classList.add("typing"); }); }
          } else if (ev.type === "done") { finished = true; renderFinal(view, ev); }
          else if (ev.type === "error") { finished = true; renderFinal(view, { status: "error", answer: ev.message }); }
        }
      }
      if (!finished) throw new Error("connection closed before the answer finished");
    } catch (e) {
      view.aiBody.classList.remove("typing");
      if (e.name === "AbortError") { view.status.className = "badge warn"; view.status.textContent = "Stopped"; }
      else { renderFinal(view, { status: "error", answer: `Request failed: ${e.message}` }); toast(e.message, "error"); }
    } finally {
      state.controller = null;
      toggleSending(false);
      loadConversations();
    }
  }

  function toggleSending(on) { $("#send").hidden = on; $("#stop").hidden = !on; }

  // ------------------------------------------------------------------ context dialog
  async function openContext(src) {
    const dlg = $("#source-dialog");
    $("#source-title").textContent = src.document;
    $("#source-sub").textContent = whereText(src);
    const open = $("#source-open");
    open.href = src.document_id ? fileUrl(src.document_id, src.page) : "#";
    open.hidden = !src.document_id;
    const body = $("#source-body");
    body.textContent = "";
    body.append(el("div", "skeleton w80"), el("div", "skeleton w60"));
    dlg.showModal();
    try {
      const r = await api(`/api/v1/chunks/${src.chunk_id}?context=3`);
      body.textContent = "";
      for (const c of r.chunks) {
        body.append(el("div", "ctx-meta", [c.section, c.page ? (c.content_type === "record" ? `row ${c.page}` : `page ${c.page}`) : ""].filter(Boolean).join(" · ")));
        const box = el("div", `ctx-chunk${c.id === src.chunk_id ? " hit" : ""}`);
        highlight(box, c.text, src.matched_ids);
        body.append(box);
      }
      $(".ctx-chunk.hit", body)?.scrollIntoView({ block: "center" });
    } catch (e) { body.textContent = `Could not load context: ${e.message}`; }
  }

  document.addEventListener("click", (e) => {
    const b = e.target.closest(".cite");
    if (!b) return;
    const msg = b.closest(".msg-bot");
    const card = msg && $(`#src-${msg.dataset.k}-${b.dataset.n}`);
    if (!card) return;
    card.closest("details").open = true;
    card.scrollIntoView({ block: "center", behavior: "smooth" });
    card.classList.add("flash");
    setTimeout(() => card.classList.remove("flash"), 1400);
  });

  // ------------------------------------------------------------------ conversations
  async function loadConversations() {
    try { state.conversations = await api("/api/v1/conversations?limit=100"); renderConversations(); } catch { /* status pill shows outages */ }
  }

  function renderConversations() {
    const list = $("#conv-list");
    const f = $("#conv-filter").value.trim().toLowerCase();
    list.textContent = "";
    const items = state.conversations.filter((c) => !f || c.title.toLowerCase().includes(f));
    if (!items.length) { list.append(el("li", "conv-empty", f ? "No matches" : "Your questions will appear here")); return; }
    for (const c of items) {
      const li = el("li", `conv-item${c.id === state.conversationId ? " active" : ""}`);
      li.tabIndex = 0; li.title = fmtDate(c.updated_at);
      const del = el("button", "icon-btn del"); del.type = "button"; del.innerHTML = icon("close"); del.setAttribute("aria-label", "Delete conversation");
      del.addEventListener("click", async (e) => {
        e.stopPropagation();
        if (!confirm(`Delete “${c.title}”?`)) return;
        try { await api(`/api/v1/conversations/${c.id}`, { method: "DELETE" }); if (c.id === state.conversationId) newChat(); loadConversations(); } catch (err) { toast(err.message, "error"); }
      });
      li.append(el("span", "title", c.title), del);
      li.addEventListener("click", () => openConversation(c.id));
      li.addEventListener("keydown", (e) => { if (e.key === "Enter") openConversation(c.id); });
      list.append(li);
    }
  }

  async function openConversation(id) {
    if (state.controller) return;
    try {
      const conv = await api(`/api/v1/conversations/${id}`);
      state.conversationId = id;
      clearMessages();
      for (const m of conv.messages) {
        if (m.role === "user") { addUserMessage(m.content); continue; }
        renderFinal(createBotMessage(), Object.assign({ answer: m.content }, m.payload || {}, { cache_hit: false }));
      }
      showView("ask"); renderConversations(); closeSidebar();
    } catch (e) { toast(e.message, "error"); }
  }

  function clearMessages() { $$("#messages > :not(#empty-state)").forEach((n) => n.remove()); }

  function newChat() {
    if (state.controller) state.controller.abort();
    state.conversationId = null;
    clearMessages();
    $("#empty-state").hidden = false;
    renderConversations(); showView("ask"); $("#query").focus();
  }

  // ------------------------------------------------------------------ filters
  async function loadFilters() {
    try {
      const r = await api("/api/v1/collections");
      for (const [sel, values] of [["#filter-collection", r.collections.map((c) => c.name)], ["#filter-type", r.file_types]]) {
        const s = $(sel); const keep = s.value; s.length = 1;
        for (const v of values) s.append(new Option(v, v));
        s.value = values.includes(keep) ? keep : "";
      }
      updateFilterCount();
    } catch { /* ignore */ }
  }
  function updateFilterCount() {
    const n = ["#filter-collection", "#filter-type"].filter((s) => $(s).value).length;
    $("#filter-count").textContent = String(n); $("#filter-count").hidden = !n;
  }

  // ------------------------------------------------------------------ documents
  async function loadDocuments() {
    const q = encodeURIComponent($("#doc-search").value.trim());
    const st = encodeURIComponent($("#doc-status").value);
    try {
      const r = await api(`/api/v1/documents?limit=${state.docLimit}&offset=${state.docOffset}&q=${q}&status=${st}`);
      const tb = $("#doc-table tbody");
      tb.textContent = "";
      for (const d of r.documents) {
        const tr = el("tr");
        const name = el("td", "name"); const cell = el("div", "doc-cell");
        const link = el("a", null, d.name); link.href = fileUrl(d.id); link.target = "_blank"; link.rel = "noopener"; link.title = d.path;
        cell.append(docBadge(d.name), link); name.append(cell);
        const status = el("td"); status.append(el("span", `badge ${d.status === "indexed" ? "ok" : d.status === "failed" ? "bad" : "warn"}`, d.status));
        if (d.error) status.title = d.error;
        const act = el("td", "actions");
        const re = el("button", "btn btn-sm", "Re-index"); re.type = "button";
        re.addEventListener("click", async () => { try { await api(`/api/v1/documents/${d.id}/reindex`, { method: "POST" }); toast(`Re-index queued: ${d.name}`, "ok"); loadJobs(); } catch (e) { toast(e.message, "error"); } });
        const del = el("button", "btn btn-sm btn-danger", "Delete"); del.type = "button";
        del.addEventListener("click", async () => { if (!confirm(`Remove “${d.name}” from the index?`)) return; try { await api(`/api/v1/documents/${d.id}`, { method: "DELETE" }); toast("Removed", "ok"); loadDocuments(); } catch (e) { toast(e.message, "error"); } });
        act.append(re, document.createTextNode(" "), del);
        tr.append(name, el("td", null, d.collection), status, el("td", "num", fmtNum(d.pages)), el("td", "num", fmtNum(d.chunk_count)), el("td", null, fmtDate(d.indexed_at)), act);
        tb.append(tr);
      }
      if (!r.documents.length) { const tr = el("tr"); const td = el("td", "muted", "No documents yet — upload manuals above or copy them into the media folder."); td.colSpan = 7; tr.append(td); tb.append(tr); }
      $("#doc-total").textContent = `(${fmtNum(r.total)})`;
      const page = Math.floor(state.docOffset / state.docLimit) + 1, pages = Math.max(1, Math.ceil(r.total / state.docLimit));
      $("#doc-page").textContent = `Page ${page} / ${pages}`;
      $("#doc-prev").disabled = page <= 1; $("#doc-next").disabled = page >= pages;
    } catch (e) { toast(e.message, "error"); }
  }

  async function loadJobs() {
    try {
      const jobs = await api("/api/v1/ingest/jobs?limit=25");
      const tb = $("#job-table tbody");
      tb.textContent = "";
      for (const j of jobs) {
        const tr = el("tr");
        const st = el("td"); st.append(el("span", `badge ${j.status === "succeeded" || j.status === "skipped" ? "ok" : j.status === "failed" ? "bad" : "warn"}`, j.status));
        const nm = el("td", "name"); const cell = el("div", "doc-cell"); cell.append(docBadge(j.source_name), el("span", null, j.source_name)); nm.append(cell);
        tr.append(nm, st, el("td", "num", fmtNum(j.chunks_total)), el("td", "num", j.attempts), el("td", null, fmtDate(j.created_at)), el("td", "err", j.error || ""));
        tb.append(tr);
      }
      if (!jobs.length) { const tr = el("tr"); const td = el("td", "muted", "No indexing jobs yet."); td.colSpan = 6; tr.append(td); tb.append(tr); }
    } catch { /* shown via status */ }
  }

  function uploadFiles(files) {
    const coll = $("#upload-collection").value.trim() || "manuals";
    for (const file of files) {
      const row = el("div", "up-row"); const pct = el("span", "muted", "0%");
      const bar = el("div", "bar"); const fill = el("i"); bar.append(fill); row.append(el("span", null, file.name), pct, bar);
      $("#upload-progress").prepend(row);
      const form = new FormData(); form.append("file", file); form.append("collection", coll);
      const xhr = new XMLHttpRequest(); xhr.open("POST", "/api/v1/upload");
      const key = store.get("rag.apiKey", ""); if (key) xhr.setRequestHeader("X-API-Key", key);
      xhr.upload.onprogress = (e) => { if (e.lengthComputable) { const p = Math.round((e.loaded / e.total) * 100); pct.textContent = `${p}%`; fill.style.width = `${p}%`; } };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) { pct.textContent = "uploaded · indexing"; fill.style.width = "100%"; loadJobs(); loadDocuments(); }
        else { let msg = xhr.statusText; try { msg = JSON.parse(xhr.responseText).detail; } catch { /* keep */ } pct.textContent = `failed — ${msg}`; toast(`Upload failed: ${msg}`, "error"); }
      };
      xhr.onerror = () => { pct.textContent = "network error"; };
      xhr.send(form);
    }
  }

  // ------------------------------------------------------------------ system
  async function loadStatus() {
    try {
      const s = await api("/api/v1/status");
      $("#status-pill .dot").className = `dot ${s.status}`;
      $("#status-text").textContent = s.status === "ok" ? "All systems OK" : s.status === "degraded" ? "Degraded" : "Down";
      const idx = s.index || {};
      const meta = $("#index-meta"); meta.textContent = "";
      meta.append(withIcon(el("span", "meta-pill"), "doc", `${fmtNum((idx.documents || {}).indexed || 0)} documents · ${fmtNum(idx.chunks)} passages`));
      if (state.view === "system") renderSystem(s);
    } catch { $("#status-pill .dot").className = "dot down"; $("#status-text").textContent = "API unreachable"; }
  }

  function renderSystem(s) {
    const cards = $("#service-cards"); cards.textContent = "";
    const add = (title, status, sub) => {
      const c = el("div", "card"); const t = el("div", "card-title");
      t.append(el("span", `dot ${status === "ok" ? "ok" : status === "down" ? "down" : "degraded"}`), el("span", null, title));
      c.append(t, el("div", "card-sub", sub)); cards.append(c);
    };
    const sv = s.services;
    add("API", sv.api.status, `Responded in ${fmtMs(s.elapsed_ms)}`);
    add("Database", sv.database.status, sv.database.status === "ok" ? "PostgreSQL connected" : "PostgreSQL unreachable");
    add("Vector index", sv.vector_store.status, sv.vector_store.status === "ok" ? "Qdrant connected" : "Qdrant unreachable");
    add("Cache", sv.cache.status, sv.cache.status === "ok" ? "Redis connected" : "Unavailable — answers not cached");
    add("Local AI", sv.llm.status === "ok" ? "ok" : "degraded", sv.llm.status === "ok" ? `${sv.llm.model} ready` : `${sv.llm.model}: ${sv.llm.status} — exact answers still work`);
    add("Search models", sv.models.loaded ? "ok" : "down", `${sv.models.embedding_model}`);
    const kv = (root, rows) => { root.textContent = ""; for (const [k, v] of rows) root.append(el("div", "k", k), el("div", "v", v)); };
    const idx = s.index || {};
    kv($("#index-kv"), [
      ["Documents", Object.entries(idx.documents || {}).map(([k, v]) => `${k}: ${fmtNum(v)}`).join(" · ") || "0"],
      ["Passages", fmtNum(idx.chunks)], ["Vectors", fmtNum(idx.vectors)],
      ["Indexing queue", Object.entries(idx.queue || {}).map(([k, v]) => `${k}: ${v}`).join(" · ") || "idle"],
    ]);
    kv($("#config-kv"), [
      ["Local AI model", s.config.llm_model], ["Passages per answer", String(s.config.top_k)],
      ["Reranking", typeof s.config.rerank === "string" ? s.config.rerank : (s.config.rerank ? "on" : "off")],
      ["API key required", s.config.auth ? "yes" : "no"],
    ]);
  }

  // ------------------------------------------------------------------ views & wiring
  function showView(name) {
    state.view = name;
    $$(".nav-btn").forEach((b) => { const on = b.dataset.view === name; b.classList.toggle("active", on); b.setAttribute("aria-selected", on); });
    $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
    $("#view-title").textContent = { ask: "Ask", docs: "Documents", system: "System status" }[name];
    clearInterval(state.timers.view);
    if (name === "docs") { loadDocuments(); loadJobs(); state.timers.view = setInterval(loadJobs, 5000); }
    if (name === "system") { loadStatus(); state.timers.view = setInterval(loadStatus, 10000); }
    closeSidebar();
  }

  function autoGrow() { const t = $("#query"); t.style.height = "auto"; t.style.height = `${Math.min(t.scrollHeight, 200)}px`; }
  function closeSidebar() { $("#sidebar").classList.remove("open"); $("#scrim").hidden = true; }
  function applyTheme(t) { if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme; }
  function setMode(mode) {
    state.mode = mode; store.set("rag.mode", mode);
    $$(".seg").forEach((b) => { const on = b.dataset.mode === mode; b.classList.toggle("active", on); b.setAttribute("aria-checked", on); });
  }

  function init() {
    hydrateIcons();
    applyTheme(store.get("rag.theme", ""));
    $("#theme-toggle").addEventListener("click", () => {
      const dark = document.documentElement.dataset.theme === "dark" || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
      const next = dark ? "light" : "dark"; store.set("rag.theme", next); applyTheme(next);
    });
    $$(".nav-btn").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
    $("#menu-btn").addEventListener("click", () => { $("#sidebar").classList.add("open"); $("#scrim").hidden = false; });
    $("#scrim").addEventListener("click", closeSidebar);
    $("#new-chat").addEventListener("click", newChat);
    $("#conv-filter").addEventListener("input", renderConversations);
    $("#composer").addEventListener("submit", (e) => { e.preventDefault(); const q = $("#query").value; $("#query").value = ""; autoGrow(); ask(q); });
    $("#query").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("#composer").requestSubmit(); } });
    $("#query").addEventListener("input", autoGrow);
    $("#stop").addEventListener("click", () => state.controller && state.controller.abort());
    $$("#examples .ex").forEach((c) => c.addEventListener("click", () => ask(c.textContent)));
    $$(".seg").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
    setMode(store.get("rag.mode", "evidence") === "auto" ? "auto" : "evidence");
    $("#filters-toggle").addEventListener("click", () => { const f = $("#filters"); f.hidden = !f.hidden; $("#filters-toggle").setAttribute("aria-expanded", String(!f.hidden)); });
    $("#filter-collection").addEventListener("change", updateFilterCount);
    $("#filter-type").addEventListener("change", updateFilterCount);

    const drop = $("#drop");
    drop.addEventListener("click", (e) => { if (e.target.closest("input, label")) return; $("#file-input").click(); });
    drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#file-input").click(); } });
    $("#file-input").addEventListener("change", (e) => { uploadFiles(e.target.files); e.target.value = ""; });
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("over"));
    drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); uploadFiles(e.dataTransfer.files); });
    let docTimer;
    $("#doc-search").addEventListener("input", () => { clearTimeout(docTimer); docTimer = setTimeout(() => { state.docOffset = 0; loadDocuments(); }, 250); });
    $("#doc-status").addEventListener("change", () => { state.docOffset = 0; loadDocuments(); });
    $("#doc-prev").addEventListener("click", () => { state.docOffset = Math.max(0, state.docOffset - state.docLimit); loadDocuments(); });
    $("#doc-next").addEventListener("click", () => { state.docOffset += state.docLimit; loadDocuments(); });
    $("#rescan").addEventListener("click", async () => { try { const r = await api("/api/v1/ingest/scan", { method: "POST" }); toast(`Scan: ${r.queued} queued, ${r.removed} removed`, "ok"); loadJobs(); } catch (e) { toast(e.message, "error"); } });

    loadStatus(); loadConversations(); loadFilters();
    state.timers.status = setInterval(loadStatus, 15000);
    state.timers.filters = setInterval(loadFilters, 60000);
    $("#query").focus();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
