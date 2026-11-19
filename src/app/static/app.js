import { FilmCartridge3D } from "/static/film_cartridge_3d.js";

const heroTitleWrap = document.getElementById("heroTitleWrap");
const composerDock = document.getElementById("composerDock");
const messagesEl = document.getElementById("messages");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");

const statusDot = document.getElementById("statusDot");
const versionBadge = document.getElementById("versionBadge");
const fovText = document.getElementById("fovText");
const disclaimerBar = document.getElementById("disclaimerBar");

const hudState = document.getElementById("hudState");
const filmPlaceholder = document.getElementById("filmPlaceholder");
const filmTray = document.getElementById("filmTray");
const filmLightbox = document.getElementById("filmLightbox");
const laserOverlay = document.getElementById("laserOverlay");

const cartridgeCanvas = document.getElementById("cartridgeCanvas");
const cartridge = new FilmCartridge3D(cartridgeCanvas);

let chatHistory = [];
let maxContextTokens = 65536;
let minSimilarityScore = 0.2;
let turnCounter = 0;

/** @type {Record<string, any>} */
const turnResponses = {};
let activeTurnId = null;
let lockedAnchorEl = null;
let hoveredAnchorEl = null;

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function runFlipTransition(mutateDomFn) {
  const els = [heroTitleWrap, composerDock];
  const firstRects = els.map((el) => el.getBoundingClientRect());
  mutateDomFn();
  const lastRects = els.map((el) => el.getBoundingClientRect());

  els.forEach((el, idx) => {
    const f = firstRects[idx];
    const l = lastRects[idx];
    if (!f.width || !l.width) return;

    const dx = f.left - l.left;
    const dy = f.top - l.top;
    // 标题块保持等比例缩放，避免文字拉扁；输入栏保持水平位移动效
    const scaleX = el === heroTitleWrap ? f.width / l.width : 1;
    const scaleY = el === heroTitleWrap ? f.width / l.width : 1;

    el.style.transition = "none";
    el.style.transformOrigin = "top left";
    el.style.transform = `translate3d(${dx}px, ${dy}px, 0) scale(${scaleX}, ${scaleY})`;
  });

  void document.body.offsetHeight;

  els.forEach((el) => {
    el.style.transition = "transform 680ms cubic-bezier(0.22, 1, 0.36, 1)";
    el.style.transform = "none";
  });

  setTimeout(() => {
    els.forEach((el) => {
      el.style.transition = "";
      el.style.transform = "";
      el.style.transformOrigin = "";
    });
  }, 720);
}

function transitionToActive() {
  if (!document.body.classList.contains("is-landing")) return;
  runFlipTransition(() => {
    document.body.classList.remove("is-landing");
    document.body.classList.add("is-active");
  });
  requestAnimationFrame(() => cartridge.resizeNow());
  setTimeout(() => cartridge.resizeNow(), 360);
}

function transitionToLanding() {
  if (document.body.classList.contains("is-landing")) return;
  runFlipTransition(() => {
    document.body.classList.remove("is-active");
    document.body.classList.add("is-landing");
  });
}

async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    maxContextTokens = Number(data.max_context_tokens) || 65536;
    minSimilarityScore = Number(data.min_similarity_score) || 0.2;
    const modeLabel = `${data.retrieval_mode || "hybrid"}${data.rerank_enabled ? "+rerank" : ""}`;
    versionBadge.textContent = modeLabel.toUpperCase();
    fovText.textContent = `FOV 0 / ${(maxContextTokens / 1000).toFixed(1)}K`;
    if (data.disclaimer) {
      disclaimerBar.textContent = data.disclaimer;
    }
  } catch (err) {
    versionBadge.textContent = "OFFLINE";
  }
}

function updateFov(resp) {
  const est = resp?.context_stats?.estimated_prompt_tokens || resp?.total_tokens || 0;
  const estStr = est >= 1000 ? `${(est / 1000).toFixed(1)}K` : String(est);
  fovText.textContent = `FOV ${estStr} / ${(maxContextTokens / 1000).toFixed(1)}K`;
}

function formatReportBody(answerText, turnId, disclaimer) {
  let mainText = String(answerText || "");
  let hasDisclaimer = false;
  if (disclaimer && mainText.includes(disclaimer)) {
    mainText = mainText.replace(disclaimer, "").trim();
    hasDisclaimer = true;
  }

  let html = escapeHtml(mainText);
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\[(\d+)\]/g, (_, num) => {
    return `<button type="button" class="cite-anchor" data-turn="${turnId}" data-cite="${num}">[⊕${num}]</button>`;
  });

  if (hasDisclaimer) {
    html += `<div class="report-disclaimer">${escapeHtml(disclaimer)}</div>`;
  }
  return html;
}

function selectFilmCard(citeIndex) {
  const cards = Array.from(filmTray.querySelectorAll(".film-card"));
  if (!cards.length) return;
  let found = false;
  cards.forEach((card) => {
    const match = Number(card.dataset.sliceIndex) === Number(citeIndex);
    card.classList.toggle("is-active", match);
    if (match) {
      found = true;
      card.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  });
  filmTray.classList.toggle("has-active", found);
}

function renderFilmTrayForTurn(turnId, targetCitationIndex = null) {
  const resp = turnResponses[turnId];
  if (!resp) return;
  activeTurnId = turnId;
  filmPlaceholder.style.display = "none";
  filmTray.innerHTML = "";
  filmTray.classList.remove("has-active");

  if (resp.refused) {
    cartridge.setState("refused");
    statusDot.classList.remove("is-scanning");
    statusDot.classList.add("is-refused");
    hudState.textContent = "EXPOSURE REJECTED";
    hudState.classList.add("is-refused");

    const maxScore = Number(resp.max_score ?? 0);
    const thPct = Math.min(95, Math.max(5, Math.round(minSimilarityScore * 100)));
    const scorePct = Math.min(98, Math.max(2, Math.round(maxScore * 100)));

    const card = document.createElement("div");
    card.className = "film-card is-rejected";
    card.style.setProperty("--tilt", "-0.6deg");
    card.innerHTML = `
      <div class="film-sprocket left"></div>
      <div class="film-sprocket right"></div>
      <div class="film-edge-top">
        <span>CMED-70MM // OVEREXPOSED</span>
        <span>EXPOSURE REJECTED</span>
      </div>
      <div class="rejected-title">无效曝光废片 · 安全护栏已阻断</div>
      <div class="rejected-reason">
        判定原因：<code>${escapeHtml(resp.refusal_reason || "low_similarity")}</code><br/>
        当前知识库信噪比不足，系统拒绝在缺乏可靠医疗依据时推测作答。
      </div>
      <div class="snr-meter">
        <div class="snr-threshold-line" style="left: ${thPct}%;" title="安全阈值 ${minSimilarityScore}"></div>
        <div class="snr-score-dot" style="left: ${scorePct}%;" title="当前峰值 ${maxScore.toFixed(4)}"></div>
      </div>
      <div class="film-edge-bottom">
        <span>PEAK ${maxScore.toFixed(4)}</span>
        <span>THRESHOLD ${minSimilarityScore.toFixed(2)}</span>
      </div>
    `;
    filmTray.appendChild(card);
    return;
  }

  const citations = resp.citations || [];
  if (citations.length === 0) {
    cartridge.setState("idle");
    hudState.textContent = "0 FRAMES";
    hudState.classList.remove("is-refused");
    filmPlaceholder.style.display = "block";
    filmPlaceholder.textContent = "本次回答未附带可显影的参考胶片。";
    return;
  }

  cartridge.triggerEject(citations.length);
  statusDot.classList.remove("is-scanning", "is-refused");
  hudState.classList.remove("is-refused");
  hudState.textContent = `EJECTED · ${citations.length} FRAMES`;

  const tilts = ["-1.1deg", "0.9deg", "-0.6deg", "1.1deg", "-0.8deg"];
  citations.forEach((c, idx) => {
    const card = document.createElement("div");
    card.className = "film-card";
    card.dataset.sliceIndex = String(c.index);
    card.style.setProperty("--tilt", tilts[idx % tilts.length]);
    card.style.setProperty("--eject-delay", `${idx * 140}ms`);
    card.innerHTML = `
      <div class="film-sprocket left"></div>
      <div class="film-sprocket right"></div>
      <div class="film-edge-top">
        <span>CMED-70MM · FRAME ${String(c.index).padStart(2, "0")}A</span>
        <span class="film-conf-badge">CONF ${Number(c.score).toFixed(4)}</span>
      </div>
      <div class="film-body">${escapeHtml(c.text)}</div>
      <div class="film-edge-bottom">
        <span>DOC // ${escapeHtml(c.doc_id)}</span>
        <span>ISO 400 · ${escapeHtml(c.source || "cMedQA2")}</span>
      </div>
    `;
    card.addEventListener("click", () => {
      const already = card.classList.contains("is-active");
      if (already) {
        card.classList.remove("is-active");
        filmTray.classList.remove("has-active");
      } else {
        selectFilmCard(c.index);
      }
      requestAnimationFrame(drawLaserOverlay);
    });
    filmTray.appendChild(card);
  });

  if (targetCitationIndex !== null) {
    selectFilmCard(targetCitationIndex);
  }
}

function clearLaserOverlay() {
  laserOverlay.innerHTML = "";
}

function drawLaserOverlay() {
  const anchor = lockedAnchorEl || hoveredAnchorEl;
  if (!anchor || !document.body.contains(anchor)) {
    clearLaserOverlay();
    return;
  }

  const citeIndex = Number(anchor.dataset.cite);
  const turnId = anchor.dataset.turn;
  if (turnId !== activeTurnId) {
    clearLaserOverlay();
    return;
  }

  const targetCard = filmTray.querySelector(`.film-card[data-slice-index="${citeIndex}"]`);
  if (!targetCard) {
    clearLaserOverlay();
    return;
  }

  const aRect = anchor.getBoundingClientRect();
  const cRect = targetCard.getBoundingClientRect();
  const x1 = aRect.right + 4;
  const y1 = aRect.top + aRect.height / 2;
  const x2 = cRect.left + 2;
  const y2 = cRect.top + Math.min(34, cRect.height / 2);

  // 沿着两栏留白走廊折弯，不遮挡左栏文字
  const lbRect = filmLightbox.getBoundingClientRect();
  const gutterX = Math.max(x1 + 14, Math.min(x2 - 12, lbRect.left - 10));
  const d = `M ${x1.toFixed(1)} ${y1.toFixed(1)} L ${gutterX.toFixed(1)} ${y1.toFixed(1)} L ${gutterX.toFixed(1)} ${y2.toFixed(1)} L ${x2.toFixed(1)} ${y2.toFixed(1)}`;

  const resp = turnResponses[turnId];
  const citeObj = (resp?.citations || []).find((c) => c.index === citeIndex);
  const confText = citeObj ? `CONF ${Number(citeObj.score).toFixed(4)}` : `FRAME #${citeIndex}`;
  const labelX = Math.max(x1 + 10, gutterX - 84);
  const labelY = y2 - 10;

  laserOverlay.innerHTML = `
    <path d="${d}" fill="none" stroke="rgba(15, 118, 110, 0.72)" stroke-width="1.5" stroke-dasharray="4 3" />
    <circle cx="${x1}" cy="${y1}" r="3" fill="#0f766e" />
    <circle cx="${x2}" cy="${y2}" r="4.5" fill="none" stroke="#14b8a6" stroke-width="1.5" />
    <circle cx="${x2}" cy="${y2}" r="2" fill="#0f766e" />
    <rect x="${labelX - 5}" y="${labelY - 11}" width="84" height="16" rx="4" fill="rgba(255,255,255,0.92)" stroke="rgba(15,118,110,0.35)" stroke-width="0.8" />
    <text x="${labelX}" y="${labelY}" fill="#0f766e" font-family="JetBrains Mono, monospace" font-size="10">${confText}</text>
  `;
}

function updateAnchorActiveClasses() {
  document.querySelectorAll(".cite-anchor.is-active").forEach((el) => el.classList.remove("is-active"));
  const active = lockedAnchorEl || hoveredAnchorEl;
  if (active) active.classList.add("is-active");
}

function handleAnchorEnter(anchor) {
  hoveredAnchorEl = anchor;
  const turnId = anchor.dataset.turn;
  const citeIndex = Number(anchor.dataset.cite);
  if (turnId !== activeTurnId) {
    renderFilmTrayForTurn(turnId, citeIndex);
  } else {
    selectFilmCard(citeIndex);
  }
  updateAnchorActiveClasses();
  requestAnimationFrame(drawLaserOverlay);
  setTimeout(drawLaserOverlay, 180);
}

function handleAnchorLeave() {
  hoveredAnchorEl = null;
  updateAnchorActiveClasses();
  if (lockedAnchorEl) {
    drawLaserOverlay();
  } else {
    clearLaserOverlay();
  }
}

function handleAnchorClick(anchor) {
  if (lockedAnchorEl === anchor) {
    lockedAnchorEl = null;
    clearLaserOverlay();
  } else {
    lockedAnchorEl = anchor;
    handleAnchorEnter(anchor);
  }
  updateAnchorActiveClasses();
}

messagesEl.addEventListener("mouseover", (e) => {
  const anchor = e.target.closest(".cite-anchor");
  if (anchor) handleAnchorEnter(anchor);
});

messagesEl.addEventListener("mouseout", (e) => {
  const anchor = e.target.closest(".cite-anchor");
  if (anchor) handleAnchorLeave();
});

messagesEl.addEventListener("click", (e) => {
  const anchor = e.target.closest(".cite-anchor");
  if (anchor) {
    e.stopPropagation();
    handleAnchorClick(anchor);
  } else if (lockedAnchorEl) {
    lockedAnchorEl = null;
    updateAnchorActiveClasses();
    clearLaserOverlay();
  }
});

messagesEl.addEventListener("scroll", () => {
  if (lockedAnchorEl || hoveredAnchorEl) requestAnimationFrame(drawLaserOverlay);
});

filmLightbox.addEventListener("scroll", () => {
  if (lockedAnchorEl || hoveredAnchorEl) requestAnimationFrame(drawLaserOverlay);
});

window.addEventListener("resize", () => {
  if (lockedAnchorEl || hoveredAnchorEl) requestAnimationFrame(drawLaserOverlay);
});

document.querySelectorAll(".example-pill").forEach((btn) => {
  btn.addEventListener("click", () => {
    queryInput.value = btn.dataset.query || "";
    queryInput.focus();
  });
});

async function sendQuery() {
  const query = queryInput.value.trim();
  if (!query) return;

  transitionToActive();

  lockedAnchorEl = null;
  hoveredAnchorEl = null;
  clearLaserOverlay();

  queryInput.value = "";
  sendBtn.disabled = true;

  turnCounter += 1;
  const turnId = `turn_${turnCounter}`;

  const userGroup = document.createElement("div");
  userGroup.className = "turn-user-group";
  const userBubble = document.createElement("div");
  userBubble.className = "msg-user";
  userBubble.textContent = query;
  userGroup.appendChild(userBubble);
  messagesEl.appendChild(userGroup);

  const reportEl = document.createElement("article");
  reportEl.className = "msg-assistant";
  reportEl.innerHTML = `
    <div class="report-label">CLINICAL REPORT // ${turnId.toUpperCase()}</div>
    <div class="report-content">正在检索医疗胶片库与精排显影...</div>
  `;
  const reportBodyEl = reportEl.querySelector(".report-content");
  messagesEl.appendChild(reportEl);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  cartridge.setState("scanning");
  statusDot.classList.remove("is-refused");
  statusDot.classList.add("is-scanning");
  hudState.classList.remove("is-refused");
  hudState.textContent = "DEVELOPING...";

  let streamedText = "";

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, history: chatHistory }),
    });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        const ev = JSON.parse(line.slice(5).trim());

        if (ev.type === "meta") {
          if (ev.rewritten_query && ev.rewritten_query !== query) {
            const rwEl = document.createElement("div");
            rwEl.className = "rewrite-hint";
            rwEl.textContent = `↳ 增强检索式 // ${ev.rewritten_query}`;
            userGroup.appendChild(rwEl);
          }
        } else if (ev.type === "delta") {
          streamedText += ev.content;
          reportBodyEl.textContent = streamedText;
          cartridge.triggerPulse();
          messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (ev.type === "done") {
          const resp = ev.response;
          turnResponses[turnId] = resp;
          if (resp.refused) {
            reportEl.classList.add("is-refused");
          }
          reportBodyEl.innerHTML = formatReportBody(resp.answer, turnId, resp.disclaimer);
          updateFov(resp);
          renderFilmTrayForTurn(turnId);
          chatHistory.push({ role: "user", content: query });
          chatHistory.push({ role: "assistant", content: resp.answer });
          messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (ev.type === "error") {
          reportBodyEl.textContent = `服务异常：${ev.message}`;
          cartridge.setState("refused");
        }
      }
    }
  } catch (err) {
    reportBodyEl.textContent = `请求失败：${err}`;
    cartridge.setState("refused");
    statusDot.classList.remove("is-scanning");
    statusDot.classList.add("is-refused");
  } finally {
    sendBtn.disabled = false;
    queryInput.focus();
  }
}

sendBtn.addEventListener("click", sendQuery);
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuery();
  }
});

clearBtn.addEventListener("click", () => {
  chatHistory = [];
  turnCounter = 0;
  activeTurnId = null;
  lockedAnchorEl = null;
  hoveredAnchorEl = null;
  clearLaserOverlay();
  messagesEl.innerHTML = "";
  filmTray.innerHTML = "";
  filmPlaceholder.style.display = "block";
  filmPlaceholder.textContent = "胶片暗盒待命中——提问后将从上方出片口吐出并显影循证胶片";
  cartridge.setState("idle");
  statusDot.classList.remove("is-scanning", "is-refused");
  hudState.classList.remove("is-refused");
  hudState.textContent = "STANDING BY";
  fovText.textContent = `FOV 0 / ${(maxContextTokens / 1000).toFixed(1)}K`;
  transitionToLanding();
});

loadHealth();
