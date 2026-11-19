import { MRIShaderCanvas } from "/static/mri_shader.js";

const messagesEl = document.getElementById("messages");
const emptyWelcomeEl = document.getElementById("emptyWelcome");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");
const statusDot = document.getElementById("statusDot");
const versionBadge = document.getElementById("versionBadge");
const fovText = document.getElementById("fovText");
const fovFill = document.getElementById("fovFill");
const disclaimerBar = document.getElementById("disclaimerBar");

const hudState = document.getElementById("hudState");
const zstackStage = document.getElementById("zstackStage");
const zstackScrubber = document.getElementById("zstackScrubber");
const zRange = document.getElementById("zRange");
const zAxisLabel = document.getElementById("zAxisLabel");
const laserOverlay = document.getElementById("laserOverlay");

const shaderCanvas = document.getElementById("mriShaderCanvas");
const shader = new MRIShaderCanvas(shaderCanvas, "/static/assets/mri_slice_texture.png");

let chatHistory = [];
let maxContextTokens = 65536;
let minSimilarityScore = 0.2;
let turnCounter = 0;

/** @type {Record<string, any>} */
const turnResponses = {};
let activeTurnId = null;
let activeSliceIndex = null;
let lockedAnchorEl = null;
let hoveredAnchorEl = null;

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    maxContextTokens = Number(data.max_context_tokens) || 65536;
    minSimilarityScore = Number(data.min_similarity_score) || 0.2;
    const modeLabel = `${data.retrieval_mode || "hybrid"}${data.rerank_enabled ? "+rerank" : ""}`;
    versionBadge.textContent = `${data.version || "v2"} · ${modeLabel}`.toUpperCase();
    fovText.textContent = `FOV: 0 / ${maxContextTokens.toLocaleString()} TOKENS`;
    if (data.disclaimer) {
      disclaimerBar.textContent = data.disclaimer;
    }
  } catch (err) {
    versionBadge.textContent = "OFFLINE";
  }
}

function updateFov(resp) {
  const est = resp?.context_stats?.estimated_prompt_tokens || resp?.total_tokens || 0;
  const pct = Math.min(100, Math.round((est / maxContextTokens) * 100));
  fovText.textContent = `FOV: ${est.toLocaleString()} / ${maxContextTokens.toLocaleString()} TOKENS`;
  fovFill.style.width = `${pct}%`;
}

function formatReportBody(answerText, turnId, disclaimer) {
  let mainText = String(answerText || "");
  let hasDisclaimer = false;
  if (disclaimer && mainText.includes(disclaimer)) {
    mainText = mainText.replace(disclaimer, "").trim();
    hasDisclaimer = true;
  }

  let html = escapeHtml(mainText);
  // 基础加粗渲染
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  // 将 [1] [2] 替换为交互式靶区锚点 [⊕1]
  html = html.replace(/\[(\d+)\]/g, (_, num) => {
    return `<button type="button" class="cite-anchor" data-turn="${turnId}" data-cite="${num}">[⊕${num}]</button>`;
  });

  if (hasDisclaimer) {
    html += `<div class="report-disclaimer">${escapeHtml(disclaimer)}</div>`;
  }
  return html;
}

function renderZStackForTurn(turnId, targetCitationIndex = null) {
  const resp = turnResponses[turnId];
  if (!resp) return;
  activeTurnId = turnId;

  if (resp.refused) {
    shader.setMode("refused", 0.0);
    statusDot.classList.remove("is-scanning");
    statusDot.classList.add("is-refused");
    hudState.textContent = "ARTIFACT INTERLOCK";
    hudState.classList.add("is-refused");
    zstackScrubber.hidden = true;

    const maxScore = Number(resp.max_score ?? 0);
    const thPct = Math.min(95, Math.max(5, Math.round(minSimilarityScore * 100)));
    const scorePct = Math.min(98, Math.max(2, Math.round(maxScore * 100)));

    zstackStage.innerHTML = `
      <div class="interlock-card">
        <div class="interlock-tag">[ARTIFACT INTERLOCK // 证据信噪比不足]</div>
        <div class="interlock-title">安全护栏已阻断本次生成</div>
        <div class="interlock-desc">
          判定原因：<code>${escapeHtml(resp.refusal_reason || "low_similarity")}</code><br/>
          系统拒绝在缺乏可靠医疗依据时进行推测性回答。
        </div>
        <div class="snr-meter">
          <div class="snr-threshold-line" style="left: ${thPct}%;" title="安全阈值 ${minSimilarityScore}"></div>
          <div class="snr-score-dot" style="left: ${scorePct}%;" title="当前峰值 ${maxScore.toFixed(4)}"></div>
        </div>
        <div class="snr-labels">
          <span>当前峰值: ${maxScore.toFixed(4)}</span>
          <span>安全线: ${minSimilarityScore.toFixed(2)}</span>
        </div>
      </div>
    `;
    return;
  }

  const citations = resp.citations || [];
  if (citations.length === 0) {
    shader.setMode("idle", 0.0);
    hudState.textContent = "NO SLICES";
    hudState.classList.remove("is-refused");
    zstackScrubber.hidden = true;
    zstackStage.innerHTML = `
      <div class="zstack-empty">
        <div class="code-line">SLICE // 0</div>
        <p>本次回答未附带可展示的断层引用切片。</p>
      </div>
    `;
    return;
  }

  const topScore = Math.max(...citations.map((c) => Number(c.score) || 0));
  shader.setMode("grounded", topScore);
  statusDot.classList.remove("is-scanning", "is-refused");
  hudState.classList.remove("is-refused");

  const resolvedTarget =
    targetCitationIndex && citations.some((c) => c.index === targetCitationIndex)
      ? targetCitationIndex
      : citations[0].index;
  activeSliceIndex = resolvedTarget;

  const deck = document.createElement("div");
  deck.className = "zstack-deck";

  citations.forEach((c, idx) => {
    const card = document.createElement("div");
    card.className = "slice-card";
    card.dataset.sliceIndex = String(c.index);
    card.dataset.pos = String(idx);
    card.innerHTML = `
      <div class="slice-dicom-top">
        <span class="slice-id">SLICE #${String(c.index).padStart(2, "0")} · ${escapeHtml(c.doc_id)}</span>
        <span class="slice-conf">CONF ${Number(c.score).toFixed(4)}</span>
      </div>
      <div class="slice-conf-bar">
        <div class="slice-conf-fill" style="width: ${Math.min(100, Math.round(Number(c.score) * 100))}%"></div>
      </div>
      <div class="slice-body">${escapeHtml(c.text)}</div>
      <div class="slice-dicom-bottom">
        <span>SRC: ${escapeHtml(c.source || "cMedQA2")}</span>
        <span>${escapeHtml(c.chunk_id)}</span>
      </div>
    `;
    card.addEventListener("click", () => {
      setActiveSlice(c.index);
    });
    deck.appendChild(card);
  });

  zstackStage.innerHTML = "";
  zstackStage.appendChild(deck);

  if (citations.length > 1) {
    zstackScrubber.hidden = false;
    zRange.min = "0";
    zRange.max = String(citations.length - 1);
    const activePos = citations.findIndex((c) => c.index === resolvedTarget);
    zRange.value = String(Math.max(0, activePos));
  } else {
    zstackScrubber.hidden = true;
  }

  applyZStackTransforms(resolvedTarget);
}

function applyZStackTransforms(targetCiteIndex) {
  activeSliceIndex = targetCiteIndex;
  const cards = Array.from(zstackStage.querySelectorAll(".slice-card"));
  if (cards.length === 0) return;

  let activePos = cards.findIndex((c) => Number(c.dataset.sliceIndex) === Number(targetCiteIndex));
  if (activePos < 0) activePos = 0;

  cards.forEach((card, pos) => {
    const rel = pos - activePos;
    const absRel = Math.abs(rel);
    const isActive = rel === 0;
    card.classList.toggle("is-active", isActive);
    card.style.zIndex = String(30 - absRel);

    if (isActive) {
      card.style.transform = "rotateY(0deg) rotateX(0deg) translate3d(0px, 0px, 36px) scale(1.01)";
      card.style.opacity = "1";
    } else {
      const tx = rel * 12;
      const ty = rel * 68;
      const tz = -absRel * 60;
      card.style.transform = `rotateY(-10deg) rotateX(3deg) translate3d(${tx}px, ${ty}px, ${tz}px) scale(0.95)`;
      card.style.opacity = String(Math.max(0.32, 0.72 - absRel * 0.16));
    }
  });

  hudState.textContent = `SLICE ${String(activePos + 1).padStart(2, "0")} / ${String(cards.length).padStart(2, "0")}`;
  if (!zstackScrubber.hidden) {
    zRange.value = String(activePos);
    zAxisLabel.textContent = `Z: ${activePos + 1}/${cards.length}`;
  }
}

function setActiveSlice(citeIndex) {
  applyZStackTransforms(citeIndex);
  requestAnimationFrame(drawLaserOverlay);
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

  const targetCard = zstackStage.querySelector(`.slice-card[data-slice-index="${citeIndex}"]`);
  if (!targetCard) {
    clearLaserOverlay();
    return;
  }

  const aRect = anchor.getBoundingClientRect();
  const cRect = targetCard.getBoundingClientRect();
  const x1 = aRect.right + 4;
  const y1 = aRect.top + aRect.height / 2;
  const x2 = cRect.left + 2;
  const y2 = cRect.top + Math.min(42, cRect.height / 2);

  // 直角折线激光路径：沿右侧留白走廊垂直转折，尽量不遮挡正文阅读区
  const panelRect = zstackStage.getBoundingClientRect();
  const gutterX = Math.max(x1 + 14, Math.min(x2 - 14, panelRect.left - 12));
  const d = `M ${x1.toFixed(1)} ${y1.toFixed(1)} L ${gutterX.toFixed(1)} ${y1.toFixed(1)} L ${gutterX.toFixed(1)} ${y2.toFixed(1)} L ${x2.toFixed(1)} ${y2.toFixed(1)}`;

  const resp = turnResponses[turnId];
  const citeObj = (resp?.citations || []).find((c) => c.index === citeIndex);
  const confText = citeObj ? `CONF ${Number(citeObj.score).toFixed(4)}` : `SLICE #${citeIndex}`;
  const labelX = Math.max(x1 + 10, gutterX - 88);
  const labelY = y2 - 10;

  laserOverlay.innerHTML = `
    <defs>
      <filter id="laserGlow" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="2.5" result="blur" />
        <feMerge>
          <feMergeNode in="blur" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
    </defs>
    <path d="${d}" fill="none" stroke="rgba(34, 211, 238, 0.88)" stroke-width="1.5" filter="url(#laserGlow)" />
    <circle cx="${x1}" cy="${y1}" r="3" fill="#22d3ee" />
    <circle cx="${x2}" cy="${y2}" r="5" fill="none" stroke="#22d3ee" stroke-width="1.5" />
    <circle cx="${x2}" cy="${y2}" r="2" fill="#34d399" />
    <rect x="${labelX - 4}" y="${labelY - 11}" width="84" height="16" rx="3" fill="rgba(5,9,17,0.88)" stroke="rgba(34,211,238,0.45)" stroke-width="0.8" />
    <text x="${labelX}" y="${labelY}" fill="#22d3ee" font-family="JetBrains Mono, monospace" font-size="10">${confText}</text>
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
    renderZStackForTurn(turnId, citeIndex);
  } else {
    setActiveSlice(citeIndex);
  }
  updateAnchorActiveClasses();
  // 等待 3D 过渡布局稳定后再次修正连线端点
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

// 事件委托处理所有动态生成的小引用胶囊 [⊕n]
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
  if (lockedAnchorEl || hoveredAnchorEl) {
    requestAnimationFrame(drawLaserOverlay);
  }
});

window.addEventListener("resize", () => {
  if (lockedAnchorEl || hoveredAnchorEl) {
    requestAnimationFrame(drawLaserOverlay);
  }
});

zRange.addEventListener("input", () => {
  const resp = turnResponses[activeTurnId];
  const citations = resp?.citations || [];
  const idx = Number(zRange.value);
  if (citations[idx]) {
    setActiveSlice(citations[idx].index);
  }
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

  if (emptyWelcomeEl) {
    emptyWelcomeEl.style.display = "none";
  }

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
    <div class="report-label">
      <span>CLINICAL RAG REPORT // ${turnId.toUpperCase()}</span>
    </div>
    <div class="report-content">正在执行双路检索与断层切片分析...</div>
  `;
  const reportBodyEl = reportEl.querySelector(".report-content");
  messagesEl.appendChild(reportEl);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  shader.setMode("scanning", 0.5);
  statusDot.classList.remove("is-refused");
  statusDot.classList.add("is-scanning");
  hudState.classList.remove("is-refused");
  hudState.textContent = "SCANNING K-SPACE";

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
          shader.triggerPulse(0.28);
          messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (ev.type === "done") {
          const resp = ev.response;
          turnResponses[turnId] = resp;
          if (resp.refused) {
            reportEl.classList.add("is-refused");
          }
          reportBodyEl.innerHTML = formatReportBody(resp.answer, turnId, resp.disclaimer);
          updateFov(resp);
          renderZStackForTurn(turnId);
          chatHistory.push({ role: "user", content: query });
          chatHistory.push({ role: "assistant", content: resp.answer });
          messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (ev.type === "error") {
          reportBodyEl.textContent = `服务异常：${ev.message}`;
          shader.setMode("refused", 0.0);
        }
      }
    }
  } catch (err) {
    reportBodyEl.textContent = `请求失败：${err}`;
    shader.setMode("refused", 0.0);
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
  activeSliceIndex = null;
  lockedAnchorEl = null;
  hoveredAnchorEl = null;
  clearLaserOverlay();
  messagesEl.innerHTML = "";
  if (emptyWelcomeEl) {
    emptyWelcomeEl.style.display = "block";
    messagesEl.appendChild(emptyWelcomeEl);
  }
  shader.setMode("idle", 0.0);
  statusDot.classList.remove("is-scanning", "is-refused");
  hudState.classList.remove("is-refused");
  hudState.textContent = "STANDING BY";
  zstackScrubber.hidden = true;
  zstackStage.innerHTML = `
    <div class="zstack-empty">
      <div class="code-line">WAITING FOR SIGNAL // 0 SLICES</div>
      <p>发起医疗问答后，召回证据将以 3D 断层胶片呈现于此，支持悬浮引用锚点激光联动。</p>
    </div>
  `;
  fovText.textContent = `FOV: 0 / ${maxContextTokens.toLocaleString()} TOKENS`;
  fovFill.style.width = "0%";
});

loadHealth();
