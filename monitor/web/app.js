// NInfer Monitor — web/app.js
// 渲染逻辑与 Strata 的 Monitor 视图同源（同样的卡片/网格/表格/火花线），
// 数据源换成 NInfer 自己的聚合接口 /api/monitor（由 server/ninfer-monitor.py 提供）。
// 布局与 Strata 完全一致，只有配色（tokens.css）与数据语义换了。
"use strict";

const $ = (id) => document.getElementById(id);
const SPRITE = "web/sprite.svg";
const icon = (name, cls = "st-icon") => `<svg class="${cls}" aria-hidden="true"><use href="${SPRITE}#i-${name}"/></svg>`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const kfmt = (n) => (n == null ? "–" : n >= 1000 ? `${fmt(n / 1000, n >= 10000 ? 0 : 1)}k` : fmt(n));
const ctxfmt = (n) => (n && n % 1024 === 0 ? `${fmt(n / 1024)}K` : kfmt(n));
const gb = (b, d = 1) => (b == null ? "–" : fmt(b / 1073741824, d));   // 内存用二进制 GB

// ------------------------------------------------------------------ theme
function setTheme(t, save) {
  document.documentElement.dataset.theme = t;
  if (save) try { localStorage.setItem("ninfer.theme", t); } catch (e) { /* ignore */ }
  $("theme-icon").setAttribute("href", `${SPRITE}#i-${t === "dark" ? "sun" : "moon"}`);
}
const flipTheme = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
$("theme-btn").onclick = flipTheme;
setTheme(document.documentElement.dataset.theme || "dark", false);

// ------------------------------------------------------------------ Monitor
const METRICS = [
  {key: "speed", label: "Speed", icon: "gauge", unit: "t/s", series: "tok_s"},
  {key: "gpu", label: "GPU load", icon: "gpu", unit: "%", series: "gpu_util", max: 100},
  {key: "vram", label: "VRAM", icon: "layers", unit: "GB", series: "gpu_mem_used"},
  {key: "temp", label: "GPU temp", icon: "thermometer", unit: "°C", series: "gpu_temp", tone: "warn"},
  {key: "power", label: "Power", icon: "bolt", unit: "W", series: "gpu_power"},
  {key: "pcie", label: "PCIe", icon: "link", unit: "", series: "gpu_pcie_rx_mb", tone: "info"},
  {key: "cpu", label: "CPU", icon: "cpu", unit: "%", series: "cpu", max: 100},
  {key: "disk", label: "Disk read", icon: "disk", unit: "MB/s", series: "disk_read_mb", tone: "info"},
];
$("metrics").innerHTML = METRICS.map((m) => `
  <div class="st-card metric-card"><div class="st-metric">
    <span class="st-metric__label">${icon(m.icon, "st-icon st-icon--sm")}${esc(m.label)}</span>
    ${m.key === "speed" ? `<div class="speed-values">
      <div><span class="st-metric__value" id="mv-speed">-</span><span class="st-metric__sub" id="ms-speed">Decode</span></div>
      <div class="speed-prefill"><span class="st-metric__value" id="mv-prefill">-</span><span class="st-metric__sub" id="ms-prefill">Prefill</span></div>
    </div>` : `<span class="st-metric__value" id="mv-${m.key}">–</span>
    <span class="st-metric__sub" id="ms-${m.key}"></span>`}
    <svg class="st-metric__spark" id="sp-${m.key}" viewBox="0 0 100 32" preserveAspectRatio="none"${m.tone ? ` data-tone="${m.tone}"` : ""}>
      <path class="area" fill="currentColor" opacity=".12"/><path class="line" fill="none" stroke="currentColor"
      stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
      ${m.key === "speed" ? `<g id="sp-prefill" class="speed-prefill"><path class="area" fill="currentColor" opacity=".12"/>
        <path class="line" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"
        stroke-linecap="round" vector-effect="non-scaling-stroke"/></g>` : ""}</svg>
  </div></div>`).join("");

function spark(id, values, max) {
  const svg = $(id);
  const v = (values || []).map((x) => (x == null ? 0 : x));
  if (v.length < 2) { svg.querySelector(".line").setAttribute("d", ""); svg.querySelector(".area").setAttribute("d", ""); return; }
  const top = Math.max(max || 0, ...v, 1e-9);
  const pts = v.map((x, i) => [(i / (v.length - 1)) * 100, 30 - (x / top) * 26]);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join("");
  svg.querySelector(".line").setAttribute("d", line);
  svg.querySelector(".area").setAttribute("d", `${line}L100,32L0,32Z`);
}
function setMetric(key, value, unit, sub) {
  $(`mv-${key}`).innerHTML = value == null ? "–" : `${esc(value)}${unit ? `<small>${esc(unit)}</small>` : ""}`;
  $(`ms-${key}`).textContent = sub || "";
}
function setPill(state, text) {
  $("pill").dataset.state = state === "error" ? "queued" : state;
  $("pill-text").textContent = text;
}
function facts(el, rows) {
  el.innerHTML = rows.filter((r) => r[1] != null && r[1] !== "").map(([k, v, copy]) =>
    `<dt>${esc(k)}</dt><dd>${copy ? `<code>${esc(v)}</code><button class="st-btn st-btn--icon" data-copy="${esc(v)}" aria-label="Copy">${icon("copy")}</button>` : esc(v)}</dd>`).join("");
}
async function copyText(text) {
  try { await navigator.clipboard.writeText(text); }
  catch (e) {
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove();
  }
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-copy]");
  if (b) copyText(b.dataset.copy);
});

let lastMetrics = null, metricsFailures = 0, reqShowAll = false;

// 与 Strata 同款：请求表尾部统计行
function renderTotals(t) {
  if (!t || !t.requests) return "";
  const since = t.since ? new Date(t.since * 1000).toLocaleString([], {weekday: "short", hour: "2-digit", minute: "2-digit"}) : null;
  const read = (t.prompt_tokens || 0) - (t.reused || 0);
  const oSpeed = t.decode_ms > 0 && t.output_tokens > 0 ? ` at ${fmt(t.output_tokens / (t.decode_ms / 1000), 1)} tok/s` : "";
  const head = since ? `Since ${since}: ` : "";
  return `${head}${fmt(t.requests)} requests · ${fmt(read)} prompt tokens read (${fmt(t.reused)} reused) · ` +
         `${fmt(t.output_tokens)} written${oSpeed}` +
         (t.accepted != null && t.drafted ? ` · MTP accepted ${fmt(100 * t.accepted / t.drafted, 1)}%` : "");
}

function renderMonitor(live, hw, st, eng, h, last, requests, totals, kept, ctx) {
  // 表里最新的一条可能是「传输途中取消」（没有读数）⇒ 取最近一条有完整读数的请求做兜底
  const done = (requests || []).find((r) => r.output_tokens != null && r.decode_tok_s != null) || null;
  // ---- model state
  const on = live.queued > 0 ? "queued" : live.state;
  for (const b of document.querySelectorAll("#state-badges .st-badge")) b.classList.toggle("on", b.dataset.s === on || b.dataset.s === live.state);
  const prog = $("state-progress");
  let label = "Waiting for a request", detail = "", pct = 0;
  if (live.state === "reading") {
    label = "Reading prompt";
    prog.dataset.tone = "info";
    detail = live.prefill_tok_s_mean != null ? `${fmt(live.prefill_tok_s_mean)} tok/s` : "prefill";
  } else if (live.state === "generating") {
    label = "Generating";
    delete prog.dataset.tone;
    detail = live.tok_s != null ? `${fmt(live.tok_s, 1)} tok/s` : "";
  } else if (done) {
    delete prog.dataset.tone;
    detail = `last: ${fmt(done.output_tokens)} tokens${done.decode_tok_s ? ` at ${fmt(done.decode_tok_s, 1)} tok/s` : ""}`;
  }
  $("state-label").textContent = label;
  $("state-detail").textContent = detail;
  $("state-bar").style.width = `${pct}%`;

  // ---- the eight cards
  const speed = live.state === "generating" ? live.tok_s : done ? done.decode_tok_s : null;
  setMetric("speed", speed == null ? null : fmt(speed, 1), "t/s",
            live.state === "generating" ? "Decode now" : done ? "Decode last request" : "Decode");
  const prefill = live.state === "reading" ? live.prefill_tok_s_mean : done && done.prefill_tok_s != null ? done.prefill_tok_s : null;
  setMetric("prefill", prefill == null ? null : fmt(prefill), "t/s",
            live.state === "reading" ? "Prefill now" : done && done.prefill_tok_s != null ? "Prefill last request" : "Prefill");
  spark("sp-speed", h.tok_s);
  spark("sp-prefill", h.prefill_tok_s_mean);
  setMetric("gpu", hw.gpu_util == null ? null : fmt(hw.gpu_util), "%", st.gpu_name || (st.gpu_note ? "not available" : ""));
  spark("sp-gpu", h.gpu_util, 100);
  setMetric("vram", hw.gpu_mem_used == null ? null : gb(hw.gpu_mem_used), hw.gpu_mem_total ? `/ ${gb(hw.gpu_mem_total, 0)} GB` : "GB",
            eng.max_context ? `${ctxfmt(eng.max_context)} ctx${eng.kv ? ` · ${eng.kv}` : ""}` : "");
  spark("sp-vram", h.gpu_mem_used, hw.gpu_mem_total);
  setMetric("temp", hw.gpu_temp == null ? null : fmt(hw.gpu_temp), "°C", "");
  spark("sp-temp", h.gpu_temp, 90);
  setMetric("power", hw.gpu_power == null ? null : fmt(hw.gpu_power), "W", hw.gpu_power_limit ? `of ${fmt(hw.gpu_power_limit)} W limit` : "");
  spark("sp-power", h.gpu_power, hw.gpu_power_limit);
  const gen = hw.gpu_pcie_gen_max || hw.gpu_pcie_gen;
  setMetric("pcie", gen ? `Gen${gen}` : null, hw.gpu_pcie_width ? `x${hw.gpu_pcie_width}` : "",
            hw.gpu_pcie_rx_mb == null ? "" : `to GPU ${fmt(hw.gpu_pcie_rx_mb, hw.gpu_pcie_rx_mb < 10 ? 1 : 0)} MB/s` +
            (hw.gpu_pcie_gen && gen && hw.gpu_pcie_gen < gen ? ` · idle Gen${hw.gpu_pcie_gen}` : ""));
  spark("sp-pcie", h.gpu_pcie_rx_mb);
  setMetric("cpu", hw.cpu == null ? null : fmt(hw.cpu), "%", st.threads ? `${st.cores ? `${st.cores} cores · ` : ""}${st.threads} threads` : "");
  spark("sp-cpu", h.cpu, 100);
  if (hw.disk_read_mb == null) {
    setMetric("disk", null, "", st.disk_dev ? `${st.disk_dev}: not readable` : "");
  } else {
    const big = hw.disk_read_mb >= 1000;
    setMetric("disk", big ? fmt(hw.disk_read_mb / 1024, 2) : fmt(hw.disk_read_mb, hw.disk_read_mb < 10 ? 1 : 0), big ? "GB/s" : "MB/s",
              hw.disk_write_mb == null ? "" : `write ${fmt(hw.disk_write_mb, 1)} MB/s`);
  }
  spark("sp-disk", h.disk_read_mb);

  // ---- context fill：KV 池占用（NInfer 的 /metrics 直接给 kv_cache_tokens / 占用比）
  const maxCtx = eng.max_context || 0;
  const used = ctx && ctx.used_tokens != null ? ctx.used_tokens : 0;
  const frac = ctx && ctx.ratio != null ? Math.min(1, ctx.ratio) : (maxCtx ? Math.min(1, used / maxCtx) : 0);
  $("ctx-fill").setAttribute("stroke-dasharray", `${(235.6 * frac).toFixed(1)} 314.2`);
  $("ctx-fill").style.opacity = 235.6 * frac >= 3 ? "1" : "0";
  $("ctx-pct").textContent = `${Math.round(frac * 100)}%`;
  $("ctx-sub").textContent = maxCtx ? `${kfmt(used)} / ${ctxfmt(maxCtx)}` : "–";
  // bar 1：模型权重在显存里的占比（对应 Strata 的「Experts in VRAM」那一条）
  const wBytes = eng.weights_bytes || 0;
  $("slots-text").textContent = wBytes && hw.gpu_mem_total ? `${gb(wBytes)} / ${gb(hw.gpu_mem_total, 0)} GB` : "–";
  $("slots-bar").style.width = hw.gpu_mem_total ? `${Math.min(100, (100 * wBytes) / hw.gpu_mem_total)}%` : "0%";
  $("ram-text").textContent = hw.ram_total ? `${gb(hw.ram_used)} / ${gb(hw.ram_total, 0)} GB` : "–";
  const ramPct = hw.ram_total ? (100 * hw.ram_used) / hw.ram_total : 0;
  $("ram-bar").style.width = `${ramPct}%`;
  if (ramPct > 92) $("ram-progress").dataset.tone = "danger"; else delete $("ram-progress").dataset.tone;
  $("temp-text").textContent = hw.gpu_temp == null ? "–" : `${fmt(hw.gpu_temp)} °C`;
  $("temp-bar").style.width = hw.gpu_temp == null ? "0%" : `${Math.min(100, hw.gpu_temp)}%`;

  // ---- recent requests
  const body = $("req-body");
  if (!requests.length) {
    body.innerHTML = `<tr><td colspan="8" class="muted">No requests yet</td></tr>`;
  } else {
    const badge = {stop: ["", "Done"], length: ["", "Max tokens"], cancel: ["st-badge--queued", "Stopped"],
                   error: ["st-badge--error", "Error"]};
    body.innerHTML = requests.slice(0, reqShowAll ? requests.length : 12).map((r) => {
      const [cls, text] = badge[r.finish] || ["", r.finish || "–"];
      const t = new Date(r.time * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
      const hit = r.hit_rate == null ? "–" : `${r.hit_rate.toFixed(1)}%` +
        (r.drafts ? ` <span class="muted" title="speculative drafts accepted for this request (MTP)">MTP ${(100 * r.accepted / r.drafts).toFixed(1)}%</span>` : "");
      return `<tr><td>${esc(t)}</td><td><span class="st-badge ${cls}" title="${esc(r.reason || text)}">${esc(text)}</span></td><td class="num">${fmt(r.prompt_tokens)}</td>
        <td class="num">${fmt(r.reused)}</td><td class="num">${fmt(r.output_tokens)}</td><td class="num">${fmt(r.decode_tok_s, 1)}</td>
        <td class="num">${hit}</td><td class="num">${fmt(r.duration_s, 1)} s</td></tr>`;
    }).join("");
  }
  const all = $("req-all");
  kept = kept == null ? requests.length : kept;
  all.hidden = kept <= 12;
  all.textContent = reqShowAll ? "Show fewer" : `Show all (${kept})`;
  $("req-wrap").classList.toggle("all", reqShowAll);
  $("req-totals").textContent = renderTotals(totals);
}

// NInfer 的上下文缓存：设备态槽位 + 宿主 KV；实时占用没有计数接口 ⇒ 只列事实，不画两条进度条
function renderConvCache(c) {
  $("cc-card").hidden = !c;
  if (!c) return;
  $("cc-bars").hidden = true;
  $("cc-sum").textContent = c.requests ? `${fmt(c.requests_reused)} of ${fmt(c.requests)} requests reused part of their prompt` : "";
  const share = c.prompt_tokens ? ` (${fmt((100 * c.reused_tokens) / c.prompt_tokens)}% of all prompt tokens)` : "";
  facts($("cc-facts"), [
    ["Last request", c.last_prompt != null ? `${fmt(c.last_reused || 0)} of ${fmt(c.last_prompt)} prompt tokens reused` : null],
    ["Reused since start", c.reused_tokens ? `${fmt(c.reused_tokens)} tokens${share}` : null],
    ["Prefix cache hits", c.prefix_hits ? `${fmt(c.prefix_hits)} tokens` : null],
    ["Device states", c.device_states || null],
    ["Host KV", c.host_kv || null],
    ["Policy", c.policy || null],
    ["Placement failures", c.exhausted ? `${fmt(c.exhausted)} requests (context cache full)` : null],
  ]);
  $("cc-note").textContent = c.note || "";
}

function render(m) {
  const live = m.live || {}, hw = m.hardware || {}, st = m.hardware_static || {}, eng = m.engine || {}, h = m.history || {};
  const last = (m.requests || [])[0];
  if (eng.up === false) {
    setPill("error", "Engine not reachable");            // ninfer 挂了/重启中：宁可说清楚，也别装作 Idle
  } else if (live.state === "reading") {
    setPill("reading", live.prefill_tok_s_mean != null ? `Reading prompt · ${fmt(live.prefill_tok_s_mean)} tok/s` : "Reading prompt");
  } else if (live.state === "generating") {
    setPill("generating", `Generating · ${fmt(live.tok_s, 1)} tok/s`);
  } else {
    setPill("idle", "Idle");
  }
  if (live.queued > 0) setPill("queued", `${live.queued} queued`);
  renderMonitor(live, hw, st, eng, h, last, m.requests || [], m.totals, m.requests_kept, m.context);
  renderConvCache(m.context_cache);
}

async function poll() {
  try {
    const r = await fetch(`api/monitor?requests=${reqShowAll ? "all" : "12"}`, {cache: "no-store"});
    if (r.ok) {
      lastMetrics = await r.json();
      metricsFailures = 0;
      render(lastMetrics);
    } else {
      throw new Error(`HTTP ${r.status}`);
    }
  } catch (e) {
    if (++metricsFailures === 3) setPill("error", "Server not reachable");
  }
  setTimeout(poll, 1000);
}
$("req-all").addEventListener("click", () => { reqShowAll = !reqShowAll; if (lastMetrics) render(lastMetrics); });
poll();
