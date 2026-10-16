// MiniMax H3 Windowed Latent Refine: UI polish. Everything is best-effort and wrapped in try/catch
// so a different frontend version degrades to the plain node instead of breaking it.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { framesFor, latentFor, planWindows as planWindowsRaw, describeLayout } from "./h3wr_layout.js";
const planWindows = (T, nw, wb, ob) => planWindowsRaw(T, nw, wb, ob).windows || null;

const NODE = "H3WindowedLatentRefine";
const EXTRA_H = 100;
const THEME = { color: "#2b3a55", bgcolor: "#1b2333", accent: "#6aa9ff", win: ["#4c8dff", "#ff9f43", "#2ecc71", "#e056fd", "#f1c40f", "#00cec9"] };
const GROUPS = [
  ["Вход", "prompt"], ["Окна", "num_windows"], ["Согласованность", "alternate_direction"],
  ["Якорь", "anchor_strength"], ["Выход", "decode"], ["Превью", "preview_frames"],
];
const LABELS = {
  prompt: "промпт", negative_prompt: "негатив", seed: "сид", steps: "шаги", cfg: "cfg",
  sampler_name: "сэмплер", scheduler: "расписание", shift_video: "shift видео", shift_audio: "shift аудио",
  denoise: "denoise", num_windows: "окон", window_blocks: "длина окна (m)", overlap_blocks: "перекрытие (o)",
  alternate_direction: "чередовать 1→N / N→1", temporal_guide: "guide от соседа", guide_from_step: "guide с шага",
  anchor_strength: "сила якоря", anchor_steps: "доля шагов с якорем", decode: "декодировать IMAGE", verbose: "подробный лог",
  preview_frames: "кадров (превью)",
};

const W = (node, name) => node.widgets?.find((w) => w.name === name);
const val = (node, name, d) => { const w = W(node, name); return w && w.value !== undefined ? w.value : d; };

function fmtT(s) { if (!isFinite(s)) return "–"; s = Math.max(0, s); return s >= 60 ? `${Math.floor(s / 60)}m${(s % 60).toFixed(0).padStart(2, "0")}s` : `${s.toFixed(1)}s`; }

function setup(node) {
  try {
    node.color = THEME.color; node.bgcolor = THEME.bgcolor;
    node.h3wr = { T: null, frames: null, state: "idle", step: -1, steps: 0, win: -1, order: [], eta: null, elapsed: 0, guide: false, anchor: 0 };
    for (const w of node.widgets || []) if (LABELS[w.name]) { try { w.label = LABELS[w.name]; } catch (e) {} }
    const panel = {
      name: "h3wr_panel", type: "h3wr_panel", value: null, serialize: false, options: { serialize: false },
      computeSize: (width) => [width, panelHeight(node)],
      draw(ctx, n, width, y) { try { drawPanel(n, ctx, y, width); } catch (e) {} },
      serializeValue: () => undefined,
    };
    try {
      const btn = node.addWidget("button", "Рассчитать окна", null, () => recompute(node, true), { serialize: false });
      if (btn) { btn.serialize = false; btn.label = "▶ Рассчитать окна"; }
    } catch (e) {}
    if (node.addCustomWidget) node.addCustomWidget(panel); else node.widgets?.push(panel);
    for (const name of ["num_windows", "window_blocks", "overlap_blocks", "preview_frames"]) {
      const w = W(node, name); if (!w) continue;
      const cb = w.callback;
      w.callback = function () { const r = cb?.apply(this, arguments); try { if (val(node, "preview_frames", 0) > 0) recompute(node, false); } catch (e) {} return r; };
    }
    const sz = node.computeSize?.(); if (sz) node.setSize?.([Math.max(sz[0], 360), sz[1]]);
  } catch (e) { console.warn("[H3WR] ui setup failed", e); }
}

function drawSections(node, ctx) {
  // vertical accent bar + rotated label in the left margin for each widget group
  const ys = GROUPS.map(([, first]) => { const w = W(node, first); return w ? (w.last_y ?? w.y) : null; });
  const panel = W(node, "h3wr_panel");
  const endY = panel ? (panel.last_y ?? panel.y) : node.size[1];
  GROUPS.forEach(([title], i) => {
    const y0 = ys[i]; if (!y0) return;
    let y1 = endY; for (let j = i + 1; j < ys.length; j++) if (ys[j]) { y1 = ys[j]; break; }
    y1 -= 4;
    ctx.save();
    ctx.fillStyle = THEME.win[i % THEME.win.length]; ctx.globalAlpha = 0.85;
    ctx.fillRect(3, y0, 3, Math.max(4, y1 - y0));
    if (y1 - y0 > 30) {
      ctx.translate(13, (y0 + y1) / 2); ctx.rotate(-Math.PI / 2);
      ctx.font = "bold 8px sans-serif"; ctx.textAlign = "center"; ctx.globalAlpha = 0.9;
      ctx.fillText(title.toUpperCase(), 0, 0);
    }
    ctx.restore();
  });
}

function layoutFor(node) {
  const st = node.h3wr || {};
  const pf = val(node, "preview_frames", 0) | 0;
  const frames = pf > 0 ? pf : (st.T ? framesFor(st.T) : 124);
  const src = pf > 0 ? "превью" : (st.T ? "последний запуск" : "пример");
  const L = describeLayout(frames, val(node, "num_windows", 2), val(node, "window_blocks", 0), val(node, "overlap_blocks", 1));
  L.src = src;
  return L;
}

function panelHeight(node) {
  let L; try { L = layoutFor(node); } catch (e) { return EXTRA_H; }
  return 80 + Math.max(1, L.rows.length) * 13 + (L.error ? 13 : 0) + L.warnings.length * 13 + 30;
}

function recompute(node, force) {
  try {
    const st = node.h3wr; if (!st) return;
    st.layout = layoutFor(node);
    const sz = node.computeSize?.();
    if (sz) node.setSize?.([Math.max(node.size[0], sz[0], 360), sz[1]]);
    node.setDirtyCanvas?.(true, true);
    if (force) console.info("[H3WR] layout", st.layout);
  } catch (e) { console.warn("[H3WR] layout failed", e); }
}

function fit(ctx, text, w) {
  if (ctx.measureText(text).width <= w) return text;
  while (text.length > 4 && ctx.measureText(text + "…").width > w) text = text.slice(0, -1);
  return text + "…";
}

function drawPanel(node, ctx, py, width) {
  const st = node.h3wr; if (!st) return;
  const L = layoutFor(node);
  const H = panelHeight(node);
  const x0 = 14, x1 = (width || node.size[0]) - 14, wd = x1 - x0, yEnd = py + H;
  let y = py + 14;
  ctx.save();
  ctx.fillStyle = "rgba(0,0,0,0.25)"; ctx.fillRect(x0 - 4, py + 2, wd + 8, H - 6);
  ctx.font = "10px sans-serif"; ctx.textAlign = "left"; ctx.fillStyle = "#9fb3d1";
  ctx.fillText(fit(ctx, `ОКНА (${L.src}) · ${L.used} кадров = ${(L.used / 24).toFixed(2)}s · T=${L.T}`, wd), x0, y);
  y += 6;
  const T = L.T;
  if (L.error) {
    y += 12; ctx.fillStyle = "#ff7675"; ctx.fillText(fit(ctx, "ошибка: " + L.error, wd), x0, y);
  } else {
    const n = L.rows.length, bh = Math.max(4, Math.min(9, 36 / n - 2));
    L.rows.forEach((r, i) => {
      const yy = y + i * (bh + 2);
      ctx.fillStyle = THEME.win[i % THEME.win.length];
      ctx.globalAlpha = st.state === "run" && st.win === i ? 1 : 0.75;
      ctx.fillRect(x0 + (r.v0 / T) * wd, yy, (r.len / T) * wd, bh);
      if (r.ov > 0) { ctx.globalAlpha = 0.35; ctx.fillStyle = "#fff"; ctx.fillRect(x0 + (r.v0 / T) * wd, y, (r.ov / T) * wd, (i + 1) * (bh + 2) - 2); }
      ctx.globalAlpha = 1;
    });
    y += n * (bh + 2) + 2;
    ctx.font = "10px monospace";
    L.rows.forEach((r, i) => {
      y += 13;
      ctx.fillStyle = THEME.win[i % THEME.win.length];
      const t = `#${r.i} lat ${r.v0}–${r.v1 - 1} (${r.len}) · кадры ${r.f0}–${r.f1} · ${r.s0.toFixed(2)}–${r.s1.toFixed(2)}s` + (r.ov ? ` · ovl ${r.ov}` : "");
      ctx.fillText(fit(ctx, t, wd), x0, y);
    });
    ctx.font = "10px sans-serif";
  }
  for (const w of L.warnings) { y += 13; ctx.fillStyle = "#ffb86b"; ctx.fillText(fit(ctx, "⚠ " + w, wd), x0, y); }
  // reference mapping (<Picture N> = input)
  ctx.fillStyle = "#c7d7ee";
  const rl = st.refs ? (st.refs.length ? st.refs.map((l) => l.replace(/^<Picture (\d+)> = /, "P$1=")).join("  ") : "референсов нет")
                     : "референсы: таблица <Picture N> появится после запуска";
  ctx.fillText(fit(ctx, rl, wd), x0, yEnd - 24);
  let s = "ожидание";
  if (st.state === "run") s = `шаг ${st.step + 1}/${st.steps} · окно ${st.win + 1}${st.order.length ? " · " + (st.order[0] === 0 ? "1→N" : "N→1") : ""}${st.guide ? " · guide" : ""}${st.anchor > 0 ? " · якорь " + st.anchor.toFixed(2) : ""} · ${fmtT(st.elapsed)} · ETA ${fmtT(st.eta)}`;
  else if (st.state === "done") s = `готово за ${fmtT(st.elapsed)}`;
  ctx.fillStyle = st.state === "run" ? "#ffd166" : "#7f8fa6";
  ctx.fillText(fit(ctx, s, wd), x0, yEnd - 10);
  if (st.state === "run" && st.steps) {
    const nw = Math.max(1, L.rows.length), f = Math.min(1, (st.step * nw + (st.pos || 0) + 1) / (st.steps * nw));
    ctx.fillStyle = THEME.accent; ctx.fillRect(x0, yEnd - 5, wd * f, 2);
  }
  ctx.restore();
}

app.registerExtension({
  name: "h3wr.windowed_refine",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      setup(this);
      return r;
    };
    const onDraw = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      const r = onDraw?.apply(this, arguments);
      if (this.flags?.collapsed) return r;
      try { drawSections(this, ctx); } catch (e) { /* never break canvas drawing */ }
      return r;
    };
  },
  async setup() {
    try {
      api.addEventListener("h3wr.progress", ({ detail: d }) => {
        try {
          const node = app.graph?.getNodeById?.(Number(d.node)) ?? app.graph?.getNodeById?.(d.node);
          if (!node?.h3wr) return;
          const st = node.h3wr;
          if (d.event === "refs") { st.refs = d.map || []; node.setDirtyCanvas?.(true, true); return; }
          if (d.event === "summary") setTimeout(() => recompute(node, false), 0);
          if (d.event === "summary") Object.assign(st, { T: d.T, frames: d.frames, steps: d.steps, state: "run", step: 0, win: -1, pos: 0 });
          else if (d.event === "step") Object.assign(st, { state: "run", step: d.step, steps: d.steps, order: d.order, guide: d.guide, anchor: d.anchor });
          else if (d.event === "window") Object.assign(st, { state: "run", step: d.step, win: d.window, pos: d.pos, elapsed: d.elapsed, eta: d.eta });
          else if (d.event === "done") Object.assign(st, { state: "done", elapsed: d.total });
          node.setDirtyCanvas?.(true, true);
        } catch (e) {}
      });
    } catch (e) { console.warn("[H3WR] progress listener unavailable", e); }
  },
});
