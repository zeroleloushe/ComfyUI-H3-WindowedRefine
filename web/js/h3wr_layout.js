// Pure layout math, mirror of h3_windows.plan_windows (video part). No ComfyUI imports:
// this file is also imported by tests/test_js_layout.py through node.js.
export const FPT = [1, 4, 4, 4, 4];
export const MIN_TRAINED_T = 37; // 124 frames

export function framesFor(t) { let s = 0; for (let k = 0; k < t; k++) s += FPT[k % 5]; return s; }
export function latentFor(frames) { return frames <= 5 ? 2 : Math.floor((frames - 5) / 17) * 5 + 2; }

// Python's round(): banker's rounding (half to even)
export function pyRound(x) {
  const f = Math.floor(x), d = x - f;
  if (Math.abs(d - 0.5) < 1e-9) return f % 2 === 0 ? f : f + 1;
  return Math.round(x);
}

export function nearestValidFrames(frames) {
  frames = Math.max(5, Math.floor(frames));
  const lower = frames - ((frames - 5) % 17), higher = (frames - 5) % 17 === 0 ? frames : lower + 17;
  const nearest = frames - lower <= higher - frames ? lower : higher;
  return { valid: (frames - 5) % 17 === 0, lower, higher, nearest };
}

// returns {windows: [[v0,v1],...]} or {error: "..."} exactly where Python raises
export function planWindows(T, nw, wb, ob) {
  nw = nw | 0; wb = wb | 0;
  if (T < 2 || (T - 2) % 5 !== 0) return { error: `T=${T} не вида 5k+2` };
  const k = (T - 2) / 5, o = Math.max(0, ob | 0);
  let m;
  if (nw && nw > 1) m = Math.ceil((k + o * (nw - 1)) / nw);
  else if (wb && wb > 0) m = wb;
  else m = k;
  m = k > 0 ? Math.max(1, Math.min(m, k)) : 0;
  if (o >= m && m < k) return { error: `перекрытие (${o}) должно быть меньше длины окна (${m})` };
  const L = 5 * m + 2, last = k - m;
  let starts;
  if (last <= 0) starts = [0];
  else {
    const stride = m - o;
    let n = nw && nw > 1 ? nw : Math.ceil(last / stride) + 1;
    n = Math.max(2, Math.min(n, last + 1));
    const set = new Set();
    for (let i = 0; i < n; i++) set.add(pyRound(i * last / (n - 1)));
    starts = [...set].sort((a, b) => a - b);
  }
  const w = starts.map((s) => [5 * s, 5 * s + L]);
  if (w.length === 1) w[0] = [0, T]; else w[w.length - 1][1] = T;
  return { windows: w };
}

// human-readable layout for the panel
export function describeLayout(frames, nw, wb, ob, fps = 24) {
  const nv = nearestValidFrames(frames);
  const used = nv.nearest, T = latentFor(used);
  const r = planWindows(T, nw, wb, ob);
  const out = { frames, used, nv, T, error: r.error || null, rows: [], warnings: [] };
  if (!nv.valid) out.warnings.push(`${frames} кадров не 17k+5: ближайшие ${nv.lower} / ${nv.higher}, взято ${used}`);
  if (r.error) return out;
  r.windows.forEach(([a, b], i) => {
    const f0 = framesFor(a), f1 = framesFor(b) - 1;
    const ov = i > 0 ? r.windows[i - 1][1] - a : 0;
    out.rows.push({ i: i + 1, v0: a, v1: b, len: b - a, f0, f1, s0: f0 / fps, s1: (f1 + 1) / fps, ov });
  });
  if (r.windows.length > 1 && r.windows.some(([a, b]) => b - a < MIN_TRAINED_T))
    out.warnings.push(`окно короче T=${MIN_TRAINED_T} (124 кадра) — вне обучающего диапазона H3`);
  return out;
}
