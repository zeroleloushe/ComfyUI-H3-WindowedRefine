"""Pure window logic + the per-step windowed model wrapper for MiniMax H3.

Facts this relies on (ComfyUI master, comfy/ldm/minimax + comfy_extras/nodes_minimax_h3.py):
- video latent [1,24,T,H/16,W/16], T = 5k+2 for 17k+5 pixel frames;
- FRAME_PER_TOKEN = (1,4,4,4,4) repeats every 5 latent frames (VAE encodes 17-frame clips
  independently), so latent index 5j is a standalone single-frame token at pixel frame 17j;
- temporal RoPE position of latent i = cursor + 5/3 * sum(FRAME_PER_TOKEN[k%5] for k<i);
  windows starting at a multiple of 5 therefore see exactly the trained position pattern;
- MiniMaxH3Model._forward rebuilds PackedLayout when the cached signature does not match,
  so a per-window payload only needs keyframes/refs/cond latents, never a layout;
- audio latent length = round(frames / 24 * 40) (5/3 per pixel frame).
"""
import math
import time
from dataclasses import dataclass

import torch

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
AUDIO_PER_FRAME = 5.0 / 3.0


def frames_for_latent_t(t):
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(t))


def audio_len_for_frames(frames):
    return round(frames / 24.0 * 40.0)


@dataclass
class Window:
    v0: int
    v1: int
    a0: int
    a1: int


def plan_windows(latent_t, audio_t, num_windows=0, window_blocks=0, overlap_blocks=1):
    """Windows over a 5k+2 latent. A window has 5*window_blocks+2 latent frames and starts at a
    multiple of 5; neighbours overlap by at least 5*overlap_blocks+2 latent frames.
    Give either num_windows (>0) or window_blocks (>0)."""
    if latent_t < 2 or (latent_t - 2) % 5 != 0:
        raise ValueError(f"H3 video latent length must be 5k+2, got {latent_t}")
    k = (latent_t - 2) // 5
    o = max(0, int(overlap_blocks))
    if num_windows and num_windows > 1:
        m = math.ceil((k + o * (num_windows - 1)) / num_windows)
    elif window_blocks and window_blocks > 0:
        m = int(window_blocks)
    else:
        m = k
    m = max(1, min(m, k)) if k > 0 else 0
    if o >= m and m < k:
        raise ValueError(f"overlap_blocks ({o}) must be smaller than window_blocks ({m})")
    L = 5 * m + 2
    last = k - m  # last start in blocks
    if last <= 0:
        starts = [0]
    else:
        stride = m - o
        n = num_windows if (num_windows and num_windows > 1) else math.ceil(last / stride) + 1
        n = max(2, min(n, last + 1))
        starts = sorted({round(i * last / (n - 1)) for i in range(n)})
    wins = []
    alen = min(audio_t, audio_len_for_frames(frames_for_latent_t(L)))
    for s in starts:
        v0 = 5 * s
        a0 = min(round(17 * s * AUDIO_PER_FRAME), audio_t - alen)
        wins.append(Window(v0, v0 + L, max(0, a0), max(0, a0) + alen))
    if len(wins) == 1:
        wins[0] = Window(0, latent_t, 0, audio_t)
    else:
        wins[-1] = Window(wins[-1].v0, latent_t, wins[-1].a0, audio_t)
    return wins


def _ramp_1d(n, lo, hi, prev_end, next_start):
    """Weights for a window covering [lo,hi): linear ramps inside overlaps with neighbours."""
    w = torch.ones(hi - lo, dtype=torch.float32)
    if prev_end is not None and prev_end > lo:
        ov = min(prev_end, hi) - lo
        w[:ov] = torch.minimum(w[:ov], torch.arange(1, ov + 1, dtype=torch.float32) / (ov + 1))
    if next_start is not None and next_start < hi:
        ov = hi - max(next_start, lo)
        w[-ov:] = torch.minimum(w[-ov:], torch.arange(ov, 0, -1, dtype=torch.float32) / (ov + 1))
    return w


def window_weights(wins):
    """Per-window (video_weights[T_w], audio_weights[A_w])."""
    out = []
    for i, w in enumerate(wins):
        p = wins[i - 1] if i > 0 else None
        nx = wins[i + 1] if i + 1 < len(wins) else None
        vw = _ramp_1d(None, w.v0, w.v1, p.v1 if p else None, nx.v0 if nx else None)
        aw = _ramp_1d(None, w.a0, w.a1, p.a1 if p else None, nx.a0 if nx else None)
        out.append((vw, aw))
    return out


def guide_indices(win, neighbour, count=1, neighbour_is_prev=True):
    """Global latent indices (multiples of 5) inside the overlap, closest to the neighbour side."""
    lo, hi = max(win.v0, neighbour.v0), min(win.v1, neighbour.v1)
    idx = [i for i in range(lo, hi) if i % 5 == 0]
    if not neighbour_is_prev:
        idx = idx[::-1]
    return sorted(idx[:max(0, count)])


def window_keyframes(base_keyframes, win):
    """Shift user keyframes (pixel-frame anchored) into window coordinates, dropping out-of-range ones."""
    f0 = frames_for_latent_t(win.v0)
    f1 = frames_for_latent_t(win.v1)
    out = []
    for kf in base_keyframes or []:
        if kf.get("latent") is None:
            continue
        r = kf["resolved_frame_index"]
        n = frames_for_latent_t(kf["latent"].shape[2])
        if r >= f0 and r + n <= f1:
            d = dict(kf)
            d["resolved_frame_index"] = r - f0
            d.pop("audio_latent", None)
            out.append(d)
    return out


def build_window_payload(payload, win, extra_keyframes):
    """Per-window copy of the H3 payload (see model_base.MiniMaxH3.extra_conds for the key order)."""
    p = {k: v for k, v in payload.items() if k not in ("layout", "keyframes", "cond_video_latents", "cond_audio_latents")}
    kfs = window_keyframes(payload.get("keyframes"), win) + list(extra_keyframes)
    cond_v, cond_a = [], []
    if kfs:
        p["keyframes"] = kfs
        cond_v += [kf["latent"] for kf in kfs if kf.get("latent") is not None]
    refs = payload.get("refs")
    if refs:
        cond_v += [r["latent"] for r in refs if "latent" in r]
        cond_a += [r["audio_latent"] for r in refs if r.get("audio_latent") is not None]
    p["cond_video_latents"] = cond_v
    p["cond_audio_latents"] = cond_a
    return p


def pack(video, audio):
    v = video.reshape(video.shape[0], 1, -1)
    a = audio.reshape(audio.shape[0], 1, -1)
    return torch.cat([v, a], dim=-1), [video.shape, audio.shape]


def unpack(x, shapes):
    vs, as_ = shapes
    n = math.prod(vs[1:])
    return x[:, :, :n].reshape([x.shape[0]] + list(vs[1:])), x[:, :, n:].reshape([x.shape[0]] + list(as_[1:]))


class WindowedH3Wrapper:
    """model_function_wrapper: every model call evaluates all windows (in alternating order)
    and returns the ramp-blended denoised prediction for the full latent."""

    def __init__(self, num_windows=0, window_blocks=0, overlap_blocks=1, alternate=True,
                 temporal_guide=True, guide_count=1, guide_from_step=1,
                 anchor_video=None, anchor_strength=0.0, anchor_steps=0, log=None, reporter=None):
        self.num_windows = num_windows
        self.window_blocks = window_blocks
        self.overlap_blocks = overlap_blocks
        self.alternate = alternate
        self.temporal_guide = temporal_guide
        self.guide_count = guide_count
        self.guide_from_step = guide_from_step
        self.anchor_video = anchor_video
        self.anchor_strength = anchor_strength
        self.anchor_steps = anchor_steps
        self.log = log  # optional list for tests: (step, order)
        self._plan = None
        self.reporter = reporter

    @staticmethod
    def step_index(timestep, transformer_options):
        sig = transformer_options.get("sample_sigmas")
        if sig is None:
            return 0
        t = timestep.flatten()[0].to(sig.device, sig.dtype)
        return int(torch.argmin((sig - t).abs()).item())

    def __call__(self, apply_model, args):
        x, t, c = args["input"], args["timestep"], dict(args["c"])
        shapes = c.get("latent_shapes")
        payload = c.get("minimax_payload")
        if shapes is None or len(shapes) != 2 or payload is None:
            return apply_model(x, t, **c)
        to = c.get("transformer_options", {})
        step = self.step_index(t, to)
        self._sigma = float(t.flatten()[0])
        self._tag = "/".join("cond" if v == 0 else "uncond" for v in args.get("cond_or_uncond", [])) or "cond"
        outs = []
        for b in range(x.shape[0]):
            cb = dict(c)
            if cb.get("c_crossattn") is not None and cb["c_crossattn"].shape[0] == x.shape[0]:
                cb["c_crossattn"] = cb["c_crossattn"][b:b + 1]
            outs.append(self._one(apply_model, x[b:b + 1], t[b:b + 1] if t.shape[0] == x.shape[0] else t, cb, shapes, payload, step))
        return torch.cat(outs, dim=0)

    def _one(self, apply_model, x, t, c, shapes, payload, step):
        video, audio = unpack(x, shapes)
        T, A = video.shape[2], audio.shape[-1]
        wins = plan_windows(T, A, self.num_windows, self.window_blocks, self.overlap_blocks)
        weights = window_weights(wins)
        order = list(range(len(wins)))
        reverse = self.alternate and step % 2 == 1
        if reverse:
            order = order[::-1]
        if self.log is not None:
            self.log.append((step, tuple(order)))
        rep = self.reporter
        guide_on = self.temporal_guide and step >= self.guide_from_step
        anchor_k = 0.0
        if self.anchor_video is not None and self.anchor_strength > 0 and step < self.anchor_steps:
            anchor_k = self.anchor_strength * (1.0 - step / max(1, self.anchor_steps))
        if rep is not None:
            rep.step_begin(step, getattr(self, "_sigma", 0.0), order, guide_on, anchor_k, getattr(self, "_tag", ""))
        acc_v = torch.zeros_like(video, dtype=torch.float32)
        acc_a = torch.zeros_like(audio, dtype=torch.float32)
        den_v = torch.zeros(T, dtype=torch.float32, device=x.device)
        den_a = torch.zeros(A, dtype=torch.float32, device=x.device)
        preds = {}
        drop = ("latent_shapes", "minimax_payload", "denoise_mask", "audio_denoise_mask")
        base_c = {k: v for k, v in c.items() if k not in drop}
        for pos, wi in enumerate(order):
            w = wins[wi]
            extra = []
            if self.temporal_guide and step >= self.guide_from_step and pos > 0:
                nb = order[pos - 1]
                nv = preds[nb]
                for gi in guide_indices(w, wins[nb], self.guide_count, neighbour_is_prev=(nb < wi)):
                    lat = nv[:, :, gi - wins[nb].v0: gi - wins[nb].v0 + 1].detach().clone()
                    extra.append({"resolved_frame_index": frames_for_latent_t(gi - w.v0), "latent": lat})
            tw0 = time.perf_counter()
            wv = video[:, :, w.v0:w.v1]
            wa = audio[..., w.a0:w.a1]
            xw, wshapes = pack(wv, wa)
            out = apply_model(xw, t, latent_shapes=wshapes,
                              minimax_payload=build_window_payload(payload, w, extra), **base_c)
            ov, oa = unpack(out, wshapes)
            preds[wi] = ov.float()
            vw, aw = weights[wi]
            vw, aw = vw.to(x.device), aw.to(x.device)
            acc_v[:, :, w.v0:w.v1] += ov.float() * vw.view(1, 1, -1, 1, 1)
            acc_a[..., w.a0:w.a1] += oa.float() * aw
            den_v[w.v0:w.v1] += vw
            den_a[w.a0:w.a1] += aw
            if rep is not None:
                rep.window_done(step, pos, wi, w, time.perf_counter() - tw0,
                                [e["resolved_frame_index"] for e in extra], getattr(self, "_tag", ""))
        out_v = acc_v / den_v.clamp(min=1e-8).view(1, 1, -1, 1, 1)
        out_a = acc_a / den_a.clamp(min=1e-8)
        if anchor_k > 0:
            k = anchor_k
            anc = self.anchor_video.to(out_v.device, out_v.dtype)
            out_v = torch.lerp(out_v, anc, k)
        return pack(out_v.to(x.dtype), out_a.to(x.dtype))[0]


def sequential_guide_indices(win, prev, count=1):
    """Sequential mode: global latent indices (multiples of 5) in the overlap with the finished previous
    window, the latest ones first (closest to the part of the window that is still unknown)."""
    lo, hi = max(win.v0, prev.v0), min(win.v1, prev.v1)
    idx = [i for i in range(lo, hi) if i % 5 == 0]
    return sorted(idx[::-1][:max(0, count)])


class SequentialWindowWrapper:
    """model_function_wrapper for window-major ("sequential") sampling of ONE window.

    The sampler runs on the window's own sub-latent. Every model call:
    - adds `keyframes` (finished latent frames of the previous window, window coordinates) to the H3 payload;
    - replaces the predicted x0 inside the overlap with the finished previous window (`lock_*`, `lock_w*`),
      so the sampler converges exactly to the already finished frames there;
    - optionally lerps the rest towards an anchor slice (same decay as the step-major mode)."""

    def __init__(self, keyframes=None, lock_video=None, lock_wv=None, lock_audio=None, lock_wa=None,
                 anchor_video=None, anchor_strength=0.0, anchor_steps=0, on_step=None):
        self.keyframes = list(keyframes or [])
        self.lock_video, self.lock_wv = lock_video, lock_wv
        self.lock_audio, self.lock_wa = lock_audio, lock_wa
        self.anchor_video = anchor_video
        self.anchor_strength = anchor_strength
        self.anchor_steps = anchor_steps
        self.on_step = on_step

    def anchor_k(self, step):
        if self.anchor_video is None or self.anchor_strength <= 0 or step >= self.anchor_steps:
            return 0.0
        return self.anchor_strength * (1.0 - step / max(1, self.anchor_steps))

    def __call__(self, apply_model, args):
        x, t, c = args["input"], args["timestep"], dict(args["c"])
        shapes = c.get("latent_shapes")
        payload = c.get("minimax_payload")
        if shapes is None or len(shapes) != 2 or payload is None:
            return apply_model(x, t, **c)
        step = WindowedH3Wrapper.step_index(t, c.get("transformer_options", {}))
        tag = "/".join("cond" if v == 0 else "uncond" for v in args.get("cond_or_uncond", [])) or "cond"
        k = self.anchor_k(step)
        if self.on_step is not None:
            self.on_step(step, float(t.flatten()[0]), k, tag)
        if self.keyframes:
            T = shapes[0][2]
            A = shapes[1][-1]
            c["minimax_payload"] = build_window_payload(payload, Window(0, T, 0, A), self.keyframes)
        out = apply_model(x, t, **c)
        ov, oa = unpack(out, shapes)
        ov, oa = ov.float(), oa.float()
        if k > 0:
            ov = torch.lerp(ov, self.anchor_video.to(ov.device, ov.dtype), k)
        if self.lock_video is not None:
            w = self.lock_wv.to(ov.device).view(1, 1, -1, 1, 1)
            ov = ov * (1 - w) + self.lock_video.to(ov.device, ov.dtype) * w
        if self.lock_audio is not None:
            w = self.lock_wa.to(oa.device)
            oa = oa * (1 - w) + self.lock_audio.to(oa.device, oa.dtype) * w
        return pack(ov.to(x.dtype), oa.to(x.dtype))[0]
