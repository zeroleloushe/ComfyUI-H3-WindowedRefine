#!/usr/bin/env python3
"""Seam check for H3WindowedLatentRefine outputs.

Per frame: mean abs diff to previous frame (MAD), PSNR to previous frame, Laplacian-variance
sharpness, mean brightness. Optional expected window overlaps (from the node's own layout
logic) are shaded on the chart; outliers are flagged by robust z-score (median/MAD).

  python seam_check.py video.mp4 --num-windows 2 --overlap-blocks 1 2 --out chart.png
  python seam_check.py video.mp4 --ranges 85-108 --out chart.png    # explicit pixel-frame ranges
"""
import argparse
import importlib.util
import os
import sys

import cv2
import numpy as np


def load_layout():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "h3_windows.py")
    spec = importlib.util.spec_from_file_location("h3_windows_layout", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def expected_overlaps(frames, num_windows, overlap_blocks, window_blocks=0):
    """Pixel-frame ranges of window overlaps, using the node's plan_windows."""
    h = load_layout()
    if frames % 17 != 5:
        print(f"warning: {frames} frames is not 17k+5; using floor grid")
    T = max(2, ((frames - 5) // 17) * 5 + 2)
    wins = h.plan_windows(T, h.audio_len_for_frames(frames), num_windows, window_blocks, overlap_blocks)
    out = []
    for a, b in zip(wins, wins[1:]):
        f0 = h.frames_for_latent_t(b.v0)
        f1 = h.frames_for_latent_t(a.v1) - 1
        out.append((f0, f1, f"nw={num_windows} ob={overlap_blocks}: latent {b.v0}..{a.v1 - 1}"))
    return T, wins, out


def read_metrics(path, max_side=512):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    prev = None
    prev_hist = None
    mad, psnr, sharp, bright, hist_d, rgb = [], [], [], [], [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        s = max_side / max(g.shape)
        gs = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
        f = gs.astype(np.float32)
        sharp.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))  # full-res sharpness
        bright.append(float(f.mean()))
        small = cv2.resize(fr, (64, int(64 * fr.shape[0] / fr.shape[1])), interpolation=cv2.INTER_AREA)
        rgb.append(small.reshape(-1, 3)[:, ::-1].mean(0))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist_d.append(np.nan if prev_hist is None else float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)))
        prev_hist = hist
        if prev is None:
            mad.append(np.nan); psnr.append(np.nan)
        else:
            d = np.abs(f - prev)
            mad.append(float(d.mean()))
            mse = float((d ** 2).mean())
            psnr.append(99.0 if mse == 0 else float(10 * np.log10(255 ** 2 / mse)))
        prev = f
    cap.release()
    return fps, (w, h), {k: np.array(v) for k, v in
                         (("mad", mad), ("psnr", psnr), ("sharpness", sharp), ("brightness", bright), ("hist", hist_d), ("rgb", rgb))}


def make_sheet(path, zones, out, cuts=(), width=240):
    seen, rows = set(), []
    for s0, e0, lbl in zones:
        if (s0, e0) in seen:
            continue
        seen.add((s0, e0))
        rows.append([(s0 - 1, "before"), (s0, "start"), ((s0 + e0) // 2, "mid"), (e0, "end"), (e0 + 1, "after")])
    cap = cv2.VideoCapture(path)
    frames = {}
    want = {i for r in rows for i, _ in r}
    i = 0
    while want - set(frames):
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            frames[i] = fr
        i += 1
    cap.release()
    tiles = []
    for (s0, e0), r in zip(sorted(seen), rows):
        row = []
        for idx, tag in r:
            fr = frames.get(idx)
            if fr is None:
                fr = np.zeros((10, 10, 3), np.uint8)
            h = int(fr.shape[0] * width / fr.shape[1])
            t = cv2.resize(fr, (width, h), interpolation=cv2.INTER_AREA)
            lab = f"{tag} f{idx}" + (" CUT" if idx in cuts else "")
            cv2.rectangle(t, (0, 0), (width, 22), (0, 0, 0), -1)
            cv2.putText(t, lab, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            row.append(t)
        hdr = np.zeros((26, width * len(row), 3), np.uint8)
        cv2.putText(hdr, f"overlap px {s0}..{e0}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 200, 255), 1)
        tiles.append(np.vstack([hdr, np.hstack(row)]))
    cv2.imwrite(out, np.vstack(tiles))


def robust_z(x):
    v = x[~np.isnan(x)]
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 or (v.std() or 1.0)
    return (x - med) / mad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--num-windows", type=int, default=0)
    ap.add_argument("--overlap-blocks", type=int, nargs="*", default=[1])
    ap.add_argument("--window-blocks", type=int, default=0)
    ap.add_argument("--ranges", nargs="*", default=[], help="explicit pixel-frame ranges a-b")
    ap.add_argument("--z", type=float, default=3.5)
    ap.add_argument("--out", default="seam_chart.png")
    ap.add_argument("--cut-hist", type=float, default=0.35, help="Bhattacharyya hist distance for a hard cut")
    ap.add_argument("--cut-z", type=float, default=8.0, help="MAD robust z for a hard cut")
    ap.add_argument("--sheet", default="", help="contact sheet PNG: frames at start/middle/end of each zone")
    a = ap.parse_args()

    fps, (w, h), m = read_metrics(a.video)
    n = len(m["mad"])
    print(f"video: {n} frames, {fps:.3f} fps, {w}x{h}, {n / fps:.2f}s")
    zones = []
    if a.num_windows or a.window_blocks:
        for ob in a.overlap_blocks:
            T, wins, ov = expected_overlaps(n, a.num_windows, ob, a.window_blocks)
            print(f"layout nw={a.num_windows} ob={ob}: T={T}, windows (latent) " +
                  ", ".join(f"{x.v0}..{x.v1 - 1}" for x in wins))
            zones += ov
    for r in a.ranges:
        s, e = map(int, r.split("-"))
        zones.append((s, e, f"range {s}-{e}"))

    rgbs = m.pop("rgb")
    hist = m.pop("hist")
    zmad0 = robust_z(m["mad"])
    cuts = [i for i in range(1, n) if hist[i] > a.cut_hist or zmad0[i] > a.cut_z]
    print(f"\nhard cuts (hist>{a.cut_hist} or MAD z>{a.cut_z}): {len(cuts)}")
    for i in cuts:
        print(f"  cut at frame {i} ({i / fps:.2f}s): hist={hist[i]:.2f} MAD={m['mad'][i]:.1f} (z={zmad0[i]:.1f})")
    near_cut = np.zeros(n, bool)
    for i in cuts:
        near_cut[max(0, i - 1):i + 2] = True
    # metrics with cut frames masked so they don't dominate the baseline
    for k in ("mad", "psnr"):
        m[k] = m[k].copy(); m[k][near_cut] = np.nan
    for k in ("sharpness", "brightness"):
        pass
    z = {k: robust_z(v) for k, v in m.items()}
    # jumps: high MAD / low PSNR; sharpness/brightness: step changes of the series itself
    flags = {}
    for i in range(1, n):
        reasons = []
        if z["mad"][i] > a.z: reasons.append(f"MAD z={z['mad'][i]:.1f}")
        if z["psnr"][i] < -a.z: reasons.append(f"PSNR z={z['psnr'][i]:.1f}")
        if near_cut[i]:
            continue
        for k in ("sharpness", "brightness"):
            d = np.diff(m[k]); d[near_cut[1:]] = np.nan
            dz = robust_z(d)
            if abs(dz[i - 1]) > a.z: reasons.append(f"d{k} z={dz[i - 1]:.1f}")
        if reasons: flags[i] = reasons

    def inzone(i):
        return [lbl for s, e, lbl in zones if s - 2 <= i <= e + 2]

    print("\nper-zone stats (inside vs outside):")
    for s, e, lbl in zones:
        inside = np.zeros(n, bool); inside[max(1, s):e + 1] = True
        out = ~inside; out[0] = False
        print(f"  {lbl} px {s}..{e}:")
        base = out & ~near_cut
        ins = inside & ~near_cut
        rin, rout = rgbs[ins].mean(0), rgbs[base].mean(0)
        print(f"    rgb        in={np.round(rin, 1)} out={np.round(rout, 1)}")
        # local step: 8 frames before zone vs first 8 in zone, last 8 in zone vs 8 after
        for k in ("sharpness", "brightness"):
            v = m[k].copy(); v[near_cut] = np.nan
            pre, post = np.nanmean(v[max(0, s - 8):s]), np.nanmean(v[e + 1:e + 9])
            print(f"    {k:<10} local: before={pre:.2f} zone-start={np.nanmean(v[s:s + 8]):.2f} "
                  f"zone-end={np.nanmean(v[e - 7:e + 1]):.2f} after={post:.2f}")
        for k in m:
            print(f"    {k:<10} in={np.nanmean(m[k][inside]):9.2f}  out={np.nanmean(m[k][out]):9.2f}  "
                  f"max|z| in={np.nanmax(np.abs(z[k][ins])):.1f} out={np.nanmax(np.abs(z[k][base])):.1f}")
    print(f"\noutlier frames (|z|>{a.z}): {len(flags)}")
    for i, r in flags.items():
        zl = inzone(i)
        print(f"  frame {i} ({i / fps:.2f}s): {', '.join(r)}" + (f"  <-- in seam zone [{'; '.join(zl)}]" if zl else ""))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    colors = ["tab:orange", "tab:green", "tab:purple", "tab:red"]
    for row, (k, lbl) in enumerate([("mad", "mean |Δ| vs prev"), ("psnr", "PSNR vs prev, dB"),
                                    ("sharpness", "Laplacian var"), ("brightness", "mean luma")]):
        ax[row].plot(m[k], lw=1)
        ax[row].set_ylabel(lbl)
        for j, (s, e, zl) in enumerate(zones):
            ax[row].axvspan(s, e, color=colors[j % 4], alpha=0.15, label=zl if row == 0 else None)
        for i in flags:
            ax[row].axvline(i, color="red", lw=0.6, alpha=0.6)
        for c in cuts:
            ax[row].axvline(c, color="black", lw=1.2, ls="--")
        for c in range(17, n, 17):  # VAE 17-frame clip grid
            ax[row].axvline(c, color="gray", lw=0.4, ls=":")
    if zones:
        ax[0].legend(loc="upper right", fontsize=8)
    ax[-1].set_xlabel(f"frame ({fps:.0f} fps); dotted = 17-frame VAE grid; red = outliers; black dashed = hard cuts")
    fig.suptitle(os.path.basename(a.video))
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f"\nchart: {a.out}")
    if a.sheet and zones:
        make_sheet(a.video, zones, a.sheet, cuts)
        print(f"contact sheet: {a.sheet}")


if __name__ == "__main__":
    sys.exit(main())
