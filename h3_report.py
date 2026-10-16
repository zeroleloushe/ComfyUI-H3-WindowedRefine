"""Console reporting / progress / interruption for the windowed H3 sampler. [H3WR] prefix."""
import logging
import sys
import time

from .h3_windows import frames_for_latent_t

logger = logging.getLogger("H3WR")
PREFIX = "[H3WR]"


def _emit(msg):
    line = f"{PREFIX} {msg}"
    try:
        logger.info(line)
        if not logger.isEnabledFor(logging.INFO) or not logging.getLogger().handlers:
            print(line, file=sys.stderr, flush=True)
    except Exception:  # logging must never break sampling
        try:
            print(line, flush=True)
        except Exception:
            pass


def _fmt_t(s):
    s = max(0.0, float(s))
    return f"{int(s // 60)}m{s % 60:04.1f}s" if s >= 60 else f"{s:.1f}s"


class Reporter:
    def __init__(self, verbose=True, total_steps=1, num_windows=1, pbar=None, interrupt=None, cuda_peak=None, emit=None, notify=None, sequential=False):
        self.sequential = sequential
        self._seq_seen = set()
        self._win_t0 = None
        self.verbose = verbose
        self.total_steps = max(1, total_steps)
        self.num_windows = max(1, num_windows)
        self.pbar = pbar
        self.interrupt = interrupt
        self.cuda_peak = cuda_peak
        self.emit = emit or _emit
        self.notify = notify
        self.t0 = time.perf_counter()
        self._step_t0 = {}
        self._last_step = None
        self.lines = []

    def log(self, msg):
        if self.verbose:
            self.lines.append(msg)
            self.emit(msg)

    def send(self, event, **data):
        if self.notify is not None:
            try:
                self.notify(event, data)
            except Exception:
                pass

    def check_interrupt(self):
        if self.interrupt is not None:
            self.interrupt()

    def summary(self, video, audio, audio_src, wins, refs, steps, denoise, sigmas, cfg, sampler, scheduler,
                shift, guide, anchor):
        T = video.shape[2]
        self.send("summary", T=int(T), frames=int(frames_for_latent_t(T)), steps=int(steps), seq=bool(self.sequential),
                  windows=[[int(w.v0), int(w.v1)] for w in wins])
        if not self.verbose:
            return
        frames = frames_for_latent_t(T)
        L = self.log
        L("=" * 72)
        L("MiniMax H3 Windowed Latent Refine")
        L(f"video latent {list(video.shape)}  audio latent {list(audio.shape)}  (audio from: {audio_src})")
        L(f"T={T} latent frames = {frames} px frames = {frames / 24:.2f}s @24fps, canvas {video.shape[4] * 16}x{video.shape[3] * 16}")
        L(f"windows: {len(wins)}   refs: {refs}   mode: {'sequential (window by window)' if self.sequential else 'step-major (all windows every step)'}")
        L(f"  {'#':>2} | {'latent':>11} | {'len':>3} | {'px frames':>11} | {'audio':>11} | ovl prev")
        for i, w in enumerate(wins):
            ov = (wins[i - 1].v1 - w.v0) if i > 0 else 0
            f0, f1 = frames_for_latent_t(w.v0), frames_for_latent_t(w.v1)
            L(f"  {i + 1:>2} | {w.v0:>4}..{w.v1 - 1:<5} | {w.v1 - w.v0:>3} | {f0:>4}..{f1 - 1:<5} | {w.a0:>4}..{w.a1 - 1:<5} | {ov}")
        L(f"steps {steps}, denoise {denoise}, cfg {cfg}, sampler {sampler}/{scheduler}, shift {shift}")
        L("sigmas: " + ", ".join(f"{s:.3f}" for s in sigmas))
        L(f"guide: {guide}")
        L(f"anchor: {anchor}")
        L("=" * 72)

    def step_begin(self, step, sigma, order, guide_on, anchor_k, tag):
        self.check_interrupt()
        if step not in self._step_t0:
            self.send("step", step=int(step), steps=self.total_steps, sigma=float(sigma), order=[int(o) for o in order],
                      guide=bool(guide_on), anchor=float(anchor_k))
            if self._last_step is not None and self._last_step in self._step_t0:
                self._step_end(self._last_step)
            self._step_t0[step] = time.perf_counter()
            self._last_step = step
            direction = "forward 1->N" if order == sorted(order) else "reverse N->1"
            self.log(f"step {step + 1}/{self.total_steps} sigma={sigma:.4f} {direction} windows={[o + 1 for o in order]} "
                     f"guide={'on' if guide_on else 'off'} anchor={anchor_k:.3f}")
        elif self.verbose:
            self.log(f"step {step + 1}/{self.total_steps} extra model call ({tag})")

    def window_done(self, step, pos, wi, w, elapsed, guides, tag):
        self.log(f"  window {pos + 1}/{self.num_windows} (#{wi + 1}) latent {w.v0}..{w.v1 - 1} [{tag}] "
                 f"guides={guides} {elapsed:.2f}s")
        el = time.perf_counter() - self.t0
        frac = (step * self.num_windows + pos + 1) / (self.total_steps * self.num_windows)
        self.send("window", step=int(step), steps=self.total_steps, pos=int(pos), window=int(wi), tag=tag,
                  v0=int(w.v0), v1=int(w.v1), elapsed=el, eta=el / max(frac, 1e-6) * (1 - frac) if frac <= 1 else 0.0)
        if self.pbar is not None:
            try:
                self.pbar.update_absolute(min(self.total_steps * self.num_windows, step * self.num_windows + pos + 1),
                                          self.total_steps * self.num_windows)
            except Exception:
                pass
        self.check_interrupt()

    def _step_end(self, step):
        dt = time.perf_counter() - self._step_t0[step]
        done = step + 1
        el = time.perf_counter() - self.t0
        eta = el / done * (self.total_steps - done)
        self.log(f"step {done}/{self.total_steps} done in {_fmt_t(dt)}; elapsed {_fmt_t(el)}, ETA {_fmt_t(eta)}")

    def finish(self):
        if self._last_step is not None and self._last_step in self._step_t0:
            self._step_end(self._last_step)
            self._last_step = None
        peak = ""
        if self.cuda_peak is not None:
            try:
                p = self.cuda_peak()
                if p:
                    peak = f", peak VRAM {p / 1024 ** 3:.2f} GiB"
            except Exception:
                pass
        self.send("done", total=time.perf_counter() - self.t0)
        self.log(f"finished in {_fmt_t(time.perf_counter() - self.t0)}{peak}")

    # ---- sequential (window-major) mode
    def seq_window_begin(self, wi, w, overlap, guides):
        self.check_interrupt()
        self._win_t0 = time.perf_counter()
        self.log(f"window {wi + 1}/{self.num_windows} latent {w.v0}..{w.v1 - 1}: locked overlap {overlap} latent frame(s), "
                 f"guides={guides}")

    def seq_step(self, wi, step, sigma, anchor_k, tag):
        self.check_interrupt()
        key = (wi, step)
        if key in self._seq_seen:
            if self.verbose:
                self.log(f"  step {step + 1}/{self.total_steps} extra model call ({tag})")
            return
        self._seq_seen.add(key)
        el = time.perf_counter() - self.t0
        done = wi * self.total_steps + step
        total = self.total_steps * self.num_windows
        eta = el / done * (total - done) if done else 0.0
        self.send("step", step=int(step), steps=self.total_steps, sigma=float(sigma), order=[int(wi)], guide=wi > 0,
                  anchor=float(anchor_k), seq=True)
        self.send("window", step=int(step), steps=self.total_steps, pos=int(wi), window=int(wi), tag=tag, seq=True,
                  elapsed=el, eta=eta)
        if self.pbar is not None:
            try:
                self.pbar.update_absolute(min(total, done + 1), total)
            except Exception:
                pass
        self.log(f"  window {wi + 1} step {step + 1}/{self.total_steps} sigma={sigma:.4f} anchor={anchor_k:.3f} [{tag}]")

    def seq_window_done(self, wi, w):
        dt = time.perf_counter() - (self._win_t0 or self.t0)
        el = time.perf_counter() - self.t0
        eta = el / (wi + 1) * (self.num_windows - wi - 1)
        self.log(f"window {wi + 1}/{self.num_windows} done in {_fmt_t(dt)}; elapsed {_fmt_t(el)}, ETA {_fmt_t(eta)}")
        self.check_interrupt()
