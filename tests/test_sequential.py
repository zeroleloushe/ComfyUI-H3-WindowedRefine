"""Sequential (window-major) mode on the tiny real H3 model (CPU)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import comfy.sample
from h3wr.h3_windows import SequentialWindowWrapper, plan_windows, sequential_guide_indices, pack, unpack, Window
from h3wr.nodes import run_windowed, MODE_SEQ, MODE_STEP
from test_tiny_model import patcher, av_latent, conds  # noqa: F401


def test_sequential_guide_indices_are_latest_in_overlap():
    prev, win = Window(0, 22, 0, 0), Window(15, 37, 0, 0)
    assert sequential_guide_indices(win, prev, 1) == [20]
    assert sequential_guide_indices(win, prev, 5) == [15, 20]


def test_wrapper_locks_overlap_and_adds_keyframes():
    T, A = 7, 10
    lock_v = torch.full((1, 24, T, 2, 2), 3.0)
    wv = torch.zeros(T); wv[:2] = 1
    lock_a = torch.full((1, 32, 2, A), -2.0)
    wa = torch.zeros(A); wa[:4] = 1
    kf = [{"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 2, 2)}]
    seen = {}

    def apply_model(x, t, **c):
        seen["payload"] = c["minimax_payload"]
        return torch.zeros_like(x)

    w = SequentialWindowWrapper(kf, lock_v, wv, lock_a, wa)
    x, shapes = pack(torch.randn(1, 24, T, 2, 2), torch.randn(1, 32, 2, A))
    out = w(apply_model, {"input": x, "timestep": torch.tensor([0.5]),
                          "c": {"latent_shapes": shapes, "minimax_payload": {"refs": []}, "transformer_options": {}}})
    ov, oa = unpack(out, shapes)
    assert torch.all(ov[:, :, :2] == 3) and torch.all(ov[:, :, 2:] == 0)
    assert torch.all(oa[..., :4] == -2) and torch.all(oa[..., 4:] == 0)
    assert seen["payload"]["keyframes"] == kf and len(seen["payload"]["cond_video_latents"]) == 1


def test_sequential_end_to_end_window_major(patcher):
    lat = {"samples": av_latent(124)}
    pos, neg = conds()
    log, lines = [], []
    anchor = {"samples": av_latent(124, 2, 3)}
    out = run_windowed(patcher, pos, neg, lat, 0, 3, 2.0, "euler", "simple", 0.5, num_windows=3, window_blocks=0,
                       overlap_blocks=1, alternate=True, temporal_guide=True, guide_from_step=1, anchor_latent=anchor,
                       anchor_strength=0.2, anchor_steps_frac=0.5, shift_video=12.0, log=log,
                       emit=lines.append, window_mode=MODE_SEQ)
    v, a = out["samples"].unbind()
    v0, a0 = lat["samples"].unbind()
    assert v.shape == v0.shape and a.shape == a0.shape
    assert torch.isfinite(v).all() and torch.isfinite(a).all()
    steps = [(e[0], e[1][0]) for e in log if e[0] != "window"]
    wins_order = [w for _, w in steps]
    assert wins_order == sorted(wins_order) and set(wins_order) == {0, 1, 2}
    for wi in range(3):
        assert sorted({s for s, w in steps if w == wi}) == [0, 1, 2]
    guides = {e[1]: e[2] for e in log if e[0] == "window"}
    assert guides[0] == () and guides[1] and guides[2]
    text = "\n".join(lines)
    assert "mode: sequential" in text and "locked overlap" in text and "finished in" in text


def test_sequential_single_window_matches_plain(patcher):
    lat = {"samples": av_latent(39)}
    pos, neg = conds()
    noise = comfy.sample.prepare_noise(lat["samples"], 0)
    a = comfy.sample.sample(patcher, noise, 2, 1.0, "euler", "simple", pos, neg, lat["samples"], denoise=0.5, seed=0, disable_pbar=True)
    b = run_windowed(patcher, pos, neg, lat, 0, 2, 1.0, "euler", "simple", 0.5, 1, 0, 1, True, True, 1,
                     verbose=False, window_mode=MODE_SEQ)["samples"]
    for x, y in zip(a.unbind(), b.unbind()):
        assert torch.allclose(x, y, atol=1e-4), (x - y).abs().max()


def test_sequential_overlap_keeps_previous_window(patcher):
    """Window 2's overlap must stay exactly as window 1 finished it (no re-blend)."""
    lat = {"samples": av_latent(124)}
    pos, neg = conds()
    wins = plan_windows(37, lat["samples"].unbind()[1].shape[-1], 2, 0, 1)
    out = run_windowed(patcher, pos, neg, lat, 0, 2, 1.0, "euler", "simple", 0.5, 2, 0, 1, True, True, 1,
                       verbose=False, window_mode=MODE_SEQ)["samples"].unbind()[0]
    v, a = lat["samples"].unbind()
    noise = comfy.sample.prepare_noise(lat["samples"], 0)
    nv, na = noise.unbind()
    w = wins[0]
    import comfy.nested_tensor as nt
    first = comfy.sample.sample(patcher, nt.NestedTensor((nv[:, :, w.v0:w.v1], na[..., w.a0:w.a1])), 2, 1.0, "euler", "simple",
                                pos, neg, nt.NestedTensor((v[:, :, w.v0:w.v1], a[..., w.a0:w.a1])), denoise=0.5, seed=0,
                                disable_pbar=True).unbind()[0]
    assert torch.allclose(out[:, :, :w.v1], first, atol=1e-4)


def test_default_mode_unchanged(patcher):
    lat = {"samples": av_latent(124)}
    pos, neg = conds()
    a = run_windowed(patcher, pos, neg, lat, 0, 2, 1.0, "euler", "simple", 0.4, 2, 0, 1, True, True, 1, verbose=False)
    b = run_windowed(patcher, pos, neg, lat, 0, 2, 1.0, "euler", "simple", 0.4, 2, 0, 1, True, True, 1, verbose=False,
                     window_mode=MODE_STEP)
    for x, y in zip(a["samples"].unbind(), b["samples"].unbind()):
        assert torch.equal(x, y)


def test_prev_clip_is_last_reference():
    from h3wr.nodes import collect_refs, mapping_lines
    imgs, mapping = collect_refs({"character_ref": torch.zeros(1, 8, 8, 3), "prev_clip": torch.zeros(1, 8, 8, 3)})
    assert mapping_lines(mapping)[-1] == "<Picture 2> = prev_clip"
