import torch
import pytest
from h3wr.h3_windows import (plan_windows, window_weights, guide_indices, frames_for_latent_t,
                             audio_len_for_frames, build_window_payload, pack, unpack, WindowedH3Wrapper, Window)


def lat_t(frames):  # nodes_minimax_h3.video_latent_t
    return 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2


@pytest.mark.parametrize("frames", [124, 243, 362])
@pytest.mark.parametrize("nw,wb,ob", [(4, 0, 1), (3, 0, 2), (2, 0, 0), (0, 3, 1), (0, 4, 2), (1, 0, 1)])
def test_plan_alignment_and_coverage(frames, nw, wb, ob):
    T = lat_t(frames)
    A = audio_len_for_frames(frames)
    wins = plan_windows(T, A, nw, wb, ob)
    assert wins[0].v0 == 0 and wins[-1].v1 == T and wins[0].a0 == 0 and wins[-1].a1 == A
    for w in wins:
        assert w.v0 % 5 == 0
        assert (w.v1 - w.v0) % 5 == 2
        assert 0 <= w.a0 < w.a1 <= A
    for a, b in zip(wins, wins[1:]):
        assert b.v0 < a.v1, "video windows must overlap"
        assert a.v1 - b.v0 >= 5 * ob + 2 or len(wins) == 1
        assert b.a0 < a.a1, "audio windows must overlap"
    if nw > 1:
        assert len(wins) <= nw


def test_audio_proportional():
    T, A = lat_t(362), audio_len_for_frames(362)
    for w in plan_windows(T, A, 4, 0, 1):
        f0 = frames_for_latent_t(w.v0)
        fl = frames_for_latent_t(w.v1 - w.v0)
        assert abs(w.a0 - f0 * 5 / 3) <= 2 or w.a1 == A
        assert abs((w.a1 - w.a0) - fl * 5 / 3) <= 2


def test_weights_partition_of_unity_and_ramps():
    T, A = 37, audio_len_for_frames(124)
    wins = plan_windows(T, A, 3, 0, 1)
    ws = window_weights(wins)
    tot = torch.zeros(T)
    for w, (vw, aw) in zip(wins, ws):
        assert vw.shape[0] == w.v1 - w.v0 and aw.shape[0] == w.a1 - w.a0
        tot[w.v0:w.v1] += vw
    assert (tot > 0).all()
    # blending a constant per window must reproduce a smooth crossfade between constants
    acc = torch.zeros(T); den = torch.zeros(T)
    for i, (w, (vw, _)) in enumerate(zip(wins, ws)):
        acc[w.v0:w.v1] += vw * i; den[w.v0:w.v1] += vw
    r = acc / den
    assert r[0] == 0 and r[-1] == len(wins) - 1
    assert torch.all(r[1:] >= r[:-1] - 1e-6), "monotone crossfade"


def test_guides_on_single_frame_tokens():
    wins = plan_windows(37, audio_len_for_frames(124), 3, 0, 1)
    g = guide_indices(wins[1], wins[0], 1, True)
    assert g == [wins[1].v0]
    g2 = guide_indices(wins[0], wins[1], 1, False)
    assert g2 and g2[0] % 5 == 0 and wins[1].v0 <= g2[0] < wins[0].v1
    assert frames_for_latent_t(10) == 34  # latent 5j <-> pixel 17j


def test_payload_rebuild_drops_layout_and_orders_cond():
    ref = {"kind": "image", "latent_h": 2, "latent_w": 2, "latent": torch.ones(1, 24, 1, 2, 2)}
    kf_in = {"resolved_frame_index": 40, "latent": torch.zeros(1, 24, 1, 4, 4)}
    kf_out = {"resolved_frame_index": 5, "latent": torch.zeros(1, 24, 1, 4, 4)}
    payload = {"layout": object(), "refs": [ref], "keyframes": [kf_in, kf_out], "seed": 3, "audio_scale": 1.0}
    w = Window(10, 22, 0, 10)  # pixel frames 34..(34+39)
    g = {"resolved_frame_index": 0, "latent": torch.full((1, 24, 1, 4, 4), 2.0)}
    p = build_window_payload(payload, w, [g])
    assert "layout" not in p
    assert [k["resolved_frame_index"] for k in p["keyframes"]] == [6, 0]
    assert len(p["cond_video_latents"]) == 3 and p["cond_video_latents"][-1] is ref["latent"]
    assert payload["keyframes"][0]["resolved_frame_index"] == 40  # original untouched


def test_pack_roundtrip():
    v, a = torch.randn(1, 24, 7, 4, 6), torch.randn(1, 32, 2, 13)
    x, s = pack(v, a)
    v2, a2 = unpack(x, s)
    assert torch.equal(v, v2) and torch.equal(a, a2)


def test_wrapper_with_identity_model_and_alternation():
    """apply_model returning its input must give back exactly the input (blend is normalized)."""
    T, A = 37, audio_len_for_frames(124)
    v, a = torch.randn(1, 24, T, 4, 4), torch.randn(1, 32, 2, A)
    x, shapes = pack(v, a)
    calls = []

    def apply_model(xw, t, latent_shapes=None, minimax_payload=None, **kw):
        calls.append((latent_shapes[0][2], latent_shapes[1][-1], len(minimax_payload.get("keyframes", []))))
        return xw

    log = []
    wr = WindowedH3Wrapper(num_windows=3, overlap_blocks=1, log=log, guide_from_step=1)
    sig = torch.tensor([1.0, 0.7, 0.3, 0.0])
    for step in range(3):
        out = wr(apply_model, {"input": x, "timestep": sig[step:step + 1], "cond_or_uncond": [0],
                                "c": {"c_crossattn": torch.zeros(1, 5, 8), "latent_shapes": shapes,
                                      "minimax_payload": {"seed": 0}, "transformer_options": {"sample_sigmas": sig}}})
        assert torch.allclose(out, x, atol=1e-6)
    assert log[0][1] == (0, 1, 2) and log[1][1] == (2, 1, 0) and log[2][1] == (0, 1, 2)
    # step 0: no guides; later steps: first window none, others one guide keyframe
    assert [c[2] for c in calls[:3]] == [0, 0, 0]
    assert [c[2] for c in calls[3:6]] == [0, 1, 1]
    assert all((c[0] - 2) % 5 == 0 for c in calls)


def test_anchor_blend():
    T, A = 12, audio_len_for_frames(39)
    v, a = torch.zeros(1, 24, T, 2, 2), torch.zeros(1, 32, 2, A)
    x, shapes = pack(v, a)
    anc = torch.ones_like(v)
    wr = WindowedH3Wrapper(num_windows=2, overlap_blocks=0, anchor_video=anc, anchor_strength=0.5, anchor_steps=2)
    sig = torch.tensor([1.0, 0.5, 0.0])
    out = wr(lambda xw, t, **k: xw, {"input": x, "timestep": sig[:1], "cond_or_uncond": [0],
            "c": {"latent_shapes": shapes, "minimax_payload": {}, "transformer_options": {"sample_sigmas": sig}}})
    ov, oa = unpack(out, shapes)
    assert torch.allclose(ov, torch.full_like(ov, 0.5)) and torch.equal(oa, a)
