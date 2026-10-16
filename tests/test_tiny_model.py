"""Runs the real comfy MiniMax H3 model class (random weights, tiny config) on CPU end to end."""
import torch
import pytest

import comfy.supported_models as sm
import comfy.model_patcher
import comfy.nested_tensor
from h3wr.h3_windows import audio_len_for_frames, WindowedH3Wrapper, pack, unpack
from h3wr.nodes import run_windowed

TEXT_DIM = 32


@pytest.fixture(scope="module")
def patcher():
    torch.manual_seed(0)
    cfg = sm.MiniMaxH3({"image_model": "minimax_h3", "hidden_size": 128, "num_layers": 2, "num_attention_heads": 1,
                        "attention_head_dim": 128, "ffn_hidden_size": 64, "text_dim": TEXT_DIM,
                        "time_embed_hidden_size": 64, "time_embed_dim": 32, "token_refiner_num_layers": 1})
    cfg.set_inference_dtype(torch.float32, None)
    m = cfg.get_model({})
    for p in m.diffusion_model.parameters():
        torch.nn.init.normal_(p, std=0.02)
    m.diffusion_model.rope.inv_freq.copy_(1.0 / (10000 ** (torch.arange(16) / 16)))
    return comfy.model_patcher.ModelPatcher(m, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


def av_latent(frames=124, h=4, w=6):
    T = ((frames - 5) // 17) * 5 + 2
    return comfy.nested_tensor.NestedTensor((torch.randn(1, 24, T, h, w), torch.randn(1, 32, 2, audio_len_for_frames(frames))))


def conds(with_ref=True):
    extra = {}
    if with_ref:
        extra["minimax_refs"] = [{"kind": "image", "latent_h": 4, "latent_w": 4, "latent": torch.randn(1, 24, 1, 4, 4)}]
    return [[torch.randn(1, 7, TEXT_DIM), dict(extra)]], [[torch.randn(1, 7, TEXT_DIM), dict(extra)]]


def test_window_forward_matches_full_when_one_window(patcher):
    """With a single window the wrapper must equal the plain model call (layout rebuilt by the model)."""
    lat = av_latent(39)
    pos, neg = conds()
    torch.manual_seed(1)
    import comfy.sample
    noise = comfy.sample.prepare_noise(lat, 0)
    a = comfy.sample.sample(patcher, noise, 2, 1.0, "euler", "simple", pos, neg, lat, denoise=0.5, seed=0, disable_pbar=True)
    p2 = patcher.clone()
    p2.set_model_unet_function_wrapper(WindowedH3Wrapper(num_windows=1))
    b = comfy.sample.sample(p2, noise, 2, 1.0, "euler", "simple", pos, neg, lat, denoise=0.5, seed=0, disable_pbar=True)
    for x, y in zip(a.unbind(), b.unbind()):
        assert torch.allclose(x, y, atol=1e-4), (x - y).abs().max()


@pytest.mark.parametrize("cfg", [1.0, 3.0])
def test_run_windowed_end_to_end(patcher, cfg):
    lat = {"samples": av_latent(124)}
    plain = {"samples": av_latent(124).unbind()[0]}
    pos, neg = conds()
    log = []
    anchor = {"samples": av_latent(124, 2, 3)}
    out = run_windowed(patcher, pos, neg, lat, 0, 3, cfg, "euler", "simple", 0.6,
                       num_windows=4, window_blocks=0, overlap_blocks=1, alternate=True, temporal_guide=True,
                       guide_from_step=1, anchor_latent=anchor, anchor_strength=0.2, anchor_steps_frac=0.5,
                       shift_video=12.0, shift_audio=3.0, log=log)
    v, a = out["samples"].unbind()
    v0, a0 = lat["samples"].unbind()
    assert v.shape == v0.shape and a.shape == a0.shape
    assert torch.isfinite(v).all() and torch.isfinite(a).all()
    steps = sorted({s for s, _ in log})
    assert steps == [0, 1, 2]
    assert any(o == (0, 1, 2, 3) for s, o in log if s % 2 == 0) and any(o == (3, 2, 1, 0) for s, o in log if s % 2 == 1)


def test_run_windowed_plain_video_input(patcher):
    pos, neg = conds()
    lat = {"samples": av_latent(124).unbind()[0]}  # e.g. VAE Encode / LatentUpscaleBy output
    out = run_windowed(patcher, pos, neg, lat, 0, 2, 1.0, "euler", "simple", 0.3,
                       num_windows=2, window_blocks=0, overlap_blocks=1, alternate=True, temporal_guide=True,
                       guide_from_step=0)
    v, a = out["samples"].unbind()
    assert v.shape == lat["samples"].shape and a.shape == (1, 32, 2, audio_len_for_frames(124))
    assert torch.isfinite(v).all()


def test_verbose_logging_and_progress(patcher, capsys):
    import comfy.utils
    seen = []
    comfy.utils.set_progress_bar_global_hook(lambda v, t, p, node_id=None: seen.append((v, t)))
    try:
        pos, neg = conds()
        lines = []
        lat = {"samples": av_latent(124)}
        run_windowed(patcher, pos, neg, lat, 0, 3, 2.0, "euler", "simple", 0.4, num_windows=3, window_blocks=0,
                     overlap_blocks=1, alternate=True, temporal_guide=True, guide_from_step=1,
                     anchor_strength=0.2, anchor_steps_frac=0.5, shift_video=12.0, verbose=True,
                     audio_src="zeros", refs_count=1, emit=lambda m: (lines.append(m), print("[H3WR] " + m)))
        text = "\n".join(lines)
        for key in ("windows: 3", "sigmas:", "step 1/3", "reverse N->1", "window 3/3", "ETA", "finished in", "guides=["):
            assert key in text, key
        assert seen and seen[-1] == (9, 9)
        quiet = []
        run_windowed(patcher, pos, neg, lat, 0, 1, 1.0, "euler", "simple", 0.4, 2, 0, 1, True, True, 1,
                     verbose=False, emit=quiet.append)
        assert quiet == []
    finally:
        comfy.utils.set_progress_bar_global_hook(None)


def test_interrupt_is_respected(patcher):
    import comfy.model_management as mm
    pos, neg = conds()
    mm.interrupt_current_processing(True)
    try:
        with pytest.raises(mm.InterruptProcessingException):
            run_windowed(patcher, pos, neg, {"samples": av_latent(124)}, 0, 2, 1.0, "euler", "simple", 0.4, 2, 0, 1, True, True, 1)
    finally:
        mm.interrupt_current_processing(False)


def test_frontend_events(patcher):
    pos, neg = conds()
    ev = []
    run_windowed(patcher, pos, neg, {"samples": av_latent(124)}, 0, 2, 1.0, "euler", "simple", 0.4, 2, 0, 1, True, True, 1,
                 verbose=False, notify=lambda e, d: ev.append((e, d)))
    kinds = [e for e, _ in ev]
    assert kinds[0] == "summary" and kinds[-1] == "done" and kinds.count("step") == 2 and kinds.count("window") == 4
    assert ev[0][1]["T"] == 37 and len(ev[0][1]["windows"]) == 2


def test_input_types_have_tooltips():
    from h3wr.nodes import H3WindowedLatentRefine as N
    it = N.INPUT_TYPES()
    for sec in ("required", "optional"):
        for name, spec in it[sec].items():
            assert len(spec) > 1 and spec[1].get("tooltip"), name
    grouped = [w for _, ws in N.WIDGET_GROUPS for w in ws]
    widgets = [k for k, v in it["required"].items() if v[0] not in ("MODEL", "CLIP", "VAE", "LATENT")]
    assert grouped == widgets
