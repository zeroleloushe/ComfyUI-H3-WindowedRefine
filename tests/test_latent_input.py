import pytest
import torch
import comfy.nested_tensor as nt
from h3wr.nodes import split_av, to_av_latent, anchor_audio, describe
from h3wr.h3_windows import audio_len_for_frames


def vid(T=37, h=4, w=6):
    return torch.randn(1, 24, T, h, w)


def test_nested_passthrough():
    v, a = vid(), torch.randn(1, 32, 2, audio_len_for_frames(124))
    v2, a2, src = split_av({"samples": nt.NestedTensor((v, a))})
    assert v2 is v and a2 is a and src == "input"


def test_list_and_tuple():
    v, a = vid(), torch.randn(1, 32, 2, 207)
    for s in ([v, a], (v, a)):
        assert split_av({"samples": s})[2] == "input"


@pytest.mark.parametrize("T,frames", [(2, 5), (37, 124), (107, 362)])
def test_plain_video_gets_zero_audio_of_right_length(T, frames):
    v, a, src = split_av({"samples": vid(T)})
    assert src == "zeros" and a.shape == (1, 32, 2, audio_len_for_frames(frames)) and not a.any()


def test_plain_video_takes_anchor_audio():
    anc = {"samples": nt.NestedTensor((vid(37, 2, 3), torch.randn(1, 32, 2, 207)))}
    v, a, src = split_av({"samples": vid()}, anchor_audio(anc))
    assert src == "anchor_latent" and torch.equal(a, anc["samples"].unbind()[1])


def test_anchor_audio_wrong_length_ignored():
    v, a, src = split_av({"samples": vid()}, torch.randn(1, 32, 2, 50))
    assert src == "zeros"


@pytest.mark.parametrize("bad,msg", [
    (torch.randn(1, 24, 48, 84), "4D"),
    (torch.randn(1, 16, 37, 4, 4), "24"),
    (torch.randn(1, 24, 36, 4, 4), "5k\\+2"),
    ([torch.randn(1, 24, 37, 4, 4), torch.randn(1, 8, 2, 10)], "audio"),
    ("nope", "unsupported"),
])
def test_clear_errors(bad, msg):
    with pytest.raises(ValueError, match=msg) as e:
        split_av({"samples": bad})
    assert "got" in str(e.value) or "unsupported" in str(e.value)


def test_output_is_av_nested():
    v, a, _ = split_av({"samples": vid()})
    out = to_av_latent({"samples": v, "noise_mask": 1}, v, a)
    assert out["samples"].is_nested and len(out["samples"].unbind()) == 2 and "noise_mask" not in out
    assert "NestedTensor" in describe(out)
