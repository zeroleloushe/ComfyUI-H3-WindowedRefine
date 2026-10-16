import pytest
import torch
from h3wr.nodes import collect_refs, mapping_lines, MAX_REF_IMAGES, H3WindowedLatentRefine


def img(b=1):
    return torch.rand(b, 32, 48, 3)


def test_order_and_batches():
    imgs, m = collect_refs({"location_ref": img(3), "character_ref": img(), "ref_5": img()})
    assert len(imgs) == 5 and all(i.shape[0] == 1 for i in imgs)
    assert mapping_lines(m) == ["<Picture 1> = character_ref", "<Picture 2> = location_ref [0]",
                                "<Picture 3> = location_ref [1]", "<Picture 4> = location_ref [2]", "<Picture 5> = ref_5"]


def test_backward_compat_two_refs():
    _, m = collect_refs({"character_ref": img(), "location_ref": img()})
    assert mapping_lines(m) == ["<Picture 1> = character_ref", "<Picture 2> = location_ref"]


def test_only_location_is_picture_1():
    _, m = collect_refs({"location_ref": img()})
    assert mapping_lines(m) == ["<Picture 1> = location_ref"]


def test_limit():
    with pytest.raises(ValueError, match="max 9"):
        collect_refs({"character_ref": img(5), "location_ref": img(5)})
    imgs, _ = collect_refs({"character_ref": img(MAX_REF_IMAGES)})
    assert len(imgs) == MAX_REF_IMAGES


def test_bad_ref_type():
    with pytest.raises(ValueError, match="ref_4"):
        collect_refs({"ref_4": torch.rand(2, 2)})


def test_inputs_declared():
    opt = H3WindowedLatentRefine.INPUT_TYPES()["optional"]
    for n in ("character_ref", "location_ref", "location_ref_2", "location_ref_3", "ref_4", "ref_5", "ref_6", "anchor_latent"):
        assert n in opt


def test_encode_ref2va_gets_one_item_per_picture():
    from h3wr.nodes import encode_ref2va
    seen = {}

    class Clip:
        def tokenize(self, prompt, minimax_ref_items=None):
            seen["items"] = minimax_ref_items
            return {}

        def encode_from_tokens_scheduled(self, tokens):
            return [[torch.zeros(1, 4, 8), {}]]

    class Vae:
        def encode(self, x):
            return torch.zeros(1, 24, 1, x.shape[1] // 16, x.shape[2] // 16)

    imgs, _ = collect_refs({"character_ref": img(), "location_ref": img(2)})
    cond = encode_ref2va(Clip(), Vae(), "p", imgs, 96, 64)
    assert [i["type"] for i in seen["items"]] == ["image"] * 3
    assert len(cond[0][1]["minimax_refs"]) == 3
