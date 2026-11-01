"""Separate Window Planner node feeding the refine node."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pytest
import torch

from h3wr.nodes import H3WindowPlanner, H3WindowedLatentRefine, apply_plan, NODE_CLASS_MAPPINGS
from test_tiny_model import av_latent  # noqa: F401


def test_planner_registered_and_typed():
    assert NODE_CLASS_MAPPINGS["H3WindowPlanner"] is H3WindowPlanner
    assert H3WindowPlanner.RETURN_TYPES == ("H3_WINDOWS",)
    assert "windows" in H3WindowedLatentRefine.INPUT_TYPES()["optional"]


def test_planner_plan_and_validation():
    plan, = H3WindowPlanner().run(3, 0, 1)
    assert plan == {"num_windows": 3, "window_blocks": 0, "overlap_blocks": 1}
    plan2, = H3WindowPlanner().run(2, 0, 1, latent={"samples": av_latent(124)})
    assert plan2["T"] == 37 and plan2["num_windows"] == 2
    with pytest.raises(ValueError):   # overlap >= window length: same error as in the refine node
        H3WindowPlanner().run(0, 2, 5, latent={"samples": av_latent(124)})


def test_wired_plan_wins_over_widgets():
    assert apply_plan(None, 2, 0, 1) == (2, 0, 1)
    assert apply_plan({"num_windows": 4, "window_blocks": 0, "overlap_blocks": 2}, 2, 0, 1) == (4, 0, 2)
    with pytest.raises(ValueError, match="windows"):
        apply_plan({"x": 1}, 2, 0, 1)
