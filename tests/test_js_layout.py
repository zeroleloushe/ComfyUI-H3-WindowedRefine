"""JS layout (web/js/h3wr_layout.js) must match h3_windows.plan_windows exactly."""
import itertools
import json
import os
import shutil
import subprocess
import tempfile

import pytest
from h3wr.h3_windows import plan_windows, audio_len_for_frames, frames_for_latent_t

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = shutil.which("node")


def py_case(T, nw, wb, ob):
    try:
        return [[w.v0, w.v1] for w in plan_windows(T, audio_len_for_frames(frames_for_latent_t(T)), nw, wb, ob)]
    except ValueError:
        return None


@pytest.mark.skipif(NODE is None, reason="node.js not installed")
def test_js_matches_python():
    cases = [(T, nw, wb, ob) for T, nw, wb, ob in itertools.product(
        range(2, 213, 5), [0, 1, 2, 3, 4, 5, 6, 8, 11], [0, 1, 2, 3, 5, 8, 13], [0, 1, 2, 3, 5])]
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(os.path.join(PKG, "web", "js", "h3wr_layout.js"), os.path.join(d, "layout.mjs"))
        with open(os.path.join(d, "cases.json"), "w") as f:
            json.dump(cases, f)
        script = os.path.join(d, "run.mjs")
        with open(script, "w") as f:
            f.write("import {planWindows, latentFor, nearestValidFrames, framesFor} from './layout.mjs';\n"
                    "import fs from 'fs';\n"
                    "const cases = JSON.parse(fs.readFileSync(new URL('./cases.json', import.meta.url)));\n"
                    "const out = cases.map(([T,nw,wb,ob]) => { const r = planWindows(T,nw,wb,ob); return r.windows || null; });\n"
                    "const lat = []; for (let f = 1; f < 800; f++) lat.push([latentFor(f), nearestValidFrames(f), framesFor(latentFor(f))]);\n"
                    "console.log(JSON.stringify({out, lat}));\n")
        res = json.loads(subprocess.check_output([NODE, script], text=True))
    mism = [(c, p, j) for c, j in zip(cases, res["out"]) if (p := py_case(*c)) != j]
    assert not mism, f"{len(mism)} mismatches, e.g. {mism[:3]}"
    for f, (lt, nv, fb) in zip(range(1, 800), res["lat"]):
        assert lt == (2 if f <= 5 else ((f - 5) // 17) * 5 + 2)  # nodes_minimax_h3.video_latent_t
        assert nv["lower"] % 17 == 5 and nv["higher"] % 17 == 5 and nv["lower"] <= max(f, 5) <= nv["higher"]
        assert nv["nearest"] in (nv["lower"], nv["higher"]) and nv["valid"] == ((max(5, f) - 5) % 17 == 0)
    assert len(cases) > 2000
