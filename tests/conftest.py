import importlib.util
import os
import sys

COMFY = os.environ.get("COMFYUI_PATH", "/workspace/ComfyUI")
sys.path.insert(0, COMFY)
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("h3wr", os.path.join(PKG, "__init__.py"), submodule_search_locations=[PKG])
mod = importlib.util.module_from_spec(spec)
sys.modules["h3wr"] = mod
spec.loader.exec_module(mod)
