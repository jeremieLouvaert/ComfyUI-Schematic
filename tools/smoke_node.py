"""Node-level smoke: drive SchematicOverlay.execute with real torch tensors.
Run with the ComfyUI embedded python (needs torch)."""
import sys, os, time
import numpy as np

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)

import torch
import importlib.util, types

# package shim so the node's relative imports resolve without ComfyUI
pkg = types.ModuleType("cs_pkg")
pkg.__path__ = [PACK]
sys.modules["cs_pkg"] = pkg
for sub in ("schematic", "nodes"):
    spec = importlib.util.spec_from_file_location(
        f"cs_pkg.{sub}", os.path.join(PACK, sub, "__init__.py"),
        submodule_search_locations=[os.path.join(PACK, sub)])
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"cs_pkg.{sub}"] = m
    spec.loader.exec_module(m)

SchematicOverlay = sys.modules["cs_pkg.nodes"].NODE_CLASS_MAPPINGS["SchematicOverlay"]

rng = np.random.default_rng(7)
img = torch.from_numpy(rng.random((2, 512, 768, 3), dtype=np.float32))  # batch of 2
mask = torch.zeros((1, 512, 768))
mask[0, 100:180, 200:280] = 1.0

node = SchematicOverlay()
defaults = {}
it = node.INPUT_TYPES()
for section in ("required", "optional"):
    for k, v in it.get(section, {}).items():
        if isinstance(v, tuple) and len(v) > 1 and isinstance(v[1], dict) and "default" in v[1]:
            defaults[k] = v[1]["default"]
        elif isinstance(v, tuple) and isinstance(v[0], list):
            defaults[k] = v[0][0]
defaults.pop("image", None); defaults.pop("pixelate_mask", None); defaults.pop("texture", None)

t0 = time.time()
out_img, out_overlay, out_alpha = node.execute(image=img, pixelate_mask=mask, texture=None, **defaults)
dt = time.time() - t0

assert out_img.shape == (2, 512, 768, 3), out_img.shape
assert out_overlay.shape == (2, 512, 768, 3), out_overlay.shape
assert out_alpha.shape[0] == 2 and out_alpha.shape[1:] == (512, 768), out_alpha.shape
assert torch.isfinite(out_img).all() and out_img.min() >= 0 and out_img.max() <= 1
assert out_alpha.min() >= 0 and out_alpha.max() <= 1
assert (out_img[0] - out_img[1]).abs().max() > 0  # different frames -> different detection
print(f"[PASS] node smoke: batch=2 512x768, mask zone, 3 outputs OK, {dt:.2f}s total")

# determinism at the node level
out2 = node.execute(image=img, pixelate_mask=mask, texture=None, **defaults)[0]
assert torch.equal(out_img, out2)
print("[PASS] node determinism: identical rerun")

# size preset path
defaults2 = dict(defaults); defaults2["size_preset"] = "Square (1080x1080)"
o3 = node.execute(image=img[:1], pixelate_mask=None, texture=None, **defaults2)[0]
assert o3.shape == (1, 1080, 1080, 3), o3.shape
print("[PASS] size preset: 1080x1080 canvas from 768x512 input")
print("ALL SMOKE PASS")
