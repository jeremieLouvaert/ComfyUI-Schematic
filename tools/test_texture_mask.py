"""
Implementation-blind teeth for the texture_mask addition to the shipped
Schematic Overlay node (docs/texture-mask-addition.md, section 4, items a-f).

Node-level only, per the spec doc's own framing ("byte-level guarantees...
each a named scalar"): drives the real SchematicOverlay.execute via the
embedded python (torch required), using the same package shim as
tools/smoke_node.py. No pytest; prints PASS/FAIL per check and exits nonzero
if any check fails.

FORBIDDEN to read as source: nodes/schematic_overlay.py, schematic/render.py,
schematic/draw_common.py, schematic/engine.py, or any other implementation
file. This suite only drives the real node's execute() as a black box.

Items a-f map onto sections below:
  a -> covered by tools/test_byteident.py (suite zero); SKIPPED here (as instructed).
  b -> [b]   c -> [c]   d -> [d]   e -> [e]   f -> note only, see bottom.
"""

import os
import sys

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)

import numpy as np
import torch
import importlib.util
import types

# package shim so the node's relative imports resolve without ComfyUI
# (identical pattern to tools/smoke_node.py)
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

PASS = True


def check(name, cond, detail=""):
    global PASS
    status = "PASS" if cond else "FAIL"
    if not cond:
        PASS = False
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


# ===========================================================================
# item (a): None-path byte-identity against suite zero's goldens is the job of
# tools/test_byteident.py --capture / --check (texture-mask-addition.md S5),
# which drives the real node across the G1-G6 config set and diffs md5 hashes
# vs a pre-change HEAD. Not reimplemented here per the task instructions
# ("is covered by the golden harness, skip it").
print("[a] None-path byte-identity vs the pre-change golden hashes: covered by "
      "tools/test_byteident.py (--capture/--check). SKIPPED here.")


# ===========================================================================
node = SchematicOverlay()
it = node.INPUT_TYPES()


def extract_defaults(input_types):
    """Same generic extraction as tools/smoke_node.py."""
    d = {}
    for section in ("required", "optional"):
        for k, v in input_types.get(section, {}).items():
            if isinstance(v, tuple) and len(v) > 1 and isinstance(v[1], dict) and "default" in v[1]:
                d[k] = v[1]["default"]
            elif isinstance(v, tuple) and isinstance(v[0], list):
                d[k] = v[0][0]
    return d


defaults = extract_defaults(it)
for sock in ("image", "pixelate_mask", "texture", "texture_mask"):
    defaults.pop(sock, None)

# Chain is drawn AFTER texture (shipped node S7 step 9, "drawn ON TOP of
# texture"). Left on, chain pixels would overwrite the texture-blended result
# wherever chain circles/labels land, masking texture_mask's own effect there
# and weakening items (d)/(e)'s locality claims without indicating any real
# bug. Disabled for these tests only -- the one deliberate deviation from
# "full widget list defaults", done purely for test isolation.
defaults["chain_enabled"] = False

H, W = 300, 400  # "small canvas" per the task; size_preset default "match input"
                 # makes canvas == input image size exactly (no cover-fit crop).

rng = np.random.default_rng(20260808)
img_np = rng.random((1, H, W, 3)).astype(np.float32)
# BRIGHT texture (near-white, slight per-pixel variation) -- required by item
# (e) so the screen-blend moves pixels by a large, unambiguous margin.
tex_np = (0.8 + 0.2 * rng.random((1, H, W, 3))).astype(np.float32)

img = torch.from_numpy(img_np)
tex = torch.from_numpy(tex_np)

ones_mask = torch.ones((1, H, W), dtype=torch.float32)
zeros_mask = torch.zeros((1, H, W), dtype=torch.float32)
half_mask = torch.zeros((1, H, W), dtype=torch.float32)
SPLIT = W // 2  # exact-canvas-size mask, no resample band -> a crisp 0/1 boundary
half_mask[:, :, :SPLIT] = 1.0

TEXTURE_OPACITY = 0.7


def run(mask, opacity=TEXTURE_OPACITY):
    conf = dict(defaults)
    conf["texture_opacity"] = opacity
    return node.execute(image=img, pixelate_mask=None, texture=tex, texture_mask=mask, **conf)


# ===========================================================================
print("\n[b] Ones-mask bitwise == mask omitted")

out_omitted = run(None)
out_ones = run(ones_mask)

check("image bitwise equal (ones-mask vs texture_mask omitted)",
      torch.equal(out_omitted[0], out_ones[0]))
check("overlay_only bitwise equal (ones-mask vs texture_mask omitted)",
      torch.equal(out_omitted[1], out_ones[1]))
check("overlay_alpha bitwise equal (ones-mask vs texture_mask omitted; texture never "
      "contributes to alpha coverage)", torch.equal(out_omitted[2], out_ones[2]))


# ===========================================================================
print("\n[c] Zeros-mask bitwise == texture_opacity=0 render")

# nonzero opacity on the zeros-mask call: proves the MASK (not an incidentally
# zero opacity) is what suppresses the effect.
out_zeros = run(zeros_mask, opacity=TEXTURE_OPACITY)
out_opt0 = run(None, opacity=0.0)

check("image bitwise equal (zeros-mask@opacity=0.7 vs texture_mask=None@opacity=0)",
      torch.equal(out_zeros[0], out_opt0[0]))
check("overlay_only bitwise equal (zeros-mask@opacity=0.7 vs texture_mask=None@opacity=0)",
      torch.equal(out_zeros[1], out_opt0[1]))
check("overlay_alpha bitwise equal (zeros-mask@opacity=0.7 vs texture_mask=None@opacity=0)",
      torch.equal(out_zeros[2], out_opt0[2]))


# ===========================================================================
print("\n[d] Locality: exact-canvas-size half mask -- m=1 side == ones-render, "
      "m=0 side == zeros-render")

out_half = run(half_mask)

img_half, ov_half, alpha_half = out_half
img_ones, ov_ones, alpha_ones = out_ones
img_zeros, ov_zeros, alpha_zeros = out_zeros

m1_ok_img = torch.equal(img_half[:, :, :SPLIT, :], img_ones[:, :, :SPLIT, :])
m0_ok_img = torch.equal(img_half[:, :, SPLIT:, :], img_zeros[:, :, SPLIT:, :])
check("image: m=1 region bitwise equals the ones-render", m1_ok_img)
check("image: m=0 region bitwise equals the zeros-render", m0_ok_img)

m1_ok_ov = torch.equal(ov_half[:, :, :SPLIT, :], ov_ones[:, :, :SPLIT, :])
m0_ok_ov = torch.equal(ov_half[:, :, SPLIT:, :], ov_zeros[:, :, SPLIT:, :])
check("overlay_only: m=1 region bitwise equals the ones-render", m1_ok_ov)
check("overlay_only: m=0 region bitwise equals the zeros-render", m0_ok_ov)

check("overlay_alpha unchanged by texture_mask (half-mask vs ones-mask, full-frame "
      "bitwise; texture never contributes to alpha)", torch.equal(alpha_half, alpha_ones))


# ===========================================================================
print("\n[e] NEGATIVE CONTROL: bright texture + texture_opacity>0 -- half-mask differs")
print("    from ones-mask exactly and only on the m=0 side (the moved scalar)")

delta = (img_half - img_ones).abs()
delta_m1 = delta[:, :, :SPLIT, :]
delta_m0 = delta[:, :, SPLIT:, :]
max_delta_m1 = float(delta_m1.max())
max_delta_m0 = float(delta_m0.max())

check("image: max|delta| over the m=1 region == 0 (untouched side)", max_delta_m1 == 0.0,
      f"max|delta|={max_delta_m1:.3e}")
check("image: max|delta| over the m=0 region > 0 (the moved scalar; teeth present)",
      max_delta_m0 > 0.0, f"max|delta|={max_delta_m0:.3e}")

# step 8 (the texture blend) runs identically for overlay_only per the doc
# ("Applies identically to... overlay_only (step 8 runs there)") -- same
# negative control, second output.
delta_ov = (ov_half - ov_ones).abs()
max_delta_ov_m1 = float(delta_ov[:, :, :SPLIT, :].max())
max_delta_ov_m0 = float(delta_ov[:, :, SPLIT:, :].max())
check("overlay_only: max|delta| over the m=1 region == 0 (untouched side)",
      max_delta_ov_m1 == 0.0, f"max|delta|={max_delta_ov_m1:.3e}")
check("overlay_only: max|delta| over the m=0 region > 0 (the moved scalar; teeth present)",
      max_delta_ov_m0 > 0.0, f"max|delta|={max_delta_ov_m0:.3e}")


# ===========================================================================
# item (f): tools/test_schematic.py staying green throughout this phase (every
# touch: draw_common extraction, texture_mask, each new node landing) is
# asserted by the CALLER as a separate CI/verification step, not by this file.
# test_texture_mask.py only exercises the new socket in isolation.
print("\n[f] tools/test_schematic.py staying green throughout this phase is asserted "
      "by the caller (a separate run of that suite), not by this file.")


# ===========================================================================
print()
if PASS:
    print("ALL CHECKS PASSED")
    sys.exit(0)
else:
    print("SOME CHECKS FAILED")
    sys.exit(1)
