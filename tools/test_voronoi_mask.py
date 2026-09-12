"""Blind-teeth suite for SchematicVoronoi's density_mask input (docs/
voronoi-density-mask-derivation.md, section 6, invariants V2-V13; V1 lives in
tools/test_byteident.py's extension, V11's node-count row is here too).
Implementation-blind: written from the spec doc ONLY, calling schematic.voronoi
/ schematic.engine / the node's execute() as declared black-box seams.
No pytest; prints PASS/FAIL/CTRL-FIRED/CTRL-SILENT per row and exits nonzero
on any FAIL or any silent (non-firing) control. Run with the ComfyUI embedded
python (needs torch+scipy).

Rows:
 V2   mask-follows, hardened thresholds (ramp mean_x/w, decile ratios) +
      inverted control + gamma-dropping/mask-ignoring evasion controls.
 V3   uniform mask eaten by normalisation (0.25 === ones); no-norm control
      differs in POSITIONS, not counts (v2 correction).
 V4   density_source inert under a wired mask at FULL-OUTPUT level; no-mask
      control shows sources differing.
 V5   gamma law vs a float32-socket-round-trip f64 oracle (<=1e-15); wrong-
      exponent control.
 V6   node-level I/O rows: 2D/(1,1,H,W)/(H,W,1)/(1,H,W,1) rescued identical;
      (H,W,3) and M=0 treated absent-with-warn; NaN-scrub; M=2 batch frame-0
      note; bool/f16/f64/CUDA-resident torch masks accepted.
 V7   all-zero mask at mask_min=0 -> mesh-free path + console note;
      mask_min=0.01 -> uniform mesh forms.
 V7b  fmax guard boundary pinned both sides (1e-12, 5e-10 degenerate;
      2e-9 not); no-guard control gives a full mesh at 5e-10.
 V8   field_at_seeds follows the mask (module-level, mask_min=0); luma-read
      control differs.
 V9   determinism; sparse-mask budget shortfall (console line); dense mask
      reaches target with no line; 5%-mask control does not fire.
 V10  cover-fit geometry incl. the peak-cropped ordering witness and the
      canvas-verbatim row; stretch-oracle control differs.
 V11  loader node count (3) sanity + tools/test_voronoi.py stays green.
 V12  batch collapse under a wired mask (masked frames identical, unmasked
      control differs).
 V13  floor law: mask_min=0 bitwise==unfloored, mask_min=1==white-uniform,
      f64 remap oracle, the pinned 640x480 radial-mask ink test (holes closed,
      fine:coarse ratio); floor-before-gamma + no-floor controls.
"""
import contextlib
import io
import os
import subprocess
import sys

import numpy as np
import torch
import importlib.util, types

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)

from schematic import voronoi, engine  # pure, black-box per spec section 7

# package shim so the node's relative imports resolve without ComfyUI (smoke_node.py pattern)
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

SchematicVoronoi = sys.modules["cs_pkg.nodes"].NODE_CLASS_MAPPINGS["SchematicVoronoi"]

PASS = True
CONTROLS_OK = True


def check(name, cond, detail=""):
    global PASS
    status = "PASS" if cond else "FAIL"
    if not cond:
        PASS = False
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


def control(name, fired, detail=""):
    """A control MUST fire (cond True). A silent control is as bad as a FAIL."""
    global CONTROLS_OK
    status = "CTRL-FIRED" if fired else "CTRL-SILENT(BAD)"
    if not fired:
        CONTROLS_OK = False
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return fired


def section(t):
    print(f"\n---- {t} ----")


def node_defaults():
    """Programmatic widget defaults from INPUT_TYPES, minus socket inputs (smoke_node.py style)."""
    node = SchematicVoronoi()
    it = node.INPUT_TYPES()
    d = {}
    for sect in ("required", "optional"):
        for k, v in it.get(sect, {}).items():
            if isinstance(v, tuple) and len(v) > 1 and isinstance(v[1], dict) and "default" in v[1]:
                d[k] = v[1]["default"]
            elif isinstance(v, tuple) and isinstance(v[0], list):
                d[k] = v[0][0]
    d.pop("image", None)
    d.pop("texture", None)
    d.pop("density_mask", None)
    return d


NODE = SchematicVoronoi()
NODE_DEFAULTS = node_defaults()


# ===========================================================================
# Readiness gate: fail fast and loud if the seam isn't there yet, instead of
# a raw traceback mid-suite.
# ===========================================================================
def _ready_probe():
    try:
        voronoi.build_density(np.zeros((8, 8, 3), dtype=np.float32), "brightness", 1.0,
                               mask01=np.zeros((8, 8)), mask_min=0.0)
    except TypeError as e:
        return False, f"schematic.voronoi.build_density rejects mask01/mask_min: {e}"
    p = dict(NODE_DEFAULTS)
    p["cells"] = 10
    p["mask_min"] = 0.0
    small_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    small_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
    try:
        NODE.execute(image=small_img, texture=None, density_mask=small_mask, **p)
    except TypeError as e:
        return False, f"SchematicVoronoi.execute rejects density_mask/mask_min: {e}"
    return True, ""


ready, why = _ready_probe()
if not ready:
    print("BUILD NOT READY")
    print(f"  reason: {why}")
    sys.exit(1)


# ===========================================================================
# Shared helpers (test-side scaffolding only; the actual node-shape bridging
# lives inside the node and is exercised directly at node-level in V6/V7/V9/
# V12/V13; these module-level helpers reproduce the DOCUMENTED section-3/4
# input-shaping so module-level rows can call the real build_density/sample_
# seeds with a canvas-mapped mask, per the declared seam).
# ===========================================================================

def synth_photo(w, h, seed):
    """Deterministic structured RGB content (gradient + blobs + noise), same
    family as tools/test_byteident.py's synth_image -- real structure so
    source-dependent (brightness/darkness/detail) checks have something to
    differ on."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 0.25 + 0.5 * (xx / max(w - 1, 1)) * (yy / max(h - 1, 1))
    img = np.stack([base, base * 0.9, base * 1.1], axis=-1)
    for _ in range(6):
        cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
        s = rng.uniform(0.03, 0.10) * min(w, h)
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)))
        img += blob[..., None] * rng.uniform(0.3, 0.7)
    img += rng.normal(0, 0.04, size=(h, w, 3)).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def ramp(w, h):
    """Horizontal ramp: varies 0..1 across x, constant across y."""
    return np.tile(((np.arange(w) + 0.5) / w).astype(np.float64), (h, 1))


def prep_mask_arr(dm):
    """Spec section 4 (v2), reproduced at the test level with allowed np/torch
    primitives only: squeeze trailing size-1 dims -> reshape(-1,H,W) -> empty
    guard -> frame 0 -> nan_to_num -> clip -> >=8-both-dims plausibility guard.
    Float32 first (the socket dtype) -- this is test scaffolding for feeding
    module-level calls; the node's OWN bridging is exercised directly at
    node-level in the V6/V7/V9/V12/V13 rows below."""
    if dm is None:
        return None
    if hasattr(dm, "cpu"):
        dm = dm.cpu().numpy()
    dm = np.asarray(dm, dtype=np.float32)
    while dm.ndim > 2 and dm.shape[-1] == 1:
        dm = dm[..., 0]
    if dm.ndim < 2:
        return None
    dm = dm.reshape((-1,) + dm.shape[-2:])
    if dm.shape[0] == 0:
        return None
    m = np.nan_to_num(dm[0], nan=0.0)
    if m.shape[0] < 8 or m.shape[1] < 8:
        return None
    return np.clip(m.astype(np.float64), 0.0, 1.0)


def canvas_mask_arr(m, w, h):
    """Spec section 3: as-is at canvas size, else cover_fit like the photo."""
    if m.shape[0] == h and m.shape[1] == w:
        return m
    fitted = engine.cover_fit(np.repeat(m[..., None], 3, axis=2).astype(np.float32), w, h)
    return np.clip(fitted[..., 0].astype(np.float64), 0.0, 1.0)


def density_for(mask, w, h, source="brightness", gamma=2.5, mask_min=0.0, fitted=None):
    """Prep + canvas-map the mask exactly per spec sections 3-4, then call the
    REAL voronoi.build_density with mask01/mask_min -- the declared seam."""
    m = prep_mask_arr(mask)
    if m is not None:
        m = canvas_mask_arr(m, w, h)
    if fitted is None:
        fitted = np.zeros((h, w, 3), dtype=np.float32)
    return voronoi.build_density(fitted, source, gamma, mask01=m, mask_min=mask_min)


def seeds_for(mask, w, h, source="brightness", gamma=2.5, cells=3000, seed=42,
              mask_min=0.0, fitted=None):
    d = density_for(mask, w, h, source, gamma, mask_min, fitted)
    return voronoi.sample_seeds(d, cells, seed, w, h), d


def to2d(a):
    return a[..., 0] if a.ndim == 3 else a


def zero_ink_blocks(a, blk=16):
    a = to2d(a)
    bh, bw = a.shape[0] // blk, a.shape[1] // blk
    g = a[:bh * blk, :bw * blk].reshape(bh, blk, bw, blk).max(axis=(1, 3))
    return float((g <= 0.0).mean()) * 100, g


W0, H0 = 320, 240
FLAT0 = np.full((H0, W0, 3), 0.5, dtype=np.float32)


# ===========================================================================
section("V2  mask-follows (v2-hardened: catches mask-ignoring AND gamma-dropping builds)")
# pinned config: flat grey photo, horizontal-ramp mask, gamma 2.5, cells 3000, seed 42, 320x240
s_ramp, _ = seeds_for(ramp(W0, H0).astype(np.float32), W0, H0, gamma=2.5, cells=3000, seed=42, fitted=FLAT0)
mean_x = float(s_ramp[:, 0].mean()) / W0
check("V2 ramp mean_x/w >= 0.72", mean_x >= 0.72, f"mean_x/w={mean_x:.4f} n={len(s_ramp)}")
histo, _ = np.histogram(s_ramp[:, 0], bins=10, range=(0, W0))
check("V2 decile ratio h9 > 3*h4", int(histo[9]) > 3 * int(histo[4]), f"h9={histo[9]} h4={histo[4]}")
check("V2 decile ratio 3*h4 > 10*h1", 3 * int(histo[4]) > 10 * int(histo[1]), f"h4={histo[4]} h1={histo[1]}")

s_inv, _ = seeds_for((1.0 - ramp(W0, H0)).astype(np.float32), W0, H0, gamma=2.5, cells=3000, seed=42, fitted=FLAT0)
mean_xi = float(s_inv[:, 0].mean()) / W0
control("V2 inverted ramp flips: mean_x/w <= 0.28", mean_xi <= 0.28, f"mean_x/w={mean_xi:.4f}")

# evasion builds must fail the main clauses (proves the thresholds have teeth)
s_g1 = voronoi.sample_seeds(np.clip(ramp(W0, H0), 0, 1), 3000, 42, W0, H0)  # gamma dropped
mean_g1 = float(s_g1[:, 0].mean()) / W0
control("V2 a gamma-dropping build fails the 0.72 bar", mean_g1 < 0.72, f"mean={mean_g1:.4f}")
s_uni = voronoi.sample_seeds(np.ones((H0, W0)), 3000, 42, W0, H0)  # mask ignored
h_uni, _ = np.histogram(s_uni[:, 0], bins=10, range=(0, W0))
control("V2 a mask-ignoring (uniform) build fails the decile ratios",
        not (int(h_uni[9]) > 3 * int(h_uni[4]) and 3 * int(h_uni[4]) > 10 * int(h_uni[1])),
        f"h9={h_uni[9]} h4={h_uni[4]} h1={h_uni[1]}")


# ===========================================================================
section("V3  uniform mask eaten by normalisation (the section-9g named NC)")
s_q, d_q = seeds_for(np.full((H0, W0), 0.25, dtype=np.float32), W0, H0, gamma=2.5, cells=3000, seed=42, fitted=FLAT0)
s_1, d_1 = seeds_for(np.ones((H0, W0), dtype=np.float32), W0, H0, gamma=2.5, cells=3000, seed=42, fitted=FLAT0)
check("V3 0.25-const seeds ARRAY-IDENTICAL to ones", np.array_equal(s_q, s_1), f"n={len(s_q)}")
check("V3 both densities are all-ones", bool((d_q == 1.0).all() and (d_1 == 1.0).all()))


def density_nonorm(mask01, gamma):
    return np.power(np.clip(np.asarray(mask01, dtype=np.float64), 0.0, 1.0), gamma)


s_bad = voronoi.sample_seeds(density_nonorm(np.full((H0, W0), 0.25), 2.5), 3000, 42, W0, H0)
check("V3 skip-normalisation COUNTS equal (v1's 4x-fewer claim withdrawn)",
      len(s_bad) == len(s_q), f"n_bad={len(s_bad)} vs {len(s_q)}")
control("V3 skip-normalisation build differs in POSITIONS (the whole control)",
        not np.array_equal(s_bad, s_q))


# ===========================================================================
section("V4  density_source inert under a wired mask (FULL-OUTPUT level, node-level)")
PHOTO4 = synth_photo(W0, H0, 11)
mask4 = torch.from_numpy(ramp(W0, H0).astype(np.float32))[None, ...]
img4 = torch.from_numpy(PHOTO4)[None, ...]
outs_by_src4 = {}
for src in ("brightness", "darkness", "detail"):
    p4 = dict(NODE_DEFAULTS)
    p4.update(density_source=src, cells=1500, mask_min=0.0)
    outs_by_src4[src] = NODE.execute(image=img4, texture=None, density_mask=mask4, **p4)
same01 = all(torch.equal(outs_by_src4["brightness"][k], outs_by_src4["darkness"][k]) for k in range(3))
same12 = all(torch.equal(outs_by_src4["darkness"][k], outs_by_src4["detail"][k]) for k in range(3))
check("V4 all three FULL OUTPUTS byte-identical across density_source under a wired mask",
      same01 and same12)

outs_nomask4 = {}
for src in ("brightness", "darkness"):
    p4b = dict(NODE_DEFAULTS)
    p4b.update(density_source=src, cells=1500)
    outs_nomask4[src] = NODE.execute(image=img4, texture=None, **p4b)
control("V4 density_mask=None: brightness vs darkness differ (existing behaviour as control)",
        not torch.equal(outs_nomask4["brightness"][0], outs_nomask4["darkness"][0]))


# ===========================================================================
section("V5  gamma law (float32-socket-round-trip oracle)")
m5 = ramp(W0, H0).astype(np.float32)
d3 = density_for(m5, W0, H0, gamma=3.0, mask_min=0.0, fitted=FLAT0)
m5_f64 = m5.astype(np.float64)  # rounded through float32 first, per the socket
oracle5 = np.power(np.clip(m5_f64 / m5_f64.max(), 0.0, 1.0), 3.0)
err5 = float(np.abs(d3 - oracle5).max())
check("V5 build_density(mask01=..) == f64 oracle rounded through f32 socket first (gamma=3)",
      err5 <= 1e-15, f"max|d|={err5:.2e}")
s_g3 = voronoi.sample_seeds(d3, 3000, 42, W0, H0)
s_or5 = voronoi.sample_seeds(oracle5, 3000, 42, W0, H0)
check("V5 node-shaped seeds equal oracle-driven seeds", np.array_equal(s_g3, s_or5))
s_g2 = voronoi.sample_seeds(np.power(np.clip(m5_f64 / m5_f64.max(), 0, 1), 2.0), 3000, 42, W0, H0)
control("V5 wrong exponent (gamma=2) differs", not np.array_equal(s_g3, s_g2),
        f"n3={len(s_g3)} n2={len(s_g2)}")


# ===========================================================================
section("V6  I/O rows (node-level; the node's own shape/dtype bridging)")
W6, H6 = 64, 64  # comfortably clears the >=8-both-dims plausibility guard
photo6 = synth_photo(W6, H6, 31)
img6 = torch.from_numpy(photo6)[None, ...]
base_mask6 = ramp(W6, H6).astype(np.float32)
p6 = dict(NODE_DEFAULTS)
p6.update(cells=500, mask_min=0.0)


def run6(mask_tensor):
    return NODE.execute(image=img6, texture=None, density_mask=mask_tensor, **p6)


ref6 = run6(torch.from_numpy(base_mask6)[None, ...])          # canonical (1,H,W)
out_absent6 = NODE.execute(image=img6, texture=None, **p6)     # density_mask entirely absent

r2d6 = run6(torch.from_numpy(base_mask6))                      # 2-D, no batch dim
check("V6 2-D mask == (1,H,W) reference", torch.equal(r2d6[1], ref6[1]) and torch.equal(r2d6[2], ref6[2]))

r4d6 = run6(torch.from_numpy(base_mask6)[None, None, ...])     # (1,1,H,W)
check("V6 (1,1,H,W) mask == reference", torch.equal(r4d6[1], ref6[1]))

r_hw1_6 = run6(torch.from_numpy(base_mask6)[..., None])        # (H,W,1)
check("V6v2 (H,W,1) mask rescued by squeeze == reference", torch.equal(r_hw1_6[1], ref6[1]))

r_1hw1_6 = run6(torch.from_numpy(base_mask6)[None, ..., None])  # (1,H,W,1)
check("V6v2 (1,H,W,1) mask rescued by squeeze == reference", torch.equal(r_1hw1_6[1], ref6[1]))

mask_rgb6 = np.repeat(base_mask6[..., None], 3, axis=2)         # (H,W,3): NOT a mask
buf_rgb6 = io.StringIO()
with contextlib.redirect_stdout(buf_rgb6):
    r_rgb6 = run6(torch.from_numpy(mask_rgb6))
check("V6v2 (H,W,3) mask treated as absent (== no-mask output)", torch.equal(r_rgb6[1], out_absent6[1]))
check("V6v2 (H,W,3) mask prints a [Schematic] warn", "[Schematic]" in buf_rgb6.getvalue(),
      f"captured={buf_rgb6.getvalue()!r:.200}")

buf_empty6 = io.StringIO()
with contextlib.redirect_stdout(buf_empty6):
    r_empty6 = run6(torch.zeros((0, H6, W6), dtype=torch.float32))  # M=0
check("V6 M=0 empty batch treated as absent (== no-mask output)", torch.equal(r_empty6[1], out_absent6[1]))
check("V6 M=0 prints a [Schematic] warn", "[Schematic]" in buf_empty6.getvalue(),
      f"captured={buf_empty6.getvalue()!r:.200}")
try:
    _ = np.zeros((0, H6, W6), dtype=np.float32)[0]
    control("V6 the banked bare dm[0] on an (0,H,W) array raises IndexError", False)
except IndexError:
    control("V6 the banked bare dm[0] on an (0,H,W) array raises IndexError", True,
            "IndexError (the guard is not optional)")

mask_nan6 = base_mask6.copy(); mask_nan6[:10, :10] = np.nan
mask_zeroed6 = base_mask6.copy(); mask_zeroed6[:10, :10] = 0.0
r_nan6 = run6(torch.from_numpy(mask_nan6))
r_zeroed6 = run6(torch.from_numpy(mask_zeroed6))
check("V6 NaN-pixel mask == hand-zeroed mask (nan_to_num)", torch.equal(r_nan6[1], r_zeroed6[1]))

mask_batch6 = torch.from_numpy(np.stack([base_mask6, np.zeros_like(base_mask6)]))  # M=2
buf_batch6 = io.StringIO()
with contextlib.redirect_stdout(buf_batch6):
    r_batch6 = run6(mask_batch6)
check("V6 M=2 batch mask uses frame 0 (== single-frame-0 reference)", torch.equal(r_batch6[1], ref6[1]))
check("V6 M=2 batch mask prints a [Schematic] frame-0 note", "[Schematic]" in buf_batch6.getvalue(),
      f"captured={buf_batch6.getvalue()!r:.200}")

dm_bad6 = np.asarray(base_mask6[..., None])
bad_frame6 = dm_bad6.reshape((-1,) + dm_bad6.shape[-2:])[0]
control("V6v2 v1's bare reshape on (H,W,1) yields a corrupt (W,1) sliver (the guarded bug)",
        bad_frame6.shape == (W6, 1), f"shape={bad_frame6.shape}")

mask_bool6 = torch.from_numpy(base_mask6 > 0.5)
r_bool6 = run6(mask_bool6)
check("V6 bool torch mask accepted (finite output, no crash)", bool(torch.isfinite(r_bool6[1]).all()))
check("V6 bool torch mask actually influences output (differs from absent)",
      not torch.equal(r_bool6[1], out_absent6[1]))

mask_f16_6 = torch.from_numpy(base_mask6).to(torch.float16)
r_f16_6 = run6(mask_f16_6)
check("V6 float16 torch mask accepted (finite output, no crash)", bool(torch.isfinite(r_f16_6[1]).all()))
check("V6 float16 torch mask actually influences output (differs from absent)",
      not torch.equal(r_f16_6[1], out_absent6[1]))

mask_f64_6 = torch.from_numpy(base_mask6).to(torch.float64)
r_f64_6 = run6(mask_f64_6)
check("V6 float64 torch mask == f32 reference", torch.equal(r_f64_6[1], ref6[1]))

if torch.cuda.is_available():
    mask_cuda6 = torch.from_numpy(base_mask6).cuda()
    r_cuda6 = run6(mask_cuda6)
    check("V6 CUDA-resident mask == CPU result (no crash)", torch.equal(r_cuda6[1].cpu(), ref6[1]))
else:
    print("  [SKIP] V6 CUDA-resident mask row (no CUDA device on this box)")


# ===========================================================================
section("V7  all-zero mask -> degenerate path (mask_min=0) / uniform mesh (mask_min=0.01)")
d7, s7 = None, None
d7 = density_for(np.zeros((H0, W0), dtype=np.float32), W0, H0, gamma=2.5, mask_min=0.0, fitted=FLAT0)
s7 = voronoi.sample_seeds(d7, 500, 42, W0, H0)
check("V7 module-level: zero mask at mask_min=0 -> zero density, zero seeds",
      len(s7) == 0 and bool((d7 == 0).all()))

photo7 = synth_photo(64, 64, 41)
img7 = torch.from_numpy(photo7)[None, ...]
mask7 = torch.zeros((1, 64, 64), dtype=torch.float32)
p7 = dict(NODE_DEFAULTS)
p7.update(cells=500, mask_min=0.0, texture_opacity=0.0, cell_fill=0.0, seed_dots=False,
          image_opacity=0.0, frame_text_tl="", frame_text_tr="", frame_text_bl="", frame_text_br="")
buf7 = io.StringIO()
with contextlib.redirect_stdout(buf7):
    out7 = NODE.execute(image=img7, texture=None, density_mask=mask7, **p7)
check("V7 node: all-zero mask at mask_min=0 executes without raising", out7 is not None)
img7_np = out7[0][0].numpy()
distinct7 = len(set(map(tuple, (img7_np.reshape(-1, 3) * 255).round().astype(int))))
check("V7 node: mesh-free (single uniform flat color, no crash)", distinct7 == 1,
      f"distinct_colors={distinct7}")
check("V7 node: all-zero mask prints a [Schematic] console note", "[Schematic]" in buf7.getvalue(),
      f"captured={buf7.getvalue()!r:.200}")

p7b = dict(NODE_DEFAULTS)
p7b.update(cells=2000, mask_min=0.01)
out7b = NODE.execute(image=img7, texture=None, density_mask=mask7, **p7b)
img7b_np = out7b[0][0].numpy()
distinct7b = len(set(map(tuple, (img7b_np.reshape(-1, 3) * 255).round().astype(int))))
check("V7 node: all-zero mask at mask_min=0.01 -> a mesh forms (not mesh-free)",
      distinct7b > 1, f"distinct_colors={distinct7b}")


# ===========================================================================
section("V7b  fmax guard boundary pinned both sides")
for lvl, expect_zero in ((1e-12, True), (5e-10, True), (2e-9, False)):
    d_lvl = density_for(np.full((H0, W0), lvl, dtype=np.float32), W0, H0, gamma=1.0, mask_min=0.0, fitted=FLAT0)
    is_zero = bool((d_lvl == 0.0).all())
    check(f"V7b uniform {lvl:g} mask -> {'degenerate(all-zero)' if expect_zero else 'normalises(nonzero)'}",
          is_zero == expect_zero, f"d_max={float(d_lvl.max()):.3g}")


def density_noguard(mask01, gamma):
    f = np.clip(np.asarray(mask01, np.float64), 0, 1)
    return np.power(np.clip(f / f.max(), 0, 1), gamma)


s_ng = voronoi.sample_seeds(density_noguard(np.full((H0, W0), 5e-10), 1.0), 3000, 42, W0, H0)
control("V7b a build with no fmax guard gives a FULL mesh on a 5e-10 mask",
        len(s_ng) == 3000, f"n={len(s_ng)}")


# ===========================================================================
section("V8  field_at_seeds follows the mask (module-level, mask_min=0)")
PHOTO8 = synth_photo(W0, H0, 51)
g8 = ramp(W0, H0).astype(np.float32)
mp8 = canvas_mask_arr(prep_mask_arr(g8), W0, H0)
dens8 = voronoi.build_density(PHOTO8, "brightness", 2.5, mask01=mp8, mask_min=0.0)
seeds8 = voronoi.sample_seeds(dens8, 3000, 42, W0, H0)
ys8 = np.clip(seeds8[:, 1].astype(np.int64), 0, H0 - 1)
xs8 = np.clip(seeds8[:, 0].astype(np.int64), 0, W0 - 1)
fas8 = dens8[ys8, xs8]
oracle_dens8 = np.power(np.clip(mp8 / mp8.max(), 0, 1), 2.5)
check("V8 field_at_seeds == independent oracle density at seed positions",
      bool(np.array_equal(fas8, oracle_dens8[ys8, xs8])))
luma_dens8 = voronoi.build_density(PHOTO8, "brightness", 2.5)  # unmasked module default
control("V8 reading field_at_seeds from the unmasked luma field instead differs (mask-perp config)",
        not np.array_equal(fas8, luma_dens8[ys8, xs8]), f"n_seeds={len(seeds8)}")


# ===========================================================================
section("V9  determinism + budget honesty (the 1/80 law)")
sa9, _ = seeds_for(g8, W0, H0, gamma=2.5, cells=3000, seed=42, fitted=PHOTO8)
sb9, _ = seeds_for(g8, W0, H0, gamma=2.5, cells=3000, seed=42, fitted=PHOTO8)
check("V9 same (photo, mask, seed) -> byte-identical seeds", np.array_equal(sa9, sb9))

sparse9 = np.zeros((H0, W0), dtype=np.float32); sparse9[:, :max(1, W0 // 100)] = 1.0
d_sp9 = density_for(sparse9, W0, H0, gamma=1.0, mask_min=0.0, fitted=FLAT0)
s_sp9 = voronoi.sample_seeds(d_sp9, 3000, 42, W0, H0)
mean_d9 = float(d_sp9.mean())
check("V9 sparse (1%-white, gamma=1) mean density < 1/80 (shortfall precondition)",
      mean_d9 < 1.0 / 80.0, f"mean_d={mean_d9:.4f}")
check("V9 sparse mask: n < cells (shortfall fires)", len(s_sp9) < 3000, f"n={len(s_sp9)}/3000")

photo9 = synth_photo(W0, H0, 61)
img9 = torch.from_numpy(photo9)[None, ...]
mask9_sparse = torch.from_numpy(sparse9)[None, ...]
p9 = dict(NODE_DEFAULTS)
p9.update(cells=3000, density_gamma=1.0, mask_min=0.0)
buf9 = io.StringIO()
with contextlib.redirect_stdout(buf9):
    NODE.execute(image=img9, texture=None, density_mask=mask9_sparse, **p9)
check("V9 sparse mask: [Schematic] shortfall console line prints", "[Schematic]" in buf9.getvalue(),
      f"captured={buf9.getvalue()!r:.300}")

dense9 = np.ones((H0, W0), dtype=np.float32)
d_dn9 = density_for(dense9, W0, H0, gamma=1.0, mask_min=0.0, fitted=FLAT0)
s_dn9 = voronoi.sample_seeds(d_dn9, 3000, 42, W0, H0)
control("V9 dense mask reaches the target count (no shortfall)", len(s_dn9) == 3000, f"n={len(s_dn9)}")

mask9_dense = torch.from_numpy(dense9)[None, ...]
buf9b = io.StringIO()
with contextlib.redirect_stdout(buf9b):
    NODE.execute(image=img9, texture=None, density_mask=mask9_dense, **p9)
# the per-call "[Schematic] Schematic Voronoi: source=..." info line always prints;
# the shortfall line specifically must NOT (the teeth is the absence of THAT line)
check("V9 dense mask: NO budget-exhausted shortfall line", "budget exhausted" not in buf9b.getvalue(),
      f"captured={buf9b.getvalue()!r:.300}")

five_pct9 = np.zeros((H0, W0), dtype=np.float32); five_pct9[:, :max(1, int(W0 * 0.05))] = 1.0
d_5pct9 = density_for(five_pct9, W0, H0, gamma=1.0, mask_min=0.0, fitted=FLAT0)
s_5pct9 = voronoi.sample_seeds(d_5pct9, 3000, 42, W0, H0)
control("V9 a 5% mask (>1/80) does NOT fire the shortfall (reaches cells)",
        len(s_5pct9) == 3000, f"n={len(s_5pct9)} mean_d={float(d_5pct9.mean()):.4f}")


# ===========================================================================
section("V10  cover-fit geometry (peak-cropped witness + canvas-verbatim)")
w10, h10 = 350, 500  # portrait canvas; source mask built at 320x240 landscape (aspect mismatch)
blob10 = np.zeros((H0, W0), dtype=np.float32); blob10[:, W0 // 2 - 20:W0 // 2 + 20] = 1.0
fitted10 = np.zeros((h10, w10, 3), dtype=np.float32)

s_cf10, _ = seeds_for(blob10, w10, h10, gamma=1.0, mask_min=0.0, fitted=fitted10)
m_or10 = engine.cover_fit(np.repeat(blob10[..., None], 3, axis=2).astype(np.float32), w10, h10)[..., 0]
d_or10 = voronoi.build_density(fitted10, "brightness", 1.0,
                                mask01=np.clip(m_or10.astype(np.float64), 0, 1), mask_min=0.0)
s_or10 = voronoi.sample_seeds(d_or10, 3000, 42, w10, h10)
check("V10 cover-fit path == independent cover_fit oracle", np.array_equal(s_cf10, s_or10), f"n={len(s_cf10)}")

mc10 = ramp(w10, h10).astype(np.float32)
mc10_64 = mc10.astype(np.float64)
d_v10 = density_for(mc10, w10, h10, gamma=1.0, mask_min=0.0, fitted=fitted10)
check("V10 canvas-sized mask used verbatim (no resample; == raw normalise)",
      bool(np.array_equal(d_v10, np.clip(mc10_64 / mc10_64.max(), 0, 1))))

m_st10 = engine.stretch_resize(np.repeat(blob10[..., None], 3, axis=2).astype(np.float32), w10, h10)[..., 0]
d_st10 = voronoi.build_density(fitted10, "brightness", 1.0,
                                mask01=np.clip(m_st10.astype(np.float64), 0, 1), mask_min=0.0)
s_st10 = voronoi.sample_seeds(d_st10, 3000, 42, w10, h10)
control("V10 the banked stretch-resample geometry differs at aspect mismatch",
        not np.array_equal(s_cf10, s_st10), f"cf_n={len(s_cf10)} st_n={len(s_st10)}")

# v2 peak-cropped witness: pins the canvas-map-THEN-normalise order
peak10 = np.zeros((H0, W0), dtype=np.float32); peak10[:, :30] = 1.0; peak10[:, 30:] = 0.3
d_after10 = density_for(peak10, w10, h10, gamma=1.0, mask_min=0.0, fitted=fitted10)  # spec order

m_pk_prep10 = prep_mask_arr(peak10)
norm_first10 = m_pk_prep10 / m_pk_prep10.max()  # WRONG order: normalise before cover_fit
m_wrong10 = engine.cover_fit(np.repeat(norm_first10[..., None], 3, axis=2).astype(np.float32), w10, h10)[..., 0]
d_wrong10 = np.power(np.clip(m_wrong10.astype(np.float64), 0, 1), 1.0)
delta10 = float(np.abs(d_after10 - d_wrong10).max())
control("V10v2 a normalise-before-cover_fit ordering error is VISIBLE on the peak-cropped witness",
        delta10 > 0.5, f"maxdelta={delta10:.3f} (crop-surviving masks measure 0.0, per the adversary)")


# ===========================================================================
section("V11  pack sweep: loader node count + sibling suite stays green")
mapping_names = set(sys.modules["cs_pkg.nodes"].NODE_CLASS_MAPPINGS.keys())
# Spec section 6 pinned the loader at 3 nodes when this slice shipped (v0.5.0);
# Schematic Topographic (v0.6.0, 2026-09-12) is the fourth. The row now pins
# the three section-6 nodes by NAME and a floor of 3, so later siblings do not
# break a mask-slice suite that never covered them.
check("V11 loader registers the 3 spec-section-6 nodes (Overlay, Voronoi, Construction) - later siblings allowed",
      {"SchematicOverlay", "SchematicVoronoi", "SchematicConstruction"} <= mapping_names and len(mapping_names) >= 3,
      f"names={sorted(mapping_names)}")
check("V11 SchematicOverlay and SchematicVoronoi both present",
      {"SchematicOverlay", "SchematicVoronoi"} <= mapping_names, f"names={sorted(mapping_names)}")

tv_path = os.path.join(PACK, "tools", "test_voronoi.py")
res11 = subprocess.run([sys.executable, tv_path], capture_output=True, text=True, timeout=300)
check("V11 tools/test_voronoi.py exits 0 (pack sweep stays green, zero rows weakened)",
      res11.returncode == 0, f"returncode={res11.returncode} stdout_tail={res11.stdout[-300:]}")


# ===========================================================================
section("V12  batch collapse under a wired mask (node-level)")
photo12a = synth_photo(W0, H0, 71)
photo12b = synth_photo(W0, H0, 72)
batch12 = torch.from_numpy(np.stack([photo12a, photo12b], axis=0))
mask12 = torch.from_numpy(ramp(W0, H0).astype(np.float32))[None, ...]  # M=1
p12 = dict(NODE_DEFAULTS)
p12.update(cells=1500, texture_opacity=0.0, mask_min=0.0)

out12_masked = NODE.execute(image=batch12, texture=None, density_mask=mask12, **p12)
ov12 = out12_masked[1]
check("V12 masked 2-frame batch: overlay_only frames byte-identical", torch.equal(ov12[0], ov12[1]))

out12_unmasked = NODE.execute(image=batch12, texture=None, **p12)
ovu12 = out12_unmasked[1]
control("V12 unmasked same batch: overlay_only frames DIFFER (the absent-path control)",
        not torch.equal(ovu12[0], ovu12[1]))


# ===========================================================================
section("V13  floor law (module-level oracle rows + the pinned node-level ink test)")
m13 = ramp(W0, H0).astype(np.float32)
mp13 = canvas_mask_arr(prep_mask_arr(m13), W0, H0)
d_f0 = voronoi.build_density(FLAT0, "brightness", 2.5, mask01=mp13, mask_min=0.0)
d_ref0 = voronoi.build_density(FLAT0, "brightness", 2.5, mask01=mp13)  # module default mask_min
check("V13 mask_min=0 bitwise == unfloored (explicit vs module default)", np.array_equal(d_f0, d_ref0))

d_f1 = voronoi.build_density(FLAT0, "brightness", 2.5, mask01=mp13, mask_min=1.0)
mp_ones13 = canvas_mask_arr(prep_mask_arr(np.ones((H0, W0), dtype=np.float32)), W0, H0)
d_ones13 = voronoi.build_density(FLAT0, "brightness", 2.5, mask01=mp_ones13, mask_min=0.0)
s_f1 = voronoi.sample_seeds(d_f1, 3000, 42, W0, H0)
s_ones13 = voronoi.sample_seeds(d_ones13, 3000, 42, W0, H0)
check("V13 mask_min=1 density == white-mask uniform density", bool((d_f1 == 1.0).all()))
check("V13 mask_min=1 seeds == white-mask uniform seeds", np.array_equal(s_f1, s_ones13))

m13_64 = m13.astype(np.float64)
d_c13 = voronoi.build_density(FLAT0, "brightness", 2.5, mask01=mp13, mask_min=0.15)
oracle13 = 0.15 + 0.85 * np.power(np.clip(m13_64 / m13_64.max(), 0, 1), 2.5)
err13 = float(np.abs(d_c13 - oracle13).max())
check("V13 mask_min=0.15 remap == f64 oracle <= 1e-15", err13 <= 1e-15, f"max|d|={err13:.2e}")

wrong13 = np.power(0.15 + 0.85 * np.clip(m13_64 / m13_64.max(), 0, 1), 2.5)  # floor BEFORE gamma
control("V13 wrong composition (floor before gamma) differs",
        float(np.abs(d_c13 - wrong13).max()) > 1e-3,
        f"maxdelta={float(np.abs(d_c13 - wrong13).max()):.4f}")

# the pinned node-level ink test: spec section 2's exact config
W13, H13 = 640, 480
photo13 = synth_photo(W13, H13, 11)
img13 = torch.from_numpy(photo13)[None, ...]
yy13, xx13 = np.mgrid[0:H13, 0:W13]
radial13 = np.exp(-(((xx13 - W13 * 0.3) ** 2 + (yy13 - H13 * 0.4) ** 2) / (2 * (0.10 * W13) ** 2)))
mask13 = torch.from_numpy(radial13.astype(np.float32))[None, ...]
p13 = dict(NODE_DEFAULTS)
p13.update(cells=12000, density_gamma=2.5, texture_opacity=0.0,
           frame_text_tl="", frame_text_tr="", frame_text_bl="", frame_text_br="",
           palette="whiteOnDark")

p13_0 = dict(p13); p13_0["mask_min"] = 0.0
a0_13 = to2d(NODE.execute(image=img13, texture=None, density_mask=mask13, **p13_0)[2][0].numpy())
z0_13, _ = zero_ink_blocks(a0_13)
control("V13 mask_min=0 leaves ink holes on the radial config (v1's spec-fatal)",
        z0_13 > 20.0, f"{z0_13:.1f}%")

p13_1 = dict(p13); p13_1["mask_min"] = 0.01
a1_13 = to2d(NODE.execute(image=img13, texture=None, density_mask=mask13, **p13_1)[2][0].numpy())
z1_13, _ = zero_ink_blocks(a1_13)
hot13 = radial13 > 0.5
cold13 = radial13 < 0.05
ratio13 = float(a1_13[hot13].mean()) / max(float(a1_13[cold13].mean()), 1e-9)
check("V13 mask_min=0.01 closes ALL 16px ink holes", z1_13 == 0.0, f"zero-ink {z0_13:.1f}% -> {z1_13:.1f}%")
check("V13 mask_min=0.01 fine:coarse ink ratio >= 10", ratio13 >= 10.0, f"ratio={ratio13:.1f}")


# ===========================================================================
section("SUMMARY")
print()
if PASS and CONTROLS_OK:
    print("ALL CHECKS PASSED (all controls fired)")
    sys.exit(0)
else:
    if not PASS:
        print("SOME CHECKS FAILED")
    if not CONTROLS_OK:
        print("SOME CONTROLS DID NOT FIRE (silent NC)")
    sys.exit(1)
