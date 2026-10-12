"""Blind-teeth suite for the Schematic Topographic node (spec section 9, items 1-11).
Implementation-blind: written against docs/schematic-topographic-derivation.md ONLY,
BEFORE schematic/topographic.py, schematic/render_topographic.py and
nodes/schematic_topographic.py existed. No pytest; prints PASS/FAIL per check and
CTRL-PASS/CTRL-FAIL per negative control, exits nonzero on any FAIL or any CTRL-FAIL.

Pure-level checks call schematic.topographic (+ schematic.engine) as black-box
APIs per the frozen module surface (spec section 7): work_grid/gradient/
contour_levels/build_contours/hachure_ticks/slope_arrows/spot_heights/benchmarks/
build_height. Node-level checks drive nodes.SchematicTopographic.execute with real
torch tensors via the smoke_node.py / test_voronoi.py package-shim pattern.
Run with the ComfyUI embedded python (needs torch).

A negative control PASSES (CTRL-PASS) when the predicate under test correctly
REJECTS a deliberately wrong or perturbed input/expectation -- it proves the
matching PASS/FAIL check has teeth (isn't vacuously true).

Items:
 1.  Contour exactness: ramp y-positions exact, cone radii within stride, level
     count per the (emin,k*interval,emax) rule, index flags every index_every-th k.
 2.  Determinism: bitwise rerun; batch of 2 == 2x single; benchmarks=False +
     texture_opacity=0 -> seed 1 vs 2 bitwise identical; benchmarks=True -> differ.
 3.  Element locality: topographic.benchmarks() identical across repeated calls,
     min-distance packing honored, count-shortfall under a huge min_dist.
 4.  Hachure direction (F(end) <= F(start)) on a cone and the hero photo; hill
     cone + depressions -> 0 ticks, pit cone + depressions -> >0, all -> >0/>0, off -> 0.
 5.  Slope arrows: ramp -> direction (0,-1), length == arrow_length; flat -> empty.
 6.  Spot heights: two-Gaussian peaks found within stride px with true labels;
     photo respects spot_count and spot_min_distance.
 7.  Grid ink: 'lines' mode has ink at every k*grid_spacing column, none at the
     half-spacing offset.
 8.  Render probes: overlay_only bg-exact with everything off; overlay_alpha zero
     at overlay_opacity=0; early-out bitwise passthrough; band_fill raises mean
     alpha monotonically.
 9.  height_mask: a mask equal to the photo's own luma (float32-socket-rounded)
     produces the same height field / contours as height_source=brightness;
     (1,H,W,1) mask accepted; 4x4 mask -> "not a plausible mask" note.
10.  Flat field: no crash, "contour-free" console note, grid+marginalia still draw.
11.  Perf print (soft, no gate): default 1600x893 hero render time.
"""
import contextlib
import io
import math
import os
import sys
import time

import numpy as np

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)

from schematic import topographic, engine, render_topographic  # pure, black-box per spec section 7

import torch
from PIL import Image
import importlib.util, types

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

SchematicTopographic = sys.modules["cs_pkg.nodes"].NODE_CLASS_MAPPINGS["SchematicTopographic"]

HERO_PATH = "C:/Users/Jeremie/comfyUI-DEV-Tools/ComfyUI-Darkroom/assets/hero.jpg"

PASS_COUNT = 0
FAIL_COUNT = 0
CTRL_COUNT = 0
CTRL_FAIL_COUNT = 0


def check(name, cond, detail=""):
    global PASS_COUNT, FAIL_COUNT
    status = "PASS" if cond else "FAIL"
    if cond:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


def control(name, rejected, detail=""):
    """A negative control: `rejected` must be True (the predicate correctly
    rejects the deliberately wrong/perturbed input) for the control to PASS."""
    global CTRL_COUNT, CTRL_FAIL_COUNT
    status = "CTRL-PASS" if rejected else "CTRL-FAIL"
    CTRL_COUNT += 1
    if not rejected:
        CTRL_FAIL_COUNT += 1
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return rejected


def node_defaults():
    """Programmatic widget defaults from INPUT_TYPES, minus socket inputs
    (make_exhibit_voronoi.py defaults_for / test_voronoi.py node_defaults pattern)."""
    node = SchematicTopographic()
    it = node.INPUT_TYPES()
    d = {}
    for section in ("required", "optional"):
        for k, v in it.get(section, {}).items():
            if isinstance(v, tuple) and len(v) > 1 and isinstance(v[1], dict) and "default" in v[1]:
                d[k] = v[1]["default"]
            elif isinstance(v, tuple) and isinstance(v[0], list):
                d[k] = v[0][0]
    for sock in ("image", "texture", "height_mask"):
        d.pop(sock, None)
    return d


DEFAULTS = node_defaults()
BG_RGB, STROKE_RGB = engine.resolve_palette("whiteOnDark", "#0a0a0a", "#d8d8d8")


# ---------------------------------------------------------------------------
# Synthetic field builders (pure numpy, (H,W) float64 -- spec section 3/4/7)
# ---------------------------------------------------------------------------

def make_ramp_H(h, w):
    """F = j ramp (spec section 4's hand-checkable ramp): H rising downwards,
    H[y,x] = y/(h-1)."""
    col = (np.arange(h, dtype=np.float64) / (h - 1))[:, None]
    return np.tile(col, (1, w))


def make_cone_H(h, w, cx, cy, radius, pit=False):
    """Hill cone H = max(0,1-r/R); pit cone H = min(1, r/R) (spec section 4/9)."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    if pit:
        return np.minimum(1.0, r / radius)
    return np.maximum(0.0, 1.0 - r / radius)


def make_two_gaussian_H(h, w, c1, c2, sigma):
    """Two bumps at known centres (spec section 9 item 6), amplitude 1 each,
    combined by max() so each true centre reads EXACTLY H=1.0 regardless of overlap."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    b1 = np.exp(-((xs - c1[0]) ** 2 + (ys - c1[1]) ** 2) / (2.0 * sigma ** 2))
    b2 = np.exp(-((xs - c2[0]) ** 2 + (ys - c2[1]) ** 2) / (2.0 * sigma ** 2))
    return np.maximum(b1, b2)


def nearest_F(F, x, y, stride):
    """Sample F at the nearest work vertex to canvas point (x,y)."""
    hj, wi = F.shape
    i = min(max(int(round(x / stride)), 0), wi - 1)
    j = min(max(int(round(y / stride)), 0), hj - 1)
    return F[j, i]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------------------------------------------------------------------
# Shared photo fixtures (hero.jpg, downscaled to 800x447 "for speed" per task
# guidance; full 1600x893 reserved for the item-11 perf print).
# ---------------------------------------------------------------------------

_hero_full_pil = Image.open(HERO_PATH).convert("RGB")
_hero_small_pil = _hero_full_pil.resize((800, 447), Image.LANCZOS)
photo_small_arr = (np.asarray(_hero_small_pil).astype(np.float32) / 255.0)
photo_tensor = torch.from_numpy(photo_small_arr[None, ...])

PHOTO_W, PHOTO_H = 800, 447
PHOTO_FITTED = engine.cover_fit(photo_small_arr, PHOTO_W, PHOTO_H)  # float32 (h,w,3), spec section 3
PHOTO_STRIDE = int(DEFAULTS["contour_resolution"])
PHOTO_H_FIELD, PHOTO_FLAT = topographic.build_height(
    PHOTO_FITTED, "brightness", DEFAULTS["smoothing"], mask01=None)
PHOTO_F = topographic.work_grid(PHOTO_H_FIELD, PHOTO_STRIDE)
PHOTO_GX, PHOTO_GY = topographic.gradient(PHOTO_F)
PHOTO_LEVELS = topographic.contour_levels(
    DEFAULTS["elevation_min"], DEFAULTS["elevation_max"],
    DEFAULTS["contour_interval"], DEFAULTS["index_every"])
PHOTO_CONTOURS = topographic.build_contours(PHOTO_F, PHOTO_LEVELS, PHOTO_STRIDE)


# ===========================================================================
print("[1] Contour exactness: ramp y-positions, cone radii, level count, index flags")

EMIN, EMAX, INTERVAL, INDEX_EVERY = 0.0, 1000.0, 20.0, 5
levels1 = topographic.contour_levels(EMIN, EMAX, INTERVAL, INDEX_EVERY)

expected_k1 = [k for k in range(1, 10000) if EMIN < k * INTERVAL < EMAX]
check("level count matches the (emin < k*interval < emax) rule",
      len(levels1) == len(expected_k1), f"got={len(levels1)} expected={len(expected_k1)}")
control("level-count predicate: an off-by-one wrong expected count is correctly rejected",
        len(levels1) != len(expected_k1) + 1)

index_ok1 = all(bool(isx) == (int(round(z / INTERVAL)) % INDEX_EVERY == 0) for (z, t, isx) in levels1)
check("index flag true iff k % index_every == 0", index_ok1)
wrong_index_every1 = INDEX_EVERY + 1
index_wrong_ok1 = all(bool(isx) == (int(round(z / INTERVAL)) % wrong_index_every1 == 0) for (z, t, isx) in levels1)
control("index-flag predicate: a wrong index_every mismatches at least one level",
        not index_wrong_ok1)

RH1, RW1 = 241, 241  # h-1 = 240, divisible by strides 1 and 2
H_ramp1 = make_ramp_H(RH1, RW1)
for s in (1, 2):
    F = topographic.work_grid(H_ramp1, s)
    contours = topographic.build_contours(F, levels1, s)
    tol = 1e-6 if s == 1 else float(s)
    open_ok, y_ok, span_ok = True, True, True
    for c in contours:
        t = (c["z"] - EMIN) / (EMAX - EMIN)
        y_expected = t * (RH1 - 1)
        for pts, closed in c["lines"]:
            if closed:
                open_ok = False
            if np.max(np.abs(pts[:, 1] - y_expected)) > tol:
                y_ok = False
        all_x = np.concatenate([pts[:, 0] for pts, _ in c["lines"]]) if c["lines"] else np.array([])
        if all_x.size and (all_x.min() > tol or all_x.max() < (RW1 - 1) - tol):
            span_ok = False
    check(f"ramp (stride={s}): every contour is a single OPEN horizontal polyline", open_ok)
    check(f"ramp (stride={s}): contour y == t_k*(h-1) within tol={tol}", y_ok)
    check(f"ramp (stride={s}): contour spans the full canvas width", span_ok)

F_ramp1_s1 = topographic.work_grid(H_ramp1, 1)
contours_ramp1_s1 = topographic.build_contours(F_ramp1_s1, levels1, 1)
c0 = contours_ramp1_s1[0]
t0 = (c0["z"] - EMIN) / (EMAX - EMIN)
y_true0 = t0 * (RH1 - 1)
y_wrong0 = y_true0 + 10.0
wrong_y_matches = all(np.max(np.abs(pts[:, 1] - y_wrong0)) <= 1e-6 for pts, _ in c0["lines"])
control("ramp-y predicate: a deliberately wrong y (+10px) is correctly rejected", not wrong_y_matches)

CW1, CH1, CX1, CY1, R1 = 300, 300, 150.0, 150.0, 60.0
H_cone1 = make_cone_H(CH1, CW1, CX1, CY1, R1, pit=False)
levels_cone1 = topographic.contour_levels(0.0, 1000.0, 100.0, 5)
for s in (1, 3):
    Fc = topographic.work_grid(H_cone1, s)
    contours_c = topographic.build_contours(Fc, levels_cone1, s)
    closed_ok, radius_ok = True, True
    for c in contours_c:
        t = c["z"] / 1000.0
        r_expected = R1 * (1.0 - t)
        for pts, closed in c["lines"]:
            if not closed:
                closed_ok = False
            r = np.sqrt((pts[:, 0] - CX1) ** 2 + (pts[:, 1] - CY1) ** 2)
            if np.max(np.abs(r - r_expected)) > s + 1e-6:
                radius_ok = False
    check(f"cone (stride={s}): every contour is a CLOSED loop", closed_ok)
    check(f"cone (stride={s}): vertex radii within stride={s}px of R*(1-t_k)", radius_ok)

Fc1 = topographic.work_grid(H_cone1, 1)
contours_c1 = topographic.build_contours(Fc1, levels_cone1, 1)
wrong_cx1 = CX1 + 20.0
bad_radius_ok = True
for c in contours_c1:
    t = c["z"] / 1000.0
    r_expected = R1 * (1.0 - t)
    for pts, closed in c["lines"]:
        r = np.sqrt((pts[:, 0] - wrong_cx1) ** 2 + (pts[:, 1] - CY1) ** 2)
        if np.max(np.abs(r - r_expected)) > 1 + 1e-6:
            bad_radius_ok = False
control("cone-radius predicate: a wrong center (+20px) is correctly rejected", not bad_radius_ok)


# ===========================================================================
print("\n[2] Determinism: bitwise rerun, batch==2x single, seed isolation via benchmarks/texture")

NODE2 = SchematicTopographic()
out1 = NODE2.execute(image=photo_tensor, **DEFAULTS)
out2 = NODE2.execute(image=photo_tensor, **DEFAULTS)
check("bitwise identical outputs on same-args rerun (all 3 outputs)",
      all(torch.equal(out1[k], out2[k]) for k in range(3)))

photo_perturbed = photo_tensor.clone()
photo_perturbed[0, 0, 0, 0] = photo_perturbed[0, 0, 0, 0] + 0.05
out_pert = NODE2.execute(image=photo_perturbed, **DEFAULTS)
control("bitwise-rerun predicate: a perturbed image correctly gives a DIFFERENT image output",
        not torch.equal(out1[0], out_pert[0]))

batch_img2 = torch.cat([photo_tensor, photo_tensor], dim=0)
out_batch2 = NODE2.execute(image=batch_img2, **DEFAULTS)
check("batch of 2 identical frames == 2x single (frame 0, all 3 outputs)",
      all(torch.equal(out_batch2[k][0], out1[k][0]) for k in range(3)))  # frame vs frame (adjudicated 2026-09-12: shape mismatch, not a build defect)
check("batch of 2 identical frames == 2x single (frame 1, all 3 outputs)",
      all(torch.equal(out_batch2[k][1], out1[k][0]) for k in range(3)))

p_iso = dict(DEFAULTS)
p_iso["benchmarks"] = False
p_iso["texture_opacity"] = 0.0
p_iso["height_source"] = "brightness"  # v2: the procedural terrain reads the seed by design (spec v2 section 3)
p_iso["seed"] = 1
out_s1 = NODE2.execute(image=photo_tensor, **p_iso)
p_iso2 = dict(p_iso)
p_iso2["seed"] = 2
out_s2 = NODE2.execute(image=photo_tensor, **p_iso2)
check("benchmarks=False & texture_opacity=0: seed 1 vs seed 2 bitwise IDENTICAL (all 3 outputs)",
      all(torch.equal(out_s1[k], out_s2[k]) for k in range(3)))

p_noiso = dict(DEFAULTS)
p_noiso["benchmarks"] = False
p_noiso["seed"] = 1
out_n1 = NODE2.execute(image=photo_tensor, **p_noiso)
p_noiso2 = dict(p_noiso)
p_noiso2["seed"] = 2
out_n2 = NODE2.execute(image=photo_tensor, **p_noiso2)
control("seed-isolation predicate: WITHOUT zeroing texture_opacity, seed 1 vs 2 correctly DIFFER "
        "(proves texture reads the seed and the isolation above is load-bearing)",
        not torch.equal(out_n1[0], out_n2[0]))

p_bm = dict(DEFAULTS)
p_bm["benchmarks"] = True
p_bm["texture_opacity"] = 0.0
p_bm["seed"] = 1
out_b1 = NODE2.execute(image=photo_tensor, **p_bm)
p_bm2 = dict(p_bm)
p_bm2["seed"] = 2
out_b2 = NODE2.execute(image=photo_tensor, **p_bm2)
check("benchmarks=True: seed 1 vs seed 2 outputs DIFFER", not torch.equal(out_b1[0], out_b2[0]))

out_b1_rerun = NODE2.execute(image=photo_tensor, **p_bm)
control("benchmarks=True seed-diff predicate: a same-seed rerun is still bitwise identical "
        "(the difference above is really seed-driven, not flaky)",
        torch.equal(out_b1[0], out_b1_rerun[0]))


# ===========================================================================
print("\n[2b] Procedural terrain (spec v2 section 3): deterministic, seed-driven, normalised, anisotropic")
grey2b = np.full((300, 500, 3), 0.5, dtype=np.float32)
terr = {"seed": 7, "scale": 120.0, "octaves": 3, "roughness": 0.45, "stretch": 1.0, "angle": 0.0}
Hp1, flat_p1 = topographic.build_height(grey2b, "procedural", 0.0, terrain=terr)
Hp2, _ = topographic.build_height(grey2b, "procedural", 0.0, terrain=terr)
check("procedural: bitwise identical on rerun", np.array_equal(Hp1, Hp2))
check("procedural: not flat on a flat grey image, range [0,1] exactly",
      (not flat_p1) and abs(float(Hp1.min())) < 1e-12 and abs(float(Hp1.max()) - 1.0) < 1e-12,
      f"min={Hp1.min():.3g} max={Hp1.max():.3g}")
Hp3, _ = topographic.build_height(grey2b, "procedural", 0.0, terrain=dict(terr, seed=8))
control("procedural determinism predicate: a different seed correctly gives a DIFFERENT field",
        not np.array_equal(Hp1, Hp3))
Hp4, _ = topographic.build_height(grey2b * 0.0 + 0.9, "procedural", 0.0, terrain=terr)
check("procedural: independent of the photo (flat 0.5 vs flat 0.9 image -> identical field)", np.array_equal(Hp1, Hp4))
Hb, _ = topographic.build_height(grey2b, "brightness", 0.0)
control("photo-independence predicate: the brightness source on a flat image correctly differs from the procedural field",
        not np.array_equal(Hp1, Hb))
Hs, _ = topographic.build_height(grey2b, "procedural", 0.0, terrain=dict(terr, stretch=4.0, angle=0.0))
gx_i, gy_i = topographic.gradient(Hp1)
gx_s, gy_s = topographic.gradient(Hs)
ratio_iso = float(np.mean(gx_i ** 2) / np.mean(gy_i ** 2))
ratio_str = float(np.mean(gx_s ** 2) / np.mean(gy_s ** 2))
check("procedural: stretch 4 along x lowers the x/y gradient-energy ratio by >= 4x vs isotropic",
      ratio_str <= ratio_iso / 4.0, f"iso={ratio_iso:.3f} stretched={ratio_str:.3f}")
control("anisotropy predicate: the isotropic field's ratio is correctly NOT <= its own quarter", not (ratio_iso <= ratio_iso / 4.0))


print("\n[3] Element locality: topographic.benchmarks() determinism and min-distance packing")

W3, H3, STRIDE3 = 400, 300, 4
H_ramp3 = make_ramp_H(H3, W3)
F3 = topographic.work_grid(H_ramp3, STRIDE3)
SPOTS3 = [(60.0, 50.0), (340.0, 80.0), (200.0, 250.0)]

bm1 = topographic.benchmarks(42, W3, H3, 6, 40.0, SPOTS3, F3, STRIDE3, 0.0, 1000.0)
bm2 = topographic.benchmarks(42, W3, H3, 6, 40.0, SPOTS3, F3, STRIDE3, 0.0, 1000.0)
check("benchmarks(): identical across repeated calls with identical arguments", list(bm1) == list(bm2))

bm_seed2 = topographic.benchmarks(43, W3, H3, 6, 40.0, SPOTS3, F3, STRIDE3, 0.0, 1000.0)
control("benchmarks() determinism predicate: a different seed correctly gives a DIFFERENT list",
        list(bm1) != list(bm_seed2))

min_dist_ok3 = True
for i, site in enumerate(bm1):
    for sp in SPOTS3:
        if dist(site, sp) < 40.0 - 1e-6:
            min_dist_ok3 = False
    for j, other in enumerate(bm1):
        if i != j and dist(site, other) < 40.0 - 1e-6:
            min_dist_ok3 = False
check("benchmarks(): every accepted site is >= min_dist from every spot and every other benchmark",
      min_dist_ok3, f"n={len(bm1)}")

bm_dense3 = topographic.benchmarks(42, W3, H3, 20, 2.0, SPOTS3, F3, STRIDE3, 0.0, 1000.0)
violation3 = any(dist(bm_dense3[i], bm_dense3[j]) < 40.0 - 1e-6
                 for i in range(len(bm_dense3)) for j in range(len(bm_dense3)) if i != j)
control("min-distance predicate: a list generated with a much smaller min_dist violates the "
        "ORIGINAL (40px) threshold", violation3, f"n_dense={len(bm_dense3)}")

bm_shortfall3 = topographic.benchmarks(42, W3, H3, 10, 100000.0, SPOTS3, F3, STRIDE3, 0.0, 1000.0)
control("count-reached property: a min_dist so large that fewer than count can be placed "
        "correctly yields a SHORTER list", len(bm_shortfall3) < 10, f"n={len(bm_shortfall3)}")


# ===========================================================================
print("\n[4] Hachure direction and depression/all/off mode gating")

CW4, CH4, CX4, CY4, R4, STRIDE4 = 240, 240, 120.0, 120.0, 50.0, 2
H_hill4 = make_cone_H(CH4, CW4, CX4, CY4, R4, pit=False)
H_pit4 = make_cone_H(CH4, CW4, CX4, CY4, R4, pit=True)
levels4 = topographic.contour_levels(0.0, 1000.0, 100.0, 5)


def _build4(H, stride):
    F = topographic.work_grid(H, stride)
    gx, gy = topographic.gradient(F)
    contours = topographic.build_contours(F, levels4, stride)
    return F, gx, gy, contours


F_hill4, gx_hill4, gy_hill4, contours_hill4 = _build4(H_hill4, STRIDE4)
F_pit4, gx_pit4, gy_pit4, contours_pit4 = _build4(H_pit4, STRIDE4)


def direction_ok(ticks, F, stride):
    return all(nearest_F(F, x1, y1, stride) <= nearest_F(F, x0, y0, stride) + 1e-9
               for (x0, y0), (x1, y1) in ticks)


ticks_hill_dep = topographic.hachure_ticks(contours_hill4, gx_hill4, gy_hill4, STRIDE4, "depressions", 8, 6)
check("hill cone + depressions -> ZERO ticks", len(ticks_hill_dep) == 0, f"n={len(ticks_hill_dep)}")

ticks_pit_dep = topographic.hachure_ticks(contours_pit4, gx_pit4, gy_pit4, STRIDE4, "depressions", 8, 6)
check("pit cone + depressions -> > 0 ticks", len(ticks_pit_dep) > 0, f"n={len(ticks_pit_dep)}")
check("pit cone + depressions: direction predicate holds (F(end) <= F(start))",
      direction_ok(ticks_pit_dep, F_pit4, STRIDE4))

ticks_hill_all = topographic.hachure_ticks(contours_hill4, gx_hill4, gy_hill4, STRIDE4, "all", 8, 6)
ticks_pit_all = topographic.hachure_ticks(contours_pit4, gx_pit4, gy_pit4, STRIDE4, "all", 8, 6)
check("hill cone + all -> > 0 ticks", len(ticks_hill_all) > 0, f"n={len(ticks_hill_all)}")
check("pit cone + all -> > 0 ticks", len(ticks_pit_all) > 0, f"n={len(ticks_pit_all)}")
check("hill cone + all: direction predicate holds", direction_ok(ticks_hill_all, F_hill4, STRIDE4))
check("pit cone + all: direction predicate holds", direction_ok(ticks_pit_all, F_pit4, STRIDE4))

ticks_off4 = topographic.hachure_ticks(contours_hill4, gx_hill4, gy_hill4, STRIDE4, "off", 8, 6)
check("hachures='off' -> ZERO ticks", len(ticks_off4) == 0, f"n={len(ticks_off4)}")

ticks_photo4 = topographic.hachure_ticks(PHOTO_CONTOURS, PHOTO_GX, PHOTO_GY, PHOTO_STRIDE,
                                          "all", DEFAULTS["hachure_spacing"], DEFAULTS["hachure_length"], F=PHOTO_F)
check("hero photo: hachure ticks non-vacuous (mode='all')", len(ticks_photo4) > 0, f"n={len(ticks_photo4)}")
check("hero photo: hachure direction predicate holds for every tick",
      direction_ok(ticks_photo4, PHOTO_F, PHOTO_STRIDE))


def _find_sloped_tick(ticks, F, stride, min_gap=1e-4):
    for (x0, y0), (x1, y1) in ticks:
        f0 = nearest_F(F, x0, y0, stride)
        f1 = nearest_F(F, x1, y1, stride)
        if f0 - f1 > min_gap:
            return f0, f1
    return None


found4 = _find_sloped_tick(ticks_pit_all, F_pit4, STRIDE4) or _find_sloped_tick(ticks_photo4, PHOTO_F, PHOTO_STRIDE)
if found4:
    f0, f1 = found4
    swapped_holds = f0 <= f1 + 1e-9  # predicate applied to the REVERSED (start<->end) tick
    control("hachure-direction predicate: a reversed tick (start<->end swapped) is correctly rejected",
            not swapped_holds, f"F(orig_start)={f0:.6f} F(orig_end)={f1:.6f}")
else:
    control("hachure-direction predicate: reversed-tick control", False, "no sufficiently sloped tick found")


# ===========================================================================
print("\n[5] Slope arrows: ramp direction/length, flat field emptiness")

H5, W5, STRIDE5, ARROW_LEN5, ARROW_SP5 = 200, 260, 5, 22.0, 40
H_ramp5 = make_ramp_H(H5, W5)
F5 = topographic.work_grid(H_ramp5, STRIDE5)
gx5, gy5 = topographic.gradient(F5)
arrows5 = topographic.slope_arrows(gx5, gy5, STRIDE5, W5, H5, ARROW_SP5, ARROW_LEN5)
check("ramp: slope_arrows non-empty", len(arrows5) > 0, f"n={len(arrows5)}")

dir_ok5, len_ok5 = True, True
for (x0, y0), (x1, y1), L in arrows5:
    dxv, dyv = x1 - x0, y1 - y0
    norm = math.hypot(dxv, dyv)
    if norm < 1e-12:
        dir_ok5 = False
        continue
    ux, uy = dxv / norm, dyv / norm
    if abs(ux - 0.0) > 1e-9 or abs(uy - (-1.0)) > 1e-9:
        dir_ok5 = False
    if abs(L - ARROW_LEN5) > 1e-9:
        len_ok5 = False
check("ramp: every arrow direction is (0,-1) within 1e-9", dir_ok5)
check("ramp: every arrow length == arrow_length exactly (uniform slope == g_ref)", len_ok5)

wrong_dir_matches5 = all(
    abs((x1 - x0) / max(1e-12, math.hypot(x1 - x0, y1 - y0)) - 1.0) <= 1e-9
    and abs((y1 - y0) / max(1e-12, math.hypot(x1 - x0, y1 - y0)) - 0.0) <= 1e-9
    for (x0, y0), (x1, y1), L in arrows5)
control("arrow-direction predicate: a deliberately wrong expected direction (1,0) is correctly rejected",
        not wrong_dir_matches5)

H_flat5 = np.zeros((H5, W5), dtype=np.float64)
F_flat5 = topographic.work_grid(H_flat5, STRIDE5)
gxf5, gyf5 = topographic.gradient(F_flat5)
arrows_flat5 = topographic.slope_arrows(gxf5, gyf5, STRIDE5, W5, H5, ARROW_SP5, ARROW_LEN5)
check("flat field: slope_arrows is EMPTY", len(arrows_flat5) == 0, f"n={len(arrows_flat5)}")

H_tiny5 = make_ramp_H(H5, W5) * 1e-6
F_tiny5 = topographic.work_grid(H_tiny5, STRIDE5)
gxt5, gyt5 = topographic.gradient(F_tiny5)
arrows_tiny5 = topographic.slope_arrows(gxt5, gyt5, STRIDE5, W5, H5, ARROW_SP5, ARROW_LEN5)
control("flat-field-emptiness predicate: a tiny NONZERO uniform slope correctly yields NON-EMPTY arrows "
        "(the empty result above isn't a universal bug)", len(arrows_tiny5) > 0, f"n={len(arrows_tiny5)}")


# ===========================================================================
print("\n[6] Spot heights: two-Gaussian peaks/labels, photo count/min-distance")

H6, W6, STRIDE6 = 200, 200, 4
C1_6, C2_6, SIGMA6 = (60.0, 60.0), (140.0, 140.0), 12.0
H_2g6 = make_two_gaussian_H(H6, W6, C1_6, C2_6, SIGMA6)
F6 = topographic.work_grid(H_2g6, STRIDE6)
spots6 = topographic.spot_heights(F6, STRIDE6, 4, 30.0, 0.0, 1000.0)
peaks6 = [(x, y, z) for (x, y, z, kind) in spots6 if kind == "peak"]
peak_xy6 = [(x, y) for (x, y, z) in peaks6]


def _nearest(pt, candidates):
    return min((dist(pt, c) for c in candidates), default=float("inf"))


d1_6 = _nearest(C1_6, peak_xy6)
d2_6 = _nearest(C2_6, peak_xy6)
check("two-Gaussian: a peak is found within stride px of true center 1", d1_6 <= STRIDE6, f"d={d1_6:.3f}")
check("two-Gaussian: a peak is found within stride px of true center 2", d2_6 <= STRIDE6, f"d={d2_6:.3f}")


def _matched_z(center, peaks):
    best = min(peaks, key=lambda p: dist(center, (p[0], p[1])))
    return best[2]


z1_6 = _matched_z(C1_6, peaks6)
z2_6 = _matched_z(C2_6, peaks6)
check("peak 1 label == round(elev_max) (H==1.0 exactly at the aligned center)",
      int(round(z1_6)) == 1000, f"z={z1_6}")
check("peak 2 label == round(elev_max) (H==1.0 exactly at the aligned center)",
      int(round(z2_6)) == 1000, f"z={z2_6}")

d_wrong6 = _nearest((0.0, 0.0), peak_xy6)
control("peak-location predicate: a wrong far-off center is correctly NOT within stride of any peak",
        d_wrong6 > STRIDE6, f"d={d_wrong6:.3f}")

SPOT_COUNT6 = DEFAULTS["spot_count"]
SPOT_MIN_DIST6 = DEFAULTS["spot_min_distance"]
spots_photo6 = topographic.spot_heights(PHOTO_F, PHOTO_STRIDE, SPOT_COUNT6, SPOT_MIN_DIST6, 0.0, 1000.0)
check("photo: spot count <= spot_count", len(spots_photo6) <= SPOT_COUNT6, f"n={len(spots_photo6)}")
xy_photo6 = [(x, y) for (x, y, z, k) in spots_photo6]
pairs6 = [(a, b) for i, a in enumerate(xy_photo6) for j, b in enumerate(xy_photo6) if i < j]
min_pair6 = min((dist(a, b) for a, b in pairs6), default=float("inf"))
check("photo: pairwise spot distances >= spot_min_distance", min_pair6 >= SPOT_MIN_DIST6 - 1e-6,
      f"min_pair={min_pair6:.3f}")

spots_tight6 = topographic.spot_heights(PHOTO_F, PHOTO_STRIDE, SPOT_COUNT6, 5.0, 0.0, 1000.0)
xy_tight6 = [(x, y) for (x, y, z, k) in spots_tight6]
pairs_tight6 = [(a, b) for i, a in enumerate(xy_tight6) for j, b in enumerate(xy_tight6) if i < j]
min_pair_tight6 = min((dist(a, b) for a, b in pairs_tight6), default=float("inf"))
control("min-distance predicate: spots generated with min_dist=5 violate the ORIGINAL "
        "spot_min_distance threshold", min_pair_tight6 < SPOT_MIN_DIST6 - 1e-6,
        f"min_pair_tight={min_pair_tight6:.3f}")


# ===========================================================================
print("\n[7] Grid ink: 'lines' mode has ink at k*grid_spacing columns, none at the half-spacing offset")

W7, H7g, SPACING7 = 800, 400, 200
grey7 = np.full((H7g, W7, 3), 0.5, dtype=np.float32)
grey7_tensor = torch.from_numpy(grey7[None, ...])

p7 = dict(DEFAULTS)
p7["height_source"] = "brightness"  # v2: a flat grey image is only FLAT through a photo source
p7.update({
    "grid": "lines", "grid_weight": 1.0, "overlay_opacity": 1.0, "band_fill": 0.0,
    "north_arrow": False, "scale_bar": False, "interval_note": False,
    "contour_labels": False, "spot_heights": False, "benchmarks": False,
    "hachures": "off", "grid_spacing": SPACING7,
    "annotations": "", "frame_text_tl": "", "frame_text_tr": "", "frame_text_bl": "", "frame_text_br": "",
    "size_preset": "match input",
})
NODE7 = SchematicTopographic()
out7 = NODE7.execute(image=grey7_tensor, **p7)
alpha7 = out7[2][0].cpu().numpy()

cols_grid7 = [k * SPACING7 for k in range(1, W7 // SPACING7)]
grid_ink_ok7 = all(np.min(alpha7[:, x]) > 0.2 for x in cols_grid7)
check("grid='lines': every grid column has ink (>0.2) in EVERY row", grid_ink_ok7, f"cols={cols_grid7}")

cols_offset7 = [x + SPACING7 // 2 for x in cols_grid7 if x + SPACING7 // 2 < W7]
# rows away from the HORIZONTAL grid lines (they cross every column; adjudicated 2026-09-12)
rows_grid7 = [k * SPACING7 for k in range(1, H7g // SPACING7 + 1) if k * SPACING7 < H7g]
row_mask7 = np.ones(H7g, dtype=bool)
for gy in rows_grid7:
    row_mask7[max(0, gy - 3):gy + 4] = False
offset_ink_ok7 = all(np.max(alpha7[row_mask7, x]) < 0.01 for x in cols_offset7)
check("grid='lines': no ink (<0.01) at the half-spacing offset columns (away from grid rows)", offset_ink_ok7, f"cols={cols_offset7}")

control("grid-ink predicate: the half-spacing offset columns correctly do NOT satisfy the '>0.2 ink in every row' test",
        all(np.min(alpha7[:, x]) <= 0.2 for x in cols_offset7))


# ===========================================================================
print("\n[8] Render probes: overlay_only bg-exact, alpha zero at op=0, early-out passthrough, band_fill monotone")

expected_bg8 = np.array(BG_RGB, dtype=np.float64) / 255.0

p8a = dict(DEFAULTS)
p8a["height_source"] = "brightness"  # v2: a flat grey image is only FLAT through a photo source
p8a.update({
    "grid": "off", "hachures": "off", "band_fill": 0.0,
    "north_arrow": False, "scale_bar": False, "interval_note": False,
    "contour_labels": False, "spot_heights": False, "benchmarks": False,
    "annotations": "", "frame_text_tl": "", "frame_text_tr": "", "frame_text_bl": "", "frame_text_br": "",
    "overlay_opacity": 1.0, "size_preset": "match input",
    "texture_opacity": 0.0,  # the grain IS content (spec section 6 step 5); adjudicated 2026-09-12
})
NODE8 = SchematicTopographic()
out8a = NODE8.execute(image=grey7_tensor, **p8a)
overlay_only8a = out8a[1][0].cpu().numpy()
bg_exact_ok8 = np.allclose(overlay_only8a, expected_bg8, atol=1e-6)
check("overlay_only with EVERYTHING off + empty frame text == bg colour exactly", bg_exact_ok8,
      f"bg_expected={expected_bg8.tolist()} sample={overlay_only8a[0, 0].tolist()}")

p8a_ctrl = dict(p8a)
p8a_ctrl["grid"] = "lines"
out8a_ctrl = NODE8.execute(image=grey7_tensor, **p8a_ctrl)
overlay_only8a_ctrl = out8a_ctrl[1][0].cpu().numpy()
control("bg-exact predicate: turning grid back on correctly breaks bg-exactness",
        not np.allclose(overlay_only8a_ctrl, expected_bg8, atol=1e-6))

p8b = dict(DEFAULTS)
p8b["overlay_opacity"] = 0.0
out8b = NODE8.execute(image=photo_tensor, **p8b)
alpha8b = out8b[2][0].cpu().numpy()
check("overlay_opacity=0 -> overlay_alpha all zero (with real default content)",
      float(np.max(alpha8b)) == 0.0, f"max={float(np.max(alpha8b))}")

p8b_ctrl = dict(DEFAULTS)
p8b_ctrl["overlay_opacity"] = 1.0
out8b_ctrl = NODE8.execute(image=photo_tensor, **p8b_ctrl)
alpha8b_ctrl = out8b_ctrl[2][0].cpu().numpy()
control("overlay-alpha-zero predicate: overlay_opacity=1 correctly gives NON-zero alpha",
        float(np.max(alpha8b_ctrl)) > 0.0)

rng8 = np.random.default_rng(13)
img8_rand = torch.from_numpy(rng8.random((1, 80, 96, 3), dtype=np.float32))
p8c = dict(DEFAULTS)
p8c.update({"image_opacity": 1.0, "overlay_opacity": 0.0, "texture_opacity": 0.0, "size_preset": "match input"})
out8c = NODE8.execute(image=img8_rand, **p8c)
check("early-out: image output bitwise identical to input", torch.equal(out8c[0], img8_rand))

p8c_ctrl = dict(p8c)
p8c_ctrl["image_opacity"] = 0.99
out8c_ctrl = NODE8.execute(image=img8_rand, **p8c_ctrl)
control("early-out predicate: breaking image_opacity>=1 correctly loses bitwise passthrough",
        not torch.equal(out8c_ctrl[0], img8_rand))

H8, W8 = 150, 200
ramp8 = (np.arange(H8, dtype=np.float32) / (H8 - 1))[:, None, None] * np.ones((1, W8, 3), dtype=np.float32)
ramp8_tensor = torch.from_numpy(ramp8[None, ...])
p8d0 = dict(DEFAULTS)
p8d0.update({"smoothing": 0, "band_fill": 0.0})
p8d3 = dict(p8d0)
p8d3["band_fill"] = 0.3
p8d5 = dict(p8d0)
p8d5["band_fill"] = 0.5
mean0_8 = float(NODE8.execute(image=ramp8_tensor, **p8d0)[2][0].mean())
mean3_8 = float(NODE8.execute(image=ramp8_tensor, **p8d3)[2][0].mean())
mean5_8 = float(NODE8.execute(image=ramp8_tensor, **p8d5)[2][0].mean())
check("band_fill=0.3 raises mean overlay_alpha vs 0.0 on a ramp", mean3_8 > mean0_8,
      f"m0={mean0_8:.5f} m3={mean3_8:.5f}")
check("band_fill=0.5 raises mean overlay_alpha vs 0.3 (monotone)", mean5_8 > mean3_8,
      f"m3={mean3_8:.5f} m5={mean5_8:.5f}")
control("band_fill-monotone predicate: the reversed inequality (0.0 > 0.3) is correctly rejected",
        not (mean0_8 > mean3_8), f"m0={mean0_8:.5f} m3={mean3_8:.5f}")


# ===========================================================================
print("\n[9] height_mask: mask==luma equals brightness source; I/O shape acceptance")

SMOOTH9 = 0
H_brightness9, flat_b9 = topographic.build_height(PHOTO_FITTED, "brightness", SMOOTH9, mask01=None)

photo_luma_f64_9 = (0.299 * PHOTO_FITTED[..., 0].astype(np.float64)
                     + 0.587 * PHOTO_FITTED[..., 1].astype(np.float64)
                     + 0.114 * PHOTO_FITTED[..., 2].astype(np.float64))
luma_f32_9 = photo_luma_f64_9.astype(np.float32)
luma_f32_as_f64_9 = luma_f32_9.astype(np.float64)

H_masked9, flat_m9 = topographic.build_height(PHOTO_FITTED, "darkness", SMOOTH9, mask01=luma_f32_as_f64_9)

check("height_mask == photo luma -> build_height('darkness', mask=luma) == "
      "build_height('brightness', mask=None)", np.array_equal(H_brightness9, H_masked9))

perturbed_mask9 = luma_f32_as_f64_9.copy()
perturbed_mask9[0, 0] = min(1.0, perturbed_mask9[0, 0] + 0.2)
H_perturbed9, _ = topographic.build_height(PHOTO_FITTED, "darkness", SMOOTH9, mask01=perturbed_mask9)
control("mask-equals-luma predicate: a perturbed mask correctly gives a DIFFERENT height field",
        not np.array_equal(H_brightness9, H_perturbed9))

F_b9 = topographic.work_grid(H_brightness9, PHOTO_STRIDE)
F_m9 = topographic.work_grid(H_masked9, PHOTO_STRIDE)
contours_b9 = topographic.build_contours(F_b9, PHOTO_LEVELS, PHOTO_STRIDE)
contours_m9 = topographic.build_contours(F_m9, PHOTO_LEVELS, PHOTO_STRIDE)

contours_match9 = len(contours_b9) == len(contours_m9)
if contours_match9:
    for cb, cm in zip(contours_b9, contours_m9):
        if cb["z"] != cm["z"] or bool(cb["index"]) != bool(cm["index"]) or len(cb["lines"]) != len(cm["lines"]):
            contours_match9 = False
            break
        for (pb, clb), (pm, clm) in zip(cb["lines"], cm["lines"]):
            if clb != clm or pb.shape != pm.shape or not np.allclose(pb, pm, atol=1e-9):
                contours_match9 = False
                break
check("build_contours() identical between the mask-equals-luma path and the brightness path", contours_match9)

NODE9 = SchematicTopographic()

mask_trailing9 = torch.from_numpy(luma_f32_9[None, :, :, None])  # (1,H,W,1)
try:
    out9_mask = NODE9.execute(image=photo_tensor, height_mask=mask_trailing9, **dict(DEFAULTS))
    accepted_ok9 = True
except Exception as e:
    accepted_ok9 = False
    print(f"    (unexpected exception: {e!r})")
check("(1,H,W,1)-trailing-dim height_mask tensor accepted without exception", accepted_ok9)

tiny_mask9 = torch.zeros((1, 4, 4), dtype=torch.float32)
buf9 = io.StringIO()
with contextlib.redirect_stdout(buf9):
    try:
        NODE9.execute(image=photo_tensor, height_mask=tiny_mask9, **dict(DEFAULTS))
        tiny_ok9 = True
    except Exception as e:
        tiny_ok9 = False
        print(f"    (unexpected exception: {e!r})")
captured9 = buf9.getvalue()
check("4x4 height_mask executes without exception (treated absent)", tiny_ok9)
control("implausible-mask guard: a 4x4 mask correctly prints the 'not a plausible mask' note",
        "not a plausible mask" in captured9, f"captured={captured9!r:.200}")


# ===========================================================================
print("\n[9b] Pixelate zones (spec v2 section 0, Overlay's pixelate_canvas verbatim): alters the photo inside the zone only")
NODE9b = SchematicTopographic()
p9b = dict(DEFAULTS)
p9b.update({"texture_opacity": 0.0, "pixelate_zones": "", "pixel_size": 16, "zone_size": 60, "pixel_stroke": False})
base9b = NODE9b.execute(image=photo_tensor, **p9b)
p9z = dict(p9b)
p9z["pixelate_zones"] = "300,200"
zon9b = NODE9b.execute(image=photo_tensor, **p9z)
# Overlay's pixelate_canvas round-trips the WHOLE canvas through 8 bits once a
# zone exists (verbatim behaviour, spec v2 section 0), so outside the zone the
# image may move by at most 1 LSB; inside it moves by real mosaic amounts.
img_abs = (base9b[0][0] - zon9b[0][0]).abs().max(dim=-1).values.numpy() * 255.0
yy9 = np.arange(img_abs.shape[0])[:, None]
xx9 = np.arange(img_abs.shape[1])[None, :]
inside_box = (xx9 >= 239) & (xx9 <= 361) & (yy9 >= 139) & (yy9 <= 261)
check("pixelate zone: the IMAGE output changes by > 8 LSB only inside the 60px half-width zone around (300,200); outside <= 1 LSB",
      float(img_abs[inside_box].max()) > 8.0 and float(img_abs[~inside_box].max()) <= 1.001,
      f"inside_max={img_abs[inside_box].max():.1f} outside_max={img_abs[~inside_box].max():.3f}")
ov_diff = (base9b[1][0] != zon9b[1][0]).any(dim=-1).numpy()
check("pixelate zone: the overlay_only output changes only inside the zone (label) - no stroke, no photo",
      bool(ov_diff[~((np.arange(ov_diff.shape[0])[:, None] >= 139) & (np.arange(ov_diff.shape[0])[:, None] <= 261)
                     & (np.arange(ov_diff.shape[1])[None, :] >= 239) & (np.arange(ov_diff.shape[1])[None, :] <= 361))].sum() == 0))
p9one = dict(p9z)
p9one["pixel_size"] = 1
one9b = NODE9b.execute(image=photo_tensor, **p9one)
check("pixel_size 1 disables pixelation bitwise (all 3 outputs equal the no-zone render)",
      all(torch.equal(base9b[k], one9b[k]) for k in range(3)))
control("zone-locality predicate: a zone at (600,300) correctly changes pixels OUTSIDE the (300,200) box",
        bool((base9b[0][0] != NODE9b.execute(image=photo_tensor, **dict(p9z, pixelate_zones="600,300"))[0][0]).any(dim=-1).numpy()[:, 500:].any()))
pm9 = np.zeros((PHOTO_H, PHOTO_W), dtype=np.float32)
pm9[280:320, 480:520] = 1.0
msk9b = NODE9b.execute(image=photo_tensor, pixelate_mask=torch.from_numpy(pm9[None]), **p9b)
m_abs = (base9b[0][0] - msk9b[0][0]).abs().max(dim=-1).values.numpy() * 255.0
m_box = (xx9 >= 439) & (xx9 <= 561) & (yy9 >= 239) & (yy9 <= 361)
check("pixelate_mask: a white square centred (500,300) pixelates (> 8 LSB) around its centroid only; outside <= 1 LSB",
      float(m_abs[m_box].max()) > 8.0 and float(m_abs[~m_box].max()) <= 1.001,
      f"inside_max={m_abs[m_box].max():.1f} outside_max={m_abs[~m_box].max():.3f}")


print("\n[10] Flat field: no crash, contour-free note, grid+marginalia still draw")

flat_grey10 = np.full((300, 300, 3), 0.5, dtype=np.float32)
flat_tensor10 = torch.from_numpy(flat_grey10[None, ...])
p10 = dict(DEFAULTS)
p10["height_source"] = "brightness"  # v2: a flat grey image is only FLAT through a photo source
p10["grid"] = "lines"
NODE10 = SchematicTopographic()
buf10 = io.StringIO()
with contextlib.redirect_stdout(buf10):
    try:
        out10 = NODE10.execute(image=flat_tensor10, **p10)
        ok10 = True
    except Exception as e:
        ok10 = False
        out10 = None
        print(f"    (unexpected exception: {e!r})")
captured10 = buf10.getvalue()
check("flat grey image: executes without exception", ok10)
check("flat grey image: console note contains 'contour-free'", "contour-free" in captured10,
      f"captured={captured10!r:.300}")
if ok10:
    alpha10 = out10[2][0].cpu().numpy()
    check("flat grey image: overlay_alpha has ink (grid+marginalia still draw)",
          float(np.max(alpha10)) > 0.01)
else:
    check("flat grey image: overlay_alpha has ink (grid+marginalia still draw) -- SKIPPED (exec failed)", False)

buf10b = io.StringIO()
with contextlib.redirect_stdout(buf10b):
    NODE10.execute(image=photo_tensor, **p10)
captured10b = buf10b.getvalue()
control("'contour-free' note predicate: a non-flat (photo) field correctly does NOT print it",
        "contour-free" not in captured10b)


# ===========================================================================
print("\n[12] Campsites: flat-ground rule, spacing, determinism, off-path identity, badge ink")

H12, _flat12 = topographic.build_height(photo_small_arr, "procedural", 10.0,
                                        terrain={"seed": 42, "scale": 420, "octaves": 3,
                                                 "roughness": 0.45, "stretch": 1.6, "angle": 30.0})
F12 = topographic.work_grid(H12, 2)
gx12, gy12 = topographic.gradient(F12)
h12, w12 = H12.shape
others12 = [(100.0, 100.0), (400.0, 220.0)]
camps12 = topographic.campsites(42, w12, h12, 8, 60, 22, others12, gx12, gy12, 2)
g_med12 = float(np.median(np.hypot(gx12, gy12)))


def _slope_ok(pts):
    return all(math.hypot(*topographic.sample_grad(gx12, gy12, x, y, 2)) <= g_med12 for (x, y) in pts)


def _spacing_ok(pts, others, md):
    allp = list(others) + list(pts)
    for a in range(len(allp)):
        for b in range(max(a + 1, len(others)), len(allp)):
            if dist(allp[a], allp[b]) < md - 1e-9:
                return False
    return True


check("campsites: count honoured (8 placed on the hero terrain)", len(camps12) == 8, f"{len(camps12)}")
check("campsites: every site on the flatter half (|grad| <= median slope)", _slope_ok(camps12))
steep = np.unravel_index(int(np.argmax(np.hypot(gx12, gy12))), gx12.shape)
control("flat-ground predicate rejects a site moved onto the steepest cell",
        not _slope_ok(camps12[:-1] + [(float(steep[1] * 2), float(steep[0] * 2))]))
check("campsites: >= max(min_dist, size) from each other and from spots/benchmarks",
      _spacing_ok(camps12, others12, 60.0))
control("spacing predicate rejects a site placed 10 px from another",
        not _spacing_ok(camps12 + [(camps12[0][0] + 10.0, camps12[0][1])], others12, 60.0))
m12 = 0.08 * min(w12, h12) + 11.0
check("campsites: inside the margin 0.08*min(w,h) + size/2",
      all(m12 <= x <= w12 - m12 and m12 <= y <= h12 - m12 for (x, y) in camps12))
check("campsites: deterministic across calls",
      camps12 == topographic.campsites(42, w12, h12, 8, 60, 22, others12, gx12, gy12, 2))
check("campsites: a different seed moves them",
      camps12 != topographic.campsites(43, w12, h12, 8, 60, 22, others12, gx12, gy12, 2))
check("campsites: huge min_dist gives a shortfall, never a crash",
      len(topographic.campsites(42, w12, h12, 8, 5000, 22, [], gx12, gy12, 2)) == 1)
check("campsites: count 0 -> empty", topographic.campsites(42, w12, h12, 0, 60, 22, [], gx12, gy12, 2) == [])

NODE12 = SchematicTopographic()
p_abs = {k: v for k, v in DEFAULTS.items() if not k.startswith("campsite")}
out_abs = NODE12.execute(image=photo_tensor, **p_abs)
out_off = NODE12.execute(image=photo_tensor, **{**p_abs, "campsites": False, "campsite_count": 40,
                                                "campsite_size": 90})
check("campsites=False is bitwise identical to the inputs being absent (all 3 outputs)",
      all(torch.equal(out_abs[k], out_off[k]) for k in range(3)))
out_on = NODE12.execute(image=photo_tensor, **{**p_abs, "campsites": True})
control("identity predicate rejects campsites=True (it does change the image)",
        not torch.equal(out_abs[0], out_on[0]))

# badge ink probe: one big badge, overlay_only, everything else off
p_ink = {**p_abs, "campsites": True, "campsite_count": 1, "campsite_size": 80, "spot_heights": False,
         "benchmarks": False, "grid": "off", "contour_weight": 0.1, "index_every": 0,
         "hachures": "off", "north_arrow": False, "scale_bar": False, "interval_note": False,
         "frame_text_tl": "", "frame_text_tr": "", "texture_opacity": 0.0, "contour_interval": 500.0}
_captured = {}
_node_topo = sys.modules["cs_pkg.schematic.topographic"]
_orig_node_cs = _node_topo.campsites


def _spy_cs(*a, **k):
    r = _orig_node_cs(*a, **k)
    _captured["c"] = r
    return r


_node_topo.campsites = _spy_cs
out_ink = NODE12.execute(image=photo_tensor, **p_ink)
_node_topo.campsites = _orig_node_cs
ov = (out_ink[1][0].numpy() * 255.0)
cx12, cy12 = _captured["c"][0]
r12 = 40.0
disc_px = ov[int(round(cy12 + 0.6 * r12)), int(round(cx12))]
stripe_px = ov[int(round(cy12 - 0.033 * r12)), int(round(cx12 + 0.245 * r12))]
ring_px = ov[int(round(cy12 + r12 - 0.03 * 80.0)), int(round(cx12))]  # mid-ring, 6% of 80 px wide
check("badge: disc below the mark is background colour", np.abs(disc_px - np.array(BG_RGB)).max() <= 2,
      f"{disc_px}")
check("badge: stripe centroid carries the mark in stroke colour",
      np.abs(stripe_px - np.array(STROKE_RGB)).max() <= 2, f"{stripe_px}")
check("badge: rim ring is stroke colour", np.abs(ring_px - np.array(STROKE_RGB)).max() <= 2, f"{ring_px}")
control("badge probe rejects the background colour at the mark point",
        np.abs(stripe_px - np.array(BG_RGB)).max() > 2)


# ===========================================================================
print("\n[11] Perf (soft, printed only, NOT gated by this suite): default full-size hero render")

hero_full_arr = (np.asarray(_hero_full_pil).astype(np.float32) / 255.0)
hero_full_tensor = torch.from_numpy(hero_full_arr[None, ...])
NODE11 = SchematicTopographic()
t0_perf = time.time()
NODE11.execute(image=hero_full_tensor, **dict(DEFAULTS))
dt_perf = time.time() - t0_perf
print(f"  default {hero_full_arr.shape[1]}x{hero_full_arr.shape[0]} full render: {dt_perf:.3f} s "
      f"(soft print only, not gated by this suite)")


# ===========================================================================
print()
print(f"{PASS_COUNT} PASS / {FAIL_COUNT} FAIL / {CTRL_COUNT} controls"
      + (f" ({CTRL_FAIL_COUNT} CTRL-FAIL)" if CTRL_FAIL_COUNT else ""))
if FAIL_COUNT > 0 or CTRL_FAIL_COUNT > 0:
    print("SOME CHECKS FAILED")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
    sys.exit(0)
