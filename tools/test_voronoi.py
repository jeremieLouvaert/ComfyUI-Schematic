"""Blind-teeth suite for the Schematic Voronoi node (spec section 8, items 1-10).
Implementation-blind: written against docs/schematic-voronoi-derivation.md ONLY.
No pytest; prints PASS/FAIL per check and exits nonzero if any check fails.

Pure-level checks call schematic.voronoi (+ schematic.engine, schematic.render_voronoi)
as black-box APIs per the frozen module surface (spec section 7). Node-level checks
drive nodes.SchematicVoronoi.execute with real torch tensors via the smoke_node.py
package-shim pattern. Run with the ComfyUI embedded python (needs torch+scipy).

Items:
 1. Field polarity (exact zero density/seeds on the dark half) + darkness-flip control.
 2. Uniform field -> target count reached, quadrant balance within tolerance.
 3. Determinism: seeds/pixels bitwise on rerun; differs on reseed; batch==single.
 4. Degenerate: all-black+brightness -> no crash, mesh-free, note printed; flat-gray runs.
 5. Closure: zero real cells with infinite regions (mirror construction).
 6. Alpha grading: dense side has strictly higher mean ridge alpha; inverted control.
 7. cells honors count: reached on bright field; budget-exhausted on near-black field.
 8. Lazy scipy: fresh-interpreter import must not pull in scipy until build_cells runs.
 9. Render probes: bg-only exact, overlay_alpha zero at op=0, early-out bitwise passthrough.
10. Perf printed (soft, no gate): default 1200x1600 render.
"""
import contextlib
import io
import math
import os
import subprocess
import sys
import time

import numpy as np

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)

from schematic import voronoi, engine, render_voronoi  # pure, black-box per spec section 7

import torch
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

SchematicVoronoi = sys.modules["cs_pkg.nodes"].NODE_CLASS_MAPPINGS["SchematicVoronoi"]

PASS = True


def check(name, cond, detail=""):
    global PASS
    status = "PASS" if cond else "FAIL"
    if not cond:
        PASS = False
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


def node_defaults():
    """Programmatic widget defaults from INPUT_TYPES, minus socket inputs (smoke_node.py style)."""
    node = SchematicVoronoi()
    it = node.INPUT_TYPES()
    d = {}
    for section in ("required", "optional"):
        for k, v in it.get(section, {}).items():
            if isinstance(v, tuple) and len(v) > 1 and isinstance(v[1], dict) and "default" in v[1]:
                d[k] = v[1]["default"]
            elif isinstance(v, tuple) and isinstance(v[0], list):
                d[k] = v[0][0]
    d.pop("image", None)
    d.pop("texture", None)
    return d


def reference_sample_seeds(density, cells, seed, w, h):
    """Independent oracle: replays spec section 3's exact triple/accept/budget rule
    using engine.MinstdLCG as the RNG (black-box oracle per task allowance). Does NOT
    call voronoi.sample_seeds internally."""
    lcg = engine.MinstdLCG(seed)
    h_arr, w_arr = density.shape
    budget = 80 * cells
    accepted = []
    draws = 0
    while len(accepted) < cells and draws < budget:
        x = lcg.draw() * w
        y = lcg.draw() * h
        u = lcg.draw()
        draws += 1
        xi = min(max(int(math.floor(x)), 0), w_arr - 1)
        yi = min(max(int(math.floor(y)), 0), h_arr - 1)
        if u < density[yi, xi]:
            accepted.append((x, y))
    return np.array(accepted, dtype=np.float64).reshape(-1, 2), draws


def build_elements(density, seeds, w, h):
    """Pure-API assembly of the render_frame_voronoi 'elements' dict (section 7)."""
    cells_out = voronoi.build_cells(seeds, w, h)
    ridge_idx = np.array([(r[2], r[3]) for r in cells_out["ridges"]], dtype=np.int64).reshape(-1, 2)
    alphas = voronoi.ridge_alpha(cells_out["areas"], ridge_idx)
    xi = np.clip(seeds[:, 0].astype(int), 0, w - 1)
    yi = np.clip(seeds[:, 1].astype(int), 0, h - 1)
    field_at_seeds = density[yi, xi]
    return {
        "regions": cells_out["regions"], "areas": cells_out["areas"],
        "ridges": cells_out["ridges"], "alphas": alphas,
        "seeds": seeds, "field_at_seeds": field_at_seeds,
    }, ridge_idx


NODE_DEFAULTS = node_defaults()
BG_RGB, STROKE_RGB = engine.resolve_palette("whiteOnDark", "#0a0a0a", "#d8d8d8")


# ===========================================================================
print("[1] Field polarity: half-black/half-white, exact zero density on the dark half")

W1, H1 = 200, 160
half = W1 // 2
fitted_bw = np.zeros((H1, W1, 3), dtype=np.float32)
fitted_bw[:, half:, :] = 1.0  # right half white, left half black

for gamma in (0.5, 2.5):
    field_b = voronoi.build_density(fitted_bw, "brightness", gamma)
    check(f"brightness density EXACTLY 0 on black (left) half, gamma={gamma}",
          bool(np.all(field_b[:, :half] == 0.0)))
    check(f"brightness density EXACTLY 1 on white (right) half, gamma={gamma}",
          bool(np.all(field_b[:, half:] == 1.0)))
    seeds_b = voronoi.sample_seeds(field_b, 3000, 42, W1, H1)
    n_black = int(np.sum(seeds_b[:, 0] < half))
    check(f"brightness: ZERO seeds land on the black half, gamma={gamma}", n_black == 0,
          f"n_black={n_black}, n_total={len(seeds_b)}")
    check(f"brightness: seeds non-vacuous (some accepted), gamma={gamma}", len(seeds_b) > 0)

# independent-oracle cross-check (bitwise) at gamma=2.5
field_b25 = voronoi.build_density(fitted_bw, "brightness", 2.5)
seeds_impl = voronoi.sample_seeds(field_b25, 3000, 42, W1, H1)
seeds_ref, draws_ref = reference_sample_seeds(field_b25, 3000, 42, W1, H1)
same_count = len(seeds_impl) == len(seeds_ref)
max_diff = float(np.max(np.abs(seeds_impl - seeds_ref))) if same_count and len(seeds_ref) else -1.0
check("sample_seeds matches independent MINSTD-replay oracle bitwise (brightness)",
      same_count and max_diff < 1e-9, f"n_impl={len(seeds_impl)} n_ref={len(seeds_ref)} max|d|={max_diff:.3e}")

# darkness control: flips which half is zero
field_d = voronoi.build_density(fitted_bw, "darkness", 2.5)
check("darkness density EXACTLY 1 on (now-hot) black/left half",
      bool(np.all(field_d[:, :half] == 1.0)))
check("darkness density EXACTLY 0 on (now-cold) white/right half",
      bool(np.all(field_d[:, half:] == 0.0)))
seeds_d = voronoi.sample_seeds(field_d, 3000, 42, W1, H1)
n_white = int(np.sum(seeds_d[:, 0] >= half))
check("CONTROL: darkness source flips the zero side -> ZERO seeds on white half", n_white == 0,
      f"n_white={n_white}, n_total={len(seeds_d)}")
check("CONTROL: darkness source yields non-vacuous seeds on black half", len(seeds_d) > 0)


# ===========================================================================
print("\n[2] Uniform field: target count reached, quadrant balance within +-25%")

W2 = H2 = 400
flat_gray = np.full((H2, W2, 3), 0.5, dtype=np.float32)
field_u = voronoi.build_density(flat_gray, "brightness", 2.5)
check("uniform gray -> density EXACTLY 1.0 everywhere (any gamma, after max-normalize)",
      bool(np.all(field_u == 1.0)))

cells2 = 4000
seeds_u = voronoi.sample_seeds(field_u, cells2, 42, W2, H2)
check("uniform field reaches the target count exactly", len(seeds_u) == cells2, f"n={len(seeds_u)}")

qx = seeds_u[:, 0] >= W2 / 2
qy = seeds_u[:, 1] >= H2 / 2
quad_counts = [int(np.sum((~qx) & (~qy))), int(np.sum(qx & (~qy))),
               int(np.sum((~qx) & qy)), int(np.sum(qx & qy))]
target = cells2 / 4.0
lo, hi = 0.75 * target, 1.25 * target
quad_ok = all(lo <= c <= hi for c in quad_counts)
check("quadrant seed counts within +-25% of n/4 (loose uniformity)", quad_ok,
      f"counts={quad_counts} target={target:.0f} range=[{lo:.0f},{hi:.0f}]")

# oracle cross-check: density==1 means every draw is accepted, so seeds are just the
# first `cells` (x,y) draws of the stream in order (fast path, exact bitwise expected)
seeds_u_ref, _ = reference_sample_seeds(field_u, cells2, 42, W2, H2)
diff_u = float(np.max(np.abs(seeds_u - seeds_u_ref)))
check("uniform-field seeds match independent MINSTD-replay oracle bitwise",
      len(seeds_u) == len(seeds_u_ref) and diff_u < 1e-9, f"max|d|={diff_u:.3e}")


# ===========================================================================
print("\n[3] Determinism: seeds/pixels bitwise on rerun; reseed differs; batch==single")

W3, H3 = 300, 240
gx = np.linspace(0, 1, W3, dtype=np.float32)
synth3 = np.tile(gx, (H3, 1))[:, :, None] * np.ones((1, 1, 3), dtype=np.float32)
density3 = voronoi.build_density(synth3, "brightness", 2.0)

seeds3a = voronoi.sample_seeds(density3, 600, 42, W3, H3)
seeds3b = voronoi.sample_seeds(density3, 600, 42, W3, H3)
check("sample_seeds: bitwise identical on same-args rerun", np.array_equal(seeds3a, seeds3b))

seeds3c = voronoi.sample_seeds(density3, 600, 43, W3, H3)
check("sample_seeds: different seed differs", not np.array_equal(seeds3a, seeds3c))

cellsout_a = voronoi.build_cells(seeds3a, W3, H3)
cellsout_b = voronoi.build_cells(seeds3a, W3, H3)
areas_equal = np.array_equal(cellsout_a["areas"], cellsout_b["areas"])
regions_equal = len(cellsout_a["regions"]) == len(cellsout_b["regions"]) and all(
    np.array_equal(np.asarray(ra), np.asarray(rb))
    for ra, rb in zip(cellsout_a["regions"], cellsout_b["regions"]))
ridges_equal = cellsout_a["ridges"] == cellsout_b["ridges"]
check("build_cells: areas bitwise identical on rerun", areas_equal)
check("build_cells: regions bitwise identical on rerun", regions_equal)
check("build_cells: ridges identical on rerun", ridges_equal)

# --- node-level: pixel determinism, reseed, batch-vs-single (all 3 outputs) ---
rng3 = np.random.default_rng(11)
imgA = torch.from_numpy(rng3.random((1, H3, W3, 3), dtype=np.float32))
imgB = torch.from_numpy(rng3.random((1, H3, W3, 3), dtype=np.float32))
p3 = dict(NODE_DEFAULTS)
p3["cells"] = 800  # keep node-level determinism checks light

node3 = SchematicVoronoi()
out1 = node3.execute(image=imgA, texture=None, **p3)
out2 = node3.execute(image=imgA, texture=None, **p3)
pix_det = all(torch.equal(out1[k], out2[k]) for k in range(3))
check("node.execute: bitwise identical pixels on same-args rerun (all 3 outputs)", pix_det)

p3_reseed = dict(p3)
p3_reseed["seed"] = p3["seed"] + 1
out3 = node3.execute(image=imgA, texture=None, **p3_reseed)
check("node.execute: different seed widget -> image output differs", not torch.equal(out1[0], out3[0]))

batch_img = torch.cat([imgA, imgB], dim=0)
out_batch = node3.execute(image=batch_img, texture=None, **p3)
out_single_A = node3.execute(image=imgA, texture=None, **p3)
out_single_B = node3.execute(image=imgB, texture=None, **p3)
batch_a_ok = all(torch.equal(out_batch[k][0], out_single_A[k][0]) for k in range(3))
batch_b_ok = all(torch.equal(out_batch[k][1], out_single_B[k][0]) for k in range(3))
check("batch frame 0 == same frame run alone (all 3 outputs, same seed every frame)", batch_a_ok)
check("batch frame 1 == same frame run alone (all 3 outputs, same seed every frame)", batch_b_ok)


# ===========================================================================
print("\n[4] Degenerate: all-black+brightness mesh-free + note; flat-gray runs")

W4, H4 = 96, 80
fitted_black4 = np.zeros((H4, W4, 3), dtype=np.float32)
density_black4 = voronoi.build_density(fitted_black4, "brightness", 2.5)
seeds_black4 = voronoi.sample_seeds(density_black4, 500, 42, W4, H4)
check("pure: all-black + brightness -> zero accepted seeds (no crash)", len(seeds_black4) == 0,
      f"n={len(seeds_black4)}")

flat_small = np.full((64, 64, 3), 0.5, dtype=np.float32)
density_flat4 = voronoi.build_density(flat_small, "brightness", 2.5)
seeds_flat4 = voronoi.sample_seeds(density_flat4, 500, 42, 64, 64)
check("pure: all-flat-gray runs without crash and yields seeds (uniform density)",
      len(seeds_flat4) > 0, f"n={len(seeds_flat4)}")

# --- node-level: degenerate path prints a note and renders mesh-free ---
img_black4 = torch.zeros((1, H4, W4, 3), dtype=torch.float32)
p4 = dict(NODE_DEFAULTS)
p4.update({"cells": 2000, "texture_opacity": 0.0, "cell_fill": 0.0, "seed_dots": False,
           "frame_text_tl": "", "frame_text_tr": "", "frame_text_bl": "", "frame_text_br": ""})
node4 = SchematicVoronoi()
buf4 = io.StringIO()
with contextlib.redirect_stdout(buf4):
    out4 = node4.execute(image=img_black4, texture=None, **p4)
captured4 = buf4.getvalue()
check("node: degenerate all-black+brightness executes without raising", out4 is not None)
check("node: '[Schematic]' note printed on degenerate skip", "[Schematic]" in captured4,
      f"captured={captured4!r:.200}")
img4_np = out4[0][0].numpy()
distinct4 = len(set(map(tuple, (img4_np.reshape(-1, 3) * 255).round().astype(int))))
check("node: degenerate render is a single uniform flat color (mesh-free interior, no text/texture/fill)",
      distinct4 == 1, f"distinct_colors={distinct4}")

# --- node-level: flat-gray runs end to end ---
img_flat4 = torch.full((1, 64, 64, 3), 0.5, dtype=torch.float32)
p4b = dict(NODE_DEFAULTS)
p4b["cells"] = 1500
out4b = node4.execute(image=img_flat4, texture=None, **p4b)
finite_ok = all(torch.isfinite(out4b[k]).all() for k in range(3))
check("node: all-flat-gray runs end to end (finite outputs, correct shape)",
      finite_ok and out4b[0].shape == (1, 64, 64, 3), f"shape={tuple(out4b[0].shape)}")


# ===========================================================================
print("\n[5] Closure: zero real cells with infinite regions (mirror construction)")
import scipy.spatial  # independent oracle only; voronoi.py lazy-imports it on its own

W5, H5 = 250, 200
gx5 = np.linspace(0, 1, W5, dtype=np.float32)
synth5 = np.tile(gx5, (H5, 1))[:, :, None] * np.ones((1, 1, 3), dtype=np.float32)
density5 = voronoi.build_density(synth5, "brightness", 2.0)
seeds5 = voronoi.sample_seeds(density5, 150, 42, W5, H5)
n5 = len(seeds5)
check("closure setup: non-degenerate seed count", n5 >= 4, f"n5={n5}")

refl_l = np.column_stack([-seeds5[:, 0], seeds5[:, 1]])
refl_r = np.column_stack([2 * W5 - seeds5[:, 0], seeds5[:, 1]])
refl_t = np.column_stack([seeds5[:, 0], -seeds5[:, 1]])
refl_b = np.column_stack([seeds5[:, 0], 2 * H5 - seeds5[:, 1]])
mirror_pts = np.vstack([seeds5, refl_l, refl_r, refl_t, refl_b])
check("closure: mirror set is exactly 5n points", len(mirror_pts) == 5 * n5)

vor_mirror = scipy.spatial.Voronoi(mirror_pts)
n_infinite_mirror = sum(1 for i in range(n5) if -1 in vor_mirror.regions[vor_mirror.point_region[i]])
check("ORACLE (independent scipy call): zero real cells with infinite region after mirror-closure",
      n_infinite_mirror == 0, f"count={n_infinite_mirror} of n={n5}")

vor_naive = scipy.spatial.Voronoi(seeds5)
n_infinite_naive = sum(1 for i in range(n5) if -1 in vor_naive.regions[vor_naive.point_region[i]])
check("CONTROL: WITHOUT mirror closure, plain Voronoi DOES have infinite regions (teeth present)",
      n_infinite_naive > 0, f"count={n_infinite_naive} of n={n5}")

cellsout5 = voronoi.build_cells(seeds5, W5, H5)
check("build_cells: one region per real seed", len(cellsout5["regions"]) == n5,
      f"regions={len(cellsout5['regions'])} n5={n5}")
all_finite5 = all(np.all(np.isfinite(np.asarray(r))) for r in cellsout5["regions"])
all_poly5 = all(len(np.asarray(r)) >= 3 for r in cellsout5["regions"])
eps5 = 1e-3
all_bounded5 = all(
    np.all(np.asarray(r)[:, 0] >= -eps5) and np.all(np.asarray(r)[:, 0] <= W5 + eps5) and
    np.all(np.asarray(r)[:, 1] >= -eps5) and np.all(np.asarray(r)[:, 1] <= H5 + eps5)
    for r in cellsout5["regions"])
check("build_cells: every real region is finite, >=3 vertices, within canvas bounds",
      all_finite5 and all_poly5 and all_bounded5)


def _shoelace(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


oracle_areas5 = np.array([_shoelace(np.asarray(r)) for r in cellsout5["regions"]])
rel_diff5 = float(np.max(np.abs(oracle_areas5 - cellsout5["areas"]))) / max(1.0, float(np.max(cellsout5["areas"])))
check("build_cells areas match independent shoelace oracle on its own region vertices",
      rel_diff5 < 1e-6, f"rel_diff={rel_diff5:.3e}")


# ===========================================================================
print("\n[6] Alpha grading: crafted density step -> mean ridge alpha higher on dense side")

W6, H6 = 300, 240
half6 = W6 / 2.0
cells6 = 1200


def _classify(i, j, seeds, boundary):
    xs = []
    if i >= 0:
        xs.append(seeds[i, 0])
    if j >= 0:
        xs.append(seeds[j, 0])
    return "L" if (sum(xs) / len(xs)) < boundary else "R"


def _ridge_sides_and_alphas(density_map, seed_val):
    seeds = voronoi.sample_seeds(density_map, cells6, seed_val, W6, H6)
    cout = voronoi.build_cells(seeds, W6, H6)
    ridge_idx = np.array([(r[2], r[3]) for r in cout["ridges"]], dtype=np.int64).reshape(-1, 2)
    alphas = voronoi.ridge_alpha(cout["areas"], ridge_idx)
    sides = [_classify(i, j, seeds, half6) for (i, j) in ridge_idx]
    return seeds, alphas, np.array(sides)


density_step = np.full((H6, W6), 0.2, dtype=np.float64)
density_step[:, : int(half6)] = 1.0  # dense LEFT, sparse RIGHT

seeds6, alphas6, sides6 = _ridge_sides_and_alphas(density_step, 42)
check("alpha grading: all ridge alphas within spec bound [0.15, 1.0]",
      bool(np.all(alphas6 >= 0.15) and np.all(alphas6 <= 1.0)))
a_dense = alphas6[sides6 == "L"]
a_sparse = alphas6[sides6 == "R"]
check("alpha grading: both sides have ridges (non-vacuous)", len(a_dense) > 0 and len(a_sparse) > 0,
      f"n_L={len(a_dense)} n_R={len(a_sparse)}")
mean_dense = float(np.mean(a_dense))
mean_sparse = float(np.mean(a_sparse))
check("mean ridge alpha strictly higher on the dense (left) side", mean_dense > mean_sparse,
      f"dense={mean_dense:.4f} sparse={mean_sparse:.4f}")

# CONTROL: invert the map (dense RIGHT, sparse LEFT) -> the left side's mean alpha must FALL
density_step_inv = np.full((H6, W6), 0.2, dtype=np.float64)
density_step_inv[:, int(half6):] = 1.0
seeds6b, alphas6b, sides6b = _ridge_sides_and_alphas(density_step_inv, 42)
mean_left_inv = float(np.mean(alphas6b[sides6b == "L"]))
mean_right_inv = float(np.mean(alphas6b[sides6b == "R"]))
check("CONTROL: inverted map LOWERS mean alpha on the (now-sparse) left side",
      mean_left_inv < mean_dense, f"orig_left(dense)={mean_dense:.4f} inverted_left(sparse)={mean_left_inv:.4f}")
check("CONTROL: inverted map RAISES mean alpha on the (now-dense) right side",
      mean_right_inv > mean_sparse, f"orig_right(sparse)={mean_sparse:.4f} inverted_right(dense)={mean_right_inv:.4f}")


# ===========================================================================
print("\n[7] cells honors count: reached on bright field; budget-exhausted on near-black field")

W7, H7 = 220, 180
density_bright7 = np.ones((H7, W7), dtype=np.float64)
cells7 = 2000
seeds_bright7 = voronoi.sample_seeds(density_bright7, cells7, 42, W7, H7)
check("count-reached scalar: n == cells exactly on a bright-enough (density=1) field",
      len(seeds_bright7) == cells7, f"n={len(seeds_bright7)}")

ref_bright7, draws_bright7 = reference_sample_seeds(density_bright7, cells7, 42, W7, H7)
check("count-reached: independent oracle also consumes exactly `cells` triples (every draw accepted)",
      draws_bright7 == cells7, f"draws={draws_bright7} cells={cells7}")
diff_bright7 = float(np.max(np.abs(seeds_bright7 - ref_bright7)))
check("bright-field seeds match independent MINSTD-replay oracle bitwise",
      len(seeds_bright7) == len(ref_bright7) and diff_bright7 < 1e-9, f"max|d|={diff_bright7:.3e}")

W7b, H7b = 300, 300
density_black7 = np.zeros((H7b, W7b), dtype=np.float64)
density_black7[0:2, 0:2] = 1.0  # tiny bright spot; overwhelming majority of area is exactly 0
cells7b = 500
budget7b = 80 * cells7b
seeds_black7 = voronoi.sample_seeds(density_black7, cells7b, 42, W7b, H7b)
check("budget-exhausted scalar: n < cells on a nearly-black field (only remaining stop condition)",
      len(seeds_black7) < cells7b, f"n={len(seeds_black7)} cells={cells7b}")

ref_black7, draws_black7 = reference_sample_seeds(density_black7, cells7b, 42, W7b, H7b)
check("budget-exhausted: independent oracle also exhausts the full 80*cells draw budget",
      draws_black7 == budget7b and len(ref_black7) < cells7b,
      f"draws={draws_black7} budget={budget7b} n_ref={len(ref_black7)}")
diff_black7 = float(np.max(np.abs(seeds_black7 - ref_black7))) if len(ref_black7) else 0.0
check("near-black seeds match independent MINSTD-replay oracle bitwise",
      len(seeds_black7) == len(ref_black7) and diff_black7 < 1e-9,
      f"n_impl={len(seeds_black7)} n_ref={len(ref_black7)} max|d|={diff_black7:.3e}")


# ===========================================================================
print("\n[8] Lazy scipy: fresh interpreter must not import scipy until build_cells runs")

pack_posix = PACK.replace("\\", "/")
lazy_script = (
    "import sys\n"
    f"sys.path.insert(0, \"{pack_posix}\")\n"
    "import schematic.voronoi as voronoi\n"
    "assert 'scipy' not in sys.modules, 'scipy imported eagerly on module import'\n"
    "import numpy as np\n"
    "seeds = np.array([[1.0,1.0],[2.0,2.0],[3.0,1.0],[1.0,3.0],[3.0,3.0]], dtype=np.float64)\n"
    "voronoi.build_cells(seeds, 10, 10)\n"
    "assert 'scipy' in sys.modules, 'scipy not imported after build_cells ran'\n"
    "print('LAZY_SCIPY_OK')\n"
)
lazy_result = subprocess.run([sys.executable, "-c", lazy_script], capture_output=True, text=True, timeout=120)
check("lazy scipy: fresh-interpreter subprocess exits 0", lazy_result.returncode == 0,
      f"stderr_tail={lazy_result.stderr[-300:]}")
check("lazy scipy: sentinel confirms scipy absent-before/present-after build_cells",
      "LAZY_SCIPY_OK" in lazy_result.stdout,
      f"stdout={lazy_result.stdout!r} stderr_tail={lazy_result.stderr[-300:]!r}")


# ===========================================================================
print("\n[9] Render probes: bg-only, overlay_alpha zero at op=0, screen-blend, early-out")

# INTERPRETATION: render_frame_voronoi's `params` blob is the widget dict minus the
# palette/size fields consumed elsewhere (mirrors test_schematic.py's DEFAULT_PARAMS
# convention for the sibling node's render.render_frame).
PURE_PARAMS = {k: v for k, v in NODE_DEFAULTS.items()
               if k not in ("palette", "bg_color", "stroke_color", "size_preset")}

w9, h9 = 64, 64
fitted9 = np.zeros((h9, w9, 3), dtype=np.float32)
empty_elements9 = {"regions": [], "areas": np.array([]), "ridges": [], "alphas": np.array([]),
                    "seeds": np.zeros((0, 2)), "field_at_seeds": np.array([])}

p9a = dict(PURE_PARAMS)
p9a.update({"overlay_opacity": 0.0, "texture_opacity": 0.0, "cell_fill": 0.0, "seed_dots": False,
            "image_opacity": 0.0, "frame_text_tl": "", "frame_text_tr": "", "frame_text_bl": "", "frame_text_br": ""})
out9a, alpha9a = render_voronoi.render_frame_voronoi(fitted9, w9, h9, (BG_RGB, STROKE_RGB), p9a,
                                                       empty_elements9, None, include_image=False, font_dir=PACK)
bg_match9 = (np.all(out9a[..., 0] == BG_RGB[0]) and np.all(out9a[..., 1] == BG_RGB[1])
             and np.all(out9a[..., 2] == BG_RGB[2]))
check("bg-only render == palette bg exactly", bg_match9, f"bg={BG_RGB} sample={out9a[0, 0].tolist()}")
check("bg-only overlay_alpha is all zero", float(np.max(alpha9a)) == 0.0)

# 9b: overlay_alpha all-zero at op=0 & texture_opacity=0 EVEN WITH real mesh content present
w9b, h9b = 120, 100
gx9b = np.linspace(0, 1, w9b, dtype=np.float32)
synth9b = np.tile(gx9b, (h9b, 1))[:, :, None] * np.ones((1, 1, 3), dtype=np.float32)
density9b = voronoi.build_density(synth9b, "brightness", 2.0)
seeds9b = voronoi.sample_seeds(density9b, 300, 42, w9b, h9b)
check("render-probe setup: non-degenerate mesh for the op=0 check", len(seeds9b) >= 4, f"n={len(seeds9b)}")
elements9b, _ = build_elements(density9b, seeds9b, w9b, h9b)
p9b = dict(PURE_PARAMS)
p9b.update({"overlay_opacity": 0.0, "texture_opacity": 0.0})
out9b, alpha9b = render_voronoi.render_frame_voronoi(synth9b, w9b, h9b, (BG_RGB, STROKE_RGB), p9b,
                                                       elements9b, None, include_image=True, font_dir=PACK)
check("overlay_alpha all-zero when overlay_opacity=0 AND texture_opacity=0 (real mesh present)",
      float(np.max(alpha9b)) == 0.0, f"max_alpha={float(np.max(alpha9b))}")

# 9c: screen-blend never darkens (end-to-end through render_frame_voronoi, mesh-free isolation)
rng9c = np.random.default_rng(5)
texture9c = rng9c.random((h9, w9, 3)).astype(np.float32)
p9c_base = dict(PURE_PARAMS)
p9c_base.update({"overlay_opacity": 0.0, "cell_fill": 0.0, "seed_dots": False,
                  "frame_text_tl": "", "frame_text_tr": "", "frame_text_bl": "", "frame_text_br": ""})
out9c_notex, _ = render_voronoi.render_frame_voronoi(fitted9, w9, h9, (BG_RGB, STROKE_RGB),
                                                       {**p9c_base, "texture_opacity": 0.0},
                                                       empty_elements9, None, include_image=False, font_dir=PACK)
for op_tex in (0.3, 0.7, 1.0):
    out9c_tex, _ = render_voronoi.render_frame_voronoi(fitted9, w9, h9, (BG_RGB, STROKE_RGB),
                                                         {**p9c_base, "texture_opacity": op_tex},
                                                         empty_elements9, texture9c, include_image=False, font_dir=PACK)
    never_darkens = np.all(out9c_tex.astype(int) >= out9c_notex.astype(int) - 1)  # +-1 uint8 rounding slack
    check(f"screen blend never darkens canvas (texture_opacity={op_tex})", bool(never_darkens),
          f"min(delta)={int(np.min(out9c_tex.astype(int) - out9c_notex.astype(int)))}")

# 9d: early-out passthrough bitwise (node level; canvas-size guard via default size_preset)
rng9d = np.random.default_rng(13)
img9d = torch.from_numpy(rng9d.random((1, 80, 96, 3), dtype=np.float32))
p9d = dict(NODE_DEFAULTS)
p9d.update({"image_opacity": 1.0, "overlay_opacity": 0.0, "texture_opacity": 0.0,
            "cell_fill": 0.0, "seed_dots": False})
node9 = SchematicVoronoi()
out9d = node9.execute(image=img9d, texture=None, **p9d)
check("early-out: image output bitwise identical to input (opaque image, overlay+texture off)",
      torch.equal(out9d[0], img9d))


# ===========================================================================
print("\n[10] Perf (soft, printed only, NOT gated per task scope): default 1200x1600 render")


def run_full_voronoi_pipeline(w, h, seed_img=1):
    rng_p = np.random.default_rng(seed_img)
    frame = rng_p.random((h, w, 3)).astype(np.float32)
    src = NODE_DEFAULTS.get("density_source", "brightness")
    gamma = NODE_DEFAULTS.get("density_gamma", 2.5)
    cells_n = NODE_DEFAULTS.get("cells", 12000)
    seed_v = NODE_DEFAULTS.get("seed", 42)
    density = voronoi.build_density(frame, src, gamma)
    seeds = voronoi.sample_seeds(density, cells_n, seed_v, w, h)
    if len(seeds) >= 4:
        elements, _ = build_elements(density, seeds, w, h)
    else:
        elements = {"regions": [], "areas": np.array([]), "ridges": [], "alphas": np.array([]),
                    "seeds": seeds, "field_at_seeds": np.array([])}
    out, alpha = render_voronoi.render_frame_voronoi(frame, w, h, (BG_RGB, STROKE_RGB), PURE_PARAMS,
                                                       elements, None, include_image=True, font_dir=PACK)
    return out, alpha, len(seeds)


t0_perf = time.time()
_, _, n_perf = run_full_voronoi_pipeline(1200, 1600)
dt_perf = time.time() - t0_perf
print(f"  1200x1600 full default pipeline (cells target={NODE_DEFAULTS.get('cells', 12000)}, "
      f"n_accepted={n_perf}): {dt_perf:.3f} s  (spec soft budget <20s embedded; not gated by this suite)")


# ===========================================================================
print()
if PASS:
    print("ALL CHECKS PASSED")
    sys.exit(0)
else:
    print("SOME CHECKS FAILED")
    sys.exit(1)
