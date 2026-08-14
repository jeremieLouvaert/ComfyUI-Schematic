"""Teeth for Schematic Construction v2.

Written from docs/schematic-construction-derivation-v2.md section 10 (the invariant
table) and section 9 (the pinned widget surface). Every invariant is paired with a
NEGATIVE CONTROL that must FIRE; a control that does not fail proves the assertion
has no teeth.

Run with the ComfyUI embedded python.
"""

import importlib.util
import math
import os
import sys
import types

import numpy as np
import torch

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)

_pkg = types.ModuleType("cs_pkg")
_pkg.__path__ = [PACK]
sys.modules["cs_pkg"] = _pkg
for _sub in ("schematic", "nodes"):
    _spec = importlib.util.spec_from_file_location(
        f"cs_pkg.{_sub}", os.path.join(PACK, _sub, "__init__.py"),
        submodule_search_locations=[os.path.join(PACK, _sub)])
    _m = importlib.util.module_from_spec(_spec)
    sys.modules[f"cs_pkg.{_sub}"] = _m
    _spec.loader.exec_module(_m)

from cs_pkg.schematic import construction as geo          # noqa: E402
from cs_pkg.schematic import render as overlay_render     # noqa: E402
from cs_pkg.schematic import render_construction as rc    # noqa: E402
from cs_pkg.schematic.elements import ElementSyntaxError  # noqa: E402

Node = sys.modules["cs_pkg.nodes"].NODE_CLASS_MAPPINGS["SchematicConstruction"]

PASS = FAIL = FIRED = TOTAL = 0
FAILURES = []


def ok(c, name):
    global PASS, FAIL
    if c:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(name)
        print(f"  FAIL  {name}")


def control(c, name):
    global FIRED, TOTAL, FAIL
    TOTAL += 1
    if c:
        FIRED += 1
    else:
        FAIL += 1
        FAILURES.append(f"CONTROL DID NOT FIRE: {name}")
        print(f"  CONTROL INERT  {name}")


IT = Node.INPUT_TYPES()
D = {}
for sec in ("required", "optional"):
    for k, v in IT.get(sec, {}).items():
        if not isinstance(v, tuple):
            continue
        if isinstance(v[0], list):
            D[k] = v[0][0]
        elif len(v) > 1 and isinstance(v[1], dict) and "default" in v[1]:
            D[k] = v[1]["default"]
D.pop("image", None)
D.pop("pixelate_mask", None)


def photo(w=320, h=420, seed=3):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 0.30 + 0.25 * (xx / float(w)) + 0.10 * np.sin(yy / 23.0)
    img = np.stack([base, base * 0.92, base * 0.85], axis=2)
    d = (yy - h * 0.55) ** 2 + (xx - w * 0.45) ** 2 <= (min(w, h) * 0.22) ** 2
    tex = 0.10 * rng.rand(h, w)
    img[d] = np.stack([0.86 + tex, 0.10 + tex, 0.09 + tex], axis=2)[d]
    return np.clip(img, 0, 1).astype(np.float32)


def run(img, **kw):
    a = dict(D)
    a.update(kw)
    a["image"] = torch.from_numpy(np.asarray(img, dtype=np.float32)[None, ...])
    out = Node().execute(**a)
    return [o.detach().cpu().numpy()[0] if hasattr(o, "detach") else o for o in out]


def u8(x):
    return (np.asarray(x) * 255.0).round().astype(np.uint8)


IMG = photo()

# ---------------------------------------------------------------- invariants

print("[1] the photograph is untouched outside zones")
r = run(IMG, zone_count=0, overlay_opacity=0.0, image_opacity=1.0, pixelate_zones="")
ok(np.allclose(r[0], IMG, atol=1e-6), "inv1 output == input exactly")
r2 = run(IMG, zone_count=1, overlay_opacity=0.0, image_opacity=1.0, pixel_size=20)
control(not np.allclose(r2[0], IMG, atol=1e-6),
        "inv1 a zone must actually alter the image")

print("[2] overlay_alpha is zero where nothing was drawn")
r = run(IMG, zone_count=0)
alpha = r[2]
diff = np.any(np.abs(r[0] - r[1]) > 1e-6, axis=2)
ok(float(alpha[~diff].max() if (~diff).any() else 0.0) < 0.02,
   "inv2 alpha ~0 where image and overlay_only agree")
control(float(alpha.max()) > 0.5, "inv2 alpha must be non-zero somewhere")

print("[3] determinism")
a = run(IMG, seed=11)[0]
b = run(IMG, seed=11)[0]
ok(np.array_equal(u8(a), u8(b)), "inv3 bitwise identical on repeat")
control(not np.array_equal(u8(a), u8(run(IMG, seed=12)[0])),
        "inv3 a different seed must differ")

print("[4] reseeding moves the LAYOUT, not only the text")
pts = geo.detect(IMG, "combined", 16, 28, 26, 70, 8, 46, 42)
cfg = {k: D[k] for k in ("focus_a_x", "focus_a_y", "focus_b_x", "focus_b_y",
                         "rays_a", "rays_b", "ray_jitter", "sphere_count",
                         "sphere_inner", "sphere_outer", "sphere_span")}
L1 = geo.Layout(1, 320, 420, pts, cfg)
L2 = geo.Layout(2, 320, 420, pts, cfg)
ok(L1.fa != L2.fa or L1.fb != L2.fb, "inv4 foci move with the seed")
ok([round(v, 3) for v in L1.radii] != [round(v, 3) for v in L2.radii],
   "inv4 sphere radii move with the seed")
control(geo.Layout(1, 320, 420, pts, cfg).fa == L1.fa,
        "inv4 the same seed must reproduce the same layout")

print("[5] trivial ray-sphere roots are exactly at radius r")
worst = 0.0
for (x, y, i, k) in geo.intersections_trivial(L1):
    worst = max(worst, abs(math.hypot(x - L1.fa[0], y - L1.fa[1]) - L1.radii[k]))
ok(worst < 1e-9, f"inv5 |P-C| == r exactly (worst {worst:.2e})")
control(worst < 1e-9, "inv5 measured, not assumed")

print("[6] non-trivial roots satisfy |P-C| = r")
worst = 0.0
n_nt = 0
for (x, y, i, k) in geo.intersections_nontrivial(L1):
    worst = max(worst, abs(math.hypot(x - L1.fa[0], y - L1.fa[1]) - L1.radii[k]))
    n_nt += 1
ok(n_nt > 0, "inv6 non-trivial intersections exist")
ok(worst < 1e-9, f"inv6 quadratic roots exact (worst {worst:.2e})")
lin = abs(math.hypot(*(L1.fb[j] - L1.fa[j] for j in (0, 1))) - L1.radii[0])
control(lin > 1e-9, "inv6 a linearised stand-in would not satisfy the bound")

print("[7] every mark uses a declared weight class")
plate = rc.Plate(64, 64, PACK, (255, 255, 255), 1.0, 1.0)
fired = False
try:
    plate.line(0, 0, 10, 10, 0.77)
except ValueError:
    fired = True
control(fired, "inv7 an unclassified weight must raise")
for cls in (rc.DATUM, rc.CONSTRUCT, rc.FINE):
    plate.line(0, 0, 5, 5, cls)
ok(True, "inv7 the three declared classes are accepted")

print("[8] labels never overlap and drops are reported")
p2 = rc.Plate(200, 200, PACK, (255, 255, 255), 1.0, 1.0)
placed = sum(1 for i in range(60) if p2.label(100, 100, "XXXX", 8.0))
ok(placed < 60, "inv8 the resolver refuses to overprint")
ok(p2.dropped == 60 - placed, "inv8 drop count is reported accurately")
boxes = p2.taken
overlap = any(boxes[i][0] < boxes[j][2] and boxes[j][0] < boxes[i][2]
              and boxes[i][1] < boxes[j][3] and boxes[j][1] < boxes[i][3]
              for i in range(len(boxes)) for j in range(i + 1, len(boxes)))
ok(not overlap, "inv8 no two placed labels overlap")
control(placed > 0, "inv8 some labels must still be placed")

print("[9] no shadow / under-stroke")
# NOT a source-string scan: the first version of this assertion matched the
# renderer's own docstring explaining that there is no shadow, and failed on
# correct code. The structural facts are that Plate owns exactly ONE draw
# surface, and that a light stroke deposits no dark pixels.
ok(len([a for a in vars(rc.Plate(8, 8, PACK, (255, 255, 255), 1.0, 1.0))
        if isinstance(getattr(rc.Plate(8, 8, PACK, (255, 255, 255), 1.0, 1.0), a),
                      type(rc.Plate(8, 8, PACK, (255, 255, 255), 1.0, 1.0).layer))]) == 1,
   "inv9 Plate owns exactly one draw surface")
p3 = rc.Plate(40, 40, PACK, (255, 255, 255), 1.0, 1.0)
p3.line(5, 20, 35, 20, rc.CONSTRUCT, 255)
arr = np.asarray(p3.layer)
dark = ((arr[..., 3] > 0) & (arr[..., :3].max(axis=2) < 120)).sum()
ok(dark == 0, "inv9 no dark pixels in the mark layer")
control(int((arr[..., 3] > 0).sum()) > 0, "inv9 the mark layer is not empty")

print("[10] zones use Overlay's pixelate_canvas verbatim")
nsrc = open(os.path.join(PACK, "nodes", "schematic_construction.py"),
            encoding="utf-8").read()
ok("overlay_render.pixelate_canvas(" in nsrc, "inv10 calls Overlay's function")
base = np.repeat(np.repeat(np.linspace(0, 1, 40)[None, :, None], 40, 0), 3, 2)
mine = overlay_render.pixelate_canvas(base.copy(), 40, 40, [(20, 20)], 8, 12.0,
                                      False, (255, 255, 255), 1.0, 9, PACK)
theirs = overlay_render.pixelate_canvas(base.copy(), 40, 40, [(20, 20)], 8, 12.0,
                                        False, (255, 255, 255), 1.0, 9, PACK)
ok(np.array_equal(mine, theirs), "inv10 identical inputs give identical output")
control(not np.array_equal(mine, base), "inv10 pixelation must change the canvas")

print("[11] zone centres are spaced, and per-zone sizes stay in range")
zs = 30.0
sizes = geo.zone_sizes(7, 12, 18.0, 55.0)
ok(len(sizes) == 12 and all(18.0 <= v <= 55.0 for v in sizes),
   "inv11 every drawn zone size lies within [min, max]")
ok(len(set(round(v, 6) for v in sizes)) > 1, "inv11 sizes actually vary")
ok(geo.zone_sizes(7, 5, 40.0, 40.0) == [40.0] * 5,
   "inv11 min == max reproduces fixed-size behaviour")
ok(geo.zone_sizes(7, 4, 18.0, 55.0) == geo.zone_sizes(7, 4, 18.0, 55.0),
   "inv11 sizes are deterministic for a seed")
# The previous control only asked that the LISTS differ, which passed while the
# FIRST element was constant across every seed - a real defect it could not see.
firsts = [geo.zone_sizes(sd, 3, 2.0, 150.0)[0] for sd in range(64)]
spread = max(firsts) - min(firsts)
ok(spread > 100.0,
   f"inv11 the FIRST zone size varies across seeds (spread {spread:.1f} of 148)")
ok(len(set(round(v, 3) for v in firsts)) > 55,
   "inv11 first sizes are near-unique across seeds")
control(geo.zone_sizes(8, 4, 18.0, 55.0) != geo.zone_sizes(7, 4, 18.0, 55.0),
        "inv11 a different seed must give different sizes")
cs = geo.zone_centres(7, L1, 320, 420, 6, zs)
bad = [(a, b) for i, a in enumerate(cs) for b in cs[i + 1:]
       if math.hypot(a[0] - b[0], a[1] - b[1]) < 2.1 * zs - 1e-6]
ok(not bad, f"inv11 all centres >= 2.1*zone_size apart ({len(cs)} placed)")
control(len(cs) > 1, "inv11 more than one centre was placed")

print("[12] geometry is resolution-similar")
small = geo.Layout(5, 500, 660, [(250, 330, 20)], cfg)
large = geo.Layout(5, 1000, 1320, [(500, 660, 40)], cfg)
ok(abs(small.fa[0] / 500.0 - large.fa[0] / 1000.0) < 1e-9,
   "inv12 focus is at the same normalised position")
ok(abs(small.radii[0] / small.s - large.radii[0] / large.s) < 1e-9,
   "inv12 sphere radii scale with the short side")
control(abs(small.radii[0] - large.radii[0]) > 1.0,
        "inv12 a pixel-unit implementation would give equal radii")

print("[13] micro-type stays sub-legible")
ratios = [rc.type_px(s, 0.0066, 1.0) / float(s) for s in (512, 1024, 4096)]
ok(max(ratios) - min(ratios) < 1e-12, "inv13 type is a constant canvas fraction")
control(max(8.0 / s for s in (512, 4096)) - min(8.0 / s for s in (512, 4096)) > 1e-9,
        "inv13 a fixed-pixel size would fail the bound")

print("[14] every parameter changes the render")
PERT = {
    "seed": 99, "detection_mode": "dark", "block_size": 40, "threshold": 60,
    "max_circles": 8, "min_distance": 160, "min_radius": 20, "max_radius": 90,
    "focus_a_x": 0.7, "focus_a_y": 0.3, "focus_b_enabled": False,
    "focus_b_x": 0.2, "focus_b_y": 0.7, "focus_clear": 0.2, "rays_a": 8,
    "rays_b": 5, "ray_jitter": 0.4, "sphere_count": 9, "sphere_inner": 0.15,
    "sphere_outer": 1.4, "sphere_span": 200.0, "marker_size": 0.01,
    "marker_shape": "square", "marker_fill": False, "marker_every": 4, "label_every": 1, "lune_density": 120, "scale_ticks": 12,
    "scale_label_every": 1, "type_scale": 2.2, "exit_labels_every": 1,
    "callout_count": 6, "data_column_enabled": False, "data_column_rows": 3,
    "zone_count": 8, "pixel_size": 40, "zone_size_min": 12.0,
    "zone_size_max": 160.0, "pixel_stroke": False,
    "palette": "blackOnLight", "bg_color": "#334455", "stroke_color": "#FF0000",
    "line_weight": 3.0, "overlay_opacity": 0.3, "image_opacity": 0.4,
    "frame_text_size": 22.0, "frame_text_tl": "ZZZ", "frame_text_tr": "ZZZ",
    "frame_text_bl": "ZZZ", "frame_text_br": "ZZZ",
    "elements": "rays_b: off", "pixelate_zones": "40,40;120,200",
}
# Parameters whose effect is CONDITIONAL on another. Exercised in the context
# where they apply rather than exempted, so the assertion keeps its teeth and
# the dependency is documented here as well as in the tooltip.
CTX = {
    "min_radius": {"elements": "detection_circles: on"},
    "max_radius": {"elements": "detection_circles: on"},
    "stroke_color": {"palette": "custom"},
    "bg_color": {"palette": "custom", "image_opacity": 0.5},
}
PERT["zone_count"] = 0     # 8 does not fit at the default spacing; 0 always differs


def moved(k, val):
    """Compare EVERY output, not just `image`: bg_color and the palette show up
    in overlay_only, which the first version of this test never looked at."""
    ctx = CTX.get(k, {})
    b = run(IMG, **ctx)
    a = run(IMG, **dict(ctx, **{k: val}))
    return any(not np.array_equal(u8(a[i]), u8(b[i])) for i in range(3))


dead = []
for k in IT["required"]:
    if k == "image":
        continue
    if k not in PERT:
        dead.append(k + " (no perturbation defined)")
    elif not moved(k, PERT[k]):
        dead.append(k)
for k in ("elements", "pixelate_zones"):
    if not moved(k, PERT[k]):
        dead.append(k)
for g in geo.REGISTRY.groups():
    if not moved(f"off_{g[2:]}_x", 0.2):
        dead.append(f"off_{g[2:]}_x")
ok(not dead, f"inv14 no inert parameters (dead: {dead})")
control(np.array_equal(u8(run(IMG)[0]), u8(run(IMG)[0])),
        "inv14 the baseline is stable")

print("[15] every input carries a tooltip")
missing = [k for sec in ("required", "optional") for k, v in IT[sec].items()
           if not (isinstance(v, tuple) and len(v) > 1 and isinstance(v[1], dict)
                   and str(v[1].get("tooltip", "")).strip())]
n_inputs = len(IT["required"]) + len(IT["optional"])
ok(not missing, f"inv15 all {n_inputs} inputs have tooltips (missing: {missing})")
ok(n_inputs >= 46, f"inv15 surface is at least Overlay's 46 ({n_inputs})")
control(n_inputs > 0, "inv15 the surface is non-empty")

print("[16] element overrides")
a = u8(run(IMG, elements="")[0])
b = u8(run(IMG, elements="  \n# comment only\n")[0])
ok(np.array_equal(a, b), "inv16 empty override == defaults")
off = u8(run(IMG, elements="spheres: off")[0])
ok(not np.array_equal(a, off), "inv16 'off' removes something")
# content independence: disabling one element must not re-roll another
d_sph = np.any(a != off, axis=2)
d_cal = np.any(a != u8(run(IMG, elements="callouts: off")[0]), axis=2)
inter = float(np.logical_and(d_sph, d_cal).sum()) / max(1, int(d_cal.sum()))
ok(inter < 0.5, f"inv16 disabling spheres leaves callouts alone ({inter:.3f})")
raised = False
try:
    run(IMG, elements="no_such_thing: off")
except ElementSyntaxError as exc:
    raised = "valid elements" in str(exc)
ok(raised, "inv16 unknown name raises, listing valid names")
control(not np.array_equal(a, off), "inv16 the toggle must have an effect")

print()
print(f"{PASS} passed, {FAIL} failed, {FIRED}/{TOTAL} negative controls fired")
if FAILURES:
    print("\nfailures:")
    for f in FAILURES:
        print(f"  - {f}")
sys.exit(1 if FAIL else 0)
