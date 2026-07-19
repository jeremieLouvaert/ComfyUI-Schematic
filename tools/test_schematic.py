"""
Offline validation for ComfyUI-Schematic — "teeth before trust".
Run with plain python + numpy + Pillow (no ComfyUI needed). No pytest;
prints PASS/FAIL per check and exits nonzero if any check fails.

Verification items (per plan "Verification" section, mapped 1:1):
  1. LCG parity (+ negative control: wrong multiplier).
  2. Block oracle: mean/std vs direct numpy; combined-score formula
     (+ negative control: flipped formula).
  3. Placement invariants: min-distance, count cap, radius range, determinism.
  4. Threshold monotonicity; flat-gray contrast-mode -> 0 circles.
  5. Chain tangency + intersection oracle (+ negative control: broken sign).
  6. Render probes: bg pixels, screen-blend never darkens, chain-above-texture,
     pixelate distinct-color bound.
  7. Cover-fit offsets (known 2:1 image into 1:1 canvas).
  8. Perf timings at 1200x1600 and 3840x2160 (printed).
"""

import math
import os
import sys
import time

PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK_ROOT)

import numpy as np
from PIL import Image

from schematic import engine, render

PASS = True


def check(name, cond, detail=""):
    global PASS
    status = "PASS" if cond else "FAIL"
    if not cond:
        PASS = False
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


DEFAULT_PARAMS = {
    "shape": "circle", "circle_stroke": 1.0, "label_size": 8,
    "overlay_opacity": 1.0, "line_weight": 0.8,
    "crosshair_enabled": True, "crosshair_frame_size": 60,
    "crosshair_dash": 8, "crosshair_stroke": 1.0,
    "crosshair_star_size": 40, "crosshair_star_points": 4,
    "pixel_size": 16, "zone_size": 100, "pixel_stroke": True,
    "image_opacity": 0.85, "texture_opacity": 0.5,
    "frame_text_size": 12, "frame_text_tl": "Design & Strategy",
    "frame_text_tr": "AKURATE", "frame_text_bl": "", "frame_text_br": "",
    "chain_intersection_size": 5.0, "seed": 42,
}


def empty_elements():
    return {"circles": [], "connections": [], "zones": [], "chain": [], "intersections": []}


# ===========================================================================
print("[1] LCG parity (Park-Miller MINSTD, seed=42) + negative control")


def reference_minstd(seed, n):
    """Independent reference implementation (not calling engine.MinstdLCG)."""
    s = abs(int(seed))
    state = s if s != 0 else 1
    out = []
    for _ in range(n):
        state = (state * 16807) % 2147483647
        out.append((state - 1) / 2147483646)
    return out


ref5 = reference_minstd(42, 5)
lcg = engine.MinstdLCG(42)
ours5 = [lcg.draw() for _ in range(5)]
check("first 5 draws match independent reference exactly", ours5 == ref5,
      f"ours={ours5[:2]}... ref={ref5[:2]}...")

lcg0 = engine.MinstdLCG(0)
check("seed 0 -> state 1 (first draw = reference_minstd(1,1)[0])",
      abs(lcg0.draw() - reference_minstd(1, 1)[0]) < 1e-15)


def reference_minstd_bad(seed, n, mult):
    s = abs(int(seed))
    state = s if s != 0 else 1
    out = []
    for _ in range(n):
        state = (state * mult) % 2147483647
        out.append((state - 1) / 2147483646)
    return out


bad5 = reference_minstd_bad(42, 5, 16808)
neg_fired = bad5 != ref5
check("NEGATIVE CONTROL: multiplier 16808 diverges from real sequence (teeth present)", neg_fired)

# lcg_draws_array vectorized parity vs sequential class (used for grain generation)
seq = reference_minstd(7, 2000)
vec = engine.lcg_draws_array(7, 2000)
max_diff = float(np.max(np.abs(np.array(seq) - vec)))
check("vectorized lcg_draws_array matches sequential draws (n=2000)", max_diff < 1e-9,
      f"max|d|={max_diff:.3e}")


# ===========================================================================
print("\n[2] Block oracle (mean/std) + combined-score formula + negative control")

rng = np.random.default_rng(123)
H, W = 64, 96
grad_x = np.linspace(0, 1, W, dtype=np.float32)
synth = np.tile(grad_x, (H, 1))[:, :, None] * np.array([1.0, 0.8, 0.6], dtype=np.float32)
synth = np.clip(synth + rng.normal(0, 0.05, size=(H, W, 3)).astype(np.float32), 0, 1)

blocks = engine.analyze_blocks(synth, 16)

# independent oracle: direct numpy on the same array, no shared code path
bs = 16
rows, cols = H // bs, W // bs
oracle_bright = np.zeros(rows * cols)
oracle_contrast = np.zeros(rows * cols)
for r in range(rows):
    for c in range(cols):
        patch = synth[r * bs:(r + 1) * bs, c * bs:(c + 1) * bs, :].astype(np.float64) * 255.0
        L = 0.299 * patch[..., 0] + 0.587 * patch[..., 1] + 0.114 * patch[..., 2]
        idx = r * cols + c
        oracle_bright[idx] = L.mean()
        oracle_contrast[idx] = math.sqrt(max(0.0, (L ** 2).mean() - L.mean() ** 2))

diff_b = float(np.max(np.abs(blocks["brightness"] - oracle_bright)))
diff_c = float(np.max(np.abs(blocks["contrast"] - oracle_contrast)))
check("block brightness matches direct-numpy oracle (< 1e-6)", diff_b < 1e-6, f"max|d|={diff_b:.2e}")
check("block contrast matches direct-numpy oracle (< 1e-6)", diff_c < 1e-6, f"max|d|={diff_c:.2e}")

score_combined = engine.score_blocks(blocks, "combined")
hand_combined = oracle_contrast * (1.0 + np.abs(oracle_bright - 128.0) / 128.0)
diff_score = float(np.max(np.abs(score_combined - hand_combined)))
check("combined score matches hand formula (< 1e-9)", diff_score < 1e-9, f"max|d|={diff_score:.2e}")

flipped = oracle_contrast * (1.0 - np.abs(oracle_bright - 128.0) / 128.0)
diff_flip = float(np.max(np.abs(score_combined - flipped)))
check("NEGATIVE CONTROL: flipped combined formula disagrees (teeth present)", diff_flip > 1e-6,
      f"max|d|={diff_flip:.2e}")


# ===========================================================================
print("\n[3] Placement invariants (min-distance, count cap, radius range, determinism)")

H2, W2 = 400, 400
rng2 = np.random.default_rng(7)
noisy = rng2.random((H2, W2, 3)).astype(np.float32)
blocks2 = engine.analyze_blocks(noisy, 16)
score2 = engine.score_blocks(blocks2, "combined")
circles = engine.place_circles(blocks2, score2, threshold=10, max_circles=60,
                                min_distance=40, min_radius=4, max_radius=24, seed=42)

ok_dist = True
for i in range(len(circles)):
    for j in range(i + 1, len(circles)):
        dx = circles[i][0] - circles[j][0]
        dy = circles[i][1] - circles[j][1]
        if dx * dx + dy * dy < 40 * 40 - 1e-6:
            ok_dist = False
check("pairwise distances >= min_distance", ok_dist, f"n={len(circles)}")
check("count <= max_circles", len(circles) <= 60, f"n={len(circles)}")

ok_radius = all(0.5 * 4 - 1e-6 <= r <= 1.5 * 24 + 1e-6 for (_, _, r, _) in circles)
check("radii within [0.5*min_radius, 1.5*max_radius]", ok_radius)

circles_a = engine.place_circles(blocks2, score2, 10, 60, 40, 4, 24, seed=42)
circles_b = engine.place_circles(blocks2, score2, 10, 60, 40, 4, 24, seed=42)
same_seed_identical = circles_a == circles_b
check("determinism: same seed -> identical set", same_seed_identical)

circles_c = engine.place_circles(blocks2, score2, 10, 60, 40, 4, 24, seed=999)
different_seed_differs = circles_a != circles_c
check("determinism: different seed -> differs", different_seed_differs)


# ===========================================================================
print("\n[4] Threshold monotonicity + flat-gray zero circles")

# Graded-contrast image: block (r,c)'s noise amplitude scales with grid position,
# so scores actually SPAN a range (i.i.d. uniform noise gives near-identical
# contrast in every block -> threshold has nothing to bite on).
rows4, cols4, bs4 = 25, 25, 16
img4 = np.full((rows4 * bs4, cols4 * bs4, 3), 0.5, dtype=np.float32)
rng4 = np.random.default_rng(11)
for r in range(rows4):
    for c in range(cols4):
        amp = (r * cols4 + c) / (rows4 * cols4 - 1)
        noise = (rng4.random((bs4, bs4, 3)).astype(np.float32) - 0.5) * amp
        img4[r * bs4:(r + 1) * bs4, c * bs4:(c + 1) * bs4, :] = np.clip(0.5 + noise, 0, 1)

blocks4 = engine.analyze_blocks(img4, bs4)
score4 = engine.score_blocks(blocks4, "combined")

counts = []
for t in (0, 20, 40, 60, 80):
    c = engine.place_circles(blocks4, score4, t, 1000, 1, 4, 24, seed=42)
    counts.append(len(c))
mono = all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))
check("higher threshold -> count non-increasing", mono, f"counts={counts}")

flat = np.full((256, 256, 3), 0.5, dtype=np.float32)
flat_blocks = engine.analyze_blocks(flat, 16)
flat_score = engine.score_blocks(flat_blocks, "contrast")
flat_circles = engine.place_circles(flat_blocks, flat_score, 30, 80, 40, 4, 24, seed=42)
check("flat gray, contrast mode, default threshold -> 0 circles", len(flat_circles) == 0,
      f"n={len(flat_circles)}")


# ===========================================================================
print("\n[5] Chain tangency + intersection oracle + negative control")

chain = engine.compute_chain(1000, 1000, 11, 45, 250, 0.79)
check("chain length == chain_count", len(chain) == 11, f"len={len(chain)}")

n_plus = (11 - 1) // 2
plus_side = [chain[0]] + chain[1:1 + n_plus]
minus_side = [chain[0]] + chain[1 + n_plus:]

ok_tangent = True
for side in (plus_side, minus_side):
    for i in range(len(side) - 1):
        c1, c2 = side[i], side[i + 1]
        d = math.hypot(c2[0] - c1[0], c2[1] - c1[1])
        if abs(d - c1[2]) > 1e-9:
            ok_tangent = False
check("consecutive same-side steps are tangent (dist == prev radius, 1e-9)", ok_tangent)

intersections = engine.compute_chain_intersections(chain)
ok_intersect = True
idx_pair = 0
for i in range(len(chain) - 1):
    c1, c2 = chain[i], chain[i + 1]
    pts = engine.two_circle_intersections(c1, c2)
    for (px, py) in pts:
        d1 = math.hypot(px - c1[0], py - c1[1])
        d2 = math.hypot(px - c2[0], py - c2[1])
        if abs(d1 - c1[2]) > 1e-6 or abs(d2 - c2[2]) > 1e-6:
            ok_intersect = False
check("intersection points satisfy |p-c1|=r1 and |p-c2|=r2 (1e-6, all consecutive array pairs)",
      ok_intersect, f"n_points={len(intersections)}")


def broken_two_circle_intersections(c1, c2):
    """NEGATIVE CONTROL: perpendicular-offset sign broken (both points use +offset)."""
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    dx, dy = x2 - x1, y2 - y1
    d = math.hypot(dx, dy)
    if d == 0 or d > r1 + r2 or d < abs(r1 - r2):
        return []
    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h2 = r1 * r1 - a * a
    if h2 < 0:
        return []
    h = math.sqrt(h2)
    midx, midy = x1 + a * dx / d, y1 + a * dy / d
    # BUG: both points identical (sign not flipped) -> only one distinct point, and
    # perpendicular direction uses wrong axis pairing (dx,dx instead of dy,-dx)
    p1 = (midx + h * dx / d, midy + h * dy / d)
    p2 = (midx + h * dx / d, midy + h * dy / d)
    return [p1, p2]


neg_ok = False
for i in range(len(chain) - 1):
    c1, c2 = chain[i], chain[i + 1]
    bad_pts = broken_two_circle_intersections(c1, c2)
    for (px, py) in bad_pts:
        d1 = math.hypot(px - c1[0], py - c1[1])
        d2 = math.hypot(px - c2[0], py - c2[1])
        if abs(d1 - c1[2]) > 1e-6 or abs(d2 - c2[2]) > 1e-6:
            neg_ok = True
check("NEGATIVE CONTROL: broken-sign intersection FAILS the distance invariant (teeth present)", neg_ok)


# ===========================================================================
print("\n[6] Render probes")

# 6a. bg-only pixels == palette.bg (everything disabled, no image, no texture)
w6, h6 = 64, 64
bg_rgb, stroke_rgb = engine.resolve_palette("whiteOnDark", "#0a0a0a", "#d8d8d8")
p6 = dict(DEFAULT_PARAMS)
p6.update({"crosshair_enabled": False, "overlay_opacity": 0.0, "texture_opacity": 0.0,
           "pixel_size": 1, "image_opacity": 0.0})
fitted6 = np.zeros((h6, w6, 3), dtype=np.float32)
out6, alpha6 = render.render_frame(fitted6, w6, h6, (bg_rgb, stroke_rgb), p6, empty_elements(),
                                    None, include_image=False, font_dir=PACK_ROOT)
bg_match = np.all(out6[..., 0] == bg_rgb[0]) and np.all(out6[..., 1] == bg_rgb[1]) and np.all(out6[..., 2] == bg_rgb[2])
check("bg-only render == palette.bg exactly", bg_match, f"bg_rgb={bg_rgb}, sample={out6[0,0].tolist()}")
check("bg-only overlay_alpha is all zero", float(np.max(alpha6)) == 0.0)

# 6b. screen blend never darkens
rng6 = np.random.default_rng(3)
canvas6 = rng6.random((32, 32, 3)).astype(np.float64)
tex6 = rng6.random((32, 32, 3)).astype(np.float64)
for op6 in (0.0, 0.3, 0.7, 1.0):
    blended = render.apply_texture_screen(canvas6, tex6, op6)
    never_darkens = np.all(blended >= canvas6 - 1e-9)
    check(f"screen blend never darkens canvas (opacity={op6})", never_darkens,
          f"min(blended-canvas)={float(np.min(blended - canvas6)):.2e}")

# 6c. chain pixels present ABOVE texture (probe on black texture, texture_opacity=1)
w6c, h6c = 200, 200
bg_black, stroke_white = (0, 0, 0), (255, 255, 255)
p6c = dict(DEFAULT_PARAMS)
p6c.update({"crosshair_enabled": False, "overlay_opacity": 1.0, "texture_opacity": 1.0, "pixel_size": 1})
chain6c = [(100.0, 100.0, 30.0)]  # single big chain circle at canvas center
elements6c = empty_elements()
elements6c["chain"] = chain6c
black_texture = np.zeros((h6c, w6c, 3), dtype=np.float32)
fitted6c = np.zeros((h6c, w6c, 3), dtype=np.float32)
out6c, _ = render.render_frame(fitted6c, w6c, h6c, (bg_black, stroke_white), p6c, elements6c,
                                black_texture, include_image=False, font_dir=PACK_ROOT)
# sample the top of the chain circle's outline (100, 100-30=70)
probe_y = 70
probe_row = out6c[probe_y, 85:116, :]
chain_visible = bool(np.any(probe_row.max(axis=1) > 10))  # not pure black despite black texture screen=1
check("chain pixels visible ABOVE a full-opacity black texture (drawn after step 8)", chain_visible,
      f"row max={int(probe_row.max())}")

# 6d. pixelate zone: <=1 distinct color per tile
w6d, h6d = 128, 128
rng6d = np.random.default_rng(9)
noisy6d = rng6d.random((h6d, w6d, 3)).astype(np.float64)
zones6d = [(64.0, 64.0)]
pixelated = render.pixelate_canvas(noisy6d, w6d, h6d, zones6d, pixel_size=16, zone_size=48,
                                    pixel_stroke=False, stroke_rgb=stroke_white, op=1.0,
                                    label_size=8, font_dir=PACK_ROOT)
tile = pixelated[16:32, 16:32, :]
distinct = len(set(map(tuple, (tile.reshape(-1, 3) * 255).round().astype(int))))
check("pixelate tile has <= 1 distinct color (solid mosaic)", distinct <= 1, f"distinct={distinct}")


# ===========================================================================
print("\n[7] Cover-fit offsets (known 2:1 image into 1:1 canvas)")

img_w7, img_h7 = 200, 100  # 2:1 image
img7 = np.zeros((img_h7, img_w7, 3), dtype=np.float32)
img7[:, :img_w7 // 2, 0] = 1.0   # left half red
img7[:, img_w7 // 2:, 2] = 1.0   # right half blue

fitted7 = engine.cover_fit(img7, 100, 100)  # 1:1 canvas
# expected crop: ia=2>ca=1 -> srcH=100, srcW=100*1=100, srcX=(200-100)/2=50, srcY=0
# crop is columns [50,150) of the 200-wide source -> spans the boundary at x=100
# so left part of crop (50..100) is red, right part (100..150) is blue
left_mean = fitted7[:, :45, 0].mean()   # comfortably inside the red half
right_mean = fitted7[:, 55:, 2].mean()  # comfortably inside the blue half
check("cover-fit crop offset: left portion is red source", left_mean > 0.9, f"mean_r={left_mean:.3f}")
check("cover-fit crop offset: right portion is blue source", right_mean > 0.9, f"mean_b={right_mean:.3f}")

# tall image (1:2) into 1:1 canvas -> crops top/bottom, centered on width match
img_w7b, img_h7b = 100, 200
img7b = np.zeros((img_h7b, img_w7b, 3), dtype=np.float32)
img7b[:img_h7b // 2, :, 1] = 1.0   # top half green
img7b[img_h7b // 2:, :, 0] = 1.0   # bottom half red
fitted7b = engine.cover_fit(img7b, 100, 100)
# ia=0.5<ca=1 -> srcW=100, srcH=100, srcY=(200-100)/2=50 -> crop rows [50,150)
top_mean = fitted7b[:45, :, 1].mean()
bot_mean = fitted7b[55:, :, 0].mean()
check("cover-fit crop offset: top portion is green source", top_mean > 0.9, f"mean_g={top_mean:.3f}")
check("cover-fit crop offset: bottom portion is red source", bot_mean > 0.9, f"mean_r={bot_mean:.3f}")


# ===========================================================================
print("\n[perf] timings (1200x1600 and 3840x2160, default-ish params)")


def run_full_pipeline(w, h, seed_img=1):
    rng_p = np.random.default_rng(seed_img)
    frame = rng_p.random((h, w, 3)).astype(np.float32)
    fitted = engine.cover_fit(frame, w, h)
    blocks = engine.analyze_blocks(fitted, 16)
    score = engine.score_blocks(blocks, "combined")
    circles = engine.place_circles(blocks, score, 30, 80, 40, 4, 24, seed=42)
    connections = engine.compute_connections(circles, 150)
    chain = engine.compute_chain(w, h, 11, 45, 250, 0.79)
    intersections = engine.compute_chain_intersections(chain)
    elements = {"circles": circles, "connections": connections, "zones": [],
                "chain": chain, "intersections": intersections}
    params = dict(DEFAULT_PARAMS)
    out, alpha = render.render_frame(fitted, w, h, (bg_rgb, stroke_rgb), params, elements,
                                      None, include_image=True, font_dir=PACK_ROOT)
    return out, alpha


t0 = time.time()
run_full_pipeline(1200, 1600)
t_1200 = time.time() - t0
print(f"  1200x1600 full pipeline (detection + render, default params, procedural grain): {t_1200:.3f} s")

t0 = time.time()
run_full_pipeline(3840, 2160)
t_4k = time.time() - t0
print(f"  3840x2160 full pipeline: {t_4k:.3f} s")

check("1200x1600 completes in a sane time (<15s, soft budget)", t_1200 < 15.0, f"{t_1200:.3f}s")
check("3840x2160 completes in a sane time (<30s, soft budget)", t_4k < 30.0, f"{t_4k:.3f}s")


# ===========================================================================
print()
if PASS:
    print("ALL CHECKS PASSED")
    sys.exit(0)
else:
    print("SOME CHECKS FAILED")
    sys.exit(1)
