# Schematic Voronoi — derivation spec (node v0.1)

Status: WRITTEN 2026-08-09 under Jeremie's end-to-end authorization ("go for it all
the way. next stop we preferably have a working tool") — derivation calls marked
here are RATIFIED OR OVERRULED AT THE WORKING-TOOL GATE rather than pre-signed.
Family D of `comfyui-brain/data-art-plan.md` §2. The LOOK passed a two-round spike
(`_eyeball/SPIKE_voronoi_r2.png`) against the reference and the shipped tool before
this document was written — the spike-first process correction from the A/I failure.

Licence (§7.4 binding): the generic technique is density-driven Voronoi tessellation
(Voronoi 1908, Fortune 1987; scipy.spatial's Qhull is BSD-3, a ComfyUI CORE dep).
The named reference ("Turbidity", signed student coursework) is the single
highest-temptation image in the set: the TECHNIQUE transfers (density-following
seeds, wireframe cells, intensity graded by cell size), its COMPOSITION never does.
Our composition is the photograph's own field — which is also what makes the node
"about the photo". No credit lines (§7.6.8).

Substrate conventions inherited verbatim from `schematic-derivation.md` §12-13
(numpy + PIL + the two-overlay 2× supersample pattern, MINSTD only, batch
same-seed, determinism, no wall-clock) plus scipy.spatial via LAZY import inside
execute (the Field Distance pattern; the pack loads and the sibling node runs even
if scipy ever vanished). Canvas px, origin top-left. "op" = overlay_opacity.
S = min(W, H).

---

## 0. Identity — the mesh IS the image

Established by the spike, the load-bearing composition decision: the photo is NOT
the ground the mesh floats on; the photo is faint (image_opacity default 0.2) and
its content EMERGES from mesh density — tiny bright-stroked cells where the field
is hot, large faint cells where it is cold. This is the family-D translation of
the pack's identity ("a machine analyzed this image"): the tessellation is a
statement about the photo by construction, and it shares zero rendering vocabulary
with Schematic Overlay (the A/I composition lesson, applied).

## 1. Pipeline

cover_fit (frozen §1, same fitted array feeds field and render) → density field
(§2) → seed sampling (§3) → Voronoi geometry (§4) → render (§5).

## 2. Density field

`density_source` dropdown:
- "brightness": field = luma01 (Rec.601, the v0.1 §2 luma on the fitted array, /255)
- "darkness": field = 1 − luma01
- "detail": field = local std of luma01 in a square window of side
  `k = odd(max(3, round(0.006 · S)))` — S-relative so presets keep their look
  across resolutions (internal constant; a new node's internals are free to be
  S-relative, as v0.1's own chain-tick constants already are).

Normalize: field /= max(field) (guard max < 1e-9 → all-zero field, §3 degenerate).
Density d(x,y) = field^`density_gamma`. Tooltip guidance (spike V7 finding): pair
"brightness" with dark-ground palettes and "darkness" with blackOnLight, so tonal
values render right-side-up; "detail" outlines structure on either ground.

## 3. Seeds — deterministic MINSTD rejection sampling

- One MinstdLCG(seed) stream, draws consumed in strict (x, y, u) triples:
  x = draw()·W, y = draw()·H, accept iff u = draw() < d(⌊y⌋,⌊x⌋) (indices clamped).
- Stop at `cells` accepted OR at the fixed draw budget `80 · cells` triples
  (deterministic termination on any input; the budget is part of the spec, not an
  implementation detail — batch frames must consume identical draw counts on
  identical inputs, which same-seed determinism gives).
- **cells (COUNT) is the user parameter, not a relative cell scale** — resolving
  §7.3-D's named fork: with count fixed, the same photo at 2× resolution yields the
  same STRUCTURE scaled ×2 (density map scales, count constant ⇒ spacing scales
  with canvas) — the resolution behavior a preset wants; and count is the honest
  knob (it is literally the number of tessellation sites).
- Degenerate: fewer than 4 accepted (black image + brightness, or an all-zero
  field) → SKIP the tessellation, render the frame without a mesh, print one
  `[Schematic]` note (the QhullError-guard pattern, §7.3-E item 10).
- Batch: same seed every frame (frozen rule); the stream never conditions on batch
  position or size.

## 4. Geometry — scipy.spatial.Voronoi on mirror-closed seeds

- Reflect the n accepted seeds across each canvas edge (x→−x; x→2W−x; y→−y;
  y→2H−y): 5n points total. Every REAL cell (the first n) becomes finite and its
  boundary tracks the canvas edge — no infinite-region special cases anywhere
  downstream. Named invariant: zero real cells with an infinite region.
- Cell area by the shoelace formula on each real region polygon.
- Ridges kept for drawing: every Voronoi ridge with finite vertices where at least
  one adjacent point is REAL (index < n).
- Per-ridge intensity from adjacent real cell areas (the Turbidity grading,
  monochrome translation): t(i) = (log A_i − log A_p2) / (log A_p98 − log A_p2)
  clamped to [0,1], where A_p2/A_p98 are the 2nd/98th percentiles of real cell
  areas (log scale because areas span decades). Ridge alpha
  `a = 1.0 − 0.85 · min(t of adjacent real cells)`, clamped [0.15, 1.0] — the
  smaller (denser) neighbor wins. Constants (0.85 floor-slope, percentile ends)
  internal, exhibit-tunable; the STRUCTURE (log-area → alpha, dense = bright) is
  the signed mechanic.

## 5. Render — sibling orchestration, pack frame

`schematic/render_voronoi.py`, OWN `render_frame_voronoi` (never shared), composing
draw_common (SS, _rgba, _load_font, _resolve_overlay) + engine as-is:

1. bg fill (palette). 2. cover-fit image at `image_opacity` (default **0.2** — §0;
   skip in overlay_only). Then overlay-A at 2×:
   a. optional cell fill: if `cell_fill` > 0, each real cell polygon filled at
      alpha `cell_fill · field(seed) · op` (plate-tone under the wirework).
   b. ridges: stroke color, alpha `a · op` (§4), width 2 at 2× (=1 px final),
      3 at 2× when a > 0.75 (the densest strokes carry slightly more ink —
      spike-derived, exhibit-tunable).
   c. optional seed dots: if `seed_dots`, r = 1.5 at 2×, alpha 0.78·op.
   d. frame text: v0.1 §7.7 verbatim (4 corners, inset 40, frame_text_size).
3. Premultiplied resolve + composite (draw_common `_resolve_overlay`).
4. Texture screen-blend at texture_opacity (socket frame 0 stretched, else
   `engine.generate_grain(seed+1, w, h)`); local 4-line screen+lerp sibling copy.
5. overlay-B EMPTY; `overlay_alpha` = overlay-A alpha. Three outputs, v0.1
   contract verbatim; early-out mirror of v0.1 §13 (with the canvas-size guard).

## 6. Node contract + widgets

ID `SchematicVoronoi`, display "Schematic Voronoi", CATEGORY `AKURATE/Schematic`,
FUNCTION "execute"; required `image` + widgets; optional `texture` (IMAGE);
RETURN ("IMAGE","IMAGE","MASK") ("image","overlay_only","overlay_alpha"); BHWC
float32; per-frame field+seeds+cells, same seed every frame; texture frame 0;
torch only in `nodes/schematic_voronoi.py`; `schematic/voronoi.py` (pure geometry,
numpy + lazy scipy) import-safe standalone.

Widgets (tooltips on all; defaults exhibit-tunable, structure signed):
density_source ["brightness","darkness","detail"] brightness; cells 12000
(200..30000/100); density_gamma 2.5 (0.5..4.0/0.1); seed 42 (0..0xFFFFFFFF);
mesh_weight 1.0 (0.5..3.0/0.1 — multiplies both ridge widths before the 2× scale);
cell_fill 0.0 (0.0..0.5/0.05); seed_dots False; overlay_opacity 1.0;
image_opacity 0.2 (0..1/0.05); texture_opacity 0.5; frame_text_size 12;
frame_text_tl "Design & Strategy"; frame_text_tr "AKURATE"; frame_text_bl "";
frame_text_br ""; palette whiteOnDark (+bg_color/stroke_color); size_preset
"match input" (v0.1 list). NO detection widgets — this node's vocabulary is the
tessellation, not marks (§0).

## 7. Module surface (the blind-teeth contract)

`schematic/voronoi.py`:
- `build_density(fitted01, source, gamma) -> (H,W) float64 in [0,1]`
- `sample_seeds(density, cells, seed, w, h) -> ndarray (n,2) float64` (n ≤ cells)
- `build_cells(seeds, w, h) -> dict(regions=list[n] of vertex arrays,
  areas=ndarray (n,), ridges=list[((x1,y1),(x2,y2), i, j)] with i/j = adjacent
  REAL seed indices or -1)`  — pure; lazy-imports scipy inside the call.
- `ridge_alpha(areas, ridge_real_indices) -> per-ridge float in [0.15, 1.0]`
  (vectorizable; exact map per §4).
`schematic/render_voronoi.py`: `render_frame_voronoi(fitted01, w, h, palette,
params, elements, texture01, include_image, font_dir) -> (uint8 (h,w,3),
float32 (h,w))`; elements keys "regions","areas","ridges","alphas","seeds",
"field_at_seeds". `nodes/schematic_voronoi.py`: class SchematicVoronoi.

## 8. Verification items (blind-teeth table of contents)

Named scalar per assertion; negative controls move THAT scalar. Suite zero: the
shipped node's byteident 18/18 + test_schematic.py stay green.

1. Field polarity: half-black/half-white image, source=brightness, any gamma ⇒
   ZERO seeds on the black half (density is exactly 0 there — exact scalar, not a
   bound); source=darkness flips the side (the control).
2. Uniform field ⇒ target count reached; quadrant seed counts within ±25% of
   n/4 (loose uniformity, stated tolerance).
3. Determinism: bitwise-identical seeds and pixels on same-args rerun; different
   seed differs; batch frame alone == same frame in batch (all 3 outputs).
4. Degenerate: all-black + brightness ⇒ no crash, mesh-free render, note printed;
   all-flat-gray ⇒ runs (uniform density).
5. Closure: zero real cells with infinite regions (mirror construction, §4).
6. Alpha grading: crafted density step (dense left / sparse right) ⇒ mean ridge
   alpha strictly higher on the dense side; control: inverted map lowers it.
7. cells honors count: accepted n == cells on a bright-enough field; n < cells
   with the budget exhausted on a nearly-black field (both scalars checked).
8. Lazy scipy: importing `schematic.voronoi` must NOT import scipy
   (`"scipy" not in sys.modules` after a fresh-interpreter import of the module);
   scipy appears only after `build_cells` runs.
9. Render probes: three-output contract, bg-only exact palette bg, overlay_alpha
   all-zero when op=0 ∧ texture_opacity=0, early-out passthrough bitwise,
   screen-blend never darkens.
10. Perf printed (soft): full default render 1200×1600 < 20 s embedded (12k cells
    is heavier than the siblings; measured at build, budget amended if needed).

## 9. Cuts — named

Extrusion / pseudo-3D (`extrude` folds in LATER, §7.2, when Isoplot's painter
machinery exists — one projection substrate built once); per-cell hue (single
stroke color is the palette system); cell-boundary smoothing/rounding; Lloyd
relaxation (fights the density-following that IS the point); animation-aware
seeding (Phase-3 rule binds if ever animated); any use of the reference image
beyond the generic technique.
