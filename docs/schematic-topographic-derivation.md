# Schematic Topographic — derivation and frozen spec (v2, 2026-09-12)

Fourth public node of ComfyUI-Schematic. Built in one plow-ahead session on
Jeremie's brief: *"topographic design overlay containing all elements mostly found
on a topographic [map], with our schematic as reference just other elements:
contour lines, index contour, hachures, spot heights and benchmarks, arrows, grid
systems... fully controllable by density, scale, weight. text annotations."*

## 0. v2 amendment (same day) — the terrain is NOT the photo

v1 traced the contours from the photo's luma and was rejected at Jeremie's eye
(*"wrong direction... it's an overlay that has nothing to do with the image...
next iteration shouldn't use this image as it's harder to evaluate"*). Three
references came with the correction; their design qualities, named:

| Reference | Quality this node now carries |
|---|---|
| 2026 calendar poster (contours over cream) | the terrain is a procedural landform; **text runs along the index contours glyph by glyph**, the line broken under it; a **dot grid** region |
| clean brown-on-grey contour map | smooth, evenly spaced contours; every 5th heavier; **inline labels with a unit suffix** sitting in a gap |
| USGS quadrangle | dense **flowing ridge/valley** contours (anisotropic terrain); x spot elevations and triangle stations |

Changes over v1: (1) `height_source` gains **`procedural` (the default)** — a
seeded fractional-Brownian gradient-noise landform (§3 step 1, new); the photo
sources stay as optional modes. (2) New widgets `terrain_scale`,
`terrain_octaves`, `terrain_roughness`, `terrain_stretch`, `terrain_angle`,
`label_unit`, `contour_text`, `contour_text_on`, `contour_text_size`,
`contour_text_gap`; grid mode `dots`. (3) `image_opacity` default 0.85 (Overlay's
value: the photo is a ground under an independent sheet). (4) Exhibits are
judged on the overlay_only output; the photo appears in one final-check panel.
The seed now drives the terrain too, so the v1 "seed-independent with
benchmarks off" tooth holds for the photo sources only.

**Pixelate zones (Jeremie, same day: "i would also leave the pixelation
functionality by input we had in the schematic to alter the underlying
image").** Overlay's `pixelate_zones` string, optional `pixelate_mask` socket,
`pixel_size`, `zone_size`, `pixel_stroke` — parsed exactly as Overlay parses
them (`engine.parse_zone_string` + `engine.mask_zone_centers` on a
nearest-resampled mask) and rendered by `render.pixelate_canvas` VERBATIM (the
Construction precedent) on the bg + photo canvas, after the image composite and
before the band raster, in both passes. It alters the photo inside each zone
only; the outline and centre label are Overlay's. `pixel_size` 1 disables it
bitwise. Inherited quirk, stated: once any zone exists, `pixelate_canvas`
round-trips the whole canvas through 8 bits, so pixels OUTSIDE the zones may
move by at most 1 LSB (measured 1.0000 on the hero); inside they move by
mosaic amounts (151 LSB max on the hero).

**Procedural field** (`topographic.procedural_field`): coordinates rotated by
`terrain_angle` and compressed along the flow axis by `terrain_stretch`;
`octaves` layers of 2-D gradient noise (lattice gradients from a 32-bit integer
hash of the lattice coords and the element seed `"terrain"`, quintic fade),
first-octave feature size `terrain_scale` px, lacunarity 2, gain
`terrain_roughness`, a per-octave lattice offset; then the common smoothing and
min/max normalisation. Hand checks: bitwise on rerun; differs by seed;
independent of the photo (a flat 0.5 and a flat 0.9 image give the same field);
stretch 4 along x lowers the x/y gradient-energy ratio by at least 4x.

**Contour text** (`topographic.text_runs`): for each polyline carrying text
(index lines, or all lines), the text plus two spaces is repeated along the
arc, one glyph per advance width, each glyph rotated to the local tangent
(never flipped — the calendar's text reads both ways around loops), a gap of
`contour_text_gap` px between repetitions, a lead of `1.2 × size` free at both
ends; the line is not drawn under a run. Numeric labels are suppressed on lines
carrying text.

## 1. Identity and named reference

**The photo is the terrain.** Its smoothed luma is the digital elevation model; the
node draws a complete topographic map SHEET over it. This is the node's own
composition (the standing law from data-art-plan.md: every new family brings its
own composition, never Overlay's marks-over-points vocabulary). A map sheet is
organised by its contour field and its grid, not by detected points.

**Reference quality, named (patterns.md 2026-08-10 rule):** the public-domain
symbol conventions of the USGS 7.5-minute quadrangle sheet and the Swisstopo /
IGN sheet marginalia:

| Element | Convention this node follows |
|---|---|
| Contour lines | intermediate contours light, evenly spaced by a fixed vertical interval |
| Index contour | every 5th contour, heavier, labelled inline with its elevation; the line is broken under the label |
| Hachures | short ticks on the DOWNHILL side of a contour; on standard sheets only depression contours carry them |
| Spot heights | peaks marked with a small triangle + elevation value; pits with a small circle + value |
| Benchmarks | survey marks: an x cross + "BM" + elevation |
| Slope arrows | terrain-analysis aspect arrows on a regular grid, pointing downhill, length proportional to slope |
| Grid | UTM-style grid as full lines, edge ticks, or interior crosses; coordinates labelled at the sheet edge in metres |
| Hypsometric bands | stepped plate tone between contours, brighter with elevation |
| Marginalia | north arrow, graphic scale bar in a 1-2-5 round length, "CONTOUR INTERVAL n M" note |
| Annotations | user call-outs: leader line with a 45° elbow and a horizontal tail, text at the tail |

No code from any map renderer is used; the conventions are public and the
implementation is from first principles. Palette, grain texture, frame text, size
presets, and the three outputs are inherited from the siblings unchanged.

## 2. Substrate (inherited, binding)

numpy + Pillow only in `schematic/topographic.py` and
`schematic/render_topographic.py` (no scipy, no torch); torch lives only in the
node file. Two 2x-supersampled RGBA overlays with premultiplied resolve
(`draw_common`), MINSTD RNG (`engine.MinstdLCG`), one seed for every frame of a
batch, no wall-clock, deterministic. Font: `fonts/SpaceGrotesk-Regular.ttf` via
`draw_common._load_font`.

## 3. Height field

Let `fitted` = `engine.cover_fit(frame, w, h)` (the same array feeds the render).

1. **Source field** `s(x,y)` from `height_source`:
   - `procedural` (default, v2): `procedural_field(w, h, seed, terrain_scale,
     terrain_octaves, terrain_roughness, terrain_stretch, terrain_angle)` — see §0
   - `brightness`: luma = 0.299 R + 0.587 G + 0.114 B, computed in float64 and
     ROUNDED THROUGH float32 (the MASK socket dtype), so a `height_mask` equal
     to the photo's luma reproduces the brightness path bitwise
   - `darkness`: 1 − luma
   - `detail`: local population std of luma in a 15×15 box (`voronoi._local_std_box`
     semantics reproduced locally, k = 15), then divided by its own max (0 if max = 0)
   - if `height_mask` is wired (prepped exactly as `nodes/schematic_voronoi._prep_density_mask`:
     trailing size-1 dims squeezed, frame 0 with a console note, nan→0, ≥8×8
     plausibility guard, clip [0,1]; cover-fitted to the canvas when sizes differ
     exactly as `_canvas_density_mask`) it REPLACES the source; `height_source`
     is inert.
2. **Smoothing**: separable Gaussian blur, σ = `smoothing` px (0 skips the op
   structurally), kernel radius `ceil(3σ)`, edge-replicate padding, kernel
   normalised to sum 1, float64.
3. **Normalise**: `H = (S − min S) / (max S − min S)`; if `max S − min S ≤ 1e-9`
   the field is FLAT: `H = 0` everywhere, no contours exist, every
   field-derived element draws nothing, one console note
   `[Schematic] Schematic Topographic: flat height field - rendering this frame
   contour-free`. Grid, marginalia, annotations and frame text still draw.
4. **Elevation**: `z(x,y) = elevation_min + H · (elevation_max − elevation_min)`
   metres. If `elevation_max ≤ elevation_min` the node swaps them (console note).
5. **Work grid**: the contour tracer runs on `H` sampled at stride
   `s = contour_resolution` px: `F[j,i] = H[j·s, i·s]` for
   `i < ceil(w/s)`, `j < ceil(h/s)`; work vertex `(i,j)` maps to canvas
   `(i·s, j·s)`. Gradient on the work grid by central differences
   (one-sided at borders), in H-units per work cell; `∇z = ∇F ·
   (elevation_max − elevation_min) / s` per canvas px is what the slope
   arrows scale by.

## 4. Contour extraction (own marching squares)

**Levels.** `Δ = contour_interval` metres. The level set is
`z_k = k·Δ` for every integer `k` with `elevation_min < z_k < elevation_max`,
i.e. levels are aligned to the interval in absolute metres (a 0…1000 m range at
Δ=20 gives 49 levels: 20, 40, …, 980), so index contours land on round numbers.
In H-units `t_k = (z_k − elevation_min) / (elevation_max − elevation_min)`.
A level is an **index** level iff `index_every > 0` and `k mod index_every == 0`.
Cap: at most 400 levels (widget ranges make more impossible: min interval 1 m
over a max range 10000 m… the cap is a guard, and it is printed when hit).

**Marching squares** per level on `F` with linear interpolation on edges,
comparing `F ≥ t_k` (vertices exactly equal to the level count as inside).
Saddle cases 5 and 10 are disambiguated by the cell centre mean (`≥ t_k` joins
the two inside corners). Edge crossing position on an edge from value `a` to `b`:
`u = (t − a) / (b − a)` (b ≠ a by construction of a crossing). Every crossing
gets a global edge id, so the two endpoints of a segment are edge ids and
segments chain by shared edge ids into polylines:

- each edge id is shared by at most two segments; chaining walks from every
  edge with exactly one segment (open polylines, they start and end on the
  canvas border) then from any remaining unvisited segment (closed loops).
- a polyline is `closed` iff its first and last edge ids coincide.
- vertices are emitted in canvas pixel coordinates (work coords × `s`).
- polylines with fewer than 3 vertices are dropped.

The pure API returns, per level, `{"z": z_k, "index": bool, "lines": [(pts (n,2) float64, closed bool), …]}`.

**Hand-checkable consequences (the teeth use these):**
- `F = j` (a vertical ramp, `H` rising downwards): every contour is a single open
  horizontal polyline at `y = t_k · (h−1)` exactly (linear interpolation is
  exact on a linear field), with `x` spanning the work grid.
- a cone `H = 1 − r/R`: each contour is one closed loop whose vertex radii all lie
  within one work cell (`s` px) of `R·(1 − t_k)`.
- level count equals the integer count from the level rule above.

## 5. Elements — geometry (`schematic/topographic.py`)

Every element that consumes randomness draws from its OWN MINSTD stream seeded by
`element_seed(seed, element_id)` (FNV-1a over the id, the `micrographics.element_seed`
rule, reproduced locally so the module stays standalone). Element ids:
`"benchmarks"` is the only consumer in v1. Consequence: with `benchmarks` off and
`texture_opacity` 0 the render is bitwise independent of `seed`.

### 5a. Contour lines
Regular contour alpha `0.55·op`, width `contour_weight` px (× SS, `max(1, round)`).
Index contour alpha `1.0·op`, width `contour_weight · index_weight` px.
Polylines are drawn with `ImageDraw.line(joint="curve")`.

### 5b. Index labels
Only on index levels when `contour_labels` is on, only on polylines whose
arc length ≥ `label_spacing`. Labels sit at arc positions
`sᵢ = label_spacing/2 + i·label_spacing` for `i ≥ 0` while `sᵢ + label_spacing/2 ≤ length`.
Text = `str(int(round(z_k)))`, font size `label_size` px, rotated to the local
tangent of the polyline at `sᵢ` (tangent from the vertices ±`label_size` px of
arc either side); if the tangent angle points into the left half-plane
(`cos θ < 0`) it is flipped by 180° so text never reads upside down. The index
polyline is NOT drawn for the arc within `text_width/2 + 0.6·label_size` of `sᵢ`
(the cartographic label gap): the line is split into sub-polylines around each
gap before drawing.

### 5c. Hachures
`hachures ∈ {off, depressions, all}`. Ticks are placed along every contour (both
kinds) at arc positions `hachure_spacing/2 + i·hachure_spacing`. At each position
the downhill unit vector `d = −∇F / |∇F|` is read at the nearest work vertex
(skip if `|∇F| < 1e-6`). The tick runs from the point to `point + hachure_length · d`,
alpha `0.55·op`, width `contour_weight` px. **A hachure never points uphill:**
a tick whose end lands on a work vertex with `F` greater than its start's is
DROPPED (it crossed a ridge within its own length; measured 0.2 % of ticks on
the hero at length 6, stride 2), so the downhill promise holds for every drawn
tick by construction. The pure API takes the work grid as `F=`; callers that
omit it get the undropped geometry.
In `depressions` mode a polyline carries hachures iff it is closed AND
`mean over vertices of (d · n_in) > 0`, where `n_in` is the inward unit normal
(from the polygon's signed area orientation): downhill points inward → the
contour encloses lower ground → a depression. Open polylines never carry
depression hachures. In `all` mode every polyline carries them.

### 5d. Slope arrows
Regular canvas grid at pitch `arrow_spacing` px, offsets `arrow_spacing/2`
(no jitter — the terrain-analysis convention is a regular grid). At each site
read `g = ∇F` at the nearest work vertex; slope magnitude reference
`g_ref` = 75th percentile of `|∇F|` over the work grid (a frame statistic; 0 →
no arrows). Arrow length `L = arrow_length · (0.35 + 0.65 · min(1, |g| / g_ref))`
(a floor of 35 % keeps every drawn arrow readable — photo slope distributions
are heavy-tailed and a pure proportional law measured a 5.7 px median at
`arrow_length` 22 on the hero; the top quarter of sites draw full length);
skipped when `|g| < 0.05·g_ref`. Shaft from `p` to `p + L·d` (downhill), head =
two lines from the tip back at ±28°, length `0.35·L`. Alpha `0.7·op`, width
`contour_weight` px.
Hand check: on the ramp `F = j` every arrow points straight up (−y) with length
`arrow_length` (all slopes equal so `|g| = g_ref`).

### 5e. Spot heights
Local extrema of `F` on the work grid: a vertex is a **peak** iff it is
strictly greater than all 8 neighbours, a **pit** iff strictly smaller (border
vertices excluded). Candidates sorted by prominence proxy = `F` descending for
peaks, `1 − F` descending for pits; the two lists are merged by that key and
accepted greedily while at least `spot_min_distance` px from every accepted
spot, until `spot_count` are accepted. Peak symbol: filled upright triangle of
side `symbol_size` px centred on the point; pit symbol: open circle of radius
`symbol_size/2` px. Label `str(int(round(z)))` at `label_size` px, anchored
"lm" at `x + symbol_size·0.8 + 2`, alpha `op`.
Hand check: on a two-Gaussian field the two peaks are found within one work
cell of the true centres and their labels are the true elevations (rounded).

### 5f. Benchmarks
`benchmark_count` sites from stream `"benchmarks"`: candidate
`(x, y) = (m + u·(w − 2m), m + v·(h − 2m))`, margin `m = 0.08·min(w,h)`, drawn
`u, v` in that order; accepted iff ≥ `spot_min_distance` px from every accepted
spot height and benchmark; at most `50·benchmark_count` draws. Symbol: x cross
of half-size `symbol_size/2` px, width `contour_weight`; label `"BM " +
str(int(round(z)))` at `label_size` px, anchored "lm" at `x + symbol_size·0.8 + 2`.
Benchmark positions depend on the seed and on the accepted spot heights (the
rejection rule), and on nothing else — toggling grid, arrows, hachures, labels or
band fill never moves them.

### 5f'. Campsites (added 2026-10-12, v0.7.0)
`campsite_count` sites from stream `"campsites"` (a sibling stream, so toggling
campsites never moves the benchmarks). Candidate as §5f with margin
`m = 0.08·min(w,h) + campsite_size/2`, `u, v` in that order; accepted iff the
slope `|∇F|` at the nearest work vertex is ≤ the MEDIAN slope of the field
(campsites are pitched on the flatter half of the terrain) AND the site is
≥ `max(spot_min_distance, campsite_size)` px from every spot height, benchmark
and accepted campsite; at most `80·campsite_count` draws. Flat field → none.
Symbol: the CanvasCamp badge, `campsite_size` px across: a disc in the stroke
colour carrying the brand mark (transcribed from
`Canvascamp001_LogoSymbol_ReDesign_Vector_Model_v001.svg`; disc centre (512,
526.1), r 104.6 in its 1024 frame, measured off the lockup) knocked out in the
background colour, inside a background-colour halo of `max(1, contour_weight)`
px. The ® is dropped (sub-pixel at map scale). Each badge is built opaque on its
own tile and composited at `overlay_opacity`. The inputs are OPTIONAL with
python defaults (shipped-node rule): absent or `campsites=False` is bitwise the
v0.6 output.

### 5g. Grid
`grid ∈ {off, lines, ticks, crosses}`, pitch `grid_spacing` px, lines at
`x = k·grid_spacing` and `y = k·grid_spacing` for `k ≥ 1` inside the canvas.
`lines`: full-length lines, alpha `0.35·op`. `ticks`: 12 px ticks inward from all
four edges at each grid coordinate, alpha `0.7·op`. `crosses`: a `+` of half-arm
`6` px at every interior intersection, alpha `0.7·op`, plus the edge ticks.
Width `grid_weight` px. `grid_labels` on: coordinate text at `label_size` px,
alpha `op`: easting `int(round(x · map_scale))` along the BOTTOM edge (anchor
"md", 6 px above the edge), northing `int(round((h − y) · map_scale))` along the
RIGHT edge rotated 90° counter-clockwise (reads bottom-to-top), 6 px inside;
labels within 80 px of any canvas corner are skipped (frame-text corner clearance).
`map_scale` is metres per canvas pixel.

### 5h. Hypsometric band fill
`band_fill` in [0, 0.5]; 0 skips. Raster at 1x on the canvas BEFORE overlay-A:
`alpha_b(x,y) = band_fill · q(x,y) · op`, `q = floor(H · n_lev) / n_lev` with
`n_lev` = number of contour levels + 1 (stepped bands whose steps are the
contour lines; `q = 0` below the first level, `q → 1` at the top). Composited
as `canvas·(1−α) + stroke·α`; contributes to `overlay_alpha` (union, like the
Voronoi cell fill). Flat field → nothing.

### 5i. Marginalia (`marginalia_scale` multiplies every size below)
- **North arrow** (`north_arrow`): at the top-right, centre `x = w − 40 −
  10·ms`, `y = 40 + frame_text_size·1.6 + 28·ms` (clears the top-right frame
  text); shaft length `44·ms` px pointing up, filled head (triangle, half-width
  `6·ms`, height `14·ms`), an "N" at `label_size·1.2` px centred `10·ms` above the tip.
- **Scale bar** (`scale_bar`): bottom-left, left end at `x = 40`, baseline `y =
  h − 40 − frame_text_size·1.6 − 14·ms`; target length `0.18·w` px converted to
  metres by `map_scale` and rounded DOWN to the nearest value in the 1-2-5 series
  (`{1,2,5}·10^n`); bar = 4 alternating segments (filled / outline) of height
  `6·ms`, end ticks, labels "0" and `f"{metres} m"` (`"{km} km"` when ≥ 1000)
  at `label_size` px, anchored above the bar ends.
- **Interval note** (`interval_note`): `f"CONTOUR INTERVAL {Δ:g} M"` at
  `label_size` px, anchor "ld", at `x = 40`, `y` = scale-bar baseline + `6·ms`
  + `label_size·1.4` (below the bar; below the frame-text line only if the
  scale bar is off, in which case it sits at `y = h − 40 − frame_text_size·1.6`).
All marginalia alpha `op`, widths `contour_weight` px.

### 5j. Annotations
`annotations` string: entries separated by `;`, each `x,y:TEXT` (canvas px;
whitespace tolerated; malformed entries skipped with one console note). For
each: dot radius `2.5` px at `(x, y)`; leader from `(x, y)` at 45° up-right for
`28·ms` px then horizontal right for `36·ms` px; text at `label_size·1.2` px,
anchor "lm", 6 px right of the tail end. If `x > 0.7·w` the leader mirrors to
up-left and the text anchors "rm". Alpha `op`, width `contour_weight`.

### 5k. Frame text
Sibling copy of the four-corner frame text (inset 40, `frame_text_size`).

## 6. Render order (`render_topographic.render_frame_topographic`)

1. bg fill
2. cover-fitted image at `image_opacity` (skipped in overlay_only)
3. hypsometric band raster (§5h)
4. overlay-A at 2x: grid → regular contours → hachures → index contours (with
   label gaps) → index labels → slope arrows → spot heights → benchmarks →
   campsites → annotations → grid labels → marginalia → frame text; premultiplied resolve;
   composite. The band raster and the resolved layer depend only on the
   elements, never on the photo, so the node draws them ONCE per frame and the
   overlay_only pass reuses them (a `cache` dict argument; bitwise-equal to
   redrawing, half the render time)
5. texture screen blend (`texture` or seeded grain, `seed + 1`)
6. overlay-B slot: EMPTY. An empty 2x layer resolves to α = 0 and
   `canvas·(1−0) + 0` is bitwise the canvas, so the resolve is elided (the
   siblings pay it; here nothing draws after the texture).
7. `overlay_alpha = 1 − (1 − α_band)(1 − α_A)`

Early-out (sibling mirror): `image_opacity ≥ 1`, `overlay_opacity ≤ 0`, no
texture, canvas == input size, AND `band_fill` irrelevant (it scales by op) →
passthrough frame, bg-only overlay, zero alpha, bitwise.

## 7. Module surface (frozen)

`schematic/topographic.py`:
- `element_seed(seed, element_id) -> int`
- `build_height(fitted01, source, smoothing, mask01=None) -> (H float64 (h,w), flat bool)`
- `work_grid(H, stride) -> F (hj, wi) float64`
- `gradient(F) -> (gx, gy)` central differences, one-sided borders
- `contour_levels(elev_min, elev_max, interval, index_every) -> list of (z, t, is_index)`
- `trace_level(F, t, stride) -> list of (pts (n,2), closed)`
- `build_contours(F, levels, stride) -> list of dicts (§4)`
- `polyline_length(pts)`, `point_at_arc(pts, s) -> (x, y, tx, ty)`
- `depression_flag(pts, closed, gx, gy, stride) -> bool`
- `hachure_ticks(contours, gx, gy, stride, mode, spacing, length, F=None) -> list of ((x0,y0),(x1,y1))`
- `slope_arrows(gx, gy, stride, w, h, spacing, length) -> list of ((x0,y0),(x1,y1), L)`
- `spot_heights(F, stride, count, min_dist, elev_min, elev_max) -> list of (x, y, z, kind)`
- `benchmarks(seed, w, h, count, min_dist, spots, F, stride, elev_min, elev_max) -> list of (x, y, z)`
- `campsites(seed, w, h, count, min_dist, size, others, gx, gy, stride) -> list of (x, y)`
- `grid_lines(w, h, spacing) -> (xs, ys)`
- `parse_annotations(s) -> list of (x, y, text)`
- `scale_bar_metres(target_px, map_scale) -> (metres, px)`
- `band_quantised(H, n_levels) -> q (h,w)`

`schematic/render_topographic.py`: `render_frame_topographic(fitted01, w, h,
palette, params, elements, texture01, include_image, font_dir)`.

## 8. Node

`SchematicTopographic`, display "Schematic Topographic", category
`AKURATE/Schematic`, outputs `image, overlay_only, overlay_alpha`. Required
widgets in the order listed in §5 (control-surface parity with Overlay's 46:
this node has 48 widgets after `image`, plus 2 optional sockets). Optional: `texture` IMAGE, `height_mask`
MASK. Console line on execute:
`[Schematic] Schematic Topographic: source=…, interval=…, levels=N, palette=…, size_preset=…, seed=…`.

Defaults: brightness, smoothing 10, resolution 2, elevation 0–1000 m, interval 40,
contour_weight 1.0, index_every 5, index_weight 2.2, labels on, label_size 9,
label_spacing 320, hachures depressions, spacing 10, length 6, arrows off,
spacing 90, length 22, spots on, count 12, min_distance 120, symbol_size 8,
benchmarks on, count 4, grid ticks, spacing 200, weight 1.0, labels on,
map_scale 1.0, band_fill 0.0, north_arrow on, scale_bar on, interval_note on,
marginalia_scale 1.0, annotations "", seed 42, overlay_opacity 1.0,
image_opacity 0.7, texture_opacity 0.5, frame text as the siblings, whiteOnDark.

## 9. Teeth (`tools/test_topographic.py`, implementation-blind against this doc)

1. Contour exactness: ramp levels at exact `y`; cone radii within `s`; level
   count per the rule; index flags every `index_every`-th `k`.
2. Determinism: bitwise rerun; batch of 2 == 2× single; with benchmarks off and
   texture 0, seed 1 vs 2 bitwise; with benchmarks on they differ.
3. Element locality: benchmark positions (pure API) identical across grid /
   hachure / arrow / label toggles; identical spots → identical benchmarks.
4. Hachure direction: every tick end has `F` ≤ `F` at its start (nearest work
   vertex), on a cone AND on the hero photo; `depressions` mode on a hill cone
   → zero ticks, on a pit cone → > 0 ticks; `all` → > 0 on both.
5. Arrows: ramp → every arrow direction (0, −1) within 1e-9, length ==
   `arrow_length`; flat → none.
6. Spots: two-Gaussian field → both peaks within `s` px, labels == true rounded
   elevations; count and min-distance honoured on the photo.
7. Grid: `lines` mode, `op` 1, `grid_weight` 1 → every column `k·spacing` has
   ink in `overlay_alpha` along its full height; no ink at `k·spacing + spacing/2`
   in a flat field with everything else off.
8. Render probes: overlay_only with everything off (texture included) is
   bg-exact; op 0 → alpha
   all-zero; early-out bitwise passthrough; band_fill raises mean alpha on a
   ramp and is monotone in `band_fill`.
9. height_mask: a mask equal to the luma of the photo produces the same
   contours as `brightness` (float32 socket rounding, patterns.md 2026-08-18);
   a (H,W,1)-trailing-dim mask is accepted; a 4×4 mask is treated absent with the note.
10. Flat field: no crash, contour-free note, grid + marginalia still draw.
11. Perf print (soft): default 1600×893 render time.

Sibling byteident (`tools/test_byteident.py`) must stay green: no sibling file
is touched by this build except `nodes/__init__.py` (registration), `README.md`,
`pyproject.toml`.

## 10. Calls made solo under plow-ahead (Jeremie's eyeball gate decides)

- `image_opacity` default 0.7 (the sheet reads over the photo; Voronoi's 0.2 is
  for a node where the mesh IS the image, this one is an overlay like Overlay's 0.85).
- Single font (Space Grotesk) for every label including coordinates; a mono
  face for coordinates was considered and left for his call.
- Coordinate labels on the bottom and right edges (keeps the top-left frame
  text clear), 80 px corner clearance.
- Alpha ladder: regular contour 0.55, index 1.0, grid lines 0.35, ticks/crosses
  0.7, arrows 0.7, everything labelled 1.0 (× op).
- Marginalia fixed to their sheet-standard corners (north arrow top-right, scale
  bar + interval note bottom-left) with one `marginalia_scale` multiplier rather
  than per-item position widgets.
