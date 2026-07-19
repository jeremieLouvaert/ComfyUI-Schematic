# Schematic Overlay — frozen derivation spec (v0.1)

Status: FROZEN 2026-07-19. Build must implement this with zero deviations.
Source: clean-room behavioral extraction of the Circles mode of Yordan Stoyanov's
Brand Assets Generator (https://brand-generator.stoyanov.works/). No code was copied;
this document is the sole build reference.

All coordinates are canvas pixels, origin top-left. All colors sRGB 8-bit unless noted.
"op" = overlay_opacity. "stroke color" = the active palette's stroke hex.

## 1. Canvas and image fit

- Canvas size comes from `size_preset`:
  - `match input` (our default): canvas = input image WxH, no crop.
  - `Portrait 3:4 (1200x1600)`, `Square (1080x1080)`, `Landscape 16:9 (1920x1080)`,
    `Instagram Story (1080x1920)`, `Poster (1400x2000)`.
- Image placement AND analysis both use **cover fit** (crop-to-fill, centered):
  let ia = imgW/imgH, ca = W/H.
  If ia > ca: srcH = imgH, srcW = srcH*ca, srcX = (imgW-srcW)/2, srcY = 0.
  Else:       srcW = imgW, srcH = srcW/ca, srcX = 0, srcY = (imgH-srcH)/2.
  The source rect is resampled to WxH (bilinear). The SAME fitted array feeds
  analysis and rendering so blocks align with visible pixels.

## 2. Block analysis

- Luma per pixel, 0-255 domain: `L = 0.299 R + 0.587 G + 0.114 B`.
- Grid: `cols = floor(W / block_size)`, `rows = floor(H / block_size)`.
  Remainder pixels at right/bottom edges belong to NO block (no partial blocks).
- Per block (all pixels in its block_size × block_size window):
  - `brightness` = mean(L)
  - `contrast` = sqrt(max(0, mean(L^2) - mean(L)^2))   (population stddev)
  - center: `x = col*bs + bs/2`, `y = row*bs + bs/2`

## 3. Seeded RNG (placement parity requirement)

Park-Miller MINSTD LCG, exact:
```
state = abs(seed) or 1          # seed 0 -> state 1
draw(): state = (state * 16807) % 2147483647
        return (state - 1) / 2147483646     # in [0, 1)
```
Python ints are exact; no float drift possible. This RNG is used ONLY for the
per-circle radius jitter (one draw per ACCEPTED circle, in acceptance order)
and for the procedural-grain fallback (separate instance, seed+1).

## 4. Scoring and placement

- Score per block by `detection_mode`:
  - `contrast`: contrast
  - `bright`:   brightness
  - `dark`:     255 - brightness
  - `combined` (default): `contrast * (1 + abs(brightness - 128) / 128)`
- Normalize: `norm = score / max(max_all_scores, 1)` → [0,1].
- Keep blocks with `norm >= threshold / 100`.
- Sort kept blocks DESCENDING by norm. (Stable sort; ties keep grid order row-major.)
- Greedy accept in sorted order:
  - reject if squared distance to ANY accepted center < min_distance².
  - stop when max_circles accepted.
  - on accept: `jitter = 0.5 + rng()` (∈ [0.5, 1.5)),
    `radius = (min_radius + (max_radius - min_radius) * norm) * jitter`.
- Output list: {x, y, r, score=norm} in acceptance order.

## 5. Connections

All unique pairs (i<j) of accepted circles with Euclidean distance < connection_dist.
No cap, no nearest-k. If connection_dist == 0, no lines.

## 6. Chain geometry

- Circle 0: center (W/2, H/2), radius chain_base_radius.
- Direction θ = (chain_angle - 90) degrees, dx = cos θ, dy = sin θ.
- Plus side, `floor((count-1)/2)` circles: walk from center;
  `next_center = current_center + (dx, dy) * current_r`; `next_r = current_r * chain_ratio`.
- Minus side, `ceil((count-1)/2)` circles: fresh walk from center with -(dx, dy), same rule.
- Array order = [center, plus side in walk order, minus side in walk order]. KEEP this order.
- Intersections (chain_intersections on): for CONSECUTIVE ARRAY pairs (i, i+1) —
  intentionally includes the non-tangent pair across the plus/minus boundary — compute
  standard two-circle intersection:
  ```
  d = hypot(c2-c1); skip if d > r1+r2 or d < |r1-r2| or d == 0
  a = (r1² - r2² + d²) / (2d);  h² = r1² - a²;  skip if h² < 0
  mid = c1 + a*(c2-c1)/d;  offset = h*( (c2y-c1y)/d, -(c2x-c1x)/d )  # note sign order
  points = mid + offset, mid - offset
  ```
  Point order: [(midx + h*dy/d, midy - h*dx/d), (midx - h*dy/d, midy + h*dx/d)]
  where dx,dy = c2-c1. A global counter n starts at 1 and increments per point drawn.

## 7. Render order (exact, non-negotiable)

1. Fill canvas with palette bg.
2. Cover-fit image composited at image_opacity over bg (skip if image_opacity == 0).
3. Pixelate zones (see §8) — sample the CURRENT canvas (bg+image only).
4. Crosshair (see §9).
5. Connection lines: stroke color, width line_weight, alpha op, round caps.
6. Detection circles, each:
   - outline circle (or axis-aligned square of side 2r centered on x,y if shape=square):
     stroke color, width circle_stroke, alpha `(0.3 + 0.4*score) * op`.
   - filled center dot r=2.5: stroke color, alpha `(0.7 + 0.3*score) * op`.
   - label `"{round(x)},{round(y)}"`: label_size px, left-aligned, vertical middle,
     at `(x + r + label_size*0.4, y)`, stroke color, alpha op.
7. Frame text: 4 corners at 40 px inset, frame_text_size px, stroke color, alpha op.
   TL left/top; TR right/top; BL left/bottom; BR right/bottom. Empty strings skipped.
8. Texture: stretched to full canvas WxH (not tiled), **screen blend**
   (`out = 1-(1-a)(1-b)` per channel, linear on 0-1 sRGB values as-is), at
   texture_opacity (lerp: `result = base + (screen - base) * texture_opacity`).
   If no texture input connected and texture_opacity > 0: procedural fallback —
   mono gaussian noise (mean 0.5, std 0.15, seeded MINSTD instance with seed+1 via
   inverse-CDF or Box-Muller from LCG draws), 1 px box blur, used as the texture image.
9. Chain (chain_enabled and chain_count > 0), drawn ON TOP of texture. Per chain circle:
   - outline: stroke color, width circle_stroke, alpha 0.5*op.
   - crosshair ticks: arm = max(8, r*0.3); horizontal + vertical segments through center;
     alpha 0.3*op, width max(0.5, S*6e-4), dash [S*0.004, S*0.004] where S=min(W,H).
   - filled center dot r = max(1.5, S*0.002), alpha 0.6*op.
   - label `"{round(x)},{round(y)}"` at `(x + arm + label_size*0.4, y)`, alpha op.
   - intersection markers: filled dot radius chain_intersection_size, alpha op;
     label `"{n} → {round(x)} – {round(y)}"` (U+2192 arrow, U+2013 en-dash) at
     `(x + chain_intersection_size + label_size*0.5, y)`, label_size px, alpha op.

## 8. Pixelate zones

- Zone sources (union, in order): (a) `pixelate_zones` STRING widget, format
  `"x,y;x,y;..."` (floats ok, whitespace tolerated, invalid pairs skipped);
  (b) `pixelate_mask` MASK input: binarize at 0.5, 8-connected components,
  each component's centroid (mean of member pixel coords) is a zone center.
  Mask is scaled to canvas size (nearest) before labeling if sizes differ.
- Active only if pixel_size > 1 and at least one zone.
- Zone rect: x0 = max(0, floor(cx - zone_size)), y0 = max(0, floor(cy - zone_size)),
  x1 = min(W, ceil(cx + zone_size)), y1 = min(H, ceil(cy + zone_size)). Skip if empty.
- Mosaic within rect on the CURRENT canvas: for each pixel_size × pixel_size tile
  (grid anchored at rect origin; partial tiles at rect edges use their own pixels),
  fill the tile with the integer-truncated mean RGB of the tile.
- If pixel_stroke: rect outline, stroke color, width 1, alpha 0.4 (NOT scaled by op).
- Zone label `"{round(cx)},{round(cy)}"` centered (h+v) in the rect, label_size px,
  stroke color, alpha op.

## 9. Crosshair

All: stroke color, width crosshair_stroke, alpha op.
- Dashed full-canvas cross: horizontal line y=H/2 across [0,W]; vertical x=W/2 across [0,H];
  dash pattern [crosshair_dash, crosshair_dash].
- Dashed square: side `a = min(W,H) * crosshair_frame_size / 100`, axis-aligned,
  centered at (W/2, H/2), same dash pattern.
- Center asterisk (SOLID, no dash): star_points diametric lines; line i at angle
  `θ_i = i / star_points * π` radians; from `(cx - cosθ*l, cy - sinθ*l)` to
  `(cx + cosθ*l, cy + sinθ*l)` with `l = crosshair_star_size / 2`.
  If crosshair_star_size == 0, skip.

## 10. Palettes

| key | bg | stroke |
|---|---|---|
| whiteOnDark  | #0a0a0a | #d8d8d8 |
| blackOnLight | #f0eeea | #111111 |
| goldOnDark   | #111110 | #c9a84c |
| greenOnDark  | #080a08 | #44cc66 |
| custom       | widget bg_color | widget stroke_color |

Custom hex parsing: accept `#rrggbb` or `rrggbb`; on parse failure fall back to
whiteOnDark values and print a `[Schematic]` warning.

## 11. Defaults (widget defaults = original tool defaults)

detection_mode combined, block_size 16 (8..64 step 4), threshold 30 (0..80),
max_circles 80 (20..400), min_distance 40 (10..200), shape circle,
min_radius 4 (1..40), max_radius 24 (4..120), circle_stroke 1.0 (0.1..8 step 0.1),
seed 42 (0..0xFFFFFFFF), label_size 8 (4..48), overlay_opacity 1.0 (0..1 step 0.05),
connection_dist 150 (0..500 step 10), line_weight 0.8 (0.1..8 step 0.1),
chain_enabled true, chain_count 11 (0..40), chain_angle 45 (0..90),
chain_base_radius 250 (20..1000), chain_ratio 0.79 (0.3..1.2 step 0.01),
chain_intersections true, chain_intersection_size 5.0 (0.5..20 step 0.5),
crosshair_enabled true, crosshair_frame_size 60 (10..100), crosshair_dash 8 (1..40),
crosshair_stroke 1.0 (0.1..8 step 0.1), crosshair_star_size 40 (0..300 step 2),
crosshair_star_points 4 (2..12), pixel_size 16 (1..80 step 1), zone_size 100 (10..500),
pixel_stroke true, image_opacity 0.85 (0..1 step 0.05), texture_opacity 0.5 (0..1 step 0.05),
frame_text_size 12 (6..72), frame_text_tl "Design & Strategy", frame_text_tr "AKURATE",
frame_text_bl "", frame_text_br "", palette whiteOnDark, size_preset "match input",
bg_color "#0a0a0a", stroke_color "#d8d8d8", pixelate_zones "" (STRING).

Every widget carries a tooltip (reuse the original UI hint text captured in the plan).

## 12. Node contract

- ID `SchematicOverlay`, display "Schematic Overlay", CATEGORY "AKURATE/Schematic",
  FUNCTION "execute".
- required: `image` (IMAGE) + all widgets. optional sockets: `pixelate_mask` (MASK),
  `texture` (IMAGE) — both default None; handle absence gracefully.
- RETURN_TYPES ("IMAGE", "IMAGE", "MASK"), RETURN_NAMES ("image", "overlay_only", "overlay_alpha"):
  - image: full composite (steps 1-9).
  - overlay_only: steps 1,3-9 WITHOUT the photo (step 2 skipped; pixelate then operates
    on plain bg — zones still draw their outline+label).
  - overlay_alpha: accumulated alpha coverage (0-1 float mask, canvas size) of every
    vector element drawn in steps 4-9 (not bg/image/pixelate/texture).
- IMAGE tensors: BHWC float32 0-1. Batch: loop frames; detection per frame; same seed
  every frame. texture input: frame 0 used for all frames (keep simple).
- Determinism: identical inputs → identical outputs. No wall-clock, no global random.

## 13. Rendering implementation requirements

- numpy + PIL only (Pillow>=9.0). No scipy, no torch math (torch only at the tensor bridge).
- Anti-aliasing: vector elements drawn at 2× supersample on transparent RGBA overlays,
  downsampled with LANCZOS, alpha-composited. TWO overlay passes to preserve order:
  overlay-A = steps 4-7 (composited BEFORE texture), overlay-B = step 9 (composited AFTER).
  All geometry (positions, radii, widths, font sizes, dashes, insets) scales ×2 on the
  supersampled canvas. overlay_alpha output = union of both overlays' downsampled alpha.
- Raster steps (1,2,3,8) run at 1× directly in numpy.
- Per-element alpha via the RGBA color's alpha channel (PIL "RGBA" draw on transparent
  overlay, then Image.alpha_composite semantics give source-over correctly).
- Dashed lines: manual dash-walking helper (PIL has no dashes).
- Font: `fonts/SpaceGrotesk-Regular.ttf` loaded pack-relative via ImageFont.truetype;
  fallback ImageFont.load_default() with a printed warning. Vertical-middle text:
  anchor "lm" (or "mm" for centered) — Pillow>=9 text anchors.
- Early-out: if image_opacity==1 and every overlay feature disabled/empty, return input unchanged.

## 14. Known accepted deltas vs the original

- Font is Space Grotesk (OFL), not Telegraf (unlicensed) — metrics differ slightly.
- AA comes from 2× supersampling, not canvas rasterizer — sub-pixel differences.
- Pixelate zones come from mask/string, not clicks.
- "Randomize All" is not a node feature (use seed + external randomization).
- Everything else must match the original tool's output structurally 1:1.
