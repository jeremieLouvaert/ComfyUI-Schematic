# texture_mask — parameter addition to the shipped Schematic Overlay

Status: SIGNED OFF BY JEREMIE 2026-08-08. FROZEN.
Family H of `data-art-plan.md` §2 (masked texture-through), the "+1 parameter
addition" of the signed §7.6 taxonomy, and one of the exactly two touches to
existing code this phase allows. The frozen spec `schematic-derivation.md` stays
untouched; this document is the v0.1.1 addendum. The frozen §7 step 8 behavior is
modified ONLY when the new socket is connected.

## 1. Socket

- `texture_mask` (MASK), optional, default None, in the node's `optional` block
  beside `pixelate_mask` and `texture` (the twice-proven optional-socket shape).
- Semantics: continuous modulator. 1.0 = full texture effect, 0.0 = none.
- Batch: frame 0 used for all frames (the exact rule `texture` and `pixelate_mask`
  already follow).
- Resample when mask size ≠ canvas: BILINEAR (via `engine._bilinear_resample` on the
  mask as a single-channel array), then clamp to [0,1]. Not nearest:
  `nearest_resample_mask` exists for pixelate_mask's BINARY centroid use; a
  continuous opacity modulator resampled nearest would alias every edge at any size
  mismatch.

## 2. The math — mask multiplies into the blend opacity BEFORE the lerp

Frozen §7 step 8 is `result = base + (screen − base) · texture_opacity` with
`screen = 1 − (1−a)(1−b)`. The addition, per pixel with m = the resampled mask:

- texture_mask connected: `result = base + (screen − base) · (texture_opacity · m)`
- texture_mask None: the ORIGINAL scalar expression executes on the original code
  path (a guarded branch, not a mask-of-ones through new code — belt and braces on
  top of the fact that ×1.0 is float-exact).

Applies identically to the procedural grain fallback (same step 8), to
`overlay_only` (step 8 runs there), and leaves `overlay_alpha` untouched (texture
never contributed to alpha coverage).

## 3. Implementation surface (exhaustive)

- `nodes/schematic_overlay.py`: socket declaration + prepare mask01 (float32,
  frame 0, clamp) + pass through.
- `schematic/render.py`: `render_frame(..., texture_mask01=None)` new default-None
  kwarg; step 8 gains the guarded branch (resample happens once in the node, not per
  render call — pass the canvas-sized array).
- `engine.py`: UNTOUCHED. No other line in any existing file changes.

## 4. Verification (blind agent builds `tools/test_texture_mask.py` from this doc)

Byte-level guarantees, each a named scalar (max |Δbytes| over the three outputs):

a. None-path: with the socket absent, all golden hashes (suite zero, §5) unchanged.
b. Ones-mask: an all-ones canvas-size mask ⇒ bitwise identical to None.
c. Zeros-mask: an all-zeros mask ⇒ bitwise identical to a texture_opacity=0 render
   (exact because (screen−base)·0 = 0 and x+0 = x for finite floats).
d. Locality: a canvas-size half mask (exact size, no resample band) ⇒ pixels under
   m=1 bitwise equal the ones-render, pixels under m=0 bitwise equal the
   zeros-render (step 8 is pointwise).
e. Negative control (the scalar moves): with a BRIGHT texture and texture_opacity >
   0, the half-mask render differs from the ones-render exactly and only on the
   m=0 side.
f. The existing `tools/test_schematic.py` suite stays green throughout.

## 5. Suite zero — the golden byte-identity harness (this phase's regression teeth)

`tools/test_byteident.py`, runnable on the embedded python (torch available; it
drives the real node `execute`). Modes: `--capture` (render the config set on
current code, store hashes) and `--check` (re-render, compare). Goldens are
LOCAL-ENVIRONMENT artifacts (font rasterization varies by Pillow build): stored in
`_goldens/` (gitignored — one .gitignore line), captured on the pre-change HEAD and
checked after EVERY touch this phase (draw_common extraction; texture_mask; each new
node landing, which must not perturb the shipped node).

Config set (deterministic synthetic photos, seeded np.random.default_rng):
- G1 all defaults, 642×480 input (odd sizes exercise cover-fit rounding).
- G2 chain-heavy: chain_count 21, chain_ratio 0.85, chain_angle 30.
- G3 crosshair off, pixelate_zones "100,100;300,200", pixel_size 12, zone_size 60.
- G4 custom palette #102030/#ffcc00, shape square, texture_opacity 1.0 (procedural
  grain path, seed+1 idiom).
- G5 batch of 2, size_preset "Instagram Story (1080x1920)", image_opacity 0.5.
- G6 blackOnLight, connection_dist 0, crosshair_star_size 0, texture_opacity 0.
Hash: md5 over each output tensor's contiguous float32 bytes, all three outputs,
every frame.
