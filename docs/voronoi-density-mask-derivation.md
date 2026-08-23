# density_mask for Schematic Voronoi — derivation (mask-audit slice 4, Schematic side)

**Status: v2 — dry-run green, two-Opus attack folded (§10). AWAITING JEREMIE'S
SIGN-OFF. DO NOT BUILD before the §9 calls are signed.**

**v2 headline (adversary B's spec-fatal, accepted):** v1's §2 argued no floor is
needed because "a tessellation fills space". True of the TOPOLOGY, false of the
INK — the render draws ridges, and measured at shipped defaults a realistic
radial mask leaves **32.67% of 32-px blocks with zero ink** (50.75% at 16 px).
§2 is re-derived below with a `mask_min` floor on the final density; every other
kill from the attack is folded and the dry-run re-run green.

Slice 4 of the cross-pack mask audit (procedural-plan §9d item 4, §9g). Sibling
doc: `ComfyUI-Field/docs/field-scatter-mask-derivation.md`. The frozen Voronoi
spec (`docs/schematic-voronoi-derivation.md`) stays frozen; this doc AMENDS it
with one optional input.

**Provenance — the banked design is the starting point, not a blank page.**
Jeremie EXPLICITLY asked for this mask during the panels thread (signed request
2026-08-10); the panels COMPOSITION died at his eye, the density_mask mechanism
did not (§9g). Extracted from `voronoi-panels-banked.patch`:

- Optional `density_mask` MASK socket; when connected the §2 density field is
  **REPLACED** by the mask (white = fine cells, black = coarse);
  `density_gamma` still applies after; frame 0 for batches; clamp [0,1];
  `build_density(fitted01, source, gamma, mask01=None)` with the None-default
  path byte-identical.
- Banked teeth intent kept: gradient mask ⇒ seed histogram follows the mask,
  not luma; control: inverted mask flips it.

Amendments to the banked code, each derived below: NaN scrub (§4), 4-D/empty
mask guards (§4), cover-fit instead of stretch for the canvas mapping (§3),
a budget-honesty console line (§5). The panels-mode plumbing is NOT extracted
— this slice touches the shipped full-frame node only.

**The public-line constraint:** Schematic v0.4.0 is pushed and public. The new
input is OPTIONAL; `density_mask=None` must stay BYTE-IDENTICAL (the pack's
byteident golden-harness law, extended never weakened). Zero new widgets are
needed (§2), so the positional-`widgets_values` hazard does not arise at all.

---

## 0. What the shipped density path actually does (read from
`schematic/voronoi.py`, not from memory)

```
build_density: field(source) → field /= max(field)   (guard max < 1e-9 → zeros)
               density = clip(normalized,0,1) ^ density_gamma          ∈ [0,1]
sample_seeds:  one MinstdLCG(seed) stream, strict (x,y,u) triples,
               accept iff u < density[⌊y⌋,⌊x⌋]; stop at `cells` accepted
               or 80·cells triples (fixed budget, deterministic termination)
build_cells:   mirror-closed scipy Voronoi — the tessellation TILES the canvas
               whatever the seed distribution is
field_at_seeds = density at each accepted seed   (drives cell_fill plate-tone)
```

Three structural facts carry the derivation:

1. **The mask cannot consume RNG draws.** The MINSTD stream draws (x, y, u)
   triples regardless of WHERE density is high; a mask changes only the accept
   test's right-hand side. Determinism at fixed (seed, mask) is inherited, and
   `density_mask=None` leaves the stream byte-identical.
2. **Normalisation eats levels.** `field /= max(field)` means only the field's
   RELATIVE structure survives — the §1 analysis.
3. **A Voronoi tessellation fills space by construction.** Zero seeds in a
   region does not mean no mesh there; it means the neighbouring cells EXPAND
   into it. Black = coarse, never empty — the §2 floor argument.

## 1. The model — the mask REPLACES the field (the signed design), and why
not multiply

When `density_mask` is connected:

```
field   = mask (canvas-mapped, §3; scrubbed, §4)      — density_source UNREAD
density = clip(field / max(field), 0, 1) ^ density_gamma
```

**Replace vs multiply, derived (the §9g composition-point question).** The §9g
brief sketches "multiply into the density field BEFORE its normalisation". Run
both through fact 2:

- *Multiply:* `density ∝ (field·mask / max(field·mask))^γ`. A UNIFORM mask of
  any level c cancels exactly (`c·field / max(c·field) = field / max(field)`)
  — the normalisation eats a uniform bias ENTIRELY. That is §9g's own named
  negative control, and it means multiply's only real content is the mask's
  relative structure — same as replace's.
- *Replace:* a uniform mask normalises to ones ⇒ uniform FINE mesh. The level
  is equally eaten; only relative structure acts (invariant V3 asserts this).
- The decisive asymmetry is COMPOSABILITY: multiply is reachable FROM replace
  externally — feed the node a pre-multiplied mask (Field From Image supplies
  luma/detail as a MASK; one **Field Combine** multiply — the MASK×MASK node;
  Field Composite is the IMAGE compositor; naming corrected by adversary B).
  But replace is NOT reachable from multiply (no mask input can un-multiply
  the photo's field). Honesty bound on the composability claim (adversary
  note 13): the external reconstruction of this node's own field is
  APPROXIMATE, not exact — `build_density` computes Rec.601 luma on the
  COVER-FITTED array and `detail`'s window is keyed to the CANVAS min-dim, so
  an externally-built luma/detail mask matches closely but not bitwise.
  Replace remains the right primitive; the claim is "reachable in practice",
  not "reachable exactly". It is the design Jeremie signed on 2026-08-10.

**Adopted: replace, the banked semantics verbatim.** Consequences:

- **`density_source` is INERT when the mask is wired** — including its ratified
  polarity pairing (brightness↔dark grounds, darkness↔blackOnLight): the
  mask's polarity is ABSOLUTE (white = fine, regardless of palette). Both
  tooltips state it; invariant V4 asserts it (byte-identical across all three
  sources under a wired mask).
- **`density_gamma` stays live** (applies to the normalised mask — the banked
  design; the knob already on the node is the mask's contrast control, so no
  new gamma widget is added).
- **`field_at_seeds` follows the mask**: `cell_fill` plate-tone and any
  field-graded rendering now grade by the MASK's value at each seed, which is
  the consistent reading (the mask IS the field). Stated; invariant V8.
  **Derived consequence (adversary-measured), stated in the cell_fill
  tooltip:** under a hard-edged (binary) mask, seeds only accept where
  density ≈ 1, so `field_at_seeds` collapses to a SINGLE value (measured std
  0.128 unmasked → 0.000000 masked) — the plate-tone renders FLAT. Values
  stay in contract ([0,1]); the per-cell grading promise degrades to uniform
  exactly when the mask is binary. A `mask_min` floor restores some spread in
  black zones (their accepted seeds carry the floor value).

## 2. The floor question, re-derived at the INK level (v2 — v1's geometric
dissolution was refuted by measurement)

v1 argued the tessellation fills space, so black = coarse cells, never empty —
and proposed no floor. Adversary B measured the claim at the level the user
actually sees, the drawn ink (`overlay_alpha` in pixel blocks, shipped
defaults, cells 12000, 640×480):

| mask | zero-ink blocks @16 px / @32 px / @64 px |
|---|---|
| unmasked photo | 0.00 / 0.00 / 0.00 % |
| radial gaussian (σ = 0.10·W — "mask the subject") | **50.75 / 32.67 / 14.29 %** |
| 3×3-px white blob | 61.33 % @16 (26 seeds, largest cell 29.3 % of canvas) |

The topology claim is true — cells tile the canvas — but a black zone's ink is
only the ridges that happen to CROSS it, and a large cell's interior is empty
canvas. `density_gamma` cannot repair this (measured monotonically worse with
γ, and 12.67 % @32 px even at the widget floor γ = 0.5); `cells` = 30000
still leaves 23 %.

**The v2 floor, derived.** The floor goes on the FINAL density, after
normalisation and gamma, mask-wired only:

```
d_eff = mask_min + (1 − mask_min) · d        (d = clip(mask/max)^γ as §1)
```

Chosen over flooring the raw mask (`max(m, c)` pre-normalisation, the
adversary's probe form) for three derived reasons: (a) LEVEL-INDEPENDENCE —
normalisation rescales by the mask's max, so a pre-normalisation floor's
effective strength varies with mask level, while a post-chain floor means
exactly "minimum acceptance probability = mask_min" always; (b) DEGENERATE
SANITY — an all-black mask + pre-norm floor normalises to ALL-ONES (uniform
FINE mesh, absurd); the post-chain form gives uniform sparse density
mask_min (measured: n = 9627 accepted at 0.01, a coarse even field);
(c) EXACTNESS — `mask_min = 0` is bitwise the unfloored chain (`0 + 1·d`,
measured), `mask_min = 1` is the white-mask uniform. Op order matches slice 3
(gamma before floor).

Measured sweep (radial config above, `_density_mask_dryrun/floor_form_sweep.py`):

| mask_min | zero-ink @16/@32/@64 | ink hot | ink cold | fine:coarse |
|---|---|---|---|---|
| 0.0 | 50.75 / 32.67 / 14.29 % | 0.634 | 0.0024 | 261:1 |
| 0.005 | 0.08 / 0 / 0 % | 0.706 | 0.027 | 26:1 |
| **0.01** | **0 / 0 / 0 %** | 0.685 | 0.041 | **16.8:1** |
| 0.05 | 0 / 0 / 0 % | 0.572 | 0.085 | 6.8:1 |
| 1.0 | 0 / 0 / 0 % | 0.181 | 0.183 | 1:1 |

The coarse-cell read is NOT destroyed by a small floor (v1's stated fear,
refuted): 16.8:1 contrast survives at 0.01. **Proposal (direction call 2,
REVISED): ship `mask_min`, ONE new FLOAT widget declared in `optional` (NOT
`required` — adversary A proved with ComfyUI's own `validate_prompt` that a
new REQUIRED input rejects every stored API-format prompt with "Required
input is missing"; an optional input with a python default in the execute
signature keeps both load paths valid while still rendering as a widget
appended at the end of widgets_values), range 0..1 step 0.005, default
0.01** — holes closed with 17:1 contrast out of
the box, matching the slice-3 floor-not-gate taste; `mask_min = 0` restores
the pure banked semantics (sparse-to-empty as data-art) bitwise. Default 0.0
offered as the alternative if he prefers the banked look first. Active only
when density_mask is wired (bitwise inert otherwise — the floor is inside the
mask branch).

Degenerate case unchanged: an all-zero mask at `mask_min = 0` yields < 4
seeds → the shipped mesh-free path + console note (V7); at mask_min > 0 it
seeds uniformly sparse (stated above).

**Sparse-geometry honesty (adversary-measured, stated not hidden):** an
extreme mask (3×3-px blob) at mask_min 0 gives 26 seeds, 79 ridges, one cell
covering 29.3 % of the canvas — no crash (the mirror closure held under every
pathological mask probed, including a single white pixel), but "coarse mesh"
at that extreme is a handful of polygons. The floor default makes this
configuration opt-in rather than a first-touch surprise.

## 3. Canvas geometry — cover-fit, an AMENDMENT to the banked stretch

The banked code stretch-resamples the mask rect to the canvas. But the field
the mask replaces is built from the COVER-FITTED photo (`build_density(fitted…)`
— frozen §1: crop to canvas aspect, then resize). A user who paints a mask ON
THE SOURCE PHOTO and picks a size preset would get a stretched mask misaligned
with the cropped photo — every masked feature displaced from the photo feature
it was painted on. Rule adopted:

- mask dimensions == canvas (w, h) exactly → used AS-IS (no resample at all);
- otherwise → **`engine.cover_fit`**, the same transform, constants and
  bilinear engine the photo itself goes through (mask lifted to (H,W,1) for
  the call). Canvas-sized masks stay verbatim; source-aligned masks stay
  aligned; mismatched-aspect masks get the same stated centred-crop behaviour
  the photo gets. Departure from the banked implementation detail, kept inside
  the signed semantics; direction call 3. Invariant V10 pins it.

No interpolation-engine departure: bilinear via the pack's OWN
`engine._bilinear_resample` (inside cover_fit) — §9g's "Schematic is its own
render path" note; the Field pack's torch align_corners convention does not
reach across packs.

## 4. Mask I/O — slice-2/3 rules inherited into numpy, banked code amended

| rule | this node |
|---|---|
| socket | optional `MASK` named `density_mask`, appended after `texture` (sockets carry no widgets_values entry — position-safe) |
| absent | `mask01=None` default → the UNTOUCHED field path — **byte-identical** (V1, harness law) |
| shape | **v2, adversary-hardened:** squeeze ALL trailing size-1 dims first, THEN `reshape((-1,) + shape[-2:])`; after resolving frame 0, REQUIRE both dims ≥ 8 — else treated absent + console warn. v1's bare reshape silently corrupted trailing-channel masks: `(H,W,1)` → a (640,1) sliver that cover-fits and RENDERS garbage without raising (measured); the plausibility guard turns every such shape into a loud no-op. Accepts 2-D, `(B,H,W)`, `(1,1,H,W)`, `(H,W,1)`, `(1,H,W,1)`; rejects-with-warn `(H,W,3)` and anything else (a 3-channel image is not a MASK) |
| batch | frame 0 (banked + pack texture convention; sibling-pack alignment) + console note when M > 1; `M = 0` → treated absent + warn (the banked `dm[0]` would IndexError — measured hazard class from slice 2) |
| values | `np.nan_to_num(dm, nan=0.0)` then clip [0,1] — **amendment**: `np.clip` propagates NaN (measured this session), and a NaN density pixel silently never accepts a seed |
| dtype | float32 in, float64 into `build_density` (the module's own dtype) |
| resize | §3: as-is at canvas size, else `engine.cover_fit` |
| all-zero mask | degenerate mesh-free path via the existing < 4-seed guard + its note; one additional `[Schematic]` line naming the wired mask as the likely cause (LoadImage alpha-less trap, slice-2 rule) |

## 5. Interplay and honesty items

- **Seed-count shortfall.** Acceptance probability ≈ mean(density). A sparse
  mask (small white area on black) can exhaust the 80·cells triple budget
  before `cells` seeds accept — shipped behaviour for dark photos too, but a
  mask makes it easy to hit. One console line when the mask is wired and
  `n < cells` (banked code has none). The budget itself is FROZEN spec (batch
  determinism depends on it) and is not touched. (Adversary-verified: a dark
  unmasked photo still reaches 12000/12000, so restricting the line to the
  mask-wired path hides no common case.)
- **Performance, measured not promised (v2 — v1's "neither speeds nor slows"
  was false as written):** a sparse mask exhausts the full 80·cells budget
  where a photo's field exits early — the seed loop alone measured 10.4×
  (0.13 s → 1.31 s at cells 12000, 1600×1200, 1 %-white mask). End-to-end it
  is diluted by render cost: 1.06–1.20× at shipped settings (+1–2 s worst
  probed). The mask_min floor default raises mean density and pulls the loop
  back toward early exit.
- **Batch collapse under a mask (adversary-found, now stated):**
  density_source is inert (§1) and the mask is frame 0 (§4), so with a mask
  wired the density — and therefore the SEEDS and the whole tessellation —
  is identical for every frame of a batch (measured byte-identical
  overlay_only across frames; only the faint background photo varies). This
  is the consistent consequence of the two rules, stated in the tooltip.
  Per-frame masks (pair mask i with frame i when M == B) are a possible
  LATER extension of the new input, noted for Jeremie, not built here.
- **Polarity**: white = fine/dense everywhere, absolute (§1). The ratified
  source↔palette pairing guidance applies only to the unmasked path; tooltip
  states it.
- **`cells` meaning unchanged**: still the target ACCEPTED count, mask or no
  mask (structure-scale resolution behaviour, frozen §3, untouched).

Tooltips (binding content, v2):

- `density_mask`: "Optional mask that DRIVES SEED DENSITY, replacing
  density_source entirely: white = fine/dense cells, black = coarse.
  density_gamma still applies. Only the mask's relative structure matters (it
  is re-normalised to its own maximum: a uniform grey mask equals a white
  one). At mask_min 0 black zones can render with NO ridges crossing them —
  raise mask_min to keep a sparse mesh everywhere. Cover-fitted like the
  photo when sizes differ; frame 0 used for batches (all frames then share
  one tessellation); an empty or non-mask-shaped input is ignored with a
  console note."
- `mask_min` (new widget, §2): "Minimum seed density where density_mask is
  black, as a fraction of the maximum. 0 = the mask alone decides (black can
  go empty); default 0.01 keeps a sparse coarse mesh everywhere. Only active
  when density_mask is wired."
- `cell_fill` (tooltip AMENDED, one clause): "…brightness-scaled by that
  cell's own field value (the mask's value when density_mask is wired — flat
  under a binary mask)."

## 6. Invariants — the failure lattice (every row DRY-RUN before sign-off;
DR markers filled by the dry-run record, §8)

Knife edges: mask ≠ canvas size with aspect mismatch; batch frames; awkward
constants (γ 2.5, cells 3000, seed 42 and 7); NCs fired at pinned configs.

| # | invariant | assertion | control (must fire) |
|---|---|---|---|
| V1 | absent byteident | `tools/test_byteident.py` EXTENDED with 4 SchematicVoronoi configs (defaults; darkness+blackOnLight+cells 4000; detail+goldOnDark+cell_fill 0.3+seed_dots; batch-2+Story preset) — goldens captured on PRE-CHANGE HEAD (artifact: `_goldens/voronoi_mask_baseline.json`, schema `{config: {image/overlay_only/overlay_alpha: md5}}`, config dicts recorded in `_density_mask_dryrun/dryrun.py` — the builder asserts against THIS file), post-build `density_mask=None` re-check → all 18 existing + 12 new hashes match. **v2 (adversary-found one-sidedness): the build ALSO captures ≥1 MASK-WIRED golden** (VG3's config + a fixed synthetic mask + an aspect-mismatched preset — crop + cell_fill + mask in one config) so the mask path is pinned against future refactors | any default nudged in a config: hash differs (harness mechanics, self-evidencing) |
| V2 | mask-follows (banked) | **v2-hardened (v1's thresholds passed a mask-ignoring AND a γ-dropping build, adversary-measured):** horizontal-ramp mask on a FLAT grey photo: mean(x)/w ≥ **0.72** (measured 0.7706 at γ 2.5; a γ=1 build measures 0.6581 and FAILS; uniform measures ~0.5 and fails); decile RATIOS `h[9] > 3·h[4] > 10·h[1]` (measured 899 / 183 / 7; uniform deciles min-diff −13 passed v1's `≥ −30` row and fails this one) | inverted ramp: mean(x)/w ≤ 0.28 (measured 0.2275); the γ=1 and uniform builds fail the main clauses |
| V3 | uniform-eaten NC | `mask = 0.25` const vs `mask = ones`: accepted seeds ARRAY-IDENTICAL (normalisation eats the level — the §9g named NC, asserted as designed behaviour) | a build that skips normalisation: seed POSITIONS differ (array-inequality), **counts do NOT** (v2 correction, adversary-measured: 3000 = 3000 — mean(0.25²·⁵)=0.031 > the 1/80 budget bound, so v1's "~4× fewer seeds, fires massively" was FALSE and is withdrawn; the position-difference clause is the whole control) |
| V4 | source inert | wired gradient mask, density_source ∈ {brightness, darkness, detail} on a structured photo: all three FULL OUTPUTS byte-identical | `density_mask=None`: brightness vs darkness differ (existing behaviour as control) |
| V5 | gamma law | module-level: `build_density(fitted, src, γ=3, mask01=m)` == float64 oracle `clip(m/max,0,1)^3` elementwise ≤ 1e-15, where the oracle rounds the mask through FLOAT32 FIRST (the socket dtype — a raw-f64 oracle misses by 1.39e-7, measured); node-level: seeds equal to oracle-driven `sample_seeds` (measured exact) | wrong exponent (γ=2): seeds differ (measured) |
| V6 | I/O rows | 4-D `(1,1,H,W)` ≡ 2-D same content (byte-identical); `(0,H,W)` → absent + warn (== V1 path); NaN-pixel mask == hand-zeroed mask (byte-identical); torch CUDA-resident mask (CUDA measured available on this box) → no crash, == CPU result; bool/float16/float64 torch masks accepted (adversary-verified); M=2 → frame 0 (== frame-0-only run) + note; **v2 shape-guard rows:** `(H,W,1)` and `(1,H,W,1)` ≡ 2-D same content (squeeze rescues them); `(H,W,3)` → absent + warn (NOT a silent sliver) | the banked bare `dm[0]` on `(0,H,W)`: IndexError; v1's bare reshape on `(H,W,1)`: a (W,1) sliver that RENDERS without raising (both measured — the guards are not optional) |
| V7 | all-zero mask | zeros mask at mask_min 0 → mesh-free render (existing degenerate path), the two console notes, no crash; at mask_min 0.01 → uniform sparse mesh (measured n=9627) | — (crash-freedom + note strings are the teeth) |
| V7b | fmax guard pinned | **v2 (adversary-found unpinned):** uniform 1e-12 AND 5e-10 masks take the degenerate all-zero path (guard `< 1e-9` fires); 2e-9 does NOT (normalises to ones) — the boundary is asserted on both sides | a `field/field.max()` build with no guard: 1e-12 mask yields a FULL 3000-seed mesh (measured) — fires |
| V8 | field_at_seeds follows mask | module-level: with mask wired, `field_at_seeds[i] == density[⌊y⌋,⌊x⌋]` where density is the MASK-derived field (drives cell_fill plate-tone) | field_at_seeds read from the luma field instead: differs on a mask⊥luma config (DR) |
| V9 | determinism + budget honesty | same (photo, mask, seed) twice: byte-identical; sparse mask (1% white, γ 1 — mean density 0.0094 < the 1/80 budget bound): `n < cells` (measured 2247/3000 accepted) AND the shortfall console line prints; dense mask: reaches `cells` (measured 3000/3000), no line | the 1/80 law itself is the control's arithmetic: mean(density)·80·cells caps the acceptances; a 5% mask does NOT fire (0.05 > 1/80, measured reasoning run 1) |
| V10 | cover-fit geometry | source-sized mask + aspect-mismatched canvas: seeds == oracle using `cover_fit(mask)`; canvas-sized mask: used verbatim; **v2 (adversary-found blind spot): the witness mask has its PEAK IN THE CROPPED REGION** — a normalise-before-cover_fit ordering error is invisible on a crop-surviving mask (measured bit-identical) and measures maxΔ 0.700 on the peak-cropped one; the row pins the §1 order (canvas-map FIRST, then normalise) | stretch-resample oracle (the banked geometry): seeds differ at aspect mismatch — the amendment's witness (adversary re-measured the displacement: 158 px / 11.3 % of canvas width, 48 % area error; cover-fit is also resolution-independent across full/half/quarter masks, measured identical centroids) |
| V12 | batch collapse stated | M=1 mask + B=2 batch: overlay_only frames byte-identical (measured; the §5 statement's tooth); mask=None same batch: frames differ | — (the absent-path clause is the control) |
| V13 | floor law | `mask_min=0` bitwise == unfloored chain (measured); `mask_min=0.01` on the radial config: ZERO empty 16-px ink blocks AND fine:coarse ink ratio ≥ 10 (measured 16.8:1); `mask_min=1` == white-mask uniform; remap: `node(mask, mask_min=c)` == oracle `c+(1−c)·d` in float64 ≤ 1e-15 | wrong composition (floor BEFORE gamma, `(c+(1−c)·n)^γ`): differs (to be fired in teeth); v1's no-floor build: 32.67 % empty blocks (measured) — the spec-fatal itself |
| V11 | pack sweep | `tools/test_voronoi.py` + all other Schematic suites stay green, zero rows weakened; loader counts **3 nodes** (Overlay, Voronoi, Construction — verified from `nodes/__init__.py` this session, NOT the misremembered 4) | — |

Teeth style: the pack's own — byteident harness extension (V1) + a new
`tools/test_voronoi_mask.py` in the `test_voronoi.py` conventions
(module-level rows against `schematic/voronoi.py` + node-level rows through
`execute`, embedded python). Implementation-blind agent; the module surface in
the frozen spec §7 is the declared seam, **extended compatibly** (v2 wording —
adding `mask01=None` does change the signature, backward-compatibly; v1's
"unchanged" was imprecise).

## 7. Implementation constraints for the builder (Sonnet, zero deviation)

1. `build_density(fitted01, source, gamma, mask01=None, mask_min=0.0)` —
   the banked signature extended with the §2 floor param (consistency fix
   flagged by the blind teeth agent: §6 already named mask_min at this seam;
   v1 of this line omitted it); the `mask01 is None` branch must be the
   UNTOUCHED existing code (byteident guards it, but structure it so
   identity is by construction).
2. Mask prep at node level (torch→numpy bridge; §4 order: squeeze-trailing-1s
   → reshape → empty guard → frame 0 → nan_to_num → clip → the ≥8-both-dims
   plausibility guard), geometry per §3; `schematic/voronoi.py` stays
   torch-free and scipy-free at import (frozen law).
3. The §2 floor: `d_eff = mask_min + (1−mask_min)·d` applied INSIDE the mask
   branch only, after gamma; `mask_min == 0.0` must skip the op structurally
   (bitwise unfloored — V13).
4. Console lines prefixed `[Schematic]`: M>1 frame-0 note, non-mask-shape
   warn, M=0 warn, all-zero line, sparse-shortfall line (ALL mask-wired only
   — the absent path's console output stays exactly as shipped).
5. ONE new widget `mask_min` declared in `optional` with a python default
   (`mask_min=0.01`) in the execute signature — NEVER in `required` (stored
   API-format prompts reject new required inputs; adversary-A-measured on
   the Field sibling with ComfyUI's own validator, and the hazard is
   identical here); socket `density_mask` appended after `texture`; tooltips
   per §5.
6. The MINSTD stream, budget, mirror-closure, ridge grading: UNTOUCHED.
7. Byteident: the 4 absent-path goldens are ALREADY captured on pre-change
   HEAD (`_goldens/voronoi_mask_baseline.json` — assert against this file,
   do not re-capture); add the mask-wired golden (V1) post-build; the
   extension commits with the build.

## 8. Dry-run record (2026-08-18, embedded python)

`_density_mask_dryrun/dryrun.py` — final tally **22 PASS / 0 FAIL, 8 NC fired
/ 0 silent**. The prototype implements the §1 replace chain + §3 geometry +
§4 prep on top of the SHIPPED `voronoi.build_density` / `sample_seeds` /
`engine.cover_fit` (nothing reimplemented below the mask branch).

Two runs to green; the single repair class is itself a spec point, folded into
V5/V10: **the socket dtype is float32** — every float64 oracle must round the
mask through f32 before comparing, exactly as a ComfyUI MASK arrives (raw-f64
oracles missed by 1.39e-7 on three rows; the prototype semantics were correct,
the oracles were not).

Measured highlights now pinned in §6: ramp-following mean(x)/w 0.7706 vs
0.2275 inverted (V2); the uniform-0.25 ≡ ones identity exact, both densities
all-ones (V3); all three density_sources byte-identical under a wired mask
while differing without one (V4); the banked `dm[0]` IndexError on an empty
batch reproduced (V6 — the guard is not optional); zero-mask → zero seeds →
the shipped mesh-free path (V7); field_at_seeds exactly the mask-derived
density (V8); the 1/80 budget-shortfall law measured at 2247/3000 on a 1%
mask (V9); cover-fit == oracle with the banked stretch differing at aspect
mismatch — the amendment's witness fired (V10). Byteident baseline for the 4
new configs captured on pre-change HEAD into
`_goldens/voronoi_mask_baseline.json`, deterministic capture verified per
config (V1).

**v2 addendum (post-attack, `floor_form_sweep.py` + the v2 dry-run rows):**
the §2 floor-form sweep measured (holes closed at mask_min 0.01, 16.8:1
contrast, floor-0 bitwise, all-black degenerate sane at n=9627), and the v2
rows (V2-hardened thresholds, V3 restated, V6 shape guards, V7b guard
boundary, V10 peak-cropped witness, V12 batch collapse, V13 floor law) were
added to `dryrun.py` and the whole suite re-run: **35 PASS / 0 FAIL, 16 NC
fired / 0 silent**. Two v2-round repairs, both recorded in-script: the V13
ink row is CONFIG-SPECIFIC (320×240 + corner labels put ink in the cold
region → ratio 9.7; re-pinned at the sweep's exact 640×480 clean config →
16.8), and the first V10v2 wrong-order oracle double-normalised and erased
its own error (maxΔ 0.000 → single-normalise → 0.700).

## 9. Direction calls for Jeremie at sign-off

1. **Replace semantics confirmed** (your 2026-08-10 signed ask, banked patch
   extracted; multiply rejected as less expressive in practice, §1).
2. **REVISED IN v2 — `mask_min` floor widget, default 0.01** (§2): v1
   proposed no floor on a geometric argument the adversarial pass refuted at
   the ink level (a realistic mask leaves a third of the canvas with zero
   ink; a 0.01 floor closes every hole at 16.8:1 fine/coarse contrast,
   measured). `mask_min = 0` restores the pure banked semantics bitwise —
   the alternative default if you prefer sparse-to-empty out of the box.
3. **Cover-fit canvas mapping** (§3) — an amendment to the banked stretch,
   keeping painted masks aligned with the photo under size presets
   (adversary re-measured the stretch displacement at 158 px / 48 % area
   error, and cover-fit resolution-independent).
4. Batch = frame 0 + console note (banked + pack convention; aligned with
   the Field sibling this slice). **Consequence, stated in §5: with a mask
   wired, every frame of a batch shares ONE tessellation** (density no
   longer reads the photo). Per-frame masks (M == B pairing) are a possible
   later extension — your call whether to want it now (it changes this new
   input only).
5. Version **v0.5.0**. **PUSH WARNING (verified this session):
   `.github/workflows/publish.yml` auto-publishes to the Comfy registry on ANY
   push to main that touches `pyproject.toml`** — same trap as Darkroom
   (2026-08-17). The version bump and the push word are yours; when you give
   the push word, the registry publish WILL ride along unless the workflow is
   held.

## 10. Adversarial pass record (2026-08-18)

Two fresh-eyes Opus adversaries attacked the slice (this doc: adversary B;
probes in `_density_mask_dryrun/adv_b/`, four scripts). Verdict on v1:
**DO-NOT-SIGN — 1 spec-fatal, 9 must-fix, 7 notes.** All folded:

- SPEC-FATAL: §2's no-floor geometric argument false at the ink level
  (zero-ink tables in §2) → the v2 floor, re-derived and re-measured.
- MUST-FIX: V3's control magnitude falsified (counts equal, positions
  differ — restated); V2 vacuous against mask-ignoring and γ-dropping
  builds (thresholds hardened to 0.72 / decile ratios); the `fmax < 1e-9`
  guard unpinned (V7b added, boundary both sides); V10 blind to a
  normalise-before-fit ordering error (peak-cropped witness); §4 shape rule
  corrupted trailing-channel masks silently (squeeze + ≥8-dims guard, V6
  rows); batch collapse undocumented (§5 + V12 + tooltip + call 4);
  byteident extension one-sided (mask-wired golden added to V1);
  §5 performance sentence false as written (replaced with measured 10.4×
  seed-loop / 1.06–1.20× end-to-end); cell_fill plate-tone flattens under a
  binary mask (§1 + tooltip).
- NOTES folded: Field Combine naming; composability stated as approximate;
  seam wording "extended compatibly"; baseline artifact named with schema;
  M=0 behaviour in tooltip; sparse-geometry numbers in §2.

What the attack could NOT break (their held list, abridged): Qhull/degenerate
geometry under every pathological mask including a single white pixel (the
mirror closure holds); the cover-fit amendment (stronger than v1 claimed:
exact photo alignment + resolution independence); V4 source-inertness at
full-output level; the torch dtype bridge (bool/f16/f64); the 1/80 law's
arithmetic and the mask-wired-only gating of the shortfall line; all frozen-
spec contracts (MINSTD order, budget, mirror closure, batch same-seed,
degenerate path); the push-trap warning; and the v1 dry-run itself,
reproduced verbatim (22/0, 8/8).
