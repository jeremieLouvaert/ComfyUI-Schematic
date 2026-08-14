# Schematic Construction v2 — derivation

**Status: SIGNED BY JEREMIE 2026-08-14** ("schematicconstruct ok. sign off"),
confirming both open items: the node reuses the SchematicConstruction name, and the
section-0 cuts stand. BUILT and shipped in this commit.

**Eyeball gate on the look: PASSED 2026-08-14** — Jeremie on spike r6: *"yes, all
looking good"*. This document specifies the node that reproduces that render, and
nothing beyond it.

Authorising arc: after Micrographics was stopped (*"back to the drawing board
completely"*), Jeremie set the direction — stop altering the image, be *"mainly
overlay"* like Schematic Overlay, content is *"a combination of micrographics and
mathematical annotations"*, *"analyse schematic overlay in depth and try again with
that one as base"*. Then three decisions in one line: *"yes law superseded, revive
the construction, build the sibling node"*.

Gated evidence: `_eyeball/construction_spike_r{1..6}.py`, `CON_r3_B1..B4.png`,
`CON_r6_P1..P4.png`. Reference set: a descriptive-geometry plate, a 17th-century
astronomical diagram, and the Stoyanov schematic Overlay itself was built from.

---

## 0. What was actually revived, stated honestly

The banked v0.1 (`docs/schematic-construction-derivation.md`, signed 2026-08-08)
specified six elements. The gated look does **not** contain all six, and pretending
otherwise would make this document lie about its own provenance.

| Banked v0.1 element | Status in v2 |
|---|---|
| 1. Detection circles around points | **DEMOTED.** Hairline, small, off by default. This is Overlay's signature read and it is what made v0.1 look like more-Overlay. |
| 2. Line bundle from a focal point | **PROMOTED to the spine.** Became the convergent pencil, now the dominant gesture, and a SECOND focus was added. |
| 3. External tangent lines | **CUT from v2.** Not in the gated render. The derived tangent maths stays in the banked v0.1 doc, which is kept locally and is not part of this repository. |
| 4. Dashed projection drop-lines | **CUT from v2.** Belonged to the Monge variant, which died at his eye ("A no"). |
| 5. Intersection markers + numbered labels | **KEPT and rebuilt.** No longer circle∩circle; now ray∩sphere, which is what generates the annotation density. |
| 6. Graduated corner arc scale | **KEPT**, moved onto the outer sphere. |

So: two of six kept, one promoted to spine, one demoted, two cut. What was genuinely
reused is the *concept* and the substrate conventions, not the apparatus wholesale.

**Why v0.1 failed, corrected against evidence rather than memory.** The brain said
"sparse and single-scale". Rendering it (`_eyeball/CON_revival_r1.png`) disproved
that: the apparatus was all present and read as *undecided*, and adding density
turned it into the "debug noise" that killed it. The true fault was **one weight,
one brightness, one length class, and no organising device**. Hence §4 and §5.

## 1. Identity, and why this is not more Overlay

The composition law that killed families A and I ("every new node brings its OWN
composition") is **superseded for this node only** by Jeremie's decision of
2026-08-13. It still binds every other future family. The differentiation argument
is therefore not a licence requirement but it is still worth stating:

| | Schematic Overlay | Schematic Construction v2 |
|---|---|---|
| Mark extent | LOCAL: a circle on a point, a line between two points; connections stop at points | GLOBAL: rays that cross the whole frame and resolve to one of two foci |
| Organising device | none; marks scatter to detected points | two foci, everything converges |
| Curves | circles on points | concentric spheres about a focus, swept as arcs |
| Density source | max_circles | **computed ray∩sphere intersections** |
| Annotation | sparse labels, corner text | dense micrographic annotation ON every geometric event |
| Circles on points | the signature | deliberately demoted or absent |

**The photograph is never altered outside a zone.** That is the non-negotiable that
Micrographics broke and this node exists to respect.

## 2. Substrate — inherited from Overlay verbatim

- numpy + PIL only in the renderer. No scipy, no torch (Overlay's §13 rule).
- 2× supersampled RGBA mark layer, resolved with premultiplied alpha.
- `engine.MinstdLCG` only. Batch: per-frame seed derived from the base seed.
- Detection: `cover_fit` → `analyze_blocks` → `score_blocks` → `place_circles`,
  imported as-is, same widget semantics.
- Shared primitives from `draw_common` (`SS`, font loading, `_rgba`, `_dashed_line`,
  `_resolve_overlay`).
- **Zones: `render.pixelate_canvas` called VERBATIM** (§8).
- Three outputs, as Overlay: `image`, `overlay_only`, `overlay_alpha`.

## 3. Pipeline order — non-destructive, fixed

```
1. bg fill
2. photo composited ONCE at image_opacity          <- never re-rendered after this
3. pixelate zones (render.pixelate_canvas, verbatim)
4. mark layer, 2x supersampled:
     4a datum weight   spheres, ground rules
     4b construct      focus furniture, graduated scales, lune
     4c fine           rays, intersection dots, all annotation
5. resolve, composite
-> image, overlay_only, overlay_alpha
```

Zones are at step 3, before the marks, so *"the effect must be behind the
schematic"* holds by construction rather than by remembering to order layers. I got
that order wrong once in the spike; the pipeline position is the fix.

## 4. Weight stratification — the fix for v0.1's core fault

Three classes, in canvas-relative units so a hairline stays a hairline at any
resolution:

| Class | Multiplier | Carries |
|---|---|---|
| `DATUM` | 1.9 | the spheres, any ground rule |
| `CONSTRUCT` | 1.0 | focus furniture, graduated scale ticks, the lune |
| `FINE` | 0.55 | rays, intersection dots, every label |

**No mark may be drawn at an unclassified weight.** v0.1 drew everything at one
`line_weight` and that single fact is most of why it failed.

## 5. The apparatus

### 5.1 Two foci and their pencils
`focus_a` and `focus_b`, each placed by normalised fraction (seed-varied, §9).
Each emits `rays_a` / `rays_b` lines aimed through detected points, with a small
seeded angular jitter, extended to the canvas boundary.

**Rays start at an inner radius** (`focus_clear`, default 0.055·S), leaving the
focus uncluttered. Thirty rays converging on a point makes a tangle; the reference
keeps clear space around its sun glyph.

### 5.2 Spheres
`sphere_count` arcs concentric about `focus_a` at normalised radii, swept over an
angular span. DATUM weight for the first, CONSTRUCT for the rest.

### 5.3 The density engine — computed intersections
- **Trivial set:** a ray from `focus_a` meets a sphere about `focus_a` at exactly
  `r`. Cheap, exact, and gives `rays_a × sphere_count` candidate points.
- **Non-trivial set:** a ray from `focus_b` meets a sphere about `focus_a` via the
  real quadratic. Both roots computed; forward roots only.

Every on-canvas intersection is a candidate for a marker and a label. **This is
where annotation density comes from.** Scattered micro-type reads as dust; the same
quantity of labels placed on genuine geometric events reads as a plate. Jeremie
chose the heaviest annotation variant, against my instinct to restrain it, which
says the failure mode was never "too much" but "not attached to anything".

### 5.4 Lune
A hatched band between two adjacent `focus_a` rays across one sphere gap, at
CONSTRUCT weight every 4th line so it registers. Captioned.

### 5.5 Graduated scale
Ticks along the outer sphere, long every 5th, with running numerals.

## 6. The annotation layer

Devices, all traceable to the references:
- primed / subscripted point labels at intersections (`a12`, `c₂`, `H₁`)
- sphere captions carrying `r=` and `n=`
- **frame-exit labels**: every Nth ray labelled where it leaves the canvas with its
  index and true bearing in degrees
- graduated numerals
- leader callouts into micro-blocks
- a running `IDX / ANG / REF` data column keyed to the ray table
- corner plate text

Type is **texture, not content**: sized as a canvas fraction so it stays
sub-legible at every resolution. Font: vendored JetBrains Mono (OFL).

### 6.1 Collision resolver — and the drop policy
Every annotation label reserves its bounding box, tries a short ordered list of
alternate placements, and is **DROPPED if none is clear**.

Dropping is correct, not a compromise: it thins annotation exactly where the
geometry is densest, which is what the reference plates do by hand. Measured on the
gated render: 4–7 labels dropped per frame. The node prints the count.

### 6.2 NO shadow strokes
The spike drew every mark twice, a dark under-stroke then the light stroke, to
survive a light busy photo. **Removed on Jeremie's call** (*"do the lines have a
shadow line? if yes, skip"*). An engraved plate has no drop shadow and it gave the
marks a digital sheen.

**Legibility is solved by INK POLARITY instead**, which Overlay already does with
its `palette` widget: dark ink on a light photo. Same mechanism, same widget
semantics, no new invention.

## 7. Seed drives PLACEMENT, not only content
The seed chooses among **valid configurations**: focus positions within legal bands,
which detected points the rays aim through, sphere radii jitter, which sphere gap
carries the lune, which rays get frame-exit labels, zone centre selection. It does
not scatter freely — the convergent structure must survive any reseed.

## 8. Zones — Overlay's pixelation, reused verbatim

`render.pixelate_canvas(canvas01, w, h, centres, pixel_size, zone_size,
pixel_stroke, stroke_rgb, op, label_size, font_dir)`, called unchanged.

- Widget semantics identical to Overlay's (`pixel_size`, `zone_size`,
  `pixel_stroke`), so the two siblings behave the same.
- Zone frame and `x,y` centre label come free in Overlay's exact style.
- **Zone centres are geometry-derived**: chosen from on-canvas ray∩sphere
  intersections, spaced at least `2.1 × zone_size` apart, so every sampled patch
  sits on an event the construction produced.
- Also accepted, as Overlay does: an explicit coordinate STRING via
  `engine.parse_zone_string`, and a MASK via `engine.mask_zone_centers`.

Eight bespoke treatments were tried and rejected before this. The rule that came out
of it is in `patterns.md`: when building a sibling, reuse the sibling's function for
a shared device rather than designing a replacement.

## 9. Widget surface

Reuse the Micrographics plumbing that survived: the element registry, group
offsets as real widget pairs, preset-style scoping made visible in names, and a
tooltip on every input.

```
CATEGORY = "AKURATE/Schematic"
detection:   detection_mode, block_size, threshold, max_circles, min_distance,
             min_radius, max_radius, seed
foci:        focus_a_x, focus_a_y, focus_b_x, focus_b_y, focus_b_enabled,
             focus_clear, rays_a, rays_b, ray_jitter
spheres:     sphere_count, sphere_inner, sphere_outer, sphere_span
intersections: marker_size, marker_every, label_every
lune:        lune_enabled, lune_density
scale:       scale_enabled, scale_ticks, scale_label_every
annotation:  type_scale, label_size, exit_labels_every, callout_count,
             data_column_enabled, data_column_rows,
             frame_text_tl / tr / bl / br, frame_text_size
zones:       zone_count, pixel_size, zone_size, pixel_stroke, pixelate_zones(STR),
             pixelate_mask(MASK)
render:      palette, bg_color, stroke_color, line_weight, overlay_opacity,
             image_opacity, size_preset, texture_opacity, texture(IMAGE)
groups:      off_<group>_x / _y for each element group
elements:    STRING multiline per-element override
returns:     IMAGE image, IMAGE overlay_only, MASK overlay_alpha, STRING element_table
```

Target is Overlay's order of magnitude (46 inputs), not Micrographics' original 10.

## 10. Invariants and negative controls

Teeth written from THIS document, before the renderer.

| # | Invariant | Control that must fire |
|---|---|---|
| 1 | The photo is untouched outside zones: with `zone_count=0` and `overlay_opacity=0`, output == input exactly | any stray tonal op must fail |
| 2 | `overlay_alpha` is zero exactly where no mark was drawn | an off-by-one resolve must fail |
| 3 | Determinism: same seed + inputs → bitwise identical | unseeded RNG must fail |
| 4 | Reseeding moves the LAYOUT, not only the text | a content-only seed must fail |
| 5 | Ray∩sphere trivial roots are exactly at radius `r` | a sloppy solver must fail |
| 6 | Non-trivial ray∩sphere roots satisfy \|P−C\|=r to 1e-9 | a linearised approximation must fail |
| 7 | Every mark is drawn at one of the three declared weights | an unclassified weight must fail |
| 8 | Labels never overlap; the drop count is reported | disabling the resolver must fail |
| 9 | No mark is drawn with a shadow/under-stroke | reintroducing one must fail |
| 10 | Zones call `render.pixelate_canvas` and match Overlay byte-for-byte on identical inputs | a reimplementation must fail |
| 11 | Zone centres are ≥ `2.1 × zone_size` apart | a naive picker must fail |
| 12 | Geometry is resolution-similar at 1024 and 2048 | any pixel-unit constant must fail |
| 13 | Micro-type stays sub-legible at every resolution | a fixed-pixel size must fail |
| 14 | Every parameter changes the render (the Micrographics lesson) | an inert widget must fail |
| 15 | Every input has a tooltip | a missing one must fail |
| 16 | Empty `elements` == defaults; `id: off` removes only that element | shared-RNG coupling must fail |

Plus the standing **byteident** harness: Overlay and Voronoi must stay
byte-identical throughout (18/18 since capture).

**Per-element RNG streams are mandatory** — keyed `(seed, element_id)` via FNV-1a,
never one shared sequential stream. This was found by the blind teeth on
Micrographics: a shared stream makes disabling any element re-roll every later one,
with no visual tell.

## 11. Licence and attribution
- The apparatus is Euclid-era projective construction and Kepler-era diagram
  technique: generic, unencumbered, implemented from THIS document.
- References inform WHICH apparatus exists, never a specific composition.
- Stoyanov credit stays scoped to Overlay, per §7.6.8. This node adds none.
- Test photo is Jeremie's own generation, so exhibits carry no licence question.

## 12. Build order
1. `schematic/construction2.py` — geometry: foci, pencils, spheres, exact
   intersections, zone-centre selection, layout-from-seed.
2. `schematic/render_construction2.py` — weight classes, mark drawing, annotation,
   collision resolver.
3. `nodes/schematic_construction2.py` — widget surface, `pixelate_canvas` call,
   three outputs plus `element_table`.
4. Teeth for all 16 invariants.
5. Battery + byteident + deploy + loader-verify.
6. Exhibit plates for the final gate on Jeremie's own photo.

Naming note: the banked v0.1 files stay untouched on disk. If v2 supersedes them
outright, the old files are deleted at commit time rather than left as dead code —
the `pixel_block` lesson.

---

## Open for Jeremie
1. **Module/class naming**: reuse `SchematicConstruction` (the banked name, v0.1
   files deleted at commit) or ship as a distinct name and keep both? I recommend
   reusing the name and deleting v0.1.
2. Confirm the §0 cuts: tangents and projection drop-lines are OUT of v2.
