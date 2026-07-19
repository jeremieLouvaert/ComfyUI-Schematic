# ComfyUI-Schematic

Turn any image into a technical/schematic annotation overlay, detection circles, tangent chains, crosshair frames, pixelation zones and all, in one node.

![screenshot placeholder](docs/screenshot.png)

## Features

**Detection**
- Four scoring modes: combined (contrast x brightness deviation), contrast, bright, dark
- Adjustable analysis grid, threshold, circle cap, and minimum spacing between circles

**Shapes**
- Circle or square outlines, with size driven by detection score plus a seeded random jitter

**Connections**
- Straight lines drawn between every pair of circles within a configurable distance

**Chain**
- A tangent chain of circles radiating outward from the canvas center at any angle, with shrinking or growing radius per step
- Optional intersection markers with coordinate labels at every adjacent-circle overlap

**Crosshair**
- Dashed full-canvas cross, dashed centered square frame, and a configurable center asterisk

**Pixelate**
- Mosaic pixelation zones placed by a coordinate string or by a mask (each connected white region becomes a zone), with optional outline and coordinate label

**Frame text**
- Four corner labels, inset from the canvas edge, for a studio-slate look

**Palette**
- Four built-in palettes (whiteOnDark, blackOnLight, goldOnDark, greenOnDark) plus a custom hex background/stroke pair

**Texture**
- An optional texture image screen-blended over the whole canvas. If none is connected and texture opacity is above zero, a seeded procedural grain fills in

**Sizes**
- Match the input image, or crop-to-fill (cover) into five fixed presets: Portrait 3:4, Square, Landscape 16:9, Instagram Story, Poster

## Outputs

- **image**: the full composite, background, photo, pixelation, crosshair, connections, circles, frame text, texture, and chain, all in order.
- **overlay_only**: the same stack without the photo, so the vector layer can be composited elsewhere.
- **overlay_alpha**: a 0 to 1 mask of everywhere a vector element (crosshair, connections, circles, frame text, chain) was drawn, for downstream masking.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/jeremieLouvaert/ComfyUI-Schematic
```

Restart ComfyUI. The node appears under **AKURATE/Schematic** as "Schematic Overlay".

## Inputs

| Input | Type | Notes |
|---|---|---|
| `image` | IMAGE | required, source photo |
| `pixelate_mask` | MASK | optional, each connected region's centroid becomes a pixelation zone |
| `texture` | IMAGE | optional, screen-blended over the canvas; frame 0 is used for every frame in a batch |

Key widgets (all carry tooltips in the node itself):

| Widget | Default | Purpose |
|---|---|---|
| `detection_mode` | combined | how blocks are scored for circle placement |
| `block_size` | 16 | analysis grid resolution |
| `threshold` | 30 | minimum score to place a circle |
| `max_circles` | 80 | cap on placed circles |
| `min_distance` | 40 | spacing between circle centers |
| `seed` | 42 | seeds the circle-size jitter and the procedural grain fallback |
| `connection_dist` | 150 | max distance for a connection line, 0 disables |
| `chain_enabled` / `chain_count` | on / 11 | the tangent chain |
| `crosshair_enabled` | on | dashed cross, frame square, and asterisk |
| `pixel_size` / `pixelate_zones` | 16 / "" | pixelation block size and zone list (`x,y;x,y;...`) |
| `palette` / `size_preset` | whiteOnDark / match input | color scheme and canvas size |
| `image_opacity` / `overlay_opacity` / `texture_opacity` | 0.85 / 1.0 / 0.5 | master opacities per layer |

## Credits and clean-room note

Inspired by Yordan Stoyanov's [Brand Assets Generator](https://brand-generator.stoyanov.works/), specifically its Circles mode. This node is a clean-room reimplementation from observed behavior, no source code was copied. Font bundled is Space Grotesk (OFL), not the original tool's typeface.

MIT license, see `LICENSE`.
