"""
PIL compositor for Schematic Voronoi. OWN render_frame_voronoi — never imports
render.render_frame or render.apply_texture_screen (sibling orchestration,
same rule as render_graph.py / render_construction.py: this module only draws
precomputed `elements`; schematic/voronoi.py owns the geometry, computed by
the node). Implements docs/schematic-voronoi-derivation.md section 5 render
order: bg -> image -> overlay-A (cell fill -> ridges -> seed dots -> frame
text) -> premultiplied resolve -> texture screen-blend -> overlay-B (EMPTY).

numpy + Pillow only (no scipy, no torch) — schematic-derivation.md section 13
substrate conventions, inherited verbatim per the voronoi derivation's header.
scipy is never touched here; it only ever enters sys.modules inside
schematic.voronoi.build_cells (lazy import), which this module does not call.
"""

import numpy as np
from PIL import Image, ImageDraw

from . import engine
from .draw_common import SS, _rgba, _load_font, _resolve_overlay


# ---------------------------------------------------------------------------
# overlay-A element drawers, in derivation section 5 order:
# cell_fill -> ridges -> seed dots -> frame text
# ---------------------------------------------------------------------------

def _draw_cell_fill(draw, regions, areas, field_at_seeds, cell_fill, stroke_rgb, op):
    """5a: each real cell polygon filled at alpha cell_fill * field(seed) * op."""
    if cell_fill <= 0.0 or not regions:
        return
    for i, poly in enumerate(regions):
        if poly.shape[0] < 3 or areas[i] <= 0.0:
            continue
        a = cell_fill * float(field_at_seeds[i]) * op
        if a <= 0.0:
            continue
        fill = _rgba(stroke_rgb, a)
        pts = [(float(vx) * SS, float(vy) * SS) for (vx, vy) in poly]
        draw.polygon(pts, fill=fill)


def _draw_ridges(draw, ridges, alphas, stroke_rgb, op, mesh_weight):
    """5b: stroke color, alpha a*op; width 2 at 2x (3 when a > 0.75), both times
    mesh_weight (multiplies before the 2x scale — see the widget tooltip)."""
    if not ridges:
        return
    for k, ((x1, y1), (x2, y2), _i, _j) in enumerate(ridges):
        a = float(alphas[k])
        fill = _rgba(stroke_rgb, a * op)
        base = 2.0 if a <= 0.75 else 3.0
        width = max(1, int(round(base * mesh_weight)))
        draw.line([(x1 * SS, y1 * SS), (x2 * SS, y2 * SS)], fill=fill, width=width)


def _draw_seed_dots(draw, seeds, stroke_rgb, op):
    """5c: r = 1.5 at 2x, alpha 0.78*op."""
    if seeds is None or len(seeds) == 0:
        return
    fill = _rgba(stroke_rgb, 0.78 * op)
    r = 1.5 * SS
    for (sx, sy) in seeds:
        x, y = float(sx) * SS, float(sy) * SS
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)


def _draw_frame_text(draw, w, h, stroke_rgb, op, params, font_dir):
    """5d: v0.1 section 7.7 verbatim — local sibling copy (see module docstring)."""
    size = params["frame_text_size"]
    inset = 40
    font = _load_font(font_dir, size * SS)
    fill = _rgba(stroke_rgb, op)
    corners = (
        (params["frame_text_tl"], (inset, inset), "la"),
        (params["frame_text_tr"], (w - inset, inset), "ra"),
        (params["frame_text_bl"], (inset, h - inset), "ld"),
        (params["frame_text_br"], (w - inset, h - inset), "rd"),
    )
    for text, pos, anchor in corners:
        if not text:
            continue
        draw.text((pos[0] * SS, pos[1] * SS), text, font=font, fill=fill, anchor=anchor)


# ---------------------------------------------------------------------------
# Full pipeline (derivation section 5)
# ---------------------------------------------------------------------------

def render_frame_voronoi(fitted01, w, h, palette, params, elements, texture01, include_image, font_dir):
    """
    fitted01: (h,w,3) float 0-1 cover-fitted source (same array feeds density + render).
    palette: (bg_rgb, stroke_rgb) uint8 tuples.
    params: dict of widget values - cell_fill, seed_dots, mesh_weight,
      overlay_opacity, image_opacity, texture_opacity, frame_text_size,
      frame_text_tl/tr/bl/br, seed.
    elements: dict with 'regions' (list[n] vertex arrays), 'areas' (n,),
      'ridges' (list of ((x1,y1),(x2,y2),i,j)), 'alphas' (parallel to ridges),
      'seeds' (n,2), 'field_at_seeds' (n,). The degenerate <4-seeds path passes
      all-empty arrays/lists here — this function stays total either way (mesh
      layers simply draw nothing; bg/image/texture/frame text still run).
    texture01: (h,w,3) float 0-1 stretched texture, or None (fallback generated here).
    include_image: whether step 2 (photo composite) runs.
    Returns (uint8 (h,w,3) composite, float32 (h,w) overlay_alpha).
    """
    bg_rgb, stroke_rgb = palette
    op = params["overlay_opacity"]

    # 1. bg fill
    canvas = np.empty((h, w, 3), dtype=np.float64)
    canvas[:, :, 0] = bg_rgb[0] / 255.0
    canvas[:, :, 1] = bg_rgb[1] / 255.0
    canvas[:, :, 2] = bg_rgb[2] / 255.0

    # 2. cover-fit image composite (skipped in overlay_only)
    image_opacity = params["image_opacity"]
    if include_image and image_opacity > 0:
        a = image_opacity
        canvas = canvas * (1.0 - a) + np.clip(fitted01, 0.0, 1.0).astype(np.float64) * a

    # ---- overlay-A: cell_fill -> ridges -> seed dots -> frame text, 2x supersampled ----
    layer_a = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    draw_a = ImageDraw.Draw(layer_a, "RGBA")

    _draw_cell_fill(draw_a, elements["regions"], elements["areas"], elements["field_at_seeds"],
                     params["cell_fill"], stroke_rgb, op)
    _draw_ridges(draw_a, elements["ridges"], elements["alphas"], stroke_rgb, op, params["mesh_weight"])
    if params["seed_dots"]:
        _draw_seed_dots(draw_a, elements["seeds"], stroke_rgb, op)
    _draw_frame_text(draw_a, w, h, stroke_rgb, op, params, font_dir)

    pm_a, alpha_a = _resolve_overlay(layer_a, w, h)
    canvas = canvas * (1.0 - alpha_a[..., None]) + pm_a

    # 3(spec)/4: texture screen-blend — local 4-line sibling copy, NOT
    # render.apply_texture_screen (see module docstring).
    texture_opacity = params["texture_opacity"]
    if texture_opacity > 0:
        tex = texture01 if texture01 is not None else engine.generate_grain(params["seed"] + 1, w, h)
        tex = np.clip(tex, 0.0, 1.0).astype(np.float64)
        screen = 1.0 - (1.0 - canvas) * (1.0 - tex)
        canvas = canvas + (screen - canvas) * texture_opacity

    # ---- overlay-B: EMPTY (two-overlay pattern kept literal; shape-parity slot) ----
    layer_b = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    pm_b, alpha_b = _resolve_overlay(layer_b, w, h)
    canvas = canvas * (1.0 - alpha_b[..., None]) + pm_b

    overlay_alpha = 1.0 - (1.0 - alpha_a) * (1.0 - alpha_b)

    out = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
    return out, overlay_alpha.astype(np.float32)
