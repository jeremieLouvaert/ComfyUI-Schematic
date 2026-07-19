"""
PIL compositor for Schematic Overlay. Implements spec sections 7-9 and 13
exactly: two 2x-supersampled RGBA vector overlays (overlay-A = steps 4-7,
overlay-B = step 9, composited after texture), manual dash walker, numpy
screen blend, procedural-grain fallback, pack-relative font loading.

numpy + Pillow only (no scipy, no torch) — see docs/schematic-derivation.md
section 13.
"""

import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import engine

SS = 2  # supersample factor for vector overlays
FONT_REL_PATH = os.path.join("fonts", "SpaceGrotesk-Regular.ttf")

_FONT_CACHE = {}
_FONT_WARNED = False


def _load_font(font_dir, size_px):
    global _FONT_WARNED
    size_px = max(1, int(round(size_px)))
    key = (font_dir, size_px)
    cached = _FONT_CACHE.get(key)
    if cached is not None:
        return cached
    path = os.path.join(font_dir, FONT_REL_PATH) if font_dir else FONT_REL_PATH
    try:
        font = ImageFont.truetype(path, size_px)
    except Exception:
        if not _FONT_WARNED:
            print(f"[Schematic] warning: could not load font at {path}, falling back to PIL default font")
            _FONT_WARNED = True
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _rgba(stroke_rgb, alpha01):
    a = int(round(np.clip(alpha01, 0.0, 1.0) * 255))
    return (int(stroke_rgb[0]), int(stroke_rgb[1]), int(stroke_rgb[2]), a)


def _dashed_line(draw, x0, y0, x1, y1, dash_on, dash_off, fill, width):
    """Manual dash-walking helper — PIL has no native dash support."""
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0:
        return
    period = dash_on + dash_off
    if period <= 0:
        draw.line([(x0, y0), (x1, y1)], fill=fill, width=width)
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    pos = 0.0
    while pos < length:
        seg_end = min(pos + dash_on, length)
        sx, sy = x0 + ux * pos, y0 + uy * pos
        ex, ey = x0 + ux * seg_end, y0 + uy * seg_end
        draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)
        pos += period


def _line_round_cap(draw, x0, y0, x1, y1, fill, width):
    draw.line([(x0, y0), (x1, y1)], fill=fill, width=width)
    r = width / 2.0
    draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=fill)
    draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=fill)


# ---------------------------------------------------------------------------
# Step 4: crosshair
# ---------------------------------------------------------------------------

def _draw_crosshair(draw, w, h, stroke_rgb, op, params):
    stroke = params["crosshair_stroke"]
    dash = params["crosshair_dash"]
    frame_pct = params["crosshair_frame_size"]
    star_size = params["crosshair_star_size"]
    star_points = params["crosshair_star_points"]

    fill = _rgba(stroke_rgb, op)
    width = max(1, int(round(stroke * SS)))
    dash_on = dash_off = dash * SS

    cx, cy = w / 2.0, h / 2.0
    _dashed_line(draw, 0, cy * SS, w * SS, cy * SS, dash_on, dash_off, fill, width)
    _dashed_line(draw, cx * SS, 0, cx * SS, h * SS, dash_on, dash_off, fill, width)

    a = min(w, h) * frame_pct / 100.0
    x0, y0 = (cx - a / 2.0) * SS, (cy - a / 2.0) * SS
    x1, y1 = (cx + a / 2.0) * SS, (cy + a / 2.0) * SS
    _dashed_line(draw, x0, y0, x1, y0, dash_on, dash_off, fill, width)
    _dashed_line(draw, x1, y0, x1, y1, dash_on, dash_off, fill, width)
    _dashed_line(draw, x1, y1, x0, y1, dash_on, dash_off, fill, width)
    _dashed_line(draw, x0, y1, x0, y0, dash_on, dash_off, fill, width)

    if star_size > 0:
        l = star_size / 2.0
        for i in range(star_points):
            theta = (i / star_points) * math.pi
            dx, dy = math.cos(theta), math.sin(theta)
            sx, sy = (cx - dx * l) * SS, (cy - dy * l) * SS
            ex, ey = (cx + dx * l) * SS, (cy + dy * l) * SS
            draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)


# ---------------------------------------------------------------------------
# Step 5: connections
# ---------------------------------------------------------------------------

def _draw_connections(draw, connections, circles, stroke_rgb, op, line_weight):
    if not connections:
        return
    fill = _rgba(stroke_rgb, op)
    width = max(1, int(round(line_weight * SS)))
    for (i, j) in connections:
        x0, y0 = circles[i][0] * SS, circles[i][1] * SS
        x1, y1 = circles[j][0] * SS, circles[j][1] * SS
        _line_round_cap(draw, x0, y0, x1, y1, fill, width)


# ---------------------------------------------------------------------------
# Step 6: detection circles
# ---------------------------------------------------------------------------

def _draw_circles(draw, circles, stroke_rgb, op, shape, circle_stroke, label_size, font_dir):
    if not circles:
        return
    outline_w = max(1, int(round(circle_stroke * SS)))
    font = _load_font(font_dir, label_size * SS)
    for (x, y, r, score) in circles:
        outline_fill = _rgba(stroke_rgb, (0.3 + 0.4 * score) * op)
        dot_fill = _rgba(stroke_rgb, (0.7 + 0.3 * score) * op)
        text_fill = _rgba(stroke_rgb, op)

        sx, sy, sr = x * SS, y * SS, r * SS
        if shape == "square":
            draw.rectangle([sx - sr, sy - sr, sx + sr, sy + sr], outline=outline_fill, width=outline_w)
        else:
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], outline=outline_fill, width=outline_w)

        dr = 2.5 * SS
        draw.ellipse([sx - dr, sy - dr, sx + dr, sy + dr], fill=dot_fill)

        label = f"{round(x)},{round(y)}"
        lx, ly = (x + r + label_size * 0.4) * SS, y * SS
        draw.text((lx, ly), label, font=font, fill=text_fill, anchor="lm")


# ---------------------------------------------------------------------------
# Step 7: frame text
# ---------------------------------------------------------------------------

def _draw_frame_text(draw, w, h, stroke_rgb, op, params, font_dir):
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
# Step 3: pixelate zones (raster, 1x)
# ---------------------------------------------------------------------------

def pixelate_canvas(canvas01, w, h, zones, pixel_size, zone_size, pixel_stroke, stroke_rgb, op, label_size, font_dir):
    """Mosaic + optional outline/label. Runs on the CURRENT canvas (bg+image only). Spec section 8."""
    canvas = canvas01.copy()
    rects = []
    for (cx, cy) in zones:
        x0 = max(0, int(math.floor(cx - zone_size)))
        y0 = max(0, int(math.floor(cy - zone_size)))
        x1 = min(w, int(math.ceil(cx + zone_size)))
        y1 = min(h, int(math.ceil(cy + zone_size)))
        if x1 <= x0 or y1 <= y0:
            continue
        rects.append((cx, cy, x0, y0, x1, y1))
        for ty in range(y0, y1, pixel_size):
            ty1 = min(ty + pixel_size, y1)
            for tx in range(x0, x1, pixel_size):
                tx1 = min(tx + pixel_size, x1)
                tile = canvas[ty:ty1, tx:tx1, :]
                mean255 = np.floor(tile.reshape(-1, 3).mean(axis=0) * 255.0)
                canvas[ty:ty1, tx:tx1, :] = mean255 / 255.0

    if not rects:
        return canvas

    img = Image.fromarray(np.clip(canvas * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    font = _load_font(font_dir, label_size)
    for (cx, cy, x0, y0, x1, y1) in rects:
        if pixel_stroke:
            draw.rectangle([x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)], outline=_rgba(stroke_rgb, 0.4), width=1)
        label = f"{round(cx)},{round(cy)}"
        draw.text(((x0 + x1) / 2.0, (y0 + y1) / 2.0), label, font=font, fill=_rgba(stroke_rgb, op), anchor="mm")

    return np.asarray(img).astype(np.float64) / 255.0


# ---------------------------------------------------------------------------
# Step 8: texture screen blend
# ---------------------------------------------------------------------------

def apply_texture_screen(canvas01, texture01, opacity):
    """out = base + (screen - base) * opacity, screen = 1-(1-a)(1-b), literal on 0-1 values."""
    tex = np.clip(texture01, 0.0, 1.0).astype(np.float64)
    base = canvas01.astype(np.float64)
    screen = 1.0 - (1.0 - base) * (1.0 - tex)
    return base + (screen - base) * opacity


# ---------------------------------------------------------------------------
# Step 9: chain (overlay-B, drawn after texture)
# ---------------------------------------------------------------------------

def _draw_chain(draw, chain, intersections, stroke_rgb, op, w, h, params, font_dir):
    circle_stroke = params["circle_stroke"]
    label_size = params["label_size"]
    s = min(w, h)
    outline_w = max(1, int(round(circle_stroke * SS)))
    font = _load_font(font_dir, label_size * SS)

    outline_fill = _rgba(stroke_rgb, 0.5 * op)
    tick_fill = _rgba(stroke_rgb, 0.3 * op)
    dot_fill = _rgba(stroke_rgb, 0.6 * op)
    text_fill = _rgba(stroke_rgb, op)

    tick_width = max(1, int(round(max(0.5, s * 6e-4) * SS)))
    dash_len = max(0.5, s * 0.004) * SS
    dot_r = max(1.5, s * 0.002) * SS

    for (x, y, r) in chain:
        sx, sy, sr = x * SS, y * SS, r * SS
        draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], outline=outline_fill, width=outline_w)

        arm = max(8.0, r * 0.3)
        sarm = arm * SS
        _dashed_line(draw, sx - sarm, sy, sx + sarm, sy, dash_len, dash_len, tick_fill, tick_width)
        _dashed_line(draw, sx, sy - sarm, sx, sy + sarm, dash_len, dash_len, tick_fill, tick_width)

        draw.ellipse([sx - dot_r, sy - dot_r, sx + dot_r, sy + dot_r], fill=dot_fill)

        label = f"{round(x)},{round(y)}"
        lx, ly = (x + arm + label_size * 0.4) * SS, y * SS
        draw.text((lx, ly), label, font=font, fill=text_fill, anchor="lm")

    marker_r = params["chain_intersection_size"]
    n = 1
    for (ix, iy) in intersections:
        sx, sy = ix * SS, iy * SS
        smr = marker_r * SS
        draw.ellipse([sx - smr, sy - smr, sx + smr, sy + smr], fill=text_fill)
        label = f"{n} → {round(ix)} – {round(iy)}"
        lx, ly = (ix + marker_r + label_size * 0.5) * SS, iy * SS
        draw.text((lx, ly), label, font=font, fill=text_fill, anchor="lm")
        n += 1


# ---------------------------------------------------------------------------
# Full pipeline (spec section 7, order 1-9; section 13 for supersampling)
# ---------------------------------------------------------------------------

def _resolve_overlay(layer, w, h):
    """
    SSxSS box-resolve of a supersampled RGBA layer with PREMULTIPLIED alpha.
    Straight-alpha resizing bleeds the transparent pixels' black into stroke
    edges (thin lines are mostly edge pixels), dimming them; premultiplication
    is the mathematically correct SSAA resolve.
    Returns (premultiplied_rgb (h,w,3), alpha (h,w)), both float64 0-1;
    composite as: canvas*(1-alpha) + premultiplied_rgb.
    """
    arr = np.asarray(layer, dtype=np.float64) / 255.0
    alpha = arr[..., 3]
    pm = arr[..., :3] * alpha[..., None]
    pm = pm.reshape(h, SS, w, SS, 3).mean(axis=(1, 3))
    alpha = alpha.reshape(h, SS, w, SS).mean(axis=(1, 3))
    return pm, alpha


def render_frame(fitted01, w, h, palette, params, elements, texture01, include_image, font_dir):
    """
    fitted01: (h,w,3) float 0-1 cover-fitted source (same array feeds analysis + render).
    palette: (bg_rgb, stroke_rgb) uint8 tuples.
    params: dict of widget values (see nodes/schematic_overlay.py for keys).
    elements: dict with 'circles', 'connections', 'zones', 'chain', 'intersections'.
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

    # 2. image composite
    image_opacity = params["image_opacity"]
    if include_image and image_opacity > 0:
        a = image_opacity
        canvas = canvas * (1.0 - a) + np.clip(fitted01, 0.0, 1.0).astype(np.float64) * a

    # 3. pixelate zones (raster, current canvas = bg+image only)
    zones = elements["zones"]
    pixel_size = params["pixel_size"]
    if pixel_size > 1 and len(zones) > 0:
        canvas = pixelate_canvas(canvas, w, h, zones, pixel_size, params["zone_size"],
                                  params["pixel_stroke"], stroke_rgb, op, params["label_size"], font_dir)

    # ---- overlay-A: steps 4-7, 2x supersampled ----
    # "RGBA" draw mode = source-over within the layer (canvas semantics at stroke
    # crossings); resolved to 1x with premultiplied alpha to avoid dark fringing.
    layer_a = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    draw_a = ImageDraw.Draw(layer_a, "RGBA")

    if params["crosshair_enabled"]:
        _draw_crosshair(draw_a, w, h, stroke_rgb, op, params)

    _draw_connections(draw_a, elements["connections"], elements["circles"], stroke_rgb, op, params["line_weight"])
    _draw_circles(draw_a, elements["circles"], stroke_rgb, op, params["shape"], params["circle_stroke"],
                  params["label_size"], font_dir)
    _draw_frame_text(draw_a, w, h, stroke_rgb, op, params, font_dir)

    pm_a, alpha_a = _resolve_overlay(layer_a, w, h)
    canvas = canvas * (1.0 - alpha_a[..., None]) + pm_a

    # 8. texture (screen blend), before chain per render order
    texture_opacity = params["texture_opacity"]
    if texture_opacity > 0:
        tex = texture01
        if tex is None:
            tex = engine.generate_grain(params["seed"] + 1, w, h)
        canvas = apply_texture_screen(canvas, tex, texture_opacity)

    # ---- overlay-B: step 9 (chain), drawn ON TOP of texture ----
    layer_b = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    draw_b = ImageDraw.Draw(layer_b, "RGBA")
    chain = elements["chain"]
    if chain:
        _draw_chain(draw_b, chain, elements["intersections"], stroke_rgb, op, w, h, params, font_dir)

    pm_b, alpha_b = _resolve_overlay(layer_b, w, h)
    canvas = canvas * (1.0 - alpha_b[..., None]) + pm_b

    overlay_alpha = 1.0 - (1.0 - alpha_a) * (1.0 - alpha_b)

    out = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
    return out, overlay_alpha.astype(np.float32)
