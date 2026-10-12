"""
PIL compositor for Schematic Topographic. OWN render_frame_topographic — never
imports render.render_frame (sibling orchestration rule: this module only draws
the precomputed `elements`; schematic/topographic.py owns the geometry, computed
by the node). Implements docs/schematic-topographic-derivation.md section 6:
bg -> image -> hypsometric band raster -> overlay-A (grid -> regular contours ->
hachures -> index contours with label gaps -> index labels -> slope arrows ->
spot heights -> benchmarks -> campsites -> annotations -> marginalia ->
frame text) ->
premultiplied resolve -> texture screen-blend -> (overlay-B slot: EMPTY,
elided -- an empty layer is a bitwise no-op).

numpy + Pillow only (no scipy, no torch).
"""

import math

import numpy as np
from PIL import Image, ImageDraw

from . import engine
from . import render as overlay_render  # ONLY pixelate_canvas, verbatim (the Construction precedent)
from . import topographic as topo
from .draw_common import SS, _rgba, _load_font, _resolve_overlay


def _w(px):
    """Stroke width in supersampled pixels."""
    return max(1, int(round(float(px) * SS)))


def _pts_ss(pts):
    return [(float(x) * SS, float(y) * SS) for (x, y) in pts]


def _text_rotated(layer, cx, cy, text, font, fill, angle_deg):
    """Draw `text` centred at canvas (cx, cy) rotated by angle_deg (CCW, degrees)
    into the supersampled RGBA layer. angle 0 uses the direct 'mm' anchor."""
    if abs(angle_deg) < 1e-9:
        ImageDraw.Draw(layer, "RGBA").text((cx * SS, cy * SS), text, font=font, fill=fill, anchor="mm")
        return
    d0 = ImageDraw.Draw(layer, "RGBA")
    bb = d0.textbbox((0, 0), text, font=font, anchor="la")
    tw = max(1, bb[2] - bb[0] + 4)
    th = max(1, bb[3] - bb[1] + 4)
    sc = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    ImageDraw.Draw(sc, "RGBA").text((tw / 2.0, th / 2.0), text, font=font, fill=fill, anchor="mm")
    sc = sc.rotate(angle_deg, expand=True, resample=Image.BICUBIC)
    ox = int(round(cx * SS - sc.width / 2.0))
    oy = int(round(cy * SS - sc.height / 2.0))
    layer.alpha_composite(sc, (ox, oy))


# ---------------------------------------------------------------------------
# element drawers (section 6 order)
# ---------------------------------------------------------------------------

def _draw_grid(draw, w, h, xs, ys, mode, stroke_rgb, op, weight):
    if mode == "off" or (not xs and not ys):
        return
    width = _w(weight)
    if mode == "lines":
        fill = _rgba(stroke_rgb, 0.35 * op)
        for x in xs:
            draw.line([(x * SS, 0), (x * SS, h * SS)], fill=fill, width=width)
        for y in ys:
            draw.line([(0, y * SS), (w * SS, y * SS)], fill=fill, width=width)
        return
    fill = _rgba(stroke_rgb, 0.7 * op)
    if mode == "dots":
        r = max(1.0, 0.9 * float(weight)) * SS
        for x in xs:
            for y in ys:
                draw.ellipse([x * SS - r, y * SS - r, x * SS + r, y * SS + r], fill=fill)
        return
    tick = 12.0 * SS
    for x in xs:
        draw.line([(x * SS, 0), (x * SS, tick)], fill=fill, width=width)
        draw.line([(x * SS, h * SS - tick), (x * SS, h * SS)], fill=fill, width=width)
    for y in ys:
        draw.line([(0, y * SS), (tick, y * SS)], fill=fill, width=width)
        draw.line([(w * SS - tick, y * SS), (w * SS, y * SS)], fill=fill, width=width)
    if mode == "crosses":
        arm = 6.0 * SS
        for x in xs:
            for y in ys:
                draw.line([(x * SS - arm, y * SS), (x * SS + arm, y * SS)], fill=fill, width=width)
                draw.line([(x * SS, y * SS - arm), (x * SS, y * SS + arm)], fill=fill, width=width)


def _draw_grid_labels(layer, draw, w, h, xs, ys, stroke_rgb, op, label_size, map_scale, font_dir):
    font = _load_font(font_dir, label_size * SS)
    fill = _rgba(stroke_rgb, op)
    clear = 80.0
    for x in xs:
        if x < clear or x > w - clear:
            continue
        txt = str(int(round(x * map_scale)))
        draw.text((x * SS, (h - 6) * SS), txt, font=font, fill=fill, anchor="md")
    for y in ys:
        if y < clear or y > h - clear:
            continue
        txt = str(int(round((h - y) * map_scale)))
        # rotated 90 CCW, centred 6 px + half the text height inside the right edge
        bb = draw.textbbox((0, 0), txt, font=font, anchor="la")
        th = (bb[3] - bb[1]) / SS
        _text_rotated(layer, w - 6 - th / 2.0, y, txt, font, fill, 90)


def _draw_polylines(draw, lines, fill, width):
    for pts in lines:
        if len(pts) < 2:
            continue
        draw.line(_pts_ss(pts), fill=fill, width=width, joint="curve")


def _draw_contours(draw, layer, contours, stroke_rgb, op, params, font_dir):
    """Regular (non-index) contours, under everything else. When contour_text
    runs on ALL lines, the arc under each text run is left open."""
    reg_fill = _rgba(stroke_rgb, 0.55 * op)
    reg_w = _w(params["contour_weight"])
    ctext = str(params.get("contour_text", "")).strip()
    text_all = bool(ctext) and params.get("contour_text_on", "index") == "all"
    advances = None
    if text_all:
        cfont = _load_font(font_dir, float(params.get("contour_text_size", 11)) * SS)
        advances = [cfont.getlength(ch) / SS for ch in ctext + "  "]
        gap = float(params.get("contour_text_gap", 60))
        lead = float(params.get("contour_text_size", 11)) * 1.2
    for lev in contours:
        if lev["index"]:
            continue
        for (pts, closed) in lev["lines"]:
            if text_all:
                _places, gaps = topo.text_runs(pts, closed, advances, gap, lead)
                if gaps:
                    for sub in topo.split_by_gaps(pts, closed, gaps):
                        draw.line(_pts_ss(sub), fill=reg_fill, width=reg_w, joint="curve")
                    continue
            p = topo._closed_pts(pts, closed)
            draw.line(_pts_ss(p), fill=reg_fill, width=reg_w, joint="curve")


def _draw_index_contours(draw, layer, contours, stroke_rgb, op, params, font_dir):
    cw = params["contour_weight"]
    iw = cw * params["index_weight"]
    idx_fill = _rgba(stroke_rgb, op)
    idx_w = _w(iw)
    labels_on = bool(params["contour_labels"])
    label_size = float(params["label_size"])
    spacing = float(params["label_spacing"])
    unit = str(params.get("label_unit", ""))
    font = _load_font(font_dir, label_size * SS) if labels_on else None

    ctext = str(params.get("contour_text", "")).strip()
    ctext_on = params.get("contour_text_on", "index")
    ctext_size = float(params.get("contour_text_size", 11))
    ctext_gap = float(params.get("contour_text_gap", 60))
    cfont = _load_font(font_dir, ctext_size * SS) if ctext else None
    advances = None
    if ctext:
        rep = ctext + "  "
        advances = [cfont.getlength(ch) / SS for ch in rep]

    labels = []
    glyphs = []
    for lev in contours:
        is_idx = lev["index"]
        text_here = bool(ctext) and (ctext_on == "all" or is_idx)
        if not is_idx and not text_here:
            continue
        text = str(int(round(lev["z"]))) + unit
        fill = idx_fill if is_idx else _rgba(stroke_rgb, 0.55 * op)
        width = idx_w if is_idx else _w(cw)
        for (pts, closed) in lev["lines"]:
            L = topo.polyline_length(pts, closed)
            gaps = []
            if text_here:
                places, tgaps = topo.text_runs(pts, closed, advances, ctext_gap, ctext_size * 1.2)
                if places:
                    glyphs.append((places, fill))
                    gaps.extend(tgaps)
            if is_idx and labels_on and spacing > 0 and L >= spacing and not gaps:
                tw = font.getlength(text) / SS
                half = tw / 2.0 + 0.6 * label_size
                s = spacing / 2.0
                while s + spacing / 2.0 <= L + 1e-9:
                    x, y, _tx, _ty = topo.point_at_arc(pts, s, closed)
                    d = max(1.0, label_size)
                    xa, ya, _, _ = topo.point_at_arc(pts, s - d, closed)
                    xb, yb, _, _ = topo.point_at_arc(pts, s + d, closed)
                    tx, ty = xb - xa, yb - ya
                    n = math.hypot(tx, ty)
                    if n <= 0:
                        tx, ty = 1.0, 0.0
                    else:
                        tx, ty = tx / n, ty / n
                    if tx < 0:
                        tx, ty = -tx, -ty
                    angle = -math.degrees(math.atan2(ty, tx))  # PIL rotates CCW; image y is down
                    labels.append((x, y, text, angle))
                    gaps.append((s - half, s + half))
                    s += spacing
            if not is_idx:
                # non-index line carrying text: it was already drawn as a regular
                # contour; redraw is not needed, the gaps are cut by the text
                # itself being drawn over it (regular alpha stays under glyphs)
                continue
            if gaps:
                for sub in topo.split_by_gaps(pts, closed, gaps):
                    draw.line(_pts_ss(sub), fill=fill, width=width, joint="curve")
            else:
                p = topo._closed_pts(pts, closed)
                draw.line(_pts_ss(p), fill=fill, width=width, joint="curve")
    for (x, y, text, angle) in labels:
        _text_rotated(layer, x, y, text, font, idx_fill, angle)
    if glyphs:
        rep = ctext + "  "
        for (places, fill) in glyphs:
            for k, (x, y, ang) in enumerate(places):
                ch = rep[k % len(rep)]
                if ch == " ":
                    continue
                _text_rotated(layer, x, y, ch, cfont, fill, ang)


def _draw_ticks(draw, ticks, stroke_rgb, op, weight):
    if not ticks:
        return
    fill = _rgba(stroke_rgb, 0.55 * op)
    width = _w(weight)
    for ((x0, y0), (x1, y1)) in ticks:
        draw.line([(x0 * SS, y0 * SS), (x1 * SS, y1 * SS)], fill=fill, width=width)


def _draw_arrows(draw, arrows, stroke_rgb, op, weight):
    if not arrows:
        return
    fill = _rgba(stroke_rgb, 0.7 * op)
    width = _w(weight)
    ang = math.radians(28.0)
    for ((x0, y0), (x1, y1), L) in arrows:
        draw.line([(x0 * SS, y0 * SS), (x1 * SS, y1 * SS)], fill=fill, width=width)
        dx, dy = x1 - x0, y1 - y0
        n = math.hypot(dx, dy)
        if n <= 0:
            continue
        ux, uy = dx / n, dy / n
        hl = 0.35 * L
        for sgn in (1.0, -1.0):
            c, s_ = math.cos(ang), math.sin(ang) * sgn
            bx = -ux * c + uy * s_
            by = -uy * c - ux * s_
            draw.line([(x1 * SS, y1 * SS), ((x1 + hl * bx) * SS, (y1 + hl * by) * SS)],
                      fill=fill, width=width)


def _draw_spots(draw, spots, stroke_rgb, op, params, font_dir):
    if not spots:
        return
    fill = _rgba(stroke_rgb, op)
    size = float(params["symbol_size"])
    width = _w(params["contour_weight"])
    font = _load_font(font_dir, params["label_size"] * SS)
    for (x, y, z, kind) in spots:
        if kind == "peak":
            r = size / math.sqrt(3.0)   # circumradius of a triangle of side `size`
            pts = [(x, y - r), (x + size / 2.0, y + r / 2.0), (x - size / 2.0, y + r / 2.0)]
            draw.polygon(_pts_ss(pts), fill=fill)
        else:
            r = size / 2.0
            draw.ellipse([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS],
                         outline=fill, width=width)
        draw.text(((x + size * 0.8 + 2) * SS, y * SS), str(int(round(z))), font=font,
                  fill=fill, anchor="lm")


def _draw_benchmarks(draw, bms, stroke_rgb, op, params, font_dir):
    if not bms:
        return
    fill = _rgba(stroke_rgb, op)
    size = float(params["symbol_size"])
    r = size / 2.0
    width = _w(params["contour_weight"])
    font = _load_font(font_dir, params["label_size"] * SS)
    for (x, y, z) in bms:
        draw.line([((x - r) * SS, (y - r) * SS), ((x + r) * SS, (y + r) * SS)], fill=fill, width=width)
        draw.line([((x - r) * SS, (y + r) * SS), ((x + r) * SS, (y - r) * SS)], fill=fill, width=width)
        draw.text(((x + size * 0.8 + 2) * SS, y * SS), "BM " + str(int(round(z))), font=font,
                  fill=fill, anchor="lm")


# CanvasCamp logo symbol, transcribed from CanvasCamp/_lib/_Gen/Logo/
# Canvascamp001_LogoSymbol_ReDesign_Vector_Model_v001.svg (viewBox 0 0 1024 1024).
# The SVG carries the mark only: the disc's centre and radius are measured off
# the brand lockup in the same frame. The (R) is dropped (sub-pixel at map scale).
_CC_DISC = (512.0, 526.1, 104.6)
_CC_STRIPE = [(511.9, 481.2), (496.1, 497.0), (555.4, 556.3), (586.9, 556.3)]


def _cubic(p0, p1, p2, p3, n=24):
    out = []
    for k in range(1, n + 1):
        t = k / n
        a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t ** 3
        out.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
                    a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return out


# left mark: M437,556.2 h4.3 c17.7,0,34.7-7,47.3-19.6 c12.5,12.5,29.5,19.6,47.3,19.6
# h4.3 l-51.5-51.5 -51.5,51.5 Z (relative commands resolved to absolute)
_CC_PEAK = ([(437.0, 556.2), (441.3, 556.2)]
            + _cubic((441.3, 556.2), (459.0, 556.2), (476.0, 549.2), (488.6, 536.6))
            + _cubic((488.6, 536.6), (501.1, 549.1), (518.1, 556.2), (535.9, 556.2))
            + [(540.2, 556.2), (488.7, 504.7), (437.2, 556.2)])


def _cc_unit(pts):
    """SVG frame -> unit-disc frame (disc centre at 0, radius 1)."""
    cx, cy, r = _CC_DISC
    return [((x - cx) / r, (y - cy) / r) for (x, y) in pts]


CC_MARK = (_cc_unit(_CC_PEAK), _cc_unit(_CC_STRIPE))


def _draw_campsites(layer, camps, stroke_rgb, bg_rgb, op, params):
    """CanvasCamp badge per site: background-colour disc carrying the mark in
    the stroke colour, a stroke-colour ring on the rim (6% of the diameter,
    min 1 px) and a thin background halo outside it so contours never touch
    the ring. Each badge is built opaque on its own tile and composited at
    `op`, so the disc stays clean at partial overlay_opacity."""
    if not camps or op <= 0.0:
        return
    size = float(params["campsite_size"])
    r = size / 2.0 * SS
    ring = max(1.0, 0.06 * size) * SS
    halo = max(1.0, float(params["contour_weight"])) * SS
    R = r + halo
    pad = int(math.ceil(R)) + 2
    a_op = int(round(np.clip(op, 0.0, 1.0) * 255))
    ink = (int(stroke_rgb[0]), int(stroke_rgb[1]), int(stroke_rgb[2]), 255)
    ground = (int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2]), 255)
    for (x, y) in camps:
        X, Y = x * SS, y * SS
        ox, oy = int(math.floor(X)) - pad, int(math.floor(Y)) - pad
        cx, cy = X - ox, Y - oy
        tile = Image.new("RGBA", (2 * pad + 1, 2 * pad + 1), (0, 0, 0, 0))
        td = ImageDraw.Draw(tile)
        td.ellipse([cx - R, cy - R, cx + R, cy + R], fill=ground)
        td.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ink)
        ri = r - ring
        td.ellipse([cx - ri, cy - ri, cx + ri, cy + ri], fill=ground)
        for poly in CC_MARK:
            td.polygon([(cx + u * r, cy + v * r) for (u, v) in poly], fill=ink)
        if a_op < 255:
            alpha = tile.getchannel("A").point(lambda a: (a * a_op + 127) // 255)
            tile.putalpha(alpha)
        layer.alpha_composite(tile, (ox, oy))


def _draw_annotations(draw, w, notes, stroke_rgb, op, params, font_dir):
    if not notes:
        return
    fill = _rgba(stroke_rgb, op)
    ms = float(params["marginalia_scale"])
    width = _w(params["contour_weight"])
    font = _load_font(font_dir, params["label_size"] * 1.2 * SS)
    dot = 2.5 * SS
    up = 28.0 * ms
    tail = 36.0 * ms
    for (x, y, text) in notes:
        draw.ellipse([x * SS - dot, y * SS - dot, x * SS + dot, y * SS + dot], fill=fill)
        sgn = -1.0 if x > 0.7 * w else 1.0
        ex, ey = x + sgn * up, y - up
        tx = ex + sgn * tail
        draw.line([(x * SS, y * SS), (ex * SS, ey * SS), (tx * SS, ey * SS)], fill=fill, width=width)
        anchor = "lm" if sgn > 0 else "rm"
        draw.text(((tx + sgn * 6) * SS, ey * SS), text, font=font, fill=fill, anchor=anchor)


def _draw_marginalia(draw, w, h, stroke_rgb, op, params, elements, font_dir):
    fill = _rgba(stroke_rgb, op)
    ms = float(params["marginalia_scale"])
    width = _w(params["contour_weight"])
    label_size = float(params["label_size"])
    fts = float(params["frame_text_size"])
    font = _load_font(font_dir, label_size * SS)

    if params["north_arrow"]:
        cx = w - 40 - 10 * ms
        cy = 40 + fts * 1.6 + 28 * ms
        shaft = 44 * ms
        tip_y = cy - shaft / 2.0
        base_y = cy + shaft / 2.0
        draw.line([(cx * SS, base_y * SS), (cx * SS, (tip_y + 14 * ms) * SS)], fill=fill, width=width)
        draw.polygon([(cx * SS, tip_y * SS), ((cx - 6 * ms) * SS, (tip_y + 14 * ms) * SS),
                      ((cx + 6 * ms) * SS, (tip_y + 14 * ms) * SS)], fill=fill)
        nfont = _load_font(font_dir, label_size * 1.2 * SS)
        draw.text((cx * SS, (tip_y - 10 * ms) * SS), "N", font=nfont, fill=fill, anchor="mm")

    bar_base = h - 40 - fts * 1.6 - 14 * ms
    if params["scale_bar"] and elements.get("scale_bar") is not None:
        metres, bar_px = elements["scale_bar"]
        if bar_px > 0:
            x0 = 40.0
            bh = 6 * ms
            seg = bar_px / 4.0
            for k in range(4):
                xa, xb = x0 + k * seg, x0 + (k + 1) * seg
                if k % 2 == 0:
                    draw.rectangle([xa * SS, (bar_base - bh) * SS, xb * SS, bar_base * SS], fill=fill)
                else:
                    draw.rectangle([xa * SS, (bar_base - bh) * SS, xb * SS, bar_base * SS],
                                   outline=fill, width=width)
            for xt in (x0, x0 + bar_px):
                draw.line([(xt * SS, (bar_base - bh - 4 * ms) * SS), (xt * SS, bar_base * SS)],
                          fill=fill, width=width)
            if metres >= 1000:
                unit = f"{metres / 1000.0:g} km"
            else:
                unit = f"{metres:g} m"
            draw.text((x0 * SS, (bar_base - bh - 6 * ms) * SS), "0", font=font, fill=fill, anchor="ld")
            draw.text(((x0 + bar_px) * SS, (bar_base - bh - 6 * ms) * SS), unit, font=font,
                      fill=fill, anchor="rd")

    if params["interval_note"]:
        note = f"CONTOUR INTERVAL {float(elements['interval']):g} M"
        if params["scale_bar"] and elements.get("scale_bar") is not None:
            ny = bar_base + 6 * ms + label_size * 1.4
        else:
            ny = h - 40 - fts * 1.6
        draw.text((40 * SS, ny * SS), note, font=font, fill=fill, anchor="ld")


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
# Full pipeline (section 6)
# ---------------------------------------------------------------------------

def render_frame_topographic(fitted01, w, h, palette, params, elements, texture01, include_image, font_dir,
                             cache=None):
    """
    fitted01: (h,w,3) float 0-1 cover-fitted source.
    palette: (bg_rgb, stroke_rgb) uint8 tuples.
    params: widget dict (see nodes/schematic_topographic.py).
    elements: dict with 'H' (h,w) or None, 'n_levels', 'contours', 'hachures',
      'arrows', 'spots', 'benchmarks', 'grid' (xs, ys), 'annotations',
      'scale_bar' (metres, px) or None, 'interval'.
    cache: optional dict shared by the image and overlay_only passes of ONE
      frame; the band raster and the resolved vector layer depend on the
      elements only (never on the photo), so the second pass reuses them
      bitwise instead of redrawing.
    Returns (uint8 (h,w,3) composite, float32 (h,w) overlay_alpha).
    """
    bg_rgb, stroke_rgb = palette
    op = float(params["overlay_opacity"])
    if cache is None:
        cache = {}

    canvas = np.empty((h, w, 3), dtype=np.float64)
    canvas[:, :, 0] = bg_rgb[0] / 255.0
    canvas[:, :, 1] = bg_rgb[1] / 255.0
    canvas[:, :, 2] = bg_rgb[2] / 255.0

    image_opacity = params["image_opacity"]
    if include_image and image_opacity > 0:
        a = image_opacity
        canvas = canvas * (1.0 - a) + np.clip(fitted01, 0.0, 1.0).astype(np.float64) * a

    # 2b. pixelate zones: Overlay's own raster step, VERBATIM, on the current
    # canvas (bg + photo). Alters the underlying image inside each zone; the
    # outline + label draw in both passes exactly as Overlay does.
    zones = elements.get("zones") or []
    pixel_size = int(params.get("pixel_size", 1))
    if pixel_size > 1 and zones:
        canvas = overlay_render.pixelate_canvas(canvas, w, h, zones, pixel_size, params["zone_size"],
                                                params["pixel_stroke"], stroke_rgb, op,
                                                params["label_size"], font_dir)

    # 3. hypsometric band raster
    if "alpha_band" not in cache:
        alpha_band = np.zeros((h, w), dtype=np.float64)
        band_fill = float(params["band_fill"])
        if band_fill > 0.0 and elements.get("H") is not None and op > 0.0:
            q = topo.band_quantised(elements["H"], elements["n_levels"])
            alpha_band = band_fill * q * op
        cache["alpha_band"] = alpha_band
    alpha_band = cache["alpha_band"]
    if np.any(alpha_band > 0.0):
        stroke = np.array([stroke_rgb[0], stroke_rgb[1], stroke_rgb[2]], dtype=np.float64) / 255.0
        canvas = canvas * (1.0 - alpha_band[..., None]) + stroke[None, None, :] * alpha_band[..., None]

    # ---- overlay-A (drawn once per frame, reused by the overlay_only pass) ----
    if "pm_a" not in cache:
        layer_a = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
        draw_a = ImageDraw.Draw(layer_a, "RGBA")

        xs, ys = elements["grid"]
        _draw_grid(draw_a, w, h, xs, ys, params["grid"], stroke_rgb, op, params["grid_weight"])
        _draw_contours(draw_a, layer_a, elements["contours"], stroke_rgb, op, params, font_dir)
        _draw_ticks(draw_a, elements["hachures"], stroke_rgb, op, params["contour_weight"])
        _draw_index_contours(draw_a, layer_a, elements["contours"], stroke_rgb, op, params, font_dir)
        _draw_arrows(draw_a, elements["arrows"], stroke_rgb, op, params["contour_weight"])
        _draw_spots(draw_a, elements["spots"], stroke_rgb, op, params, font_dir)
        _draw_benchmarks(draw_a, elements["benchmarks"], stroke_rgb, op, params, font_dir)
        _draw_campsites(layer_a, elements.get("campsites") or [], stroke_rgb, bg_rgb, op, params)
        _draw_annotations(draw_a, w, elements["annotations"], stroke_rgb, op, params, font_dir)
        if params["grid"] != "off" and params["grid_labels"]:
            _draw_grid_labels(layer_a, draw_a, w, h, xs, ys, stroke_rgb, op, params["label_size"],
                              params["map_scale"], font_dir)
        _draw_marginalia(draw_a, w, h, stroke_rgb, op, params, elements, font_dir)
        _draw_frame_text(draw_a, w, h, stroke_rgb, op, params, font_dir)
        cache["pm_a"], cache["alpha_a"] = _resolve_overlay(layer_a, w, h)
    pm_a, alpha_a = cache["pm_a"], cache["alpha_a"]
    canvas = canvas * (1.0 - alpha_a[..., None]) + pm_a

    # texture screen-blend (local sibling copy)
    texture_opacity = params["texture_opacity"]
    if texture_opacity > 0:
        tex = texture01 if texture01 is not None else engine.generate_grain(params["seed"] + 1, w, h)
        tex = np.clip(tex, 0.0, 1.0).astype(np.float64)
        screen = 1.0 - (1.0 - canvas) * (1.0 - tex)
        canvas = canvas + (screen - canvas) * texture_opacity

    # ---- overlay-B: EMPTY in this node. The sibling two-overlay pattern keeps a
    # post-texture vector slot; here nothing draws after the texture, and an
    # empty layer resolves to alpha 0 (canvas*(1-0)+0 == canvas bitwise), so the
    # 2x resolve is elided rather than paid for nothing. ----
    overlay_alpha = 1.0 - (1.0 - alpha_band) * (1.0 - alpha_a)

    out = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
    return out, overlay_alpha.astype(np.float32)
