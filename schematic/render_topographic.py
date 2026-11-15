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


def _w(px, S):
    """Stroke width in layer pixels (S = layer px per layout unit)."""
    return max(1, int(round(float(px) * S)))


def _pts_ss(pts, S):
    return [(float(x) * S, float(y) * S) for (x, y) in pts]


def _text_rotated(layer, cx, cy, text, font, fill, angle_deg, S):
    """Draw `text` centred at canvas (cx, cy) rotated by angle_deg (CCW, degrees)
    into the supersampled RGBA layer. angle 0 uses the direct 'mm' anchor."""
    if abs(angle_deg) < 1e-9:
        ImageDraw.Draw(layer, "RGBA").text((cx * S, cy * S), text, font=font, fill=fill, anchor="mm")
        return
    d0 = ImageDraw.Draw(layer, "RGBA")
    bb = d0.textbbox((0, 0), text, font=font, anchor="la")
    tw = max(1, bb[2] - bb[0] + 4)
    th = max(1, bb[3] - bb[1] + 4)
    sc = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    ImageDraw.Draw(sc, "RGBA").text((tw / 2.0, th / 2.0), text, font=font, fill=fill, anchor="mm")
    sc = sc.rotate(angle_deg, expand=True, resample=Image.BICUBIC)
    ox = int(round(cx * S - sc.width / 2.0))
    oy = int(round(cy * S - sc.height / 2.0))
    layer.alpha_composite(sc, (ox, oy))


# ---------------------------------------------------------------------------
# element drawers (section 6 order)
# ---------------------------------------------------------------------------

def _draw_grid(draw, w, h, xs, ys, mode, stroke_rgb, op, weight, S):
    if mode == "off" or (not xs and not ys):
        return
    width = _w(weight, S)
    if mode == "lines":
        fill = _rgba(stroke_rgb, 0.35 * op)
        for x in xs:
            draw.line([(x * S, 0), (x * S, h * S)], fill=fill, width=width)
        for y in ys:
            draw.line([(0, y * S), (w * S, y * S)], fill=fill, width=width)
        return
    fill = _rgba(stroke_rgb, 0.7 * op)
    if mode == "dots":
        r = max(1.0, 0.9 * float(weight)) * S
        for x in xs:
            for y in ys:
                draw.ellipse([x * S - r, y * S - r, x * S + r, y * S + r], fill=fill)
        return
    tick = 12.0 * S
    for x in xs:
        draw.line([(x * S, 0), (x * S, tick)], fill=fill, width=width)
        draw.line([(x * S, h * S - tick), (x * S, h * S)], fill=fill, width=width)
    for y in ys:
        draw.line([(0, y * S), (tick, y * S)], fill=fill, width=width)
        draw.line([(w * S - tick, y * S), (w * S, y * S)], fill=fill, width=width)
    if mode == "crosses":
        arm = 6.0 * S
        for x in xs:
            for y in ys:
                draw.line([(x * S - arm, y * S), (x * S + arm, y * S)], fill=fill, width=width)
                draw.line([(x * S, y * S - arm), (x * S, y * S + arm)], fill=fill, width=width)


def _draw_grid_labels(layer, draw, w, h, xs, ys, stroke_rgb, op, label_size, map_scale, font_dir, S):
    font = _load_font(font_dir, label_size * S)
    fill = _rgba(stroke_rgb, op)
    clear = 80.0
    for x in xs:
        if x < clear or x > w - clear:
            continue
        txt = str(int(round(x * map_scale)))
        draw.text((x * S, (h - 6) * S), txt, font=font, fill=fill, anchor="md")
    for y in ys:
        if y < clear or y > h - clear:
            continue
        txt = str(int(round((h - y) * map_scale)))
        # rotated 90 CCW, centred 6 px + half the text height inside the right edge
        bb = draw.textbbox((0, 0), txt, font=font, anchor="la")
        th = (bb[3] - bb[1]) / S
        _text_rotated(layer, w - 6 - th / 2.0, y, txt, font, fill, 90, S)


def _draw_polylines(draw, lines, fill, width, S):
    for pts in lines:
        if len(pts) < 2:
            continue
        draw.line(_pts_ss(pts, S), fill=fill, width=width, joint="curve")


def _draw_contours(draw, layer, contours, stroke_rgb, op, params, font_dir, S):
    """Regular (non-index) contours, under everything else. When contour_text
    runs on ALL lines, the arc under each text run is left open."""
    reg_fill = _rgba(stroke_rgb, 0.55 * op)
    reg_w = _w(params["contour_weight"], S)
    ctext = str(params.get("contour_text", "")).strip()
    text_all = bool(ctext) and params.get("contour_text_on", "index") == "all"
    advances = None
    if text_all:
        cfont = _load_font(font_dir, float(params.get("contour_text_size", 11)) * S)
        advances = [cfont.getlength(ch) / S for ch in ctext + "  "]
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
                        draw.line(_pts_ss(sub, S), fill=reg_fill, width=reg_w, joint="curve")
                    continue
            p = topo._closed_pts(pts, closed)
            draw.line(_pts_ss(p, S), fill=reg_fill, width=reg_w, joint="curve")


def _draw_index_contours(draw, layer, contours, stroke_rgb, op, params, font_dir, S):
    cw = params["contour_weight"]
    iw = cw * params["index_weight"]
    idx_fill = _rgba(stroke_rgb, op)
    idx_w = _w(iw, S)
    labels_on = bool(params["contour_labels"])
    label_size = float(params["label_size"])
    spacing = float(params["label_spacing"])
    unit = str(params.get("label_unit", ""))
    font = _load_font(font_dir, label_size * S) if labels_on else None

    ctext = str(params.get("contour_text", "")).strip()
    ctext_on = params.get("contour_text_on", "index")
    ctext_size = float(params.get("contour_text_size", 11))
    ctext_gap = float(params.get("contour_text_gap", 60))
    cfont = _load_font(font_dir, ctext_size * S) if ctext else None
    advances = None
    if ctext:
        rep = ctext + "  "
        advances = [cfont.getlength(ch) / S for ch in rep]

    labels = []
    glyphs = []
    for lev in contours:
        is_idx = lev["index"]
        text_here = bool(ctext) and (ctext_on == "all" or is_idx)
        if not is_idx and not text_here:
            continue
        text = str(int(round(lev["z"]))) + unit
        fill = idx_fill if is_idx else _rgba(stroke_rgb, 0.55 * op)
        width = idx_w if is_idx else _w(cw, S)
        for (pts, closed) in lev["lines"]:
            L = topo.polyline_length(pts, closed)
            gaps = []
            if text_here:
                places, tgaps = topo.text_runs(pts, closed, advances, ctext_gap, ctext_size * 1.2)
                if places:
                    glyphs.append((places, fill))
                    gaps.extend(tgaps)
            if is_idx and labels_on and spacing > 0 and L >= spacing and not gaps:
                tw = font.getlength(text) / S
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
                    draw.line(_pts_ss(sub, S), fill=fill, width=width, joint="curve")
            else:
                p = topo._closed_pts(pts, closed)
                draw.line(_pts_ss(p, S), fill=fill, width=width, joint="curve")
    for (x, y, text, angle) in labels:
        _text_rotated(layer, x, y, text, font, idx_fill, angle, S)
    if glyphs:
        rep = ctext + "  "
        for (places, fill) in glyphs:
            for k, (x, y, ang) in enumerate(places):
                ch = rep[k % len(rep)]
                if ch == " ":
                    continue
                _text_rotated(layer, x, y, ch, cfont, fill, ang, S)


def _draw_ticks(draw, ticks, stroke_rgb, op, weight, S):
    if not ticks:
        return
    fill = _rgba(stroke_rgb, 0.55 * op)
    width = _w(weight, S)
    for ((x0, y0), (x1, y1)) in ticks:
        draw.line([(x0 * S, y0 * S), (x1 * S, y1 * S)], fill=fill, width=width)


def _draw_arrows(draw, arrows, stroke_rgb, op, weight, S):
    if not arrows:
        return
    fill = _rgba(stroke_rgb, 0.7 * op)
    width = _w(weight, S)
    ang = math.radians(28.0)
    for ((x0, y0), (x1, y1), L) in arrows:
        draw.line([(x0 * S, y0 * S), (x1 * S, y1 * S)], fill=fill, width=width)
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
            draw.line([(x1 * S, y1 * S), ((x1 + hl * bx) * S, (y1 + hl * by) * S)],
                      fill=fill, width=width)


def _draw_spots(draw, spots, stroke_rgb, op, params, font_dir, S):
    if not spots:
        return
    fill = _rgba(stroke_rgb, op)
    size = float(params["symbol_size"])
    width = _w(params["contour_weight"], S)
    font = _load_font(font_dir, params["label_size"] * S)
    for (x, y, z, kind) in spots:
        if kind == "peak":
            r = size / math.sqrt(3.0)   # circumradius of a triangle of side `size`
            pts = [(x, y - r), (x + size / 2.0, y + r / 2.0), (x - size / 2.0, y + r / 2.0)]
            draw.polygon(_pts_ss(pts, S), fill=fill)
        else:
            r = size / 2.0
            draw.ellipse([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S],
                         outline=fill, width=width)
        draw.text(((x + size * 0.8 + 2) * S, y * S), str(int(round(z))), font=font,
                  fill=fill, anchor="lm")


def _draw_benchmarks(draw, bms, stroke_rgb, op, params, font_dir, S):
    if not bms:
        return
    fill = _rgba(stroke_rgb, op)
    size = float(params["symbol_size"])
    r = size / 2.0
    width = _w(params["contour_weight"], S)
    font = _load_font(font_dir, params["label_size"] * S)
    for (x, y, z) in bms:
        draw.line([((x - r) * S, (y - r) * S), ((x + r) * S, (y + r) * S)], fill=fill, width=width)
        draw.line([((x - r) * S, (y + r) * S), ((x + r) * S, (y - r) * S)], fill=fill, width=width)
        draw.text(((x + size * 0.8 + 2) * S, y * S), "BM " + str(int(round(z))), font=font,
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


def _cc_bare(mark):
    """Unit-disc frame -> bare-mark frame: centred on the mark's own bbox,
    half-width 1, so a bare mark of diameter `size` spans `size` across."""
    pts = [p for poly in mark for p in poly]
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    mx, my, half = (x0 + x1) / 2.0, (y0 + y1) / 2.0, (x1 - x0) / 2.0
    return tuple([((u - mx) / half, (v - my) / half) for (u, v) in poly] for poly in mark)


CC_MARK_BARE = _cc_bare(CC_MARK)


def _draw_campsites(layer, camps, stroke_rgb, bg_rgb, op, params, S):
    """CanvasCamp campsite per site, two styles.
    badge: background-colour disc carrying the mark in the stroke colour, a
    stroke-colour ring on the rim (campsite_ring x diameter, min 1 px; 0 = no
    ring) and a thin background halo outside it so contours never touch the
    ring. mark: the logo alone in the stroke colour, no disc, no halo, its
    width spanning campsite_size. campsite_logo_scale scales the logo in both.
    Each site is built opaque on its own tile and composited at `op`, so the
    disc stays clean at partial overlay_opacity."""
    if not camps or op <= 0.0:
        return
    size = float(params["campsite_size"])
    bare = params.get("campsite_style", "badge") == "mark"
    ring_frac = float(params.get("campsite_ring", 0.06))
    logo = float(params.get("campsite_logo_scale", 1.0))
    r = size / 2.0 * S
    ring = max(1.0, ring_frac * size) * S if ring_frac > 0.0 else 0.0
    halo = max(1.0, float(params["contour_weight"])) * S
    R = r if bare else r + halo
    rl = r * logo
    pad = int(math.ceil(max(R, rl))) + 2
    a_op = int(round(np.clip(op, 0.0, 1.0) * 255))
    ink = (int(stroke_rgb[0]), int(stroke_rgb[1]), int(stroke_rgb[2]), 255)
    ground = (int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2]), 255)
    mark = CC_MARK_BARE if bare else CC_MARK
    for (x, y) in camps:
        X, Y = x * S, y * S
        ox, oy = int(math.floor(X)) - pad, int(math.floor(Y)) - pad
        cx, cy = X - ox, Y - oy
        tile = Image.new("RGBA", (2 * pad + 1, 2 * pad + 1), (0, 0, 0, 0))
        td = ImageDraw.Draw(tile)
        if not bare:
            td.ellipse([cx - R, cy - R, cx + R, cy + R], fill=ground)
            if ring > 0.0:
                td.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ink)
                ri = r - ring
                td.ellipse([cx - ri, cy - ri, cx + ri, cy + ri], fill=ground)
        for poly in mark:
            td.polygon([(cx + u * rl, cy + v * rl) for (u, v) in poly], fill=ink)
        if a_op < 255:
            alpha = tile.getchannel("A").point(lambda a: (a * a_op + 127) // 255)
            tile.putalpha(alpha)
        layer.alpha_composite(tile, (ox, oy))


def _draw_annotations(draw, w, notes, stroke_rgb, op, params, font_dir, S):
    if not notes:
        return
    fill = _rgba(stroke_rgb, op)
    ms = float(params["marginalia_scale"])
    width = _w(params["contour_weight"], S)
    font = _load_font(font_dir, params["label_size"] * 1.2 * S)
    dot = 2.5 * S
    up = 28.0 * ms
    tail = 36.0 * ms
    for (x, y, text) in notes:
        draw.ellipse([x * S - dot, y * S - dot, x * S + dot, y * S + dot], fill=fill)
        sgn = -1.0 if x > 0.7 * w else 1.0
        ex, ey = x + sgn * up, y - up
        tx = ex + sgn * tail
        draw.line([(x * S, y * S), (ex * S, ey * S), (tx * S, ey * S)], fill=fill, width=width)
        anchor = "lm" if sgn > 0 else "rm"
        draw.text(((tx + sgn * 6) * S, ey * S), text, font=font, fill=fill, anchor=anchor)


def _draw_marginalia(draw, w, h, stroke_rgb, op, params, elements, font_dir, S):
    fill = _rgba(stroke_rgb, op)
    ms = float(params["marginalia_scale"])
    width = _w(params["contour_weight"], S)
    label_size = float(params["label_size"])
    fts = float(params["frame_text_size"])
    font = _load_font(font_dir, label_size * S)

    if params["north_arrow"]:
        cx = w - 40 - 10 * ms
        cy = 40 + fts * 1.6 + 28 * ms
        shaft = 44 * ms
        tip_y = cy - shaft / 2.0
        base_y = cy + shaft / 2.0
        draw.line([(cx * S, base_y * S), (cx * S, (tip_y + 14 * ms) * S)], fill=fill, width=width)
        draw.polygon([(cx * S, tip_y * S), ((cx - 6 * ms) * S, (tip_y + 14 * ms) * S),
                      ((cx + 6 * ms) * S, (tip_y + 14 * ms) * S)], fill=fill)
        nfont = _load_font(font_dir, label_size * 1.2 * S)
        draw.text((cx * S, (tip_y - 10 * ms) * S), "N", font=nfont, fill=fill, anchor="mm")

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
                    draw.rectangle([xa * S, (bar_base - bh) * S, xb * S, bar_base * S], fill=fill)
                else:
                    draw.rectangle([xa * S, (bar_base - bh) * S, xb * S, bar_base * S],
                                   outline=fill, width=width)
            for xt in (x0, x0 + bar_px):
                draw.line([(xt * S, (bar_base - bh - 4 * ms) * S), (xt * S, bar_base * S)],
                          fill=fill, width=width)
            if metres >= 1000:
                unit = f"{metres / 1000.0:g} km"
            else:
                unit = f"{metres:g} m"
            draw.text((x0 * S, (bar_base - bh - 6 * ms) * S), "0", font=font, fill=fill, anchor="ld")
            draw.text(((x0 + bar_px) * S, (bar_base - bh - 6 * ms) * S), unit, font=font,
                      fill=fill, anchor="rd")

    if params["interval_note"]:
        note = f"CONTOUR INTERVAL {float(elements['interval']):g} M"
        if params["scale_bar"] and elements.get("scale_bar") is not None:
            ny = bar_base + 6 * ms + label_size * 1.4
        else:
            ny = h - 40 - fts * 1.6
        draw.text((40 * S, ny * S), note, font=font, fill=fill, anchor="ld")


def _draw_frame_text(draw, w, h, stroke_rgb, op, params, font_dir, S):
    size = params["frame_text_size"]
    inset = 40
    font = _load_font(font_dir, size * S)
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
        draw.text((pos[0] * S, pos[1] * S), text, font=font, fill=fill, anchor=anchor)


# ---------------------------------------------------------------------------
# Full pipeline (section 6)
# ---------------------------------------------------------------------------

def render_frame_topographic(fitted01, w, h, palette, params, elements, texture01, include_image, font_dir,
                             cache=None, unit=1.0):
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
    unit: canvas px per layout unit (the node's reference_size lock). Every
      element coordinate and every px-valued param is in LAYOUT units; the
      vector layer is drawn at S = SS * unit layer px per unit, so the whole
      sheet (strokes, fonts, insets, marginalia) scales as one. unit == 1.0
      is the literal-pixel path, bitwise the pre-lock render. `zones` alone
      arrive in canvas px (the mosaic is a raster step on the real canvas).
    Returns (uint8 (h,w,3) composite, float32 (h,w) overlay_alpha).
    """
    bg_rgb, stroke_rgb = palette
    op = float(params["overlay_opacity"])
    if cache is None:
        cache = {}
    unit = float(unit)
    locked = unit != 1.0
    S = SS * unit if locked else SS            # layer px per layout unit
    wu, hu = (w / unit, h / unit) if locked else (w, h)   # canvas in layout units

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
        zone_size, zlabel = params["zone_size"], params["label_size"]
        if locked:
            pixel_size = max(1, int(round(pixel_size * unit)))
            zone_size, zlabel = zone_size * unit, zlabel * unit
        canvas = overlay_render.pixelate_canvas(canvas, w, h, zones, pixel_size, zone_size,
                                                params["pixel_stroke"], stroke_rgb, op,
                                                zlabel, font_dir, unit=unit)

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
        _draw_grid(draw_a, wu, hu, xs, ys, params["grid"], stroke_rgb, op, params["grid_weight"], S)
        _draw_contours(draw_a, layer_a, elements["contours"], stroke_rgb, op, params, font_dir, S)
        _draw_ticks(draw_a, elements["hachures"], stroke_rgb, op, params["contour_weight"], S)
        _draw_index_contours(draw_a, layer_a, elements["contours"], stroke_rgb, op, params, font_dir, S)
        _draw_arrows(draw_a, elements["arrows"], stroke_rgb, op, params["contour_weight"], S)
        _draw_spots(draw_a, elements["spots"], stroke_rgb, op, params, font_dir, S)
        _draw_benchmarks(draw_a, elements["benchmarks"], stroke_rgb, op, params, font_dir, S)
        _draw_campsites(layer_a, elements.get("campsites") or [], stroke_rgb, bg_rgb, op, params, S)
        _draw_annotations(draw_a, wu, elements["annotations"], stroke_rgb, op, params, font_dir, S)
        if params["grid"] != "off" and params["grid_labels"]:
            _draw_grid_labels(layer_a, draw_a, wu, hu, xs, ys, stroke_rgb, op, params["label_size"],
                              params["map_scale"], font_dir, S)
        _draw_marginalia(draw_a, wu, hu, stroke_rgb, op, params, elements, font_dir, S)
        _draw_frame_text(draw_a, wu, hu, stroke_rgb, op, params, font_dir, S)
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
