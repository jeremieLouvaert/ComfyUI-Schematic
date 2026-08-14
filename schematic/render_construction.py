"""Schematic Construction v2 - rendering.

Spec: docs/schematic-construction-derivation-v2.md (signed 2026-08-14).

Two rules govern this file and both come from why v0.1 failed:

  * NO MARK MAY BE DRAWN AT AN UNCLASSIFIED WEIGHT (spec section 4). v0.1 drew
    everything at one line_weight, and that single fact is most of why it read as
    an undecided web.
  * NO SHADOW / UNDER-STROKE (spec section 6.2). An engraved plate has no drop
    shadow. Legibility is ink polarity, handled by the palette, not by doubling
    every stroke.
"""

import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import construction as geo
from .draw_common import SS, _resolve_overlay
from .elements import rng_for

# ---- weight classes (spec section 4) --------------------------------------
DATUM, CONSTRUCT, FINE = 1.9, 1.0, 0.55
_WEIGHTS = (DATUM, CONSTRUCT, FINE)

_MONO_REL = os.path.join("fonts", "JetBrainsMono-Regular.ttf")
_FONT_CACHE = {}

_HEX = "0123456789ABCDEF"
_POINT_LETTERS = "abcdefghikmnpqr"


def type_px(canvas_min, frac, type_scale):
    """Type as a constant fraction of canvas, so micro-type stays sub-legible at
    every resolution: it is texture, not content."""
    return max(1.0, float(canvas_min) * float(frac) * float(type_scale))


def _font(pack_dir, px):
    key = (pack_dir, int(round(px * SS)))
    if key not in _FONT_CACHE:
        path = os.path.join(pack_dir, _MONO_REL)
        try:
            _FONT_CACHE[key] = ImageFont.truetype(path, max(1, key[1]))
        except Exception:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


def _hexs(rng, n):
    return "".join(_HEX[int(rng.draw() * 16) % 16] for _ in range(n))


class Plate:
    """Supersampled mark layer with a label-collision resolver."""

    def __init__(self, w, h, pack_dir, stroke_rgb, op, line_weight):
        self.w, self.h = int(w), int(h)
        self.cmin = min(self.w, self.h)
        self.pack = pack_dir
        self.stroke = tuple(int(v) for v in stroke_rgb)
        self.op = float(max(0.0, min(1.0, op)))
        self.lw_scale = float(line_weight)
        self.layer = Image.new("RGBA", (self.w * SS, self.h * SS), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.layer, "RGBA")
        self.taken = []
        self.dropped = 0

    # -- weight discipline --------------------------------------------------
    def _w(self, cls):
        if cls not in _WEIGHTS:
            raise ValueError(
                f"[Construction] mark drawn at unclassified weight {cls!r}; "
                f"use DATUM, CONSTRUCT or FINE (spec section 4).")
        px = self.cmin / 1000.0 * cls * self.lw_scale
        return max(1, int(round(px * SS)))

    def _c(self, a):
        return (self.stroke[0], self.stroke[1], self.stroke[2],
                int(round(max(0, min(255, a)) * self.op)))

    # -- primitives ---------------------------------------------------------
    def line(self, x0, y0, x1, y1, cls=CONSTRUCT, a=230):
        self.d.line([(x0 * SS, y0 * SS), (x1 * SS, y1 * SS)],
                    fill=self._c(a), width=self._w(cls))

    def arc(self, cx, cy, r, a0, a1, cls=CONSTRUCT, a=225):
        self.d.arc([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS],
                   a0, a1, fill=self._c(a), width=self._w(cls))

    def circle(self, cx, cy, r, cls=FINE, a=210):
        self.d.ellipse([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS],
                       outline=self._c(a), width=self._w(cls))

    def dot(self, cx, cy, r, a=250, hollow=False):
        box = [(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS]
        if hollow:
            self.d.ellipse(box, outline=self._c(a), width=self._w(FINE))
        else:
            self.d.ellipse(box, fill=self._c(a))

    def marker(self, cx, cy, r, shape="circle", fill=True, a=250):
        """Intersection marker. Shape and fill are user-chosen; the two crosses
        are stroke-only by nature and ignore `fill`."""
        col = self._c(a)
        w = self._w(FINE)
        if shape == "cross_plus":
            self.d.line([(cx - r) * SS, cy * SS, (cx + r) * SS, cy * SS],
                        fill=col, width=w)
            self.d.line([cx * SS, (cy - r) * SS, cx * SS, (cy + r) * SS],
                        fill=col, width=w)
            return
        if shape == "cross_x":
            k = r * 0.7071
            self.d.line([(cx - k) * SS, (cy - k) * SS, (cx + k) * SS, (cy + k) * SS],
                        fill=col, width=w)
            self.d.line([(cx - k) * SS, (cy + k) * SS, (cx + k) * SS, (cy - k) * SS],
                        fill=col, width=w)
            return
        if shape == "triangle":
            pts = [(cx * SS, (cy - r) * SS),
                   ((cx + r * 0.866) * SS, (cy + r * 0.5) * SS),
                   ((cx - r * 0.866) * SS, (cy + r * 0.5) * SS)]
            if fill:
                self.d.polygon(pts, fill=col)
            else:
                self.d.polygon(pts, outline=col, width=w)
            return
        if shape == "square":
            box = [(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS]
            if fill:
                self.d.rectangle(box, fill=col)
            else:
                self.d.rectangle(box, outline=col, width=w)
            return
        self.dot(cx, cy, r, a, hollow=not fill)

    def text(self, x, y, s, px, a=250, anchor="la"):
        self.d.text((x * SS, y * SS), s, font=_font(self.pack, px),
                    fill=self._c(a), anchor=anchor)

    # -- collision resolver (spec section 6.1) ------------------------------
    def _free(self, box):
        x0, y0, x1, y1 = box
        if x0 < 2 or y0 < 2 or x1 > self.w - 2 or y1 > self.h - 2:
            return False
        for (a0, b0, a1, b1) in self.taken:
            if x0 < a1 and a0 < x1 and y0 < b1 and b0 < y1:
                return False
        return True

    def label(self, x, y, s, px, a=250, anchor="la", pad=1.6):
        """Reserve a box, try alternates, DROP if none is clear. Dropping is
        correct: it thins annotation exactly where the geometry is densest."""
        f = _font(self.pack, px)
        bb = self.d.textbbox((0, 0), s, font=f, anchor=anchor)
        tw = (bb[2] - bb[0]) / float(SS)
        th = (bb[3] - bb[1]) / float(SS)
        for (ox, oy) in ((0, 0), (0, -th - 3), (0, th + 3), (6, 0), (-6, 0),
                         (0, -2 * th - 5), (10, -th - 3)):
            bx = x + ox + (0.0 if anchor[0] == "l" else -tw)
            by = y + oy
            box = (bx - pad, by - pad, bx + tw + pad, by + th + pad)
            if self._free(box):
                self.taken.append(box)
                self.text(x + ox, y + oy, s, px, a, anchor)
                return True
        self.dropped += 1
        return False

    def resolve_onto(self, canvas01):
        pm, alpha = _resolve_overlay(self.layer, self.w, self.h)
        return canvas01 * (1.0 - alpha[..., None]) + pm, alpha


# ---------------------------------------------------------------------------
# The plate
# ---------------------------------------------------------------------------

def draw_plate(canvas01, layout, st, seed, params, pack_dir, stroke_rgb):
    w, h = layout.w, layout.h
    cmin = min(w, h)
    p = Plate(w, h, pack_dir, stroke_rgb, params["overlay_opacity"],
              params["line_weight"])

    ts = float(params["type_scale"])
    micro = type_px(cmin, 0.0066, ts)
    small = type_px(cmin, 0.0082, ts)
    big = type_px(cmin, 0.0125, ts)

    def off(eid):
        return st.px(eid, w, h)

    # ---- spheres (DATUM for the first, CONSTRUCT for the rest) ------------
    a0 = math.degrees(layout.arc_from)
    a1 = math.degrees(layout.arc_to)
    if st.on("spheres"):
        ox, oy = off("spheres")
        for k, r in enumerate(layout.radii):
            p.arc(layout.fa[0] + ox, layout.fa[1] + oy, r, a0, a1,
                  DATUM if k == 0 else CONSTRUCT, 240)

    # ---- rays -------------------------------------------------------------
    clear_a = cmin * float(params["focus_clear"])
    if st.on("rays_a"):
        ox, oy = off("rays_a")
        for (ux, uy) in layout.rays_a:
            sx, sy = layout.fa[0] + ux * clear_a, layout.fa[1] + uy * clear_a
            ex, ey = geo.edge_hit(layout.fa[0], layout.fa[1], ux, uy, w, h)
            p.line(sx + ox, sy + oy, ex + ox, ey + oy, FINE, 175)
    if st.on("rays_b"):
        # focus_clear applies IDENTICALLY to both foci. It previously carried an
        # undocumented 0.72 factor here, so the widget lied about focus B and its
        # rays started closer than the value said.
        ox, oy = off("rays_b")
        for (ux, uy) in layout.rays_b:
            sx, sy = layout.fb[0] + ux * clear_a, layout.fb[1] + uy * clear_a
            ex, ey = geo.edge_hit(layout.fb[0], layout.fb[1], ux, uy, w, h)
            p.line(sx + ox, sy + oy, ex + ox, ey + oy, FINE, 150)

    # ---- lune -------------------------------------------------------------
    if st.on("lune") and len(layout.rays_a) > layout.lune_at + 4 and len(layout.radii) > 1:
        ox, oy = off("lune")
        i0 = layout.lune_at
        u0, u1 = layout.rays_a[i0], layout.rays_a[i0 + 4]
        ang0 = math.atan2(u0[1], u0[0])
        ang1 = math.atan2(u1[1], u1[0])
        k = min(layout.lune_gap, len(layout.radii) - 2)
        r0, r1 = layout.radii[k], layout.radii[k + 1]
        n = max(4, int(params["lune_density"]))
        for i in range(n):
            t = i / float(n - 1)
            ang = ang0 + (ang1 - ang0) * t
            ca, sa = math.cos(ang), math.sin(ang)
            p.line(layout.fa[0] + ca * r0 + ox, layout.fa[1] + sa * r0 + oy,
                   layout.fa[0] + ca * r1 + ox, layout.fa[1] + sa * r1 + oy,
                   CONSTRUCT if i % 4 == 0 else FINE, 205)

    # ---- intersections: the density engine --------------------------------
    m_every = max(1, int(params["marker_every"]))
    l_every = max(1, int(params["label_every"]))
    m_size = cmin * float(params["marker_size"])
    m_shape = str(params.get("marker_shape", "circle"))
    m_fill = bool(params.get("marker_fill", True))

    if st.on("intersections_a"):
        ox, oy = off("intersections_a")
        for n, (x, y, i, k) in enumerate(geo.intersections_trivial(layout)):
            if not geo.on_canvas((x, y), w, h) or n % m_every:
                continue
            p.marker(x + ox, y + oy, m_size, m_shape, m_fill, 250)
            if st.on("intersection_labels") and n % (m_every * l_every) == 0:
                lo = off("intersection_labels")
                p.label(x + lo[0] + 4, y + lo[1] - 9,
                        f"{_POINT_LETTERS[k % len(_POINT_LETTERS)]}{i}", micro, 240)

    if st.on("intersections_b"):
        ox, oy = off("intersections_b")
        for n, (x, y, i, k) in enumerate(geo.intersections_nontrivial(layout)):
            if not geo.on_canvas((x, y), w, h) or n % m_every:
                continue
            p.marker(x + ox, y + oy, m_size * 1.1, m_shape, False, 250)
            if st.on("intersection_labels") and n % (m_every * l_every) == 0:
                lo = off("intersection_labels")
                p.label(x + lo[0] + 4, y + lo[1] - 9, f"{k}·{i}", micro, 235)

    # ---- focus furniture --------------------------------------------------
    if st.on("focus_a_mark"):
        ox, oy = off("focus_a_mark")
        p.dot(layout.fa[0] + ox, layout.fa[1] + oy, cmin * 0.0042)
        p.circle(layout.fa[0] + ox, layout.fa[1] + oy, cmin * 0.014, CONSTRUCT, 245)
        p.circle(layout.fa[0] + ox, layout.fa[1] + oy, cmin * 0.021, FINE, 200)
    if st.on("focus_b_mark") and params["focus_b_enabled"]:
        ox, oy = off("focus_b_mark")
        p.dot(layout.fb[0] + ox, layout.fb[1] + oy, cmin * 0.0036)
        p.circle(layout.fb[0] + ox, layout.fb[1] + oy, cmin * 0.011, CONSTRUCT, 245)
    if st.on("focus_labels"):
        ox, oy = off("focus_labels")
        p.label(layout.fa[0] + ox + cmin * 0.020, layout.fa[1] + oy - cmin * 0.026,
                "H₁", big, 255)
        if params["focus_b_enabled"]:
            p.label(layout.fb[0] + ox - cmin * 0.016, layout.fb[1] + oy + cmin * 0.014,
                    "H₂", big, 255, anchor="ra")

    # ---- sphere captions ---------------------------------------------------
    if st.on("sphere_captions"):
        ox, oy = off("sphere_captions")
        rng = rng_for(seed, "sphere_captions")
        for k, r in enumerate(layout.radii):
            ang = layout.arc_to - (k + 1) * 0.05
            cx = layout.fa[0] + math.cos(ang) * r + ox
            cy = layout.fa[1] + math.sin(ang) * r + oy
            p.label(cx + 7, cy - 11, f"SPHAERA {'I' * (k + 1) if k < 3 else 'IV'}",
                    small, 245)
            p.label(cx + 7, cy - 1, f"r={int(r)}  n={len(layout.rays_a) - k * 4}",
                    micro, 225)

    # ---- graduated scale ---------------------------------------------------
    if st.on("scale_ticks") and layout.radii:
        ox, oy = off("scale_ticks")
        r = layout.radii[-1]
        n = max(2, int(params["scale_ticks"]))
        for i in range(n):
            t = i / float(n - 1)
            ang = layout.arc_from + (layout.arc_to - layout.arc_from) * t
            ca, sa = math.cos(ang), math.sin(ang)
            ln = cmin * (0.012 if i % 5 == 0 else 0.006)
            p.line(layout.fa[0] + ca * r + ox, layout.fa[1] + sa * r + oy,
                   layout.fa[0] + ca * (r + ln) + ox, layout.fa[1] + sa * (r + ln) + oy,
                   FINE, 235)
            if st.on("scale_numerals") and i % max(1, int(params["scale_label_every"])) == 0:
                no = off("scale_numerals")
                p.label(layout.fa[0] + ca * (r + cmin * 0.017) + no[0],
                        layout.fa[1] + sa * (r + cmin * 0.017) + no[1] - 4,
                        str(i * 2), micro, 235)

    # ---- frame-exit labels --------------------------------------------------
    if st.on("exit_labels"):
        ox, oy = off("exit_labels")
        every = max(1, int(params["exit_labels_every"]))
        for i, (ux, uy) in enumerate(layout.rays_a):
            if i % every:
                continue
            ex, ey = geo.edge_hit(layout.fa[0], layout.fa[1], ux, uy, w, h)
            deg = (math.degrees(math.atan2(uy, ux)) + 360.0) % 360.0
            anchor, tx = "la", ex + 5
            if ex > w - cmin * 0.06:
                anchor, tx = "ra", ex - 5
            ty = ey + 3 if ey < cmin * 0.014 else ey - 5
            p.label(tx + ox, ty + oy, f"r{i:02d} {deg:.1f}°", micro, 235,
                    anchor=anchor)

    # ---- callouts -----------------------------------------------------------
    if st.on("callouts"):
        ox, oy = off("callouts")
        rng = rng_for(seed, "callouts")
        tri = [q for q in geo.intersections_trivial(layout)
               if geo.on_canvas(q, w, h, cmin * 0.12)]
        for c in range(max(0, int(params["callout_count"]))):
            if not tri:
                break
            q = tri[int(rng.draw() * len(tri)) % len(tri)]
            ax, ay = q[0] + ox, q[1] + oy
            p.marker(ax, ay, m_size, m_shape, False, 250)
            p.line(ax, ay, ax + cmin * 0.026, ay - cmin * 0.020, FINE, 220)
            p.line(ax + cmin * 0.026, ay - cmin * 0.020,
                   ax + cmin * 0.060, ay - cmin * 0.020, FINE, 220)
            for k in range(4):
                p.label(ax + cmin * 0.063, ay - cmin * 0.027 + k * micro * 1.45,
                        _hexs(rng, 11), micro, 235)

    # ---- data column --------------------------------------------------------
    if st.on("data_column") and params["data_column_enabled"]:
        ox, oy = off("data_column")
        rng = rng_for(seed, "data_column")
        rows = max(1, int(params["data_column_rows"]))
        x0 = w - cmin * 0.116 + ox
        y0 = h * 0.28 + oy
        lead = micro * 2.05
        p.label(x0, y0 - lead, "IDX   ANG   REF", micro, 250)
        p.line(x0 - cmin * 0.006, y0 - lead * 0.3,
               x0 - cmin * 0.006, y0 + rows * lead, FINE, 200)
        for i in range(rows):
            if 2 * i >= len(layout.rays_a):
                break
            ux, uy = layout.rays_a[2 * i]
            deg = (math.degrees(math.atan2(uy, ux)) + 360.0) % 360.0
            p.label(x0, y0 + i * lead, f"{2 * i:02d} {deg:6.2f} {_hexs(rng, 4)}",
                    micro, 235)

    # ---- demoted detection circles -----------------------------------------
    if st.on("detection_circles"):
        ox, oy = off("detection_circles")
        for (x, y, r) in layout_points(layout):
            p.circle(x + ox, y + oy, r * 0.55, FINE, 165)

    # ---- frame text ---------------------------------------------------------
    if st.on("frame_text"):
        ox, oy = off("frame_text")
        fs = type_px(cmin, float(params["frame_text_size"]) / 1000.0, ts)
        m = cmin * 0.014
        for key, x, y, anch in (("frame_text_tl", m, m, "la"),
                                ("frame_text_tr", w - m, m, "ra"),
                                ("frame_text_bl", m, h - m - fs, "la"),
                                ("frame_text_br", w - m, h - m - fs, "ra")):
            s = str(params.get(key, "") or "")
            if s:
                p.text(x + ox, y + oy, s, fs, 245, anchor=anch)

    return p


def layout_points(layout):
    return getattr(layout, "_pts", ())
