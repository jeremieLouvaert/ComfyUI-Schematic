"""Schematic Construction v2 - geometry.

Spec: docs/schematic-construction-derivation-v2.md (signed 2026-08-14).
Supersedes the v0.1 apparatus; see spec section 0 for exactly what was kept, cut,
promoted and demoted.

Nothing here draws. It computes foci, pencils, spheres, EXACT intersections, the
seed-chosen layout, and the zone centres. Rendering lives in
render_construction.py.
"""

import math

import numpy as np

from . import engine
from .elements import Registry, rng_for

# ---------------------------------------------------------------------------
# Element registry (spec section 9). Order is draw order.
# ---------------------------------------------------------------------------

REGISTRY = Registry((
    ("zones", "g_zones", True),

    ("spheres", "g_spheres", True),
    ("sphere_captions", "g_spheres", True),

    ("rays_a", "g_rays", True),
    ("rays_b", "g_rays", True),

    ("lune", "g_marks", True),

    ("intersections_a", "g_marks", True),
    ("intersections_b", "g_marks", True),
    ("intersection_labels", "g_marks", True),

    ("focus_a_mark", "g_focus", True),
    ("focus_b_mark", "g_focus", True),
    ("focus_labels", "g_focus", True),

    ("scale_ticks", "g_scale", True),
    ("scale_numerals", "g_scale", True),

    ("exit_labels", "g_annot", True),
    ("callouts", "g_annot", True),
    ("data_column", "g_annot", True),

    # Overlay's signature read, deliberately OFF by default: circles around
    # detected points are what made v0.1 look like more-Overlay (spec section 0).
    ("detection_circles", "g_annot", False),

    ("frame_text", "g_frame", True),
))


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect(fitted01, detection_mode, block_size, threshold, max_circles,
           min_distance, min_radius, max_radius, seed):
    """Overlay's detection, imported as-is, same widget semantics."""
    blocks = engine.analyze_blocks(fitted01, block_size)
    score = engine.score_blocks(blocks, detection_mode)
    circles = engine.place_circles(blocks, score, threshold, max_circles,
                                   min_distance, min_radius, max_radius, seed)
    pts = [(float(c[0]), float(c[1]), float(c[2])) for c in circles]
    return pts


# ---------------------------------------------------------------------------
# Layout chosen by the seed (spec section 7)
# ---------------------------------------------------------------------------

class Layout:
    """The seed chooses among VALID configurations. It never scatters freely: the
    convergent structure must survive any reseed."""

    def __init__(self, seed, w, h, pts, cfg):
        r = rng_for(seed, "__layout__")
        self.w, self.h = w, h
        self.s = float(min(w, h))

        def band(base, spread, lo, hi):
            v = base + (r.draw() - 0.5) * 2.0 * spread
            return float(np.clip(v, lo, hi))

        self.fa = (band(cfg["focus_a_x"], 0.06, 0.04, 0.96) * w,
                   band(cfg["focus_a_y"], 0.06, 0.04, 0.96) * h)
        self.fb = (band(cfg["focus_b_x"], 0.06, 0.04, 0.96) * w,
                   band(cfg["focus_b_y"], 0.06, 0.04, 0.96) * h)

        inner = float(cfg["sphere_inner"])
        outer = float(cfg["sphere_outer"])
        n = max(1, int(cfg["sphere_count"]))
        self.radii = []
        for i in range(n):
            t = i / float(max(1, n - 1))
            rf = inner + (outer - inner) * t
            rf *= 1.0 + (r.draw() - 0.5) * 0.06
            self.radii.append(self.s * rf)

        span = math.radians(float(cfg["sphere_span"]))
        base_ang = math.atan2(h * 0.5 - self.fa[1], w * 0.5 - self.fa[0])
        self.arc_from = base_ang - span / 2.0
        self.arc_to = base_ang + span / 2.0

        self.rays_a = self._pencil(r, self.fa, pts, int(cfg["rays_a"]),
                                   float(cfg["ray_jitter"]))
        self.rays_b = self._pencil(r, self.fb, pts, int(cfg["rays_b"]),
                                   float(cfg["ray_jitter"]))

        self.lune_at = int(r.draw() * max(1, len(self.rays_a) - 4))
        self.lune_gap = int(r.draw() * max(1, n - 1))

    @staticmethod
    def _pencil(r, focus, pts, count, jitter):
        out = []
        if count <= 0 or not pts:
            return out
        for i in range(count):
            t = i / float(max(1, count - 1))
            p = pts[int(t * (len(pts) - 1))]
            dx, dy = p[0] - focus[0], p[1] - focus[1]
            d = math.hypot(dx, dy) or 1.0
            j = (r.draw() - 0.5) * 2.0 * jitter
            ca, sa = math.cos(j), math.sin(j)
            out.append(((dx * ca - dy * sa) / d, (dx * sa + dy * ca) / d))
        return out


def edge_hit(px, py, ux, uy, w, h):
    """Forward intersection of a ray with the canvas boundary. Frame-CROSSING
    lines are the thing Overlay never draws: its connections stop at points."""
    ts = []
    if abs(ux) > 1e-12:
        ts += [(0.0 - px) / ux, (float(w) - px) / ux]
    if abs(uy) > 1e-12:
        ts += [(0.0 - py) / uy, (float(h) - py) / uy]
    pos = [t for t in ts if t > 1e-9]
    t = min(pos) if pos else 0.0
    return px + ux * t, py + uy * t


# ---------------------------------------------------------------------------
# The density engine: EXACT intersections (spec section 5.3)
# ---------------------------------------------------------------------------

def ray_circle(px, py, ux, uy, cx, cy, r):
    """Forward roots of |P + t*u - C| = r. Both roots computed; a linearised
    approximation must fail invariant 6."""
    fx, fy = px - cx, py - cy
    b = 2.0 * (fx * ux + fy * uy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return []
    sq = math.sqrt(disc)
    return [t for t in ((-b - sq) * 0.5, (-b + sq) * 0.5) if t > 1e-9]


def intersections_trivial(layout):
    """A ray from focus_a meets a sphere about focus_a at exactly r. Exact by
    construction, which invariant 5 pins."""
    out = []
    fx, fy = layout.fa
    for i, (ux, uy) in enumerate(layout.rays_a):
        for k, r in enumerate(layout.radii):
            out.append((fx + ux * r, fy + uy * r, i, k))
    return out


def intersections_nontrivial(layout):
    """A ray from focus_b meets a sphere about focus_a via the real quadratic."""
    out = []
    ax, ay = layout.fa
    bx, by = layout.fb
    for i, (ux, uy) in enumerate(layout.rays_b):
        for k, r in enumerate(layout.radii):
            for t in ray_circle(bx, by, ux, uy, ax, ay, r)[:1]:
                out.append((bx + ux * t, by + uy * t, i, k))
    return out


def on_canvas(p, w, h, margin=0.0):
    return margin <= p[0] <= w - margin and margin <= p[1] <= h - margin


# ---------------------------------------------------------------------------
# Zone centres (spec section 8)
# ---------------------------------------------------------------------------

def zone_sizes(seed, n, lo, hi):
    """One size per zone, drawn from [lo, hi].

    Spacing elsewhere uses the LARGEST possible size, not each zone's own, so a
    small zone can never end up overlapping a large neighbour that was placed
    later - the guarantee has to hold for the worst case, not the average one.
    """
    lo, hi = float(min(lo, hi)), float(max(lo, hi))
    if n <= 0:
        return []
    if hi - lo < 1e-9:
        return [lo] * int(n)
    r = rng_for(seed, "__zone_sizes__")
    return [lo + r.draw() * (hi - lo) for _ in range(int(n))]


def zone_centres(seed, layout, w, h, count, zone_size, extra=()):
    """Chosen from on-canvas intersections so every sampled patch sits on an event
    the construction produced, spaced at least 2.1 * zone_size apart.

    `zone_size` here is the MAXIMUM zone half-extent (see zone_sizes)."""
    picked = [(float(x), float(y)) for (x, y) in extra]
    if count <= 0:
        return picked
    r = rng_for(seed, "__zones__")
    cand = [(p[0], p[1]) for p in intersections_trivial(layout)
            if on_canvas(p, w, h, zone_size)]
    cand += [(p[0], p[1]) for p in intersections_nontrivial(layout)
             if on_canvas(p, w, h, zone_size)]
    if not cand:
        return picked
    gap = 2.1 * float(zone_size)
    tries = 0
    while len(picked) < int(count) + len(extra) and tries < 400:
        tries += 1
        c = cand[int(r.draw() * len(cand)) % len(cand)]
        if all(math.hypot(c[0] - q[0], c[1] - q[1]) >= gap for q in picked):
            picked.append(c)
    return picked
