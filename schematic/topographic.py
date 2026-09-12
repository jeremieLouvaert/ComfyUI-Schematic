"""
Pure-numpy geometry engine for Schematic Topographic: height field, own
marching-squares contour tracer with polyline chaining, and every map-sheet
element's geometry (hachures, slope arrows, spot heights, benchmarks, grid,
annotations, scale bar, hypsometric bands). No torch, no PIL, no scipy -- see
docs/schematic-topographic-derivation.md for the frozen spec this file
implements (sections cited per function).

Kept import-safe standalone: `sys.path.insert(0, pack_root); from schematic
import topographic` works with nothing but numpy installed.
"""

import math

import numpy as np

from . import engine


# ---------------------------------------------------------------------------
# Per-element RNG streams (spec section 5 preamble; micrographics rule)
# ---------------------------------------------------------------------------

def element_seed(seed, element_id):
    """Stable per-element seed: FNV-1a over the id mixed with the user seed.
    Each element draws from its own stream so toggling a sibling never
    re-rolls it (patterns.md 2026-08-12)."""
    h = 2166136261
    for ch in str(element_id):
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    mixed = (int(seed) & 0x7FFFFFFF) ^ h
    return (mixed % (engine.MinstdLCG.MOD - 1)) + 1


# ---------------------------------------------------------------------------
# 3. Height field
# ---------------------------------------------------------------------------

def _luma(fitted01):
    """Rec.601 luma in float64, then ROUNDED THROUGH float32 - the MASK socket
    dtype - so a height_mask equal to the photo's luma reproduces the
    brightness path bitwise (spec section 3; patterns.md 2026-08-18)."""
    f = np.asarray(fitted01, dtype=np.float64)
    lum = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    return lum.astype(np.float32).astype(np.float64)


def _box_filter_mean(arr, k):
    """k x k box mean via 2D prefix sums, edge-replicate padding (k odd, >= 1).
    Local copy of the voronoi helper so this module stays standalone."""
    if k <= 1:
        return arr.astype(np.float64, copy=True)
    r = k // 2
    padded = np.pad(arr, ((r, r), (r, r)), mode="edge").astype(np.float64)
    c0 = np.cumsum(padded, axis=0)
    p0 = np.concatenate([np.zeros((1, c0.shape[1]), dtype=np.float64), c0], axis=0)
    row_sum = p0[k:, :] - p0[:-k, :]
    c1 = np.cumsum(row_sum, axis=1)
    p1 = np.concatenate([np.zeros((c1.shape[0], 1), dtype=np.float64), c1], axis=1)
    box_sum = p1[:, k:] - p1[:, :-k]
    return box_sum / float(k * k)


def _local_std_box(luma01, k):
    m = _box_filter_mean(luma01, k)
    m2 = _box_filter_mean(luma01 * luma01, k)
    return np.sqrt(np.maximum(0.0, m2 - m * m))


def gaussian_blur(arr, sigma):
    """Separable Gaussian, radius ceil(3 sigma), edge-replicate padding, kernel
    normalised to 1, float64. sigma <= 0 skips the op structurally (copy)."""
    a = np.asarray(arr, dtype=np.float64)
    if sigma <= 0.0:
        return a.copy()
    r = int(math.ceil(3.0 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    n = k.shape[0]
    # rows
    p = np.pad(a, ((0, 0), (r, r)), mode="edge")
    out = np.zeros_like(a)
    for i in range(n):
        out += k[i] * p[:, i:i + a.shape[1]]
    # cols
    p = np.pad(out, ((r, r), (0, 0)), mode="edge")
    out2 = np.zeros_like(a)
    for i in range(n):
        out2 += k[i] * p[i:i + a.shape[0], :]
    return out2


def _hash2(ix, iy, seed):
    """Deterministic 32-bit integer hash of lattice coords (vectorised)."""
    h = (ix.astype(np.uint64) * np.uint64(374761393)
         + iy.astype(np.uint64) * np.uint64(668265263)
         + np.uint64(int(seed) & 0xFFFFFFFF) * np.uint64(1442695041)) & np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(13)
    h = (h * np.uint64(1274126177)) & np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(16)
    return h


def _gradient_noise(x, y, seed):
    """2-D gradient (Perlin-style) noise at float coords x, y (arrays), lattice
    gradients from _hash2, quintic fade. Range about [-1, 1]."""
    xi = np.floor(x)
    yi = np.floor(y)
    xf = x - xi
    yf = y - yi
    xi = xi.astype(np.int64)
    yi = yi.astype(np.int64)

    def grad(ix, iy):
        ang = (_hash2(ix, iy, seed).astype(np.float64) / 4294967296.0) * (2.0 * math.pi)
        return np.cos(ang), np.sin(ang)

    def fade(t):
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    g00x, g00y = grad(xi, yi)
    g10x, g10y = grad(xi + 1, yi)
    g01x, g01y = grad(xi, yi + 1)
    g11x, g11y = grad(xi + 1, yi + 1)
    d00 = g00x * xf + g00y * yf
    d10 = g10x * (xf - 1.0) + g10y * yf
    d01 = g01x * xf + g01y * (yf - 1.0)
    d11 = g11x * (xf - 1.0) + g11y * (yf - 1.0)
    u = fade(xf)
    v = fade(yf)
    nx0 = d00 + u * (d10 - d00)
    nx1 = d01 + u * (d11 - d01)
    return nx0 + v * (nx1 - nx0)


def procedural_field(w, h, seed, scale, octaves, roughness, stretch, angle_deg):
    """
    Spec v2 section 3: seeded fractional-Brownian gradient noise, the terrain
    that has nothing to do with the photo. scale = feature size in canvas px
    of the first octave; octaves 1..6, lacunarity 2, gain = roughness;
    stretch >= 1 elongates features along angle_deg (ridge/valley flow).
    Returns (h, w) float64, raw (normalised by the caller).
    """
    scale = max(4.0, float(scale))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    a = math.radians(float(angle_deg))
    ca, sa = math.cos(a), math.sin(a)
    # rotate into the flow frame, compress the along-flow axis by stretch
    u = (xs * ca + ys * sa) / max(1.0, float(stretch))
    v = (-xs * sa + ys * ca)
    out = np.zeros((h, w), dtype=np.float64)
    amp = 1.0
    freq = 1.0 / scale
    s0 = element_seed(seed, "terrain")
    for o in range(max(1, int(octaves))):
        # per-octave offset so octaves never share a lattice origin
        ox = ((s0 * (o + 1) * 7919) % 1000) * 0.37
        oy = ((s0 * (o + 1) * 104729) % 1000) * 0.61
        out += amp * _gradient_noise(u * freq + ox, v * freq + oy, s0 + 131 * o)
        amp *= float(roughness)
        freq *= 2.0
    return out


def build_height(fitted01, source, smoothing, mask01=None, terrain=None):
    """
    Spec section 3 steps 1-3. fitted01: (H,W,3) float 0-1 cover-fitted source.
    mask01: optional canvas-sized (H,W) float64 in [0,1] that REPLACES source.
    terrain: dict(seed, scale, octaves, roughness, stretch, angle) used when
      source == "procedural" (spec v2; the default source - the terrain is
      its own landform, never the photo).
    Returns (H float64 (h,w) in [0,1], flat bool). Flat -> H is all zeros.
    """
    if mask01 is not None:
        s = np.asarray(mask01, dtype=np.float64)
    elif source == "procedural":
        hh, ww = np.asarray(fitted01).shape[0], np.asarray(fitted01).shape[1]
        t = terrain or {}
        s = procedural_field(ww, hh, t.get("seed", 0), t.get("scale", 420.0), t.get("octaves", 3),
                             t.get("roughness", 0.45), t.get("stretch", 1.6), t.get("angle", 30.0))
    else:
        lum = _luma(fitted01)
        if source == "darkness":
            s = 1.0 - lum
        elif source == "detail":
            sd = _local_std_box(lum, 15)
            mx = float(sd.max()) if sd.size else 0.0
            s = sd / mx if mx > 0.0 else np.zeros_like(sd)
        else:
            s = lum
    s = gaussian_blur(s, float(smoothing))
    lo = float(s.min())
    hi = float(s.max())
    if hi - lo <= 1e-9:
        return np.zeros_like(s), True
    return (s - lo) / (hi - lo), False


def work_grid(H, stride):
    """Spec section 3 step 5: F[j,i] = H[j*s, i*s] (ceil(h/s) x ceil(w/s) samples)."""
    s = max(1, int(stride))
    return np.ascontiguousarray(np.asarray(H, dtype=np.float64)[::s, ::s])


def gradient(F):
    """Central differences, one-sided at the borders; returns (gx, gy) in
    F-units per work cell."""
    F = np.asarray(F, dtype=np.float64)
    if F.shape[0] < 2 or F.shape[1] < 2:
        return np.zeros_like(F), np.zeros_like(F)
    gy, gx = np.gradient(F)
    return gx, gy


def sample_grad(gx, gy, x, y, stride):
    """Gradient at the work vertex nearest canvas point (x, y)."""
    j = int(min(gx.shape[0] - 1, max(0, round(y / stride))))
    i = int(min(gx.shape[1] - 1, max(0, round(x / stride))))
    return float(gx[j, i]), float(gy[j, i])


def grad_at(gx, gy, xs, ys, stride):
    """Vectorised sample_grad: arrays of canvas points -> (ex, ey) arrays."""
    j = np.clip(np.rint(np.asarray(ys, dtype=np.float64) / stride).astype(np.int64), 0, gx.shape[0] - 1)
    i = np.clip(np.rint(np.asarray(xs, dtype=np.float64) / stride).astype(np.int64), 0, gx.shape[1] - 1)
    return gx[j, i], gy[j, i]


def sample_F(F, x, y, stride):
    j = int(min(F.shape[0] - 1, max(0, round(y / stride))))
    i = int(min(F.shape[1] - 1, max(0, round(x / stride))))
    return float(F[j, i])


# ---------------------------------------------------------------------------
# 4. Contour levels + marching squares
# ---------------------------------------------------------------------------

MAX_LEVELS = 400


def contour_levels(elev_min, elev_max, interval, index_every):
    """Spec section 4 'Levels': z_k = k*interval strictly inside (elev_min, elev_max).
    Returns list of (z, t, is_index); at most MAX_LEVELS (guard)."""
    emin, emax = float(elev_min), float(elev_max)
    if emax < emin:
        emin, emax = emax, emin
    d = float(interval)
    if d <= 0.0 or emax - emin <= 0.0:
        return []
    k0 = int(math.floor(emin / d)) + 1
    k1 = int(math.ceil(emax / d)) - 1
    out = []
    rng = emax - emin
    for k in range(k0, k1 + 1):
        z = k * d
        if not (emin < z < emax):
            continue
        t = (z - emin) / rng
        is_index = bool(index_every > 0 and (k % int(index_every) == 0))
        out.append((z, t, is_index))
        if len(out) >= MAX_LEVELS:
            print(f"[Schematic] Schematic Topographic: level cap {MAX_LEVELS} reached, "
                  "raise contour_interval")
            break
    return out


# case -> list of (edgeA, edgeB) with edge codes T=0, R=1, B=2, L=3
_CASES = {
    0: [], 15: [],
    1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)],
    6: [(0, 2)], 7: [(3, 2)], 8: [(2, 3)], 9: [(0, 2)],
    11: [(1, 2)], 12: [(1, 3)], 13: [(0, 1)], 14: [(3, 0)],
}
_SADDLE_5_IN = [(0, 1), (3, 2)]     # centre inside: tl-br band; cut off tr and bl
_SADDLE_5_OUT = [(3, 0), (1, 2)]    # centre outside: isolate tl and br
_SADDLE_10_IN = [(3, 0), (1, 2)]    # centre inside: tr-bl band; cut off tl and br
_SADDLE_10_OUT = [(0, 1), (2, 3)]   # centre outside: isolate tr and bl


def trace_level(F, t, stride):
    """
    Marching squares at level t on F (>= t is inside), linear edge interpolation,
    saddles by centre mean, segments chained by shared edge ids into polylines.
    Returns list of (pts (n,2) float64 canvas px, closed bool); polylines with
    fewer than 3 vertices dropped. Spec section 4.
    """
    F = np.asarray(F, dtype=np.float64)
    nj, ni = F.shape
    if nj < 2 or ni < 2:
        return []
    s = float(stride)
    inside = F >= t

    tl = inside[:-1, :-1]
    tr = inside[:-1, 1:]
    br = inside[1:, 1:]
    bl = inside[1:, :-1]
    case = (tl.astype(np.int64) + 2 * tr.astype(np.int64)
            + 4 * br.astype(np.int64) + 8 * bl.astype(np.int64))

    nh = nj * (ni - 1)            # horizontal edge ids: j*(ni-1)+i
    stride_h = ni - 1

    cj, ci = np.nonzero((case != 0) & (case != 15))
    if cj.size == 0:
        return []
    cc = case[cj, ci]

    def eid(code, jj, ii):
        if code == 0:     # T
            return jj * stride_h + ii
        if code == 2:     # B
            return (jj + 1) * stride_h + ii
        if code == 3:     # L
            return nh + jj * ni + ii
        return nh + jj * ni + ii + 1   # R

    seg_a, seg_b = [], []
    for c in range(1, 15):
        sel = cc == c
        if not np.any(sel):
            continue
        jj, ii = cj[sel], ci[sel]
        if c == 5 or c == 10:
            centre = 0.25 * (F[jj, ii] + F[jj, ii + 1] + F[jj + 1, ii + 1] + F[jj + 1, ii])
            cin = centre >= t
            for sub, table in ((cin, _SADDLE_5_IN if c == 5 else _SADDLE_10_IN),
                               (~cin, _SADDLE_5_OUT if c == 5 else _SADDLE_10_OUT)):
                j2, i2 = jj[sub], ii[sub]
                if j2.size == 0:
                    continue
                for (a, b) in table:
                    seg_a.append(eid(a, j2, i2))
                    seg_b.append(eid(b, j2, i2))
        else:
            for (a, b) in _CASES[c]:
                seg_a.append(eid(a, jj, ii))
                seg_b.append(eid(b, jj, ii))
    if not seg_a:
        return []
    S = np.stack([np.concatenate(seg_a), np.concatenate(seg_b)], axis=1).astype(np.int64)
    m = S.shape[0]

    # compact edge ids (only edges that actually carry a crossing)
    uniq, inv, counts = np.unique(S.ravel(), return_inverse=True, return_counts=True)
    inv = inv.reshape(m, 2)
    E = uniq.shape[0]

    # crossing positions per compact edge (canvas px)
    is_h = uniq < nh
    px = np.empty(E)
    py = np.empty(E)
    hid = uniq[is_h]
    hj, hi = hid // stride_h, hid % stride_h
    a = F[hj, hi]
    b = F[hj, hi + 1]
    u = (t - a) / (b - a)
    px[is_h] = (hi + u) * s
    py[is_h] = hj * s
    vid = uniq[~is_h] - nh
    vj, vi = vid // ni, vid % ni
    a = F[vj, vi]
    b = F[vj + 1, vi]
    u = (t - a) / (b - a)
    px[~is_h] = vi * s
    py[~is_h] = (vj + u) * s

    # adjacency: each compact edge touches at most two segments
    e = inv.ravel()
    seg_of = np.repeat(np.arange(m, dtype=np.int64), 2)
    order = np.argsort(e, kind="stable")
    es, ss = e[order], seg_of[order]
    first = np.searchsorted(es, np.arange(E), side="left")
    adj0 = ss[first]
    adj1 = np.full(E, -1, dtype=np.int64)
    two = counts >= 2
    adj1[two] = ss[first[two] + 1]

    visited = np.zeros(m, dtype=bool)
    adj0l, adj1l = adj0.tolist(), adj1.tolist()
    S0l, S1l = inv[:, 0].tolist(), inv[:, 1].tolist()
    vis = visited.tolist()

    def walk(start_edge, start_seg):
        path = [start_edge]
        cur_e, cur_s = start_edge, start_seg
        while True:
            vis[cur_s] = True
            other = S1l[cur_s] if S0l[cur_s] == cur_e else S0l[cur_s]
            path.append(other)
            nxt = adj0l[other] if adj0l[other] != cur_s else adj1l[other]
            if nxt < 0 or vis[nxt]:
                break
            cur_e, cur_s = other, nxt
        return path

    lines = []
    for e0 in np.nonzero(counts == 1)[0].tolist():
        s0 = adj0l[e0]
        if vis[s0]:
            continue
        lines.append((walk(e0, s0), False))
    for k in range(m):
        if vis[k]:
            continue
        path = walk(S0l[k], k)
        closed = path[0] == path[-1]
        if closed:
            path = path[:-1]
        lines.append((path, closed))

    out = []
    for path, closed in lines:
        ids = np.asarray(path, dtype=np.int64)
        pts = np.stack([px[ids], py[ids]], axis=1)
        if pts.shape[0] < 3:
            continue
        out.append((pts, bool(closed)))
    return out


def build_contours(F, levels, stride):
    """Spec section 4 API: per level {'z','t','index','lines'}."""
    out = []
    for (z, t, is_index) in levels:
        out.append({"z": float(z), "t": float(t), "index": bool(is_index),
                    "lines": trace_level(F, t, stride)})
    return out


# ---------------------------------------------------------------------------
# Polyline helpers
# ---------------------------------------------------------------------------

def _closed_pts(pts, closed):
    if closed and pts.shape[0] > 1:
        return np.concatenate([pts, pts[:1]], axis=0)
    return pts


def _cumlen(pts):
    d = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    return np.concatenate([[0.0], np.cumsum(d)])


def polyline_length(pts, closed=False):
    p = _closed_pts(np.asarray(pts, dtype=np.float64), closed)
    if p.shape[0] < 2:
        return 0.0
    return float(_cumlen(p)[-1])


def arc_samples(pts, closed, positions):
    """Vectorised arc sampling: positions (k,) arc lengths (clamped) ->
    (x, y, tx, ty) arrays; tangent is the containing segment's unit direction."""
    p = _closed_pts(np.asarray(pts, dtype=np.float64), closed)
    pos = np.asarray(positions, dtype=np.float64)
    if p.shape[0] < 2:
        n = pos.shape[0]
        return (np.full(n, p[0, 0]), np.full(n, p[0, 1]), np.ones(n), np.zeros(n))
    c = _cumlen(p)
    L = c[-1]
    sv = np.clip(pos, 0.0, L)
    k = np.searchsorted(c, sv, side="right") - 1
    k = np.clip(k, 0, p.shape[0] - 2)
    seg = c[k + 1] - c[k]
    u = np.where(seg > 0, (sv - c[k]) / np.where(seg > 0, seg, 1.0), 0.0)
    dx = p[k + 1, 0] - p[k, 0]
    dy = p[k + 1, 1] - p[k, 1]
    x = p[k, 0] + u * dx
    y = p[k, 1] + u * dy
    n = np.hypot(dx, dy)
    ok = n > 0
    tx = np.where(ok, dx / np.where(ok, n, 1.0), 1.0)
    ty = np.where(ok, dy / np.where(ok, n, 1.0), 0.0)
    return x, y, tx, ty


def point_at_arc(pts, s, closed=False):
    """Position and unit tangent at arc length s (clamped). Returns (x, y, tx, ty)."""
    x, y, tx, ty = arc_samples(pts, closed, [float(s)])
    return float(x[0]), float(y[0]), float(tx[0]), float(ty[0])


def split_by_gaps(pts, closed, gaps):
    """Cut a polyline into sub-polylines, dropping the arc ranges in gaps
    [(s0, s1), ...]. Returns list of (n,2) arrays (open)."""
    p = _closed_pts(np.asarray(pts, dtype=np.float64), closed)
    if p.shape[0] < 2:
        return [p]
    c = _cumlen(p)
    L = c[-1]
    keep = []
    pos = 0.0
    for (g0, g1) in sorted(gaps):
        g0, g1 = max(0.0, g0), min(L, g1)
        if g1 <= pos:
            continue
        if g0 > pos:
            keep.append((pos, g0))
        pos = max(pos, g1)
    if pos < L:
        keep.append((pos, L))
    out = []
    for (s0, s1) in keep:
        if s1 - s0 <= 1e-9:
            continue
        x0, y0, _, _ = point_at_arc(p, s0)
        x1, y1, _, _ = point_at_arc(p, s1)
        k0 = int(np.searchsorted(c, s0, side="right"))
        k1 = int(np.searchsorted(c, s1, side="left"))
        mid = p[k0:k1]
        sub = np.concatenate([[[x0, y0]], mid, [[x1, y1]]], axis=0)
        out.append(sub)
    return out


def _inward_normals(p):
    """p closed ring (n,2) without duplicate end. Returns per-edge unit inward normals (n,2)."""
    q = np.concatenate([p, p[:1]], axis=0)
    dx = np.diff(q[:, 0])
    dy = np.diff(q[:, 1])
    area = 0.5 * float(np.sum(q[:-1, 0] * q[1:, 1] - q[1:, 0] * q[:-1, 1]))
    n = np.hypot(dx, dy)
    n[n == 0] = 1.0
    if area > 0:
        nx, ny = -dy / n, dx / n
    else:
        nx, ny = dy / n, -dx / n
    return np.stack([nx, ny], axis=1)


def depression_flag(pts, closed, gx, gy, stride):
    """Spec 5c: closed AND mean(downhill . inward normal) > 0."""
    if not closed:
        return False
    p = np.asarray(pts, dtype=np.float64)
    if p.shape[0] < 3:
        return False
    nrm = _inward_normals(p)
    q = np.concatenate([p, p[:1]], axis=0)
    mids = 0.5 * (q[:-1] + q[1:])
    ex, ey = grad_at(gx, gy, mids[:, 0], mids[:, 1], stride)
    g = np.hypot(ex, ey)
    ok = g >= 1e-12
    if not np.any(ok):
        return False
    dots = (-ex[ok] / g[ok]) * nrm[ok, 0] + (-ey[ok] / g[ok]) * nrm[ok, 1]
    return bool(dots.mean() > 0.0)


# ---------------------------------------------------------------------------
# 5c. Hachures
# ---------------------------------------------------------------------------

def _F_at(F, xs, ys, stride):
    j = np.clip(np.rint(np.asarray(ys, dtype=np.float64) / stride).astype(np.int64), 0, F.shape[0] - 1)
    i = np.clip(np.rint(np.asarray(xs, dtype=np.float64) / stride).astype(np.int64), 0, F.shape[1] - 1)
    return F[j, i]


def hachure_ticks(contours, gx, gy, stride, mode, spacing, length, F=None):
    """Spec 5c. F (the work grid) enables the never-uphill drop; when omitted it
    is reconstructed from nothing and the drop is skipped (pure-geometry callers)."""
    if mode == "off" or spacing <= 0 or length <= 0:
        return []
    F_ref = F
    ticks = []
    for lev in contours:
        for (pts, closed) in lev["lines"]:
            if mode == "depressions" and not depression_flag(pts, closed, gx, gy, stride):
                continue
            L = polyline_length(pts, closed)
            pos = np.arange(spacing / 2.0, L + 1e-9, spacing)
            if pos.size == 0:
                continue
            x, y, _tx, _ty = arc_samples(pts, closed, pos)
            ex, ey = grad_at(gx, gy, x, y, stride)
            g = np.hypot(ex, ey)
            ok = g >= 1e-6
            if not np.any(ok):
                continue
            dx = -ex[ok] / g[ok]
            dy = -ey[ok] / g[ok]
            x0, y0 = x[ok], y[ok]
            x1 = x0 + length * dx
            y1 = y0 + length * dy
            # a hachure never points uphill: a tick whose END sits on a higher
            # work vertex than its start (it crossed a ridge within its own
            # length; ~0.2 % of ticks on a photo) is dropped, so the downhill
            # promise holds for every drawn tick by construction.
            if F_ref is not None:
                keep = _F_at(F_ref, x1, y1, stride) <= _F_at(F_ref, x0, y0, stride)
            else:
                keep = np.ones(x0.shape[0], dtype=bool)
            for a, b, c, d in zip(x0[keep].tolist(), y0[keep].tolist(), x1[keep].tolist(), y1[keep].tolist()):
                ticks.append(((a, b), (c, d)))
    return ticks


# ---------------------------------------------------------------------------
# 5d. Slope arrows
# ---------------------------------------------------------------------------

def slope_arrows(gx, gy, stride, w, h, spacing, length):
    """Spec 5d: regular grid, downhill direction, L = length * (0.35 + 0.65 * min(1, g/g_ref)),
    g_ref = 75th percentile of |grad F|; sites with g < 0.05*g_ref draw nothing."""
    if spacing <= 0 or length <= 0:
        return []
    mag = np.hypot(gx, gy)
    if mag.size == 0:
        return []
    g_ref = float(np.percentile(mag, 75))
    if g_ref <= 0.0:
        return []
    out = []
    y = spacing / 2.0
    while y < h:
        x = spacing / 2.0
        while x < w:
            ex, ey = sample_grad(gx, gy, x, y, stride)
            g = math.hypot(ex, ey)
            if g >= 0.05 * g_ref:
                L = length * (0.35 + 0.65 * min(1.0, g / g_ref))
                dx, dy = -ex / g, -ey / g
                out.append(((x, y), (x + L * dx, y + L * dy), L))
            x += spacing
        y += spacing
    return out


# ---------------------------------------------------------------------------
# 5e. Spot heights, 5f. Benchmarks
# ---------------------------------------------------------------------------

def _extrema(F):
    """Strict 8-neighbour maxima / minima on the interior. Returns (peaks (j,i), pits (j,i))."""
    nj, ni = F.shape
    if nj < 3 or ni < 3:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 2), dtype=np.int64)
    c = F[1:-1, 1:-1]
    gt = np.ones_like(c, dtype=bool)
    lt = np.ones_like(c, dtype=bool)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            if dj == 0 and di == 0:
                continue
            nb = F[1 + dj:nj - 1 + dj, 1 + di:ni - 1 + di]
            gt &= c > nb
            lt &= c < nb
    pj, pi = np.nonzero(gt)
    qj, qi = np.nonzero(lt)
    return (np.stack([pj + 1, pi + 1], axis=1), np.stack([qj + 1, qi + 1], axis=1))


def spot_heights(F, stride, count, min_dist, elev_min, elev_max):
    """Returns list of (x, y, z, kind) with kind in {'peak', 'pit'}; greedy by
    prominence proxy (F for peaks, 1-F for pits), min_dist apart, up to count."""
    if count <= 0:
        return []
    F = np.asarray(F, dtype=np.float64)
    peaks, pits = _extrema(F)
    cands = []
    for (j, i) in peaks.tolist():
        cands.append((float(F[j, i]), 0, j, i, "peak"))
    for (j, i) in pits.tolist():
        cands.append((1.0 - float(F[j, i]), 1, j, i, "pit"))
    # descending key, deterministic tie-break on (kind, j, i)
    cands.sort(key=lambda c: (-c[0], c[1], c[2], c[3]))
    emin, emax = float(elev_min), float(elev_max)
    if emax < emin:
        emin, emax = emax, emin
    acc = []
    md2 = float(min_dist) ** 2
    for (_k, _o, j, i, kind) in cands:
        x, y = i * stride, j * stride
        ok = True
        for (ax, ay, _z, _kd) in acc:
            if (ax - x) ** 2 + (ay - y) ** 2 < md2:
                ok = False
                break
        if not ok:
            continue
        z = emin + float(F[j, i]) * (emax - emin)
        acc.append((float(x), float(y), z, kind))
        if len(acc) >= count:
            break
    return acc


def benchmarks(seed, w, h, count, min_dist, spots, F, stride, elev_min, elev_max):
    """Spec 5f: own MINSTD stream 'benchmarks'; margin 0.08*min(w,h); u then v;
    accepted iff >= min_dist from every spot and accepted benchmark; at most
    50*count draws. Returns list of (x, y, z)."""
    if count <= 0:
        return []
    rng = engine.MinstdLCG(element_seed(seed, "benchmarks"))
    m = 0.08 * min(w, h)
    emin, emax = float(elev_min), float(elev_max)
    if emax < emin:
        emin, emax = emax, emin
    md2 = float(min_dist) ** 2
    acc = []
    others = [(float(sp[0]), float(sp[1])) for sp in spots]  # (x, y, ...) tuples of any length
    for _ in range(50 * int(count)):
        u = rng.draw()
        v = rng.draw()
        x = m + u * (w - 2 * m)
        y = m + v * (h - 2 * m)
        ok = True
        for (ox, oy) in others:
            if (ox - x) ** 2 + (oy - y) ** 2 < md2:
                ok = False
                break
        if ok:
            for (ax, ay, _z) in acc:
                if (ax - x) ** 2 + (ay - y) ** 2 < md2:
                    ok = False
                    break
        if not ok:
            continue
        z = emin + sample_F(F, x, y, stride) * (emax - emin)
        acc.append((float(x), float(y), z))
        if len(acc) >= count:
            break
    return acc


# ---------------------------------------------------------------------------
# 5g. Grid, 5i. Scale bar, 5j. Annotations, 5h. Bands
# ---------------------------------------------------------------------------

def grid_lines(w, h, spacing):
    if spacing <= 0:
        return [], []
    xs, ys = [], []
    k = 1
    while k * spacing < w:
        xs.append(float(k * spacing))
        k += 1
    k = 1
    while k * spacing < h:
        ys.append(float(k * spacing))
        k += 1
    return xs, ys


def text_runs(pts, closed, advances, gap, lead):
    """
    Spec v2 5b-bis: glyph placements for text repeated along a polyline.
    advances: list of per-glyph advance widths (px) for ONE repetition of the
    text; gap: px between repetitions; lead: px left free at both ends.
    Returns (placements [(x, y, angle_deg)] one per glyph, gaps [(s0, s1)] one
    per repetition (the arc the line must not be drawn under)).
    """
    L = polyline_length(pts, closed)
    total = float(sum(advances))
    if total <= 0 or L < total + 2 * lead:
        return [], []
    places, gaps = [], []
    s = lead
    while s + total <= L - lead + 1e-9:
        centres = []
        acc = s
        for a in advances:
            centres.append(acc + a / 2.0)
            acc += a
        xs, ys, txs, tys = arc_samples(pts, closed, centres)
        for k in range(len(advances)):
            ang = -math.degrees(math.atan2(float(tys[k]), float(txs[k])))
            places.append((float(xs[k]), float(ys[k]), ang))
        gaps.append((s - 0.35 * lead, s + total + 0.35 * lead))
        s += total + gap
    return places, gaps


def parse_annotations(s):
    """'x,y:TEXT;x,y:TEXT' -> list of (x, y, text); malformed entries skipped
    with one console note."""
    out = []
    bad = 0
    if not s:
        return out
    for entry in str(s).split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            bad += 1
            continue
        coords, text = entry.split(":", 1)
        parts = coords.split(",")
        if len(parts) != 2:
            bad += 1
            continue
        try:
            x = float(parts[0].strip())
            y = float(parts[1].strip())
        except ValueError:
            bad += 1
            continue
        text = text.strip()
        if not text:
            bad += 1
            continue
        out.append((x, y, text))
    if bad:
        print(f"[Schematic] Schematic Topographic: skipped {bad} malformed annotation "
              "entr" + ("y" if bad == 1 else "ies") + " (expected 'x,y:TEXT;...')")
    return out


def scale_bar_metres(target_px, map_scale):
    """Largest {1,2,5}*10^n metres <= target_px*map_scale. Returns (metres, px)."""
    if target_px <= 0 or map_scale <= 0:
        return 0.0, 0.0
    target_m = float(target_px) * float(map_scale)
    best = None
    for n in range(-3, 9):
        for base in (1.0, 2.0, 5.0):
            v = base * (10.0 ** n)
            if v <= target_m and (best is None or v > best):
                best = v
    if best is None:
        best = 0.001
    return best, best / float(map_scale)


def band_quantised(H, n_levels):
    n = max(1, int(n_levels))
    return np.clip(np.floor(np.asarray(H, dtype=np.float64) * n) / n, 0.0, 1.0)
