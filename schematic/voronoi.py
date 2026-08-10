"""
Pure-numpy density + seed-sampling + Voronoi-geometry engine for Schematic
Voronoi. No torch, no PIL — see docs/schematic-voronoi-derivation.md for the
frozen spec this file implements section-by-section (cited per function).

scipy.spatial is imported LAZILY, inside build_cells only — the module itself
stays scipy-free at import time (numpy + the sibling `engine` module, itself
scipy-free, are the only top-level imports), so the pack loads and every
sibling node runs even if scipy ever vanished. scipy only enters
sys.modules once build_cells actually runs.

Kept import-safe standalone: `sys.path.insert(0, pack_root); from schematic
import voronoi` works with nothing but numpy installed.
"""

import numpy as np

from . import engine


# ---------------------------------------------------------------------------
# Local box filter (pure numpy, no scipy.ndimage) — used only by the "detail"
# density source. 2D prefix-sum (cumsum) sliding-window mean, edge-replicate
# padding (same padding convention as engine._box_blur_1px).
# ---------------------------------------------------------------------------

def _odd(n):
    """Round a value up to the nearest odd integer (n already >= 1 by caller)."""
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _box_filter_mean(arr, k):
    """
    k x k uniform box filter mean via 2D prefix sums, edge-replicate padding.
    Pure numpy — deliberately NOT scipy.ndimage.uniform_filter, so this stays
    usable before scipy is ever imported. k must be odd, k >= 1.
    """
    if k <= 1:
        return arr.astype(np.float64, copy=True)
    r = k // 2
    padded = np.pad(arr, ((r, r), (r, r)), mode="edge").astype(np.float64)

    c0 = np.cumsum(padded, axis=0)
    p0 = np.concatenate([np.zeros((1, c0.shape[1]), dtype=np.float64), c0], axis=0)
    row_sum = p0[k:, :] - p0[:-k, :]                      # (h, w + 2r)

    c1 = np.cumsum(row_sum, axis=1)
    p1 = np.concatenate([np.zeros((c1.shape[0], 1), dtype=np.float64), c1], axis=1)
    box_sum = p1[:, k:] - p1[:, :-k]                      # (h, w)

    return box_sum / float(k * k)


def _local_std_box(luma01, k):
    """Local population std of luma01 in a k x k window (box-filter mean/mean-of-squares)."""
    m = _box_filter_mean(luma01, k)
    m2 = _box_filter_mean(luma01 * luma01, k)
    return np.sqrt(np.maximum(0.0, m2 - m * m))


# ---------------------------------------------------------------------------
# 2. Density field (spec section 2)
# ---------------------------------------------------------------------------

def build_density(fitted01, source, gamma):
    """
    fitted01: (H,W,3) float array, 0-1 (the cover-fitted source, same array
    that feeds the render). source: "brightness" | "darkness" | "detail".
    Returns (H,W) float64 density d(x,y) = normalized_field ** gamma, in [0,1].
    Spec section 2.
    """
    img255 = np.asarray(fitted01, dtype=np.float64) * 255.0
    luma = 0.299 * img255[..., 0] + 0.587 * img255[..., 1] + 0.114 * img255[..., 2]
    luma01 = luma / 255.0

    if source == "brightness":
        field = luma01
    elif source == "darkness":
        field = 1.0 - luma01
    elif source == "detail":
        h, w = luma01.shape
        s = min(w, h)
        k = _odd(max(3, round(0.006 * s)))
        field = _local_std_box(luma01, k)
    else:
        raise ValueError(f"unknown density_source: {source!r}")

    fmax = float(field.max()) if field.size else 0.0
    if fmax < 1e-9:
        # guard: max < 1e-9 -> all-zero field (spec section 3 degenerate trigger)
        normalized = np.zeros_like(field, dtype=np.float64)
    else:
        normalized = field / fmax

    density = np.power(np.clip(normalized, 0.0, 1.0), gamma)
    return density.astype(np.float64)


# ---------------------------------------------------------------------------
# 3. Seeds — deterministic MINSTD rejection sampling (spec section 3)
# ---------------------------------------------------------------------------

def sample_seeds(density, cells, seed, w, h):
    """
    One engine.MinstdLCG(seed) stream, draws consumed in strict (x, y, u)
    triples: x = draw()*w, y = draw()*h, accept iff u = draw() < density at
    the (floored, clamped) pixel under (x,y). Stops at `cells` accepted OR at
    the fixed budget of 80*cells triples (deterministic termination on any
    input). Spec section 3. Returns (n,2) float64, n <= cells.
    """
    cells = int(cells)
    if cells <= 0:
        return np.zeros((0, 2), dtype=np.float64)

    dens = np.asarray(density, dtype=np.float64)
    dens_h, dens_w = dens.shape[0], dens.shape[1]
    if dens_h <= 0 or dens_w <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    # Nested python lists: plain scalar indexing in the hot loop is
    # meaningfully faster than repeated numpy scalar gathers here (same
    # values either way — pure implementation speedup, this loop runs up to
    # 80*cells times).
    dens_rows = dens.tolist()

    rng = engine.MinstdLCG(seed)
    draw = rng.draw  # cache the bound method (same object/state, call-site speedup only)

    w = float(w)
    h = float(h)
    x_max_idx = dens_w - 1
    y_max_idx = dens_h - 1
    budget = 80 * cells

    pts = np.empty((cells, 2), dtype=np.float64)
    n = 0
    triples = 0

    while n < cells and triples < budget:
        x = draw() * w
        y = draw() * h
        u = draw()
        triples += 1

        xi = int(x)
        if xi > x_max_idx:
            xi = x_max_idx
        elif xi < 0:
            xi = 0
        yi = int(y)
        if yi > y_max_idx:
            yi = y_max_idx
        elif yi < 0:
            yi = 0

        if u < dens_rows[yi][xi]:
            pts[n, 0] = x
            pts[n, 1] = y
            n += 1

    return pts[:n].copy()


# ---------------------------------------------------------------------------
# 4. Geometry — scipy.spatial.Voronoi on mirror-closed seeds (spec section 4)
# ---------------------------------------------------------------------------

def build_cells(seeds, w, h):
    """
    Reflect the n accepted seeds across each canvas edge (5n points total),
    run scipy.spatial.Voronoi, and extract the n REAL cells (index < n) as
    finite polygons with shoelace areas, plus the drawable ridge list. Spec
    section 4. Pure; lazy-imports scipy.spatial (not touched at module import).

    Returns dict(regions=list[n] of (k,2) float64 vertex arrays, areas=(n,)
    float64, ridges=list[((x1,y1),(x2,y2), i, j)] with i/j = adjacent REAL
    seed indices or -1).
    """
    from scipy.spatial import Voronoi  # lazy: module import must stay scipy-free

    seeds = np.asarray(seeds, dtype=np.float64).reshape(-1, 2)
    n = seeds.shape[0]
    W = float(w)
    H = float(h)

    refl_x0 = seeds * np.array([-1.0, 1.0])                              # x -> -x
    refl_x1 = seeds * np.array([-1.0, 1.0]) + np.array([2.0 * W, 0.0])   # x -> 2W - x
    refl_y0 = seeds * np.array([1.0, -1.0])                              # y -> -y
    refl_y1 = seeds * np.array([1.0, -1.0]) + np.array([0.0, 2.0 * H])   # y -> 2H - y
    allpts = np.vstack([seeds, refl_x0, refl_x1, refl_y0, refl_y1])

    vor = Voronoi(allpts)

    regions = []
    areas = np.zeros(n, dtype=np.float64)
    for i in range(n):
        region = vor.regions[vor.point_region[i]]
        if len(region) == 0 or -1 in region:
            # Unreachable given the mirror construction (named invariant, spec
            # section 4: zero real cells with an infinite region) — kept as a
            # numerical safety net so build_cells stays total rather than
            # raising on a pathological/degenerate seed configuration.
            regions.append(np.zeros((0, 2), dtype=np.float64))
            continue
        poly = vor.vertices[region]
        regions.append(poly)
        x = poly[:, 0]
        y = poly[:, 1]
        areas[i] = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    ridges = []
    for (p1, p2), rv in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in rv:
            continue
        i_real = int(p1) if p1 < n else -1
        j_real = int(p2) if p2 < n else -1
        if i_real == -1 and j_real == -1:
            continue
        v1 = vor.vertices[rv[0]]
        v2 = vor.vertices[rv[1]]
        ridges.append(((float(v1[0]), float(v1[1])), (float(v2[0]), float(v2[1])), i_real, j_real))

    return {"regions": regions, "areas": areas, "ridges": ridges}


# ---------------------------------------------------------------------------
# Ridge intensity — log-area percentile map (spec section 4)
# ---------------------------------------------------------------------------

def ridge_alpha(areas, ridge_real_indices):
    """
    Per-ridge intensity from adjacent real cell areas, spec section 4:
    t(i) = (log A_i - log A_p2) / (log A_p98 - log A_p2), clamped [0,1], where
    A_p2/A_p98 are the 2nd/98th percentiles of `areas` (log scale — areas span
    decades). alpha = 1.0 - 0.85 * min(t of adjacent real cells), clamped
    [0.15, 1.0] (the smaller/denser neighbor wins).

    areas: (n,) float64 real-cell areas (build_cells()['areas']).
    ridge_real_indices: array-like (n_ridges, 2) int — REAL seed index or -1
    per side (the (i, j) tail of each build_cells()['ridges'] entry).
    Returns (n_ridges,) float64 in [0.15, 1.0].
    """
    areas = np.asarray(areas, dtype=np.float64).reshape(-1)
    idx = np.asarray(ridge_real_indices, dtype=np.int64).reshape(-1, 2)
    n_ridges = idx.shape[0]
    if n_ridges == 0:
        return np.zeros((0,), dtype=np.float64)

    a_lo = float(np.percentile(areas, 2)) if areas.size else 0.0
    a_hi = float(np.percentile(areas, 98)) if areas.size else 0.0
    lo = np.log(a_lo + 1e-9)
    hi = np.log(a_hi + 1e-9)
    denom = max(hi - lo, 1e-9)

    log_areas = np.log(areas + 1e-9) if areas.size else np.zeros((0,), dtype=np.float64)
    t_per_cell = np.clip((log_areas - lo) / denom, 0.0, 1.0)

    i_idx = idx[:, 0]
    j_idx = idx[:, 1]
    i_real = i_idx >= 0
    j_real = j_idx >= 0

    # gather, masking non-real sides to +inf so they never win the min()
    safe_i = np.maximum(i_idx, 0)
    safe_j = np.maximum(j_idx, 0)
    t_i = np.where(i_real, t_per_cell[safe_i], np.inf)
    t_j = np.where(j_real, t_per_cell[safe_j], np.inf)
    t_min = np.minimum(t_i, t_j)

    alpha = np.clip(1.0 - 0.85 * t_min, 0.15, 1.0)
    return alpha.astype(np.float64)
