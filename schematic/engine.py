"""
Pure-numpy analysis + geometry engine for Schematic Overlay.
No torch, no PIL — see docs/schematic-derivation.md for the frozen spec this
file implements section-by-section (cited in each docstring/comment).

Kept import-safe standalone: `sys.path.insert(0, pack_root); from schematic import engine`
works with nothing but numpy installed.
"""

import math

import numpy as np


# ---------------------------------------------------------------------------
# 1. Canvas / cover-fit (spec section 1)
# ---------------------------------------------------------------------------

def _bilinear_resample(img, src_x, src_y, src_w, src_h, out_w, out_h):
    """
    Bilinear-resample the sub-rect [src_x, src_x+src_w) x [src_y, src_y+src_h)
    of `img` (H,W,C float) to exactly out_w x out_h.
    """
    img_h, img_w = img.shape[0], img.shape[1]
    out_w = max(1, int(round(out_w)))
    out_h = max(1, int(round(out_h)))

    xs = src_x + (np.arange(out_w, dtype=np.float64) + 0.5) * (src_w / out_w) - 0.5
    ys = src_y + (np.arange(out_h, dtype=np.float64) + 0.5) * (src_h / out_h) - 0.5
    xs = np.clip(xs, 0.0, img_w - 1)
    ys = np.clip(ys, 0.0, img_h - 1)

    x0 = np.floor(xs).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, img_w - 1)
    y0 = np.floor(ys).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, img_h - 1)

    fx = (xs - x0).astype(np.float64)[None, :, None]
    fy = (ys - y0).astype(np.float64)[:, None, None]

    Ia = img[np.ix_(y0, x0)]
    Ib = img[np.ix_(y0, x1)]
    Ic = img[np.ix_(y1, x0)]
    Id = img[np.ix_(y1, x1)]

    top = Ia * (1.0 - fx) + Ib * fx
    bot = Ic * (1.0 - fx) + Id * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float32)


def cover_fit(img, canvas_w, canvas_h):
    """
    Crop-to-fill (cover), centered, then bilinear-resample to canvas_w x canvas_h.
    img: (H,W,C) float array (any range; caller decides 0-1 vs 0-255).
    Spec section 1.
    """
    img_h, img_w = img.shape[0], img.shape[1]
    ia = img_w / img_h
    ca = canvas_w / canvas_h
    if ia > ca:
        src_h = float(img_h)
        src_w = src_h * ca
        src_x = (img_w - src_w) / 2.0
        src_y = 0.0
    else:
        src_w = float(img_w)
        src_h = src_w / ca
        src_x = 0.0
        src_y = (img_h - src_h) / 2.0
    return _bilinear_resample(img, src_x, src_y, src_w, src_h, canvas_w, canvas_h)


def stretch_resize(img, out_w, out_h):
    """Non-cropping stretch (bilinear) of the whole image to out_w x out_h. Used for texture."""
    img_h, img_w = img.shape[0], img.shape[1]
    return _bilinear_resample(img, 0.0, 0.0, float(img_w), float(img_h), out_w, out_h)


def nearest_resample_mask(mask_bool, out_w, out_h):
    """Nearest-neighbor resize of a boolean (H,W) mask to out_w x out_h. Spec section 8(b)."""
    h0, w0 = mask_bool.shape
    if h0 == out_h and w0 == out_w:
        return mask_bool
    ys = np.floor((np.arange(out_h, dtype=np.float64) + 0.5) * h0 / out_h).astype(np.int64)
    xs = np.floor((np.arange(out_w, dtype=np.float64) + 0.5) * w0 / out_w).astype(np.int64)
    ys = np.clip(ys, 0, h0 - 1)
    xs = np.clip(xs, 0, w0 - 1)
    return mask_bool[np.ix_(ys, xs)]


# ---------------------------------------------------------------------------
# 2. Block analysis (spec section 2)
# ---------------------------------------------------------------------------

def analyze_blocks(fitted01, block_size):
    """
    fitted01: (H,W,3) float in 0-1. Returns dict with 'brightness', 'contrast',
    'x', 'y' arrays (n_blocks,) in row-major grid order (remainder pixels dropped).
    """
    h, w = fitted01.shape[0], fitted01.shape[1]
    bs = int(block_size)
    cols = w // bs
    rows = h // bs
    if cols <= 0 or rows <= 0:
        empty = np.zeros((0,), dtype=np.float64)
        return {"brightness": empty, "contrast": empty, "x": empty, "y": empty,
                "rows": rows, "cols": cols}

    img255 = fitted01[: rows * bs, : cols * bs, :].astype(np.float64) * 255.0
    luma = 0.299 * img255[..., 0] + 0.587 * img255[..., 1] + 0.114 * img255[..., 2]
    # (rows,bs,cols,bs) -> (rows,cols,bs,bs) -> (rows*cols, bs*bs), row-major grid order
    blocks = luma.reshape(rows, bs, cols, bs).transpose(0, 2, 1, 3).reshape(rows * cols, bs * bs)

    mean = blocks.mean(axis=1)
    meansq = (blocks * blocks).mean(axis=1)
    contrast = np.sqrt(np.maximum(0.0, meansq - mean * mean))

    n = rows * cols
    row_idx = np.arange(n) // cols
    col_idx = np.arange(n) % cols
    x = col_idx * bs + bs / 2.0
    y = row_idx * bs + bs / 2.0
    return {"brightness": mean, "contrast": contrast, "x": x, "y": y, "rows": rows, "cols": cols}


def score_blocks(blocks, mode):
    """Spec section 4 scoring. mode in {contrast, bright, dark, combined}."""
    brightness = blocks["brightness"]
    contrast = blocks["contrast"]
    if mode == "contrast":
        return contrast.copy()
    if mode == "bright":
        return brightness.copy()
    if mode == "dark":
        return 255.0 - brightness
    # combined (default)
    return contrast * (1.0 + np.abs(brightness - 128.0) / 128.0)


# ---------------------------------------------------------------------------
# 3. Seeded RNG — Park-Miller MINSTD LCG (spec section 3)
# ---------------------------------------------------------------------------

class MinstdLCG:
    """
    Park-Miller MINSTD LCG, exact per spec section 3.
    state = abs(seed) or 1; draw(): state = state*16807 % 2147483647; return (state-1)/2147483646.
    """

    MULT = 16807
    MOD = 2147483647

    def __init__(self, seed):
        s = abs(int(seed))
        self.state = s if s != 0 else 1

    def draw(self):
        self.state = (self.state * self.MULT) % self.MOD
        return (self.state - 1) / (self.MOD - 1)


def _modpow_array(base, exp_arr, mod):
    """Vectorized base**exp_arr % mod (square-and-multiply, base is a python int scalar)."""
    result = np.ones_like(exp_arr, dtype=np.int64)
    b = int(base) % mod
    e = exp_arr.copy()
    while np.any(e > 0):
        odd = (e & 1).astype(bool)
        if np.any(odd):
            result[odd] = (result[odd] * b) % mod
        b = (b * b) % mod
        e = e >> 1
    return result


def lcg_draws_array(seed, n):
    """
    Vectorized equivalent of calling MinstdLCG(seed).draw() n times in a row,
    returned as an (n,) float64 array in emission order. Used only where N is
    large (procedural grain) — exact parity with the sequential class is a
    verification item in tools/test_schematic.py.
    """
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    s0 = abs(int(seed))
    if s0 == 0:
        s0 = 1
    exps = np.arange(1, n + 1, dtype=np.int64)
    powmod = _modpow_array(MinstdLCG.MULT, exps, MinstdLCG.MOD)
    states = (powmod * s0) % MinstdLCG.MOD
    return (states - 1).astype(np.float64) / (MinstdLCG.MOD - 1)


# ---------------------------------------------------------------------------
# 4. Scoring and placement (spec section 4)
# ---------------------------------------------------------------------------

def place_circles(blocks, score, threshold, max_circles, min_distance, min_radius, max_radius, seed):
    """Returns list of (x, y, r, score) tuples in acceptance order. Spec section 4."""
    xs = blocks["x"]
    ys = blocks["y"]
    n = xs.shape[0]
    if n == 0 or score.size == 0:
        return []

    max_score = max(float(np.max(score)), 1.0)
    norm = score / max_score
    keep = norm >= (threshold / 100.0)
    idx = np.nonzero(keep)[0]
    if idx.size == 0:
        return []

    # stable sort descending by norm; ties keep grid (ascending index) order
    order = idx[np.argsort(-norm[idx], kind="stable")]

    rng = MinstdLCG(seed)
    min_d2 = float(min_distance) * float(min_distance)
    max_circles = int(max_circles)

    acc_x = np.empty(max(max_circles, 1), dtype=np.float64)
    acc_y = np.empty(max(max_circles, 1), dtype=np.float64)
    n_acc = 0
    result = []

    for i in order:
        if n_acc >= max_circles:
            break
        x = float(xs[i])
        y = float(ys[i])
        if n_acc > 0:
            dx = acc_x[:n_acc] - x
            dy = acc_y[:n_acc] - y
            if np.any(dx * dx + dy * dy < min_d2):
                continue
        jitter = 0.5 + rng.draw()
        s = float(norm[i])
        r = (min_radius + (max_radius - min_radius) * s) * jitter
        result.append((x, y, r, s))
        acc_x[n_acc] = x
        acc_y[n_acc] = y
        n_acc += 1

    return result


# ---------------------------------------------------------------------------
# 5. Connections (spec section 5)
# ---------------------------------------------------------------------------

def compute_connections(circles, connection_dist):
    """All unique (i<j) pairs with distance < connection_dist. Spec section 5."""
    conns = []
    if connection_dist <= 0 or len(circles) < 2:
        return conns
    d2 = float(connection_dist) * float(connection_dist)
    n = len(circles)
    for i in range(n):
        xi, yi = circles[i][0], circles[i][1]
        for j in range(i + 1, n):
            xj, yj = circles[j][0], circles[j][1]
            dx = xi - xj
            dy = yi - yj
            if dx * dx + dy * dy < d2:
                conns.append((i, j))
    return conns


# ---------------------------------------------------------------------------
# 6. Chain geometry + intersections (spec section 6)
# ---------------------------------------------------------------------------

def compute_chain(canvas_w, canvas_h, count, angle_deg, base_radius, ratio):
    """
    Returns list of (x, y, r) tuples: [center, plus-side in walk order, minus-side
    in walk order]. Array order preserved exactly per spec section 6.
    """
    count = int(count)
    if count <= 0:
        return []
    cx, cy = canvas_w / 2.0, canvas_h / 2.0
    theta = math.radians(angle_deg - 90.0)
    dx, dy = math.cos(theta), math.sin(theta)

    center = (cx, cy, float(base_radius))
    chain = [center]

    n_plus = (count - 1) // 2
    n_minus = count // 2  # ceil((count-1)/2)

    cur = center
    for _ in range(n_plus):
        nx = cur[0] + dx * cur[2]
        ny = cur[1] + dy * cur[2]
        nr = cur[2] * ratio
        cur = (nx, ny, nr)
        chain.append(cur)

    cur = center
    for _ in range(n_minus):
        nx = cur[0] - dx * cur[2]
        ny = cur[1] - dy * cur[2]
        nr = cur[2] * ratio
        cur = (nx, ny, nr)
        chain.append(cur)

    return chain


def two_circle_intersections(c1, c2):
    """
    Standard two-circle intersection, exact point order per spec section 6:
    [(midx + h*dy/d, midy - h*dx/d), (midx - h*dy/d, midy + h*dx/d)] where dx,dy = c2-c1.
    Returns [] if the circles don't intersect (or are coincident).
    """
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    dx = x2 - x1
    dy = y2 - y1
    d = math.hypot(dx, dy)
    if d == 0 or d > r1 + r2 or d < abs(r1 - r2):
        return []
    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h2 = r1 * r1 - a * a
    if h2 < 0:
        return []
    h = math.sqrt(h2)
    midx = x1 + a * dx / d
    midy = y1 + a * dy / d
    p1 = (midx + h * dy / d, midy - h * dx / d)
    p2 = (midx - h * dy / d, midy + h * dx / d)
    return [p1, p2]


def compute_chain_intersections(chain):
    """
    Consecutive ARRAY pairs (i, i+1) — including the intentional non-tangent
    pair across the plus/minus boundary. Spec section 6.
    """
    points = []
    for i in range(len(chain) - 1):
        points.extend(two_circle_intersections(chain[i], chain[i + 1]))
    return points


# ---------------------------------------------------------------------------
# 8. Pixelate zone parsing (spec section 8)
# ---------------------------------------------------------------------------

def parse_zone_string(s):
    """'x,y;x,y;...' -> list of (float x, float y). Invalid pairs skipped."""
    zones = []
    if not s:
        return zones
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        pieces = part.split(",")
        if len(pieces) != 2:
            continue
        try:
            x = float(pieces[0].strip())
            y = float(pieces[1].strip())
        except ValueError:
            continue
        zones.append((x, y))
    return zones


def label_mask_8conn(mask_bool):
    """
    8-connected component labeling, pure numpy (no scipy). Vectorized per-row
    run detection + union-find merge across rows (checks column offsets -1/0/1
    for 8-connectivity). Returns int64 label array (0 = background).
    """
    h, w = mask_bool.shape
    m = mask_bool.astype(np.int8)
    if not m.any():
        return np.zeros((h, w), dtype=np.int64)

    starts = (m == 1) & np.concatenate(
        [np.ones((h, 1), dtype=bool), m[:, :-1] == 0], axis=1
    )
    run_id = np.cumsum(starts.reshape(-1)).reshape(h, w)
    run_id = np.where(m == 1, run_id, 0).astype(np.int64)

    max_id = int(run_id.max())
    parent = list(range(max_id + 1))

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for y in range(1, h):
        row = run_id[y]
        prev = run_id[y - 1]
        for shift in (-1, 0, 1):
            if shift == -1:
                a = row[1:]
                b = prev[:-1]
            elif shift == 1:
                a = row[:-1]
                b = prev[1:]
            else:
                a = row
                b = prev
            valid = (a > 0) & (b > 0)
            aa = a[valid]
            bb = b[valid]
            for k in range(aa.shape[0]):
                union(int(aa[k]), int(bb[k]))

    lookup = np.zeros(max_id + 1, dtype=np.int64)
    for i in range(1, max_id + 1):
        lookup[i] = find(i)
    # canonicalize to compact 1..K ids (not required, but keeps labels tidy)
    return lookup[run_id]


def mask_zone_centers(mask_bool):
    """Each 8-connected component's centroid (mean member coords) is a zone center."""
    labels = label_mask_8conn(mask_bool)
    ys, xs = np.nonzero(labels)
    if ys.size == 0:
        return []
    lbls = labels[ys, xs]
    ids = np.unique(lbls)
    centers = []
    for lid in ids:
        sel = lbls == lid
        cx = float(xs[sel].mean())
        cy = float(ys[sel].mean())
        centers.append((cx, cy))
    return centers


# ---------------------------------------------------------------------------
# Palettes (spec section 10)
# ---------------------------------------------------------------------------

PALETTES = {
    "whiteOnDark": ("#0a0a0a", "#d8d8d8"),
    "blackOnLight": ("#f0eeea", "#111111"),
    "goldOnDark": ("#111110", "#c9a84c"),
    "greenOnDark": ("#080a08", "#44cc66"),
}


def parse_hex(s):
    """Accept '#rrggbb' or 'rrggbb'. Returns (r,g,b) int tuple or None on failure."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        return None
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        return None


def resolve_palette(palette, bg_color, stroke_color):
    """Returns ((r,g,b) bg, (r,g,b) stroke). Falls back to whiteOnDark on parse failure."""
    if palette == "custom":
        bg = parse_hex(bg_color)
        stroke = parse_hex(stroke_color)
        if bg is None or stroke is None:
            print("[Schematic] warning: invalid custom hex color, falling back to whiteOnDark")
            bg = parse_hex(PALETTES["whiteOnDark"][0])
            stroke = parse_hex(PALETTES["whiteOnDark"][1])
        return bg, stroke
    if palette not in PALETTES:
        print(f"[Schematic] warning: unknown palette '{palette}', falling back to whiteOnDark")
        palette = "whiteOnDark"
    bgh, sh = PALETTES[palette]
    return parse_hex(bgh), parse_hex(sh)


# ---------------------------------------------------------------------------
# Procedural grain fallback (spec section 7 step 8 / section 13)
# ---------------------------------------------------------------------------

def _box_blur_1px(img2d):
    """3x3 uniform box blur (radius 1), edge-replicated padding."""
    padded = np.pad(img2d, 1, mode="edge")
    out = np.zeros_like(img2d, dtype=np.float64)
    h, w = img2d.shape
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out += padded[dy:dy + h, dx:dx + w]
    return out / 9.0


def generate_grain(seed, w, h):
    """
    Mono gaussian noise (mean 0.5, std 0.15) via Box-Muller from a MINSTD(seed)
    instance, 1px box blur. Returns (h,w,3) float32 0-1 (grayscale broadcast to RGB).
    """
    n_pixels = w * h
    n_pairs = (n_pixels + 1) // 2
    n_draws = n_pairs * 2
    u = lcg_draws_array(seed, n_draws)
    u = np.clip(u, 1e-12, 1.0)
    u1 = u[0::2][:n_pairs]
    u2 = u[1::2][:n_pairs]
    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    z0 = r * np.cos(theta)
    z1 = r * np.sin(theta)
    z = np.empty(n_pairs * 2, dtype=np.float64)
    z[0::2] = z0
    z[1::2] = z1
    z = z[:n_pixels]

    noise = np.clip(0.5 + 0.15 * z, 0.0, 1.0).reshape(h, w)
    noise = _box_blur_1px(noise)
    noise = np.clip(noise, 0.0, 1.0)
    return np.stack([noise, noise, noise], axis=-1).astype(np.float32)
