"""
Shared draw substrate for Schematic Overlay and its Phase 1 sibling nodes.
Extracted 1:1 (verbatim, zero logic edits) from schematic/render.py.
"""

import math
import os

import numpy as np
from PIL import ImageFont

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
