"""
Schematic Voronoi node for ComfyUI-Schematic.

Density-driven Voronoi tessellation over the photo's own field: seed density
follows brightness/darkness/detail, cells draw as a fine wireframe mesh graded
by cell size (small/dense cells read brighter). See
docs/schematic-voronoi-derivation.md for the frozen behavioral spec this
implements, and docs/voronoi-density-mask-derivation.md for the optional
density_mask + mask_min amendment. torch only lives in this file (the tensor
bridge); schematic/voronoi.py and schematic/render_voronoi.py stay torch-free.
schematic/voronoi.py additionally stays scipy-free at import time (scipy.spatial
is lazy-imported inside build_cells) so this node still loads if scipy is
ever missing; only calling execute() with >=4 accepted seeds touches scipy.
"""

import os

import numpy as np
import torch

from ..schematic import engine, voronoi, render_voronoi

PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Local tensor-bridge helpers (node modules independent — not shared with
# nodes/schematic_overlay.py or the other sibling nodes).
# ---------------------------------------------------------------------------

def tensor_to_numpy_batch(tensor):
    """ComfyUI IMAGE tensor (B,H,W,C) -> list of (H,W,C) float32 numpy arrays."""
    return [tensor[i].cpu().numpy().astype(np.float32) for i in range(tensor.shape[0])]


def numpy_batch_to_tensor(arrays):
    """list of (H,W,C) float32 numpy arrays -> ComfyUI IMAGE tensor (B,H,W,C)."""
    tensors = [torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).unsqueeze(0) for arr in arrays]
    return torch.cat(tensors, dim=0)


def mask_batch_to_tensor(arrays):
    """list of (H,W) float32 numpy arrays -> ComfyUI MASK tensor (B,H,W)."""
    stacked = np.stack([np.ascontiguousarray(a, dtype=np.float32) for a in arrays], axis=0)
    return torch.from_numpy(stacked)


def _prep_density_mask(dm):
    """
    Node-level density_mask prep, docs/voronoi-density-mask-derivation.md
    section 4 (v2): squeeze ALL trailing size-1 dims -> reshape((-1,)+shape[-2:])
    -> empty-batch guard -> frame 0 (+ console note) -> nan_to_num -> the
    >=8-both-dims plausibility guard -> clip [0,1], float64. Ported verbatim
    from the certified prototype (_density_mask_dryrun/dryrun.py:prep_mask).

    Returns (H,W) float64 in the mask's OWN resolution (not yet canvas-mapped
    -- see _canvas_density_mask), or None (treated absent; absent is silent,
    matches the None-default path).
    """
    if dm is None:
        return None
    if hasattr(dm, "cpu"):
        dm = dm.cpu().numpy()
    dm = np.asarray(dm, dtype=np.float32)
    while dm.ndim > 2 and dm.shape[-1] == 1:
        dm = dm[..., 0]
    if dm.ndim < 2:
        print("[Schematic] density_mask has no 2-D content, treating as absent")
        return None
    dm = dm.reshape((-1,) + dm.shape[-2:])
    if dm.shape[0] == 0:
        print("[Schematic] empty density_mask batch, treating as absent")
        return None
    if dm.shape[0] > 1:
        print(f"[Schematic] density_mask batch {dm.shape[0]} frames, using frame 0")
    m = np.nan_to_num(dm[0], nan=0.0)
    if m.shape[0] < 8 or m.shape[1] < 8:
        print(f"[Schematic] density_mask resolves to {m.shape} which is not a "
              "plausible mask, treating as absent")
        return None
    return np.clip(m.astype(np.float64), 0.0, 1.0)


def _canvas_density_mask(m, w, h):
    """
    Canvas geometry, section 3: mask dims == canvas (w,h) exactly -> used
    AS-IS (no resample); otherwise engine.cover_fit, the SAME transform the
    photo itself goes through (mask lifted to (H,W,1)->3-channel for the
    call, channel 0 taken back out, clipped [0,1]).
    """
    if m.shape[0] == h and m.shape[1] == w:
        return m
    fitted = engine.cover_fit(np.repeat(m[..., None], 3, axis=2).astype(np.float32), w, h)
    return np.clip(fitted[..., 0].astype(np.float64), 0.0, 1.0)


SIZE_PRESETS = {
    "match input": None,
    "Portrait 3:4 (1200x1600)": (1200, 1600),
    "Square (1080x1080)": (1080, 1080),
    "Landscape 16:9 (1920x1080)": (1920, 1080),
    "Instagram Story (1080x1920)": (1080, 1920),
    "Poster (1400x2000)": (1400, 2000),
}


class SchematicVoronoi:
    """Density-following Voronoi mesh overlay. See docs/schematic-voronoi-derivation.md
    and docs/voronoi-density-mask-derivation.md (optional density_mask + mask_min)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "density_source": (["brightness", "darkness", "detail"], {
                    "default": "brightness",
                    "tooltip": "Field the seed mesh follows. Brightness = pixel luma - pair with "
                               "dark-ground palettes (whiteOnDark, goldOnDark, greenOnDark). Darkness = "
                               "inverted luma - pair with blackOnLight instead, so tonal values render "
                               "right-side-up. Detail = local contrast (std of luma) - outlines structure "
                               "on either ground."
                }),
                "cells": ("INT", {
                    "default": 12000, "min": 200, "max": 30000, "step": 100,
                    "tooltip": "Target seed count - the actual number of tessellation sites, not a "
                               "relative density scale. The same photo at higher canvas resolution keeps "
                               "the same mesh structure, scaled up with the canvas."
                }),
                "density_gamma": ("FLOAT", {
                    "default": 2.5, "min": 0.5, "max": 4.0, "step": 0.1,
                    "tooltip": "Contrast curve applied to the normalized field before seed sampling. "
                               "Higher = seeds concentrate more sharply into the hottest regions of the field."
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xFFFFFFFF, "step": 1,
                    "tooltip": "Random seed for seed placement (MINSTD rejection sampling). Same seed "
                               "used every frame in a batch."
                }),
                "mesh_weight": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1,
                    "tooltip": "Multiplies both ridge stroke widths (normal and the extra-dense tier) "
                               "before the 2x supersample scale - thicken or thin the wireframe without "
                               "changing seed density."
                }),
                "cell_fill": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.05,
                    "tooltip": "Optional flat fill under each cell's wireframe, brightness-scaled by that "
                               "cell's own field value (the mask's value when density_mask is wired - flat "
                               "under a binary mask; plate-tone under the wirework). 0 disables."
                }),
                "seed_dots": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Draw a small dot at each accepted seed location."
                }),
                "overlay_opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Master opacity for the mesh: cell fill, ridges, seed dots, and frame text."
                }),
                "image_opacity": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Fade the background photo. Low by default - the mesh IS the image here "
                               "(density follows the photo's own field), not an overlay drawn on top of it."
                }),
                "texture_opacity": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Opacity of the noise overlay (screen blend)"
                }),
                "frame_text_size": ("INT", {
                    "default": 12, "min": 6, "max": 72, "step": 1,
                    "tooltip": "Font size of the 4 corner labels"
                }),
                "frame_text_tl": ("STRING", {
                    "default": "Design & Strategy",
                    "tooltip": "Top-left corner frame text"
                }),
                "frame_text_tr": ("STRING", {
                    "default": "AKURATE",
                    "tooltip": "Top-right corner frame text"
                }),
                "frame_text_bl": ("STRING", {
                    "default": "",
                    "tooltip": "Bottom-left corner frame text"
                }),
                "frame_text_br": ("STRING", {
                    "default": "",
                    "tooltip": "Bottom-right corner frame text"
                }),
                "palette": (["whiteOnDark", "blackOnLight", "goldOnDark", "greenOnDark", "custom"], {
                    "default": "whiteOnDark",
                    "tooltip": "Color palette for background and stroke color"
                }),
                "bg_color": ("STRING", {
                    "default": "#0a0a0a",
                    "tooltip": "Custom palette background color (hex). Used when palette = custom"
                }),
                "stroke_color": ("STRING", {
                    "default": "#d8d8d8",
                    "tooltip": "Custom palette stroke color (hex). Used when palette = custom"
                }),
                "size_preset": (["match input", "Portrait 3:4 (1200x1600)", "Square (1080x1080)",
                                  "Landscape 16:9 (1920x1080)", "Instagram Story (1080x1920)",
                                  "Poster (1400x2000)"], {
                    "default": "match input",
                    "tooltip": "Canvas size. 'match input' keeps the source image dimensions; presets "
                               "cover-fit (crop-to-fill, centered) the source before resizing"
                }),
            },
            "optional": {
                "texture": ("IMAGE", {
                    "tooltip": "Optional texture/noise image, screen-blended over the canvas. If empty "
                               "and texture_opacity > 0, a procedural grain fallback is used"
                }),
                "density_mask": ("MASK", {
                    "tooltip": "Optional mask that DRIVES SEED DENSITY, replacing density_source "
                               "entirely: white = fine/dense cells, black = coarse. density_gamma "
                               "still applies. Only the mask's relative structure matters (it is "
                               "re-normalised to its own maximum: a uniform grey mask equals a white "
                               "one). At mask_min 0 black zones can render with NO ridges crossing "
                               "them - raise mask_min to keep a sparse mesh everywhere. Cover-fitted "
                               "like the photo when sizes differ; frame 0 used for batches (all frames "
                               "then share one tessellation); an empty or non-mask-shaped input is "
                               "ignored with a console note."
                }),
                "mask_min": ("FLOAT", {
                    "default": 0.01, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Minimum seed density where density_mask is black, as a fraction of "
                               "the maximum. 0 = the mask alone decides (black can go empty); default "
                               "0.01 keeps a sparse coarse mesh everywhere. Only active when "
                               "density_mask is wired."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "overlay_only", "overlay_alpha")
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Schematic"

    def execute(self, image, density_source, cells, density_gamma, seed, mesh_weight,
                cell_fill, seed_dots, overlay_opacity, image_opacity, texture_opacity,
                frame_text_size, frame_text_tl, frame_text_tr, frame_text_bl, frame_text_br,
                palette, bg_color, stroke_color, size_preset, texture=None,
                density_mask=None, mask_min=0.01):

        print(f"[Schematic] Schematic Voronoi: source={density_source}, cells={cells}, "
              f"gamma={density_gamma}, palette={palette}, size_preset={size_preset}, seed={seed}")

        bg_rgb, stroke_rgb = engine.resolve_palette(palette, bg_color, stroke_color)

        frames = tensor_to_numpy_batch(image)

        texture_np = None
        if texture is not None:
            tex_frames = tensor_to_numpy_batch(texture)
            if tex_frames:
                texture_np = tex_frames[0]  # frame 0 used for all frames

        # density_mask prep (docs/voronoi-density-mask-derivation.md section 4):
        # once per execute, frame 0 of the mask batch -- independent of the
        # image batch loop below (section 5: one tessellation shared by every
        # image frame when a mask is wired). density_mask=None keeps this a
        # no-op with zero console output (the byte-identical absent path).
        mask01 = _prep_density_mask(density_mask)

        rparams = {
            "cell_fill": cell_fill, "seed_dots": seed_dots, "mesh_weight": mesh_weight,
            "overlay_opacity": overlay_opacity, "image_opacity": image_opacity,
            "texture_opacity": texture_opacity, "frame_text_size": frame_text_size,
            "frame_text_tl": frame_text_tl, "frame_text_tr": frame_text_tr,
            "frame_text_bl": frame_text_bl, "frame_text_br": frame_text_br, "seed": seed,
        }

        out_images, out_overlays, out_alphas = [], [], []

        for frame in frames:
            img_h, img_w = frame.shape[0], frame.shape[1]
            preset = SIZE_PRESETS.get(size_preset)
            w, h = (img_w, img_h) if preset is None else preset

            no_texture = texture_opacity <= 0.0
            # overlay_opacity<=0 mathematically zeroes every alpha in the mesh
            # (cell fill, ridges, seed dots, frame text all scale by op) ->
            # equivalent to "every overlay feature disabled". Early-out mirror
            # of v0.1 section 13, with the canvas-size guard.
            if (image_opacity >= 1.0 and overlay_opacity <= 0.0 and no_texture
                    and w == img_w and h == img_h):
                out_images.append(frame.copy())
                bg_arr = np.empty((h, w, 3), dtype=np.float32)
                bg_arr[:, :, 0] = bg_rgb[0] / 255.0
                bg_arr[:, :, 1] = bg_rgb[1] / 255.0
                bg_arr[:, :, 2] = bg_rgb[2] / 255.0
                out_overlays.append(bg_arr)
                out_alphas.append(np.zeros((h, w), dtype=np.float32))
                continue

            fitted = engine.cover_fit(frame, w, h)

            if mask01 is None:
                density = voronoi.build_density(fitted, density_source, density_gamma)
            else:
                canvas_mask = _canvas_density_mask(mask01, w, h)
                density = voronoi.build_density(fitted, density_source, density_gamma,
                                                 mask01=canvas_mask, mask_min=mask_min)
            seeds = voronoi.sample_seeds(density, cells, seed, w, h)
            n = seeds.shape[0]

            if mask01 is not None:
                if float(density.max()) <= 0.0:
                    print("[Schematic] density_mask is entirely zero: no seeds can accept "
                          "(an alpha-less LoadImage mask does this)")
                elif n < cells:
                    print(f"[Schematic] density_mask: only {n}/{cells} seeds accepted "
                          "(budget exhausted) - a sparse mask can leave the mesh coarser than requested")

            if n < 4:
                # Degenerate path (spec section 3): SKIP the tessellation entirely,
                # render the frame mesh-free, print one note (QhullError-guard pattern).
                print(f"[Schematic] Schematic Voronoi: only {n} seed(s) accepted (< 4) - "
                      f"rendering this frame mesh-free")
                elements = {
                    "regions": [], "areas": np.zeros((0,), dtype=np.float64), "ridges": [],
                    "alphas": np.zeros((0,), dtype=np.float64),
                    "seeds": np.zeros((0, 2), dtype=np.float64),
                    "field_at_seeds": np.zeros((0,), dtype=np.float64),
                }
            else:
                cells_geo = voronoi.build_cells(seeds, w, h)
                ridge_idx = [(i, j) for (_p1, _p2, i, j) in cells_geo["ridges"]]
                alphas = voronoi.ridge_alpha(cells_geo["areas"], ridge_idx)

                ys = np.clip(seeds[:, 1].astype(np.int64), 0, h - 1)
                xs = np.clip(seeds[:, 0].astype(np.int64), 0, w - 1)
                field_at_seeds = density[ys, xs]

                elements = {
                    "regions": cells_geo["regions"], "areas": cells_geo["areas"],
                    "ridges": cells_geo["ridges"], "alphas": alphas,
                    "seeds": seeds, "field_at_seeds": field_at_seeds,
                }

            texture_canvas = None
            if texture_np is not None:
                texture_canvas = engine.stretch_resize(texture_np, w, h)
            elif texture_opacity > 0.0:
                # generate once per frame; both render passes share identical grain
                texture_canvas = engine.generate_grain(seed + 1, w, h)

            img_out, alpha = render_voronoi.render_frame_voronoi(
                fitted, w, h, (bg_rgb, stroke_rgb), rparams, elements, texture_canvas, True, PACK_ROOT)
            overlay_out, _ = render_voronoi.render_frame_voronoi(
                fitted, w, h, (bg_rgb, stroke_rgb), rparams, elements, texture_canvas, False, PACK_ROOT)

            out_images.append(img_out.astype(np.float32) / 255.0)
            out_overlays.append(overlay_out.astype(np.float32) / 255.0)
            out_alphas.append(alpha)

        return (
            numpy_batch_to_tensor(out_images),
            numpy_batch_to_tensor(out_overlays),
            mask_batch_to_tensor(out_alphas),
        )


NODE_CLASS_MAPPINGS = {"SchematicVoronoi": SchematicVoronoi}
NODE_DISPLAY_NAME_MAPPINGS = {"SchematicVoronoi": "Schematic Voronoi"}
