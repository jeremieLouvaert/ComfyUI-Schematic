"""
Schematic Overlay node for ComfyUI-Schematic.

Single all-in-one node: clean-room reimplementation of the Circles mode from
Yordan Stoyanov's Brand Assets Generator (https://brand-generator.stoyanov.works/).
No code was copied — see docs/schematic-derivation.md for the frozen behavioral
spec this implements. torch only lives in this file (the tensor bridge);
schematic/engine.py and schematic/render.py stay torch-free.
"""

import os

import numpy as np
import torch

from ..schematic import engine, render

PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


SIZE_PRESETS = {
    "match input": None,
    "Portrait 3:4 (1200x1600)": (1200, 1600),
    "Square (1080x1080)": (1080, 1080),
    "Landscape 16:9 (1920x1080)": (1920, 1080),
    "Instagram Story (1080x1920)": (1080, 1920),
    "Poster (1400x2000)": (1400, 2000),
}


class SchematicOverlay:
    """All-in-one schematic/technical-annotation overlay generator. See docs/schematic-derivation.md."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detection_mode": (["combined", "contrast", "bright", "dark"], {
                    "default": "combined",
                    "tooltip": "Circle placement scoring mode. Combined = contrast x brightness deviation "
                               "(default). Contrast, Bright, Dark score blocks by that single measure."
                }),
                "block_size": ("INT", {
                    "default": 16, "min": 8, "max": 64, "step": 4,
                    "tooltip": "Analysis grid resolution - smaller = more detail"
                }),
                "threshold": ("INT", {
                    "default": 30, "min": 0, "max": 80, "step": 1,
                    "tooltip": "Min score to place a circle - higher = fewer, stronger hits"
                }),
                "max_circles": ("INT", {
                    "default": 80, "min": 20, "max": 400, "step": 1,
                    "tooltip": "Cap on total circles placed"
                }),
                "min_distance": ("INT", {
                    "default": 40, "min": 10, "max": 200, "step": 1,
                    "tooltip": "Spacing between circle centers - prevents clustering"
                }),
                "shape": (["circle", "square"], {
                    "default": "circle",
                    "tooltip": "Detection circle outline shape"
                }),
                "min_radius": ("INT", {
                    "default": 4, "min": 1, "max": 40, "step": 1,
                    "tooltip": "Smallest circle size (low-score regions)"
                }),
                "max_radius": ("INT", {
                    "default": 24, "min": 4, "max": 120, "step": 1,
                    "tooltip": "Largest circle size (high-score regions)"
                }),
                "circle_stroke": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "Circle outline thickness"
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xFFFFFFFF, "step": 1,
                    "tooltip": "Random seed for size variation - change for different distributions"
                }),
                "label_size": ("INT", {
                    "default": 8, "min": 4, "max": 48, "step": 1,
                    "tooltip": "Font size of x,y coordinate labels"
                }),
                "overlay_opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Master opacity for all circles, lines, dots, and text"
                }),
                "connection_dist": ("INT", {
                    "default": 150, "min": 0, "max": 500, "step": 10,
                    "tooltip": "Circles within this range get connected - 0 hides lines"
                }),
                "line_weight": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "Thickness of connecting lines"
                }),
                "chain_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable the tangent chain of circles radiating from canvas center"
                }),
                "chain_count": ("INT", {
                    "default": 11, "min": 0, "max": 40, "step": 1,
                    "tooltip": "Number of circles in the chain"
                }),
                "chain_angle": ("INT", {
                    "default": 45, "min": 0, "max": 90, "step": 1,
                    "tooltip": "Direction of chain - 0 vertical, 45 diagonal, 90 horizontal"
                }),
                "chain_base_radius": ("INT", {
                    "default": 250, "min": 20, "max": 1000, "step": 1,
                    "tooltip": "Size of the center circle"
                }),
                "chain_ratio": ("FLOAT", {
                    "default": 0.79, "min": 0.3, "max": 1.2, "step": 0.01,
                    "tooltip": "Size multiplier per step - below 1 shrinks outward"
                }),
                "chain_intersections": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Draw intersection markers where consecutive chain circles overlap"
                }),
                "chain_intersection_size": ("FLOAT", {
                    "default": 5.0, "min": 0.5, "max": 20.0, "step": 0.5,
                    "tooltip": "Dot size at each adjacent-circle intersection"
                }),
                "crosshair_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable the dashed crosshair, frame square, and center asterisk"
                }),
                "crosshair_frame_size": ("INT", {
                    "default": 60, "min": 10, "max": 100, "step": 1,
                    "tooltip": "Square size as % of canvas short side"
                }),
                "crosshair_dash": ("INT", {
                    "default": 8, "min": 1, "max": 40, "step": 1,
                    "tooltip": "Length of each dash + gap"
                }),
                "crosshair_stroke": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "Line thickness for frame and crosshair"
                }),
                "crosshair_star_size": ("INT", {
                    "default": 40, "min": 0, "max": 300, "step": 2,
                    "tooltip": "Diameter of the center asterisk - 0 hides it"
                }),
                "crosshair_star_points": ("INT", {
                    "default": 4, "min": 2, "max": 12, "step": 1,
                    "tooltip": "Number of crossing lines - 2 is a plus, 4 is an 8-pointed star"
                }),
                "pixel_size": ("INT", {
                    "default": 16, "min": 1, "max": 80, "step": 1,
                    "tooltip": "Mosaic block size - larger = blockier"
                }),
                "zone_size": ("INT", {
                    "default": 100, "min": 10, "max": 500, "step": 1,
                    "tooltip": "Half-width of each square pixelation zone"
                }),
                "pixel_stroke": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Draw an outline around each pixelation zone"
                }),
                "image_opacity": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Fade the background image"
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
                "size_preset": (["match input", "Portrait 3:4 (1200x1600)", "Square (1080x1080)",
                                  "Landscape 16:9 (1920x1080)", "Instagram Story (1080x1920)",
                                  "Poster (1400x2000)"], {
                    "default": "match input",
                    "tooltip": "Canvas size. 'match input' keeps the source image dimensions; presets "
                               "cover-fit (crop-to-fill, centered) the source before resizing"
                }),
                "bg_color": ("STRING", {
                    "default": "#0a0a0a",
                    "tooltip": "Custom palette background color (hex). Used when palette = custom"
                }),
                "stroke_color": ("STRING", {
                    "default": "#d8d8d8",
                    "tooltip": "Custom palette stroke color (hex). Used when palette = custom"
                }),
                "pixelate_zones": ("STRING", {
                    "default": "",
                    "tooltip": "Pixelation zone centers as 'x,y;x,y;...' in canvas pixels. "
                               "Union with pixelate_mask centroids"
                }),
            },
            "optional": {
                "pixelate_mask": ("MASK", {
                    "tooltip": "Optional mask; each 8-connected white region's centroid becomes a "
                               "pixelation zone center"
                }),
                "texture": ("IMAGE", {
                    "tooltip": "Optional texture/noise image, screen-blended over the canvas. If empty "
                               "and texture_opacity > 0, a procedural grain fallback is used"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "overlay_only", "overlay_alpha")
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Schematic"

    def execute(self, image, detection_mode, block_size, threshold, max_circles, min_distance,
                shape, min_radius, max_radius, circle_stroke, seed, label_size, overlay_opacity,
                connection_dist, line_weight, chain_enabled, chain_count, chain_angle,
                chain_base_radius, chain_ratio, chain_intersections, chain_intersection_size,
                crosshair_enabled, crosshair_frame_size, crosshair_dash, crosshair_stroke,
                crosshair_star_size, crosshair_star_points, pixel_size, zone_size, pixel_stroke,
                image_opacity, texture_opacity, frame_text_size, frame_text_tl, frame_text_tr,
                frame_text_bl, frame_text_br, palette, size_preset, bg_color, stroke_color,
                pixelate_zones, pixelate_mask=None, texture=None):

        print(f"[Schematic] Schematic Overlay: mode={detection_mode}, block_size={block_size}, "
              f"palette={palette}, size_preset={size_preset}, seed={seed}")

        bg_rgb, stroke_rgb = engine.resolve_palette(palette, bg_color, stroke_color)

        frames = tensor_to_numpy_batch(image)

        texture_np = None
        if texture is not None:
            tex_frames = tensor_to_numpy_batch(texture)
            if tex_frames:
                texture_np = tex_frames[0]  # frame 0 used for all frames (spec section 12)

        mask_bool = None
        if pixelate_mask is not None:
            m = pixelate_mask
            if hasattr(m, "cpu"):
                m = m.cpu().numpy()
            m = np.asarray(m, dtype=np.float32)
            if m.ndim == 3:
                m = m[0]
            mask_bool = m > 0.5

        rparams = {
            "shape": shape, "circle_stroke": circle_stroke, "label_size": label_size,
            "overlay_opacity": overlay_opacity, "line_weight": line_weight,
            "crosshair_enabled": crosshair_enabled, "crosshair_frame_size": crosshair_frame_size,
            "crosshair_dash": crosshair_dash, "crosshair_stroke": crosshair_stroke,
            "crosshair_star_size": crosshair_star_size, "crosshair_star_points": crosshair_star_points,
            "pixel_size": pixel_size, "zone_size": zone_size, "pixel_stroke": pixel_stroke,
            "image_opacity": image_opacity, "texture_opacity": texture_opacity,
            "frame_text_size": frame_text_size, "frame_text_tl": frame_text_tl,
            "frame_text_tr": frame_text_tr, "frame_text_bl": frame_text_bl, "frame_text_br": frame_text_br,
            "chain_intersection_size": chain_intersection_size, "seed": seed,
        }

        out_images, out_overlays, out_alphas = [], [], []

        for frame in frames:
            img_h, img_w = frame.shape[0], frame.shape[1]
            preset = SIZE_PRESETS.get(size_preset)
            if preset is None:
                w, h = img_w, img_h
            else:
                w, h = preset

            zones = engine.parse_zone_string(pixelate_zones)
            if mask_bool is not None:
                mb = engine.nearest_resample_mask(mask_bool, w, h)
                zones = zones + engine.mask_zone_centers(mb)

            no_pixelate = pixel_size <= 1 or len(zones) == 0
            no_texture = texture_opacity <= 0.0
            # overlay_opacity<=0 mathematically zeroes every alpha in steps 4,5,6,7,9
            # (every element's alpha is a multiple of op) -> equivalent to "all disabled".
            if (image_opacity >= 1.0 and overlay_opacity <= 0.0 and no_pixelate and no_texture
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

            blocks = engine.analyze_blocks(fitted, block_size)
            score = engine.score_blocks(blocks, detection_mode)
            circles = engine.place_circles(blocks, score, threshold, max_circles, min_distance,
                                            min_radius, max_radius, seed)
            connections = engine.compute_connections(circles, connection_dist)

            chain, intersections = [], []
            if chain_enabled and chain_count > 0:
                chain = engine.compute_chain(w, h, chain_count, chain_angle, chain_base_radius, chain_ratio)
                if chain_intersections:
                    intersections = engine.compute_chain_intersections(chain)

            elements = {
                "circles": circles, "connections": connections, "zones": zones,
                "chain": chain, "intersections": intersections,
            }

            texture_canvas = None
            if texture_np is not None:
                texture_canvas = engine.stretch_resize(texture_np, w, h)
            elif texture_opacity > 0.0:
                # generate once per frame; both render passes share identical grain
                texture_canvas = engine.generate_grain(seed + 1, w, h)

            img_out, alpha = render.render_frame(fitted, w, h, (bg_rgb, stroke_rgb), rparams,
                                                  elements, texture_canvas, True, PACK_ROOT)
            overlay_out, _ = render.render_frame(fitted, w, h, (bg_rgb, stroke_rgb), rparams,
                                                  elements, texture_canvas, False, PACK_ROOT)

            out_images.append(img_out.astype(np.float32) / 255.0)
            out_overlays.append(overlay_out.astype(np.float32) / 255.0)
            out_alphas.append(alpha)

        return (
            numpy_batch_to_tensor(out_images),
            numpy_batch_to_tensor(out_overlays),
            mask_batch_to_tensor(out_alphas),
        )


NODE_CLASS_MAPPINGS = {"SchematicOverlay": SchematicOverlay}
NODE_DISPLAY_NAME_MAPPINGS = {"SchematicOverlay": "Schematic Overlay"}
