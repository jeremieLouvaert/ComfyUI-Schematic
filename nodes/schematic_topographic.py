"""
Schematic Topographic node for ComfyUI-Schematic.

The photo's smoothed luma is the terrain; the node draws a complete topographic
map sheet over it: contour lines, index contours with inline elevation labels,
depression hachures, spot heights, benchmarks, CanvasCamp campsite badges,
slope arrows, grid with edge
coordinates, hypsometric band tone, north arrow, scale bar, contour-interval
note and user annotations. See docs/schematic-topographic-derivation.md for
the frozen spec. torch only lives in this file (the tensor bridge);
schematic/topographic.py and schematic/render_topographic.py stay torch-free.
"""

import os

import numpy as np
import torch

from ..schematic import engine, topographic as topo, render_topographic

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


def _prep_height_mask(dm):
    """Voronoi density_mask prep, verbatim semantics (spec section 3 step 1)."""
    if dm is None:
        return None
    if hasattr(dm, "cpu"):
        dm = dm.cpu().numpy()
    dm = np.asarray(dm, dtype=np.float32)
    while dm.ndim > 2 and dm.shape[-1] == 1:
        dm = dm[..., 0]
    if dm.ndim < 2:
        print("[Schematic] height_mask has no 2-D content, treating as absent")
        return None
    dm = dm.reshape((-1,) + dm.shape[-2:])
    if dm.shape[0] == 0:
        print("[Schematic] empty height_mask batch, treating as absent")
        return None
    if dm.shape[0] > 1:
        print(f"[Schematic] height_mask batch {dm.shape[0]} frames, using frame 0")
    m = np.nan_to_num(dm[0], nan=0.0)
    if m.shape[0] < 8 or m.shape[1] < 8:
        print(f"[Schematic] height_mask resolves to {m.shape} which is not a "
              "plausible mask, treating as absent")
        return None
    return np.clip(m.astype(np.float64), 0.0, 1.0)


def _canvas_mask(m, w, h):
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


class SchematicTopographic:
    """Topographic map-sheet overlay: the photo is the terrain. See docs/schematic-topographic-derivation.md."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "height_source": (["procedural", "brightness", "darkness", "detail"], {
                    "default": "procedural",
                    "tooltip": "Where the terrain comes from. Procedural (default) = a seeded landform "
                               "that has nothing to do with the photo - the sheet is an overlay. "
                               "Brightness / darkness / detail derive it from the photo instead."
                }),
                "terrain_scale": ("INT", {
                    "default": 420, "min": 40, "max": 3000, "step": 10,
                    "tooltip": "Procedural terrain: size (px) of the largest landforms."
                }),
                "terrain_octaves": ("INT", {
                    "default": 3, "min": 1, "max": 6, "step": 1,
                    "tooltip": "Procedural terrain: detail layers. 1 = broad swells only; more adds "
                               "finer wrinkles to the contours."
                }),
                "terrain_roughness": ("FLOAT", {
                    "default": 0.45, "min": 0.1, "max": 0.9, "step": 0.05,
                    "tooltip": "Procedural terrain: how much each finer layer contributes (gain)."
                }),
                "terrain_stretch": ("FLOAT", {
                    "default": 1.6, "min": 1.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Procedural terrain: elongates landforms along terrain_angle so contours "
                               "flow as ridges and valleys. 1 = isotropic blobs."
                }),
                "terrain_angle": ("FLOAT", {
                    "default": 30.0, "min": 0.0, "max": 180.0, "step": 5.0,
                    "tooltip": "Procedural terrain: direction (degrees) of the ridge/valley flow."
                }),
                "smoothing": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 40.0, "step": 0.5,
                    "tooltip": "Gaussian blur sigma (px) on the height field before contouring. Higher "
                               "= broader, calmer contours; 0 = raw pixels (noisy)."
                }),
                "contour_resolution": ("INT", {
                    "default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Pixels per contour-tracing cell. 1 = finest (slowest); 2 is "
                               "indistinguishable at default smoothing and twice as fast."
                }),
                "elevation_min": ("FLOAT", {
                    "default": 0.0, "min": -10000.0, "max": 10000.0, "step": 1.0,
                    "tooltip": "Elevation (m) of the lowest point of the field. Labels, spot heights "
                               "and benchmarks read in this range."
                }),
                "elevation_max": ("FLOAT", {
                    "default": 1000.0, "min": -10000.0, "max": 10000.0, "step": 1.0,
                    "tooltip": "Elevation (m) of the highest point of the field."
                }),
                "contour_interval": ("FLOAT", {
                    "default": 40.0, "min": 1.0, "max": 500.0, "step": 1.0,
                    "tooltip": "Vertical distance (m) between contour lines - the contour DENSITY. "
                               "0-1000 m at 40 m = 24 lines; 20 m = 49."
                }),
                "contour_weight": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "Stroke width (px) of regular contours; also the base width of "
                               "hachures, arrows, symbols, grid ticks and marginalia."
                }),
                "index_every": ("INT", {
                    "default": 5, "min": 0, "max": 20, "step": 1,
                    "tooltip": "Every Nth contour (counted in absolute metres, so 200, 400... at "
                               "40 m x 5) is an INDEX contour: heavier and labelled. 0 = none."
                }),
                "index_weight": ("FLOAT", {
                    "default": 2.2, "min": 1.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Index contour width as a multiple of contour_weight."
                }),
                "contour_labels": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Inline elevation labels on index contours, rotated along the line, "
                               "with the line broken under each label."
                }),
                "label_size": ("INT", {
                    "default": 9, "min": 4, "max": 48, "step": 1,
                    "tooltip": "Font size (px) of contour labels, spot heights, benchmarks and grid "
                               "coordinates."
                }),
                "label_spacing": ("INT", {
                    "default": 320, "min": 60, "max": 2000, "step": 10,
                    "tooltip": "Distance (px) along an index contour between labels; contours "
                               "shorter than this get no label."
                }),
                "label_unit": ("STRING", {
                    "default": "",
                    "tooltip": "Suffix after every contour label, e.g. ' m'. Empty = bare numbers."
                }),
                "contour_text": ("STRING", {
                    "default": "",
                    "tooltip": "Text repeated along the contours, glyph by glyph, following the line "
                               "(the calendar-poster device). Empty = off. Replaces the numeric labels "
                               "on the lines it runs on."
                }),
                "contour_text_on": (["index", "all"], {
                    "default": "index",
                    "tooltip": "Run contour_text on the index contours only, or on every contour."
                }),
                "contour_text_size": ("INT", {
                    "default": 11, "min": 4, "max": 48, "step": 1,
                    "tooltip": "Font size (px) of contour_text."
                }),
                "contour_text_gap": ("INT", {
                    "default": 60, "min": 0, "max": 1000, "step": 5,
                    "tooltip": "Gap (px) between repetitions of contour_text along a line."
                }),
                "hachures": (["off", "depressions", "all"], {
                    "default": "depressions",
                    "tooltip": "Short downhill ticks on contours. Depressions = only on closed "
                               "contours enclosing LOWER ground (the map-sheet convention). All = "
                               "every contour."
                }),
                "hachure_spacing": ("INT", {
                    "default": 10, "min": 2, "max": 100, "step": 1,
                    "tooltip": "Distance (px) between hachure ticks along a contour - tick DENSITY."
                }),
                "hachure_length": ("INT", {
                    "default": 6, "min": 1, "max": 60, "step": 1,
                    "tooltip": "Length (px) of each hachure tick."
                }),
                "slope_arrows": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Regular grid of arrows pointing downhill, length proportional to "
                               "slope (terrain-analysis aspect arrows)."
                }),
                "arrow_spacing": ("INT", {
                    "default": 90, "min": 20, "max": 400, "step": 5,
                    "tooltip": "Grid pitch (px) of the slope arrows - arrow DENSITY."
                }),
                "arrow_length": ("INT", {
                    "default": 22, "min": 4, "max": 120, "step": 1,
                    "tooltip": "Length (px) of the steepest arrow; flatter sites draw shorter, "
                               "near-flat sites draw none."
                }),
                "spot_heights": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Mark local peaks (triangle) and pits (circle) with their elevation."
                }),
                "spot_count": ("INT", {
                    "default": 12, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Maximum spot heights, most prominent first."
                }),
                "spot_min_distance": ("INT", {
                    "default": 120, "min": 10, "max": 800, "step": 5,
                    "tooltip": "Minimum spacing (px) between spot heights and benchmarks."
                }),
                "symbol_size": ("INT", {
                    "default": 8, "min": 2, "max": 40, "step": 1,
                    "tooltip": "Size (px) of the peak triangle, pit circle and benchmark cross."
                }),
                "benchmarks": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Survey benchmarks: an x cross with 'BM' + elevation, at seeded "
                               "positions kept clear of the spot heights."
                }),
                "benchmark_count": ("INT", {
                    "default": 4, "min": 0, "max": 40, "step": 1,
                    "tooltip": "Number of benchmarks (the only element that uses the seed)."
                }),
                "grid": (["off", "lines", "ticks", "crosses", "dots"], {
                    "default": "ticks",
                    "tooltip": "Map grid: full lines, edge ticks only, edge ticks + interior crosses "
                               "(the UTM-sheet look), or a dot at every intersection."
                }),
                "grid_spacing": ("INT", {
                    "default": 200, "min": 20, "max": 1000, "step": 10,
                    "tooltip": "Grid pitch (px) - grid DENSITY."
                }),
                "grid_weight": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "Stroke width (px) of grid lines, ticks and crosses."
                }),
                "grid_labels": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Coordinate labels (m) along the bottom and right edges, from "
                               "map_scale; skipped near the corners to keep the frame text clear."
                }),
                "map_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 1000.0, "step": 0.01,
                    "tooltip": "Metres per canvas pixel. Drives the grid coordinates and the scale bar."
                }),
                "band_fill": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.05,
                    "tooltip": "Hypsometric plate tone: stepped stroke-colour fill between contours, "
                               "stronger with elevation. 0 disables."
                }),
                "north_arrow": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "North arrow with 'N' at the top-right, under the frame text."
                }),
                "scale_bar": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Graphic scale bar at the bottom-left (a round 1-2-5 length from "
                               "map_scale), above the frame text."
                }),
                "interval_note": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "'CONTOUR INTERVAL n M' note under the scale bar."
                }),
                "marginalia_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.3, "max": 4.0, "step": 0.1,
                    "tooltip": "Size multiplier for the north arrow, scale bar, interval note "
                               "offsets and annotation leaders."
                }),
                "annotations": ("STRING", {
                    "default": "",
                    "tooltip": "Custom call-outs as 'x,y:TEXT;x,y:TEXT' in canvas pixels: a dot at "
                               "the point, a 45-degree leader with a horizontal tail, text at the end."
                }),
                "pixel_size": ("INT", {
                    "default": 16, "min": 1, "max": 80, "step": 1,
                    "tooltip": "Mosaic block size of the pixelate zones - larger = blockier; 1 disables. "
                               "Identical to Schematic Overlay's."
                }),
                "zone_size": ("INT", {
                    "default": 100, "min": 10, "max": 500, "step": 1,
                    "tooltip": "Half-width of each square pixelation zone"
                }),
                "pixel_stroke": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Draw an outline around each pixelation zone"
                }),
                "pixelate_zones": ("STRING", {
                    "default": "",
                    "tooltip": "Pixelation zone centers as 'x,y;x,y;...' in canvas pixels. "
                               "Union with pixelate_mask centroids. Alters the underlying photo only."
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xFFFFFFFF, "step": 1,
                    "tooltip": "Seed for the procedural terrain, benchmark placement and the grain texture."
                }),
                "overlay_opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Master opacity for every drawn element."
                }),
                "image_opacity": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Fade the background photo under the map sheet."
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
                "pixelate_mask": ("MASK", {
                    "tooltip": "Optional mask; each 8-connected white region's centroid becomes a "
                               "pixelation zone center (as Schematic Overlay)"
                }),
                "texture": ("IMAGE", {
                    "tooltip": "Optional texture/noise image, screen-blended over the canvas. If empty "
                               "and texture_opacity > 0, a procedural grain fallback is used"
                }),
                "height_mask": ("MASK", {
                    "tooltip": "Optional mask used AS the terrain height instead of height_source "
                               "(white = high). Smoothing still applies. Cover-fitted like the photo "
                               "when sizes differ; frame 0 used for batches."
                }),
                "campsites": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Scatter CanvasCamp campsite badges over the sheet, pitched on flat "
                               "ground (the flatter half of the terrain), clear of spot heights and "
                               "benchmarks."
                }),
                "campsite_count": ("INT", {
                    "default": 6, "min": 0, "max": 60, "step": 1,
                    "tooltip": "Maximum campsites; fewer when flat ground runs out at "
                               "spot_min_distance spacing. Placement follows the seed."
                }),
                "campsite_size": ("INT", {
                    "default": 22, "min": 8, "max": 160, "step": 1,
                    "tooltip": "Diameter (px) of each campsite badge."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "overlay_only", "overlay_alpha")
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Schematic"

    def execute(self, image, height_source, terrain_scale, terrain_octaves, terrain_roughness,
                terrain_stretch, terrain_angle, smoothing, contour_resolution, elevation_min,
                elevation_max, contour_interval, contour_weight, index_every, index_weight,
                contour_labels, label_size, label_spacing, label_unit, contour_text,
                contour_text_on, contour_text_size, contour_text_gap, hachures, hachure_spacing,
                hachure_length, slope_arrows, arrow_spacing, arrow_length, spot_heights,
                spot_count, spot_min_distance, symbol_size, benchmarks, benchmark_count, grid,
                grid_spacing, grid_weight, grid_labels, map_scale, band_fill, north_arrow,
                scale_bar, interval_note, marginalia_scale, annotations, pixel_size, zone_size,
                pixel_stroke, pixelate_zones, seed, overlay_opacity,
                image_opacity, texture_opacity, frame_text_size, frame_text_tl, frame_text_tr,
                frame_text_bl, frame_text_br, palette, bg_color, stroke_color, size_preset,
                pixelate_mask=None, texture=None, height_mask=None, campsites=False,
                campsite_count=6, campsite_size=22):

        if elevation_max <= elevation_min:
            print(f"[Schematic] Schematic Topographic: elevation_max ({elevation_max}) <= "
                  f"elevation_min ({elevation_min}), swapping")
            elevation_min, elevation_max = min(elevation_min, elevation_max), max(elevation_min, elevation_max)
            if elevation_max == elevation_min:
                elevation_max = elevation_min + 1.0

        levels = topo.contour_levels(elevation_min, elevation_max, contour_interval, index_every)
        print(f"[Schematic] Schematic Topographic: source={height_source}, interval={contour_interval:g}, "
              f"levels={len(levels)}, palette={palette}, size_preset={size_preset}, seed={seed}")

        bg_rgb, stroke_rgb = engine.resolve_palette(palette, bg_color, stroke_color)
        frames = tensor_to_numpy_batch(image)

        texture_np = None
        if texture is not None:
            tex_frames = tensor_to_numpy_batch(texture)
            if tex_frames:
                texture_np = tex_frames[0]

        mask01 = _prep_height_mask(height_mask)
        notes = topo.parse_annotations(annotations)

        pix_mask_bool = None
        if pixelate_mask is not None:
            pmk = pixelate_mask
            if hasattr(pmk, "cpu"):
                pmk = pmk.cpu().numpy()
            pmk = np.asarray(pmk, dtype=np.float32)
            if pmk.ndim == 3:
                pmk = pmk[0]
            pix_mask_bool = pmk > 0.5
        stride = int(contour_resolution)

        rparams = {
            "contour_weight": contour_weight, "index_weight": index_weight,
            "contour_labels": contour_labels, "label_size": label_size,
            "label_spacing": label_spacing, "label_unit": label_unit,
            "contour_text": contour_text, "contour_text_on": contour_text_on,
            "contour_text_size": contour_text_size, "contour_text_gap": contour_text_gap,
            "symbol_size": symbol_size, "campsite_size": campsite_size,
            "grid": grid, "grid_weight": grid_weight, "grid_labels": grid_labels,
            "map_scale": map_scale, "band_fill": band_fill, "north_arrow": north_arrow,
            "scale_bar": scale_bar, "interval_note": interval_note,
            "marginalia_scale": marginalia_scale, "pixel_size": pixel_size,
            "zone_size": zone_size, "pixel_stroke": pixel_stroke,
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

            canvas_mask = _canvas_mask(mask01, w, h) if mask01 is not None else None
            terrain = {"seed": seed, "scale": terrain_scale, "octaves": terrain_octaves,
                       "roughness": terrain_roughness, "stretch": terrain_stretch,
                       "angle": terrain_angle}
            H, flat = topo.build_height(fitted, height_source, smoothing, mask01=canvas_mask,
                                        terrain=terrain)

            if flat:
                print("[Schematic] Schematic Topographic: flat height field - rendering this "
                      "frame contour-free")
                contours, ticks, arrows, spots, bms, camps = [], [], [], [], [], []
            else:
                F = topo.work_grid(H, stride)
                gx, gy = topo.gradient(F)
                contours = topo.build_contours(F, levels, stride)
                ticks = topo.hachure_ticks(contours, gx, gy, stride, hachures,
                                           hachure_spacing, hachure_length, F=F)
                arrows = (topo.slope_arrows(gx, gy, stride, w, h, arrow_spacing, arrow_length)
                          if slope_arrows else [])
                spots = (topo.spot_heights(F, stride, spot_count, spot_min_distance,
                                           elevation_min, elevation_max) if spot_heights else [])
                bms = (topo.benchmarks(seed, w, h, benchmark_count, spot_min_distance, spots, F,
                                       stride, elevation_min, elevation_max) if benchmarks else [])
                camps = (topo.campsites(seed, w, h, campsite_count, spot_min_distance, campsite_size,
                                        list(spots) + list(bms), gx, gy, stride) if campsites else [])

            zones = engine.parse_zone_string(pixelate_zones)
            if pix_mask_bool is not None:
                zones = zones + engine.mask_zone_centers(engine.nearest_resample_mask(pix_mask_bool, w, h))

            xs, ys = topo.grid_lines(w, h, grid_spacing) if grid != "off" else ([], [])
            sbar = topo.scale_bar_metres(0.18 * w, map_scale) if scale_bar else None

            elements = {
                "H": None if flat else H, "n_levels": len(levels) + 1,
                "contours": contours, "hachures": ticks, "arrows": arrows,
                "spots": spots, "benchmarks": bms, "campsites": camps, "grid": (xs, ys),
                "annotations": notes, "scale_bar": sbar, "interval": contour_interval,
                "zones": zones,
            }

            texture_canvas = None
            if texture_np is not None:
                texture_canvas = engine.stretch_resize(texture_np, w, h)
            elif texture_opacity > 0.0:
                texture_canvas = engine.generate_grain(seed + 1, w, h)

            cache = {}  # vector layer + band raster shared by both passes of this frame
            img_out, alpha = render_topographic.render_frame_topographic(
                fitted, w, h, (bg_rgb, stroke_rgb), rparams, elements, texture_canvas, True, PACK_ROOT,
                cache=cache)
            overlay_out, _ = render_topographic.render_frame_topographic(
                fitted, w, h, (bg_rgb, stroke_rgb), rparams, elements, texture_canvas, False, PACK_ROOT,
                cache=cache)

            out_images.append(img_out.astype(np.float32) / 255.0)
            out_overlays.append(overlay_out.astype(np.float32) / 255.0)
            out_alphas.append(alpha)

        return (
            numpy_batch_to_tensor(out_images),
            numpy_batch_to_tensor(out_overlays),
            mask_batch_to_tensor(out_alphas),
        )


NODE_CLASS_MAPPINGS = {"SchematicTopographic": SchematicTopographic}
NODE_DISPLAY_NAME_MAPPINGS = {"SchematicTopographic": "Schematic Topographic"}
