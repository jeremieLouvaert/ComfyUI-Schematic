"""Schematic Construction v2 - the annotation sibling of Schematic Overlay.

Spec: docs/schematic-construction-derivation-v2.md (signed by Jeremie 2026-08-14).
Supersedes the v0.1 apparatus; spec section 0 accounts for every banked element
that was kept, promoted, demoted or cut.

The photograph is composited ONCE and never re-rendered outside a pixelate zone.
Zones use Overlay's own `render.pixelate_canvas`, called verbatim, at pipeline
step 3 so the schematic is always drawn on top of them.
"""

import os

import numpy as np
import torch

from ..schematic import construction as geo
from ..schematic import engine
from ..schematic import render as overlay_render
from ..schematic import render_construction as rc
from ..schematic.elements import ElementState, ElementSyntaxError

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_GROUP_WHAT = {
    "g_zones": "the pixelate sample zones",
    "g_spheres": "the concentric spheres and their captions",
    "g_rays": "both ray pencils",
    "g_marks": "the computed intersections, their labels and the lune",
    "g_focus": "both focus marks and their labels",
    "g_scale": "the graduated scale and its numerals",
    "g_annot": "frame-exit labels, callouts, the data column, detection circles",
    "g_frame": "the four corner plate texts",
}


def _f(d, lo, hi, step, tip):
    return ("FLOAT", {"default": d, "min": lo, "max": hi, "step": step, "tooltip": tip})


def _i(d, lo, hi, tip):
    return ("INT", {"default": d, "min": lo, "max": hi, "tooltip": tip})


def _b(d, tip):
    return ("BOOLEAN", {"default": d, "tooltip": tip})


def _e(ch, d, tip):
    return (ch, {"default": d, "tooltip": tip})


def _s(d, tip, multiline=False):
    return ("STRING", {"default": d, "multiline": multiline, "tooltip": tip})


class SchematicConstruction:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The photograph the plate is drawn over. It is never altered outside a pixelate zone."}),
                "seed": _i(42, 0, 0x7FFFFFFF, "Chooses among valid LAYOUTS as well as content: focus positions within their legal bands, which detected points the rays aim through, sphere radii, which gap carries the lune, which intersections become zones."),

                "detection_mode": _e(["combined", "contrast", "bright", "dark"], "combined", "What the detector scores. Drives where every ray aims and where zones land."),
                "block_size": _i(16, 4, 96, "Detection block size in pixels, as Overlay's."),
                "threshold": _i(28, 0, 255, "Minimum block score to accept a point."),
                "max_circles": _i(26, 1, 200, "Maximum detected points the pencils can aim through."),
                "min_distance": _i(70, 1, 1000, "Minimum spacing between detected points, in pixels."),
                "min_radius": _i(8, 1, 500, "Smallest detected-point radius. Only visible when detection_circles is enabled in `elements`, since the circles are the one element that draws a point's radius."),
                "max_radius": _i(46, 1, 500, "Largest detected-point radius. Only visible when detection_circles is enabled in `elements`."),

                "focus_a_x": _f(0.15, 0.0, 1.0, 0.01, "Primary focus X, as a fraction of width. Everything converges here; the seed varies it within a band."),
                "focus_a_y": _f(0.88, 0.0, 1.0, 0.01, "Primary focus Y, as a fraction of height."),
                "focus_b_enabled": _b(True, "The second focus. The astronomical reference has two, and the second pencil is what produces the non-trivial intersections."),
                "focus_b_x": _f(0.82, 0.0, 1.0, 0.01, "Secondary focus X, as a fraction of width."),
                "focus_b_y": _f(0.14, 0.0, 1.0, 0.01, "Secondary focus Y, as a fraction of height."),
                "focus_clear": _f(0.055, 0.0, 0.3, 0.005, "Rays start this far out from the focus, as a fraction of the short side, so the focus stays clear instead of tangling. Applies identically to BOTH foci."),
                "rays_a": _i(30, 0, 200, "Rays in the primary pencil."),
                "rays_b": _i(20, 0, 200, "Rays in the secondary pencil."),
                "ray_jitter": _f(0.05, 0.0, 0.5, 0.005, "Angular scatter applied to each ray, in radians."),

                "sphere_count": _i(4, 0, 24, "Concentric spheres about the primary focus."),
                "sphere_inner": _f(0.34, 0.02, 2.0, 0.01, "Innermost sphere radius, as a fraction of the short side."),
                "sphere_outer": _f(0.84, 0.02, 3.0, 0.01, "Outermost sphere radius, as a fraction of the short side."),
                "sphere_span": _f(98.0, 5.0, 360.0, 1.0, "Angular sweep of the sphere arcs, in degrees."),

                "marker_size": _f(0.0020, 0.0002, 0.02, 0.0002, "Intersection marker radius, as a fraction of the short side."),
                "marker_shape": _e(["circle", "cross_x", "cross_plus", "triangle", "square"], "circle", "Shape drawn at every computed intersection. The two crosses are stroke-only by nature and ignore marker_fill."),
                "marker_fill": _b(True, "Fill the marker, or draw its outline only. Ignored for the cross shapes."),
                "marker_every": _i(1, 1, 20, "Mark every Nth computed intersection. Raise it to thin the plate."),
                "label_every": _i(3, 1, 40, "Label every Nth marked intersection."),

                "lune_density": _i(34, 4, 200, "Hatch lines in the lune, the shaded band between two rays across one sphere gap."),
                "scale_ticks": _i(50, 2, 400, "Graduations along the outer sphere."),
                "scale_label_every": _i(5, 1, 40, "Number every Nth graduation."),

                "type_scale": _f(1.0, 0.3, 3.0, 0.05, "Scales all annotation. Type stays a fixed fraction of canvas, so it is sub-legible at any resolution: it is texture, not content."),
                "exit_labels_every": _i(2, 1, 40, "Label every Nth ray where it leaves the frame, with its index and true bearing."),
                "callout_count": _i(2, 0, 20, "Leader callouts into micro-blocks."),
                "data_column_enabled": _b(True, "The running IDX / ANG / REF column keyed to the ray table."),
                "data_column_rows": _i(15, 1, 60, "Rows in the data column."),

                "zone_count": _i(3, 0, 40, "Pixelate sample zones, placed on computed intersections so each patch sits on an event the construction produced."),
                "pixel_size": _i(14, 1, 200, "Mosaic tile size, identical to Overlay's. 1 disables pixelation."),
                "zone_size_min": _f(44.0, 2.0, 1000.0, 1.0, "Smallest zone half-extent in pixels. Each zone is drawn a size at random between min and max, so the patches vary instead of being a uniform grid. Set min equal to max for Overlay's fixed-size behaviour."),
                "zone_size_max": _f(92.0, 2.0, 1000.0, 1.0, "Largest zone half-extent in pixels. Zone spacing is computed from THIS value, so patches never overlap even when a small one lands beside a large one."),
                "pixel_stroke": _b(True, "Outline each zone, as Overlay does."),

                "palette": _e(["whiteOnDark", "blackOnLight", "custom"], "whiteOnDark", "Ink polarity. Use blackOnLight on a light photograph: polarity is how this node stays legible, rather than shadowing every stroke."),
                "bg_color": _s("#101014", "Background behind the photograph and the ground of the overlay_only output. Only applies when palette is set to custom, and it is hidden in `image` at image_opacity 1.0 since the photograph covers it."),
                "stroke_color": _s("#FFFFFF", "Ink colour. Only applies when palette is set to custom."),
                "line_weight": _f(1.0, 0.1, 6.0, 0.05, "Scales all three weight classes together. The classes themselves are fixed by the spec."),
                "overlay_opacity": _f(1.0, 0.0, 1.0, 0.01, "Opacity of every drawn mark."),
                "image_opacity": _f(1.0, 0.0, 1.0, 0.01, "Opacity of the photograph under the plate."),

                "frame_text_size": _f(9.0, 1.0, 60.0, 0.5, "Corner text size, per mille of the short side."),
                "frame_text_tl": _s("DESIGN & STRATEGY", "Top-left plate text."),
                "frame_text_tr": _s("AKURATE", "Top-right plate text."),
                "frame_text_bl": _s("", "Bottom-left plate text."),
                "frame_text_br": _s("", "Bottom-right plate text."),
            },
            "optional": {
                "elements": _s("", "Per-ELEMENT overrides, one per line: 'rays_b: off', 'spheres: +0.02,-0.01', 'g_annot: +0.00,+0.03'. Offsets are normalised canvas fractions applied after anchoring. Read element_table for every valid name.", multiline=True),
                "pixelate_zones": _s("", "Extra zone centres as 'x,y;x,y', exactly as Overlay parses them."),
                "pixelate_mask": ("MASK", {"tooltip": "Extra zone centres taken from a mask, exactly as Overlay does."}),
                **{f"off_{g[2:]}_x": _f(0.0, -1.0, 1.0, 0.005,
                                        f"Offset the {g} group as a unit: {_GROUP_WHAT[g]}. Normalised canvas fraction, applied after anchoring, so members keep their shared edges.")
                   for g in geo.REGISTRY.groups()},
                **{f"off_{g[2:]}_y": _f(0.0, -1.0, 1.0, 0.005,
                                        f"Offset the {g} group as a unit: {_GROUP_WHAT[g]}. Normalised canvas fraction, applied after anchoring, so members keep their shared edges.")
                   for g in geo.REGISTRY.groups()},
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "overlay_only", "overlay_alpha", "element_table")
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Schematic"

    def execute(self, image, seed, **kw):
        # Group widgets feed the SAME override string as `elements`, so both
        # routes share one resolution path.
        lines = []
        for g in geo.REGISTRY.groups():
            dx = float(kw.get(f"off_{g[2:]}_x", 0.0) or 0.0)
            dy = float(kw.get(f"off_{g[2:]}_y", 0.0) or 0.0)
            if dx or dy:
                lines.append(f"{g}: {dx:+.6f},{dy:+.6f}")
        txt = kw.get("elements", "") or ""
        combined = "\n".join(lines + ([txt] if txt.strip() else []))

        # Parse FIRST so a typo fails before any render spend.
        st = ElementState(geo.REGISTRY, combined)

        bg_rgb, stroke_rgb = engine.resolve_palette(
            kw.get("palette", "whiteOnDark"), kw.get("bg_color", "#101014"),
            kw.get("stroke_color", "#FFFFFF"))

        params = dict(kw)
        params["overlay_opacity"] = float(kw.get("overlay_opacity", 1.0))
        params["line_weight"] = float(kw.get("line_weight", 1.0))

        cfg = {k: kw.get(k) for k in (
            "focus_a_x", "focus_a_y", "focus_b_x", "focus_b_y", "rays_a", "rays_b",
            "ray_jitter", "sphere_count", "sphere_inner", "sphere_outer",
            "sphere_span")}

        out_img, out_only, out_alpha = [], [], []
        table = ""
        dropped_total = 0

        for b in range(image.shape[0]):
            frame = np.clip(image[b].detach().cpu().numpy().astype(np.float32), 0.0, 1.0)
            h, w = int(frame.shape[0]), int(frame.shape[1])
            fitted = frame.astype(np.float64)
            frame_seed = (int(seed) + b * 7919) % 0x7FFFFFFF

            pts = geo.detect(frame, kw.get("detection_mode", "combined"),
                             int(kw.get("block_size", 16)), int(kw.get("threshold", 28)),
                             int(kw.get("max_circles", 26)), int(kw.get("min_distance", 70)),
                             int(kw.get("min_radius", 8)), int(kw.get("max_radius", 46)),
                             frame_seed)
            if not pts:
                pts = [(w * 0.5, h * 0.5, min(w, h) * 0.05)]

            if not kw.get("focus_b_enabled", True):
                cfg_b = dict(cfg)
                cfg_b["rays_b"] = 0
            else:
                cfg_b = cfg
            layout = geo.Layout(frame_seed, w, h, pts, cfg_b)
            layout._pts = pts

            # ---- 1/2: bg then the photograph, composited ONCE ---------------
            canvas = np.empty((h, w, 3), dtype=np.float64)
            canvas[:, :, 0] = bg_rgb[0] / 255.0
            canvas[:, :, 1] = bg_rgb[1] / 255.0
            canvas[:, :, 2] = bg_rgb[2] / 255.0
            io_ = float(kw.get("image_opacity", 1.0))
            if io_ > 0:
                canvas = canvas * (1.0 - io_) + fitted * io_

            # ---- 3: zones, Overlay's own pixelate_canvas, VERBATIM ----------
            extra = engine.parse_zone_string(kw.get("pixelate_zones", "") or "")
            pm = kw.get("pixelate_mask")
            if pm is not None:
                mb = pm[min(b, pm.shape[0] - 1)].detach().cpu().numpy() > 0.5
                extra = list(extra) + list(engine.mask_zone_centers(
                    engine.nearest_resample_mask(mb, w, h)))
            zlo = float(kw.get("zone_size_min", 44.0))
            zhi = float(kw.get("zone_size_max", 92.0))
            zmax = max(zlo, zhi)
            centres = []
            if st.on("zones"):
                centres = geo.zone_centres(frame_seed, layout, w, h,
                                           int(kw.get("zone_count", 3)), zmax,
                                           extra=extra)
            # The zones group offset must reach the CENTRES; without this the
            # off_zones_* widgets are inert, which invariant 14 caught.
            zox, zoy = st.px("zones", w, h)
            if zox or zoy:
                centres = [(cx + zox, cy + zoy) for (cx, cy) in centres]
            # No silent caps: if the spacing guarantee could not fit everything
            # asked for, say so rather than quietly placing fewer.
            want = int(kw.get("zone_count", 3)) + len(extra)
            if st.on("zones") and len(centres) < want:
                print(f"[Schematic] Construction: placed {len(centres)} of {want} "
                      f"zones; at zone_size_max={zmax:.0f} the 2.1x spacing rule "
                      f"leaves no room for more. Lower zone_size_max or "
                      f"zone_count.")
            psize = int(kw.get("pixel_size", 14))
            if psize > 1 and centres:
                sizes = geo.zone_sizes(frame_seed, len(centres), zlo, zhi)
                # Overlay's pixelate_canvas takes ONE size for the whole list, so
                # a per-zone size means calling it once per zone. Still the exact
                # same function, which is what keeps the siblings identical.
                for c, zs in zip(centres, sizes):
                    canvas = overlay_render.pixelate_canvas(
                        canvas, w, h, [c], psize, zs,
                        bool(kw.get("pixel_stroke", True)), stroke_rgb,
                        params["overlay_opacity"], 11, PACK_DIR)

            # ---- 4: the plate ----------------------------------------------
            plate = rc.draw_plate(canvas, layout, st, frame_seed, params,
                                  PACK_DIR, stroke_rgb)
            composed, alpha = plate.resolve_onto(canvas)
            dropped_total += plate.dropped

            only_bg = np.empty((h, w, 3), dtype=np.float64)
            only_bg[:, :, 0] = bg_rgb[0] / 255.0
            only_bg[:, :, 1] = bg_rgb[1] / 255.0
            only_bg[:, :, 2] = bg_rgb[2] / 255.0
            only, _ = plate.resolve_onto(only_bg)

            out_img.append(np.clip(composed, 0, 1).astype(np.float32))
            out_only.append(np.clip(only, 0, 1).astype(np.float32))
            out_alpha.append(alpha.astype(np.float32))
            if b == 0:
                table = st.table("Schematic Construction")

        print(f"[Schematic] Construction: seed={seed}, "
              f"{len(out_img)} frame(s), labels dropped={dropped_total}")

        return (torch.from_numpy(np.stack(out_img, 0)),
                torch.from_numpy(np.stack(out_only, 0)),
                torch.from_numpy(np.stack(out_alpha, 0)),
                table)


NODE_CLASS_MAPPINGS = {"SchematicConstruction": SchematicConstruction}
NODE_DISPLAY_NAME_MAPPINGS = {"SchematicConstruction": "Schematic Construction"}
