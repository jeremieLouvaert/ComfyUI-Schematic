"""
Byte-identity golden harness for the SHIPPED SchematicOverlay node — suite zero of
the Phase 1 expansion (docs/texture-mask-addition.md section 5).

Usage (embedded python, needs torch):
  python tools/test_byteident.py --capture   # render config set, store golden hashes
  python tools/test_byteident.py --check     # re-render, compare, exit nonzero on drift

Goldens are LOCAL-ENVIRONMENT artifacts (font rasterization varies by Pillow build):
stored in _goldens/ (gitignored), captured on pre-change HEAD, re-checked after every
touch this phase. Configs G1-G6 cover: defaults, chain-heavy, pixelate zones with
crosshair off, custom palette + square + procedural grain, batch + size preset,
blackOnLight with connections/star/texture disabled.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)

import torch
import importlib.util, types

# package shim so the node's relative imports resolve without ComfyUI
pkg = types.ModuleType("cs_pkg")
pkg.__path__ = [PACK]
sys.modules["cs_pkg"] = pkg
for sub in ("schematic", "nodes"):
    spec = importlib.util.spec_from_file_location(
        f"cs_pkg.{sub}", os.path.join(PACK, sub, "__init__.py"),
        submodule_search_locations=[os.path.join(PACK, sub)])
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"cs_pkg.{sub}"] = m
    spec.loader.exec_module(m)

SchematicOverlay = sys.modules["cs_pkg.nodes"].NODE_CLASS_MAPPINGS["SchematicOverlay"]

GOLDEN_DIR = os.path.join(PACK, "_goldens")
GOLDEN_PATH = os.path.join(GOLDEN_DIR, "schematic_v01.json")

DEFAULTS = {
    "detection_mode": "combined", "block_size": 16, "threshold": 30,
    "max_circles": 80, "min_distance": 40, "shape": "circle",
    "min_radius": 4, "max_radius": 24, "circle_stroke": 1.0, "seed": 42,
    "label_size": 8, "overlay_opacity": 1.0, "connection_dist": 150,
    "line_weight": 0.8, "chain_enabled": True, "chain_count": 11,
    "chain_angle": 45, "chain_base_radius": 250, "chain_ratio": 0.79,
    "chain_intersections": True, "chain_intersection_size": 5.0,
    "crosshair_enabled": True, "crosshair_frame_size": 60, "crosshair_dash": 8,
    "crosshair_stroke": 1.0, "crosshair_star_size": 40, "crosshair_star_points": 4,
    "pixel_size": 16, "zone_size": 100, "pixel_stroke": True,
    "image_opacity": 0.85, "texture_opacity": 0.5, "frame_text_size": 12,
    "frame_text_tl": "Design & Strategy", "frame_text_tr": "AKURATE",
    "frame_text_bl": "", "frame_text_br": "", "palette": "whiteOnDark",
    "size_preset": "match input", "bg_color": "#0a0a0a",
    "stroke_color": "#d8d8d8", "pixelate_zones": "",
}


def synth_image(w, h, seed):
    """Deterministic structured content: gradients + bright blobs + noise, so the
    detector has real features to bite on."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 0.25 + 0.5 * (xx / max(w - 1, 1)) * (yy / max(h - 1, 1))
    img = np.stack([base, base * 0.9, base * 1.1], axis=-1)
    for _ in range(6):
        cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
        s = rng.uniform(0.03, 0.10) * min(w, h)
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)))
        img += blob[..., None] * rng.uniform(0.3, 0.7)
    img += rng.normal(0, 0.04, size=(h, w, 3)).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


CONFIGS = {
    "G1_defaults": {"frames": [(642, 480, 11)], "params": {}},
    "G2_chain_heavy": {"frames": [(642, 480, 12)],
                       "params": {"chain_count": 21, "chain_ratio": 0.85, "chain_angle": 30}},
    "G3_pixelate_no_crosshair": {"frames": [(642, 480, 13)],
                                 "params": {"crosshair_enabled": False,
                                            "pixelate_zones": "100,100;300,200",
                                            "pixel_size": 12, "zone_size": 60}},
    "G4_custom_square_grain": {"frames": [(642, 480, 14)],
                               "params": {"palette": "custom", "bg_color": "#102030",
                                          "stroke_color": "#ffcc00", "shape": "square",
                                          "texture_opacity": 1.0}},
    "G5_batch_story_preset": {"frames": [(640, 640, 50), (640, 640, 51)],
                              "params": {"size_preset": "Instagram Story (1080x1920)",
                                         "image_opacity": 0.5}},
    "G6_light_min": {"frames": [(642, 480, 16)],
                     "params": {"palette": "blackOnLight", "connection_dist": 0,
                                "crosshair_star_size": 0, "texture_opacity": 0.0}},
}


def run_config(node, cfg):
    frames = [synth_image(w, h, s) for (w, h, s) in cfg["frames"]]
    img = torch.from_numpy(np.stack(frames, axis=0))
    params = dict(DEFAULTS)
    params.update(cfg["params"])
    outs = node.execute(image=img, pixelate_mask=None, texture=None, **params)
    hashes = {}
    for name, t in zip(("image", "overlay_only", "overlay_alpha"), outs):
        arr = np.ascontiguousarray(t.cpu().numpy().astype(np.float32))
        hashes[name] = hashlib.md5(arr.tobytes()).hexdigest()
    return hashes


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = ap.parse_args()

    node = SchematicOverlay()
    results = {}
    for name, cfg in CONFIGS.items():
        t0 = time.time()
        results[name] = run_config(node, cfg)
        print(f"  [{name}] rendered in {time.time() - t0:.2f}s")

    if args.capture:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(GOLDEN_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"CAPTURED {len(results)} configs -> {GOLDEN_PATH}")
        return 0

    if not os.path.exists(GOLDEN_PATH):
        print(f"FAIL: no goldens at {GOLDEN_PATH}; run --capture on the baseline first")
        return 1
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    ok = True
    for name, hashes in results.items():
        for out_name, h in hashes.items():
            g = golden.get(name, {}).get(out_name)
            match = (g == h)
            if not match:
                ok = False
            print(f"  [{'PASS' if match else 'FAIL'}] {name}.{out_name}"
                  + ("" if match else f"  golden={g} now={h}"))
    print("BYTE-IDENTITY " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
