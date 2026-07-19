"""
Pure-python engine + PIL renderer for ComfyUI-Schematic.

No torch here (and no ComfyUI imports) — `engine.py` and `render.py` only need
numpy + Pillow, so `tools/test_schematic.py` can import this package directly
via `sys.path.insert(0, pack_root); from schematic import engine, render`
without ComfyUI installed.
"""
