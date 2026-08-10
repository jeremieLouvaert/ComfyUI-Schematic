from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f"[Schematic] {len(NODE_CLASS_MAPPINGS)} nodes loaded: " + ", ".join(sorted(NODE_DISPLAY_NAME_MAPPINGS.values())))
