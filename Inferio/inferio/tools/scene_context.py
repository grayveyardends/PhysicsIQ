"""scene_context.py — turns the document tree into short text for the LLM.

The model can't see the 3D view (well, it can via screenshots, but text is
1000x cheaper in tokens). Every chat turn we describe what exists:

    - Box001 (Part::Box) label='MountingPlate' Length=80.0 Width=60.0 Height=5.0
    - Cut (Part::Cut) label='PlateWithHole' bbox=80x60x5mm

That is usually all a 2B model needs to write correct edits by Label.
"""

# Properties worth mentioning, in the order we mention them. Anything
# else (colors, placement matrices...) is token noise for a small model.
_INTERESTING_PROPS = [
    "Length", "Width", "Height", "Radius", "Radius1", "Radius2", "Angle",
]

_MAX_OBJECTS = 40   # a huge assembly would blow the context window


def describe_object(obj) -> str:
    bits = [f"- {obj.Name} ({obj.TypeId}) label='{obj.Label}'"]
    for prop in _INTERESTING_PROPS:
        if hasattr(obj, prop):
            try:
                bits.append(f"{prop}={float(getattr(obj, prop)):g}")
            except (TypeError, ValueError):
                pass  # property exists but isn't a plain number — skip
    # Bounding box gives the model a sense of scale even for weird shapes.
    shape = getattr(obj, "Shape", None)
    if shape is not None and not shape.isNull():
        bb = shape.BoundBox
        bits.append(f"bbox={bb.XLength:.0f}x{bb.YLength:.0f}x{bb.ZLength:.0f}mm")
    return " ".join(bits)


def describe_document() -> str:
    import FreeCAD as App
    doc = App.ActiveDocument
    if doc is None or not doc.Objects:
        return "(empty document — nothing created yet)"
    lines = [describe_object(o) for o in doc.Objects[:_MAX_OBJECTS]]
    if len(doc.Objects) > _MAX_OBJECTS:
        lines.append(f"... and {len(doc.Objects) - _MAX_OBJECTS} more objects")
    return "\n".join(lines)
