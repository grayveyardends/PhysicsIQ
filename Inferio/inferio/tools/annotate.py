"""annotate.py — put text labels / dimension callouts on 3D objects.

This powers requests like "annotate the gears": the LLM calls
inferio_tools.annotate("Gear1") inside its code block and a floating
label appears next to that object in the 3D view.

We use Draft.make_text because it renders in the 3D viewport (App::
Annotation is the fallback for builds without Draft).
"""


def _find_by_label(label: str):
    import FreeCAD as App
    doc = App.ActiveDocument
    if doc is None:
        raise ValueError("No document open")
    for obj in doc.Objects:
        if obj.Label == label or obj.Name == label:
            return obj
    raise ValueError(
        f"No object labelled '{label}'. Existing labels: "
        + ", ".join(o.Label for o in doc.Objects[:20]))


def annotate(label: str, text: str = None, offset_mm: float = 10.0):
    """Attach a text annotation next to the object with this Label.

    text=None auto-generates "Label (WxLxH mm)" from the bounding box —
    handy for "annotate everything with its dimensions" requests.
    """
    import FreeCAD as App
    obj = _find_by_label(label)

    shape = getattr(obj, "Shape", None)
    if shape is not None and not shape.isNull():
        bb = shape.BoundBox
        anchor = App.Vector(bb.XMax + offset_mm, bb.YMax + offset_mm, bb.ZMax)
        if text is None:
            text = (f"{obj.Label}  "
                    f"{bb.XLength:.1f} x {bb.YLength:.1f} x {bb.ZLength:.1f} mm")
    else:
        anchor = App.Vector(0, 0, 0)
        text = text or obj.Label

    try:
        import Draft
        note = Draft.make_text([text], placement=App.Placement(
            anchor, App.Rotation()))
        note.ViewObject.FontSize = 4  # mm — readable at part scale
    except ImportError:
        note = App.ActiveDocument.addObject("App::Annotation", "Note")
        note.LabelText = [text]
        note.Position = anchor
    note.Label = f"Note_{obj.Label}"
    App.ActiveDocument.recompute()
    return note.Label
