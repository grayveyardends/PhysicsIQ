"""ui_control.py — lets the LLM (and the user) rearrange FreeCAD's windows.

FreeCAD's central area is a QMdiArea holding the 3D views, so we can tile,
split and re-focus programmatically — this is the "reorder my workspace"
feature. All functions here must run on the main thread (they touch Qt).
"""

from PySide6 import QtWidgets

import FreeCADGui as Gui


def _mdi_area() -> QtWidgets.QMdiArea:
    return Gui.getMainWindow().findChild(QtWidgets.QMdiArea)


def arrange_views(mode: str = "tile"):
    """mode: 'tile' | 'cascade' | 'tabbed'. Rearranges all open views."""
    area = _mdi_area()
    if mode == "tile":
        area.setViewMode(QtWidgets.QMdiArea.SubWindowView)
        area.tileSubWindows()
    elif mode == "cascade":
        area.setViewMode(QtWidgets.QMdiArea.SubWindowView)
        area.cascadeSubWindows()
    elif mode == "tabbed":
        area.setViewMode(QtWidgets.QMdiArea.TabbedView)
    else:
        raise ValueError("mode must be tile, cascade or tabbed")
    return f"views arranged: {mode}"


def split_model_and_sketch(sketch_label: str = None):
    """The layout from the plan: main 3D viewport left, a second view
    focused on a sketch (or top-down 2D look) right, side by side."""
    import FreeCAD as App
    doc = App.ActiveDocument
    if doc is None:
        raise ValueError("No document open")

    # Second view of the SAME document — FreeCAD supports this natively.
    Gui.runCommand("Std_ViewCreate", 0)
    area = _mdi_area()
    area.setViewMode(QtWidgets.QMdiArea.SubWindowView)
    area.tileSubWindows()

    # Point the new (= active) view straight down for a 2D drawing feel.
    view = Gui.ActiveDocument.ActiveView
    view.viewTop()
    view.fitAll()

    if sketch_label:
        for obj in doc.Objects:
            if obj.Label == sketch_label and hasattr(obj, "ViewObject"):
                Gui.Selection.clearSelection()
                Gui.Selection.addSelection(obj)
                break
    return "split layout: 3D left, top-down right"


def set_view(direction: str = "iso"):
    """Snap the active 3D view: iso/front/top/right/left/rear/bottom."""
    view = Gui.ActiveDocument.ActiveView
    fn = {
        "iso": view.viewIsometric, "front": view.viewFront,
        "top": view.viewTop, "right": view.viewRight,
        "left": view.viewLeft, "rear": view.viewRear,
        "bottom": view.viewBottom,
    }.get(direction)
    if fn is None:
        raise ValueError(f"unknown direction '{direction}'")
    fn()
    view.fitAll()
    return f"view set to {direction}"
