"""heatmap_overlay.py — paints the PINN stress field onto the 3D viewport.

How: FreeCAD's 3D view is an Open Inventor (coin3d) scene graph, and pivy
gives us Python access to it. We build our own colored copy of the part's
mesh — one color per vertex, blue (relaxed) -> red (screaming) — and add it
ON TOP of the scene, plus a red marker sphere at the worst weak point.
The real part is made mostly transparent underneath so the colors read.

Nothing here touches the document data — it is pure eye candy, safe to
clear and redo any number of times. Main thread only (Qt/coin law).
"""

import math
import time

import numpy as np

# Keep references to what we added so a re-run can cleanly undo it first,
# and so a live update can recolor in place instead of rebuilding the mesh.
_state = {"node": None, "target": None, "old_transparency": None,
          # cached geometry for fast live recoloring
          "vp": None, "verts": None, "L": 0.0, "target_label": None,
          "marker_tr": None,
          # marker animation bits
          "marker": None, "ball": None, "ball_mat": None,
          "base_radius": 0.0, "timer": None}

_geomkit = None


def _load_geomkit():
    """ctypes handle to pinn/native/libgeomkit.so, or None. The lib is
    plain C, so FreeCAD's interpreter can use the venv-built binary."""
    global _geomkit
    if _geomkit is not None:
        return _geomkit if _geomkit else None
    import ctypes
    from physicsiq import REPO_DIR
    so = f"{REPO_DIR}/pinn/native/libgeomkit.so"
    try:
        lib = ctypes.CDLL(so)
        lib.nearest_index.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32)]
        _geomkit = lib
    except OSError:
        _geomkit = False
    return _geomkit if _geomkit else None


def _pulse_tick():
    """One animation frame: the hotspot ball breathes (radius + glow).
    Driven by a QTimer on the main thread — coin3d re-renders whenever a
    scene-graph field changes, so this is all it takes to animate."""
    ball = _state["ball"]
    mat = _state["ball_mat"]
    if ball is None or mat is None:
        return
    # 0..1 sine wave, ~1.6 s period
    w = 0.5 + 0.5 * math.sin(time.time() * 4.0)
    ball.radius = _state["base_radius"] * (0.8 + 0.7 * w)
    mat.emissiveColor = (0.25 + 0.75 * w, 0.0, 0.0)
    mat.transparency = 0.15 * (1.0 - w)


def _start_pulse():
    from PySide6 import QtCore
    if _state["timer"] is None:
        _state["timer"] = QtCore.QTimer()
        _state["timer"].setInterval(60)          # ~16 fps, plenty smooth
        _state["timer"].timeout.connect(_pulse_tick)
    _state["timer"].start()


def stop_pulse():
    if _state["timer"] is not None:
        _state["timer"].stop()


def remove_marker():
    """Take the hotspot ball out of the scene but keep the color map.
    Called when the analysis loop is done with a spot — the sphere is a
    'look HERE right now' cue, not permanent decoration."""
    stop_pulse()
    if _state["node"] is not None and _state["marker"] is not None:
        try:
            _state["node"].removeChild(_state["marker"])
        except Exception:
            pass  # scene already torn down — nothing to remove
    _state["marker"] = None
    _state["ball"] = None
    _state["ball_mat"] = None


def _jet(vals):
    """Normalized values (0..1) -> RGB float triplets, blue->cyan->green->
    yellow->red. Hand-rolled so we don't need matplotlib inside FreeCAD."""
    v = np.clip(vals, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=1)


def _nearest_values(verts, points, values, chunk=512):
    """For each mesh vertex, the stress of the nearest PINN sample point.
    Brute force in numpy chunks: verts ~ few thousand, points ~ few
    thousand -> tens of millions of float ops, milliseconds. No KD-tree
    dependency needed."""
    lib = _load_geomkit()
    if lib is not None:
        import ctypes
        s = np.ascontiguousarray(points, dtype=np.float64)
        q = np.ascontiguousarray(verts, dtype=np.float64)
        idx = np.empty(len(q), dtype=np.int32)
        dp = ctypes.POINTER(ctypes.c_double)
        lib.nearest_index(s.ctypes.data_as(dp), len(s),
                          q.ctypes.data_as(dp), len(q),
                          idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        return values[idx]
    out = np.empty(len(verts), dtype=float)
    for i in range(0, len(verts), chunk):
        block = verts[i:i + chunk]                       # (c, 3)
        d2 = ((block[:, None, :] - points[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = values[d2.argmin(axis=1)]
    return out


def _set_colors(vp, vertex_vm, lo, hi):
    norm = (vertex_vm - lo) / max(hi - lo, 1e-9)
    rgb = (_jet(norm) * 255).astype(np.uint32)
    # coin wants packed 0xRRGGBBAA integers for per-vertex color
    rgba = ((rgb[:, 0] << 24) | (rgb[:, 1] << 16) | (rgb[:, 2] << 8) | 0xFF)
    vp.orderedRGBA.setValues(0, len(rgba), [int(x) for x in rgba])


def update_from_files(field_npz_path, target_label=None):
    """Live repaint: new stresses onto the ALREADY-built overlay mesh.
    Falls through to a full apply_from_files the first time (or if the
    target changed). Cheap enough to run on every training snapshot."""
    if (_state["node"] is None or _state["vp"] is None
            or (target_label and target_label != _state["target_label"])):
        return apply_from_files(field_npz_path, target_label)

    data = np.load(field_npz_path)
    points = data["points"].astype(float)
    vm = data["von_mises"].astype(float)

    vertex_vm = _nearest_values(_state["verts"], points, vm)
    lo, hi = float(vm.min()), float(vm.max())
    _set_colors(_state["vp"], vertex_vm, lo, hi)
    if _state["marker_tr"] is not None:
        worst = points[vm.argmax()]
        _state["marker_tr"].translation = tuple(float(x) for x in worst)
    return f"heatmap updated (max {hi:.0f} MPa)"


def clear_overlay():
    import FreeCADGui as Gui
    stop_pulse()
    _state["marker"] = None
    _state["ball"] = None
    _state["ball_mat"] = None
    if _state["node"] is not None:
        try:
            sg = Gui.ActiveDocument.ActiveView.getSceneGraph()
            sg.removeChild(_state["node"])
        except Exception:
            pass  # view may be gone; nothing to clean then
        _state["node"] = None
    _state["vp"] = None
    _state["verts"] = None
    _state["marker_tr"] = None
    _state["target_label"] = None
    if _state["target"] is not None and _state["old_transparency"] is not None:
        try:
            _state["target"].ViewObject.Transparency = _state["old_transparency"]
        except Exception:
            pass
        _state["target"] = None
        _state["old_transparency"] = None


def apply_from_files(field_npz_path, target_label=None):
    """Load field.npz (points + von_mises) and paint the heatmap."""
    import FreeCAD as App
    import FreeCADGui as Gui
    from pivy import coin

    data = np.load(field_npz_path)
    points = data["points"].astype(float)
    vm = data["von_mises"].astype(float)

    # -- find the part to paint
    doc = App.ActiveDocument
    candidates = [o for o in doc.Objects
                  if getattr(o, "Shape", None) is not None
                  and not o.Shape.isNull() and o.Shape.Solids]
    if target_label:
        target = next(o for o in candidates if o.Label == target_label)
    else:
        target = max(candidates, key=lambda o: o.Shape.Volume)

    clear_overlay()

    # -- mesh the shape and color each vertex by nearest stress
    L = max(target.Shape.BoundBox.XLength, target.Shape.BoundBox.YLength,
            target.Shape.BoundBox.ZLength)
    verts, facets = target.Shape.tessellate(0.005 * L)
    v = np.array([[p.x, p.y, p.z] for p in verts])
    vertex_vm = _nearest_values(v, points, vm)

    lo, hi = float(vm.min()), float(vm.max())

    vp = coin.SoVertexProperty()
    vp.vertex.setValues(0, len(v), v.tolist())
    _set_colors(vp, vertex_vm, lo, hi)
    vp.materialBinding = coin.SoVertexProperty.PER_VERTEX_INDEXED

    face_set = coin.SoIndexedFaceSet()
    face_set.vertexProperty = vp
    idx = []
    for (a, b, c) in facets:
        idx += [a, b, c, -1]         # -1 terminates each polygon
    face_set.coordIndex.setValues(0, len(idx), idx)

    root = coin.SoSeparator()
    hints = coin.SoShapeHints()      # light both sides, tolerate any winding
    hints.vertexOrdering = coin.SoShapeHints.COUNTERCLOCKWISE
    hints.shapeType = coin.SoShapeHints.UNKNOWN_SHAPE_TYPE
    root.addChild(hints)
    root.addChild(face_set)

    # -- PULSING red marker ball at the WORST point
    worst = points[vm.argmax()]
    marker = coin.SoSeparator()
    tr = coin.SoTranslation()
    tr.translation = tuple(float(x) for x in worst)
    mat = coin.SoMaterial()
    mat.diffuseColor = (1.0, 0.0, 0.0)
    mat.emissiveColor = (0.5, 0.0, 0.0)   # glows so it can't be missed
    ball = coin.SoSphere()
    ball.radius = 0.02 * L
    marker.addChild(tr)
    marker.addChild(mat)
    marker.addChild(ball)
    root.addChild(marker)

    # -- ghost the real part and mount our overlay
    # Not every view provider has Transparency (an App::Link's doesn't), and
    # the heatmap is worth showing even when we can't fade the part behind it.
    _state["old_transparency"] = getattr(
        target.ViewObject, "Transparency", None)
    if _state["old_transparency"] is not None:
        target.ViewObject.Transparency = 80
    Gui.ActiveDocument.ActiveView.getSceneGraph().addChild(root)
    _state["node"] = root
    _state["target"] = target
    _state["target_label"] = target.Label
    _state["vp"] = vp
    _state["verts"] = v
    _state["L"] = L
    _state["marker_tr"] = tr
    _state["marker"] = marker
    _state["ball"] = ball
    _state["ball_mat"] = mat
    _state["base_radius"] = 0.02 * L
    _start_pulse()   # breathe until the analysis loop settles the spot
    return f"heatmap painted on '{target.Label}' (max {hi:.0f} MPa, pulsing ball = hotspot)"


def summarize_report(report: dict) -> str:
    """result.json -> the short text the chat shows and the model reads."""
    lines = [f"Stress analysis of '{report.get('target_label', 'part')}': "
             f"max von Mises {report['max_von_mises_mpa']} MPa, "
             f"yield {report['yield_mpa']} MPa, "
             f"min safety factor {report['min_safety_factor']}"]
    for i, wp in enumerate(report.get("weak_points", []), 1):
        x, y, z = wp["location_mm"]
        lines.append(f"  {i}. weak point at ({x}, {y}, {z}) mm — "
                     f"{wp['von_mises_mpa']} MPa, SF {wp['safety_factor']}")
    return "\n".join(lines)
