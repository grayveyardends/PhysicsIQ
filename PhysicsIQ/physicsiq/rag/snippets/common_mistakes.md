# Common FreeCAD scripting mistakes (read when something errors)

Keywords: error, fail, exception, wrong, mistake, fix, debug, null, shape, invalid, assembly, link, frame, drone, stress, analysis

- **A part you want to STRESS-ANALYZE must be ONE fused solid.** An assembly
  of separate pieces has no load path, so the PINN cannot solve it. Build the
  pieces, then fuse them into a single object and analyze THAT:

  ```python
  frame = doc.addObject("Part::MultiFuse", "Frame")
  frame.Shapes = [plate, arm1, arm2, arm3, arm4]   # they must OVERLAP
  frame.Label = "F450 Frame"
  doc.recompute()
  piq_tools.run_stress_analysis("F450 Frame", force_N=(0, 0, -500))
  ```
  The pieces must genuinely overlap (share volume), not merely touch at a
  coincident face, or the fuse leaves them as separate solids.
- **Do not use `App::Link` / `App::LinkGroup` to repeat a part** (e.g. four
  identical drone arms). A Link has a `.Shape` but is NOT a `Part::Feature`,
  so half of FreeCAD's API silently ignores it (`Part.export` prints
  *"'X' is not a shape, export will be ignored"*). To repeat a shape, copy
  the geometry instead:
  ```python
  for i, angle in enumerate((45, 135, 225, 315)):
      a = doc.addObject("Part::Feature", f"Arm{i}")
      a.Shape = arm.Shape.copy()
      a.Placement = App.Placement(App.Vector(0, 0, 0),
                                  App.Rotation(App.Vector(0, 0, 1), angle))
  ```

- Forgot `doc.recompute()` → objects exist but shapes are empty/invisible.
  Recompute once at the END of the script.
- A `Part::Box` is NOT centered at the origin: its local origin is a
  CORNER, so it spans [0..Length]×[0..Width]×[0..Height] and its center
  is (Length/2, Width/2, Height/2). To drill through a box's center,
  place the drill at (Length/2, Width/2, -1) — never at (0, 0, -1).
  (`Part::Cylinder` and `Part::Sphere` ARE centered on their own axis.)
- `App.Rotation(axis, angle)` takes DEGREES. `math` functions take radians.
  Converting the wrong way = parts at weird angles.
- Booleans: the Tool must fully pierce the Base for a clean through-hole —
  make the drill cylinder ~2mm longer and start it 1mm below the face.
- `Part::Cut/Fuse` referencing an object that was deleted → "Links go out
  of scope". Never delete objects that a boolean still references.
- Look up objects by `Label`, delete by `Name` (see the query recipe).
- A `Part::Feature` needs `obj.Shape = some_shape` — assigning a Wire or
  Face where a Solid is expected makes downstream booleans fail. Use
  `_P.Face(wire)` then `.extrude(...)` / `.revolve(...)` to get a solid.
- `fillet.Edges` indexes start at 1, not 0. A too-large radius makes the
  fillet fail entirely — halve it and retry.
- Vectors must be `App.Vector(x, y, z)` — plain tuples raise TypeError.
- Placement of a sketch inside a PartDesign Body: leave it default (XY)
  unless you know the attachment API; wrong attachments break pads.
- Don't call `Gui.*` dialogs or `input()` — the script runs unattended.
