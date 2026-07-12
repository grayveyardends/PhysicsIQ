# Common FreeCAD scripting mistakes (read when something errors)

Keywords: error, fail, exception, wrong, mistake, fix, debug, null, shape, invalid

- Forgot `doc.recompute()` → objects exist but shapes are empty/invisible.
  Recompute once at the END of the script.
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
