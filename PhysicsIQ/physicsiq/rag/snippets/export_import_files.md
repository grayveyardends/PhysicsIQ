# Export / import: STEP, STL, screenshots

Keywords: export, import, save, step, stl, mesh, file, screenshot, image, open

```python
import Part as _P
doc = App.ActiveDocument
obj = next(o for o in doc.Objects if o.Label == "Bracket")

# STEP export (the CAD interchange format — also what the PINN consumes):
_P.export([obj], "/tmp/bracket.step")

# STL export (3D printing):
import Mesh
Mesh.export([obj], "/tmp/bracket.stl")

# Import a STEP file into the current document:
_P.insert("/tmp/other_part.step", doc.Name)

# Screenshot the 3D view to a file:
Gui.ActiveDocument.ActiveView.saveImage("/tmp/view.png", 1280, 960, "White")
doc.recompute()
```
