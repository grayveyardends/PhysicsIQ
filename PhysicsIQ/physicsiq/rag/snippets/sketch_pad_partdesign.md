# PartDesign workflow: Body → Sketch → Pad (extrude) / Pocket

Keywords: sketch, pad, extrude, pocket, partdesign, body, profile, 2d, drawing, extrusion

```python
# A 40x30 rectangle sketch padded 8mm tall, the "proper CAD" way.
doc = App.ActiveDocument

body = doc.addObject("PartDesign::Body", "Body")
sketch = body.newObject("Sketcher::SketchObject", "Profile")
# A fresh sketch lies on the XY plane by default — good enough here.

import Part as _P
W, L = 40.0, 30.0
pts = [App.Vector(0, 0, 0), App.Vector(W, 0, 0),
       App.Vector(W, L, 0), App.Vector(0, L, 0)]
for i in range(4):                       # four lines make a closed rectangle
    sketch.addGeometry(_P.LineSegment(pts[i], pts[(i + 1) % 4]), False)

pad = body.newObject("PartDesign::Pad", "Pad")
pad.Profile = sketch
pad.Length = 8.0                          # mm, extrusion height
doc.recompute()
```

Pocket (cut INTO the pad with a second sketch): create another sketch on
the pad's top face, then `pk = body.newObject("PartDesign::Pocket", "Pocket");
pk.Profile = sketch2; pk.Length = 3.0; doc.recompute()`.
Circles in sketches: `sketch.addGeometry(_P.Circle(App.Vector(x, y, 0), App.Vector(0, 0, 1), radius), False)`.
