# Patterns: linear array, polar/circular array (bolt holes)

Keywords: array, pattern, repeat, copies, polar, circular, bolt, holes, grid, linear, ring

```python
# Ring of 6 bolt holes in a plate — the manual (always-works) way:
import math
doc = App.ActiveDocument
plate = next(o for o in doc.Objects if o.Label == "BasePlate")

N, RING_R, HOLE_R = 6, 22.0, 2.5
cx, cy = 30.0, 30.0                       # ring center
result = plate
for i in range(N):
    a = 2 * math.pi * i / N
    drill = doc.addObject("Part::Cylinder", f"Hole{i}")
    drill.Radius, drill.Height = HOLE_R, 50.0
    drill.Placement.Base = App.Vector(cx + RING_R * math.cos(a),
                                      cy + RING_R * math.sin(a), -1)
    cut = doc.addObject("Part::Cut", f"CutH{i}")
    cut.Base, cut.Tool = result, drill
    result = cut                          # chain the cuts
result.Label = "PlateWithBoltHoles"
doc.recompute()
```

Linear grid: same loop, `Base = App.Vector(x0 + i*dx, y0 + j*dy, z)`.
For a copy-array of SOLIDS (not holes), use Draft:
`Draft.make_polar_array(obj, number=6, angle=360)` or
`Draft.make_ortho_array(obj, App.Vector(20,0,0), App.Vector(0,20,0), n_x=3, n_y=2)`.
