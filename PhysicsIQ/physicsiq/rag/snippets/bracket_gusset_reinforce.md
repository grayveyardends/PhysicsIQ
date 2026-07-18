# L-bracket with gusset — and how to reinforce a weak part

Keywords: bracket, mount, L-bracket, gusset, rib, reinforce, stiffen, support, drone, arm, load, strengthen

```python
# Parametric L-bracket: vertical wall + horizontal base + triangular gusset.
doc = App.ActiveDocument
BASE_L, BASE_W, T = 60.0, 40.0, 5.0     # base plate size, thickness
WALL_H = 50.0                            # vertical wall height
GUSSET = 25.0                            # gusset triangle leg length

base = doc.addObject("Part::Box", "Base")
base.Length, base.Width, base.Height = BASE_L, BASE_W, T

wall = doc.addObject("Part::Box", "Wall")
wall.Length, wall.Width, wall.Height = T, BASE_W, WALL_H
wall.Placement.Base = App.Vector(0, 0, T)

# Gusset = triangular prism made from a wire, then extruded across width.
import Part as _P
tri = _P.makePolygon([App.Vector(T, 0, T),
                      App.Vector(T + GUSSET, 0, T),
                      App.Vector(T, 0, T + GUSSET),
                      App.Vector(T, 0, T)])
gusset_shape = _P.Face(tri).extrude(App.Vector(0, BASE_W, 0))
gusset = doc.addObject("Part::Feature", "Gusset")
gusset.Shape = gusset_shape

f1 = doc.addObject("Part::Fuse", "F1"); f1.Base = base; f1.Tool = wall
f2 = doc.addObject("Part::Fuse", "Bracket"); f2.Base = f1; f2.Tool = gusset
f2.Label = "Bracket"
doc.recompute()
```

To reinforce an existing part at a weak point: add a gusset/rib solid at
that location and fuse it on, or fillet the sharp inner corner (see the
fillet recipe). Bigger fillet radius = lower stress concentration.
