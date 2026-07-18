# Boolean operations: hole (cut), fuse (union), common (intersection), hollow

Keywords: hole, drill, cut, subtract, fuse, union, combine, merge, intersect, hollow, shell, pocket, through

```python
# A 20mm cube with a 5mm diameter hole straight through it.
doc = App.ActiveDocument
SIZE, HOLE_D = 20.0, 5.0

cube = doc.addObject("Part::Box", "Cube")
cube.Length = cube.Width = cube.Height = SIZE

drill = doc.addObject("Part::Cylinder", "Drill")
drill.Radius = HOLE_D / 2
drill.Height = SIZE + 2                     # 1mm longer on both ends: clean cut
drill.Placement.Base = App.Vector(SIZE/2, SIZE/2, -1)

cut = doc.addObject("Part::Cut", "CubeWithHole")
cut.Base = cube                             # the body being drilled
cut.Tool = drill                            # the drill bit
cut.Label = "CubeWithHole"
doc.recompute()
```

Fuse two solids into one: `f = doc.addObject("Part::Fuse", "Fused"); f.Base = a; f.Tool = b`.
Hollow part (wall thickness t): cut a copy of the shape scaled/offset inward,
or simplest: outer solid minus an inner solid that is `2*t` smaller and
placed `t` deeper. Booleans hide their children automatically after recompute.
