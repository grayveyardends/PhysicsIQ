# Primitive solids: box, cylinder, sphere, cone (parametric)

Keywords: box, cube, plate, block, cylinder, rod, shaft, tube, sphere, ball, cone, primitive, create, make

```python
# Parametric primitives: dimensions stay editable in the model tree.
L, W, H = 80.0, 60.0, 5.0          # mm
doc = App.ActiveDocument

plate = doc.addObject("Part::Box", "Plate")
plate.Length, plate.Width, plate.Height = L, W, H
plate.Label = "BasePlate"

rod = doc.addObject("Part::Cylinder", "Rod")
rod.Radius, rod.Height = 4.0, 40.0
rod.Placement = App.Placement(App.Vector(20, 30, H), App.Rotation())  # sits on plate

ball = doc.addObject("Part::Sphere", "Ball")
ball.Radius = 8.0
ball.Placement.Base = App.Vector(60, 30, H + 8)

doc.recompute()
```

A tube (hollow cylinder) is a cylinder minus a smaller cylinder — see the
boolean cut recipe.
