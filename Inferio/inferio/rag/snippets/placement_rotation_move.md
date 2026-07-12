# Moving and rotating objects (Placement)

Keywords: move, translate, rotate, position, place, placement, orient, angle, flip, align

```python
doc = App.ActiveDocument
obj = next(o for o in doc.Objects if o.Label == "Rod")

# Absolute position: put the object's local origin at (x, y, z).
obj.Placement.Base = App.Vector(10.0, 25.0, 0.0)

# Rotation: axis + angle in DEGREES (never radians here).
obj.Placement.Rotation = App.Rotation(App.Vector(0, 1, 0), 90)  # tip over Y

# Position AND rotation in one go:
obj.Placement = App.Placement(
    App.Vector(10, 25, 0),
    App.Rotation(App.Vector(0, 0, 1), 45))

# Relative nudge from wherever it is now:
obj.Placement.move(App.Vector(0, 0, 5.0))
doc.recompute()
```

Rotation happens around the object's LOCAL origin (a Part::Box rotates
around its corner, a Part::Cylinder around its bottom-center axis). To
rotate about the center, move first so the center sits at the origin of
rotation, or use `App.Placement(App.Vector(...), App.Rotation(...), App.Vector(cx, cy, cz))`
where the third argument is the center point.
