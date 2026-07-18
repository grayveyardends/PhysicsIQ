# Revolve (lathe parts), sweep and loft

Keywords: revolve, lathe, revolution, turn, sweep, loft, pipe, bend, profile, funnel, vase, shaft, pulley

```python
# Revolve: a profile spun around an axis — pulleys, shafts, vases, flanges.
import Part as _P
doc = App.ActiveDocument

# Profile in the XZ plane (closed polygon), revolved around the Z axis.
profile_pts = [App.Vector(5, 0, 0),  App.Vector(15, 0, 0),
               App.Vector(15, 0, 4), App.Vector(8, 0, 4),
               App.Vector(8, 0, 20), App.Vector(5, 0, 20),
               App.Vector(5, 0, 0)]
wire = _P.makePolygon(profile_pts)
face = _P.Face(wire)
solid = face.revolve(App.Vector(0, 0, 0),   # a point on the axis
                     App.Vector(0, 0, 1),   # axis direction (Z)
                     360)                   # degrees
obj = doc.addObject("Part::Feature", "Pulley")
obj.Shape = solid
doc.recompute()
```

Loft (skin between two profiles, e.g. square-to-circle funnel):
```python
sq = _P.makePolygon([App.Vector(-20,-20,0), App.Vector(20,-20,0),
                     App.Vector(20,20,0), App.Vector(-20,20,0), App.Vector(-20,-20,0)])
ci = _P.Wire(_P.makeCircle(8, App.Vector(0, 0, 40)))
loft = _P.makeLoft([sq, ci], True)      # True = make it a solid
obj = doc.addObject("Part::Feature", "Funnel"); obj.Shape = loft
```
