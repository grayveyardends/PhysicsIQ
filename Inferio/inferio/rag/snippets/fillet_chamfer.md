# Fillet (round edges) and chamfer — also THE stress-concentration fix

Keywords: fillet, round, radius, chamfer, bevel, edge, smooth, stress, concentration, sharp, corner, reinforce

```python
# Fillet ALL edges of an object with a 2mm radius.
doc = App.ActiveDocument
base = next(o for o in doc.Objects if o.Label == "CubeWithHole")

fillet = doc.addObject("Part::Fillet", "Fillet")
fillet.Base = base
edges = []
for i in range(len(base.Shape.Edges)):
    edges.append((i + 1, 2.0, 2.0))     # (edge index STARTING AT 1, r_start, r_end)
fillet.Edges = edges
base.Visibility = False
doc.recompute()
```

Fillet only edges near a point (e.g. the corner the PINN flagged as a weak
point): filter by edge midpoint distance before appending —

```python
target = App.Vector(25, 0, 10)          # weak point from the stress report
for i, e in enumerate(base.Shape.Edges):
    mid = e.valueAt(0.5 * (e.FirstParameter + e.LastParameter))
    if (mid - target).Length < 8.0:     # only edges within 8mm of the hotspot
        edges.append((i + 1, 3.0, 3.0))
```

Chamfer is identical with `"Part::Chamfer"`. If a fillet fails to compute,
the radius is too big for that edge — halve it and retry.
