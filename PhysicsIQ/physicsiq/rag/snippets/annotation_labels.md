# Annotating the model: text notes and dimension callouts

Keywords: annotate, annotation, label, note, text, dimension, callout, mark, document

```python
# Easiest: the built-in helper puts a note next to any object by Label,
# auto-writing its dimensions when text is omitted.
piq_tools.annotate("Gear")                       # "Gear 36.0 x 36.0 x 6.0 mm"
piq_tools.annotate("BasePlate", "mounting side") # custom text

# Manual Draft text at an exact 3D point:
note = Draft.make_text(["max stress here"],
                       placement=App.Placement(App.Vector(25, 0, 12),
                                               App.Rotation()))
note.ViewObject.FontSize = 4

# A linear dimension between two points:
dim = Draft.make_linear_dimension(App.Vector(0, 0, 0),
                                  App.Vector(80, 0, 0),
                                  App.Vector(40, -12, 0))  # where the text sits
App.ActiveDocument.recompute()
```

"Annotate the gears" style requests = loop over matching labels:
```python
for o in App.ActiveDocument.Objects:
    if "gear" in o.Label.lower() and hasattr(o, "Shape"):
        piq_tools.annotate(o.Label)
```
