# Spreadsheet-driven parameters (master dimensions)

Keywords: spreadsheet, parameter, parametric, variable, formula, configure, driven, master, table

```python
# One spreadsheet cell drives many dimensions — change the cell, the whole
# model updates. This is how you make properly configurable parts.
doc = App.ActiveDocument

sheet = doc.addObject("Spreadsheet::Sheet", "Params")
sheet.set("A1", "plate_len");  sheet.set("B1", "90")
sheet.set("A2", "plate_th");   sheet.set("B2", "6")
sheet.setAlias("B1", "plate_len")     # aliases make cells referencable
sheet.setAlias("B2", "plate_th")
doc.recompute()                       # compute aliases BEFORE binding them

plate = doc.addObject("Part::Box", "Plate")
plate.setExpression("Length", "Params.plate_len")   # bound, not copied
plate.setExpression("Height", "Params.plate_th")
plate.Width = 50.0                                   # plain value is fine too
doc.recompute()

# Later, to resize the whole model:
sheet.set("B1", "120"); doc.recompute()
```
