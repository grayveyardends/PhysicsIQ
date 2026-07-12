# Finding and editing objects that already exist

Keywords: edit, change, modify, resize, find, select, existing, update, rename, delete, remove

```python
doc = App.ActiveDocument

# Find by Label (the human-visible name in the model tree — USE THIS):
plate = next(o for o in doc.Objects if o.Label == "BasePlate")

# Change its dimensions — parametric objects update on recompute:
plate.Length = 100.0
plate.Height = 8.0

# What the user has clicked/selected in the GUI:
sel = Gui.Selection.getSelection()        # list of objects
if sel:
    print("selected:", sel[0].Label)

# Delete an object (by internal Name, not Label):
# doc.removeObject(plate.Name)

# List everything with types — useful before deciding what to edit:
for o in doc.Objects:
    print(o.Name, o.TypeId, o.Label)

doc.recompute()
```

Label vs Name: `Label` is what the user sees and can contain spaces;
`Name` is the immutable internal id ("Box001"). Look up by Label,
delete by Name. After changing any property, ALWAYS `doc.recompute()`.
