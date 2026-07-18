# Simple spur gear (approximate teeth)

Keywords: gear, spur, teeth, cog, sprocket, pinion, mesh

```python
# Approximate spur gear: disc + trapezoid teeth in a polar pattern.
# Good for visual mockups & 3D prints, NOT an involute profile.
import math
import Part as _P
doc = App.ActiveDocument

TEETH, MODULE, THICK = 16, 2.0, 6.0      # module = mm of pitch dia per tooth
pitch_r = MODULE * TEETH / 2
root_r = pitch_r - 1.25 * MODULE
tip_r = pitch_r + 1.0 * MODULE
BORE_R = 4.0

disc = _P.makeCylinder(root_r, THICK)
gear = disc
half_t = math.pi * pitch_r / TEETH / 2 * 0.9   # half tooth width at pitch circle
for i in range(TEETH):
    a = 2 * math.pi * i / TEETH
    # Tooth = tapered box from root to tip, rotated into place.
    tooth = _P.makeBox(tip_r - root_r + 0.5, 2 * half_t, THICK)
    tooth.translate(App.Vector(root_r - 0.25, -half_t, 0))
    tooth.rotate(App.Vector(0, 0, 0), App.Vector(0, 0, 1), math.degrees(a))
    gear = gear.fuse(tooth)
bore = _P.makeCylinder(BORE_R, THICK + 2)
bore.translate(App.Vector(0, 0, -1))
gear = gear.cut(bore)

obj = doc.addObject("Part::Feature", "Gear")
obj.Shape = gear.removeSplitter()        # cleans internal faces after fusing
obj.Label = "Gear"
doc.recompute()
```

Two meshing gears: center distance = (teeth1 + teeth2) * module / 2.
