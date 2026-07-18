#!/usr/bin/env python3
"""make_test_bracket.py — builds the demo L-bracket STEP file.

Run under FreeCAD's headless python (NOT the venv):

    freecadcmd pinn/examples/make_test_bracket.py

Writes runs/bracket/part.step: an L-bracket with a deliberately sharp
inner corner — exactly the stress concentration the PINN should flag.
"""

import os

import FreeCAD as App
import Part

out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "..", "..", "runs", "bracket")
os.makedirs(out_dir, exist_ok=True)

doc = App.newDocument("bracket")

# Base plate 60x40x6 on the floor, wall 6x40x50 rising at x=0.
base = Part.makeBox(60, 40, 6)
wall = Part.makeBox(6, 40, 50)
bracket = base.fuse(wall).removeSplitter()

obj = doc.addObject("Part::Feature", "Bracket")
obj.Shape = bracket
doc.recompute()

step_path = os.path.join(out_dir, "part.step")
Part.export([obj], step_path)
print(f"wrote {step_path}")
