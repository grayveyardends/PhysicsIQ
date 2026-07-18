#!/usr/bin/env python3
"""bracket_3d.py — end-to-end 3D pipeline test on the demo L-bracket:
geometry sampling -> BCs -> training -> weak-point report.

    ~/.venvs/usage/bin/python pinn/examples/bracket_3d.py [epochs]

(Builds the bracket STEP first via freecadcmd if it's missing.)
Expected outcome: the reported weak points cluster around the bracket's
sharp inner corner (x≈6, z≈6) — where any engineer would point too.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

epochs = sys.argv[1] if len(sys.argv) > 1 else "3000"
out_dir = os.path.join(HERE, "..", "..", "runs", "bracket")
step_path = os.path.join(out_dir, "part.step")

if not os.path.exists(step_path):
    print("building test bracket STEP via freecadcmd…")
    subprocess.run(["freecadcmd",
                    os.path.join(HERE, "make_test_bracket.py")], check=True)

bcs_path = os.path.join(out_dir, "bcs.json")
with open(bcs_path, "w", encoding="utf-8") as fh:
    json.dump({
        # bolted to the floor, 500 N pulling the top of the wall in +x
        "fixed": {"rule": "zmin"},
        "load": {"rule": "zmax", "force_N": [500.0, 0.0, 0.0]},
        "material": {"E": 70000, "nu": 0.33, "yield_mpa": 276},
    }, fh, indent=2)

from physicsiq_pinn import cli  # noqa: E402

sys.argv = ["cli", "--step", step_path, "--bcs", bcs_path,
            "--out", out_dir, "--epochs", epochs]
cli.main()

with open(os.path.join(out_dir, "result.json"), encoding="utf-8") as fh:
    result = json.load(fh)
print()
print("weak points found:")
for wp in result["weak_points"]:
    print(f"  {wp['location_mm']}  SF={wp['safety_factor']}")
