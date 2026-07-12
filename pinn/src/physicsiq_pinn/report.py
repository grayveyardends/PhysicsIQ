"""report.py — writes the two files that ARE the FreeCAD↔PINN contract.

    result.json   summary + ranked weak points (for the LLM agent)
    field.npz     points (N,3|2) + von_mises (N,)  (for the heatmap overlay)

Keep this format stable: the workbench (tools/pinn_runner.py and
tools/heatmap_overlay.py) parses exactly these keys.
"""

import json
import os

import numpy as np


def write(outdir, points_mm, von_mises_mpa, breakpoint_report, meta=None):
    os.makedirs(outdir, exist_ok=True)

    field_file = os.path.join(outdir, "field.npz")
    np.savez_compressed(field_file,
                        points=np.asarray(points_mm, dtype=np.float32),
                        von_mises=np.asarray(von_mises_mpa, dtype=np.float32))

    result = dict(breakpoint_report)
    result["field_file"] = field_file
    if meta:
        result["meta"] = meta
    result_file = os.path.join(outdir, "result.json")
    with open(result_file, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    # The magic line pinn_runner.py greps for in our stdout:
    print(f"RESULT_JSON {result_file}", flush=True)
    return result_file
