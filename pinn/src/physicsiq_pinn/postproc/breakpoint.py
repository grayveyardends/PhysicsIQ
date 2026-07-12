"""breakpoint.py — THE break-point framework: stress field in, ranked
weak points out. This dict is the contract the CAD agent consumes to
decide WHERE to add a fillet or gusset.

safety factor SF = yield_strength / von_Mises_stress
    SF < 1   the part yields (breaks/deforms permanently) at this load
    SF ~ 2   the usual "sleep well at night" engineering target
"""

import numpy as np


def rank_weak_points(points_mm, von_mises_mpa, yield_mpa,
                     top_k=5, min_separation_mm=None):
    """Return the top_k weakest SPOTS (not just the k lowest points —
    those would all be neighbours of the single worst point, telling the
    agent the same thing five times). Greedy de-duplication: take the
    worst point, blank out its neighbourhood, repeat."""
    points = np.asarray(points_mm, dtype=float)
    vm = np.asarray(von_mises_mpa, dtype=float)
    sf = yield_mpa / np.maximum(vm, 1e-9)

    if min_separation_mm is None:
        # Default: 10% of the part's largest extent — one hotspot per
        # geometric feature, roughly.
        extent = points.max(axis=0) - points.min(axis=0)
        min_separation_mm = 0.10 * float(extent.max())

    order = np.argsort(sf)          # weakest first
    alive = np.ones(len(sf), dtype=bool)
    picks = []
    for idx in order:
        if not alive[idx]:
            continue
        picks.append(idx)
        if len(picks) >= top_k:
            break
        # blank the neighbourhood of this pick
        d = np.linalg.norm(points - points[idx], axis=1)
        alive[d < min_separation_mm] = False

    weak_points = [{
        "location_mm": [round(float(c), 2) for c in points[i]],
        "von_mises_mpa": round(float(vm[i]), 1),
        "safety_factor": round(float(sf[i]), 2),
    } for i in picks]

    return {
        "yield_mpa": yield_mpa,
        "max_von_mises_mpa": round(float(vm.max()), 1),
        "min_safety_factor": round(float(sf.min()), 2),
        "weak_points": weak_points,
    }
