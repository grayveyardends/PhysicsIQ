"""bc_extract.py — turns a human-friendly bcs.json into boundary groups
the Elasticity3D problem understands.

bcs.json (written by the FreeCAD workbench, usually filled by the LLM
from the user's words: "bolted at the base, 500 N downward on the tip"):

    {
      "fixed": {"rule": "zmin"},
      "load":  {"rule": "xmax", "force_N": [0, 0, -500]},
      "material": {"E": 70000, "nu": 0.33, "yield_mpa": 276}
    }

Rules pick FACES:
    "xmin"/"xmax"/"ymin"/"ymax"/"zmin"/"zmax" — faces whose center sits at
        that extreme of the part (within 5% of its size). "The bottom" = zmin.
    {"indices": [3, 7]} — explicit face indices for when you know exactly.
The applied traction is force / picked-face-area (MPa), i.e. we spread
the load evenly over the chosen face(s). Every unpicked face is "free".
"""

import numpy as np

from physicsiq_pinn.problems.base import Material

_AXIS = {"x": 0, "y": 1, "z": 2}


def _faces_for_rule(rule, faces, bbox):
    if isinstance(rule, dict) and "indices" in rule:
        return list(rule["indices"])
    axis = _AXIS[rule[0]]
    lo = bbox[axis]
    hi = bbox[axis + 3]
    tol = 0.05 * max(bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2])
    target = lo if rule.endswith("min") else hi
    picked = [f["index"] for f in faces
              if abs(f["center"][axis] - target) < tol]
    if not picked:
        # Fall back to the single closest face — a slightly slanted base
        # should still count as "the bottom".
        picked = [min(faces,
                      key=lambda f: abs(f["center"][axis] - target))["index"]]
    return picked


def build_groups(npz_data, faces_meta, spec):
    """Returns (boundary_groups list for Elasticity3D, Material)."""
    faces = faces_meta["faces"]
    bbox = faces_meta["bbox"]
    ids = npz_data["boundary_face_ids"]
    pts = npz_data["boundary_pts"]
    nrm = npz_data["boundary_normals"]

    fixed_faces = set(_faces_for_rule(spec["fixed"]["rule"], faces, bbox))
    load_faces = set(_faces_for_rule(spec["load"]["rule"], faces, bbox))
    load_faces -= fixed_faces          # a face can't be both

    force = np.asarray(spec["load"]["force_N"], dtype=float)
    load_area = sum(f["area"] for f in faces if f["index"] in load_faces)
    traction_mpa = force / max(load_area, 1e-9)   # N / mm² = MPa
    print(f"fixed faces {sorted(fixed_faces)}, load faces "
          f"{sorted(load_faces)}, traction {traction_mpa.round(3)} MPa",
          flush=True)

    def mask(face_set):
        return np.isin(ids, list(face_set))

    m_fixed, m_load = mask(fixed_faces), mask(load_faces)
    m_free = ~(m_fixed | m_load)

    groups = [
        {"role": "fixed", "points": pts[m_fixed], "normals": nrm[m_fixed]},
        {"role": "traction", "points": pts[m_load], "normals": nrm[m_load],
         "traction_mpa": traction_mpa},
        {"role": "free", "points": pts[m_free], "normals": nrm[m_free]},
    ]

    mat_spec = spec.get("material", {})
    material = Material(E=mat_spec.get("E", 70000.0),
                        nu=mat_spec.get("nu", 0.33),
                        yield_mpa=mat_spec.get("yield_mpa", 276.0))
    return groups, material
