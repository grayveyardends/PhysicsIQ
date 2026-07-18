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

    def area_of(face_set):
        return sum(f["area"] for f in faces if f["index"] in face_set)

    free_faces = ({f["index"] for f in faces}
                  - fixed_faces - load_faces)
    groups = [
        {"role": "fixed", "points": pts[m_fixed], "normals": nrm[m_fixed],
         "area_mm2": area_of(fixed_faces)},
        {"role": "traction", "points": pts[m_load], "normals": nrm[m_load],
         "traction_mpa": traction_mpa, "area_mm2": area_of(load_faces)},
        {"role": "free", "points": pts[m_free], "normals": nrm[m_free],
         "area_mm2": area_of(free_faces)},
    ]

    mat_spec = spec.get("material", {})
    material = Material(E=mat_spec.get("E", 70000.0),
                        nu=mat_spec.get("nu", 0.33),
                        yield_mpa=mat_spec.get("yield_mpa", 276.0))
    return groups, material


def beam_prior(interior_pts, faces_meta, groups):
    """Treat the part as a crude beam along its load path and return both
    an ANALYTIC STRESS FIELD GUESS and the peak-stress estimate.

    Why: a slender part in bending carries stress 10–100× the applied
    traction. A PINN trained from scratch never finds that field (it
    settles for a bogus zero-stress solution — found the hard way). But
    given this rough beam field as a hard baseline, the net only has to
    learn the CORRECTION — 3D effects and the stress concentrations at
    corners, i.e. exactly the part beam theory can't see and exactly what
    we're here for. Same trick that makes the 2D Kirsch problem converge
    (far-field warm start), generalized to arbitrary geometry.

    Slicing: cut the interior point cloud into cross-sections along the
    fixed-face->load-face axis; each slice's area A and second moment I
    come from point counts (Monte-Carlo integration, volume is exact from
    OCCT). Beam formulas then give σ_axial = M·ξ/I + F∥/A and shear F⊥/A.
    """
    pts = np.asarray(interior_pts, dtype=float)
    volume = faces_meta.get("volume_mm3")
    if volume is None:   # older sample files — fall back to bbox estimate
        bb = faces_meta["bbox"]
        volume = 0.5 * (bb[3]-bb[0]) * (bb[4]-bb[1]) * (bb[5]-bb[2])

    g_fix = next(g for g in groups if g["role"] == "fixed")
    g_load = next(g for g in groups if g["role"] == "traction")
    force = np.asarray(g_load["traction_mpa"]) * g_load["area_mm2"]  # back to N
    t_load = float(np.linalg.norm(g_load["traction_mpa"]))

    c_fix = g_fix["points"].mean(axis=0)
    c_load = g_load["points"].mean(axis=0)
    axis = c_load - c_fix                     # the "beam axis"
    arm = float(np.linalg.norm(axis))
    empty = {"sigma_est": max(t_load, 1e-6), "valid": False}
    if arm < 1e-9:
        return empty
    e = axis / arm
    f_ax = float(force @ e)                   # axial component
    f_perp_vec = force - f_ax * e             # bending component
    f_perp = float(np.linalg.norm(f_perp_vec))
    b = f_perp_vec / f_perp if f_perp > 1e-9 else np.zeros(3)

    # Slice the interior points into ~15 cross-sections along the axis.
    s = (pts - c_fix) @ e
    per_pt_vol = volume / len(pts)
    n_bins = 15
    edges = np.linspace(s.min(), s.max(), n_bins + 1)
    ds = edges[1] - edges[0]

    s_centers, areas, inertias, centroids = [], [], [], []
    worst = t_load
    for i in range(n_bins):
        m = (s >= edges[i]) & (s < edges[i + 1])
        if m.sum() < 20:
            continue  # too few points for a trustworthy slice
        block = pts[m]
        area = m.sum() * per_pt_vol / ds              # slice cross-section
        centroid = block.mean(axis=0)
        sigma = abs(f_ax) / area + f_perp / area      # axial + shear
        inertia = 0.0
        if f_perp > 1e-9:
            xi = (block - centroid) @ b               # distance from N.A.
            inertia = float((xi ** 2).sum()) * per_pt_vol / ds
            c = float(np.percentile(np.abs(xi), 95))
            moment = f_perp * (arm - float(edges[i]))  # worst at fixed end
            if inertia > 1e-9:
                sigma += abs(moment) * c / inertia
        worst = max(worst, sigma)
        s_centers.append(0.5 * (edges[i] + edges[i + 1]))
        areas.append(area)
        inertias.append(max(inertia, 1e-9))
        centroids.append(centroid)

    print(f"estimated peak stress scale: {worst:.1f} MPa "
          f"(traction alone: {t_load:.2f})", flush=True)
    if len(s_centers) < 2:
        return empty
    return {
        "sigma_est": worst, "valid": True,
        "c_fix": c_fix, "e": e, "b": b,
        "arm": arm, "f_ax": f_ax, "f_perp": f_perp,
        "s": np.asarray(s_centers), "A": np.asarray(areas),
        "I": np.asarray(inertias), "centroid": np.asarray(centroids),
    }


def estimate_stress_scale(interior_pts, faces_meta, groups):
    """Peak-stress estimate only (see beam_prior for the full story)."""
    return beam_prior(interior_pts, faces_meta, groups)["sigma_est"]
