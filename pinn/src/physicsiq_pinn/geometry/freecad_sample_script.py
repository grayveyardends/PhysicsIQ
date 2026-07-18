"""freecad_sample_script.py — runs under `freecadcmd` (FreeCAD's headless
Python), NOT in the JAX venv. It is the only place we touch OCCT.

Job: STEP file in -> point clouds out (.npz + .faces.json):
    interior            (N,3) points strictly inside the solid
    boundary_pts        (M,3) points on the surface
    boundary_normals    (M,3) outward unit normals
    boundary_face_ids   (M,)  which face each point belongs to
    <out>.faces.json    per-face center/area/normal (bc_extract uses this)

Parameters come in via environment variables because freecadcmd's argv
handling is unreliable across versions:
    PIQ_STEP, PIQ_OUT, PIQ_N_INTERIOR, PIQ_N_BOUNDARY
"""

import json
import os
import sys

import numpy as np

import FreeCAD  # noqa: F401 — importing it initializes the OCCT kernel
import Part

step_path = os.environ["PIQ_STEP"]
out_npz = os.environ["PIQ_OUT"]
n_interior = int(os.environ.get("PIQ_N_INTERIOR", "2000"))
n_boundary = int(os.environ.get("PIQ_N_BOUNDARY", "1500"))

shape = Part.Shape()
shape.read(step_path)
solids = shape.Solids
if not solids:
    raise SystemExit("STEP file contains no solid")

# An assembly arrives as several solids. Elasticity needs ONE connected body
# — the load must have a path from the fixed face to the loaded face. The
# workbench already fuses before exporting, but a STEP handed to us directly
# (examples/, a file from disk) may not be, and taking Solids[0] and quietly
# analyzing one arm of a drone frame is the worst possible failure: it looks
# like it worked.
if len(solids) > 1:
    fused = solids[0].multiFuse(solids[1:]).removeSplitter()
    if len(fused.Solids) != 1:
        raise SystemExit(
            f"STEP holds {len(fused.Solids)} solids that do not touch. "
            "A stress analysis needs one connected body — fuse the parts.")
    solid = fused.Solids[0]
    print(f"fused {len(solids)} solids into one body", flush=True)
else:
    solid = solids[0]

bb = solid.BoundBox
L = max(bb.XLength, bb.YLength, bb.ZLength)
rng = np.random.default_rng(0)

# The native geomkit lib (built by the venv side, loaded here through
# ctypes — it has no python in it, so the interpreter mismatch doesn't
# matter) turns thousands of OCCT isInside round trips into one call
# against the tessellated surface. Optional: everything below falls
# back to the OCCT loops when it is missing.
_geom = None
try:
    pinn_src = os.environ.get("PIQ_PINN_SRC")
    if pinn_src and pinn_src not in sys.path:
        sys.path.insert(0, pinn_src)
    from physicsiq_pinn.geometry import native as _geom
    if _geom.load() is None:
        _geom = None
except Exception:
    _geom = None

_mesh_tris = None
if _geom is not None:
    mv, mf = solid.tessellate(0.01 * L)
    mv = np.array([[p.x, p.y, p.z] for p in mv])
    mf = np.array(mf)
    _mesh_tris = mv[mf]          # (T,3,3)

lo = np.array([bb.XMin, bb.YMin, bb.ZMin])
hi = np.array([bb.XMax, bb.YMax, bb.ZMax])

interior = np.empty((0, 3))
attempts = 0
if _mesh_tris is not None:
    # batch rejection sampling against the mesh; fill ratio adapts the
    # batch size so hollow/lattice parts don't need dozens of rounds
    fill = max(solid.Volume / max(np.prod(hi - lo), 1e-9), 1e-3)
    got = []
    while sum(len(g) for g in got) < n_interior and attempts < 100 * n_interior:
        want = n_interior - sum(len(g) for g in got)
        batch = int(min(4 * want / fill, 2e6))
        cand = rng.uniform(lo, hi, size=(batch, 3))
        attempts += batch
        keep = _geom.points_in_mesh(_mesh_tris, cand)
        got.append(cand[keep])
    interior = np.vstack(got)[:n_interior] if got else interior
if len(interior) < n_interior:
    # OCCT fallback: dumb, robust, slow. Also catches the (unlikely)
    # case of a mesh so coarse the batch sampler starved.
    pts = list(interior)
    occt_attempts = 0
    while len(pts) < n_interior and occt_attempts < 60 * n_interior:
        occt_attempts += 1
        p = FreeCAD.Vector(*rng.uniform(lo, hi))
        if solid.isInside(p, 1e-6, False):
            pts.append(np.array([p.x, p.y, p.z]))
    interior = np.array(pts)
    attempts += occt_attempts
print(f"sampled {len(interior)} interior points "
      f"({attempts} attempts, native={'yes' if _mesh_tris is not None else 'no'})",
      flush=True)

# Boundary: tessellate each face into triangles, then sample points on
# the triangles (area-weighted). Tessellation handles ANY face shape —
# trimmed, curved, whatever — where naive UV-grid sampling falls apart.
faces_meta = []
b_pts, b_nrm, b_ids = [], [], []
total_area = sum(f.Area for f in solid.Faces)

for fi, face in enumerate(solid.Faces):
    verts, tris = face.tessellate(0.02 * L)
    verts = np.array([[v.x, v.y, v.z] for v in verts])
    tris = np.array(tris)
    if len(tris) == 0:
        continue

    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    tri_area = 0.5 * np.linalg.norm(cross, axis=1)
    tri_n = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12)

    # this face's share of the point budget, at least a few points
    n_face = max(8, int(n_boundary * face.Area / total_area))
    probs = tri_area / tri_area.sum()
    chosen = rng.choice(len(tris), size=n_face, p=probs)
    # random barycentric coordinates = uniform points on a triangle
    r1, r2 = rng.uniform(size=n_face), rng.uniform(size=n_face)
    s = np.sqrt(r1)
    pts = ((1 - s)[:, None] * v0[chosen]
           + (s * (1 - r2))[:, None] * v1[chosen]
           + (s * r2)[:, None] * v2[chosen])
    nrm = tri_n[chosen]

    # Make normals point OUTWARD: nudge along the normal; if we end up
    # inside the solid, the normal was inward — flip it. Ground truth
    # beats trusting tessellation orientation.
    eps = 1e-3 * L
    if _mesh_tris is not None:
        flip = _geom.points_in_mesh(_mesh_tris, pts + eps * nrm)
        nrm[flip] = -nrm[flip]
    else:
        for k in range(len(pts)):
            probe = FreeCAD.Vector(*(pts[k] + eps * nrm[k]))
            if solid.isInside(probe, 1e-9, False):
                nrm[k] = -nrm[k]

    b_pts.append(pts)
    b_nrm.append(nrm)
    b_ids.append(np.full(n_face, fi))
    c = face.CenterOfMass
    faces_meta.append({"index": fi, "area": face.Area,
                       "center": [c.x, c.y, c.z],
                       "normal": [float(x) for x in nrm.mean(axis=0)]})

np.savez_compressed(
    out_npz,
    interior=interior,
    boundary_pts=np.vstack(b_pts),
    boundary_normals=np.vstack(b_nrm),
    boundary_face_ids=np.concatenate(b_ids),
)

with open(out_npz + ".faces.json", "w", encoding="utf-8") as fh:
    json.dump({"bbox": [bb.XMin, bb.YMin, bb.ZMin,
                        bb.XMax, bb.YMax, bb.ZMax],
               "volume_mm3": solid.Volume,   # exact, from OCCT — used to
                                             # turn point counts into areas
               "faces": faces_meta}, fh, indent=2)
print(f"wrote {out_npz}", flush=True)
