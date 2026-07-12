"""freecad_sample_script.py — runs under `freecadcmd` (FreeCAD's headless
Python), NOT in the JAX venv. It is the only place we touch OCCT.

Job: STEP file in → point clouds out (.npz + .faces.json):
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

import numpy as np

import FreeCAD  # noqa: F401 — importing it initializes the OCCT kernel
import Part

step_path = os.environ["PIQ_STEP"]
out_npz = os.environ["PIQ_OUT"]
n_interior = int(os.environ.get("PIQ_N_INTERIOR", "2000"))
n_boundary = int(os.environ.get("PIQ_N_BOUNDARY", "1500"))

shape = Part.Shape()
shape.read(step_path)
if not shape.Solids:
    raise SystemExit("STEP file contains no solid")
solid = shape.Solids[0]

bb = solid.BoundBox
L = max(bb.XLength, bb.YLength, bb.ZLength)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------------
# Interior: rejection sampling. Throw random points into the bounding
# box, keep the ones OCCT says are inside. Dumb, robust, fast enough.
# ---------------------------------------------------------------------
interior = []
attempts = 0
while len(interior) < n_interior and attempts < 60 * n_interior:
    attempts += 1
    p = FreeCAD.Vector(rng.uniform(bb.XMin, bb.XMax),
                       rng.uniform(bb.YMin, bb.YMax),
                       rng.uniform(bb.ZMin, bb.ZMax))
    if solid.isInside(p, 1e-6, False):   # False = strictly inside, not on skin
        interior.append([p.x, p.y, p.z])
interior = np.array(interior)
print(f"sampled {len(interior)} interior points "
      f"({attempts} attempts)", flush=True)

# ---------------------------------------------------------------------
# Boundary: tessellate each face into triangles, then sample points on
# the triangles (area-weighted). Tessellation handles ANY face shape —
# trimmed, curved, whatever — where naive UV-grid sampling falls apart.
# ---------------------------------------------------------------------
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
               "faces": faces_meta}, fh, indent=2)
print(f"wrote {out_npz}", flush=True)
