"""elasticity3d.py — 3D Navier-Cauchy elasticity on REAL CAD geometry.

Same math as elasticity2d, one dimension up, and the geometry comes from
files instead of formulas:

    interior points   (N,3)  — sampled inside the STEP solid (geometry/sampler)
    boundary groups           — points+normals per role, from bc_extract:
        "fixed"    u = 0            (the bolted/clamped face)
        "traction" σ·n = t          (the loaded face, t = force/area in MPa)
        "free"     σ·n = 0          (every other surface)

Nondimensionalization (crucial for convergence, don't remove it):
    x̂ = x / L        L = longest bbox edge of the part
    σ̂ = σ / t_ref    t_ref = magnitude of the applied traction
    û = u · E / (t_ref · L)
"""

import jax
import jax.numpy as jnp
import numpy as np

from physicsiq_pinn.models import mlp
from physicsiq_pinn.problems import base


class Elasticity3D:
    dim = 3

    def __init__(self, interior_pts, boundary_groups, material=None,
                 hidden=(64, 64, 64, 64)):
        """
        interior_pts: (N,3) float array, mm, inside the solid.
        boundary_groups: list of dicts:
            {"role": "fixed"|"traction"|"free",
             "points": (M,3) mm, "normals": (M,3) unit,
             "traction_mpa": (3,) — only for role "traction"}
        """
        self.mat = material or base.Material()
        self.layers = [3, *hidden, 3]

        pts = np.asarray(interior_pts, dtype=np.float64)
        # --- nondimensionalize once, keep the factors to convert back ---
        self.origin = pts.min(axis=0)
        self.L = float((pts.max(axis=0) - self.origin).max())
        tractions = [np.linalg.norm(g["traction_mpa"])
                     for g in boundary_groups if g["role"] == "traction"]
        self.t_ref = float(max(tractions)) if tractions else 1.0

        self.x_pde = jnp.asarray(self._hat(pts))
        self.groups = []
        for g in boundary_groups:
            item = {"role": g["role"],
                    "points": jnp.asarray(self._hat(np.asarray(g["points"]))),
                    "normals": jnp.asarray(np.asarray(g["normals"]))}
            if g["role"] == "traction":
                item["t_hat"] = jnp.asarray(
                    np.asarray(g["traction_mpa"]) / self.t_ref)
            self.groups.append(item)

    def _hat(self, x_mm):
        return (x_mm - self.origin) / self.L

    # ------------------------------------------------------------------
    def params_init(self, key):
        return mlp.init_params(key, self.layers)

    def _u(self, params, x):
        # No hard constraints here: on arbitrary CAD shapes we can't bake
        # the fixed face into a formula, so "fixed" is a soft loss below.
        return mlp.apply(params, x)

    def _sigma(self, params, x):
        J = jax.jacfwd(lambda p: self._u(params, p))(x)
        eps = base.strain_from_jacobian(J)
        return base.iso3d_sigma(eps, self.mat.nu)

    def _div_sigma(self, params, x):
        dS = jax.jacfwd(lambda p: self._sigma(params, p))(x)  # (3,3,3)
        return jnp.array([
            dS[0, 0, 0] + dS[0, 1, 1] + dS[0, 2, 2],
            dS[1, 0, 0] + dS[1, 1, 1] + dS[1, 2, 2],
            dS[2, 0, 0] + dS[2, 1, 1] + dS[2, 2, 2]])

    # ------------------------------------------------------------------
    def loss_terms(self, params):
        vdiv = jax.vmap(lambda p: self._div_sigma(params, p))
        vsig = jax.vmap(lambda p: self._sigma(params, p))
        vu = jax.vmap(lambda p: self._u(params, p))

        terms = {"pde": jnp.mean(vdiv(self.x_pde) ** 2)}
        for i, g in enumerate(self.groups):
            if g["role"] == "fixed":
                terms[f"bc_fixed{i}"] = jnp.mean(vu(g["points"]) ** 2)
            else:
                target = g.get("t_hat", jnp.zeros(3))  # free faces: t = 0
                tr = jnp.einsum("nij,nj->ni", vsig(g["points"]), g["normals"])
                terms[f"bc_{g['role']}{i}"] = jnp.mean((tr - target) ** 2)
        return terms

    # ------------------------------------------------------------------
    def eval_points(self):
        """Points where we report stress, in ORIGINAL mm coordinates.
        Interior + all boundary points (hotspots live on surfaces!)."""
        hat = [np.asarray(self.x_pde)]
        hat += [np.asarray(g["points"]) for g in self.groups]
        hat = np.vstack(hat)
        return hat * self.L + self.origin

    def von_mises(self, params, x_mm):
        x_hat = jnp.asarray(self._hat(np.asarray(x_mm)))
        vsig = jax.vmap(lambda p: self._sigma(params, p))
        vm_hat = jax.vmap(base.von_mises_3d)(vsig(x_hat))
        return np.asarray(vm_hat) * self.t_ref   # back to MPa
