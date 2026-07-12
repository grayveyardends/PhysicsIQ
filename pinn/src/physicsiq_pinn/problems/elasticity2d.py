"""elasticity2d.py — plane-stress Navier-Cauchy on the classic benchmark:
a plate with a circular hole pulled from one side (the "Kirsch problem").

WHY THIS PROBLEM: it has a textbook analytical answer — the stress at the
edge of the hole is 3x the applied stress (stress concentration factor
Kt = 3 for an infinite plate). If our PINN reproduces that, the whole
pipeline (autodiff → PDE residual → training → von Mises) is CORRECT,
and we can trust the same machinery on real 3D parts where no textbook
answer exists.

Geometry (quarter model, using symmetry — standard trick, converges 4x faster):

        ŷ=1  ────────────── (traction-free)
        │                  │
   sym  │                  │ ← pulled with stress S (σxx = S)
   ux=0 │   ◜ hole r=â     │
        └──╯───────────────  ŷ=0, sym uy=0
        x̂=0                x̂=1

Everything inside is NONDIMENSIONAL: coordinates /L, stress /S,
displacement /(S·L/E). See base.py for why.
"""

import jax
import jax.numpy as jnp
import numpy as np

from physicsiq_pinn.models import mlp
from physicsiq_pinn.problems import base


class KirschPlate:
    dim = 2

    def __init__(self, material=None, hole_ratio=0.2, seed=0,
                 n_interior=2500, n_ring=1200, n_edge=250,
                 hidden=(64, 64, 64)):
        self.mat = material or base.Material()
        self.a = hole_ratio          # hole radius / plate size
        self.layers = [2, *hidden, 2]
        rng = np.random.default_rng(seed)
        self._sample_points(rng, n_interior, n_ring, n_edge)

    # ------------------------------------------------------------------
    # geometry sampling (pure numpy — happens once, before training)
    # ------------------------------------------------------------------
    def _sample_points(self, rng, n_interior, n_ring, n_edge):
        a = self.a

        # Interior: uniform points in the unit square, rejecting the hole.
        pts = rng.uniform(0, 1, size=(int(n_interior * 1.6), 2))
        pts = pts[np.hypot(pts[:, 0], pts[:, 1]) > a][:n_interior]

        # Extra ring of points hugging the hole — that is where the stress
        # gradient is steep, so the PDE loss needs the most eyes there.
        r = rng.uniform(a, 3 * a, size=n_ring)
        th = rng.uniform(0, np.pi / 2, size=n_ring)
        ring = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
        ring = ring[(ring[:, 0] <= 1) & (ring[:, 1] <= 1)]

        self.x_pde = jnp.asarray(np.vstack([pts, ring]))

        # Boundary points, one array per edge:
        t = rng.uniform(0, 1, size=n_edge)
        self.x_right = jnp.asarray(np.stack([np.ones_like(t), t], 1))   # x̂=1: pulled
        self.x_top = jnp.asarray(np.stack([t, np.ones_like(t)], 1))     # ŷ=1: free
        th = rng.uniform(0, np.pi / 2, size=n_edge)
        self.x_hole = jnp.asarray(
            a * np.stack([np.cos(th), np.sin(th)], 1))                  # hole arc: free
        self.hole_n = jnp.asarray(
            -np.stack([np.cos(th), np.sin(th)], 1))                     # inward normal
        self.x_symx = jnp.asarray(np.stack([np.zeros_like(t), t], 1))   # x̂=0
        self.x_symy = jnp.asarray(np.stack([t, np.zeros_like(t)], 1))   # ŷ=0

    # ------------------------------------------------------------------
    # the displacement field (net + hard boundary conditions)
    # ------------------------------------------------------------------
    def params_init(self, key):
        return mlp.init_params(key, self.layers)

    def _u(self, params, x):
        """Displacement at ONE point, with symmetry built into the FORMULA:
        ûx = x̂·N(...) is zero at x̂=0 no matter what the net says. Hard
        constraints like this beat soft penalty losses every time you can
        afford them."""
        n = mlp.apply(params, x)
        return jnp.array([x[0] * n[0], x[1] * n[1]])

    def _sigma(self, params, x):
        """Nondimensional stress tensor at one point: differentiate the
        displacement (jacfwd), symmetrize to strain, Hooke's law."""
        J = jax.jacfwd(lambda p: self._u(params, p))(x)
        eps = base.strain_from_jacobian(J)
        return base.plane_stress_sigma(eps, self.mat.nu)

    def _div_sigma(self, params, x):
        """Equilibrium residual ∇·σ at one point. jacfwd of the STRESS
        function gives dσij/dxk in one shot — this is the 'second
        derivative of the net' that PINNs are famous for."""
        dS = jax.jacfwd(lambda p: self._sigma(params, p))(x)  # (2,2,2)
        # row i: dσ_i0/dx0 + dσ_i1/dx1
        return jnp.array([dS[0, 0, 0] + dS[0, 1, 1],
                          dS[1, 0, 0] + dS[1, 1, 1]])

    # ------------------------------------------------------------------
    # loss terms — the trainer just sums these
    # ------------------------------------------------------------------
    def loss_terms(self, params):
        vsig = jax.vmap(lambda p: self._sigma(params, p))
        vdiv = jax.vmap(lambda p: self._div_sigma(params, p))

        pde = jnp.mean(vdiv(self.x_pde) ** 2)

        s_right = vsig(self.x_right)         # pulled edge: σxx=1, σxy=0
        bc_load = (jnp.mean((s_right[:, 0, 0] - 1.0) ** 2)
                   + jnp.mean(s_right[:, 0, 1] ** 2))

        s_top = vsig(self.x_top)             # free edge: σyy=0, σxy=0
        bc_top = jnp.mean(s_top[:, 1, 1] ** 2) + jnp.mean(s_top[:, 0, 1] ** 2)

        s_hole = vsig(self.x_hole)           # hole: traction σ·n = 0
        tr = jnp.einsum("nij,nj->ni", s_hole, self.hole_n)
        bc_hole = jnp.mean(tr ** 2)

        # Symmetry shear must vanish (displacement part is hard-enforced).
        s_sx = vsig(self.x_symx)
        s_sy = vsig(self.x_symy)
        bc_sym = (jnp.mean(s_sx[:, 0, 1] ** 2) + jnp.mean(s_sy[:, 0, 1] ** 2))

        return {"pde": pde, "bc_load": bc_load, "bc_free": bc_top,
                "bc_hole": bc_hole, "bc_sym": bc_sym}

    # ------------------------------------------------------------------
    # evaluation / postprocessing hooks
    # ------------------------------------------------------------------
    def eval_points(self):
        return np.asarray(self.x_pde)

    def von_mises(self, params, x, applied_stress_mpa=100.0):
        """Physical von Mises in MPa: nondimensional σ̂ times the applied S."""
        vsig = jax.vmap(lambda p: self._sigma(params, p))
        vm = jax.vmap(base.von_mises_2d)(vsig(jnp.asarray(x)))
        return np.asarray(vm * applied_stress_mpa)

    def stress_concentration(self, params):
        """Kt = σxx at the top of the hole (x̂=0, ŷ=â) / applied stress.
        Kirsch says ≈ 3.0 — THE number that validates this package."""
        s = self._sigma(params, jnp.array([0.0, self.a]))
        return float(s[0, 0])
