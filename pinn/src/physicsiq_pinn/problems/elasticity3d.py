"""elasticity3d.py — 3D linear elasticity on REAL CAD geometry, in the
MIXED (displacement + stress) formulation.

Why mixed? Our first attempt output only displacement u and derived stress
by differentiating twice. On bending-dominated parts (brackets!) that
trains into a bogus near-zero field: the true stress is ~(length/thickness)²
times the applied traction, and Adam never finds the path there — every
loss term except the traction BC is happily minimized by "nothing moves".
The known fix (Rao/Sun/Liu-style mixed PINNs): the net ALSO outputs the 6
stress components, and physics becomes three well-conditioned couplings:

    constitutive:  σ_net  ==  Hooke(∇u_net)     (1st derivatives only)
    equilibrium:   div σ_net == 0               (1st derivatives only)
    traction BCs:  σ_net·n == t                 (NO derivatives — linear!)

Traction BCs acting directly on an output is what drags the stress head —
and through the constitutive law the displacement head — up to the right
magnitude. Two learnable log-scales let each head grow to whatever size
the part's slenderness demands without the net leaving its O(1) comfort
zone.

Geometry comes in as point clouds (see geometry/):
    interior_pts (N,3) mm; boundary groups with points+normals per role:
        "fixed"    u = 0
        "traction" σ·n = t   (t = force/area, MPa)
        "free"     σ·n = 0

Nondimensional throughout: x̂=x/L, σ̂=σ/t_ref, û=u·E/(t_ref·L).
"""

import jax
import jax.numpy as jnp
import numpy as np

from physicsiq_pinn.models import mlp
from physicsiq_pinn.problems import base

# order of the 6 unique stress components in the net output
# (xx, yy, zz, xy, yz, xz)
_IDX = jnp.array([[0, 3, 5],
                  [3, 1, 4],
                  [5, 4, 2]])


class Elasticity3D:
    dim = 3

    def __init__(self, interior_pts, boundary_groups, material=None,
                 hidden=(64, 64, 64, 64), sigma_ref_mpa=None,
                 beam=None):
        """sigma_ref_mpa: the expected PEAK stress (use
        bc_extract.estimate_stress_scale). Nondimensionalizing by it puts
        the true solution at σ̂ ~ O(1), which is the difference between
        converging and settling into a bogus zero-stress field. Falls
        back to the applied traction magnitude when not given (fine for
        stubby tension-dominated parts, wrong for slender bending ones).

        beam: dict from bc_extract.beam_prior. When given, the stress head
        becomes  σ̂ = σ̂_beam(x) + net_correction  — the net starts at the
        beam-theory answer and only learns 3D effects and concentrations.
        Without it, slender bending parts do not converge at all.
        """
        self.mat = material or base.Material()
        self.layers = [3, *hidden, 9]      # 3 displacement + 6 stress
        self.beam = beam if (beam and beam.get("valid")) else None

        pts = np.asarray(interior_pts, dtype=np.float64)
        self.origin = pts.min(axis=0)
        self.L = float((pts.max(axis=0) - self.origin).max())
        tractions = [np.linalg.norm(g["traction_mpa"])
                     for g in boundary_groups if g["role"] == "traction"]
        t_load = float(max(tractions)) if tractions else 1.0
        self.sigma_ref = float(sigma_ref_mpa) if sigma_ref_mpa else t_load

        self.x_pde = jnp.asarray(self._hat(pts))

        # Force-consistent BC weights: a stress error ε on a face carries
        # ε·(face area) Newtons of bogus force, so each traction/free
        # group is weighted by ((σ_ref·area)/‖F_applied‖)² — "how many
        # applied-loads of force could this face's error hide?". Two
        # failure modes this kills (both found the hard way):
        #   * big free faces leaking the load out as invisible tractions
        #     (area factor), and
        #   * the optimizer abandoning the σ̂=t/σ_ref≈0.02 traction target
        #     entirely because missing it costs ~0.0004 (the σ_ref/t
        #     factor makes missing it cost ~1, like before rescaling).
        load_area = max((g.get("area_mm2", 0.0) for g in boundary_groups
                         if g["role"] == "traction"), default=0.0) or 1.0
        f_char = t_load * load_area if tractions else 1.0

        self.groups = []
        for g in boundary_groups:
            area = g.get("area_mm2", load_area)
            # (σ_ref/t_load)² makes ignoring the tiny σ̂-target as costly
            # as before rescaling; ×(area ratio) grows the guard on big
            # faces LINEARLY — the quadratic version ("all errors sum
            # coherently into force") over-guards so hard the optimizer
            # can't move the field at all near shared edges.
            w = ((self.sigma_ref / t_load) ** 2) * (area / load_area)
            item = {"role": g["role"],
                    "points": jnp.asarray(self._hat(np.asarray(g["points"]))),
                    "normals": jnp.asarray(np.asarray(g["normals"])),
                    "weight": w}
            if g["role"] == "traction":
                item["t_hat"] = jnp.asarray(
                    np.asarray(g["traction_mpa"]) / self.sigma_ref)
            self.groups.append(item)

    def _hat(self, x_mm):
        return (x_mm - self.origin) / self.L

    def params_init(self, key):
        # log_su / log_ss: one trainable number per head that scales the
        # whole field by exp(·). Slender parts need û and σ̂ tens-to-
        # hundreds of times larger than the naive scale — the net stays
        # O(1) and these absorb the magnitude.
        return {"net": mlp.init_params(key, self.layers),
                "log_su": jnp.zeros(()),
                "log_ss": jnp.zeros(())}

    def _u(self, params, x):
        out = mlp.apply(params["net"], x)
        return jnp.exp(params["log_su"]) * out[:3]

    def _sigma_beam(self, x):
        """The beam-theory baseline stress σ̂ at one hat-point. Pure
        analytic function of position — no trainable parts, so its
        derivatives (for the equilibrium loss) come for free from JAX."""
        beam = self.beam
        x_mm = x * self.L + jnp.asarray(self.origin)
        e = jnp.asarray(beam["e"])
        b = jnp.asarray(beam["b"])
        s = (x_mm - jnp.asarray(beam["c_fix"])) @ e

        # per-slice section properties, interpolated smoothly along s
        s_tab = jnp.asarray(beam["s"])
        A = jnp.interp(s, s_tab, jnp.asarray(beam["A"]))
        I = jnp.interp(s, s_tab, jnp.asarray(beam["I"]))
        centroid = jnp.stack([
            jnp.interp(s, s_tab, jnp.asarray(beam["centroid"][:, k]))
            for k in range(3)])

        moment = beam["f_perp"] * jnp.clip(beam["arm"] - s, 0.0, None)
        xi = (x_mm - centroid) @ b               # distance from neutral axis
        sig_axial = moment * xi / I + beam["f_ax"] / A   # bending + axial
        sig_shear = beam["f_perp"] / A

        ee = jnp.outer(e, e)
        eb = jnp.outer(e, b)
        sigma = sig_axial * ee + sig_shear * 0.5 * (eb + eb.T)
        # beam theory can blow up where a slice was thin/noisy — clamp to
        # a sane multiple of the expected peak, the net corrects the rest
        return jnp.clip(sigma / self.sigma_ref, -3.0, 3.0)

    def _sigma_net(self, params, x):
        """Stress head: beam-theory baseline + learned correction, as a
        symmetric 3x3 tensor. This is what BCs and equilibrium act on."""
        out = mlp.apply(params["net"], x)
        correction = jnp.exp(params["log_ss"]) * out[3:][_IDX]
        if self.beam is None:
            return correction
        return self._sigma_beam(x) + correction

    def _sigma_from_u(self, params, x):
        """Stress derived from the displacement head via Hooke's law —
        only used to TIE the two heads together (constitutive loss)."""
        J = jax.jacfwd(lambda p: self._u(params, p))(x)
        return base.iso3d_sigma(base.strain_from_jacobian(J), self.mat.nu)

    def _div_sigma(self, params, x):
        dS = jax.jacfwd(lambda p: self._sigma_net(params, p))(x)  # (3,3,3)
        return jnp.array([
            dS[0, 0, 0] + dS[0, 1, 1] + dS[0, 2, 2],
            dS[1, 0, 0] + dS[1, 1, 1] + dS[1, 2, 2],
            dS[2, 0, 0] + dS[2, 1, 1] + dS[2, 2, 2]])

    def loss_terms(self, params):
        vdiv = jax.vmap(lambda p: self._div_sigma(params, p))
        vsig = jax.vmap(lambda p: self._sigma_net(params, p))
        vsig_u = jax.vmap(lambda p: self._sigma_from_u(params, p))
        vu = jax.vmap(lambda p: self._u(params, p))

        terms = {
            "pde": jnp.mean(vdiv(self.x_pde) ** 2),
            "constitutive": jnp.mean(
                (vsig(self.x_pde) - vsig_u(self.x_pde)) ** 2),
        }
        for i, g in enumerate(self.groups):
            if g["role"] == "fixed":
                terms[f"bc_fixed{i}"] = jnp.mean(vu(g["points"]) ** 2)
            else:
                target = g.get("t_hat", jnp.zeros(3))  # free faces: t = 0
                tr = jnp.einsum("nij,nj->ni", vsig(g["points"]), g["normals"])
                terms[f"bc_{g['role']}{i}"] = (
                    g["weight"] * jnp.mean((tr - target) ** 2))
        return terms

    def eval_points(self):
        """Points where we report stress, in ORIGINAL mm coordinates.
        Interior + all boundary points (hotspots live on surfaces!)."""
        hat = [np.asarray(self.x_pde)]
        hat += [np.asarray(g["points"]) for g in self.groups]
        hat = np.vstack(hat)
        return hat * self.L + self.origin

    def von_mises(self, params, x_mm):
        x_hat = jnp.asarray(self._hat(np.asarray(x_mm)))
        vsig = jax.vmap(lambda p: self._sigma_net(params, p))
        vm_hat = jax.vmap(base.von_mises_3d)(vsig(x_hat))
        return np.asarray(vm_hat) * self.sigma_ref   # back to MPa
