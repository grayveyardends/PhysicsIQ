"""base.py — what every PDE problem must provide, plus shared mechanics math.

To add a new physics (heat, CFD, ...) you write ONE new file in this
folder that satisfies the Problem contract below. The trainer and the CLI
never need to change — they only know this interface:

    problem.params_init(key)      -> initial net parameters
    problem.loss_terms(params)    -> {"pde": scalar, "bc_load": scalar, ...}
    problem.eval_points()         -> (N, dim) points to evaluate results at
    problem.von_mises(params, x)  -> (N,) stress in MPa at those points

Units convention across the whole package:
    lengths mm, forces N, stress/pressure MPa (N/mm²), E in MPa.
"""

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class Material:
    E: float = 70000.0        # Young's modulus, MPa (default: aluminium)
    nu: float = 0.33          # Poisson's ratio
    yield_mpa: float = 276.0  # yield strength (6061-T6)


# ----------------------------------------------------------------------
# Small-strain mechanics helpers shared by 2D and 3D problems.
# All are written for ONE point and vmapped by the problem classes —
# single-point code is far easier to read and to differentiate.
# ----------------------------------------------------------------------

def strain_from_jacobian(J):
    """ε = ½(∇u + ∇uᵀ). J is (dim, dim) = jacobian of displacement."""
    return 0.5 * (J + J.T)


def plane_stress_sigma(eps, nu):
    """2D plane-stress constitutive law, NONDIMENSIONAL form.

    Input strain is ε̂ (strain per unit S/E); output is σ̂ = σ/S.
    Dividing everything by S and E keeps numbers ~1, which tanh nets
    and Adam both love. Convert back: σ = σ̂ * S.
    """
    exx, eyy, exy = eps[0, 0], eps[1, 1], eps[0, 1]
    sxx = (exx + nu * eyy) / (1 - nu**2)
    syy = (eyy + nu * exx) / (1 - nu**2)
    sxy = exy / (1 + nu)
    return jnp.array([[sxx, sxy], [sxy, syy]])


def iso3d_sigma(eps, nu):
    """3D isotropic Hooke's law, nondimensional (σ̂ = σ/t_ref, ε̂ = ε·E/t_ref):
    σ̂ = (λ/E)·tr(ε̂)·I + 2(μ/E)·ε̂ — the E's cancel into pure ν ratios."""
    lam_over_E = nu / ((1 + nu) * (1 - 2 * nu))
    mu_over_E = 1.0 / (2 * (1 + nu))
    return lam_over_E * jnp.trace(eps) * jnp.eye(3) + 2 * mu_over_E * eps


def von_mises_2d(s):
    """Plane stress von Mises from a 2x2 stress tensor."""
    sxx, syy, sxy = s[0, 0], s[1, 1], s[0, 1]
    return jnp.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2)


def von_mises_3d(s):
    sxx, syy, szz = s[0, 0], s[1, 1], s[2, 2]
    sxy, syz, sxz = s[0, 1], s[1, 2], s[0, 2]
    return jnp.sqrt(0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2)
                    + 3 * (sxy**2 + syz**2 + sxz**2))
