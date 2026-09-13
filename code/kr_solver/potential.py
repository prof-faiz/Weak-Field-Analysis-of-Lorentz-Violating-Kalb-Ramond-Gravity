"""Update rule for the Newtonian potential Phi(r).

Solves the modified Poisson equation

    [1 - B(r)] nabla^2 Phi(r) = 4 pi G [rho_M(r) + A(r)],

with
    B(r) = (3/2) xi_2 (a - b_{01}(r))^2,
    A(r) = T^{KR}_{00}(r)  excluding the term proportional to nabla^2 Phi,

as given in Sec. 3.3 of manuscript.tex. A one-step Jacobi update is
applied at each iteration.
"""

from __future__ import annotations

import numpy as np

from .kr_fields import _kr_scalar
from .parameters import Parameters


def _A_source(
    b01: np.ndarray,
    b23: np.ndarray,
    r: np.ndarray,
    dr: float,
    params: Parameters,
) -> np.ndarray:
    """KR energy density excluding the curvature-mixing term."""
    a = params.a
    mu, Lambda = params.mu, params.Lambda
    xi = params.xi

    db01_dr = np.gradient(b01, dr)
    db23_dr = np.gradient(b23, dr)
    X = _kr_scalar(b01, b23, params)

    kinetic = 0.5 * db01_dr ** 2 + (1.0 / 12.0) * db23_dr ** 2
    linear_pot = 2.0 * (mu ** 2 * X + Lambda * X ** 3) * (
        a ** 2 - 2.0 * a * b01 + b01 ** 2
    )
    pot_energy = -((1.0 / 12.0) * mu ** 2 * X ** 2 - 0.25 * Lambda * X ** 4)
    curv_self = xi * (2.0 * (a + b01) ** 2 * b01 + 3.0 * b01 ** 2)
    return kinetic + linear_pot + pot_energy + curv_self


def _B_factor(b01: np.ndarray, params: Parameters) -> np.ndarray:
    """Return B(r) = (3/2) xi (a - b_{01})^2."""
    return 1.5 * params.xi * (params.a - b01) ** 2


def solve_phi(
    phi: np.ndarray,
    rho: np.ndarray,
    b01: np.ndarray,
    b23: np.ndarray,
    r: np.ndarray,
    dr: float,
    params: Parameters,
) -> np.ndarray:
    """One Jacobi sweep for Phi over the modified Poisson equation."""
    A_src = _A_source(b01, b23, r, dr, params)
    B_fac = _B_factor(b01, params)
    denom = np.where(np.abs(1.0 - B_fac) < 1e-12, 1e-12, 1.0 - B_fac)
    rhs = 4.0 * np.pi * params.G * (rho + A_src) / denom

    new = phi.copy()
    for i in range(1, len(phi) - 1):
        alpha = 1.0 + dr / r[i]
        beta = 1.0 - dr / r[i]
        new[i] = 0.5 * (alpha * phi[i + 1] + beta * phi[i - 1] - rhs[i] * dr ** 2)

    new[0] = new[1]      # regularity: dPhi/dr = 0 at inner boundary
    new[-1] = 0.0        # asymptotic flatness
    return np.clip(new, -params.clip, params.clip)
