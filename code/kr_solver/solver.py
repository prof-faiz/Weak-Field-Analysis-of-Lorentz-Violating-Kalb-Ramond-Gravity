"""Iterative relaxation solver for the coupled (Phi, b_{01}, b_{23}) system.

Implements the scheme described in Sec. 4.2 of manuscript.tex: at each
iteration the KR field components and Phi are updated via under-relaxed
Jacobi sweeps until the maximum relative change of all three fields falls
below the tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kr_fields import update_b01, update_b23
from .matter import gaussian_density
from .operators import compute_laplacian
from .parameters import Parameters
from .potential import solve_phi


@dataclass
class SolverResult:
    r: np.ndarray
    phi: np.ndarray
    b01: np.ndarray
    b23: np.ndarray
    rho: np.ndarray
    iterations: int
    converged: bool
    residuals: dict[str, list[float]] = field(default_factory=dict)


def _initial_fields(r: np.ndarray, params: Parameters, seed: int = 0):
    rng = np.random.default_rng(seed)
    phi = -params.G * params.M_total / np.maximum(r, params.r_min)
    b01 = rng.normal(0.0, 1e-3, size=r.size)
    b23 = rng.normal(0.0, 1e-3, size=r.size)
    # Apply boundary conditions
    phi[0] = phi[1]
    phi[-1] = 0.0
    b01[0] = b01[-1] = 0.0
    b23[0] = b23[-1] = 0.0
    return phi, b01, b23


def solve_coupled_fields(
    params: Parameters | None = None,
    verbose: bool = True,
    seed: int = 0,
) -> SolverResult:
    """Run the coupled relaxation solver and return the converged fields."""
    if params is None:
        params = Parameters()

    r, dr = params.radial_grid()
    rho = gaussian_density(r, params)
    phi, b01, b23 = _initial_fields(r, params, seed=seed)

    hist = {"phi": [], "b01": [], "b23": []}
    converged = False
    it = 0

    for it in range(1, params.max_iters + 1):
        old_phi, old_b01, old_b23 = phi.copy(), b01.copy(), b23.copy()

        lap_phi = compute_laplacian(phi, r, dr)
        b01_new = update_b01(b01, b23, lap_phi, r, dr, params)
        b23_new = update_b23(b01, b23, lap_phi, r, dr, params)
        phi_new = solve_phi(phi, rho, b01, b23, r, dr, params)

        w = params.relax
        b01 = (1.0 - w) * b01 + w * b01_new
        b23 = (1.0 - w) * b23 + w * b23_new
        phi = (1.0 - w) * phi + w * phi_new

        # Bail out if the solution diverges.
        if not (np.all(np.isfinite(phi)) and np.all(np.isfinite(b01)) and np.all(np.isfinite(b23))):
            if verbose:
                print(f"[!] Divergence detected at iteration {it}. Aborting.")
            break

        rel_phi = np.max(np.abs(phi - old_phi)) / (np.max(np.abs(phi)) + 1e-10)
        rel_b01 = np.max(np.abs(b01 - old_b01)) / (np.max(np.abs(b01)) + 1e-10)
        rel_b23 = np.max(np.abs(b23 - old_b23)) / (np.max(np.abs(b23)) + 1e-10)
        hist["phi"].append(rel_phi)
        hist["b01"].append(rel_b01)
        hist["b23"].append(rel_b23)

        if (
            it >= params.min_iters
            and rel_phi < params.tol
            and rel_b01 < params.tol
            and rel_b23 < params.tol
        ):
            converged = True
            if verbose:
                print(f"[+] Converged after {it} iterations.")
            break

        if verbose and it % 1000 == 0:
            print(
                f"iter {it:6d} | rel_phi={rel_phi:.2e} "
                f"rel_b01={rel_b01:.2e} rel_b23={rel_b23:.2e}"
            )

    if not converged and verbose:
        print(f"[!] Maximum iterations ({params.max_iters}) reached without convergence.")

    return SolverResult(
        r=r, phi=phi, b01=b01, b23=b23, rho=rho,
        iterations=it, converged=converged, residuals=hist,
    )
