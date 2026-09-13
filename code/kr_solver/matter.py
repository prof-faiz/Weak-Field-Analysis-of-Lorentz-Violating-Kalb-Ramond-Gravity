"""Matter density for the wide-binary source.

The two point-like stars of the wide binary are represented by narrow
Gaussians (Eq. rho_gauss of manuscript.tex) so that the source is smooth
enough for finite-difference solvers.
"""

from __future__ import annotations

import numpy as np

from .parameters import Parameters


def gaussian_density(r: np.ndarray, params: Parameters) -> np.ndarray:
    """Return the double-Gaussian radial density used as source term."""
    norm1 = params.M1 / (params.sigma * np.sqrt(2.0 * np.pi))
    norm2 = params.M2 / (params.sigma * np.sqrt(2.0 * np.pi))
    rho1 = norm1 * np.exp(-0.5 * ((r - params.r1) / params.sigma) ** 2)
    rho2 = norm2 * np.exp(-0.5 * ((r - params.r2) / params.sigma) ** 2)
    return rho1 + rho2
