"""Finite-difference operators on the uniform radial grid.

The spherically symmetric Laplacian is discretised with the standard
second-order stencil plus the 2/r first-derivative correction; regularity
at the innermost node is enforced by f[0] = f[1].
"""

from __future__ import annotations

import numpy as np


def compute_laplacian(f: np.ndarray, r: np.ndarray, dr: float) -> np.ndarray:
    """Second-order radial Laplacian in spherical symmetry."""
    N = len(f)
    lap = np.zeros(N)
    for i in range(1, N - 1):
        d2f = (f[i + 1] - 2.0 * f[i] + f[i - 1]) / dr ** 2
        df = (f[i + 1] - f[i - 1]) / (2.0 * dr)
        if r[i] > 1e-6:
            lap[i] = d2f + (2.0 / r[i]) * df
        else:
            lap[i] = d2f
    lap[0] = lap[1]
    lap[-1] = lap[-2]
    return lap


def first_derivative(f: np.ndarray, dr: float) -> np.ndarray:
    """Central-difference first derivative with one-sided ends."""
    df = np.zeros_like(f)
    df[1:-1] = (f[2:] - f[:-2]) / (2.0 * dr)
    df[0] = (f[1] - f[0]) / dr
    df[-1] = (f[-1] - f[-2]) / dr
    return df
