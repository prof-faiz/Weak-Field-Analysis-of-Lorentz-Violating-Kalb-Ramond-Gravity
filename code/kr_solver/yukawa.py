"""Yukawa fit of the numerical gravitational potential.

Implements Eq. (yukawa) of manuscript.tex: Phi_fit(r) = -A * exp(-mu r) / r.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class YukawaFit:
    A: float
    mu: float
    lambda_eff: float
    covariance: np.ndarray


def yukawa_potential(r: np.ndarray, A: float, mu: float) -> np.ndarray:
    return -A * np.exp(-mu * r) / r


def fit_yukawa(
    r: np.ndarray,
    phi: np.ndarray,
    r_min_fit: float = 100.0,
    p0: tuple[float, float] = (2.0, 1e-5),
) -> YukawaFit:
    """Nonlinear least-squares fit of the Yukawa form to Phi(r)."""
    mask = r >= r_min_fit
    popt, pcov = curve_fit(yukawa_potential, r[mask], phi[mask], p0=p0)
    A, mu = popt
    return YukawaFit(A=A, mu=mu, lambda_eff=1.0 / mu, covariance=pcov)
