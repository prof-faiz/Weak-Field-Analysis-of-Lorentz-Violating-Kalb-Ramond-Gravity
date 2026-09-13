"""Update rules for the Kalb--Ramond field components b_{01} and b_{23}.

The equations solved here are Eqs. (b01-eom) and (b23-eom) of
manuscript.tex, obtained by varying the action with respect to B^{mu nu}
in the static spherically symmetric ansatz.
"""

from __future__ import annotations

import numpy as np

from .parameters import Parameters


def _kr_scalar(b01: np.ndarray, b23: np.ndarray, params: Parameters) -> np.ndarray:
    """Return X = B_{mu nu} B^{mu nu} to first order in fluctuations."""
    a, b = params.a, params.b
    return -2.0 * a ** 2 + 2.0 * b ** 2 - 4.0 * a * b01 + 4.0 * b * b23


def update_b01(
    b01: np.ndarray,
    b23: np.ndarray,
    lap_phi: np.ndarray,
    r: np.ndarray,
    dr: float,
    params: Parameters,
) -> np.ndarray:
    a, b = params.a, params.b
    m_sq = params.m ** 2
    lam = params.lam
    xi, xi3 = params.xi, params.xi3
    N = len(b01)
    new = b01.copy()

    for i in range(1, N - 1):
        d2b = (b01[i + 1] - 2.0 * b01[i] + b01[i - 1]) / dr ** 2
        db = (b01[i + 1] - b01[i - 1]) / (2.0 * dr)
        lap = d2b + (2.0 / r[i]) * db if r[i] > 1e-6 else d2b

        S = a * b01[i] + b * b23[i]
        Veff = m_sq - 2.0 * lam * (a ** 2 + b ** 2)
        coupling = lap_phi[i] * (xi * a - 8.0 * xi3 * a)

        rhs = lap + m_sq * b01[i] + a * (Veff + 2.0 * lam * S) + coupling
        new[i] = b01[i] - params.step * rhs * dr ** 2

    new[0] = 0.0
    new[-1] = 0.0
    return np.clip(new, -params.clip, params.clip)


def update_b23(
    b01: np.ndarray,
    b23: np.ndarray,
    lap_phi: np.ndarray,
    r: np.ndarray,
    dr: float,
    params: Parameters,
) -> np.ndarray:
    a, b = params.a, params.b
    m_sq = params.m ** 2
    lam = params.lam
    xi, xi3 = params.xi, params.xi3
    N = len(b23)
    new = b23.copy()

    for i in range(1, N - 1):
        d2b = (b23[i + 1] - 2.0 * b23[i] + b23[i - 1]) / dr ** 2
        db = (b23[i + 1] - b23[i - 1]) / (2.0 * dr)
        lap = d2b + (2.0 / r[i]) * db if r[i] > 1e-6 else d2b

        S = a * b01[i] + b * b23[i]
        Veff = m_sq - 2.0 * lam * (a ** 2 + b ** 2)
        coupling = lap_phi[i] * (xi * b + 4.0 * xi3 * b)

        rhs = lap + m_sq * b23[i] + b * (Veff + 2.0 * lam * S) + coupling
        new[i] = b23[i] - params.step * rhs * dr ** 2

    new[0] = 0.0
    new[-1] = 0.0
    return np.clip(new, -params.clip, params.clip)
