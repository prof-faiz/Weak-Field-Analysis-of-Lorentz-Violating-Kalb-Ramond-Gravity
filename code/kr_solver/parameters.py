"""Physical and numerical parameters for the Kalb--Ramond gravity solver.

All values follow the fiducial choices used in the thesis (Section 3.3 of
manuscript.tex): geometrised units with M_total = 2, distances in AU, and a
radial grid from r_min to r_max with N points.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Parameters:
    # ---- Physical constants (geometrised units) ----
    G: float = 1.0
    M_total: float = 2.0

    # ---- Kalb--Ramond field parameters ----
    m: float = 1e-15          # KR mass (Planck units)
    lam: float = 1.0          # self-coupling lambda
    xi: float = 1e-2          # curvature coupling xi (= xi_2 in the paper)
    xi3: float = 1.0          # secondary curvature coupling xi_3
    a: float = 1e-2           # Lorentz-violating VEV: beta_{01} = -a
    b: float = 1e-2           # Lorentz-violating VEV: beta_{23} =  b
    mu: float = 1e-15         # potential mass parameter
    Lambda: float = 0.0       # quartic coupling in V(X)

    # ---- Binary system (Gaussian double source) ----
    r1: float = 1000.0        # star 1 position (AU)
    r2: float = 3000.0        # star 2 position (AU)
    sigma: float = 50.0       # Gaussian width (AU)

    # ---- Numerical grid ----
    r_min: float = 1e-3       # inner radius (AU)
    r_max: float = 5000.0     # outer radius (AU)
    N: int = 1000             # number of radial nodes

    # ---- Relaxation solver controls ----
    relax: float = 1e-3       # under-relaxation weight
    step: float = 1e-4        # residual step size for KR field updates
    tol: float = 1e-5         # convergence tolerance
    min_iters: int = 1000     # minimum iterations before convergence check
    max_iters: int = 100000   # hard iteration limit
    clip: float = 1e3         # amplitude clip to prevent runaway

    # ---- Split masses (derived) ----
    M1: float = field(init=False)
    M2: float = field(init=False)

    def __post_init__(self) -> None:
        self.M1 = 0.5 * self.M_total
        self.M2 = 0.5 * self.M_total

    def radial_grid(self) -> tuple[np.ndarray, float]:
        r = np.linspace(self.r_min, self.r_max, self.N)
        dr = r[1] - r[0]
        return r, dr
