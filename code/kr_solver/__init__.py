"""Kalb--Ramond post-Newtonian gravity solver.

Modules:
    parameters : physical and numerical parameters (Parameters dataclass)
    matter     : Gaussian double source for the wide binary
    operators  : finite-difference Laplacian and derivatives
    kr_fields  : update rules for b_{01}(r) and b_{23}(r)
    potential  : Jacobi update for Phi(r) from the modified Poisson eq.
    solver     : coupled iterative relaxation loop
    yukawa     : Yukawa-form fit of the numerical potential
    plotting   : matplotlib plots of the results
"""

from .parameters import Parameters
from .solver import solve_coupled_fields, SolverResult
from .yukawa import fit_yukawa, YukawaFit, yukawa_potential

__all__ = [
    "Parameters",
    "SolverResult",
    "solve_coupled_fields",
    "fit_yukawa",
    "YukawaFit",
    "yukawa_potential",
]
