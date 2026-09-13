"""Driver script: solve the coupled Phi/b_{01}/b_{23} system and fit Yukawa.

Run from the repository root:

    python -m code.run_simulation

Outputs (in code/output/):
    potential.png     - Phi(r)
    kr_fields.png     - b_{01}(r), b_{23}(r)
    yukawa_fit.png    - numerical Phi vs. Yukawa fit
    convergence.png   - residual history
    fields.npz        - r, phi, b01, b23 arrays
    fit.txt           - Yukawa fit parameters
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kr_solver import Parameters, fit_yukawa, solve_coupled_fields
from kr_solver.plotting import (
    plot_convergence,
    plot_kr_fields,
    plot_potential,
    plot_yukawa_fit,
)


def main() -> None:
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    params = Parameters()
    result = solve_coupled_fields(params, verbose=True)

    plot_potential(result.r, result.phi, out_dir / "potential.png")
    plot_kr_fields(result.r, result.b01, result.b23, out_dir / "kr_fields.png")
    plot_convergence(result.residuals, out_dir / "convergence.png")

    fit = fit_yukawa(result.r, result.phi, r_min_fit=100.0)
    plot_yukawa_fit(result.r, result.phi, fit, out_dir / "yukawa_fit.png")

    np.savez(
        out_dir / "fields.npz",
        r=result.r, phi=result.phi, b01=result.b01, b23=result.b23, rho=result.rho,
    )
    with open(out_dir / "fit.txt", "w") as fh:
        fh.write("Yukawa fit: Phi(r) = -A * exp(-mu r) / r\n")
        fh.write(f"A          = {fit.A:.6f}\n")
        fh.write(f"mu         = {fit.mu:.6e}  [AU^-1]\n")
        fh.write(f"lambda_eff = {fit.lambda_eff:.6e}  [AU]\n")

    print("---")
    print(f"Yukawa fit: A = {fit.A:.4f}, mu = {fit.mu:.4e} AU^-1, "
          f"lambda_eff = {fit.lambda_eff:.4e} AU")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
