# Kalb--Ramond post-Newtonian gravity solver

Numerical code accompanying *Post-Newtonian Analysis of Lorentz-Violating
Kalb--Ramond Gravity* (see `../manuscript.tex`).

## Layout

```
code/
├── kr_solver/            # importable package
│   ├── __init__.py
│   ├── parameters.py     # Parameters dataclass (fiducial values)
│   ├── matter.py         # double-Gaussian wide-binary source
│   ├── operators.py      # spherical Laplacian, finite differences
│   ├── kr_fields.py      # update_b01, update_b23
│   ├── potential.py      # solve_phi (modified Poisson equation)
│   ├── solver.py         # coupled iterative relaxation loop
│   ├── yukawa.py         # Yukawa-form nonlinear fit
│   └── plotting.py       # matplotlib figures
├── run_simulation.py     # end-to-end driver script
├── requirements.txt
└── README.md
```

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
cd code
python run_simulation.py
```

Outputs go to `code/output/` and include the potential, KR field
profiles, convergence history, and the Yukawa fit.

## Reproducing the paper values

The default `Parameters()` match Table I of the manuscript. Overriding
any subset works as expected:

```python
from kr_solver import Parameters, solve_coupled_fields, fit_yukawa

params = Parameters(xi=5e-3, a=2e-2)
result = solve_coupled_fields(params)
fit = fit_yukawa(result.r, result.phi)
print(fit.A, fit.mu, fit.lambda_eff)
```
