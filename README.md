## Driven Pendulum Analysis

Author: Toni Schollum


### Context

A pendulum of length L and mass m hangs from a pivot that is driven horizontally, so the pivot position is x_p(t) = A·sin(ωt). With gravity g, linear damping coefficient b, and θ measured from the downward vertical, the equation of motion is:

>
> θ̈  + b·θ̇ + (g/L)·sin θ = (A·ω²/L)·sin(ωt)·cos θ
>

Using baseline values:
> L = 1.0 m
> g = 9.81 m/s²
> m = 1.0 kg
> b = 0.5 s⁻¹

And exploring the drive amplitude (A) and the frequency (ω).


### Project layout

| File | Purpose |
| --- | --- |
| [driven_pendulum.py](driven_pendulum.py) | `DrivenPendulum` class — the model, an RK4 integrator, and the small-angle analytic solution. |
| [analysis_plotting_helpers.py](analysis_plotting_helpers.py) | Plotting helpers that run a simulation and save a figure (pivot motion, full dynamics, parameter sweeps, RK4 vs small-angle). |
| [main.py](main.py) | Defines the parameter search space and drives the helpers to produce the full set of figures. |

Each file is documented with module and method docstrings — read those for the detail.


### Setup (uv)

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management. With uv installed, sync the environment from the lockfile:

```bash
uv sync
```

That creates a `.venv` with the pinned dependencies (Python ≥ 3.12, matplotlib).


### Running the analysis

```bash
uv run main.py
```

This runs all the simulations and writes the figures to a `figures/` directory (created if it doesn't exist). Progress is printed to the console as each figure is saved.
