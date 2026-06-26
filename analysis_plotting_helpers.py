"""Helper methods to plot and analyse driven-pendulum behaviour.

Each function takes one DrivenPendulum (or a list of them), runs the
relevant simulation, and writes a PNG into FIGURE_DIR. A non-interactive
matplotlib backend is used so the script runs headless and just saves files.

Conventions
-----------
* "RK4"        : the full non-linear model integrated by DrivenPendulum.step
* "small-angle": the analytic steady-state from DrivenPendulum.small_angle_state
* All angles in rad, angular speeds in rad/s, angular accelerations in rad/s^2.
"""

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: render straight to file
import matplotlib.pyplot as plt
import numpy as np

from driven_pendulum import DrivenPendulum


FIGURE_DIR = "figures"

# Caption stamped on every figure: names the model and the fixed baselines.
# g and b are read from the class so the caption tracks any change there.
MODEL_CAPTION = (
    "Model: horizontally driven, damped pendulum  "
    f"(g={DrivenPendulum.g:g} m/s$^2$, b={DrivenPendulum.b:g} s$^{{-1}}$)"
)


# ---------------------------------------------------------------------- #
# Small internal helpers
# ---------------------------------------------------------------------- #
def _save(fig, filename):
    """Save a figure into FIGURE_DIR and close it. Returns the path."""
    # Caption below the figure; bbox_inches='tight' grows the canvas to fit it.
    fig.text(0.5, -0.01, MODEL_CAPTION, ha="center", va="top",
             fontsize="small", style="italic", color="gray")
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")
    return path


def _label(p):
    """Short legend label describing a pendulum's drive/length parameters."""
    return f"A={p.A:g} m, w={p.w:g} rad/s, L={p.L:g} m"


def _legend_outside(ax):
    """Place the legend just outside the axes (top-right) so it never sits on
    top of the data. _save uses bbox_inches='tight', so the saved figure grows
    to include the legend rather than clipping it."""
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")


def _slug(p):
    """Filename-safe parameter string, e.g. 'A0p1_w4_L1'."""
    fmt = lambda v: f"{v:g}".replace(".", "p")
    return f"A{fmt(p.A)}_w{fmt(p.w)}_L{fmt(p.L)}"


def simulate_run(pendulum, theta0=0.0, theta_dot0=0.0, t_end=30.0, dt=1e-3):
    """Run the RK4 model and return (times, thetas, theta_dots, theta_ddots).

    Thin wrapper around DrivenPendulum.simulate that also evaluates the
    angular-acceleration series, so callers get all four time series at once.
    """
    times, thetas, theta_dots = pendulum.simulate(
        theta0, theta_dot0, t_end=t_end, dt=dt
    )
    theta_ddots = pendulum.angular_acceleration_series(times, thetas, theta_dots)

    return times, thetas, theta_dots, theta_ddots


# ---------------------------------------------------------------------- #
# 1. Pivot motion over time
# ---------------------------------------------------------------------- #
def plot_pivot_position(pendulums, t_end=10.0, n=2000, filename="pivot_position.png"):
    """Overlay the horizontal pivot position x_p(t) = A*sin(w*t).

    The pivot motion is a property of the drive alone (A and w), independent
    of the pendulum's response, so this is a quick way to see the forcing that
    every run shares.

    Parameters
    ----------
    pendulums : iterable of DrivenPendulum
    t_end     : final time [s]
    n         : number of sample points
    """
    times = np.linspace(0.0, t_end, n)
    fig, ax = plt.subplots(figsize=(10, 5))

    for p in pendulums:
        x_p = [p.get_pivot_position(t) for t in times]
        ax.plot(times, x_p, label=_label(p))

    ax.set_title("Pivot position over time")
    ax.set_xlabel("time t [s]")
    ax.set_ylabel("pivot position $x_p$ [m]")
    ax.grid(True, alpha=0.3)
    _legend_outside(ax)
    return _save(fig, filename)


# ---------------------------------------------------------------------- #
# 2. Full dynamics of a single pendulum: angle, speed, acceleration
# ---------------------------------------------------------------------- #
def plot_dynamics(pendulum, theta0=0.0, theta_dot0=0.0, t_end=30.0, dt=1e-3,
                  filename=None):
    """Plot theta, theta_dot and theta_ddot over time for one RK4 run.

    Produces a 3-row figure (shared time axis) for a single pendulum, which is
    the most detailed view of how that particular drive settles into its
    steady oscillation.
    """
    times, thetas, theta_dots, theta_ddots = simulate_run(
        pendulum, theta0, theta_dot0, t_end, dt
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(f"Pendulum dynamics  ({_label(pendulum)})")

    axes[0].plot(times, thetas, color="tab:blue")
    axes[0].set_ylabel(r"angle $\theta$ [rad]")

    axes[1].plot(times, theta_dots, color="tab:orange")
    axes[1].set_ylabel(r"angular speed $\dot\theta$ [rad/s]")

    axes[2].plot(times, theta_ddots, color="tab:green")
    axes[2].set_ylabel(r"angular accel. $\ddot\theta$ [rad/s$^2$]")
    axes[2].set_xlabel("time t [s]")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    if filename is None:
        filename = f"dynamics_{_slug(pendulum)}.png"
    return _save(fig, filename)


# ---------------------------------------------------------------------- #
# 3. Sweep: overlay one quantity for several pendulums
# ---------------------------------------------------------------------- #
def plot_dynamics_sweep(pendulums, title, filename, theta0=0.0, theta_dot0=0.0,
                        t_end=30.0, dt=1e-3):
    """Overlay theta, theta_dot and theta_ddot for a list of pendulums.

    Intended for parameter sweeps: hold all parameters fixed except one (e.g.
    vary A, or vary w to see resonance) and pass the resulting list so the
    responses are compared on shared axes.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(title)

    for p in pendulums:
        times, thetas, theta_dots, theta_ddots = simulate_run(
            p, theta0, theta_dot0, t_end, dt
        )
        label = _label(p)
        axes[0].plot(times, thetas, label=label)
        axes[1].plot(times, theta_dots, label=label)
        axes[2].plot(times, theta_ddots, label=label)

    axes[0].set_ylabel(r"angle $\theta$ [rad]")
    axes[1].set_ylabel(r"angular speed $\dot\theta$ [rad/s]")
    axes[2].set_ylabel(r"angular accel. $\ddot\theta$ [rad/s$^2$]")
    axes[2].set_xlabel("time t [s]")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    _legend_outside(axes[0])
    return _save(fig, filename)


# ---------------------------------------------------------------------- #
# 4. Method comparison: RK4 vs small-angle approximation
# ---------------------------------------------------------------------- #
def plot_method_comparison(pendulum, theta0=0.0, theta_dot0=0.0, t_end=30.0,
                           dt=1e-3, filename=None):
    """Compare the full RK4 solution against the small-angle approximation.

    The small-angle result is the *steady-state* particular solution only, so
    it carries no transient. Expect the RK4 curve to start from theta0 and
    converge onto the small-angle curve once the damped transient has decayed
    (good agreement for small A; visible divergence once theta grows and the
    sin(theta) ~ theta assumption breaks down).
    """
    times, thetas, theta_dots, _ = simulate_run(
        pendulum, theta0, theta_dot0, t_end, dt
    )
    sa = [pendulum.small_angle_state(t) for t in times]
    sa_thetas = [s[0] for s in sa]
    sa_theta_dots = [s[1] for s in sa]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"RK4 vs small-angle approximation  ({_label(pendulum)})")

    axes[0].plot(times, thetas, label="RK4 (full non-linear)", color="tab:blue")
    axes[0].plot(times, sa_thetas, label="small-angle (steady state)",
                 color="tab:red", linestyle="--")
    axes[0].set_ylabel(r"angle $\theta$ [rad]")

    axes[1].plot(times, theta_dots, label="RK4 (full non-linear)",
                 color="tab:blue")
    axes[1].plot(times, sa_theta_dots, label="small-angle (steady state)",
                 color="tab:red", linestyle="--")
    axes[1].set_ylabel(r"angular speed $\dot\theta$ [rad/s]")
    axes[1].set_xlabel("time t [s]")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        _legend_outside(ax)

    if filename is None:
        filename = f"comparison_{_slug(pendulum)}.png"
    return _save(fig, filename)
