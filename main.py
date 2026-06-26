"""Create driven-pendulum objects and run the behavioural analysis.

Defines the parameter search space at the top, then builds DrivenPendulum
objects and hands them to the plotting helpers. All figures are written to
the /figures directory.

The drive amplitude A and frequency w are the parameters of interest; length
L is also explored, and mass has no effect on the dynamics (it does
not appear in the equation of motion) so it is held fixed.
"""

from driven_pendulum import DrivenPendulum
import analysis_plotting_helpers as helpers

# --------------------------------------------------------------------- #
# Parameter search space
# --------------------------------------------------------------------- #
AMPLITUDES  = [0.05, 0.1, 0.2, 0.5]   # m,     pivot drive amplitude (A)
FREQUENCIES = [1.0, 2.0, 4.0, 8.0]    # rad/s, pivot drive frequency (w)
LENGTHS     = [0.5, 1.0, 2.0]         # m,     pendulum length (L)

MASS = 1.0   # kg, no effect on the dynamics (kept only for the constructor)

# Baseline values used when holding parameters fixed while sweeping another.
A0 = 0.1   # m
W0 = 4.0   # rad/s
L0 = 1.0   # m

# With L0 = 1.0 m the natural frequency is w0 = sqrt(g/L) ~ 3.13 rad/s, so the frequency sweep brackets resonance.

# --------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------- #
THETA0     = 0.0    # rad,   initial angle (hanging straight down)
THETA_DOT0 = 0.0    # rad/s, initially at rest
T_END      = 60.0   # s,     long window so the damped transient -> steady state is clear
DT         = 1e-3   # s,     RK4 time step
# --------------------------------------------------------------------- #

def make_pendulum(A=A0, w=W0, L=L0):
    """Build a DrivenPendulum at the given A/w/L (mass fixed at MASS)."""
    return DrivenPendulum(amplitude=A, frequency=w, length=L, mass=MASS)


def main():
    print("Driven-pendulum analysis -> writing figures to /figures\n")

    # ----------------------------------------------------------------- #
    # Pivot motion (drive only): how A and w shape the forcing.
    # ----------------------------------------------------------------- #
    print("Pivot position:")
    helpers.plot_pivot_position(
        [make_pendulum(A=A, w=W0) for A in AMPLITUDES],
        filename="pivot_position_vary_A.png",
    )
    helpers.plot_pivot_position(
        [make_pendulum(w=w) for w in FREQUENCIES],
        filename="pivot_position_vary_w.png",
    )

    # ----------------------------------------------------------------- #
    # Detailed single-run dynamics at the baseline parameters.
    # ----------------------------------------------------------------- #
    print("Baseline dynamics:")
    baseline = make_pendulum()
    helpers.plot_dynamics(baseline, THETA0, THETA_DOT0, T_END, DT)

    # ----------------------------------------------------------------- #
    # Parameter sweeps: vary one of A / w / L, hold the rest at baseline.
    # ----------------------------------------------------------------- #
    print("Amplitude sweep:")
    helpers.plot_dynamics_sweep(
        [make_pendulum(A=A) for A in AMPLITUDES],
        title=f"Amplitude sweep  (w={W0:g} rad/s, L={L0:g} m)",
        filename="sweep_amplitude.png",
        t_end=T_END, dt=DT,
    )

    print("Frequency sweep (resonance):")
    helpers.plot_dynamics_sweep(
        [make_pendulum(w=w) for w in FREQUENCIES],
        title=f"Frequency sweep  (A={A0:g} m, L={L0:g} m)",
        filename="sweep_frequency.png",
        t_end=T_END, dt=DT,
    )

    print("Length sweep:")
    helpers.plot_dynamics_sweep(
        [make_pendulum(L=L) for L in LENGTHS],
        title=f"Length sweep  (A={A0:g} m, w={W0:g} rad/s)",
        filename="sweep_length.png",
        t_end=T_END, dt=DT,
    )

    # ----------------------------------------------------------------- #
    # RK4 vs small-angle approximation.
    # Small A should agree well; large A should diverge as sin(theta) ~ theta
    # stops holding.
    # ----------------------------------------------------------------- #
    print("Method comparison (small vs large amplitude):")
    helpers.plot_method_comparison(
        make_pendulum(A=min(AMPLITUDES)), THETA0, THETA_DOT0, T_END, DT,
        filename="comparison_small_amplitude.png",
    )
    helpers.plot_method_comparison(
        make_pendulum(A=max(AMPLITUDES)), THETA0, THETA_DOT0, T_END, DT,
        filename="comparison_large_amplitude.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
