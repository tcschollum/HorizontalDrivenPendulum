""" Containing class defining the driven pendulum and its behaviour.

Model: a simple pendulum of length L and mass m whose pivot is driven
horizontally according to x_p(t) = A * sin(w * t).

Let theta be the angle of the rod measured from the downward vertical.
The mass position is:
    x(t) = x_p(t) + L * sin(theta)
    y(t) =        - L * cos(theta)

Let "_dot" refer to the first derivative,a nd "_ddot" refer to the second derivative.

With a linear damping term b * theta_dot, the equation of motion is:

    theta_ddot = -(g / L) * sin(theta)
                 - b * theta_dot
                 + (A * w**2 / L) * sin(w * t) * cos(theta)

                 (note this is just a re-arranged version of what is in the readme)

The driving term comes from the pivot's acceleration x_p_ddot = -A*w**2*sin(w*t)
entering the rotating frame as -(x_p_ddot / L) * cos(theta).

"""

import math


class DrivenPendulum:
    # Class attributes
    g = 9.81  # m s^-2, acceleration of gravity on Earth
    b = 0.5   # s^-1,   damping coefficient

    # Constructor to initialise parameters for this instance
    def __init__(self, amplitude, frequency, length, mass):
        self.A = amplitude   # m,     drive amplitude of the pivot
        self.w = frequency   # rad/s, drive angular frequency
        self.L = length      # m,     pendulum length
        self.m = mass        # kg,    mass at end of pendulum

    # ------------------------------------------------------------------ #
    # Pivot motion
    # ------------------------------------------------------------------ #
    def get_pivot_position(self, t):
        """Horizontal position of the driven pivot at time t."""
        return self.A * math.sin(self.w * t)

    ''' note derived attributes below; x_p' and x_p" (x_p_dot, and x_p_ddot)'''
    def get_pivot_velocity(self, t):
        """Horizontal velocity of the driven pivot at time t."""
        return self.A * self.w * math.cos(self.w * t)

    def get_pivot_acceleration(self, t):
        """Horizontal acceleration of the driven pivot at time t."""
        return -self.A * self.w ** 2 * math.sin(self.w * t)

    # ------------------------------------------------------------------ #
    # Equation of Motion, EoM (exact, non-linear ODE right-hand side)
    # ------------------------------------------------------------------ #
    def angular_acceleration(self, theta, theta_dot, t):
        """Angular acceleration (theta_ddot) for the full non-linear model.

        Parameters
        ----------
        theta :     float   current angle from the downward vertical [rad]
        theta_dot : float   current angular speed (theta_dot)        [rad/s]
        t     :     float   time                                     [s]
        """
        # calculate each component of the EoM
        natural = -(self.g / self.L) * math.sin(theta)
        damping = -self.b * theta_dot
        horizontal_drive = (self.A * self.w ** 2 / self.L) * math.sin(self.w * t) * math.cos(theta)

        return natural + damping + horizontal_drive

    ''' note derived attribute below '''
    @property
    def natural_frequency(self):
        """Undriven small-oscillation angular frequency w0 = sqrt(g / L)."""
        return math.sqrt(self.g / self.L)

    # ------------------------------------------------------------------ #
    # Small-angle approximation analytic solution

    # Note - derived by Claude Code. I understand the concept of small angle 
    #        approximation & mostly follow the maths
    # ------------------------------------------------------------------ #
    def small_angle_state(self, t):
        """Steady-state (theta, theta_dot) under the small-angle approximation.

        For small theta, sin(theta) ~ theta and cos(theta) ~ 1, so the EoM
        becomes a linearly driven, damped harmonic oscillator:

            theta_ddot + b*theta_dot + w0**2 * theta = F * sin(w*t)

        with w0**2 = g/L and F = A*w**2/L. The long-time (transient-decayed)
        particular solution is:

            theta(t) = Theta * sin(w*t - phi)

        Returns the steady-state angle and angular speed at time t.

        """
        w0_sq = self.g / self.L
        F = self.A * self.w ** 2 / self.L

        denom = math.sqrt((w0_sq - self.w ** 2) ** 2 + (self.b * self.w) ** 2)
        amplitude = F / denom
        phase = math.atan2(self.b * self.w, w0_sq - self.w ** 2)

        theta = amplitude * math.sin(self.w * t - phase)
        theta_dot = amplitude * self.w * math.cos(self.w * t - phase)
    
        return theta, theta_dot

    # ------------------------------------------------------------------ #
    # Numerical integration (4th-order Runge-Kutta) of the full model

    # Note - this RK4 method is coded by Claude Code
    # ------------------------------------------------------------------ #
    def step(self, theta, theta_dot, t, dt):
        """Advance the state (theta, theta_dot) by one RK4 step of size dt."""
        def deriv(th, th_d, tt):
            return th_d, self.angular_acceleration(th, th_d, tt)

        k1_th, k1_om = deriv(theta, theta_dot, t)
        k2_th, k2_om = deriv(theta + 0.5 * dt * k1_th,
                             theta_dot + 0.5 * dt * k1_om, t + 0.5 * dt)
        k3_th, k3_om = deriv(theta + 0.5 * dt * k2_th,
                             theta_dot + 0.5 * dt * k2_om, t + 0.5 * dt)
        k4_th, k4_om = deriv(theta + dt * k3_th,
                             theta_dot + dt * k3_om, t + dt)

        theta_next = theta + (dt / 6.0) * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
        theta_dot_next = theta_dot + (dt / 6.0) * (k1_om + 2 * k2_om + 2 * k3_om + k4_om)
    
        return theta_next, theta_dot_next

    def simulate(self, theta0, theta_dot0, t_end, dt=1e-3, t0=0.0):
        """Integrate the full non-linear EoM from t0 to t_end.

        Parameters
        ----------
        theta0, theta_dot0  : initial angle [rad] and angular speed [rad/s]
        t_end               : final time [s]
        dt                  : time step [s]
        t0                  : start time [s]

        Returns
        -------
        (times, thetas, theta_dots) : three lists sampled at each step.
        """
        times = [t0]
        thetas = [theta0]
        theta_dots = [theta_dot0]

        t, theta, theta_dot = t0, theta0, theta_dot0
        n_steps = int(round((t_end - t0) / dt))
        for _ in range(n_steps):
            theta, theta_dot = self.step(theta, theta_dot, t, dt)
            t += dt
            times.append(t)
            thetas.append(theta)
            theta_dots.append(theta_dot)

        return times, thetas, theta_dots




if __name__ == "__main__":
    # Quick sanity check / demo
    pendulum = DrivenPendulum(amplitude=0.05, frequency=4.0, length=1.0, mass=1.0)

    print(f"natural frequency w0 = {pendulum.natural_frequency:.3f} rad/s")

    t = 1.0
    print(f"pivot x_p(t={t})       = {pendulum.get_pivot_position(t):.4f} m")

    theta_lin, omega_lin = pendulum.small_angle_state(t)
    print(f"small-angle theta(t)   = {theta_lin:.4f} rad")

    times, thetas, omegas = pendulum.simulate(theta0=0.0, theta_dot0=0.0, t_end=10.0, dt=1e-3)
    print(f"numerical theta(t={times[-1]:.1f}) = {thetas[-1]:.4f} rad "
          f"(over {len(times)} samples)")
