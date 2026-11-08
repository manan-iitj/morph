"""
Hopf Oscillator - Single CPG primitive.

Implements the standard Hopf oscillator formulation:

    dx/dt = alpha * (mu - r^2) * x - omega * y
    dy/dt = alpha * (mu - r^2) * y + omega * x

where:
    x, y : oscillator states
    alpha: convergence rate to limit cycle
    mu   : limit-cycle amplitude squared (steady-state r = sqrt(mu))
    omega: angular frequency (rad/s)
    r^2  = x^2 + y^2

Theoretical properties:
    - Steady-state amplitude: r_ss = sqrt(mu)
    - Oscillation frequency: f = omega / (2*pi)

Does NOT depend on MuJoCo. Pure mathematical implementation.
"""

import numpy as np


class HopfOscillator:
    """
    Single Hopf oscillator implementing the standard CPG formulation.

    Parameters
    ----------
    alpha : float
        Convergence rate to the limit cycle. Higher = faster convergence.
    mu : float
        Limit-cycle amplitude squared. Steady-state radius r = sqrt(mu).
    omega : float
        Angular frequency in rad/s. Frequency f = omega / (2*pi).
    x0, y0 : float
        Initial oscillator states.
    dt : float
        Integration time step in seconds.
    """

    def __init__(self, alpha=20.0, mu=1.0, omega=6.28, x0=0.1, y0=0.0, dt=0.001):
        self.alpha = alpha
        self.mu = mu
        self.omega = omega
        self.dt = dt

        # State
        self.x = x0
        self.y = y0

        # History for plotting
        self.t_history = [0.0]
        self.x_history = [x0]
        self.y_history = [y0]
        self.r_history = [np.sqrt(x0**2 + y0**2)]

        self.t = 0.0

    def _derivatives(self, x, y):
        """Compute dx/dt and dy/dt for given states."""
        r_sq = x**2 + y**2
        dx = self.alpha * (self.mu - r_sq) * x - self.omega * y
        dy = self.alpha * (self.mu - r_sq) * y + self.omega * x
        return dx, dy

    def step(self):
        """Advance the oscillator by one time step using RK4 integration."""
        k1x, k1y = self._derivatives(self.x, self.y)
        k2x, k2y = self._derivatives(
            self.x + 0.5 * self.dt * k1x,
            self.y + 0.5 * self.dt * k1y,
        )
        k3x, k3y = self._derivatives(
            self.x + 0.5 * self.dt * k2x,
            self.y + 0.5 * self.dt * k2y,
        )
        k4x, k4y = self._derivatives(
            self.x + self.dt * k3x,
            self.y + self.dt * k3y,
        )

        self.x += (self.dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        self.y += (self.dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
        self.t += self.dt

        # Record history
        self.t_history.append(self.t)
        self.x_history.append(self.x)
        self.y_history.append(self.y)
        self.r_history.append(np.sqrt(self.x**2 + self.y**2))

    def run(self, duration):
        """Run the oscillator for a given duration (seconds)."""
        n_steps = int(duration / self.dt)
        for _ in range(n_steps):
            self.step()

    def get_amplitude(self):
        """Return current amplitude r = sqrt(x^2 + y^2)."""
        return np.sqrt(self.x**2 + self.y**2)

    def get_phase(self):
        """Return current phase angle atan2(y, x)."""
        return np.arctan2(self.y, self.x)

    def reset(self, x0=0.1, y0=0.0):
        """Reset oscillator state."""
        self.x = x0
        self.y = y0
        self.t = 0.0
        self.t_history = [0.0]
        self.x_history = [x0]
        self.y_history = [y0]
        self.r_history = [np.sqrt(x0**2 + y0**2)]

    def set_parameters(self, alpha=None, mu=None, omega=None):
        """Update oscillator parameters."""
        if alpha is not None:
            self.alpha = alpha
        if mu is not None:
            self.mu = max(mu, 0.01)  # Keep positive
        if omega is not None:
            self.omega = omega