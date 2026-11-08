"""
Coupled CPG Network - Four-Leg Oscillator Network.

Extends single Hopf oscillators into a four-leg network with configurable
phase offsets. Implements diagonal trot and walk gaits.

Leg naming convention:
    FL: Front Left
    FR: Front Right
    RL: Rear Left
    RR: Rear Right

Phase relationships:
    Trot (diagonal): FL=0, FR=pi, RL=pi, RR=0
    Walk (sequential): FL=0, FR=pi/2, RL=pi, RR=3*pi/2

Each leg has its own independent Hopf oscillator. Phase coordination is
achieved through static phase offsets applied to oscillator states.
"""

import numpy as np
from .hopf_oscillator import HopfOscillator


class CoupledCPG:
    """
    Four-leg CPG network using independent Hopf oscillators with phase offsets.

    Parameters
    ----------
    alpha : float
        Convergence rate for all oscillators.
    mu : float
        Limit-cycle amplitude squared.
    omega : float
        Base angular frequency (rad/s).
    leg_phases : dict
        Phase offset per leg: {'FL': 0.0, 'FR': pi, 'RL': pi, 'RR': 0.0}
    dt : float
        Integration time step.
    """

    def __init__(self, alpha=20.0, mu=1.0, omega=6.28,
                 leg_phases=None, dt=0.001):
        self.alpha = alpha
        self.mu = mu
        self.omega = omega
        self.dt = dt

        if leg_phases is None:
            # Default: diagonal trot
            leg_phases = {'FL': 0.0, 'FR': np.pi, 'RL': np.pi, 'RR': 0.0}

        self.leg_phases = leg_phases
        self.legs = ['FL', 'FR', 'RL', 'RR']

        # Create one oscillator per leg
        self.oscillators = {}
        for leg in self.legs:
            phase_offset = leg_phases[leg]
            # Initialize state on the limit cycle at the phase offset
            x0 = np.sqrt(mu) * np.cos(phase_offset)
            y0 = np.sqrt(mu) * np.sin(phase_offset)
            self.oscillators[leg] = HopfOscillator(
                alpha=alpha, mu=mu, omega=omega,
                x0=x0, y0=y0, dt=dt
            )

    def step(self):
        """Advance all oscillators by one time step."""
        for osc in self.oscillators.values():
            osc.step()

    def run(self, duration):
        """Run the network for a given duration (seconds)."""
        n_steps = int(duration / self.dt)
        for _ in range(n_steps):
            self.step()

    def get_states(self):
        """Return dict of (x, y) for each leg."""
        return {leg: (osc.x, osc.y) for leg, osc in self.oscillators.items()}

    def get_amplitudes(self):
        """Return dict of amplitude r for each leg."""
        return {leg: osc.get_amplitude() for leg, osc in self.oscillators.items()}

    def get_phases(self):
        """Return dict of phase angle for each leg."""
        return {leg: osc.get_phase() for leg, osc in self.oscillators.items()}

    def get_histories(self):
        """Return full history dict for each leg."""
        return {leg: {
            't': osc.t_history,
            'x': osc.x_history,
            'y': osc.y_history,
            'r': osc.r_history,
        } for leg, osc in self.oscillators.items()}

    def reset(self):
        """Reset all oscillators to their initial phase-offset states."""
        for leg, osc in self.oscillators.items():
            phase_offset = self.leg_phases[leg]
            x0 = np.sqrt(self.mu) * np.cos(phase_offset)
            y0 = np.sqrt(self.mu) * np.sin(phase_offset)
            osc.reset(x0=x0, y0=y0)

    def set_omega(self, omega):
        """Update frequency for all oscillators."""
        self.omega = omega
        for osc in self.oscillators.values():
            osc.set_parameters(omega=omega)

    def set_mu(self, mu):
        """Update amplitude parameter for all oscillators."""
        self.mu = max(mu, 0.01)
        for osc in self.oscillators.values():
            osc.set_parameters(mu=self.mu)

    def set_leg_phase(self, leg, phase):
        """Update phase offset for a specific leg."""
        self.leg_phases[leg] = phase


def create_trot_cpg(dt=0.001):
    """Create a CPG network configured for diagonal trot gait."""
    leg_phases = {'FL': 0.0, 'FR': np.pi, 'RL': np.pi, 'RR': 0.0}
    return CoupledCPG(alpha=20.0, mu=1.0, omega=6.28,
                      leg_phases=leg_phases, dt=dt)


def create_walk_cpg(dt=0.001):
    """Create a CPG network configured for walk gait."""
    leg_phases = {'FL': 0.0, 'FR': np.pi/2, 'RL': np.pi, 'RR': 3*np.pi/2}
    return CoupledCPG(alpha=20.0, mu=1.0, omega=6.28,
                      leg_phases=leg_phases, dt=dt)


def create_run_cpg(dt=0.001):
    """Create a CPG network configured for run gait (faster trot)."""
    leg_phases = {'FL': 0.0, 'FR': np.pi, 'RL': np.pi, 'RR': 0.0}
    return CoupledCPG(alpha=20.0, mu=1.0, omega=12.56,
                      leg_phases=leg_phases, dt=dt)


# Alias for compatibility
CPGNetwork = CoupledCPG