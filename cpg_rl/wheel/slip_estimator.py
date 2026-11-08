"""
Wheel Slip Estimator for DHAV1W.

Estimates wheel slip by comparing wheel angular velocity with the robot's
actual ground-relative linear velocity.

Slip ratio:
    slip = (omega * r - v_ground) / (omega * r)

where:
    omega : wheel angular velocity (rad/s)
    r     : wheel radius (m)
    v_ground : robot ground-relative linear velocity (m/s)

Positive slip = wheel spinning faster than ground speed (acceleration)
Negative slip = wheel slower than ground speed (braking)
"""

import numpy as np


class SlipEstimator:
    """
    Estimates wheel slip for DHAV1W wheeled-legged quadruped.

    Parameters
    ----------
    wheel_radius : float
        Wheel radius in meters (default 0.077 m).
    """

    def __init__(self, wheel_radius=0.077):
        self.wheel_radius = wheel_radius

        # History for plotting
        self.history = {leg: {'t': [], 'slip': [], 'omega': [], 'v_ground': []}
                        for leg in ['FL', 'FR', 'RL', 'RR']}

    def compute_slip(self, leg, omega_rad_s, v_ground_m_s, dt=0.001):
        """
        Compute wheel slip ratio.

        Parameters
        ----------
        leg : str
            Wheel identifier: 'FL', 'FR', 'RL', 'RR'.
        omega_rad_s : float
            Wheel angular velocity (rad/s).
        v_ground_m_s : float
            Robot ground-relative linear velocity at wheel contact (m/s).
        dt : float
            Time step (s).

        Returns
        -------
        slip : float
            Slip ratio. Positive = wheel spinning faster than ground speed.
            Negative = wheel slower than ground speed.
        """
        v_wheel = omega_rad_s * self.wheel_radius

        if abs(v_wheel) < 1e-6:
            slip = 0.0
        else:
            slip = (v_wheel - v_ground_m_s) / abs(v_wheel)

        # Clamp to reasonable range
        slip = np.clip(slip, -2.0, 2.0)

        # Record history
        t_prev = self.history[leg]['t'][-1] if self.history[leg]['t'] else 0.0
        self.history[leg]['t'].append(t_prev + dt)
        self.history[leg]['slip'].append(slip)
        self.history[leg]['omega'].append(omega_rad_s)
        self.history[leg]['v_ground'].append(v_ground_m_s)

        return slip

    def get_slip(self, leg):
        """Get latest slip value for a leg."""
        if self.history[leg]['slip']:
            return self.history[leg]['slip'][-1]
        return 0.0

    def get_average_slip(self, leg, window=10):
        """Get average slip over a window of recent samples."""
        slips = self.history[leg]['slip'][-window:]
        if slips:
            return np.mean(slips)
        return 0.0

    def is_slipping(self, leg, threshold=0.3):
        """Check if wheel is slipping excessively."""
        return abs(self.get_slip(leg)) > threshold

    def get_all_slips(self):
        """Get slip values for all wheels."""
        return {leg: self.get_slip(leg) for leg in ['FL', 'FR', 'RL', 'RR']}

    def reset(self):
        """Reset all slip estimator history."""
        for leg in ['FL', 'FR', 'RL', 'RR']:
            self.history[leg] = {'t': [], 'slip': [], 'omega': [], 'v_ground': []}