"""
Wheel Constraints for DHAV1W.

Enforces safety constraints on wheel control commands:
- Velocity limits
- Torque limits
- Acceleration limits (smooth acceleration/deceleration)
- Emergency reduction during unstable conditions
"""

import numpy as np


class WheelConstraints:
    """
    Enforces safety constraints on wheel control commands.

    Parameters
    ----------
    wheel_radius : float
        Wheel radius in meters.
    max_velocity : float
        Maximum wheel angular velocity in rad/s.
    max_torque : float
        Maximum wheel torque in N·m.
    max_acceleration : float
        Maximum angular acceleration in rad/s².
    """

    def __init__(self, wheel_radius=0.077,
                 max_velocity=73.0,
                 max_torque=14.0,
                 max_acceleration=100.0):
        self.wheel_radius = wheel_radius
        self.max_velocity = max_velocity
        self.max_torque = max_torque
        self.max_acceleration = max_acceleration

        # Previous commands for acceleration limiting
        self.prev_velocity_cmd = {leg: 0.0 for leg in ['FL', 'FR', 'RL', 'RR']}

        # Emergency reduction factor (1.0 = normal, 0.0 = stop)
        self.emergency_reduction = 1.0

    def constrain_velocity(self, leg, velocity_cmd, dt=0.001):
        """
        Apply velocity and acceleration limits to a wheel command.

        Parameters
        ----------
        leg : str
            Wheel identifier.
        velocity_cmd : float
            Requested angular velocity (rad/s).
        dt : float
            Time step (s).

        Returns
        -------
        constrained_vel : float
            Constrained angular velocity (rad/s).
        """
        # Apply velocity limits
        velocity_cmd = np.clip(velocity_cmd, -self.max_velocity, self.max_velocity)

        # Apply acceleration limits
        max_delta = self.max_acceleration * dt
        delta = velocity_cmd - self.prev_velocity_cmd[leg]
        delta = np.clip(delta, -max_delta, max_delta)
        constrained_vel = self.prev_velocity_cmd[leg] + delta

        # Apply emergency reduction
        constrained_vel *= self.emergency_reduction

        # Update previous command
        self.prev_velocity_cmd[leg] = constrained_vel

        return constrained_vel

    def constrain_torque(self, leg, torque_cmd):
        """
        Apply torque limits to a wheel command.

        Parameters
        ----------
        leg : str
            Wheel identifier.
        torque_cmd : float
            Requested torque (N·m).

        Returns
        -------
        constrained_torque : float
            Constrained torque (N·m).
        """
        torque_cmd = np.clip(torque_cmd, -self.max_torque, self.max_torque)
        torque_cmd *= self.emergency_reduction
        return torque_cmd

    def set_emergency_reduction(self, reduction):
        """
        Set emergency reduction factor.

        Parameters
        ----------
        reduction : float
            Reduction factor in [0, 1]. 1.0 = normal operation,
            0.0 = all wheel commands zeroed.
        """
        self.emergency_reduction = np.clip(reduction, 0.0, 1.0)

    def get_emergency_reduction(self):
        """Get current emergency reduction factor."""
        return self.emergency_reduction

    def reset_emergency(self):
        """Reset emergency reduction to normal (1.0)."""
        self.emergency_reduction = 1.0

    def reset(self):
        """Reset all constraints."""
        for leg in ['FL', 'FR', 'RL', 'RR']:
            self.prev_velocity_cmd[leg] = 0.0
        self.reset_emergency()

    def linear_speed_to_velocity(self, linear_speed_m_s):
        """Convert linear speed to wheel angular velocity (rad/s)."""
        return linear_speed_m_s / self.wheel_radius

    def velocity_to_linear_speed(self, angular_velocity_rad_s):
        """Convert wheel angular velocity to linear speed (m/s)."""
        return angular_velocity_rad_s * self.wheel_radius