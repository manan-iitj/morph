"""
Wheel Control Module for DHAV1W.

Independent wheel control module supporting velocity, torque, and position
control modes. Separate from CPG and foot trajectory generator.

Wheel specifications (from URDF/MJCF):
    - Joint type: continuous
    - Axis: (0, 1, 0) — wheel rolls in XZ plane
    - Radius: 0.077 m
    - Length: 0.052 m
    - Max effort (torque): 14 N·m
    - Max velocity: 73 rad/s (5.61 m/s at r=0.0769 m)
    - Motor effort range: [-14, 14] N·m
"""

import numpy as np


class WheelController:
    """
    Independent wheel control module for DHAV1W.

    Provides three control modes:
        1. Velocity mode: command desired angular velocity
        2. Torque mode: command desired torque
        3. Position mode: hold wheel at fixed angle

    Parameters
    ----------
    wheel_radius : float
        Wheel radius in meters (default 0.077 m).
    max_velocity : float
        Maximum wheel angular velocity in rad/s.
    max_torque : float
        Maximum wheel torque in N·m.
    control_mode : str
        'velocity', 'torque', or 'position'.
    """

    def __init__(self, wheel_radius=0.077,
                 max_velocity=73.0,
                 max_torque=14.0,
                 control_mode='velocity'):
        self.wheel_radius = wheel_radius
        self.max_velocity = max_velocity
        self.max_torque = max_torque
        self.control_mode = control_mode

        # Current wheel states
        self.wheel_angles = {leg: 0.0 for leg in ['FL', 'FR', 'RL', 'RR']}
        self.wheel_velocities = {leg: 0.0 for leg in ['FL', 'FR', 'RL', 'RR']}

        # Commands
        self.velocity_commands = {leg: 0.0 for leg in ['FL', 'FR', 'RL', 'RR']}
        self.torque_commands = {leg: 0.0 for leg in ['FL', 'FR', 'RL', 'RR']}
        self.position_commands = {leg: 0.0 for leg in ['FL', 'FR', 'RL', 'RR']}

        # PD gains for position mode (holding wheel stationary)
        self.kp_position = 100.0   # N·m/rad
        self.kd_position = 10.0    # N·m·s/rad

        # Velocity tracking gain
        self.kp_velocity = 5.0  # N·m/(rad/s)

        # History for plotting
        self.history = {leg: {'t': [], 'angle': [], 'velocity': [],
                              'cmd_velocity': [], 'cmd_torque': []}
                        for leg in ['FL', 'FR', 'RL', 'RR']}

    def set_velocity(self, leg, velocity_rad_s):
        """Set wheel angular velocity command (rad/s)."""
        velocity_rad_s = np.clip(velocity_rad_s, -self.max_velocity, self.max_velocity)
        self.velocity_commands[leg] = velocity_rad_s
        if self.control_mode == 'velocity':
            self.torque_commands[leg] = 0.0
        elif self.control_mode == 'torque':
            self.position_commands[leg] = 0.0

    def set_velocities(self, velocities):
        """Set wheel velocities for all legs. dict: {leg: rad/s}."""
        for leg, vel in velocities.items():
            self.set_velocity(leg, vel)

    def set_torque(self, leg, torque_nm):
        """Set wheel torque command (N·m)."""
        torque_nm = np.clip(torque_nm, -self.max_torque, self.max_torque)
        self.torque_commands[leg] = torque_nm
        if self.control_mode == 'torque':
            self.velocity_commands[leg] = 0.0
        elif self.control_mode == 'position':
            self.position_commands[leg] = 0.0

    def set_torques(self, torques):
        """Set wheel torques for all legs. dict: {leg: N·m}."""
        for leg, trq in torques.items():
            self.set_torque(leg, trq)

    def set_position(self, leg, angle_rad):
        """Set wheel position command (radians) for position hold mode."""
        self.position_commands[leg] = angle_rad
        if self.control_mode == 'position':
            self.velocity_commands[leg] = 0.0
            self.torque_commands[leg] = 0.0

    def set_all_velocities(self, velocity_rad_s):
        """Set all wheels to the same velocity (forward/backward)."""
        for leg in ['FL', 'FR', 'RL', 'RR']:
            self.set_velocity(leg, velocity_rad_s)

    def set_all_torques(self, torque_nm):
        """Set all wheels to the same torque."""
        for leg in ['FL', 'FR', 'RL', 'RR']:
            self.set_torque(leg, torque_nm)

    def stop_all(self):
        """Stop all wheels (zero velocity command)."""
        for leg in ['FL', 'FR', 'RL', 'RR']:
            self.set_velocity(leg, 0.0)
            self.set_torque(leg, 0.0)

    def get_control_signal(self, leg, current_angle, current_velocity, dt=0.001):
        """
        Compute the control signal (torque) for a wheel based on current mode.

        Parameters
        ----------
        leg : str
            Wheel identifier.
        current_angle : float
            Current wheel angle (radians).
        current_velocity : float
            Current wheel angular velocity (rad/s).
        dt : float
            Time step (s).

        Returns
        -------
        torque : float
            Torque command to apply (N·m).
        """
        if self.control_mode == 'velocity':
            vel_error = self.velocity_commands[leg] - current_velocity
            torque = self.kp_velocity * vel_error
            torque = np.clip(torque, -self.max_torque, self.max_torque)

        elif self.control_mode == 'torque':
            torque = self.torque_commands[leg]

        elif self.control_mode == 'position':
            pos_error = self.position_commands[leg] - current_angle
            vel_error = 0.0 - current_velocity
            torque = (self.kp_position * pos_error +
                      self.kd_position * vel_error)
            torque = np.clip(torque, -self.max_torque, self.max_torque)

        else:
            torque = 0.0

        return torque

    def update_state(self, leg, angle, velocity, torque_applied, dt=0.001):
        """Record wheel state for logging/plotting."""
        self.wheel_angles[leg] = angle
        self.wheel_velocities[leg] = velocity

        t_prev = self.history[leg]['t'][-1] if self.history[leg]['t'] else 0.0
        self.history[leg]['t'].append(t_prev + dt)
        self.history[leg]['angle'].append(angle)
        self.history[leg]['velocity'].append(velocity)
        self.history[leg]['cmd_velocity'].append(self.velocity_commands[leg])
        self.history[leg]['cmd_torque'].append(self.torque_commands[leg])

    def velocity_to_linear_speed(self, angular_velocity_rad_s):
        """Convert wheel angular velocity to linear speed (m/s)."""
        return angular_velocity_rad_s * self.wheel_radius

    def linear_speed_to_velocity(self, linear_speed_m_s):
        """Convert linear speed to wheel angular velocity (rad/s)."""
        return linear_speed_m_s / self.wheel_radius

    def compute_turning_velocities(self, forward_vel_m_s, yaw_rate_rad_s,
                                   track_width=0.107):
        """
        Compute individual wheel velocities for forward motion + turning.

        Uses differential drive kinematics:
            v_left  = v_forward + yaw_rate * track_width / 2
            v_right = v_forward - yaw_rate * track_width / 2

        Parameters
        ----------
        forward_vel_m_s : float
            Desired forward linear velocity (m/s).
        yaw_rate_rad_s : float
            Desired yaw rate (rad/s). Positive = turn left.
        track_width : float
            Distance between left and right wheels (m).
            DHAV1W hip separation: 2 * 0.0535 = 0.107 m.

        Returns
        -------
        velocities : dict
            {leg: angular_velocity_rad_s}
        """
        v_forward_ang = self.linear_speed_to_velocity(forward_vel_m_s)
        v_turn_ang = self.linear_speed_to_velocity(
            yaw_rate_rad_s * track_width / 2.0
        )

        velocities = {
            'FL': v_forward_ang + v_turn_ang,  # Left front
            'RL': v_forward_ang + v_turn_ang,  # Left rear
            'FR': v_forward_ang - v_turn_ang,  # Right front
            'RR': v_forward_ang - v_turn_ang,  # Right rear
        }

        for leg in ['FL', 'FR', 'RL', 'RR']:
            velocities[leg] = np.clip(
                velocities[leg], -self.max_velocity, self.max_velocity
            )

        return velocities

    def reset(self):
        """Reset all wheel states and commands."""
        for leg in ['FL', 'FR', 'RL', 'RR']:
            self.wheel_angles[leg] = 0.0
            self.wheel_velocities[leg] = 0.0
            self.velocity_commands[leg] = 0.0
            self.torque_commands[leg] = 0.0
            self.position_commands[leg] = 0.0
            self.history[leg] = {'t': [], 'angle': [], 'velocity': [],
                                 'cmd_velocity': [], 'cmd_torque': []}