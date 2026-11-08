"""
Base controller class for Dhav-Hy locomotion.

Provides shared functionality for all motion controllers:
- Joint address setup
- PD + gravity feedforward torque computation
- Base orientation extraction (roll, pitch, yaw)
- Common constants
"""

import numpy as np
import mujoco as mj

# Leg names
LEGS = ['FL', 'FR', 'RL', 'RR']

# Joint types in order
JOINT_TYPES = ['hip', 'thigh', 'calf']

# Actuator map: leg -> joint_type -> actuator index
ACTUATOR_MAP = {
    'FL': {'hip': 0, 'thigh': 1, 'calf': 2, 'wheel': 3},
    'FR': {'hip': 4, 'thigh': 5, 'calf': 6, 'wheel': 7},
    'RL': {'hip': 8, 'thigh': 9, 'calf': 10, 'wheel': 11},
    'RR': {'hip': 12, 'thigh': 13, 'calf': 14, 'wheel': 15},
}

# Effort limits (from original MJCF)
EFFORT_LIMITS = {
    'hip':   40.0,
    'thigh': 40.0,
    'calf':  80.0,
}

# Wheel effort limit
WHEEL_EFFORT_LIMIT = 14.0

# Wheel radius
WHEEL_RADIUS = 0.077

# Wheel velocity limits
MAX_VELOCITY = 73.0          # rad/s (5.6 m/s at r=0.077m)
MAX_TORQUE = 14.0            # Nm

# Wheel effort limit
WHEEL_EFFORT_LIMIT = 14.0

# Wheel velocity limits
MAX_VELOCITY = 73.0

# Stance foot z-height relative to hip (wheel on ground)
STANCE_FOOT_Z = -0.370

# Stand pose (from MJCF keyframe)
STAND_POSE = {
    'FL': (0.1,  0.70, -1.33),
    'FR': (-0.1, 0.70, -1.33),
    'RL': (0.1,  0.87, -1.33),
    'RR': (-0.1, 0.70, -1.33),
}

# PD gains per joint type: (kp, kd)
# Default gains — individual controllers can override via class attribute
PD_GAINS = {
    'hip':   (300.0, 12.0),
    'thigh': (300.0, 12.0),
    'calf':  (500.0, 20.0),
}

# Track width (hip separation: 2 * 0.0535 m) for differential drive
TRACK_WIDTH = 0.107

# Wheel acceleration limit for ramping (rad/s^2)
WHEEL_ACCEL_LIMIT = 80.0


def compute_differential_wheel_velocities(forward_vel_m_s, yaw_rate_rad_s,
                                          track_width=TRACK_WIDTH,
                                          max_velocity=MAX_VELOCITY):
    """Convert body forward velocity + yaw rate to per-wheel angular velocities.

    Differential drive:
        v_left  = v_forward + yaw_rate * track_width / 2
        v_right = v_forward - yaw_rate * track_width / 2

    Returns
    -------
    velocities : dict
        {leg: angular_velocity_rad_s} for FL, FR, RL, RR.
    """
    v_forward_ang = forward_vel_m_s / WHEEL_RADIUS
    v_turn_ang = (yaw_rate_rad_s * track_width / 2.0) / WHEEL_RADIUS

    velocities = {
        'FL': v_forward_ang + v_turn_ang,
        'RL': v_forward_ang + v_turn_ang,
        'FR': v_forward_ang - v_turn_ang,
        'RR': v_forward_ang - v_turn_ang,
    }

    for leg in velocities:
        velocities[leg] = float(np.clip(velocities[leg],
                                         -max_velocity, max_velocity))
    return velocities


def ramp_velocity(current, target, dt, accel_limit=WHEEL_ACCEL_LIMIT):
    """Slew-limit a single velocity command toward target."""
    diff = target - current
    max_step = accel_limit * dt
    if diff > max_step:
        diff = max_step
    elif diff < -max_step:
        diff = -max_step
    return float(current + diff)


def setup_joint_addresses(model):
    """Map leg/joint-type to qpos and dof addresses using MuJoCo name lookup."""
    qpos_addr = {}
    dof_addr = {}
    for leg in LEGS:
        qpos_addr[leg] = {}
        dof_addr[leg] = {}
        for joint_type in ['hip', 'thigh', 'calf', 'wheel']:
            joint_name = f'{leg}_{joint_type}_joint'
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_addr[leg][joint_type] = int(model.jnt_qposadr[jid])
            dof_addr[leg][joint_type] = int(model.jnt_dofadr[jid])
    return qpos_addr, dof_addr


class BaseMotion:
    """Base class for all motion controllers.

    Provides:
      - Joint address setup
      - PD + gravity feedforward torque computation
      - Base orientation extraction (roll, pitch, yaw)
      - Common constants
    """

    def __init__(self, model, data, qpos_addr, dof_addr):
        self.model = model
        self.data = data
        self.qpos_addr = qpos_addr
        self.dof_addr = dof_addr

    def extract_base_rpy(self, data=None):
        """Extract roll, pitch, yaw rate from base state."""
        if data is None:
            data = self.data
        w, x, y, z = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        roll_rate = data.qvel[3]
        pitch_rate = data.qvel[4]
        yaw_rate = data.qvel[5]
        return roll, pitch, roll_rate, pitch_rate, yaw_rate

    def compute_gravity_feedforward(self, data=None):
        """Return qfrc_bias (gravity + Coriolis/centrifugal compensation)."""
        if data is None:
            data = self.data
        mj.mj_forward(self.model, data)
        return data.qfrc_bias.copy()

    def compute_pd_torque(self, target_pos, current_pos, current_vel, kp, kd, effort_limit):
        """PD torque with effort clipping."""
        torque = kp * (target_pos - current_pos) - kd * current_vel
        torque = np.clip(torque, -effort_limit, effort_limit)
        if not np.isfinite(torque):
            torque = 0.0
        return torque

    def compute_leg_torques(self, data, gravity, joint_targets):
        """Compute PD + gravity feedforward torques for hip/thigh/calf joints.

        Parameters
        ----------
        data : MjData
            Current MuJoCo data.
        gravity : ndarray
            Gravity feedforward torques (qfrc_bias).
        joint_targets : dict
            {leg: (theta_hip, theta_thigh, theta_calf)} target joint positions.

        Returns
        -------
        torques : ndarray
            Actuator torques array of size model.nu.
        """
        torques = np.zeros(self.model.nu)
        for leg in LEGS:
            targets = joint_targets[leg]
            for k, joint_type in enumerate(JOINT_TYPES):
                act_idx = ACTUATOR_MAP[leg][joint_type]
                # Use per-controller PD_GAINS if defined, else module-level
                pd_gains = getattr(self, 'PD_GAINS', PD_GAINS)
                kp, kd = pd_gains[joint_type]
                qp = self.qpos_addr[leg][joint_type]
                df = self.dof_addr[leg][joint_type]

                torque = kp * (targets[k] - data.qpos[qp]) - kd * data.qvel[df]
                torque += gravity[df]
                torque = np.clip(torque, -EFFORT_LIMITS[joint_type], EFFORT_LIMITS[joint_type])
                if not np.isfinite(torque):
                    torque = 0.0
                torques[act_idx] = torque
        return torques
