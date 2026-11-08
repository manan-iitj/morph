"""
Stabilizer — Common Obstacle-Adaptive Stance + Body Stabilization.

Provides shared stabilization logic for all locomotion controllers:
- Full-leg IK adaptation during stance to keep base at constant height
- Body roll/pitch/yaw offsets during swing foot placement
- Yaw damping to keep body moving in a straight line
- IK caching for real-time performance

Each controller passes its own tunable parameters via the constructor.
"""

import numpy as np
import mujoco as mj

from cpg_rl.kinematics.inverse_kinematics import (
    forward_kinematics, inverse_kinematics, HIP_OFFSETS, JOINT_LIMITS, STAND_POSE
)
from cpg_rl.controllers.base_controller import (
    LEGS, JOINT_TYPES
)


class Stabilizer:
    """
    Stabilizes legs + body roll/pitch/yaw offsets during the entire motion.

    Parameters
    ----------
    model : MjModel
        MuJoCo model.
    data : MjData
        MuJoCo data.
    qpos_addr : dict
        Joint qpos addresses per leg.
    dof_addr : dict
        Joint dof addresses per leg.
    leg_adapt_kp : float
        Height error to foot-position gain (N/m).
    max_adapt_rad : float
        Max deviation per joint from stand pose (radians).
    height_adapt_scale : float
        Fraction of height error applied to foot z target.
    ik_phase_bucket : float
        Bucket size for IK cache granularity (meters).
    height_adapt_deadband : float
        Deadband below which legs stay at stand pose (meters).
    kp_height : float
        Base height stiffness (N/m).
    kp_pitch : float
        Pitch stiffness (N/m/rad).
    kp_roll : float
        Roll stiffness (N/m/rad).
    kp_yaw : float
        Yaw stiffness (N/m/rad) — keeps body moving straight.
    yaw_damping : float
        Yaw rate damping (N·m·s/rad) — counters unwanted rotation.
    friction_scale : float
        Multiplier for effective friction via increased joint torque.
    """

    def __init__(self, model, data, qpos_addr, dof_addr,
                 leg_adapt_kp=1000.0,
                 max_adapt_rad=0.50,
                 height_adapt_scale=0.15,
                 ik_phase_bucket=0.10,
                 height_adapt_deadband=0.10,
                 kp_height=6000.0,
                 kp_pitch=4000.0,
                 kp_roll=4000.0,
                 kp_yaw=6000.0,
                 yaw_damping=1000.0,
                 friction_scale=1.0,
                 disable_yaw=False):
        self.model = model
        self.data = data
        self.qpos_addr = qpos_addr
        self.dof_addr = dof_addr

        # Adaptation gains
        self.leg_adapt_kp = leg_adapt_kp
        self.max_adapt_rad = max_adapt_rad
        self.height_adapt_scale = height_adapt_scale
        self.ik_phase_bucket = ik_phase_bucket
        self.height_adapt_deadband = height_adapt_deadband

        # Body stabilization gains — VERY HIGH for stability
        self.kp_height = kp_height
        self.kp_pitch = kp_pitch
        self.kp_roll = kp_roll
        self.kp_yaw = kp_yaw
        self.yaw_damping = yaw_damping
        self.friction_scale = friction_scale

        # Disable yaw corrections entirely (for straight-line gaits)
        self.disable_yaw = disable_yaw

        # Runtime state
        self._target_base_z = None
        self._target_yaw = None
        self._ik_cache = {}          # {(leg, phase_bucket): (hip, thigh, calf)}
        self._last_z_err = 0.0
        self._last_yaw_err = 0.0
        self._last_yaw_rate = 0.0
        self._last_roll = 0.0
        self._last_pitch = 0.0
        self._last_roll_rate = 0.0
        self._last_pitch_rate = 0.0

    def capture_initial_state(self, data):
        """Capture the initial base height and yaw on first call."""
        if self._target_base_z is None:
            self._target_base_z = float(data.qpos[2])
        if self._target_yaw is None:
            w, x, y, z = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            self._target_yaw = float(np.arctan2(siny_cosp, cosy_cosp))

    def get_height_error(self, base_pos):
        """Return base height error: target_base_z - current_base_z."""
        if self._target_base_z is None:
            return 0.0
        return float(self._target_base_z - base_pos[2])

    def get_yaw_error(self, data=None):
        """Return yaw error: target_yaw - current_yaw."""
        if self._target_yaw is None:
            return 0.0
        if data is None:
            data = self.data
        w, x, y, z = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        current_yaw = float(np.arctan2(siny_cosp, cosy_cosp))
        yaw_err = current_yaw - self._target_yaw
        while yaw_err > np.pi:
            yaw_err -= 2.0 * np.pi
        while yaw_err < -np.pi:
            yaw_err += 2.0 * np.pi
        return float(yaw_err)

    def extract_base_rpy(self, data=None):
        """Extract roll, pitch, yaw from base quaternion."""
        if data is None:
            data = self.data
        w, x, y, z = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = float(np.arctan2(sinr_cosp, cosr_cosp))
        sinp = 2.0 * (w * y - z * x)
        sinp = float(np.clip(sinp, -1.0, 1.0))
        pitch = float(np.arcsin(sinp))
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = float(np.arctan2(siny_cosp, cosy_cosp))
        return roll, pitch, yaw

    def compute_body_corrections(self, data=None):
        """
        Compute body leveling corrections for swing foot placement and
        stance leg torque adjustments.

        Returns
        -------
        z_offset : float
            Vertical offset applied to swing foot z target.
        pitch_offset : float
            Pitch-based offset applied per-leg hip x position.
        roll_offset : float
            Roll-based offset applied per-leg hip y position.
        yaw_correction : float
            Yaw correction torque applied to keep body straight.
        roll_correction : float
            Roll correction torque applied to keep body level.
        pitch_correction : float
            Pitch correction torque applied to keep body level.
        """
        if data is None:
            data = self.data
        base_pos = data.qpos[0:3]
        roll, pitch, yaw = self.extract_base_rpy(data)

        z_err = self.get_height_error(base_pos)
        z_offset = z_err * 0.5
        pitch_offset = 0.0 - pitch
        roll_offset = 0.0 - roll

        # Yaw correction: keep body facing initial direction
        if self.disable_yaw:
            yaw_correction = 0.0
        else:
            yaw_err = self.get_yaw_error(data)
            yaw_err = 0.7 * self._last_yaw_err + 0.3 * yaw_err
            self._last_yaw_err = yaw_err
            yaw_rate = data.qvel[5]
            yaw_rate = 0.7 * self._last_yaw_rate + 0.3 * yaw_rate
            self._last_yaw_rate = yaw_rate
            yaw_correction = self.kp_yaw * yaw_err - self.yaw_damping * yaw_rate

        # Roll correction: keep body level left-right
        roll_err = 0.0 - roll
        roll_err = 0.7 * self._last_roll + 0.3 * roll_err
        self._last_roll = roll_err
        roll_rate = data.qvel[4]
        roll_rate = 0.7 * self._last_roll_rate + 0.3 * roll_rate
        self._last_roll_rate = roll_rate
        roll_correction = self.kp_roll * roll_err - self.yaw_damping * roll_rate

        # Pitch correction: keep body level front-back
        pitch_err = 0.0 - pitch
        pitch_err = 0.7 * self._last_pitch + 0.3 * pitch_err
        self._last_pitch = pitch_err
        pitch_rate = data.qvel[3]
        pitch_rate = 0.7 * self._last_pitch_rate + 0.3 * pitch_rate
        self._last_pitch_rate = pitch_rate
        pitch_correction = self.kp_pitch * pitch_err - self.yaw_damping * pitch_rate

        return z_offset, pitch_offset, roll_offset, yaw_correction, roll_correction, pitch_correction

    def compute_adaptive_stance(self, leg, z_err, p_stand):
        """
        Compute stance joint targets using full-leg IK adaptation.

        If |z_err| is within deadband, returns stand pose (no IK needed).
        Otherwise, adjusts foot z target and runs IK for all three joints.
        """
        z_err = 0.7 * self._last_z_err + 0.3 * z_err
        self._last_z_err = z_err

        if not self.leg_adapt_enabled or abs(z_err) < self.height_adapt_deadband:
            return STAND_POSE[leg]

        p_des = np.array([
            p_stand[0],
            HIP_OFFSETS[leg][1],
            p_stand[2] - z_err * self.height_adapt_scale,
        ])

        height_bucket = int(z_err / self.ik_phase_bucket)
        cache_key = (leg, height_bucket)
        cached = self._ik_cache.get(cache_key)
        if cached is not None:
            return cached

        ik_result = inverse_kinematics(
            p_des, leg=leg, joint_limits=JOINT_LIMITS
        )
        if ik_result['success']:
            targets = (
                ik_result['theta_hip'],
                ik_result['theta_thigh'],
                ik_result['theta_calf'],
            )
            self._ik_cache[cache_key] = targets
            return targets
        return STAND_POSE[leg]

    def compute_swing_ik(self, leg, p_des_xz):
        """
        Compute swing joint targets using knee-dominant IK.

        Hip is fixed at stand value; thigh and calf solve for desired foot x,z.
        """
        hip_q = STAND_POSE[leg][0]

        best_error = float('inf')
        best_angles = (hip_q, STAND_POSE[leg][1], STAND_POSE[leg][2])

        for thigh in np.linspace(JOINT_LIMITS['thigh'][0],
                                 JOINT_LIMITS['thigh'][1], 21):
            for calf in np.linspace(JOINT_LIMITS['calf'][0],
                                    JOINT_LIMITS['calf'][1], 21):
                p = forward_kinematics(hip_q, thigh, calf, leg=leg)
                err = abs(p[0] - p_des_xz[0]) + abs(p[2] - p_des_xz[1])
                if err < best_error:
                    best_error = err
                    best_angles = (hip_q, thigh, calf)

        return best_angles

    def apply_orientation_correction_to_targets(self, targets, leg,
                                                yaw_correction,
                                                roll_correction,
                                                pitch_correction):
        """
        Apply orientation corrections to joint targets to counter body rotation.

        Adjusts hip/thigh/calf angles per leg to generate counter-torques
        that keep the body level and moving straight.

        Parameters
        ----------
        targets : tuple
            Current (theta_hip, theta_thigh, theta_calf) joint targets.
        leg : str
            Leg identifier.
        yaw_correction : float
            Yaw correction torque.
        roll_correction : float
            Roll correction torque.
        pitch_correction : float
            Pitch correction torque.

        Returns
        -------
        targets : tuple
            Adjusted (theta_hip, theta_thigh, theta_calf) joint targets.
        """
        hip_x = HIP_OFFSETS[leg][0]  # forward/back position
        hip_y = HIP_OFFSETS[leg][1]  # lateral position
        side = 1.0 if hip_y > 0 else -1.0  # left=+1, right=-1
        front = 1.0 if hip_x > 0 else -1.0  # front=+1, back=-1

        yaw_calf_adj = yaw_correction * side * 0.1

        # Roll correction: counter body rolling left-right
        roll_calf_adj = roll_correction * side * 0.08

        # Pitch correction: counter body pitching front-back
        pitch_calf_adj = pitch_correction * front * 0.08

        # Combine all adjustments on calf (primary motion joint)
        total_calf_adj = yaw_calf_adj + roll_calf_adj + pitch_calf_adj
        total_calf_adj = np.clip(total_calf_adj,
                                 -self.max_adapt_rad,
                                 self.max_adapt_rad)

        # Also adjust thigh slightly for additional leverage
        thigh_adj = total_calf_adj * 0.3

        return (targets[0],
                targets[1] + thigh_adj,
                targets[2] + total_calf_adj)

    @property
    def leg_adapt_enabled(self):
        return self.leg_adapt_kp is not None and self.leg_adapt_kp > 0
