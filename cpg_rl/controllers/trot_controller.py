"""
Trot motion controller for DHAV-HY — Hybrid wheeled trotting.

The robot moves using a combination of:
1. Leg-generated propulsion through coordinated diagonal trotting.
2. Continuous wheel rolling during stance.

The wheel and leg motion are coordinated to produce smooth locomotion
throughout the gait cycle. Improved for higher speeds while maintaining
stability and base height.
"""

import numpy as np
import mujoco as mj

from .base_controller import (
    BaseMotion, STAND_POSE, PD_GAINS, EFFORT_LIMITS,
    WHEEL_EFFORT_LIMIT, WHEEL_RADIUS, ACTUATOR_MAP, LEGS, JOINT_TYPES,
    STANCE_FOOT_Z, compute_differential_wheel_velocities, ramp_velocity,
    TRACK_WIDTH
)
from cpg_rl.cpg.coupled_cpg import CPGNetwork, create_trot_cpg
from cpg_rl.cpg.foot_trajectory import FootTrajectoryGenerator
from cpg_rl.kinematics.inverse_kinematics import (
    forward_kinematics, inverse_kinematics as _ik,
    HIP_OFFSETS, JOINT_LIMITS, rotation_matrix_y
)
from cpg_rl.wheel.wheel_controller import WheelController
from .stabilization import Stabilizer


# Physical parameters (from the MuJoCo model)
MAX_VELOCITY = 200.0          # rad/s (5.6 m/s at r=0.077m)
MAX_TORQUE = 14.0            # Nm
WHEEL_RADIUS = 0.077

# Wheel coordination parameters
WHEEL_FORWARD_VEL_SCALE = 1.0
WHEEL_TURN_VEL_SCALE = 0.5

# Optimized CPG parameters for stable trotting
CPG_OMEGA_BASE = 14.0          # base CPG omega (rad/s) ~2.2 Hz - slower, more stable
CPG_OMEGA_MAX = 18.0           # max CPG omega (rad/s)
CPG_OMEGA_SCALE = 1.0         # omega scaling per m/s velocity

# Optimized step parameters for stable trotting
STEP_LENGTH_BASE = 0.18       # shorter step for less yaw moment and more stability
STEP_LENGTH_MAX = 0.30        # max step length (m)
STEP_LENGTH_VEL_SCALE = 0.05  # additional step length per m/s

# Higher duty factor: 70% stance, 30% swing - more stability
DUTY_FACTOR_BASE = 0.70       # increased from 0.55 for stability

# Stability gains (high for stability)
YAW_STIFFNESS = 80000.0
YAW_DAMPING = 8000.0
ROLL_STIFFNESS = 40000.0      # increased from 35000.0
PITCH_STIFFNESS = 35000.0     # increased from 25000.0

# Calf angle adjustment gains
YAW_CALF_GAIN = 0.20
ROLL_CALF_GAIN = 0.12
PITCH_CALF_GAIN = 0.12

# Smoothing parameters
MAX_JOINT_VEL_CHANGE = 10.0   # further reduced for smoother motion clamping
SMOOTH_ALPHA = 0.70           # increased from 0.50 for more smoothing (stronger low-pass)

# Velocity regulation gains
VEL_KP = 0.5
VEL_MAX_SCALE = 1.3
VEL_MIN_SCALE = 0.75

# Height control parameters - optimized for smooth trotting
KP_HEIGHT_TROT = 35000.0      # increased from 20000.0 for less z oscillation
KD_HEIGHT_TROT = 1500.0       # increased from 1000.0 for stability

# Straight-line heading correction: P controller
# Negative KP corrects the systematic rightward drift
YAW_CORRECTION_KP_TROT = -100.0    # proportional gain (strong correction)
YAW_DEADBAND_TROT = 0.005          # rad - tiny deadband
MAX_YAW_CORRECTION_TROT = 6.0      # rad/s - hard safety cap


class TrotMotion(BaseMotion):
    """Diagonal trot gait — diagonal pairs swing together.

    The robot moves using a combination of:
    1. Leg-generated propulsion through coordinated diagonal trotting.
    2. Continuous wheel rolling during stance.

    Gait parameters:
    - CPG with diagonal phase offsets: FL=0, FR=pi, RL=pi, RR=0
    - Duty factor: 50% stance, 50% swing
    - Swing height: 0.10 m (original)
    - Step length: 0.10 m (original)
    - Omega: configurable rad/s (~1.3 Hz base)
    """

    def __init__(self, model, data, qpos_addr, dof_addr):
        super().__init__(model, data, qpos_addr, dof_addr)

        # --- Stabilizer for body height, pitch, roll, yaw ---
        # Use high stabilizer parameters for stability
        self.stabilizer = Stabilizer(
            model, data, qpos_addr, dof_addr,
            leg_adapt_kp=2000.0,
            max_adapt_rad=0.25,
            height_adapt_scale=0.10,
            ik_phase_bucket=0.10,
            height_adapt_deadband=0.05,
            kp_height=KP_HEIGHT_TROT,     # increased for smooth height
            kp_pitch=25000.0,      # increased for stability
            kp_roll=30000.0,       # increased for stability
            kp_yaw=50000.0,        # ORIGINAL value
            yaw_damping=5000.0,    # ORIGINAL value
            friction_scale=1.0,
            disable_yaw=True,  # Zero yaw for straight-line trotting
        )

        # --- CPG network for diagonal trot generation ---
        # Optimized omega for 2.5 m/s trotting
        omega = CPG_OMEGA_BASE
        omega = np.clip(omega, CPG_OMEGA_BASE, CPG_OMEGA_MAX)

        self.cpg = create_trot_cpg(dt=model.opt.timestep)
        self.cpg.alpha = 20.0
        self.cpg.mu = 1.0
        self.cpg.set_omega(omega)
        self.cpg.leg_phases = {'FL': 0.0, 'FR': np.pi, 'RL': np.pi, 'RR': 0.0}

        # --- Foot trajectory generator ---
        # Use ORIGINAL parameters for stability
        self.traj_gen = FootTrajectoryGenerator(
            swing_height=0.15,       # optimized for 2.5 m/s
            step_length=STEP_LENGTH_BASE,  # 0.18 m optimized
            duty_factor=DUTY_FACTOR_BASE,  # 0.48 - reduced for faster gait
        )

        # --- Wheel controller (velocity mode) ---
        self.wheel_ctrl = WheelController(
            wheel_radius=WHEEL_RADIUS,
            max_velocity=MAX_VELOCITY,
            max_torque=MAX_TORQUE,
            control_mode='velocity',
        )

        # --- Current wheel velocity commands ---
        self._current_wheel_cmds = {leg: 0.0 for leg in LEGS}

        # --- Wheel velocity smoothing (low-pass filter) ---
        self._wheel_cmd_smoothed = {leg: 0.0 for leg in LEGS}
        self._WHEEL_SMOOTH_ALPHA = 0.50  # increased from 0.30 for smoother motion

        # --- Constant wheel velocities (4 constants) ---
        self._use_constant_wheels = False
        self._constant_wheel_vels = {leg: 0.0 for leg in LEGS}

        # --- Velocity command tracking ---
        self._cmd_vel_x = 0.0
        self._cmd_vel_y = 0.0
        self._cmd_yaw_rate = 0.0
        self._target_yaw = None       # locked heading for straight-line travel
        self._yaw_locked = False
        self._yaw_integral = 0.0      # integral of yaw error
        self._y_init = None           # initial Y position for position-based correction

        # --- Actual velocity tracking ---
        self._actual_vel_x = 0.0

        # --- Gait phase tracking ---
        self._phase = {leg: 0.0 for leg in LEGS}

        # --- Stance foot world positions (cached per stance phase) ---
        self._stance_foot_world = {leg: None for leg in LEGS}
        self._leg_in_stance = {leg: False for leg in LEGS}

        # --- Cached IK targets for stance legs ---
        self._stance_target_cache = {leg: None for leg in LEGS}

        # --- Swing trajectory caches ---
        self._swing_traj_cache = {}
        self._precompute_swing_trajectories()

        # --- Velocity regulation ---
        self._vel_integral = 0.0

        # --- Smoothing state for joint targets ---
        self._prev_joint_targets = {leg: STAND_POSE[leg] for leg in LEGS}
        self._filtered_targets = {leg: STAND_POSE[leg] for leg in LEGS}

        # --- Configurable gait parameters (optimized for speed) ---
        self._gait_frequency = CPG_OMEGA_BASE  # 22.0 rad/s
        self._duty_factor = DUTY_FACTOR_BASE   # 0.60 - reduced for faster gait
        self._stride_length = STEP_LENGTH_BASE   # 0.22 m optimized
        self._swing_height = 0.10                # normal swing height

    def _precompute_swing_trajectories(self):
        """Pre-compute swing phase joint targets for all legs."""
        stance_threshold = self.traj_gen.duty_factor * 2.0 * np.pi
        swing_range = 2.0 * np.pi - stance_threshold

        for leg in LEGS:
            hip_off = HIP_OFFSETS[leg]

            traj_targets = []
            for i in range(51):
                s = i / 50.0
                cpg_phase = stance_threshold + s * swing_range

                foot_x, foot_z = self.traj_gen.compute_foot_position(
                    cpg_phase, stance_z=STANCE_FOOT_Z
                )

                # Scale step length by velocity regulation
                speed = abs(self._cmd_vel_x)
                step_len_scale = 1.0 + speed * STEP_LENGTH_VEL_SCALE / STEP_LENGTH_BASE
                step_len_scale = np.clip(step_len_scale, 0.8, 2.0)
                foot_x = foot_x * step_len_scale

                p_des_xz = (
                    hip_off[0] + foot_x,
                    hip_off[2] + foot_z,
                )

                hip_q = STAND_POSE[leg][0]
                targets = self.stabilizer.compute_swing_ik(leg, p_des_xz)
                traj_targets.append(targets)

            self._swing_traj_cache[leg] = {
                'targets': traj_targets,
                'stance_threshold': stance_threshold,
                'swing_range': swing_range,
            }

    def set_velocity_command(self, vel_x, vel_y=0.0, yaw_rate=0.0):
        """Set the driving velocity command.

        When yaw_rate is 0, heading is locked for straight-line travel.
        """
        self._cmd_vel_x = float(vel_x)
        self._cmd_vel_y = float(vel_y)
        self._cmd_yaw_rate = float(yaw_rate)
        # Mark heading lock as pending — capture actual yaw on next compute call
        if abs(yaw_rate) < 0.01:
            self._yaw_locked = True
            self._target_yaw = None  # will be captured in compute_torques
        else:
            self._yaw_locked = False
            self._target_yaw = None

    def set_target_velocity(self, velocity_m_s, vel_y=0.0, yaw_rate=0.0):
        """Set the desired linear velocity (m/s) for the robot."""
        self._cmd_vel_x = float(velocity_m_s)
        self._cmd_vel_y = float(vel_y)
        self._cmd_yaw_rate = float(yaw_rate)

    def set_wheel_velocities(self, fl_wheel, fr_wheel, rl_wheel, rr_wheel):
        """Set 4 constant wheel velocities directly (rad/s).

        Bypasses differential drive — each wheel gets its own fixed speed.
        Use this for precise control of all 4 wheels independently.
        """
        self._cmd_vel_x = 0.0
        self._cmd_yaw_rate = 0.0
        self._constant_wheel_vels = {
            'FL': float(fl_wheel),
            'FR': float(fr_wheel),
            'RL': float(rl_wheel),
            'RR': float(rr_wheel),
        }
        self._use_constant_wheels = True

    def set_gait_parameters(self, frequency=None, duty_factor=None,
                            stride_length=None, swing_height=None):
        """Configure gait parameters at runtime.

        Parameters
        ----------
        frequency : float or None
            CPG omega (rad/s). If None, use optimized 14.0 rad/s for 2.5 m/s.
        duty_factor : float or None
            Stance-phase duty factor (0 < df < 1). If None, use 0.48.
        stride_length : float or None
            Nominal step length in meters. If None, use optimized 0.18 m.
        swing_height : float or None
            Swing leg clearance in meters. If None, use optimized 0.15 m.
        """
        if frequency is not None:
            self._gait_frequency = float(frequency)
            self.cpg.set_omega(self._gait_frequency)
        else:
            self._gait_frequency = CPG_OMEGA_BASE
            self.cpg.set_omega(self._gait_frequency)
        if duty_factor is not None:
            self._duty_factor = float(duty_factor)
            self.traj_gen.duty_factor = self._duty_factor
            # Re-precompute trajectories with new duty factor
            self._precompute_swing_trajectories()
        else:
            self._duty_factor = DUTY_FACTOR_BASE
        if stride_length is not None:
            self._stride_length = float(stride_length)
            self.traj_gen.step_length = self._stride_length
        else:
            self._stride_length = STEP_LENGTH_BASE
        if swing_height is not None:
            self._swing_height = float(swing_height)
            self.traj_gen.swing_height = self._swing_height
        else:
            self._swing_height = 0.10

    def _get_stance_target(self, leg, phase, z_err, z_offset, pitch_offset,
                           roll_offset, yaw_correction, roll_correction,
                           pitch_correction, base_pos, base_quat):
        """Stance target: foot LOCKED at world position captured at stance entry."""
        cache = self._stance_foot_world[leg]
        if cache is None:
            # Fallback: compute IK at current position
            hip_q = self.data.qpos[self.qpos_addr[leg]['hip']]
            thigh_q = self.data.qpos[self.qpos_addr[leg]['thigh']]
            calf_q = self.data.qpos[self.qpos_addr[leg]['calf']]
            foot_world = forward_kinematics(hip_q, thigh_q, calf_q, leg=leg)
            foot_world = foot_world + base_pos
            self._stance_foot_world[leg] = foot_world
            cache = foot_world

        # Convert world foot position to base-relative position
        w, x_val, y_val, z_val = base_quat
        yaw = np.arctan2(2.0 * (w * z_val + x_val * y_val),
                         1.0 - 2.0 * (y_val * y_val + z_val * z_val))
        R_inv = rotation_matrix_y(-yaw)
        foot_world_rel = cache - base_pos
        foot_base = R_inv @ foot_world_rel

        p_des = np.array([
            foot_base[0],
            HIP_OFFSETS[leg][1],
            foot_base[2],
        ])

        z_adapt = z_err * self.stabilizer.height_adapt_scale
        z_adapt = np.clip(z_adapt,
                          -self.stabilizer.max_adapt_rad,
                          self.stabilizer.max_adapt_rad)
        p_des[2] += z_adapt

        ik_result = _ik(
            p_des, leg=leg, joint_limits=JOINT_LIMITS
        )
        if ik_result['success']:
            targets = (
                ik_result['theta_hip'],
                ik_result['theta_thigh'],
                ik_result['theta_calf'],
            )
        else:
            targets = STAND_POSE[leg]

        # Body leveling offsets
        pitch_term = -pitch_offset * hip_off[0] * 0.4
        roll_term = -roll_offset * hip_off[1] * 0.4
        z_adj = z_offset + pitch_term + roll_term

        calf_adj = z_adj * 0.6
        calf_adj = np.clip(calf_adj,
                           -self.stabilizer.max_adapt_rad,
                           self.stabilizer.max_adapt_rad)

        targets = (targets[0], targets[1], targets[2] + calf_adj)

        side = -1.0 if HIP_OFFSETS[leg][1] > 0 else 1.0
        front = 1.0 if hip_off[0] > 0 else -1.0

        yaw_calf_adj = yaw_correction * side * YAW_CALF_GAIN
        roll_calf_adj = roll_correction * side * ROLL_CALF_GAIN
        pitch_calf_adj = pitch_correction * front * PITCH_CALF_GAIN

        total_calf_adj = yaw_calf_adj + roll_calf_adj + pitch_calf_adj
        total_calf_adj = np.clip(total_calf_adj,
                                 -self.stabilizer.max_adapt_rad,
                                 self.stabilizer.max_adapt_rad)

        targets = (targets[0], targets[1], targets[2] + total_calf_adj)

        return targets

    def _get_swing_target(self, leg, phase, z_offset, pitch_offset, roll_offset,
                           yaw_correction, roll_correction, pitch_correction):
        """Swing phase joint target via pre-computed CPG trajectory."""
        cache = self._swing_traj_cache[leg]
        stance_threshold = cache['stance_threshold']
        swing_range = cache['swing_range']
        targets_list = cache['targets']

        s = (phase - stance_threshold) / swing_range
        s = np.clip(s, 0.0, 1.0)

        idx = s * 50.0
        i0 = int(idx)
        i1 = min(i0 + 1, 50)
        frac = idx - i0

        t0 = targets_list[i0]
        t1 = targets_list[i1]

        targets = tuple(
            t0[k] + frac * (t1[k] - t0[k])
            for k in range(3)
        )

        # Velocity-based step length scaling
        vel_error = self._cmd_vel_x - self._actual_vel_x
        # Smooth velocity error with a first-order filter
        self._vel_integral = self._vel_integral * 0.85 + vel_error * 0.15
        vel_error_smooth = self._vel_integral
        step_scale = 1.0 + vel_error_smooth * VEL_KP
        step_scale = np.clip(step_scale, VEL_MIN_SCALE, VEL_MAX_SCALE)
        hip_deviation = targets[0] - STAND_POSE[leg][0]
        targets = (STAND_POSE[leg][0] + hip_deviation * step_scale,
                   targets[1], targets[2])

        # Body leveling offsets
        hip_off = HIP_OFFSETS[leg]
        hip_x = hip_off[0]
        hip_y = hip_off[1]

        pitch_term = -pitch_offset * hip_x * 0.4
        roll_term = -roll_offset * hip_y * 0.4
        z_adj = z_offset + pitch_term + roll_term

        calf_adj = z_adj * 0.6
        calf_adj = np.clip(calf_adj, -self.stabilizer.max_adapt_rad,
                           self.stabilizer.max_adapt_rad)

        targets = (targets[0], targets[1], targets[2] + calf_adj)

        # Apply orientation corrections (small, for stability)
        targets = self.stabilizer.apply_orientation_correction_to_targets(
            targets, leg,
            yaw_correction * 0.3,
            roll_correction * 0.3,
            pitch_correction * 0.3
        )

        # Smooth joint targets
        targets = self._smooth_actions(leg, targets)

        return targets

    def _smooth_actions(self, leg, new_targets):
        """Smooth joint targets: velocity clamping + low-pass filter."""
        prev = self._prev_joint_targets[leg]
        clamped = []
        max_change = MAX_JOINT_VEL_CHANGE * self.model.opt.timestep
        for k in range(3):
            diff = new_targets[k] - prev[k]
            if diff > max_change:
                diff = max_change
            elif diff < -max_change:
                diff = -max_change
            clamped.append(prev[k] + diff)
        clamped = tuple(clamped)

        filtered_prev = self._filtered_targets[leg]
        filtered = []
        for k in range(3):
            filtered_val = (SMOOTH_ALPHA * clamped[k] +
                            (1.0 - SMOOTH_ALPHA) * filtered_prev[k])
            filtered.append(filtered_val)
        filtered = tuple(filtered)

        self._prev_joint_targets[leg] = filtered
        self._filtered_targets[leg] = filtered

        return filtered

    def compute_leg_torques(self, data, gravity):
        """Compute leg torques using CPG-driven stance/swing logic."""
        self.cpg.step()

        mj.mj_forward(self.model, self.data)
        cpg_states = self.cpg.get_states()
        stance_threshold = self.traj_gen.duty_factor * 2.0 * np.pi

        # --- Stabilizer body corrections ---
        self.stabilizer.capture_initial_state(self.data)
        base_pos = self.data.qpos[0:3]
        base_vel = self.data.qvel[0:3]
        self._actual_vel_x = float(base_vel[0])
        z_err = self.stabilizer.get_height_error(base_pos)
        (z_offset, pitch_offset, roll_offset,
         yaw_correction, roll_correction, pitch_correction) = (
             self.stabilizer.compute_body_corrections(self.data)
         )

        # --- Zero out yaw correction for trotting gait ---
        yaw_correction = 0.0

        base_quat = data.qpos[3:7]

        # --- Joint targets per leg ---
        joint_targets = {}
        for leg in LEGS:
            x_osc, y_osc = cpg_states[leg]
            phase = np.arctan2(y_osc, x_osc)
            if phase < 0:
                phase += 2.0 * np.pi
            in_stance = phase < stance_threshold

            # --- Stance phase entry: capture foot position once ---
            if in_stance and not self._leg_in_stance[leg]:
                hip_q = data.qpos[self.qpos_addr[leg]['hip']]
                thigh_q = data.qpos[self.qpos_addr[leg]['thigh']]
                calf_q = data.qpos[self.qpos_addr[leg]['calf']]
                foot_world = forward_kinematics(
                    hip_q, thigh_q, calf_q, leg=leg
                )
                self._stance_foot_world[leg] = foot_world + base_pos
                # Cache IK target for stance phase — compute once
                w, x_val, y_val, z_val = base_quat
                yaw = np.arctan2(2.0 * (w * z_val + x_val * y_val),
                                 1.0 - 2.0 * (y_val * y_val + z_val * z_val))
                R_inv = rotation_matrix_y(-yaw)
                foot_world_rel = self._stance_foot_world[leg] - base_pos
                foot_base = R_inv @ foot_world_rel
                p_des = np.array([
                    foot_base[0],
                    HIP_OFFSETS[leg][1],
                    foot_base[2],
                ])
                z_adapt = z_err * self.stabilizer.height_adapt_scale
                z_adapt = np.clip(z_adapt,
                                  -self.stabilizer.max_adapt_rad,
                                  self.stabilizer.max_adapt_rad)
                p_des[2] += z_adapt
                ik_result = _ik(
                    p_des, leg=leg, joint_limits=JOINT_LIMITS
                )
                if ik_result['success']:
                    targets = (
                        ik_result['theta_hip'],
                        ik_result['theta_thigh'],
                        ik_result['theta_calf'],
                    )
                else:
                    targets = STAND_POSE[leg]
                self._stance_target_cache[leg] = targets
                self._leg_in_stance[leg] = True

            # --- Exit stance when CPG phase enters swing ---
            elif not in_stance:
                self._leg_in_stance[leg] = False
                self._stance_foot_world[leg] = None
                self._stance_target_cache[leg] = None

            # --- Assign joint targets based on phase ---
            if in_stance:
                targets = self._stance_target_cache[leg]
                targets = self._smooth_actions(leg, targets)
            else:
                targets = self._get_swing_target(
                    leg, phase, z_offset, pitch_offset, roll_offset,
                    yaw_correction, roll_correction, pitch_correction
                )

            joint_targets[leg] = targets

        torques = super().compute_leg_torques(self.data, gravity, joint_targets)
        return torques

    def compute_wheel_torques(self, data):
        """Compute wheel torques with velocity tracking and gait coordination."""
        dt = self.model.opt.timestep

        # --- Zero y-velocity command ---
        self._cmd_vel_y = 0.0

        # --- Compute desired wheel velocities ---
        if self._use_constant_wheels:
            # Use 4 constant wheel velocities directly
            target_cmds = dict(self._constant_wheel_vels)
        else:
            # Differential drive from velocity command
            target_cmds = compute_differential_wheel_velocities(
                self._cmd_vel_x, self._cmd_yaw_rate, track_width=TRACK_WIDTH
            )

        # --- Ramp wheel velocities for smooth acceleration ---
        ramped_cmds = {}
        for leg in LEGS:
            ramped_cmds[leg] = ramp_velocity(
                self._current_wheel_cmds[leg], target_cmds[leg], dt
            )
        self._current_wheel_cmds = ramped_cmds

        # --- Additional low-pass smoothing for ultra-smooth wheel motion ---
        smoothed_cmds = {}
        for leg in LEGS:
            prev = self._wheel_cmd_smoothed[leg]
            smoothed_cmds[leg] = (
                self._WHEEL_SMOOTH_ALPHA * self._current_wheel_cmds[leg] +
                (1.0 - self._WHEEL_SMOOTH_ALPHA) * prev
            )
        self._wheel_cmd_smoothed = smoothed_cmds
        self.wheel_ctrl.set_velocities(smoothed_cmds)

        # --- Compute wheel torques via PID control ---
        torques = np.zeros(self.model.nu)
        for leg in LEGS:
            act_idx = ACTUATOR_MAP[leg]['wheel']
            qp = self.qpos_addr[leg]['wheel']
            df = self.dof_addr[leg]['wheel']
            ca = self.data.qpos[qp]
            cv = self.data.qvel[df]
            torque = self.wheel_ctrl.get_control_signal(
                leg, ca, cv, dt=self.model.opt.timestep)
            torque = np.clip(torque, -WHEEL_EFFORT_LIMIT, WHEEL_EFFORT_LIMIT)
            torques[act_idx] = torque

        # Yaw rate damping: actively counter any yaw rotation to keep yaw = 0
        yaw_rate = self.data.qvel[5]
        YAW_DAMPING_KP = 80.0  # N*m/(rad/s)
        YAW_POSITION_KP = 50.0  # N*m/rad

        w, x_val, y_val, z_val = self.data.qpos[3:7]
        current_yaw = np.arctan2(
            2.0 * (w * z_val + x_val * y_val),
            1.0 - 2.0 * (y_val * y_val + z_val * z_val)
        )
        yaw_counter_torque = -YAW_POSITION_KP * current_yaw - YAW_DAMPING_KP * yaw_rate

        # Apply yaw counter-torque differentially: left wheels -, right wheels +
        for i, leg in enumerate(LEGS):
            act_idx = ACTUATOR_MAP[leg]['wheel']
            side = -1.0 if HIP_OFFSETS[leg][1] > 0 else 1.0
            torques[act_idx] += yaw_counter_torque * side * 0.5

        return torques

    def compute_torques(self, model, data):
        """Compute torques for all leg and wheel actuators.

        Called by the MuJoCo simulation loop in mujoco_viz.py.

        Returns
        -------
        torques : ndarray
            Actuator torques array of size model.nu.
        """
        mj.mj_forward(model, data)
        gravity = data.qfrc_bias.copy()
        self.data = data
        self.model = model

        # --- Yaw: completely zero for trotting gait ---
        self._cmd_yaw_rate = 0.0
        self._target_yaw = None
        self._yaw_locked = False

        # --- Compute leg torques via CPG-driven stance/swing ---
        leg_torques = self.compute_leg_torques(self.data, gravity)

        # Light yaw damping on leg hip joints to supplement wheel damping
        yaw_rate = self.data.qvel[5]
        w, x_val, y_val, z_val = self.data.qpos[3:7]
        current_yaw = np.arctan2(
            2.0 * (w * z_val + x_val * y_val),
            1.0 - 2.0 * (y_val * y_val + z_val * z_val)
        )
        LEG_YAW_DAMPING = 15.0  # N*m/(rad/s) - light leg yaw damping
        LEG_YAW_POSITION = 10.0  # N*m/rad
        leg_yaw_torque = -LEG_YAW_POSITION * current_yaw - LEG_YAW_DAMPING * yaw_rate
        
        # Apply leg yaw counter-torque differentially to hip joints
        for i, leg in enumerate(LEGS):
            act_idx = ACTUATOR_MAP[leg]['hip']
            side = -1.0 if HIP_OFFSETS[leg][1] > 0 else 1.0
            leg_torques[act_idx] += leg_yaw_torque * side * 0.2

        # --- Compute wheel torques ---
        wheel_torques = self.compute_wheel_torques(self.data)

        return leg_torques + wheel_torques

    def reset(self):
        """Reset controller state."""
        self._current_wheel_cmds = {leg: 0.0 for leg in LEGS}
        self._wheel_cmd_smoothed = {leg: 0.0 for leg in LEGS}
        self._cmd_vel_x = 0.0
        self._cmd_vel_y = 0.0
        self._cmd_yaw_rate = 0.0
        self._actual_vel_x = 0.0
        self._phase = {leg: 0.0 for leg in LEGS}
        self._stance_foot_world = {leg: None for leg in LEGS}
        self._leg_in_stance = {leg: False for leg in LEGS}
        self._stance_target_cache = {leg: None for leg in LEGS}
        self._swing_traj_cache = {}
        self._precompute_swing_trajectories()
        self._vel_integral = 0.0
        self._use_constant_wheels = False
        self._constant_wheel_vels = {leg: 0.0 for leg in LEGS}
        self._target_yaw = None
        self._yaw_locked = False
        self._yaw_integral = 0.0
        self._y_init = None


# Backwards-compatibility alias
TrotController = TrotMotion