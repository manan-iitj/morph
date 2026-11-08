"""
Stair-climbing stepping controller for DHAV1W wheeled-legged quadruped.

Based on Unitree Go2 stair-climbing:
- Approach: wheels roll forward to reach stairs
- Climb: wheels lock, legs step up one at a time onto each stair tread
- Body lifts slightly between stairs
- 4-beat sequential gait: FL -> FR -> RL -> RR
"""

import os
import re

import numpy as np
import mujoco as mj

from .base_controller import (
    BaseMotion, STAND_POSE, PD_GAINS, EFFORT_LIMITS,
    WHEEL_EFFORT_LIMIT, WHEEL_RADIUS, WHEEL_ACCEL_LIMIT, ACTUATOR_MAP,
    LEGS, JOINT_TYPES, STANCE_FOOT_Z, setup_joint_addresses, ramp_velocity
)
from .stabilization import Stabilizer
from cpg_rl.kinematics.inverse_kinematics import (
    forward_kinematics, HIP_OFFSETS, THIGH_Y_OFFSETS,
    WHEEL_Y_OFFSETS, JOINT_LIMITS, rotation_matrix_y
)
from cpg_rl.wheel.wheel_controller import WheelController


# Gait timing
STANCE_DURATION = 0.50
SWING_DURATION = 0.50
SWING_HEIGHT = 0.12
APPROACH_SPEED = 1.0
WHEEL_KP = 200.0

# Body lift parameters
BODY_LIFT_DURATION = 0.40
BODY_LIFT_HEIGHT = 0.08

DEFAULT_RISER_HEIGHT = 0.20
DEFAULT_TREAD_DEPTH = 0.15
MAX_STAIRS = 20

# Stair parameters
MAX_STAIRS = 20

FOOT_Y_OFFSETS = {
    leg: HIP_OFFSETS[leg][1] + THIGH_Y_OFFSETS[leg] + WHEEL_Y_OFFSETS[leg]
    for leg in LEGS
}


def simple_ik(p_desired, leg, current_joints):
    hip_lim = JOINT_LIMITS['hip']
    thigh_lim = JOINT_LIMITS['thigh']
    calf_lim = JOINT_LIMITS['calf']
    hip_c, thigh_c, calf_c = current_joints
    best = None
    best_err = float('inf')
    for hip in np.linspace(hip_lim[0], hip_lim[1], 9):
        for thigh in np.linspace(max(thigh_lim[0], thigh_c - 0.7),
                                 min(thigh_lim[1], thigh_c + 0.7), 6):
            for calf in np.linspace(max(calf_lim[0], calf_c - 0.7),
                                    min(calf_lim[1], calf_c + 0.7), 6):
                p = forward_kinematics(hip, thigh, calf, leg=leg)
                err = np.linalg.norm(p - p_desired)
                if err < best_err:
                    best_err = err
                    best = (hip, thigh, calf)
    if best_err > 0.15:
        return None
    return best


class StairGeometry:
    def __init__(self):
        self.num_steps = 0
        self.tread_positions = []
        self.riser_positions = []
        self.first_stair_x = 0.0


class LegState:
    def __init__(self, leg: str):
        self.leg = leg
        self.phase = 'stance'
        self.phase_timer = 0.0
        self.phase_duration = STANCE_DURATION
        self.foot_start_pos = np.zeros(3)
        self.foot_target_pos = np.zeros(3)
        self.joint_targets = STAND_POSE[leg]
        self.swing_trajectory = None
        self.traj_index = 0
        self.step_count = 0


class StairProfileParser:
    _GEOM_RE = re.compile(
        r'<geom\s+name="([^"]*)"\s+type="box"[^>]*?'
        r'size="([^"]*)"\s+pos="([^"]*)"[^>]*?'
    )

    def parse_stairs_xml(self, filepath: str) -> StairGeometry:
        geom = StairGeometry()
        if not os.path.exists(filepath):
            return geom
        treads = []
        risers = []
        with open(filepath, 'r') as f:
            content = f.read()
        for match in self._GEOM_RE.finditer(content):
            name = match.group(1)
            sx, sy, sz = [float(v) for v in match.group(2).split()]
            x, y, z = [float(v) for v in match.group(3).split()]
            if 'tread' in name:
                treads.append((name, x, y, z, sx, sy, sz))
            elif 'riser' in name:
                risers.append((name, x, y, z, sx, sy, sz))
        treads.sort(key=lambda t: t[1])
        risers.sort(key=lambda r: r[1])
        for name, x, y, z, sx, sy, sz in treads:
            geom.tread_positions.append(np.array([x, y, z + sz]))
        for name, x, y, z, sx, sy, sz in risers:
            geom.riser_positions.append(np.array([x, y, z + sz]))
        geom.num_steps = len(geom.tread_positions)
        if geom.tread_positions:
            geom.first_stair_x = float(geom.tread_positions[0][0])
        return geom


class SwingTrajectoryGenerator:
    @staticmethod
    def cubic_ease_in_out(t):
        t = np.clip(t, 0.0, 1.0)
        return 3.0 * t * t - 2.0 * t * t * t

    @staticmethod
    def generate_positions(start_pos, target_pos, swing_height, n_steps=10):
        trajectory = []
        for i in range(n_steps + 1):
            t = i / float(n_steps)
            s = SwingTrajectoryGenerator.cubic_ease_in_out(t)
            x = (1.0 - s) * start_pos[0] + s * target_pos[0]
            z_base = (1.0 - s) * start_pos[2] + s * target_pos[2]
            lift = swing_height * np.sin(np.pi * s)
            z = z_base + lift
            trajectory.append((x, z))
        return trajectory

    @staticmethod
    def trajectory_to_joint_targets(leg, trajectory, start_joint_targets,
                                    base_pos, yaw):
        joint_targets = []
        hip_q0, thigh_q0, calf_q0 = start_joint_targets
        R_inv = rotation_matrix_y(-yaw)

        def foot_to_ik(x, z):
            p_world = np.array([x, 0.0, z])
            p_rel = R_inv @ (p_world - base_pos)
            p_des = np.array([
                p_rel[0],
                FOOT_Y_OFFSETS[leg],
                p_rel[2],
            ])
            result = simple_ik(p_des, leg, (hip_q0, thigh_q0, calf_q0))
            if result is not None:
                return result
            return (hip_q0, thigh_q0, calf_q0)

        start_ik = foot_to_ik(trajectory[0][0], trajectory[0][1])
        end_ik = foot_to_ik(trajectory[-1][0], trajectory[-1][1])
        for i in range(len(trajectory)):
            t = i / float(max(len(trajectory) - 1, 1))
            joint_targets.append((
                start_ik[0] + t * (end_ik[0] - start_ik[0]),
                start_ik[1] + t * (end_ik[1] - start_ik[1]),
                start_ik[2] + t * (end_ik[2] - start_ik[2]),
            ))
        return joint_targets


class SteppingMotion(BaseMotion):
    def __init__(self, model, data, qpos_addr, dof_addr):
        super().__init__(model, data, qpos_addr, dof_addr)

        self._cmd_vel_x = 0.0
        self._cmd_vel_y = 0.0
        self._cmd_yaw_rate = 0.0
        self._yaw_locked = False

        self.stabilizer = Stabilizer(
            model, data, qpos_addr, dof_addr,
            leg_adapt_kp=2000.0,
            max_adapt_rad=0.30,
            height_adapt_scale=0.15,
            ik_phase_bucket=0.10,
            height_adapt_deadband=0.05,
            kp_height=30000.0,
            kp_pitch=30000.0,
            kp_roll=30000.0,
            kp_yaw=50000.0,
            yaw_damping=5000.0,
            friction_scale=1.0,
        )

        self.wheel_ctrl = WheelController(
            wheel_radius=WHEEL_RADIUS,
            max_velocity=73.0, max_torque=14.0,
            control_mode='velocity',
        )

        self.stair_geometry = StairGeometry()
        self._stair_index = 0
        self._stairs_completed = 0
        self._stair_profile_loaded = False

        self._step_sequence = ['FL', 'FR', 'RL', 'RR']
        self._seq_index = 0
        self._leg_states = {leg: LegState(leg) for leg in LEGS}

        self._prev_joint_targets = {leg: STAND_POSE[leg] for leg in LEGS}
        self._filtered_targets = {leg: STAND_POSE[leg] for leg in LEGS}

        self._step_count = 0
        self._total_steps = 0
        self._stair_climb_count = 0

        self._n_substeps = 1
        self._approach_speed = APPROACH_SPEED
        self._wheel_vel_cmd = 0.0

        self._load_stair_profile()

    def _load_stair_profile(self):
        candidates = [
            'models/dhav1w_description/mujoco/stairs.xml',
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'models', 'dhav1w_description', 'mujoco', 'stairs.xml'),
            '/Users/mananjain/Desktop/xTerra/models/dhav1w_description/mujoco/stairs.xml',
        ]
        for path in candidates:
            if os.path.exists(path):
                parser = StairProfileParser()
                self.stair_geometry = parser.parse_stairs_xml(path)
                self._stair_profile_loaded = self.stair_geometry.num_steps > 0
                if self.stair_geometry.num_steps > 0:
                    print(f"[SteppingMotion] Loaded {self.stair_geometry.num_steps} stairs, "
                          f"first stair x={self.stair_geometry.first_stair_x:.2f}m, "
                          f"first tread z={self.stair_geometry.tread_positions[0][2]:.3f}m")
                return
        print("[SteppingMotion] No stairs.xml found.")

    def set_wheel_controller(self, wheel_ctrl: WheelController):
        self.wheel_ctrl = wheel_ctrl

    def set_velocity_command(self, vel_x, vel_y=0.0, yaw_rate=0.0):
        self._cmd_vel_x = float(vel_x)
        self._cmd_vel_y = float(vel_y)
        self._cmd_yaw_rate = float(yaw_rate)
        self._yaw_locked = abs(yaw_rate) < 0.01

    def _compute_foot_world_pos(self, leg):
        data = self.data
        hip_q = data.qpos[self.qpos_addr[leg]['hip']]
        thigh_q = data.qpos[self.qpos_addr[leg]['thigh']]
        calf_q = data.qpos[self.qpos_addr[leg]['calf']]
        p_foot = forward_kinematics(hip_q, thigh_q, calf_q, leg=leg)
        base_pos = data.qpos[0:3].copy()
        base_quat = data.qpos[3:7].copy()
        w, x_val, y_val, z_val = base_quat
        yaw = np.arctan2(2.0 * (w * z_val + x_val * y_val),
                         1.0 - 2.0 * (y_val * y_val + z_val * z_val))
        R = rotation_matrix_y(yaw)
        return base_pos + R @ p_foot

    def _get_base_yaw(self):
        q = self.data.qpos[3:7]
        w, x_val, y_val, z_val = q
        return np.arctan2(2.0 * (w * z_val + x_val * y_val),
                          1.0 - 2.0 * (y_val * y_val + z_val * z_val))

    def _plan_foot_target(self, leg):
        base_pos = self.data.qpos[0:3].copy()
        base_quat = self.data.qpos[3:7].copy()
        w, x_val, y_val, z_val = base_quat
        yaw = np.arctan2(2.0 * (w * z_val + x_val * y_val),
                         1.0 - 2.0 * (y_val * y_val + z_val * z_val))

        target = None
        if self._stair_profile_loaded and self._stair_index < self.stair_geometry.num_steps:
            first_stair_x = self.stair_geometry.first_stair_x
            if self._stair_index == 0 and base_pos[0] < first_stair_x - 0.20:
                forward_dir = np.array([np.cos(yaw), np.sin(yaw), 0.0])
                current_foot = self._compute_foot_world_pos(leg)
                target = current_foot.copy()
                step_len = min(DEFAULT_TREAD_DEPTH, first_stair_x - base_pos[0] - 0.10)
                step_len = max(step_len, 0.05)
                target[0] += forward_dir[0] * step_len
                target[1] += forward_dir[1] * step_len
                target[2] = base_pos[2] + STANCE_FOOT_Z
            else:
                target = self.stair_geometry.tread_positions[self._stair_index].copy()
                target[2] += 0.005
        else:
            forward_dir = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            current_foot = self._compute_foot_world_pos(leg)
            target = current_foot.copy()
            target[0] += forward_dir[0] * DEFAULT_TREAD_DEPTH
            target[1] += forward_dir[1] * DEFAULT_TREAD_DEPTH
            target[2] = base_pos[2] + STANCE_FOOT_Z

        R_inv = rotation_matrix_y(-yaw)
        p_des_rel = R_inv @ (target - base_pos)
        p_des = np.array([p_des_rel[0], FOOT_Y_OFFSETS[leg], p_des_rel[2]])

        current_joints = (
            self.data.qpos[self.qpos_addr[leg]['hip']],
            self.data.qpos[self.qpos_addr[leg]['thigh']],
            self.data.qpos[self.qpos_addr[leg]['calf']],
        )
        result = simple_ik(p_des, leg, current_joints)
        if result is None:
            return None
        return target

    def _precompute_swing_trajectory(self, leg):
        state = self._leg_states[leg]
        start_pos = state.foot_start_pos.copy()
        target_pos = state.foot_target_pos.copy()
        traj_positions = SwingTrajectoryGenerator.generate_positions(
            start_pos, target_pos, SWING_HEIGHT, n_steps=12
        )
        start_joints = (
            self.data.qpos[self.qpos_addr[leg]['hip']],
            self.data.qpos[self.qpos_addr[leg]['thigh']],
            self.data.qpos[self.qpos_addr[leg]['calf']],
        )
        base_pos = self.data.qpos[0:3].copy()
        yaw = self._get_base_yaw()
        joint_targets = SwingTrajectoryGenerator.trajectory_to_joint_targets(
            leg, traj_positions, start_joints, base_pos, yaw
        )
        state.swing_trajectory = joint_targets
        state.traj_index = 0

    def _select_next_leg(self):
        return self._step_sequence[self._seq_index]

    def _advance_sequence(self):
        self._seq_index = (self._seq_index + 1) % len(self._step_sequence)

    def _enter_body_lift(self, leg, data):
        state = self._leg_states[leg]
        state.phase = 'body_lift'
        state.phase_timer = 0.0
        state.phase_duration = 0.40

        if self._stair_profile_loaded and self._stair_index < self.stair_geometry.num_steps:
            target_tread_z = self.stair_geometry.tread_positions[self._stair_index][2]
            base_z = data.qpos[2]
            target_z = base_z + 0.08
            target_z = min(target_z, target_tread_z + 0.25)
            self._target_base_z = target_z
        else:
            self._target_base_z = data.qpos[2] + 0.05

        state.swing_trajectory = None
        state.traj_index = 0

    def _enter_lift(self, leg, data):
        state = self._leg_states[leg]
        state.phase = 'lift'
        state.phase_timer = 0.0
        state.phase_duration = SWING_DURATION * 0.4
        state.foot_start_pos = self._compute_foot_world_pos(leg).copy()

        target = self._plan_foot_target(leg)
        if target is None:
            state.phase = 'stance'
            state.joint_targets = STAND_POSE[leg]
            return False

        state.foot_target_pos = target
        self._precompute_swing_trajectory(leg)
        return True

    def _enter_stance(self, leg, data):
        state = self._leg_states[leg]
        state.phase = 'stance'
        state.phase_timer = 0.0
        state.phase_duration = STANCE_DURATION
        state.in_contact = True
        state.step_count += 1
        self._total_steps += 1

        current_joints = (
            data.qpos[self.qpos_addr[leg]['hip']],
            data.qpos[self.qpos_addr[leg]['thigh']],
            data.qpos[self.qpos_addr[leg]['calf']],
        )
        state.joint_targets = current_joints
        state.swing_trajectory = None
        state.traj_index = 0

        if self._stair_profile_loaded and self._stair_index < self.stair_geometry.num_steps:
            self._stair_index += 1
            if self._stair_index >= self.stair_geometry.num_steps:
                self._stairs_completed += 1
        self._advance_sequence()

    def _update_gait(self, data, dt):
        eff_dt = dt * self._n_substeps
        current_leg = self._select_next_leg()
        state = self._leg_states[current_leg]

        base_pos_w = data.qpos[0:3]
        near_stair = True
        if self._stair_profile_loaded:
            first_stair_x = self.stair_geometry.first_stair_x
            if base_pos_w[0] < first_stair_x - 0.20:
                near_stair = False

        if state.phase == 'stance':
            state.phase_timer += eff_dt
            if state.phase_timer >= state.phase_duration and near_stair:
                self._enter_body_lift(current_leg, data)

        elif state.phase == 'body_lift':
            state.phase_timer += eff_dt
            if state.phase_timer >= state.phase_duration:
                if not self._enter_lift(current_leg, data):
                    self._advance_sequence()

        elif state.phase in ('lift', 'swing', 'descend', 'landing'):
            state.phase_timer += eff_dt
            if state.swing_trajectory is not None:
                traj_progress = state.phase_timer / state.phase_duration
                state.traj_index = int(
                    traj_progress * (len(state.swing_trajectory) - 1)
                )
                state.traj_index = min(
                    state.traj_index, len(state.swing_trajectory) - 1
                )
            if state.phase_timer >= state.phase_duration:
                self._enter_stance(current_leg, data)

        for leg in LEGS:
            if leg == current_leg:
                continue
            ls = self._leg_states[leg]
            if ls.phase != 'stance':
                ls.phase = 'stance'
                ls.swing_trajectory = None
                ls.traj_index = 0

    def compute_torques(self, model, data):
        dt = model.opt.timestep
        gravity = data.qfrc_bias.copy()
        base_pos = data.qpos[0:3].copy()
        self.data = data
        self.model = model

        self._cmd_yaw_rate = 0.0

        self._update_gait(data, dt)

        self.stabilizer.capture_initial_state(data)
        z_err = self.stabilizer.get_height_error(base_pos)
        (z_offset, pitch_offset, roll_offset,
         yaw_correction, roll_correction, pitch_correction) = (
             self.stabilizer.compute_body_corrections(data)
         )

        joint_targets = {}
        current_leg = self._select_next_leg()

        for leg in LEGS:
            state = self._leg_states[leg]
            if state.phase == 'stance':
                targets = state.joint_targets
                if self._stair_profile_loaded and self._stair_index > 0:
                    targets = self.stabilizer.compute_adaptive_stance(
                        leg, z_err, targets
                    )
                    hip_off = HIP_OFFSETS[leg]
                    side = 1.0 if hip_off[1] > 0 else -1.0
                    front = 1.0 if hip_off[0] > 0 else -1.0
                    pitch_calf_adj = pitch_correction * front * 0.10
                    roll_calf_adj = roll_correction * side * 0.10
                    calf_adj = np.clip(pitch_calf_adj + roll_calf_adj,
                                       -self.stabilizer.max_adapt_rad,
                                       self.stabilizer.max_adapt_rad)
                    targets = (targets[0], targets[1], targets[2] + calf_adj)
            else:
                if state.swing_trajectory is not None and state.traj_index < len(state.swing_trajectory):
                    idx = min(state.traj_index, len(state.swing_trajectory) - 1)
                    targets = state.swing_trajectory[idx]
                else:
                    targets = STAND_POSE[leg]
            joint_targets[leg] = targets

        leg_torques = super().compute_leg_torques(data, gravity, joint_targets)

        wheel_torques = np.zeros(model.nu)
        current_state = self._leg_states[self._select_next_leg()]

        if self._stair_profile_loaded and self._stair_index == 0:
            base_pos_w = data.qpos[0:3]
            first_stair_x = self.stair_geometry.first_stair_x
            if base_pos_w[0] < first_stair_x - 0.20:
                target_wheel_vel = self._approach_speed / WHEEL_RADIUS
            else:
                target_wheel_vel = 0.0
        elif self._stair_profile_loaded and self._stair_index > 0:
            target_wheel_vel = 0.0
        elif current_state.phase != 'stance':
            target_wheel_vel = SWING_HEIGHT * 8.0 * np.sign(self._cmd_vel_x)
        else:
            target_wheel_vel = 0.0

        KP_WHEEL = 5.0
        dt = model.opt.timestep
        for leg in LEGS:
            act_idx = ACTUATOR_MAP[leg]['wheel']
            df = self.dof_addr[leg]['wheel']
            cv = data.qvel[df]
            target_wheel_vel = ramp_velocity(
                self._wheel_vel_cmd, target_wheel_vel, dt,
                accel_limit=WHEEL_ACCEL_LIMIT
            )
            self._wheel_vel_cmd = target_wheel_vel
            torque = KP_WHEEL * (target_wheel_vel - cv)
            torque = np.clip(torque, -WHEEL_EFFORT_LIMIT, WHEEL_EFFORT_LIMIT)
            wheel_torques[act_idx] = torque

        return leg_torques + wheel_torques

    def reset(self):
        self._seq_index = 0
        self._stair_index = 0
        self._stairs_completed = 0
        self._step_count = 0
        self._total_steps = 0
        self._stair_climb_count = 0
        self._wheel_vel_cmd = 0.0
        self._leg_states = {leg: LegState(leg) for leg in LEGS}
        self._prev_joint_targets = {leg: STAND_POSE[leg] for leg in LEGS}
        self._filtered_targets = {leg: STAND_POSE[leg] for leg in LEGS}
        self._target_yaw = None
        self._yaw_locked = False


SteppingController = SteppingMotion
