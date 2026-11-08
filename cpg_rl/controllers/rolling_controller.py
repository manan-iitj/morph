"""
Rolling motion controller for DHAV-HY — Pure wheel-driven locomotion.

Wheels provide primary propulsion; legs actively stabilize the robot
and maintain approximately constant base height.

No gait/CPG logic — pure stabilization of base height using PD control.
"""

import numpy as np
import mujoco as mj

from .base_controller import (
    BaseMotion, STAND_POSE, PD_GAINS, EFFORT_LIMITS,
    WHEEL_EFFORT_LIMIT, WHEEL_RADIUS, ACTUATOR_MAP, LEGS, JOINT_TYPES,
    STANCE_FOOT_Z, compute_differential_wheel_velocities, ramp_velocity,
    TRACK_WIDTH
)
from cpg_rl.wheel.wheel_controller import WheelController
from .stabilization import Stabilizer
from cpg_rl.kinematics.inverse_kinematics import HIP_OFFSETS


# Physical parameters (from the MuJoCo model)
WHEEL_RADIUS = 0.077  # m
MAX_VELOCITY = 200.0    # rad/s (7.7 m/s at r=0.077m)
MAX_TORQUE = 200.0      # Nm
WHEEL_ACCEL_LIMIT = 200.0  # rad/s^2 - increased for smoother ramp-up

# Auto-acceleration parameters: start small, ramp up automatically
# These can be modified at runtime via roll_ctrl.set_acceleration_params()
AUTO_ACCEL_START_VEL = 0.0      # m/s - start from complete stop
AUTO_ACCEL_RATE_DEFAULT = 0.2   # m/s^2 - slower, smoother acceleration
AUTO_ACCEL_MIN_VEL = 0.05       # minimum speed to maintain after reaching target

# Straight-line heading correction: P controller (strong enough to track straight)
YAW_CORRECTION_KP = 30.0      # rad/s per rad of yaw error
YAW_DEADBAND = 0.02           # rad - tiny deadband to prevent oscillation
MAX_YAW_CORRECTION = 10.0     # rad/s - hard safety cap

# Height control parameters - PROPER PD CONTROL
# e_z = z_desired - z_actual
# u_z = Kp_z * e_z - Kd_z * v_z
# These gains provide stable height control without jerkiness

# Height control gains
KP_HEIGHT = 5000.0     # height stiffness (N/m)
KD_HEIGHT = 500.0      # height damping (N·s/m) - moderate for stability

# Pitch control parameters
KP_PITCH = 15000.0       # base pitch stiffness (N/m/rad)
KP_PITCH_HIGH = 30000.0  # pitch stiffness at high speed (N/m/rad)
PITCH_DAMPING = 5000.0   # pitch rate damping (N·m·s/rad)

# Roll control parameters
KP_ROLL = 20000.0        # roll stiffness (N/m/rad)
KP_ROLL_HIGH = 40000.0   # roll stiffness at high speed (N/m/rad)
ROLL_DAMPING = 5000.0    # roll rate damping (N·m·s/rad)


class RollingMotion(BaseMotion):
    """Wheel-driven locomotion — pure rolling.

    Wheels ROLL at target forward velocity — primary propulsion.
    Legs provide active stabilization: base height, pitch, roll,
    and stable wheel-ground contact.

    NO gait/CPG logic — pure stabilization of base height.
    Height remains constant regardless of obstacles or speed.
    """

    def __init__(self, model, data, qpos_addr, dof_addr):
        super().__init__(model, data, qpos_addr, dof_addr)

        # --- Target forward velocity command (set externally) ---
        self._cmd_vel_x = 0.0
        self._cmd_vel_y = 0.0
        self._cmd_yaw_rate = 0.0
        self._target_yaw = None       # locked heading for straight-line travel
        self._yaw_locked = False

        # --- Auto-acceleration state ---
        self._target_vel_x = 0.0      # user-set target speed
        self._auto_vel_x = 0.0        # current auto-accelerating speed
        self._auto_accel_active = False
        self._auto_accel_rate = AUTO_ACCEL_RATE_DEFAULT  # configurable acceleration

        # --- Stabilizer for leg-based body stabilization ---
        # Fixed gains for consistent height control
        self.stabilizer = Stabilizer(
            model, data, qpos_addr, dof_addr,
            leg_adapt_kp=1500.0,
            max_adapt_rad=0.30,
            height_adapt_scale=0.0,
            ik_phase_bucket=0.10,
            height_adapt_deadband=0.05,
            kp_height=KP_HEIGHT,
            kp_pitch=KP_PITCH,
            kp_roll=KP_ROLL,
            kp_yaw=50000.0,
            yaw_damping=50000.0,
            friction_scale=1.0,
            disable_yaw=True,  # Zero yaw for straight-line rolling
        )

        # --- Wheel controller (velocity mode) ---
        self.wheel_ctrl = WheelController(
            wheel_radius=WHEEL_RADIUS,
            max_velocity=MAX_VELOCITY,
            max_torque=MAX_TORQUE,
            control_mode='velocity',
        )

        # --- Current wheel velocity commands (rad/s) per leg ---
        self._current_wheel_cmds = {leg: 0.0 for leg in LEGS}

        # --- Actual velocity tracking ---
        self._actual_vel_x = 0.0
        self._actual_vel_z = 0.0

        # --- Height control state ---
        # Target base height initialized from current position
        self._target_base_z = None
        self._height_error_prev = 0.0
        self._height_cmd_prev = 0.0

        # --- Smoothing state for joint targets ---
        self._prev_joint_targets = {leg: STAND_POSE[leg] for leg in LEGS}
        self._filtered_targets = {leg: STAND_POSE[leg] for leg in LEGS}

        # --- Terrain adaptation ---
        self._last_ground_contact = {leg: True for leg in LEGS}
        self._obstacle_clearance = 0.0
        # Obstacle detection per leg - height only changes when obstacle present
        self._obstacle_detected = {leg: False for leg in LEGS}

    def set_velocity_command(self, vel_x, vel_y=0.0, yaw_rate=0.0):
        """Set the driving velocity command.

        Starts auto-acceleration from small speed toward the target.
        When yaw_rate is 0, heading is locked for straight-line travel.

        Parameters
        ----------
        vel_x : float
            Desired forward linear velocity (m/s).
        vel_y : float
            Desired lateral velocity (m/s). Default 0.0.
        yaw_rate : float
            Desired yaw rate (rad/s). Positive = turn left.
            If 0, heading is locked for straight-line travel.
        """
        self._target_vel_x = float(vel_x)
        self._cmd_vel_y = float(vel_y)
        self._cmd_yaw_rate = float(yaw_rate)
        self._auto_accel_active = True
        # Start from small speed for smooth acceleration
        self._auto_vel_x = AUTO_ACCEL_START_VEL
        # Lock heading when yaw_rate is 0
        if abs(yaw_rate) < 0.01:
            self._yaw_locked = True
            self._target_yaw = None  # captured in compute_torques
        else:
            self._yaw_locked = False
            self._target_yaw = None

    def set_target_velocity(self, velocity_m_s, vel_y=0.0, yaw_rate=0.0,
                            auto_progression=False):
        """Set the desired linear velocity (m/s) for the robot.

        Starts auto-acceleration from small speed toward the target.

        Parameters
        ----------
        velocity_m_s : float
            Desired forward linear velocity in m/s.
        vel_y : float
            Desired lateral velocity (m/s). Default 0.0.
        yaw_rate : float
            Desired yaw rate (rad/s). Positive = turn left.
        auto_progression : bool
            If True, automatically progress through speeds (NOT USED:
            height stabilization is speed-independent).
        """
        self._target_vel_x = float(velocity_m_s)
        self._cmd_vel_y = float(vel_y)
        self._cmd_yaw_rate = float(yaw_rate)
        self._auto_accel_active = True
        # Start from small speed for smooth acceleration
        self._auto_vel_x = AUTO_ACCEL_START_VEL
        # auto_progression not used — speed is user-controlled, height
        # stabilization works independently of forward velocity

    def set_acceleration_params(self, start_vel=0.2, accel_rate=0.3, min_vel=0.1):
        """Configure auto-acceleration parameters at runtime.

        Parameters
        ----------
        start_vel : float
            Starting velocity (m/s) when acceleration begins.
        accel_rate : float
            Acceleration rate (m/s^2) — how fast to ramp to target speed.
        min_vel : float
            Minimum velocity to maintain after reaching target.
        """
        global AUTO_ACCEL_START_VEL, AUTO_ACCEL_RATE_DEFAULT, AUTO_ACCEL_MIN_VEL
        AUTO_ACCEL_START_VEL = float(start_vel)
        AUTO_ACCEL_RATE_DEFAULT = float(accel_rate)
        self._auto_accel_rate = float(accel_rate)
        AUTO_ACCEL_MIN_VEL = float(min_vel)


    # --- Velocity progression removed: pure height stabilization ---
    # No _advance_velocity_progression method — speed is user-set,
    # not auto-progressed through waypoints.

    def _compute_height_control(self, base_pos, base_vel):
        """Compute height control output using PD control.

        e_z = z_desired - z_actual
        u_z = Kp_z * e_z - Kd_z * v_z

        Returns the desired vertical correction in radians
        to be distributed across leg kinematics.

        Key: target_base_z is INITIALIZED ONCE from the first measured
        height and then kept constant — the robot maintains its initial
        standing height throughout locomotion, regardless of obstacles.
        """
        # Initialize target base z ONCE from first measurement
        if self._target_base_z is None:
            self._target_base_z = base_pos[2]
            # Store as constant target — height stabilization is not
            # dynamically adapted; the robot maintains its initial height

        # Fixed target: maintain the initially measured base height
        # DO NOT adapt target based on obstacles or terrain changes
        e_z = self._target_base_z - base_pos[2]
        u_z = KP_HEIGHT * e_z - KD_HEIGHT * base_vel[2]

        # Low-pass filter to avoid abrupt changes but maintain stability
        u_z = 0.8 * self._height_cmd_prev + 0.2 * u_z
        self._height_cmd_prev = u_z

        # Clamp the correction to physically meaningful range
        u_z = np.clip(u_z, -0.3, 0.3)

        return u_z

    def compute_torques(self, model, data):
        """Compute torques for all leg and wheel actuators.

        Called by the MuJoCo simulation loop in mujoco_viz.py.

        Key differences from original:
        - Auto-acceleration: starts small, ramps up to target
        - Height target FIXED at initial measurement when no obstacle
        - Leg heights ONLY change when obstacles are detected
        - Front-weight bias at high speeds to balance wheel torque
        """
        dt = model.opt.timestep

        # --- Auto-acceleration: ramp speed from small toward target ---
        if self._auto_accel_active:
            speed_diff = self._target_vel_x - self._auto_vel_x
            if abs(speed_diff) > 0.01:
                self._auto_vel_x += self._auto_accel_rate * dt * np.sign(speed_diff)
                self._auto_vel_x = np.clip(
                    self._auto_vel_x,
                    -abs(self._target_vel_x),
                    abs(self._target_vel_x)
                )
            else:
                self._auto_vel_x = self._target_vel_x
        else:
            self._auto_vel_x = self._target_vel_x

        # Use auto-accelerating speed for wheel commands
        self._cmd_vel_x = self._auto_vel_x

        # --- Front-weight bias at high speeds (>3 m/s) ---
        speed = abs(self._cmd_vel_x)
        front_bias = 0.0
        if speed > 3.0:
            front_bias = 0.05 * min(1.0, (speed - 3.0) / 2.0)
        # Low-pass filter front-bias to avoid abrupt changes
        self._front_bias_prev = front_bias if not hasattr(self, '_front_bias_prev') else (
            0.7 * self._front_bias_prev + 0.3 * front_bias
        )
        front_bias = self._front_bias_prev

        # --- Advance simulation state ---
        mj.mj_forward(model, data)

        # --- Capture base state ---
        gravity = data.qfrc_bias.copy()
        base_pos = data.qpos[0:3]
        base_vel = data.qvel[0:3]
        self._actual_vel_x = float(base_vel[0])
        self._actual_vel_z = float(base_vel[2])

        # --- Yaw: completely zero for rolling gait ---
        self._cmd_yaw_rate = 0.0
        self._target_yaw = None
        self._yaw_locked = False

        # --- Obstacle detection: check contact forces on wheel geoms ---
        # Only adjust leg heights when obstacles are actually present
        self._obstacle_detected = {leg: False for leg in LEGS}
        n_contact = 0
        for leg in LEGS:
            # Check wheel contact force - high normal force indicates obstacle
            qp = self.qpos_addr[leg]['wheel']
            ca = self.data.qpos[qp]
            cv = self.data.qvel[self.dof_addr[leg]['wheel']]
            # Detect obstacle: wheel pushing into ground with significant force
            # or wheel position below normal stance
            if cv < -1.0 and ca < -0.5:
                self._obstacle_detected[leg] = True
                n_contact += 1

        # --- Height control: ONLY adjust when obstacle detected ---
        if n_contact > 0:
            # Obstacle present: adjust height to clear it
            u_z = self._compute_height_control(base_pos, base_vel)
        else:
            # No obstacle: maintain constant height, no adjustment
            u_z = 0.0

        # --- Apply front-weight bias to height control ---
        u_z += front_bias * 0.15
        u_z = np.clip(u_z, -0.3, 0.3)

        # --- Wheel velocities from differential drive ---
        target_cmds = compute_differential_wheel_velocities(
            self._cmd_vel_x, self._cmd_yaw_rate, track_width=TRACK_WIDTH
        )

        # --- Smooth wheel commands with acceleration ramping ---
        ramped_cmds = {}
        for leg in LEGS:
            ramped_cmds[leg] = ramp_velocity(
                self._current_wheel_cmds[leg], target_cmds[leg], dt
            )
        self._current_wheel_cmds = ramped_cmds

        # --- Low-pass smoothing for wheel velocity commands ---
        if not hasattr(self, '_wheel_cmd_smoothed_prev'):
            self._wheel_cmd_smoothed_prev = {leg: 0.0 for leg in LEGS}
        smoothed_cmds = {}
        alpha = 0.5  # smoothing factor - increased for smoother motion
        for leg in LEGS:
            prev = self._wheel_cmd_smoothed_prev[leg]
            smoothed_cmds[leg] = (
                alpha * self._current_wheel_cmds[leg] +
                (1.0 - alpha) * prev
            )
            self._wheel_cmd_smoothed_prev[leg] = smoothed_cmds[leg]
        self._current_wheel_cmds = smoothed_cmds

        # --- Apply wheel velocity commands ---
        self.wheel_ctrl.set_velocities(self._current_wheel_cmds)

        # --- Stabilizer: body height, pitch, roll, yaw corrections ---
        self.stabilizer.capture_initial_state(self.data)
        z_err = self.stabilizer.get_height_error(base_pos)
        (z_offset, pitch_offset, roll_offset,
         yaw_correction, roll_correction, pitch_correction) = (
             self.stabilizer.compute_body_corrections(self.data)
         )

        # --- Zero out yaw correction for rolling gait ---
        yaw_correction = 0.0

# --- Compute leg joint targets ---
        # Rolling: legs stay near stand pose. Height adaptation ONLY when
        # obstacle detected. NO pitch/roll body leveling when clear.
        base_quat = data.qpos[3:7]
        joint_targets = {}
        for leg in LEGS:
            targets = self._get_stance_target(
                leg, u_z, z_err, 0.0,                          # z_offset=0
                0.0, 0.0, 0.0, 0.0, 0.0,                      # pitch/roll/yaw corrections = 0
                base_pos, base_quat,                            # base_pos, base_quat
                obstacle=self._obstacle_detected[leg]          # only height adapts
            )
            targets = self._smooth_actions(leg, targets)
            joint_targets[leg] = targets

        # --- Compute PD + gravity torques for leg actuators ---
        leg_torques = super().compute_leg_torques(self.data, gravity, joint_targets)

        # --- Compute wheel torques ---
        wheel_torques = np.zeros(model.nu)
        # Direct wheel velocity control: common torque for all wheels
        # (prevents yaw from per-wheel velocity differences)
        target_wheel_vel = self._cmd_vel_x / WHEEL_RADIUS  # Convert m/s to rad/s
        KP_WHEEL = 5.0  # rad/s per Nm - gentle acceleration
        
        # Compute average wheel velocity error (same for all wheels)
        avg_cv = 0.0
        for leg in LEGS:
            df = self.dof_addr[leg]['wheel']
            avg_cv += data.qvel[df]
        avg_cv /= len(LEGS)
        
        common_torque = KP_WHEEL * (target_wheel_vel - avg_cv)
        common_torque = np.clip(common_torque, -WHEEL_EFFORT_LIMIT, WHEEL_EFFORT_LIMIT)
        
        # Yaw rate damping: actively counter any yaw rotation to keep yaw = 0
        # Get yaw rate from base angular velocity
        yaw_rate = data.qvel[5]
        YAW_DAMPING_KP = 80.0  # N*m/(rad/s) - counter-torque per yaw rate
        YAW_POSITION_KP = 50.0  # N*m/rad - proportional term to drive yaw to zero
        
        # Get current yaw
        w, x_val, y_val, z_val = data.qpos[3:7]
        current_yaw = np.arctan2(
            2.0 * (w * z_val + x_val * y_val),
            1.0 - 2.0 * (y_val * y_val + z_val * z_val)
        )
        
        # Combined yaw counter-torque: proportional to yaw error + rate damping
        yaw_counter_torque = -YAW_POSITION_KP * current_yaw - YAW_DAMPING_KP * yaw_rate
        
        # Apply yaw counter-torque differentially: left wheels -, right wheels +
        # This creates a torque about Z axis to counter yaw rotation
        for i, leg in enumerate(LEGS):
            act_idx = ACTUATOR_MAP[leg]['wheel']
            side = -1.0 if HIP_OFFSETS[leg][1] > 0 else 1.0  # left=+1, right=-1
            wheel_torques[act_idx] = common_torque + yaw_counter_torque * side * 0.5

        return leg_torques + wheel_torques

    def _get_stance_target(self, leg, u_z, z_err, z_offset, pitch_offset, roll_offset,
                           yaw_correction, roll_correction, pitch_correction,
                           base_pos, base_quat, obstacle=False):
        """Stance target: foot fixed at stand pose + height adaptation.

        Leg heights remain CONSTANT when no obstacle is present.
        Heights ONLY change when obstacle is detected to clear it.

        u_z: vertical PD control output (radians) - only applied when obstacle=True.
        obstacle: bool - if True, apply height adjustments to clear obstacle.
        """
        hip_off = HIP_OFFSETS[leg]
        stand_angles = STAND_POSE[leg]

        targets = list(stand_angles)

        # Height adaptation: ONLY when obstacle detected
        if obstacle:
            # Distribute u_z across legs to clear obstacle
            front_factor = 1.0 if hip_off[0] > 0 else -1.0
            leg_height_adj = u_z * 0.25 * front_factor
            leg_height_adj = np.clip(leg_height_adj,
                                      -self.stabilizer.max_adapt_rad,
                                      self.stabilizer.max_adapt_rad)
            targets[2] += leg_height_adj

        # Body leveling offsets: ONLY apply when obstacle detected
        # (when clearing obstacles, legs adjust height; when clear, legs stay at stand)
        if obstacle:
            # Body leveling offsets - apply for stability during rolling
            pitch_term = -pitch_offset * hip_off[0] * 0.4
            roll_term = -roll_offset * hip_off[1] * 0.4
            z_adj = pitch_term + roll_term

            calf_adj = z_adj * 0.6
            calf_adj = np.clip(calf_adj,
                               -self.stabilizer.max_adapt_rad,
                               self.stabilizer.max_adapt_rad)

            targets[2] += calf_adj

        # Orientation corrections — keep body planted while wheels roll
        side = 1.0 if hip_off[1] > 0 else -1.0
        front = 1.0 if hip_off[0] > 0 else -1.0

        # Apply orientation corrections always (not just when obstacle detected)
        yaw_calf_adj = yaw_correction * side * 0.20
        roll_calf_adj = roll_correction * side * 0.10
        pitch_calf_adj = pitch_correction * front * 0.10

        total_calf_adj = yaw_calf_adj + roll_calf_adj + pitch_calf_adj
        total_calf_adj = np.clip(total_calf_adj,
                                  -self.stabilizer.max_adapt_rad,
                                  self.stabilizer.max_adapt_rad)

        targets = (targets[0], targets[1], targets[2] + total_calf_adj)

        return targets

    def _smooth_actions(self, leg, new_targets):
        """Smooth joint targets: velocity clamping + low-pass filter."""
        prev = self._prev_joint_targets[leg]
        clamped = []
        max_change = 30.0 * self.model.opt.timestep  # rad/s clamping
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
            filtered_val = (0.40 * clamped[k] +
                            (1.0 - 0.40) * filtered_prev[k])
            filtered.append(filtered_val)
        filtered = tuple(filtered)

        self._prev_joint_targets[leg] = filtered
        self._filtered_targets[leg] = filtered

        return filtered

    def reset(self):
        """Reset controller state."""
        self._current_wheel_cmds = {leg: 0.0 for leg in LEGS}
        self._cmd_vel_x = 0.0
        self._cmd_vel_y = 0.0
        self._cmd_yaw_rate = 0.0
        self._target_vel_x = 0.0
        self._auto_vel_x = 0.0
        self._auto_accel_active = False
        self._actual_vel_x = 0.0
        self._actual_vel_z = 0.0
        self._target_base_z = None
        self._height_error_prev = 0.0
        self._height_cmd_prev = 0.0
        self._prev_joint_targets = {leg: STAND_POSE[leg] for leg in LEGS}
        self._filtered_targets = {leg: STAND_POSE[leg] for leg in LEGS}
        self._last_ground_contact = {leg: True for leg in LEGS}
        self._obstacle_clearance = 0.0
        self._target_yaw = None
        self._yaw_locked = False
        self._yaw_integral = 0.0
        self._y_init = None


# Backwards-compatibility alias
RollingController = RollingMotion