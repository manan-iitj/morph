# DHAV1W Technical Report — Phase 0: Repository Inspection

**Source of truth:** `models/dhav1w_description/mujoco/dhav1w.xml` and `urdf/dhav1w_description.urdf`  
**Project directory:** `/Users/mananjain/Desktop/xTerra/dhav1w_cpg_rl/`  
**Model XML:** `models/dhav1w_description/mujoco/dhav1w.xml`

---

## 1. Robot Joint Inventory

### 16 joints total, arranged as 4 legs × 4 joints per leg + 1 floating base

| Joint Name | Type | Axis | Range [rad] | Effort Limit [Nm] | Parent Link | Child Link |
|---|---|---|---|---|---|---|
| FL_hip_joint | revolute | (1,0,0) | [-0.8378, 0.8378] | 40 | base | FL_hip_link |
| FL_thigh_joint | revolute | (0,1,0) | [-1.7169, 2.5758] | 40 | FL_hip_link | FL_thigh_link |
| FL_calf_joint | revolute | (0,1,0) | [-2.800, -0.212] | 80 | FL_thigh_link | FL_calf_link |
| FL_wheel_joint | continuous | (0,1,0) | unconstrained | 14 | FL_calf_link | FL_wheel |
| FR_hip_joint | revolute | (1,0,0) | [-0.8378, 0.8378] | 40 | base | FR_hip_link |
| FR_thigh_joint | revolute | (0,1,0) | [-1.7169, 2.5758] | 40 | FR_hip_link | FR_thigh_link |
| FR_calf_joint | revolute | (0,1,0) | [-2.800, -0.212] | 80 | FR_thigh_link | FR_calf_link |
| FR_wheel_joint | continuous | (0,1,0) | unconstrained | 14 | FR_calf_link | FR_wheel |
| RL_hip_joint | revolute | (1,0,0) | [-0.8378, 0.8378] | 40 | base | RL_hip_link |
| RL_thigh_joint | revolute | (0,1,0) | [-1.7169, 2.5758] | 40 | RL_hip_link | RL_thigh_link |
| RL_calf_joint | revolute | (0,1,0) | [-2.800, -0.212] | 80 | RL_thigh_link | RL_calf_link |
| RL_wheel_joint | continuous | (0,1,0) | unconstrained | 14 | RL_calf_link | RL_wheel |
| RR_hip_joint | revolute | (1,0,0) | [-0.8378, 0.8378] | 40 | base | RR_hip_link |
| RR_thigh_joint | revolute | (0,1,0) | [-1.7169, 2.5758] | 40 | RR_hip_link | RR_thigh_link |
| RR_calf_joint | revolute | (0,1,0) | [-2.800, -0.212] | 80 | RR_thigh_link | RR_calf_link |
| RR_wheel_joint | continuous | (0,1,0) | unconstrained | 14 | RR_calf_link | RR_wheel |

Plus 1 floating base joint (`floating_base`, type `free`) connecting `base` to world.

---

## 2. Actuator Control Mode

**All 16 actuators are `<motor>` type = torque control.**  
No position servos exist in the XML. The RL policy must output torque commands.

| Actuator Index | Joint | Type | Ctrl Range [Nm] |
|---|---|---|---|
| 0 | FL_hip | motor | [-40, 40] |
| 1 | FL_thigh | motor | [-40, 40] |
| 2 | FL_calf | motor | [-80, 80] |
| 3 | FL_wheel | motor | [-14, 14] |
| 4 | FR_hip | motor | [-40, 40] |
| 5 | FR_thigh | motor | [-40, 40] |
| 6 | FR_calf | motor | [-80, 80] |
| 7 | FR_wheel | motor | [-14, 14] |
| 8 | RL_hip | motor | [-40, 40] |
| 9 | RL_thigh | motor | [-40, 40] |
| 10 | RL_calf | motor | [-80, 80] |
| 11 | RL_wheel | motor | [-14, 14] |
| 12 | RR_hip | motor | [-40, 40] |
| 13 | RR_thigh | motor | [-40, 40] |
| 14 | RR_calf | motor | [-80, 80] |
| 15 | RR_wheel | motor | [-14, 14] |

---

## 3. Leg-to-Joint Mapping (from config/dhav1w.yaml)

| Leg | Joint Indices | Joint Names |
|---|---|---|
| FL | [0, 1, 2, 3] | hip, thigh, calf, wheel |
| FR | [4, 5, 6, 7] | hip, thigh, calf, wheel |
| RL | [8, 9, 10, 11] | hip, thigh, calf, wheel |
| RR | [12, 13, 14, 15] | hip, thigh, calf, wheel |

---

## 4. Wheel-to-Actuator Mapping

| Wheel | Actuator Index | Joint Name | Radius [m] | Max Velocity [rad/s] | Max Torque [Nm] |
|---|---|---|---|---|---|
| FL | 3 | FL_wheel_joint | 0.077 | 73 | 14 |
| FR | 7 | FR_wheel_joint | 0.077 | 73 | 14 |
| RL | 11 | RL_wheel_joint | 0.077 | 73 | 14 |
| RR | 15 | RR_wheel_joint | 0.077 | 73 | 14 |

---

## 5. Available Observations and Sensors (56 total)

### Joint Sensors (32): jointpos + jointvel for all 16 joints
### Actuator Force Sensors (16): actuatorfrc for all 16 motors
### IMU Sensors (4): accelerometer, gyro, framequat (site `imu`)
### Base Frame Sensors (8): framepos, framelinvel, frameangvel, framelinacc (site `base`)

**Contact/foot sites (4):** `FL`, `FR`, `RL`, `RR` — sphere sites at wheel centers, size 0.025 m, group 5

---

## 6. Kinematic Chain Per Leg

```
base ── hip ── thigh ── calf ── wheel
```

Joint axes:
- Hip (abduction): (1, 0, 0) — rotation about X
- Thigh: (0, 1, 0) — rotation about Y
- Calf (knee): (0, 1, 0) — rotation about Y
- Wheel: (0, 1, 0) — rotation about Y

Link offsets:
- Hip → thigh: [0, ±0.09435, 0]
- Thigh → calf: [0, 0, -0.25]
- Calf → wheel: [0, ±0.04597, -0.22088]

Wheel radius: 0.077 m

---

## 7. Stand Pose (Keyframe)

| Leg | Hip | Thigh | Calf | Wheel |
|---|---|---|---|---|
| FL | 0.1 | 0.65 | -1.33 | 0 |
| FR | -0.1 | 0.65 | -1.33 | 0 |
| RL | 0.1 | 0.87 | -1.33 | 0 |
| RR | -0.1 | 0.87 | -1.33 | 0 |

Base height: 0.4341 m

---

## 8. Simulation Parameters

| Parameter | Value |
|---|---|
| Timestep | 0.0005 s |
| Integrator | RK4 |
| Gravity | (0, 0, -9.81) m/s² |
| Total bodies | 18 |
| Total joints | 17 |
| Total actuators | 16 |
| Total sensors | 56 |

---

## 9. Existing Simulation Architecture

| File | Purpose |
|---|---|
| `models/dhav1w_description/mujoco/dhav1w.xml` | Robot model (DO NOT modify geometry) |
| `models/dhav1w_description/mujoco/scene.xml` | Scene with floor only |
| `models/dhav1w_description/config/dhav1w.yaml` | Joint limits, poses, gains |
| `run.py` | Simple MuJoCo viewer launcher |

No IK, trajectory generation, or gait controller exists yet. All must be built from scratch.

---

## 10. Control Mode Determination

| Actuator Group | XML Tag | Control Mode |
|---|---|---|
| Hip (abduction) | `<motor>` | Torque |
| Thigh | `<motor>` | Torque |
| Calf (knee) | `<motor>` | Torque |
| Wheel | `<motor>` | Torque |

**All actuators use torque control.** The RL policy and CPG must output torque commands, which are applied via PD control or direct torque.
