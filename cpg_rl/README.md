# DHAV1W CPG-RL Locomotion Controller

## Project Overview

Complete CPG-based locomotion controller for the DHAV1W wheeled-legged quadruped robot in MuJoCo, with optional PPO-based Reinforcement Learning adaptation.

**Research Direction:** CPG-Based Gait Generation with Reinforcement Learning-Based Gait Adaptation for Wheeled-Legged Quadrupeds

**Model:** DHAV1W (DHAV 1.3 wheeled) - 16-actuator wheeled quadruped

**Project Directory:** `/Users/mananjain/Desktop/xTerra/dhav1w_cpg_rl/`

---

## Architecture

```
dhav1w_cpg_rl/
├── README.md
├── robot/
│   ├── __init__.py
│   └── TECHNICAL_REPORT.md       # Phase 0: Model inspection report
├── cpg/
│   ├── __init__.py
│   ├── hopf_oscillator.py        # Phase 1: Single Hopf oscillator
│   ├── coupled_cpg.py            # Phase 1: Four-leg CPG network
│   └── foot_trajectory.py        # Phase 2: Foot trajectory generation
├── kinematics/
│   ├── __init__.py
│   └── inverse_kinematics.py     # Phase 3: DHAV1W-specific IK
├── wheel_control/
│   ├── __init__.py
│   ├── wheel_controller.py       # Phase 4: Wheel control module
│   ├── slip_estimator.py         # Phase 4: Wheel slip estimation
│   └── wheel_constraints.py      # Phase 4: Wheel constraint enforcement
├── environment/
│   ├── __init__.py
│   ├── dhav1w_env.py             # Phase 6: Gymnasium RL environment
│   ├── terrain_generator.py      # Phase 10: Terrain generation
│   ├── domain_randomization.py   # Phase 10: Domain randomization
│   └── reward.py                 # Phase 7: Reward computation
├── controllers/
│   ├── __init__.py
│   ├── baseline_controller.py    # Phase 5: Controller A (Baseline)
│   ├── cpg_rl_controller.py      # Phase 7: Controller B (CPG-RL)
│   ├── wheel_rl_controller.py    # Phase 8: Controller C (Wheel-RL)
│   └── combined_controller.py    # Phase 9: Controller D (Combined)
├── environment/
│   ├── __init__.py
│   ├── dhav1w_env.py             # Phase 6: RL environment
│   ├── terrain_generator.py      # Phase 10: Terrain generation
│   ├── domain_randomization.py   # Phase 10: Domain randomization
│   └── reward.py                 # Phase 7: Reward computation
├── training/
│   ├── __init__.py
│   ├── train_ppo.py              # PPO training script
│   ├── evaluate_policy.py        # Evaluation script
│   ├── curriculum.py             # Phase 10: Curriculum learning
│   └── main_train.py             # Main training entry point
├── analysis/
│   ├── __init__.py
│   └── plot_results.py           # Phase 12: Visualization
├── experiments/
├── tests/
├── configs/
├── results/
│   ├── models/
│   │   ├── ppo_controller_D.zip      # Trained PPO model
│   │   ├── ppo_controller_D.pt       # PyTorch model weights
│   │   ├── vec_normalize_D.pkl       # VecNormalize stats
│   │   └── best_model.zip            # Best model during training
│   ├── plots/
│   │   ├── hopf_*.png                    # Phase 1 oscillator plots
│   │   ├── cpg_*.png                     # Phase 1/4 CPG plots
│   │   ├── foot_trajectories.png       # Phase 2 trajectory plots
│   │   ├── gait_cycle.png              # Phase 2 gait cycle
│   │   ├── ik_validation.png           # Phase 3 IK validation
│   │   ├── wheel_control.png           # Phase 4 wheel control
│   │   ├── wheel_velocities.png        # Phase 4 wheel velocities
│   │   └── *.png                       # Training/evaluation plots
│   └── logs/
│       └── rl/                       # Tensorboard logs
└── videos/
    └── dhav1w_walk.mp4               # Trained policy video
```

---

## Quick Start

### 1. Launch Robot in MuJoCo (No Control)

```bash
cd /Users/mananjain/Desktop/xTerra
./venv/bin/mjpython run.py
```

This loads the DHAV1W model and opens the MuJoCo viewer. The robot will fall under gravity (no control applied).

---

### 2. Run Baseline CPG Controller (Controller A)

```bash
cd /Users/mananjain/Desktop/xTerra/dhav1w_cpg_rl/controllers
./venv/bin/python -c "
import sys
sys.path.insert(0, '.')
from baseline_controller import BaselineController
import mujoco as mj

model = mj.MjModel.from_xml_path('/Users/mananjain/Desktop/xTerra/models/dhav1w_description/mujoco/scene.xml')
data = mj.MjData(model)

controller = BaselineController(model, data, gait_mode='trot', forward_vel=0.5)

# Run 100 steps
for _ in range(100):
    controller.step()
print('Baseline controller test complete!')
"
```

---

### 3. Train RL Policy (Controller D - Combined CPG-RL + Wheel-RL)

```bash
cd /Users/mananjain/Desktop/xTerra/dhav1w_cpg_rl/training
./venv/bin/python train_ppo.py \
    --controller D \
    --timesteps 500000 \
    --curriculum \
    --save-path results/models/ppo_controller_D.zip
```

Training options:
- `--controller`: A (Baseline), B (CPG-RL), C (Wheel-RL), D (Combined)
- `--timesteps`: Number of training timesteps (default: 500000)
- `--curriculum`: Enable curriculum learning (default: True)
- `--save-path`: Path to save trained model
- `--log-dir`: Tensorboard log directory

---

### 4. Test Trained Policy

```bash
cd /Users/mananjain/Desktop/xTerra/dhav1w_cpg_rl/training
./venv/bin/python train_ppo.py \
    --controller D \
    --test \
    --model results/models/ppo_controller_D.zip \
    --vec-norm results/models/vec_normalize_D.pkl \
    --duration 10
```

---

### 5. Generate Video of Trained Policy

```bash
cd /Users/mananjain/Desktop/xTerra/dhav1w_cpg_rl/training
./venv/bin/python main_train.py \
    --video \
    --model results/models/ppo_controller_D.zip \
    --vec-norm results/models/vec_normalize_D.pkl \
    --duration 10
```

Output: `dhav1w_cpg_rl/results/videos/dhav1w_walk.mp4`

---

### 6. Save Policy as .pt File

```bash
cd /Users/mananjain/Desktop/xTerra/dhav1w_cpg_rl/training
./venv/bin/python main_train.py \
    --save-pt \
    --model results/models/ppo_controller_D.zip
```

Output: `dhav1w_cpg_rl/results/models/ppo_controller_D.pt`

---

### 7. Controller Comparison (Ablation Study)

```bash
cd /Users/mananjain/Desktop/xTerra/dhav1w_cpg_rl/training
./venv/bin/python evaluate_policy.py \
    --compare \
    --duration 10 \
    --episodes 5 \
    --output-dir results/evaluation
```

---

## Controller Architectures

| Controller | Type | CPG | Wheel | Action Dim |
|------------|------|-----|-------|------------|
| A | Baseline | Fixed | Fixed | 0 (no RL) |
| B | CPG-RL | Adaptive | Fixed | 5 |
| C | Wheel-RL | Fixed | Adaptive | 4 |
| D | Combined | Adaptive | Adaptive | 9 |

---

## Observation Space (63 dims)

| Component | Dim | Description |
|-----------|-----|-------------|
| base_lin_vel | 3 | Base linear velocity (m/s) |
| base_ang_vel | 3 | Base angular velocity (rad/s) |
| base_quat | 4 | Base orientation quaternion |
| joint_pos_rel | 16 | Joint positions relative to stand |
| joint_vel | 16 | Joint velocities |
| wheel_vel | 4 | Wheel angular velocities |
| cpg_phase_sin | 4 | CPG phase sine components |
| cpg_phase_cos | 4 | CPG phase cosine components |
| contact | 4 | Foot contact states |
| cpg_params | 5 | Normalized CPG parameters |

---

## Action Space

| Controller | Dim | Actions |
|------------|-----|---------|
| B (CPG-RL) | 5 | [d_omega, d_swing_height, d_step_length, d_duty_factor, d_wheel_vel] |
| C (Wheel-RL) | 4 | [d_wheel_vel_fl, d_wheel_vel_fr, d_wheel_vel_rl, d_wheel_vel_rr] |
| D (Combined) | 9 | All of the above |

---

## Reward Function Components

| Component | Weight | Description |
|-----------|--------|-------------|
| forward_velocity | +5.0 | Reward for forward velocity tracking |
| upright | +2.0 | Reward for staying upright |
| roll_pitch | -5.0 | Penalty for roll/pitch deviations |
| height | -10.0 | Penalty for height deviation |
| torque | -0.001 | Energy penalty |
| action_smoothness | -0.1 | Penalty for action changes |
| lateral_vel | -2.0 | Penalty for lateral movement |
| fall | -100.0 | Penalty for falling |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO |
| Learning rate | 3e-4 |
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.01 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |
| net_arch | [256, 256] |

---

## Curriculum Learning Stages

| Stage | Terrain | Threshold |
|-------|---------|-----------|
| 0 | Flat | - |
| 1 | Uneven (small bumps) | > 100 reward |
| 2 | Slope (10°) | > 200 reward |
| 3 | Obstacles | > 300 reward |
| 4 | Stairs (8cm) | > 400 reward |
| 5 | Stairs + Slope | > 500 reward |

---

## Files Created

### Core Source Files
| File | Lines | Description |
|------|-------|-------------|
| `cpg/hopf_oscillator.py` | ~120 | Single Hopf oscillator |
| `cpg/coupled_cpg.py` | ~150 | Four-leg CPG network |
| `cpg/foot_trajectory.py` | ~180 | Foot trajectory generator |
| `kinematics/inverse_kinematics.py` | ~200 | DHAV1W-specific IK |
| `wheel_control/wheel_controller.py` | ~150 | Wheel control module |
| `wheel_control/wheel_constraints.py` | ~80 | Wheel constraints |
| `wheel_control/slip_estimator.py` | ~90 | Slip estimation |
| `environment/dhav1w_env.py` | ~620 | Gymnasium RL environment |
| `environment/terrain_generator.py` | ~120 | Terrain generation |
| `environment/domain_randomization.py` | ~90 | Domain randomization |
| `environment/reward.py` | ~100 | Reward computation |
| `controllers/baseline_controller.py` | ~150 | Controller A |
| `controllers/cpg_rl_controller.py` | ~200 | Controller B |
| `controllers/wheel_rl_controller.py` | ~200 | Controller C |
| `controllers/combined_controller.py` | ~250 | Controller D |
| `training/train_ppo.py` | ~200 | PPO training script |
| `training/evaluate_policy.py` | ~200 | Evaluation script |
| `training/curriculum.py` | ~80 | Curriculum manager |
| `training/main_train.py` | ~120 | Main entry point |
| `analysis/plot_results.py` | ~200 | Visualization |

### Key Output Files
| File | Description |
|------|-------------|
| `results/models/ppo_controller_D.zip` | Trained PPO model (Controller D) |
| `results/models/ppo_controller_D.pt` | PyTorch policy weights |
| `results/models/best_model.zip` | Best model during training |
| `results/models/vec_normalize_D.pkl` | Observation/reward normalization |
| `results/plots/*.png` | 21 validation and training plots |
| `results/logs/rl/` | Tensorboard logs |
| `results/videos/dhav1w_walk.mp4` | Trained policy video |

---

## Dependencies

| Package | Version |
|---------|---------|
| mujoco | >=2.3.0 |
| gymnasium | >=1.4.0 |
| stable-baselines3 | >=2.9.0 |
| numpy | >=1.24.0 |
| matplotlib | >=3.7.0 |
| scipy | >=1.10.0 |
| PyYAML | >=6.0 |
| torch | >=2.0.0 |
| imageio | >=2.0.0 |
| tensorboard | >=2.0.0 |

---

## References

- RL_learning.pdf: Motion Control of Wheeled-Legged Quadrupeds on Complex Terrains via Integrated Latent Encoding (2025)
- Hopf Oscillator: Standard CPG formulation (Ijspeert 2008)
- DHAV1W Model: models/dhav1w_description/mujoco/dhav1w.xml

---

## License

MIT License

---

## Acknowledgments

- XTER Robotics for the DHAV1W model
- MuJoCo physics engine
- Stable-Baselines3 RL framework