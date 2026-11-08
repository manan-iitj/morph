# MORPH — Mobility Optimization through Reinforcement and Primitive-based Hybrid Control for Wheeled Quadrupeds

## Introduction

MORPH is a comprehensive framework for wheeled quadruped locomotion that combines reinforcement learning with primitive-based hybrid control. This repository provides the model descriptions, configuration, and control infrastructure needed to develop and deploy advanced locomotion policies for the dhav-hy wheeled quadruped platform.

## Core Philosophy

The MORPH approach integrates:
- **Reinforcement Learning** for adaptive policy optimization
- **Primitive-based hybrids** for structured, interpretable behavior
- **CPG (Central Pattern Generator)** networks for rhythmic movement generation
- **Wheel-control coordination** for legged-wheel locomotion fusion

## Model Assets

This repository contains description assets for the dhav-hy wheeled quadruped, including:
- URDF and MuJoCo simulation models
- Mesh files for collision and visual geometry
- Per-robot configuration files (joint limits, actuator gains, inertial properties)
- CPG network configurations and oscillator parameters

## Available Models

- **dhav1_description**: 12-motor quadruped base model
- **dhav1w_description**: 16-motor quadruped with foot-wheels
- **m2_metal_description**: Metal-framed variant

## Expected Layout

Each robot directory follows this structure:

```
models/<robot_name>_description/
├── urdf/                # URDF and RL-friendly variants
├── mujoco/              # MuJoCo scene and body XML
├── meshes/              # STL collision/visual meshes
└── config/<robot_name>.yaml  # Canonical robot configuration
```

The configuration file serves as a single source of truth for:
- Poses (sleep, stand, rest positions)
- Per-group joint limits
- Inertial properties
- Actuator gains and PWM maps
- RL/MBC tuning parameters
- Safety thresholds and hardware motor maps

## Getting Started

Model paths are located through an environment variable pointing to the repository root. Configure your simulation or policy framework to load models from the appropriate directory.

## Repository Structure

```
/dhav1w_cpg_rl/
├── README.md              # This file
├── robot/
│   ├── __init__.py
│   └── TECHNICAL_REPORT.md
├── cpg/
│   ├── hopf_oscillator.py
│   ├── coupled_cpg.py
│   └── foot_trajectory.py
├── kinematics/
│   └── inverse_kinematics.py
├── wheel_control/
│   ├── wheel_controller.py
│   ├── slip_estimator.py
│   └── wheel_constraints.py
├── environment/
│   ├── dhav1w_env.py
│   ├── terrain_generator.py
│   ├── domain_randomization.py
│   └── reward.py
├── controllers/
│   ├── baseline_controller.py
│   ├── cpg_rl_controller.py
│   ├── wheel_rl_controller.py
│   └── combined_controller.py
├── training/
│   ├── train_ppo.py
│   ├── evaluate_policy.py
│   ├── curriculum.py
│   └── main_train.py
├── analysis/
│   └── plot_results.py
├── configs/
├── results/
│   ├── models/
│   └── plots/
└── videos/
```

## Citation

If you use MORPH in your research, please cite the relevant papers and acknowledge the dhav-hy wheeled quadruped asset from the robotics company.

## License

MIT License — free for academic and research use.
