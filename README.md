# xterra_robots

Robot description assets — URDFs, MuJoCo scenes, meshes, and per-robot
configuration — for the xTerra robot family.

## How to use it

This repository is not a CMake package. Tools locate models through
`XTERRA_ROBOTS_PATH`.

```bash
export XTERRA_ROBOTS_PATH=/path/to/xterra_robots
```

Example paths:

```text
$XTERRA_ROBOTS_PATH/models/dhav1_description/urdf/dhav1_description.urdf
$XTERRA_ROBOTS_PATH/models/dhav1_description/mujoco/scene.xml
$XTERRA_ROBOTS_PATH/models/dhav1w_description/config/dhav1w.yaml
```

## Available model directories

- `models/dhav1_description/` — 12-motor quadruped
- `models/dhav1w_description/` — 16-motor quadruped + foot-wheels (branch `feat/dhav1w`)
- `models/m2_metal_description/`

## Expected layout

Each robot directory follows this structure:

```text
models/<robot_name>_description/
├── urdf/                         # URDF and any RL-friendly variants
├── mujoco/                       # MuJoCo scene + body XML
├── meshes/                       # STL collision/visual meshes
└── config/<robot_name>.yaml      # canonical robot config (optional)
```

The `config/<robot_name>.yaml` file follows the schema in
[`robot_config_template.yaml`](robot_config_template.yaml) at the repo root —
a single source of truth for poses (sleep, stand), per-group joint limits,
inertial properties, actuator gains, RL/MBC tuning, safety thresholds, and
the hardware motor map. Keeping configuration alongside the description
means a robot's URDF, meshes, and tuning travel as one bundle.

What's actually checked in today:

- `dhav1_description`: `urdf/`, `mujoco/`, `meshes/`
- `dhav1w_description`: `urdf/`, `mujoco/`, `meshes/`, and
  `config/dhav1w.yaml` (canonical schema; calibrated sleep pose and
  spawn-hold gains)
- `m2_metal_description`: `urdf/`, `mujoco/`, `meshes/`
