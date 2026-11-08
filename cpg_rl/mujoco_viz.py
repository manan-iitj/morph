#!/usr/bin/env python3
"""
Dhav-Hy Quadruped Viewer — Live MuJoCo Viewer.

Plug-and-play gait selector. Choose gait via command-line argument:

    ./venv/bin/mjpython mujoco_viz.py --gait roll   (default)
    ./venv/bin/mjpython mujoco_viz.py --gait trot
    ./venv/bin/mjpython mujoco_viz.py --gait step

Optionally load terrain:
    ./venv/bin/mjpython mujoco_viz.py --gait step --terrain stairs.xml

Close the viewer window to stop.
"""

import sys
import os
import argparse
import time
import numpy as np
import mujoco as mj
import mujoco.viewer
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

# Physics substeps per controller update for real-time performance
N_SUBSTEPS = 1


def extract_geoms_from_terrain(filepath):
    """Extract <geom> and <material> tags from a terrain XML file.

    Terrain XML files are complete MuJoCo documents. We only need the
    <geom> elements (to insert into worldbody) and <material> elements
    (to insert into asset) from them.
    """
    if not os.path.exists(filepath):
        return "", ""
    with open(filepath, 'r') as f:
        content = f.read()
    # Extract all <geom ... /> tags, exclude ground (scene.xml has one)
    geom_pattern = re.compile(r'<geom[^>]*/?>')
    geoms = [g for g in geom_pattern.findall(content)
             if 'name="ground"' not in g]

    # Extract material definitions
    mat_pattern = re.compile(r'<material[^>]*/?>')
    materials = mat_pattern.findall(content)

    return '\n'.join(geoms), '\n'.join(materials)


def create_combined_scene_xml(robot_xml_path, terrain_filepath, output_path):
    """Create a combined MuJoCo XML with robot model + terrain geoms.

    Reads the robot scene XML and inserts terrain geoms + materials
    into it, producing a complete loadable XML file.
    """
    with open(robot_xml_path, 'r') as f:
        robot_content = f.read()

    # Extract geoms and materials from terrain file
    terrain_geoms, terrain_materials = extract_geoms_from_terrain(terrain_filepath)

    if not terrain_geoms:
        # No terrain to add, just use the robot XML as-is
        with open(output_path, 'w') as f:
            f.write(robot_content)
        return output_path

    # Insert materials into <asset> section
    if terrain_materials and '</asset>' in robot_content:
        mats_lines = terrain_materials.split('\n')
        mats_str = '\n        '.join(mats_lines)
        robot_content = robot_content.replace(
            '</asset>',
            mats_str + '\n        </asset>'
        )

    # Insert geoms before </worldbody>, replacing step placeholders
    if '</worldbody>' in robot_content:
        robot_content = robot_content.replace(
            '<!--step test2-->',
            terrain_geoms + '\n  <!-- Terrain from step_controller -->'
        )
        robot_content = robot_content.replace(
            '<!--step test3-->', '<!-- Removed for terrain -->'
        )

    with open(output_path, 'w') as f:
        f.write(robot_content)

    return output_path


from controllers import (
    RollingMotion, TrotMotion, SteppingMotion,
    setup_joint_addresses
)

XML_PATH = os.path.join(
    _ROOT,
    'models', 'dhav1w_description', 'mujoco', 'scene.xml'
)

STAND_JOINT_VALUES = [
    ('FL', 0.1,  0.70, -1.33, 0.0),
    ('FR', -0.1, 0.70, -1.33, 0.0),
    ('RL', 0.1,  0.70, -1.33, 0.0),
    ('RR', -0.1, 0.70, -1.33, 0.0),
]


def set_stand_pose(data):
    """Set robot to standing pose."""
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    qpos[0:3] = [0.0, 0.0, 0.4341]
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    qvel[0:6] = 0.0
    for i, (leg, hip, thigh, calf, wheel) in enumerate(STAND_JOINT_VALUES):
        qpos[7 + i * 4] = hip
        qpos[7 + i * 4 + 1] = thigh
        qpos[7 + i * 4 + 2] = calf
        qpos[7 + i * 4 + 3] = wheel
    qvel[6:] = 0.0
    data.qpos = qpos
    data.qvel = qvel


def parse_terrain_xml(terrain_path):
    """Parse terrain XML file to extract obstacle/stair positions.

    Returns
    -------
    obstacles : list of tuples
        Each is (x, y, z_center, sx, sy, sz) representing box center
        and half-sizes.
    """
    import re as re_mod

    if not os.path.exists(terrain_path):
        return [], []

    with open(terrain_path, 'r') as f:
        content = f.read()

    obstacles = []

    # Find all <geom> tags with box type
    geom_pattern = re_mod.compile(
        r'<geom\s+name="([^"]*)"\s+type="box"[^>]*?'
        r'size="([^"]*)"\s+pos="([^"]*)"[^>]*?'
    )

    for match in geom_pattern.finditer(content):
        name = match.group(1)
        size_str = match.group(2)
        pos_str = match.group(3)

        try:
            sx, sy, sz = [float(v) for v in size_str.split()]
            x, y, z = [float(v) for v in pos_str.split()]
            obstacles.append((x, y, z, sx, sy, sz))
        except ValueError:
            continue

    return obstacles, []


def make_query_fns(obstacles, stairs):
    """Create query functions from terrain data.

    Returns
    -------
    query_fn : callable
        (x, y) -> z ground height at world position.
    obstacle_fn : callable
        (x, y) -> bool, True if obstacle present at (x, y).
    """
    def query_fn(x, y):
        """Return ground height at world position (x, y)."""
        z = 0.0  # Base ground level

        # Check obstacle boxes
        for ox, oy, oz, ox_s, oy_s, oz_s in obstacles:
            if abs(x - ox) <= ox_s and abs(y - oy) <= oy_s:
                z = max(z, oz + oz_s)  # Top of obstacle

        return z

    def obstacle_fn(x, y):
        """Return True if obstacle present at (x, y)."""
        for ox, oy, oz, ox_s, oy_s, oz_s in obstacles:
            if abs(x - ox) <= ox_s and abs(y - oy) <= oy_s:
                return True
        return False

    return query_fn, obstacle_fn


def main():
    parser = argparse.ArgumentParser(description='Dhav-Hy quadruped viewer')
    parser.add_argument('--gait', choices=['roll', 'trot', 'step'],
                        default='roll',
                        help='Gait to run: roll (default), trot, step')
    parser.add_argument('--terrain', type=str, default=None,
                        help='Path to terrain XML file (from terrain_generator.py)')
    parser.add_argument('--staircase', action='store_true',
                        help='Enable staircase climbing mode')
    parser.add_argument('--stair-path', type=str, default=None,
                        help='Path to staircase XML file')
    args = parser.parse_args()

    # Load model with optional terrain
    xml_path = XML_PATH
    terrain_xml_path = args.terrain
    staircase_xml_path = args.stair_path

    # Determine which terrain file to use
    terrain_file = terrain_xml_path or staircase_xml_path

    # Create combined scene XML if terrain is specified
    if terrain_file and os.path.exists(terrain_file):
        combined_path = os.path.join(
            os.path.dirname(XML_PATH),
            f'scene_with_{os.path.basename(terrain_file)}'
        )
        xml_path = create_combined_scene_xml(XML_PATH, terrain_file, combined_path)

    if not os.path.exists(xml_path):
        print(f"ERROR: Model file not found at {xml_path}")
        sys.exit(1)

    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)

    # Increase timestep for real-time performance (0.0005 -> 0.002 s)
    model.opt.timestep = 0.002

    print(f"Model loaded: {model.nbody} bodies, {model.njnt} joints, {model.nu} actuators")
    print(f"Timestep: {model.opt.timestep} s")

    qpos_addr, dof_addr = setup_joint_addresses(model)

    # Build terrain query functions if terrain file provided
    query_fn = None
    obstacle_fn = None
    if args.terrain and os.path.exists(args.terrain):
        obstacles, stairs = parse_terrain_xml(args.terrain)
        query_fn, obstacle_fn = make_query_fns(obstacles, stairs)
        print(f"Terrain loaded: {len(obstacles)} obstacles, {len(stairs)} stairs")

    # Initialize stepping controller for stair climbing
    stepping_controller = None
    if args.gait == 'step' or args.staircase:
        from cpg_rl.controllers.step_controller import SteppingMotion
        stepping_controller = SteppingMotion(model, data, qpos_addr, dof_addr)
        stepping_controller.set_velocity_command(0.6, 0.0, 0.0)  # approach toward stairs
        stepping_controller._n_substeps = N_SUBSTEPS

        # Attach wheel controller
        from cpg_rl.wheel.wheel_controller import WheelController
        wheel_ctrl = WheelController(
            wheel_radius=0.077,
            max_velocity=73.0,
            max_torque=14.0,
            control_mode='velocity',
        )
        stepping_controller.set_wheel_controller(wheel_ctrl)

        # If staircase mode, enable online stair detection
        if args.staircase:
            # Set terrain query if available
            if query_fn is not None:
                pass  # online perception handles detection
            print("STAIRCASE CLIMBING MODE - Adaptive stair detection enabled")

    if args.gait == 'trot':
        controller = TrotMotion(model, data, qpos_addr, dof_addr)
        controller.set_velocity_command(3.0, 0.0, 0.0)
        print("Starting TROT simulation (diagonal pairs, wheels rolling)...")
    elif args.gait == 'step':
        if stepping_controller is not None:
            controller = stepping_controller
            print("STEPPING simulation (wheels locked, legs step over obstacles...)")
        else:
            controller = SteppingMotion(model, data, qpos_addr, dof_addr)
            controller.set_velocity_command(1.5, 0.0, 0.0)
            print("STEPPING simulation (wheels locked, legs step over obstacles...)")
    else:
        controller = RollingMotion(model, data, qpos_addr, dof_addr)
        controller.set_velocity_command(4.5, 0.0, 0.0)
        print("ROLLING simulation (legs stabilize, wheels rolling)...")

    set_stand_pose(data)
    mj.mj_forward(model, data)

    print("Close the viewer window to stop.")
    print()

    step_count = 0
    sim_time = 0.0
    wall_start = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Controls:")
        print("  ESC or Close window: Stop simulation")
        print("  'r': Reset robot pose")
        print("  's': Print current step statistics")
        print()

        while viewer.is_running():
            wall_now = time.time()
            wall_elapsed = wall_now - wall_start

            # Real-time sync: if sim is ahead of real time, sleep
            if sim_time > wall_elapsed:
                time.sleep(sim_time - wall_elapsed)

            mj.mj_forward(model, data)
            gravity = data.qfrc_bias.copy()

            # Compute torques from controller (updated every N_SUBSTEPS steps)
            if step_count % N_SUBSTEPS == 0:
                data.ctrl[:] = controller.compute_torques(model, data)

            # Substep physics for better real-time performance
            for _ in range(N_SUBSTEPS):
                mj.mj_step(model, data)
                sim_time += model.opt.timestep
            step_count += N_SUBSTEPS

            # Print status periodically
            if step_count % 200 == 0:
                base_pos = data.qpos[0:3]
                base_lin_vel = data.qvel[0:3]
                print(f"t={sim_time:6.2f}s  base_z={base_pos[2]:+.3f}  "
                      f"vel_x={base_lin_vel[0]:+.3f}  vel_y={base_lin_vel[1]:+.3f}")

                # Log stair geometry if using step_controller
                if stepping_controller and hasattr(stepping_controller, 'get_statistics'):
                    try:
                        g = stepping_controller.get_stair_geometry()
                        if g.num_steps > 0:
                            print(f"  Stair: {g.num_steps} steps, "
                                  f"riser={g.riser_heights[0]:.3f}m, "
                                  f"tread={g.tread_depths[0]:.3f}m")
                    except Exception as e:
                        pass  # Ignore errors in geometry logging

            viewer.sync()

    # Final summary
    bp = data.qpos[0:3]
    bv = data.qvel[0:3]
    print(f"\nSimulation stopped after {step_count} steps ({sim_time:.2f}s)")
    print(f"Final: base_z={bp[2]:+.3f} base_x={bp[0]:+.3f} base_y={bp[1]:+.3f} vel_x={bv[0]:+.3f}")


if __name__ == "__main__":
    main()