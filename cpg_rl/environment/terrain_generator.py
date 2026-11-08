"""
Proper terrain generator for DHAV1W — Complete MuJoCo XML terrain files.

Generates complete, loadable MuJoCo XML terrain files for testing rolling,
trotting, and stepping locomotion. Each file includes ground plane, materials,
and terrain geoms ready to be loaded by mujoco_viz.py --terrain.

Terrain types:
    - flat: Flat ground plane
    - stairs: Staircase (multiple riser heights)
    - obstacles: Rectangular blocks/barriers
    - ramps: Inclined ramps at various angles
    - uneven: Random bumps and depressions
    - mixed: Combination of different elements
"""

import math
import os

import numpy as np

# ---------------------------------------------------------------------------
# MuJoCo XML templates
# ---------------------------------------------------------------------------

MUJOCO_HEADER = """<mujoco model="{name}">
    <option timestep="0.002" gravity="0 0 -9.81"/>
    
    <asset>
        <material name="ground" rgba="0.8 0.8 0.8 1"/>
        <material name="stair" rgba="0.6 0.4 0.2 1"/>
        <material name="obstacle" rgba="0.5 0.5 0.5 1"/>
        <material name="ramp" rgba="0.7 0.7 0.3 1"/>
        <material name="hazard" rgba="0.8 0.2 0.2 1"/>
    </asset>

    <worldbody>
        <!-- Ground plane -->
        <geom name="ground" type="plane" size="50 50 0.1" pos="0 0 0" material="ground"/>
        
        {terrain_geoms}
    </worldbody>
</mujoco>
"""

GEOM_BOX = '    <geom name="{name}" type="box" size="{sx} {sy} {sz}" pos="{x} {y} {z}" material="{mat}"/>\n'
GEOM_SPHERE = '    <geom name="{name}" type="sphere" size="{r}" pos="{x} {y} {z}" material="{mat}"/>\n'


# ---------------------------------------------------------------------------
# Terrain generators
# ---------------------------------------------------------------------------

def generate_flat(length=50.0, width=50.0):
    """Flat ground plane only."""
    return ""


def generate_stairs(num_steps=6, riser_height=0.05, tread_depth=0.13,
                    stair_width=4.0, start_x=1.0, material="stair"):
    """Generate staircase terrain.

    Creates ascending stairs with proper riser/tread geometry.
    """
    geoms = []
    current_x = start_x
    current_z = 0.0

    for i in range(num_steps):
        # Riser (vertical face)
        riser_z = current_z + riser_height / 2.0
        geoms.append(GEOM_BOX.format(
            name=f"stair_{i}_riser",
            sx=0.02, sy=stair_width / 2.0, sz=riser_height / 2.0,
            x=current_x + tread_depth / 2.0, y=0.0, z=riser_z,
            mat=material,
        ))

        # Tread (horizontal surface)
        tread_z = current_z + riser_height - 0.01
        geoms.append(GEOM_BOX.format(
            name=f"stair_{i}_tread",
            sx=tread_depth / 2.0, sy=stair_width / 2.0, sz=0.01,
            x=current_x + tread_depth / 2.0, y=0.0, z=tread_z,
            mat=material,
        ))

        current_x += tread_depth
        current_z += riser_height

    return "".join(geoms)


def generate_obstacles(num_obstacles=8, min_size=0.1, max_size=0.4,
                       min_height=0.05, max_height=0.3,
                       area_x=(1.0, 15.0), area_y=(-2.0, 2.0),
                       seed=42, material="obstacle"):
    """Generate random rectangular obstacles."""
    rng = np.random.RandomState(seed)
    geoms = []

    for i in range(num_obstacles):
        sx = rng.uniform(min_size, max_size)
        sy = rng.uniform(min_size, max_size)
        sz = rng.uniform(min_height, max_height)
        x = rng.uniform(area_x[0], area_x[1])
        y = rng.uniform(area_y[0], area_y[1])
        z = sz / 2.0

        geoms.append(GEOM_BOX.format(
            name=f"obs_{i}",
            sx=sx / 2.0, sy=sy / 2.0, sz=sz / 2.0,
            x=x, y=y, z=z,
            mat=material,
        ))

    return "".join(geoms)


def generate_ramps(num_ramps=3, length=5.0, width=4.0,
                   angle_range=(5.0, 20.0),
                   start_x=1.0, material="ramp"):
    """Generate inclined ramps."""
    geoms = []
    current_x = start_x
    current_z = 0.0

    for i in range(num_ramps):
        angle = np.random.uniform(angle_range[0], angle_range[1])
        theta = math.radians(angle)
        dx = length * math.cos(theta)
        dz = length * math.sin(theta)

        center_x = current_x + dx / 2.0
        center_z = current_z + dz / 2.0

        # Ramp as a rotated box
        geoms.append(
            f'    <geom name="ramp_{i}" type="box" '
            f'size="{length/2:.4f} {width/2:.4f} 0.05" '
            f'pos="{center_x:.3f} 0.0 {center_z:.3f}" '
            f'axisangle="0 1 0 {-theta:.4f}" material="{material}"/>\n'
        )

        current_x += dx
        current_z += dz

    return "".join(geoms)


def generate_uneven(num_bumps=20, length=20.0, width=4.0,
                    min_height=-0.1, max_height=0.1,
                    min_size=0.15, max_size=0.4,
                    seed=42, material="ground"):
    """Generate random bumps and depressions."""
    rng = np.random.RandomState(seed)
    geoms = []

    for i in range(num_bumps):
        sx = rng.uniform(min_size, max_size)
        sy = rng.uniform(min_size, max_size)
        h = rng.uniform(min_height, max_height)
        x = rng.uniform(1.0, length - 1.0)
        y = rng.uniform(-width / 2.0 + 0.5, width / 2.0 - 0.5)
        z = h / 2.0 if h > 0 else h / 2.0

        geoms.append(GEOM_BOX.format(
            name=f"bump_{i}",
            sx=sx / 2.0, sy=sy / 2.0, sz=abs(h) / 2.0,
            x=x, y=y, z=z,
            mat=material,
        ))

    return "".join(geoms)


def generate_mixed(seed=42, length=20.0, width=4.0):
    """Generate mixed terrain with stairs, obstacles, ramps, and bumps."""
    rng = np.random.RandomState(seed)
    geoms = []

    section_len = length / 5.0

    # Section 1: Flat start (no geoms needed)

    # Section 2: Uneven bumps
    n_bumps = int(section_len * 0.4)
    for i in range(n_bumps):
        x = 1.0 + section_len + rng.uniform(0, section_len)
        y = rng.uniform(-width / 2.0 + 0.5, width / 2.0 - 0.5)
        h = rng.uniform(-0.05, 0.05)
        sx = rng.uniform(0.15, 0.3)
        sy = rng.uniform(0.15, 0.3)
        if abs(h) < 0.01:
            continue
        geoms.append(GEOM_BOX.format(
            name=f"mixed_bump_{i}",
            sx=sx / 2.0, sy=sy / 2.0, sz=abs(h) / 2.0,
            x=x, y=y, z=h / 2.0,
            mat="ground",
        ))

    # Section 3: Obstacles
    n_obs = max(2, int(5))
    for i in range(n_obs):
        x = 1.0 + section_len * 2 + rng.uniform(0, section_len)
        y = rng.uniform(-width / 2.0 + 0.5, width / 2.0 - 0.5)
        sx = rng.uniform(0.15, 0.35)
        sy = rng.uniform(0.15, 0.35)
        sz = rng.uniform(0.05, 0.25)
        geoms.append(GEOM_BOX.format(
            name=f"mixed_obs_{i}",
            sx=sx / 2.0, sy=sy / 2.0, sz=sz / 2.0,
            x=x, y=y, z=sz / 2.0,
            mat="obstacle",
        ))

    # Section 4: Stairs (small)
    n_stairs = 4
    riser_h = 0.05
    tread_d = 0.13
    stair_x = 1.0 + section_len * 3
    stair_z = 0.0
    for i in range(n_stairs):
        geoms.append(GEOM_BOX.format(
            name=f"mixed_stair_{i}_riser",
            sx=0.02, sy=width / 2.0, sz=riser_h / 2.0,
            x=stair_x + tread_d / 2.0, y=0.0, z=stair_z + riser_h / 2.0,
            mat="stair",
        ))
        geoms.append(GEOM_BOX.format(
            name=f"mixed_stair_{i}_tread",
            sx=tread_d / 2.0, sy=width / 2.0, sz=0.01,
            x=stair_x + tread_d / 2.0, y=0.0, z=stair_z + riser_h - 0.01,
            mat="stair",
        ))
        stair_x += tread_d
        stair_z += riser_h

    # Section 5: Ramp
    angle = math.radians(10.0)
    ramp_len = 5.0
    ramp_x = stair_x + 1.0
    ramp_z = stair_z
    center_x = ramp_x + ramp_len * math.cos(angle) / 2.0
    center_z = ramp_z + ramp_len * math.sin(angle) / 2.0
    geoms.append(
        f'    <geom name="mixed_ramp" type="box" '
        f'size="{ramp_len/2:.4f} {width/2:.4f} 0.05" '
        f'pos="{center_x:.3f} 0.0 {center_z:.3f}" '
        f'axisangle="0 1 0 {-angle:.4f}" material="ramp"/>\n'
    )

    return "".join(geoms)


# ---------------------------------------------------------------------------
# File writer
# ---------------------------------------------------------------------------

TERRAIN_GENERATORS = {
    'flat': generate_flat,
    'stairs': generate_stairs,
    'obstacles': generate_obstacles,
    'ramps': generate_ramps,
    'uneven': generate_uneven,
    'mixed': generate_mixed,
}


def generate_terrain_xml(terrain_type='mixed', **kwargs):
    """Generate complete MuJoCo XML terrain string.

    Parameters
    ----------
    terrain_type : str
        One of: flat, stairs, obstacles, ramps, uneven, mixed.
    **kwargs : dict
        Parameters passed to the specific terrain generator.
    """
    if terrain_type not in TERRAIN_GENERATORS:
        raise ValueError(f"Unknown terrain type: {terrain_type}. "
                         f"Available: {list(TERRAIN_GENERATORS.keys())}")

    generator = TERRAIN_GENERATORS[terrain_type]

    # Filter kwargs to only those accepted by the generator
    import inspect
    sig = inspect.signature(generator)
    valid_kwargs = {}
    for k, v in kwargs.items():
        if k in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        ):
            valid_kwargs[k] = v

    terrain_geoms = generator(**valid_kwargs)

    return MUJOCO_HEADER.format(
        name=f"{terrain_type}_terrain",
        terrain_geoms=terrain_geoms,
    )


def save_terrain(terrain_type='mixed', filepath='terrain.xml', **kwargs):
    """Generate and save a terrain XML file.

    Parameters
    ----------
    terrain_type : str
        Type of terrain to generate.
    filepath : str
        Output file path.
    **kwargs : dict
        Parameters passed to the terrain generator.
    """
    xml_str = generate_terrain_xml(terrain_type, **kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(xml_str)
    print(f"Terrain saved: {filepath} ({terrain_type})")
    return filepath


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate MuJoCo terrain XML")
    parser.add_argument('--type', choices=list(TERRAIN_GENERATORS.keys()),
                        default='mixed', help='Terrain type')
    parser.add_argument('--output', default='terrain.xml',
                        help='Output file path')
    parser.add_argument('--num-steps', type=int, default=6,
                        help='Number of stairs (for stairs type)')
    parser.add_argument('--riser-height', type=float, default=0.05,
                        help='Stair riser height in meters')
    parser.add_argument('--tread-depth', type=float, default=0.13,
                        help='Stair tread depth in meters')
    parser.add_argument('--num-obstacles', type=int, default=8,
                        help='Number of obstacles')
    parser.add_argument('--num-ramps', type=int, default=3,
                        help='Number of ramps')
    parser.add_argument('--num-bumps', type=int, default=20,
                        help='Number of bumps (uneven type)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    kwargs = {'seed': args.seed}
    if args.type == 'stairs':
        kwargs.update({
            'num_steps': args.num_steps,
            'riser_height': args.riser_height,
            'tread_depth': args.tread_depth,
        })
    elif args.type == 'obstacles':
        kwargs['num_obstacles'] = args.num_obstacles
    elif args.type == 'ramps':
        kwargs['num_ramps'] = args.num_ramps
    elif args.type == 'uneven':
        kwargs['num_bumps'] = args.num_bumps

    save_terrain(args.type, args.output, **kwargs)
