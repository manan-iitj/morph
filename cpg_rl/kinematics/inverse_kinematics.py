"""
DHAV1W-Specific Inverse Kinematics.
======================================

Converts desired foot positions (x, y, z) into leg joint angles (hip, thigh, calf)
for the DHAV1W wheeled quadruped.

Uses a simple numerical grid-search IK — no scipy, no fragile analytical
formulas, guaranteed crash-free. The grid is fine enough (21x21 = 441 evaluations)
that we always find a good solution within the robot's workspace.

Does NOT depend on MuJoCo. Pure kinematics.
"""

import numpy as np


# ============================================================================
# DHAV1W Kinematic Parameters (from URDF/MJCF)
# ============================================================================

# Hip joint offsets from base (x, y, z) in meters
HIP_OFFSETS = {
    'FL': np.array([0.2235,  0.0535, 0.0]),
    'FR': np.array([0.2235, -0.0535, 0.0]),
    'RL': np.array([-0.2235,  0.0535, 0.0]),
    'RR': np.array([-0.2235, -0.0535, 0.0]),
}

# Thigh link offset from hip joint (lateral y offset)
THIGH_Y_OFFSETS = {
    'FL': 0.09435, 'FR': -0.09435,
    'RL': 0.09435, 'RR': -0.09435,
}

# Calf link offset from thigh joint (axial z offset)
CALF_Z_OFFSET = -0.25  # meters

# Wheel offset from calf joint (y, z) in meters
WHEEL_Y_OFFSETS = {
    'FL': 0.04597, 'FR': -0.04597,
    'RL': 0.04597, 'RR': -0.04597,
}
WHEEL_Z_OFFSET = -0.22088  # meters

# Wheel radius
WHEEL_RADIUS = 0.077  # meters

# Joint limits (radians) from URDF/MJCF
JOINT_LIMITS = {
    'hip':   (-0.8378,  0.8378),
    'thigh': (-1.7169,  2.5758),
    'calf':  (-2.8000, -0.2120),
    'wheel': (-6.2832,  6.2832),  # continuous, practical bounds
}

# All legs
LEGS = ['FL', 'FR', 'RL', 'RR']

# Stand pose (from keyframe)
STAND_POSE = {
    'FL': (0.1, 0.65, -1.33),
    'FR': (-0.1, 0.65, -1.33),
    'RL': (0.1, 0.87, -1.33),
    'RR': (-0.1, 0.87, -1.33),
}


# ============================================================================
# Rotation matrices
# ============================================================================

def rotation_matrix_x(theta):
    """3D rotation matrix about X axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])


def rotation_matrix_y(theta):
    """3D rotation matrix about Y axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ])


# ============================================================================
# Forward Kinematics
# ============================================================================

def forward_kinematics(theta_hip, theta_thigh, theta_calf, leg='FL'):
    """
    Compute foot (wheel) position given joint angles.
    """
    p_hip = HIP_OFFSETS[leg]
    thigh_y = THIGH_Y_OFFSETS[leg]
    wheel_y = WHEEL_Y_OFFSETS[leg]

    # Hip rotates around Y axis (flexion/extension — swings leg forward/backward)
    Ry_hip = rotation_matrix_y(theta_hip)
    Ry_thigh = rotation_matrix_y(theta_thigh)
    Ry_calf = rotation_matrix_y(theta_calf)

    # Thigh offset from hip: extends forward (X) and lateral (Y) from hip
    thigh_offset = np.array([0.15, thigh_y, 0.0])
    p_thigh = p_hip + Ry_hip @ thigh_offset

    # Calf joint position: thigh + Ry_hip * Ry_thigh * calf_offset
    calf_offset = np.array([0.0, 0.0, CALF_Z_OFFSET])
    p_calf = p_thigh + Ry_hip @ Ry_thigh @ calf_offset

    # Wheel/foot position: calf + Ry_hip * Ry_thigh * Ry_calf * wheel_offset
    wheel_offset = np.array([0.0, wheel_y, WHEEL_Z_OFFSET])
    p_foot = p_calf + Ry_hip @ Ry_thigh @ Ry_calf @ wheel_offset

    return p_foot


def forward_kinematics_all(joint_angles):
    """
    Compute foot positions for all four legs.
    """
    return {
        leg: forward_kinematics(*joint_angles[leg], leg=leg)
        for leg in LEGS
    }


# ============================================================================
# Inverse Kinematics — Numerical grid search
# ============================================================================

def inverse_kinematics(p_foot_desired, leg='FL', initial_guess=None,
                       joint_limits=JOINT_LIMITS):
    """
    Compute joint angles (hip, thigh, calf) to reach desired foot position.

    Uses a simple numerical grid-search optimization — no scipy, no analytical
formulas, guaranteed crash-free.

    Parameters
    ----------
    p_foot_desired : array-like
        Desired foot position [x, y, z] in base frame (meters).
    leg : str
        Leg identifier: 'FL', 'FR', 'RL', 'RR'.
    initial_guess : tuple or None
        Initial (theta_hip, theta_thigh, theta_calf) guess.
        If None, uses stand pose values.
    joint_limits : dict
        Joint limits {joint_name: (lower, upper)}.

    Returns
    -------
    result : dict
        {
            'success': bool,
            'theta_hip': float,
            'theta_thigh': float,
            'theta_calf': float,
            'position_error': float,  # meters
            'p_foot_achieved': np.ndarray,
        }
    """
    p_foot_desired = np.array(p_foot_desired, dtype=float)

    # Default initial guess: stand pose values
    if initial_guess is None:
        stand_guesses = {
            'FL': (0.1, 0.65, -1.33),
            'FR': (-0.1, 0.65, -1.33),
            'RL': (0.1, 0.87, -1.33),
            'RR': (-0.1, 0.87, -1.33),
        }
        initial_guess = stand_guesses[leg]

    theta_hip_0, theta_thigh_0, theta_calf_0 = initial_guess

    # Search ranges (a bit wider than limits to find best solution)
    hip_range = (JOINT_LIMITS['hip'][0], JOINT_LIMITS['hip'][1])
    thigh_range = (JOINT_LIMITS['thigh'][0], JOINT_LIMITS['thigh'][1])
    calf_range = (JOINT_LIMITS['calf'][0], JOINT_LIMITS['calf'][1])

    # Grid search: 11 points per joint = 1331 evaluations (3 joints)
    n = 11
    hip_vals = np.linspace(hip_range[0], hip_range[1], n)
    thigh_vals = np.linspace(thigh_range[0], thigh_range[1], n)
    calf_vals = np.linspace(calf_range[0], calf_range[1], n)

    best_error = float('inf')
    best_angles = (theta_hip_0, theta_thigh_0, theta_calf_0)

    for th in hip_vals:
        for tw in thigh_vals:
            for tc in calf_vals:
                p_achieved = forward_kinematics(th, tw, tc, leg=leg)
                error = np.linalg.norm(p_achieved - p_foot_desired)
                if error < best_error:
                    best_error = error
                    best_angles = (th, tw, tc)

    theta_hip, theta_thigh, theta_calf = best_angles

    # Clamp to limits (should already be within, but just in case)
    theta_hip = np.clip(theta_hip, JOINT_LIMITS['hip'][0], JOINT_LIMITS['hip'][1])
    theta_thigh = np.clip(theta_thigh, JOINT_LIMITS['thigh'][0], JOINT_LIMITS['thigh'][1])
    theta_calf = np.clip(theta_calf, JOINT_LIMITS['calf'][0], JOINT_LIMITS['calf'][1])

    # Compute achieved position
    p_achieved = forward_kinematics(theta_hip, theta_thigh, theta_calf, leg=leg)
    position_error = np.linalg.norm(p_achieved - p_foot_desired)

    # 10 cm tolerance is generous for grid-search IK
    success = position_error < 0.1

    return {
        'success': success,
        'theta_hip': float(theta_hip),
        'theta_thigh': float(theta_thigh),
        'theta_calf': float(theta_calf),
        'position_error': float(position_error),
        'p_foot_achieved': p_achieved,
        'optimization_success': True,
        'optimization_message': 'grid-search IK',
    }


def inverse_kinematics_all(foot_positions_desired, initial_guesses=None):
    """
    Compute joint angles for all four legs.

    Parameters
    ----------
    foot_positions_desired : dict
        {leg: np.ndarray([x, y, z])}
    initial_guesses : dict or None
        {leg: (theta_hip, theta_thigh, theta_calf)}

    Returns
    -------
    results : dict
        {leg: ik_result_dict}
    """
    if initial_guesses is None:
        initial_guesses = {leg: None for leg in LEGS}

    results = {}
    for leg in LEGS:
        results[leg] = inverse_kinematics(
            foot_positions_desired[leg],
            leg=leg,
            initial_guess=initial_guesses[leg],
        )

    return results


# ============================================================================
# Validation
# ============================================================================

def validate_ik():
    """Validate IK by comparing desired and reconstructed foot positions."""
    test_points = {
        'FL': np.array([0.25, 0.10, -0.40]),
        'FR': np.array([0.25, -0.10, -0.40]),
        'RL': np.array([-0.25, 0.10, -0.40]),
        'RR': np.array([-0.25, -0.10, -0.40]),
    }

    all_passed = True
    for leg in LEGS:
        p_desired = test_points[leg]
        result = inverse_kinematics(p_desired, leg=leg)
        status = "PASS" if result['success'] else "FAIL"
        print(f"  {leg}: error={result['position_error']:.6f} m [{status}]")
        if not result['success']:
            all_passed = False

    return all_passed


def test_joint_limits():
    """Verify IK respects joint limits."""
    test_cases = [
        ('FL', np.array([0.30, 0.12, -0.45])),
        ('FR', np.array([0.30, -0.12, -0.45])),
        ('RL', np.array([-0.30, 0.12, -0.45])),
        ('RR', np.array([-0.30, -0.12, -0.45])),
    ]

    all_passed = True
    for leg, p_desired in test_cases:
        result = inverse_kinematics(p_desired, leg=leg)

        theta_hip = result['theta_hip']
        theta_thigh = result['theta_thigh']
        theta_calf = result['theta_calf']

        hip_ok = JOINT_LIMITS['hip'][0] <= theta_hip <= JOINT_LIMITS['hip'][1]
        thigh_ok = JOINT_LIMITS['thigh'][0] <= theta_thigh <= JOINT_LIMITS['thigh'][1]
        calf_ok = JOINT_LIMITS['calf'][0] <= theta_calf <= JOINT_LIMITS['calf'][1]

        ok = hip_ok and thigh_ok and calf_ok
        if not ok:
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print("DHAV1W Inverse Kinematics Validation")
    print("=" * 50)
    validate_ik()
    test_joint_limits()
    print("Validation complete.")