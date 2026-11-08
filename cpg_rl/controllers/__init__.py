"""
Dhav-Hy Locomotion Controllers.

Exports motion controller classes used by mujoco_viz.py and other
entry points.

Available controllers:
- RollingMotion: Wheel-driven locomotion (high speed)
- TrotMotion: Diagonal pairs gait (balanced speed/stability)
- RollingMotion: Wheel-driven locomotion (high speed)
- TrotMotion: Diagonal pairs gait (balanced speed/stability)
- SteppingMotion: Leg-based stepping for obstacle negotiation
- Stabilizer: Common obstacle-adaptive stance + body stabilization
"""

from .base_controller import (
    BaseMotion, STAND_POSE, PD_GAINS, EFFORT_LIMITS,
    WHEEL_EFFORT_LIMIT, WHEEL_RADIUS, STANCE_FOOT_Z,
    ACTUATOR_MAP, LEGS, JOINT_TYPES, setup_joint_addresses
)
from .rolling_controller import RollingMotion
from .trot_controller import TrotMotion
from .stabilization import Stabilizer
from .step_controller import SteppingMotion

__all__ = [
    'BaseMotion',
    'STAND_POSE',
    'PD_GAINS',
    'EFFORT_LIMITS',
    'WHEEL_EFFORT_LIMIT',
    'WHEEL_RADIUS',
    'STANCE_FOOT_Z',
    'ACTUATOR_MAP',
    'LEGS',
    'JOINT_TYPES',
    'setup_joint_addresses',
    'RollingMotion',
    'TrotMotion',
    'SteppingMotion',
    'Stabilizer',
]
