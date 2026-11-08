"""CPG (Central Pattern Generator) module for DHAV1W wheeled-legged quadruped."""

from .hopf_oscillator import HopfOscillator
from .coupled_cpg import CoupledCPG, CPGNetwork
from .foot_trajectory import FootTrajectoryGenerator

__all__ = ["HopfOscillator", "CoupledCPG", "CPGNetwork", "FootTrajectoryGenerator"]
