"""
Foot Trajectory Generator.

Converts CPG oscillator phase into desired foot positions (x, z) for each leg.
Implements stance/swing phases with smooth trajectories.

For each leg, the oscillator phase (0 to 2π) determines the gait phase:
    - Stance phase: phase % (2π) < duty_factor * 2π
      Foot is on ground, moves backward (propulsive)
    - Swing phase: phase % (2π) >= duty_factor * 2π
      Foot is lifted, moves forward (recovery)

Trajectory design (mathematically smooth):
    Stance: x = step_length/2 * cos(π * phase_stance), z = 0
            Foot moves from +step_length/2 (front) to -step_length/2 (back)
    Swing:  x = -step_length/2 * cos(π * phase_swing), z = swing_height * sin(π * phase_swing)
            Foot lifts, moves forward, and places down

Parameters:
    swing_height: max foot height during swing (m)
    step_length: total forward-backward foot travel (m)
    duty_factor: fraction of cycle in stance (0.0 to 1.0)

Does NOT depend on MuJoCo. Pure trajectory generation.
"""

import numpy as np


class FootTrajectoryGenerator:
    """
    Generates desired foot positions from CPG oscillator phase.

    Parameters
    ----------
    swing_height : float
        Maximum foot height during swing phase (m).
    step_length : float
        Total forward-backward foot travel during stance (m).
    duty_factor : float
        Fraction of gait cycle in stance phase (0.0 to 1.0).
        0.5 = equal stance/swing time.
    """

    def __init__(self, swing_height=0.10, step_length=0.20, duty_factor=0.5):
        self.swing_height = swing_height
        self.step_length = step_length
        self.duty_factor = duty_factor

        assert 0.0 < duty_factor < 1.0, "duty_factor must be in (0, 1)"
        assert swing_height > 0, "swing_height must be positive"
        assert step_length > 0, "step_length must be positive"

    def compute_foot_position(self, phase, leg_offset=0.0, stance_z=0.0):
        """
        Compute desired foot position (x, z) for a given oscillator phase.

        Parameters
        ----------
        phase : float
            Oscillator phase in radians (0 to 2π).
        leg_offset : float
            Additional phase offset for this leg (radians).
        stance_z : float
            Foot z-height during stance phase relative to hip (m).
            Typically negative (foot below hip), e.g. -0.357 for wheel on ground.

        Returns
        -------
        x : float
            Desired foot x-position relative to hip (m).
            Positive = forward, negative = backward.
        z : float
            Desired foot z-position relative to hip (m).
        """
        # Wrap phase to [0, 2π)
        phi = (phase + leg_offset) % (2.0 * np.pi)

        # Determine stance vs swing
        stance_threshold = self.duty_factor * 2.0 * np.pi

        if phi < stance_threshold:
            # === STANCE PHASE ===
            # Normalize phase within stance: 0 to 1
            ps = phi / stance_threshold

            # Foot moves from front (+step_length/2) to back (-step_length/2)
            # Using cosine for smooth velocity profile (zero velocity at endpoints)
            x = (self.step_length / 2.0) * np.cos(np.pi * ps)
            z = stance_z  # Foot at ground level (below hip)
        else:
            # === SWING PHASE ===
            # Normalize phase within swing: 0 to 1
            pw = (phi - stance_threshold) / (2.0 * np.pi - stance_threshold)

            # Foot lifts, moves forward, and places down
            # x: from back (-step_length/2) to front (+step_length/2)
            # z: from stance_z up to stance_z + swing_height and back
            x = -(self.step_length / 2.0) * np.cos(np.pi * pw)
            z = stance_z + self.swing_height * np.sin(np.pi * pw)

        return x, z

    def compute_foot_position_array(self, phases, leg_offsets=None):
        """
        Compute foot positions for multiple legs at once.

        Parameters
        ----------
        phases : array-like
            Oscillator phases for each leg (radians).
        leg_offsets : array-like or None
            Additional phase offsets per leg.

        Returns
        -------
        positions : list of (x, z) tuples
        """
        if leg_offsets is None:
            leg_offsets = [0.0] * len(phases)

        positions = []
        for phase, offset in zip(phases, leg_offsets):
            x, z = self.compute_foot_position(phase, offset)
            positions.append((x, z))

        return positions

    def generate_trajectory(self, duration, omega=6.28, leg_phases=None, dt=0.001):
        """
        Generate complete foot trajectories for all four legs over a duration.

        Parameters
        ----------
        duration : float
            Simulation duration (s).
        omega : float
            Gait frequency (rad/s).
        leg_phases : dict or None
            Phase offsets per leg. If None, uses diagonal trot.
        dt : float
            Time step (s).

        Returns
        -------
        trajectories : dict
            {leg: {'t': array, 'x': array, 'z': array, 'phase': array}}
        """
        if leg_phases is None:
            leg_phases = {'FL': 0.0, 'FR': np.pi, 'RL': np.pi, 'RR': 0.0}

        legs = ['FL', 'FR', 'RL', 'RR']
        n_steps = int(duration / dt)
        t_array = np.linspace(0, duration, n_steps + 1)

        trajectories = {}
        for leg in legs:
            x_array = np.zeros(n_steps + 1)
            z_array = np.zeros(n_steps + 1)
            phase_array = np.zeros(n_steps + 1)

            for i in range(n_steps + 1):
                t = t_array[i]
                # Oscillator phase: omega * t + leg_offset
                phase = (omega * t + leg_phases[leg]) % (2.0 * np.pi)
                phase_array[i] = phase

                x, z = self.compute_foot_position(phase, 0.0)
                x_array[i] = x
                z_array[i] = z

            trajectories[leg] = {
                't': t_array,
                'x': x_array,
                'z': z_array,
                'phase': phase_array,
            }

        return trajectories

    def plot_trajectories(self, trajectories, filename="foot_trajectories.png"):
        """Plot X-Z foot trajectories for all four legs."""
        import matplotlib.pyplot as plt
        import os

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        colors = {'FL': 'tab:blue', 'FR': 'tab:orange', 'RL': 'tab:green', 'RR': 'tab:red'}

        for idx, leg in enumerate(['FL', 'FR', 'RL', 'RR']):
            ax = axes[idx]
            traj = trajectories[leg]

            ax.plot(traj['x'], traj['z'], color=colors[leg], linewidth=2, label=leg)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('z (m)')
            ax.set_title(f'{leg} Leg Foot Trajectory')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_aspect('equal')
            ax.set_xlim(-0.15, 0.15)
            ax.set_ylim(-0.02, 0.15)

            # Mark stance and swing regions
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)

        plt.suptitle('Foot Trajectories (X-Z Plane) - Diagonal Trot', fontsize=14)
        plt.tight_layout()

        plot_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'results', 'plots'
        )
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, filename), dpi=150)
        plt.close()

        return fig

    def plot_gait_cycle(self, trajectories, filename="gait_cycle.png"):
        """Plot foot position vs time for all legs (gait cycle visualization)."""
        import matplotlib.pyplot as plt
        import os

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        colors = {'FL': 'tab:blue', 'FR': 'tab:orange', 'RL': 'tab:green', 'RR': 'tab:red'}

        for idx, leg in enumerate(['FL', 'FR', 'RL', 'RR']):
            ax = axes[idx]
            traj = trajectories[leg]

            ax.plot(traj['t'], traj['x'], color=colors[leg], linewidth=2, label='x')
            ax.plot(traj['t'], traj['z'], color=colors[leg], linewidth=2,
                    alpha=0.5, linestyle='--', label='z')
            ax.set_ylabel(f'{leg}\n(m)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

        axes[-1].set_xlabel('Time (s)')
        axes[0].set_title('Gait Cycle: Foot Position vs Time')
        plt.tight_layout()

        plot_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'results', 'plots'
        )
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, filename), dpi=150)
        plt.close()

        return fig


def test_trajectory_continuity():
    """Verify trajectory continuity at stance/swing transitions."""
    tg = FootTrajectoryGenerator(swing_height=0.10, step_length=0.20, duty_factor=0.5)

    stance_threshold = tg.duty_factor * 2.0 * np.pi

    # Stance->swing boundary
    phi_before = stance_threshold - 0.001
    phi_after = stance_threshold + 0.001
    x_before, z_before = tg.compute_foot_position(phi_before)
    x_after, z_after = tg.compute_foot_position(phi_after)

    x_diff = abs(x_after - x_before)
    z_diff = abs(z_after - z_before)

    assert x_diff < 0.01, f"x discontinuity at stance->swing: {x_diff}"
    assert z_diff < 0.01, f"z discontinuity at stance->swing: {z_diff}"

    # Swing->stance boundary
    phi_before = 2.0 * np.pi - 0.001
    phi_after = 0.001
    x_before, z_before = tg.compute_foot_position(phi_before)
    x_after, z_after = tg.compute_foot_position(phi_after)

    x_diff = abs(x_after - x_before)
    z_diff = abs(z_after - z_before)

    assert x_diff < 0.01, f"x discontinuity at swing->stance: {x_diff}"
    assert z_diff < 0.01, f"z discontinuity at swing->stance: {z_diff}"

    return True


def test_swing_height():
    """Verify foot reaches configured swing height."""
    for sh in [0.05, 0.10, 0.15]:
        tg = FootTrajectoryGenerator(swing_height=sh, step_length=0.20, duty_factor=0.5)
        trajectories = tg.generate_trajectory(duration=2.0, omega=6.28,
                                              leg_phases={'FL': 0.0, 'FR': np.pi,
                                                          'RL': np.pi, 'RR': 0.0})
        max_z = max(np.max(traj['z']) for traj in trajectories.values())
        assert abs(max_z - sh) < 0.01, f"Swing height error: {max_z} vs {sh}"
    return True


def test_duty_factor():
    """Verify stance/swing time ratio matches duty factor."""
    for df in [0.3, 0.5, 0.7]:
        tg = FootTrajectoryGenerator(swing_height=0.10, step_length=0.20, duty_factor=df)
        trajectories = tg.generate_trajectory(duration=2.0, omega=6.28,
                                              leg_phases={'FL': 0.0, 'FR': np.pi,
                                                          'RL': np.pi, 'RR': 0.0})
        traj = trajectories['FL']
        z = traj['z']
        stance_samples = np.sum(z < 0.01)
        measured_df = stance_samples / len(z)
        assert abs(measured_df - df) < 0.05, f"Duty factor error: {measured_df} vs {df}"
    return True