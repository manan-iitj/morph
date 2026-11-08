"""
Analysis and visualization for DHAV1W CPG-RL results.

Generates plots for:
- Training curves (reward over time)
- Controller comparison
- Robot state (base position, joint angles, wheel velocities)
- CPG phase coordination
- Foot trajectories
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def plot_training_curves(log_dir, save_path=None):
    """Plot training reward curves from tensorboard logs."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard not installed. Install with: pip install tensorboard")
        return

    event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    if not event_files:
        print(f"No tensorboard event files found in {log_dir}")
        return

    event_path = os.path.join(log_dir, event_files[0])
    ea = EventAccumulator(event_path)
    ea.Reload()

    tags = ea.Tags()['scalars']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    plot_idx = 0
    for tag in ['rollout/ep_len_mean', 'rollout/ep_rew_mean', 'train/loss', 'train/value_loss']:
        if tag in tags and plot_idx < 4:
            ax = axes[plot_idx]
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            ax.plot(steps, values, 'b-', linewidth=2)
            ax.set_xlabel('Steps')
            ax.set_ylabel(tag.split('/')[-1])
            ax.set_title(tag)
            ax.grid(True, alpha=0.3)
            plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, 4):
        axes[i].set_visible(False)

    plt.suptitle('PPO Training Curves', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training curves saved to: {save_path}")
    plt.close()


def plot_controller_comparison(results, save_path=None):
    """Plot controller comparison results."""
    import json

    if isinstance(results, str):
        with open(results, 'r') as f:
            results = json.load(f)

    controllers = list(results.keys())
    stages = sorted(set(k.split('_')[1] for v in results.values() for k in v.keys()))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Mean reward comparison
    ax = axes[0, 0]
    x = np.arange(len(stages))
    width = 0.2
    for i, ctrl in enumerate(controllers):
        rewards = [results[ctrl][f'stage_{s}']['mean_reward'] for s in stages]
        ax.bar(x + i * width, rewards, width, label=ctrl, alpha=0.7)
    ax.set_xlabel('Terrain Stage')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Mean Reward by Controller and Stage')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(stages)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Forward velocity comparison
    ax = axes[0, 1]
    for i, ctrl in enumerate(controllers):
        velocities = [results[ctrl][f'stage_{s}']['mean_forward_vel'] for s in stages]
        ax.bar(x + i * width, velocities, width, label=ctrl, alpha=0.7)
    ax.set_xlabel('Terrain Stage')
    ax.set_ylabel('Forward Velocity (m/s)')
    ax.set_title('Forward Velocity by Controller and Stage')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(stages)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Fall rate comparison
    ax = axes[1, 0]
    for i, ctrl in enumerate(controllers):
        fall_rates = [results[ctrl][f'stage_{s}']['fall_rate'] for s in stages]
        ax.bar(x + i * width, fall_rates, width, label=ctrl, alpha=0.7)
    ax.set_xlabel('Terrain Stage')
    ax.set_ylabel('Fall Rate')
    ax.set_title('Fall Rate by Controller and Stage')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(stages)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Reward std comparison
    ax = axes[1, 1]
    for i, ctrl in enumerate(controllers):
        stds = [results[ctrl][f'stage_{s}']['std_reward'] for s in stages]
        ax.bar(x + i * width, stds, width, label=ctrl, alpha=0.7)
    ax.set_xlabel('Terrain Stage')
    ax.set_ylabel('Reward Std Dev')
    ax.set_title('Reward Variability by Controller and Stage')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(stages)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Controller Ablation Study', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Comparison plot saved to: {save_path}")
    plt.close()


def plot_robot_state(log_data, save_path=None):
    """Plot robot state from simulation log."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    t = log_data.get('t', np.arange(len(log_data.get('base_pos', []))))

    # Base position
    ax = axes[0, 0]
    base_pos = np.array(log_data['base_pos'])
    ax.plot(t, base_pos[:, 0], 'b-', label='x', linewidth=2)
    ax.plot(t, base_pos[:, 1], 'g-', label='y', linewidth=2)
    ax.plot(t, base_pos[:, 2], 'r-', label='z', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('Base Position')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Base linear velocity
    ax = axes[0, 1]
    base_vel = np.array(log_data['base_lin_vel'])
    ax.plot(t, base_vel[:, 0], 'b-', label='vx', linewidth=2)
    ax.plot(t, base_vel[:, 1], 'g-', label='vy', linewidth=2)
    ax.plot(t, base_vel[:, 2], 'r-', label='vz', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Base Linear Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Joint positions
    ax = axes[1, 0]
    joint_pos = np.array(log_data['joint_pos'])
    for i in range(min(4, joint_pos.shape[1])):
        ax.plot(t, joint_pos[:, i], linewidth=1.5, label=f'J{i}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (rad)')
    ax.set_title('Joint Positions (First 4)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Joint velocities
    ax = axes[1, 1]
    joint_vel = np.array(log_data['joint_vel'])
    for i in range(min(4, joint_vel.shape[1])):
        ax.plot(t, joint_vel[:, i], linewidth=1.5, label=f'J{i}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (rad/s)')
    ax.set_title('Joint Velocities (First 4)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Wheel velocities
    ax = axes[2, 0]
    wheel_vel = np.array(log_data['wheel_vel'])
    leg_names = ['FL', 'FR', 'RL', 'RR']
    for i in range(4):
        ax.plot(t, wheel_vel[:, i], linewidth=2, label=leg_names[i])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (rad/s)')
    ax.set_title('Wheel Velocities')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # CPG phases
    ax = axes[2, 1]
    cpg_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for i, leg in enumerate(LEGS):
        cpg_data = np.array(log_data[f'cpg_phase_{leg}'])
        ax.plot(t, cpg_data, color=cpg_colors[i], linewidth=2, label=leg)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CPG Phase (rad)')
    ax.set_title('CPG Oscillator Phases')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Robot State Analysis', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Robot state plot saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Analysis module ready. Use plot_training_curves, plot_controller_comparison, or plot_robot_state.")