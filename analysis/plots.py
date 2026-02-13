"""
Plotting functions for Safe MBRL paper-style figures.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import os


def smooth(data, window=10):
    """Moving average smoothing."""
    data = np.array(data, dtype=float)
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    smoothed = np.convolve(data, kernel, mode='valid')
    return smoothed


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)


def plot_learning_curve(tracker, ax=None, label=None, color=None, window=10):
    """Paper Fig: Integral cost J(x0) per episode."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    raw = tracker.episode_cumulative_cost if tracker.episode_cumulative_cost else [-r for r in tracker.episode_rewards]
    smoothed = smooth(raw, window=window)
    offset = window // 2
    x = np.arange(len(smoothed)) + offset
    ax.plot(x, smoothed, label=label, color=color, linewidth=1.5)
    ax.fill_between(x, smoothed * 0.9, smoothed * 1.1, alpha=0.1, color=color)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Cost per Step')
    ax.set_title('Learning Curve (Avg Step Cost)')
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax


def plot_episode_reward(tracker, ax=None, label=None, color=None, window=10):
    """Episode return (reward) over training."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    smoothed = smooth(tracker.episode_rewards, window=window)
    offset = window // 2
    x = np.arange(len(smoothed)) + offset
    ax.plot(x, smoothed, label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Return')
    ax.set_title('Episode Return')
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax


def plot_goal_distance(tracker, ax=None, label=None, color=None, window=10):
    """Final distance to goal per episode."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    smoothed = smooth(tracker.episode_goal_distances, window=window)
    offset = window // 2
    x = np.arange(len(smoothed)) + offset
    ax.plot(x, smoothed, label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Final Distance to Goal')
    ax.set_title('Goal Distance Convergence')
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax


def plot_safety_violations(tracker, ax=None, label=None, color=None):
    """Cumulative safety violations over episodes (Paper safety metric)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    cum_violations = tracker.get_cumulative_violations()
    ax.plot(cum_violations, label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Safety Violations')
    ax.set_title('Safety Violations')
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax


def plot_min_cbf(tracker, ax=None, label=None, color=None, window=5):
    """Minimum CBF value h(x) per episode (should stay > 0 with safety filter)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    data = [h for h in tracker.episode_min_h if h < 1e6]
    if not data:
        return ax
    smoothed = smooth(data, window=window)
    offset = window // 2
    x = np.arange(len(smoothed)) + offset
    ax.plot(x, smoothed, label=label, color=color, linewidth=1.5)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Safety boundary')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Min $h(x)$')
    ax.set_title('Minimum CBF Value per Episode')
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_parameter_error(tracker, ax=None, label=None, color=None, window=5):
    """Parameter estimation error ||theta_hat - theta*|| (Paper Fig 5)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    if not tracker.episode_theta_errors:
        ax.text(0.5, 0.5, 'No parameter errors recorded',
                transform=ax.transAxes, ha='center')
        return ax
    smoothed = smooth(tracker.episode_theta_errors, window=window)
    offset = window // 2
    x = np.arange(len(smoothed)) + offset
    ax.plot(x, smoothed, label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel(r'$\|\hat{\theta} - \theta^*\|$')
    ax.set_title('Parameter Estimation Error')
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax


def plot_bellman_error(tracker, ax=None, label=None, color=None, window=10):
    """Mean Bellman error per episode (convergence metric)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    if not tracker.episode_mean_bellman_errors:
        ax.text(0.5, 0.5, 'No Bellman errors recorded',
                transform=ax.transAxes, ha='center')
        return ax
    smoothed = smooth(tracker.episode_mean_bellman_errors, window=window)
    offset = window // 2
    x = np.arange(len(smoothed)) + offset
    ax.plot(x, smoothed, label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel(r'Mean $|\delta|$')
    ax.set_title('Bellman Error')
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax


def plot_weight_norms(tracker, ax=None, label_prefix='', window=5):
    """Critic and Actor weight norms over episodes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    if tracker.episode_wc_norms:
        wc_s = smooth(tracker.episode_wc_norms, window=window)
        offset = window // 2
        x = np.arange(len(wc_s)) + offset
        ax.plot(x, wc_s, label=f'{label_prefix}$\\|W_c\\|$', linewidth=1.5)
    if tracker.episode_wa_norms:
        wa_s = smooth(tracker.episode_wa_norms, window=window)
        offset = window // 2
        x = np.arange(len(wa_s)) + offset
        ax.plot(x, wa_s, label=f'{label_prefix}$\\|W_a\\|$', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Weight Norm')
    ax.set_title('Critic/Actor Weight Convergence')
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_trajectory_2d(tracker, ax=None, traj_idx=None, area_bounds=None):
    """
    2D state-space trajectory with obstacles (Paper Fig 3).
    Plots stored trajectories in the x-y plane.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    # Select which trajectories to plot
    if traj_idx is not None:
        trajs = [(i, t) for i, t in tracker.stored_trajectories if i in traj_idx]
        obs_list = [(i, o) for i, o in tracker.stored_obstacles if i in traj_idx]
    else:
        trajs = tracker.stored_trajectories
        obs_list = tracker.stored_obstacles

    # Plot obstacles from first available episode
    if obs_list:
        _, obstacles = obs_list[0]
        for obs in obstacles:
            circle = plt.Circle((obs['x'], obs['y']), obs['r'],
                                color='red', alpha=0.3, linewidth=1.5)
            ax.add_patch(circle)
            circle_edge = plt.Circle((obs['x'], obs['y']), obs['r'],
                                     fill=False, edgecolor='red', linewidth=1.5)
            ax.add_patch(circle_edge)

    # Plot goal
    goal_circle = plt.Circle((0, 0), 0.25, color='green', alpha=0.2, linewidth=1.5)
    ax.add_patch(goal_circle)
    ax.plot(0, 0, 'g*', markersize=15, label='Goal')

    # Color map for trajectories
    cmap = plt.cm.viridis
    n_trajs = len(trajs)
    for k, (ep_idx, traj) in enumerate(trajs):
        color = cmap(k / max(n_trajs - 1, 1))
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.0,
                alpha=0.7, label=f'Ep {ep_idx}')
        ax.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=5)
        ax.plot(traj[-1, 0], traj[-1, 1], 's', color=color, markersize=5)

    # Area bounds
    if area_bounds is not None:
        xmin, xmax = area_bounds[0]
        ymin, ymax = area_bounds[1]
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                  linewidth=2, edgecolor='red',
                                  facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.set_xlim(xmin - 0.5, xmax + 0.5)
        ax.set_ylim(ymin - 0.5, ymax + 0.5)
    else:
        ax.set_xlim(-7, 7)
        ax.set_ylim(-7, 7)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('State Trajectories')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_cbf_trace(tracker, ax=None, ep_indices=None):
    """CBF value h(x) within selected episodes (Paper Fig 6)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    if not tracker.stored_h_traces:
        ax.text(0.5, 0.5, 'No CBF traces stored', transform=ax.transAxes, ha='center')
        return ax

    traces = tracker.stored_h_traces
    if ep_indices is not None:
        traces = [(i, h) for i, h in traces if i in ep_indices]

    for ep_idx, h_vals in traces:
        ax.plot(h_vals, linewidth=1.0, alpha=0.7, label=f'Ep {ep_idx}')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Safety bound')
    ax.set_xlabel('Step')
    ax.set_ylabel('$h(x)$')
    ax.set_title('CBF Value Along Trajectory')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def plot_success_rate(tracker, ax=None, label=None, color=None, window=20):
    """Rolling success rate over episodes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    if not tracker.episode_success:
        return ax
    success = np.array(tracker.episode_success, dtype=float)
    if len(success) >= window:
        rolling = np.convolve(success, np.ones(window) / window, mode='valid')
        x = np.arange(len(rolling)) + window // 2
        ax.plot(x, rolling * 100, label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Rolling Success Rate')
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax


def plot_obstacle_curriculum(tracker, ax=None, label=None, color=None, window=20):
    """Plot obstacle count over episodes with success-rate overlay."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    if not getattr(tracker, 'episode_obstacle_count', None):
        return ax

    episodes = np.arange(len(tracker.episode_obstacle_count))
    obs_counts = np.array(tracker.episode_obstacle_count)

    ax.step(episodes, obs_counts, where='post', linewidth=2,
            color=color or '#2196F3', label=label or 'Max Obstacles')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Max Obstacles', color='#2196F3')
    ax.set_title('Obstacle Curriculum')
    ax.tick_params(axis='y', labelcolor='#2196F3')
    ax.set_ylim(0, max(obs_counts) + 1)
    ax.grid(True, alpha=0.3)

    # Overlay rolling success rate on secondary axis
    if tracker.episode_success and len(tracker.episode_success) >= window:
        ax2 = ax.twinx()
        success = np.array(tracker.episode_success, dtype=float)
        rolling = np.convolve(success, np.ones(window) / window, mode='valid')
        x = np.arange(len(rolling)) + window // 2
        ax2.plot(x, rolling * 100, color='#4CAF50', alpha=0.7,
                 linewidth=1.2, linestyle='--', label='Success Rate')
        ax2.set_ylabel('Success Rate (%)', color='#4CAF50')
        ax2.tick_params(axis='y', labelcolor='#4CAF50')
        ax2.set_ylim(-5, 105)
        # Threshold line at 80%
        ax2.axhline(80, color='#FF9800', linewidth=1, linestyle=':',
                     alpha=0.6, label='80% threshold')
        ax2.legend(loc='center right', fontsize=8)

    ax.legend(loc='upper left', fontsize=8)
    return ax

def generate_all_plots(tracker, save_dir='plots', scenario_name='', basis_name='',
                       area_bounds=None, show=False):
    """Generate all paper-style plots and save to directory."""
    os.makedirs(save_dir, exist_ok=True)
    prefix = f"{scenario_name}_{basis_name}_" if scenario_name else ""

    # 1. Learning Curve (Integral Cost)
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_learning_curve(tracker, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{prefix}learning_curve.png'), dpi=150)
    if not show:
        plt.close(fig)

    # 2. Episode Return
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_episode_reward(tracker, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{prefix}episode_return.png'), dpi=150)
    if not show:
        plt.close(fig)

    # 3. Goal Distance
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_goal_distance(tracker, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{prefix}goal_distance.png'), dpi=150)
    if not show:
        plt.close(fig)

    # 4. Safety Violations
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_safety_violations(tracker, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{prefix}safety_violations.png'), dpi=150)
    if not show:
        plt.close(fig)

    # 5. Parameter Estimation Error
    if tracker.episode_theta_errors:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_parameter_error(tracker, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'{prefix}parameter_error.png'), dpi=150)
        if not show:
            plt.close(fig)

    # 7. Bellman Error
    if tracker.episode_mean_bellman_errors:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_bellman_error(tracker, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'{prefix}bellman_error.png'), dpi=150)
        if not show:
            plt.close(fig)

    # 8. Weight Norms
    if tracker.episode_wc_norms or tracker.episode_wa_norms:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_weight_norms(tracker, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'{prefix}weight_norms.png'), dpi=150)
        if not show:
            plt.close(fig)

    # 9. Trajectories
    if tracker.stored_trajectories:
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_trajectory_2d(tracker, ax=ax, area_bounds=area_bounds)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'{prefix}trajectories.png'), dpi=150)
        if not show:
            plt.close(fig)

    # 10. CBF Trace
    if tracker.stored_h_traces:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_cbf_trace(tracker, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'{prefix}cbf_trace.png'), dpi=150)
        if not show:
            plt.close(fig)

    # 11. Success Rate
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_success_rate(tracker, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{prefix}success_rate.png'), dpi=150)
    if not show:
        plt.close(fig)

    # 12. Obstacle Curriculum
    if getattr(tracker, 'episode_obstacle_count', None):
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_obstacle_curriculum(tracker, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'{prefix}obstacle_curriculum.png'), dpi=150)
        if not show:
            plt.close(fig)

    # 13. Summary dashboard (multi-panel)
    _generate_dashboard(tracker, save_dir, prefix, area_bounds, show)

    print(f"All plots saved to {save_dir}/")


def _generate_dashboard(tracker, save_dir, prefix, area_bounds, show):
    """Single multi-panel figure summarizing all key metrics."""
    has_curriculum = bool(getattr(tracker, 'episode_obstacle_count', None))
    n_rows = 4 if has_curriculum else 3
    fig = plt.figure(figsize=(20, 5.5 * n_rows))
    gs = GridSpec(n_rows, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Row 1
    ax1 = fig.add_subplot(gs[0, 0])
    plot_learning_curve(tracker, ax=ax1)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_goal_distance(tracker, ax=ax2)
    ax3 = fig.add_subplot(gs[0, 2])
    plot_success_rate(tracker, ax=ax3)

    # Row 2
    ax4 = fig.add_subplot(gs[1, 0])
    plot_safety_violations(tracker, ax=ax4)
    ax5 = fig.add_subplot(gs[1, 1])
    if tracker.episode_mean_bellman_errors:
        plot_bellman_error(tracker, ax=ax5)
    else:
        plot_episode_reward(tracker, ax=ax5)
    ax6 = fig.add_subplot(gs[1, 2])
    if tracker.episode_theta_errors:
        plot_parameter_error(tracker, ax=ax6)
    else:
        plot_weight_norms(tracker, ax=ax6)

    # Row 3
    ax7 = fig.add_subplot(gs[2, 0])
    if tracker.episode_mean_bellman_errors:
        plot_bellman_error(tracker, ax=ax7)
    ax8 = fig.add_subplot(gs[2, 1])
    if tracker.stored_h_traces:
        plot_cbf_trace(tracker, ax=ax8)
    ax9 = fig.add_subplot(gs[2, 2])
    if tracker.stored_trajectories:
        plot_trajectory_2d(tracker, ax=ax9, area_bounds=area_bounds)

    # Row 4 — Obstacle Curriculum (only when data exists)
    if has_curriculum:
        ax10 = fig.add_subplot(gs[3, 0:2])
        plot_obstacle_curriculum(tracker, ax=ax10)
        ax11 = fig.add_subplot(gs[3, 2])
        plot_episode_reward(tracker, ax=ax11)

    fig.suptitle(f'{prefix.replace("_", " ").strip()} — Training Summary',
                 fontsize=16, fontweight='bold')
    fig.savefig(os.path.join(save_dir, f'{prefix}dashboard.png'), dpi=150)
    if not show:
        plt.close(fig)


def generate_comparison_plots(tracker_safe, tracker_unsafe,
                              save_dir='plots', scenario_name='',
                              area_bounds=None, show=False):
    """
    Compare safety-filter ON vs OFF.
    Generates side-by-side and overlay plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    prefix = f"{scenario_name}_compare_" if scenario_name else "compare_"

    # 1. Learning curves overlay
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_learning_curve(tracker_safe, ax=ax, label='Safety ON', color='blue')
    plot_learning_curve(tracker_unsafe, ax=ax, label='Safety OFF', color='red')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{prefix}learning_curve.png'), dpi=150)
    if not show:
        plt.close(fig)

    # 2. Goal distance overlay
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_goal_distance(tracker_safe, ax=ax, label='Safety ON', color='blue')
    plot_goal_distance(tracker_unsafe, ax=ax, label='Safety OFF', color='red')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{prefix}goal_distance.png'), dpi=150)
    if not show:
        plt.close(fig)

    # 3. Safety violations overlay
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_safety_violations(tracker_safe, ax=ax, label='Safety ON', color='blue')
    plot_safety_violations(tracker_unsafe, ax=ax, label='Safety OFF', color='red')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{prefix}safety_violations.png'), dpi=150)
    if not show:
        plt.close(fig)

    # 4. Success rate overlay
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_success_rate(tracker_safe, ax=ax, label='Safety ON', color='blue')
    plot_success_rate(tracker_unsafe, ax=ax, label='Safety OFF', color='red')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{prefix}success_rate.png'), dpi=150)
    if not show:
        plt.close(fig)

    # 5. Trajectory comparison side by side
    if tracker_safe.stored_trajectories and tracker_unsafe.stored_trajectories:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 8))
        plot_trajectory_2d(tracker_safe, ax=ax_l, area_bounds=area_bounds)
        ax_l.set_title('Trajectories (Safety ON)')
        plot_trajectory_2d(tracker_unsafe, ax=ax_r, area_bounds=area_bounds)
        ax_r.set_title('Trajectories (Safety OFF)')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'{prefix}trajectories.png'), dpi=150)
        if not show:
            plt.close(fig)

    # 6. CBF traces comparison
    if tracker_safe.stored_h_traces or tracker_unsafe.stored_h_traces:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6))
        plot_cbf_trace(tracker_safe, ax=ax_l)
        ax_l.set_title('CBF Trace (Safety ON)')
        plot_cbf_trace(tracker_unsafe, ax=ax_r)
        ax_r.set_title('CBF Trace (Safety OFF)')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'{prefix}cbf_trace.png'), dpi=150)
        if not show:
            plt.close(fig)

    # 7. Summary comparison dashboard
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    plot_learning_curve(tracker_safe, ax=ax, label='Safety ON', color='blue')
    plot_learning_curve(tracker_unsafe, ax=ax, label='Safety OFF', color='red')
    ax.legend()

    ax = fig.add_subplot(gs[0, 1])
    plot_goal_distance(tracker_safe, ax=ax, label='Safety ON', color='blue')
    plot_goal_distance(tracker_unsafe, ax=ax, label='Safety OFF', color='red')
    ax.legend()

    ax = fig.add_subplot(gs[0, 2])
    plot_success_rate(tracker_safe, ax=ax, label='Safety ON', color='blue')
    plot_success_rate(tracker_unsafe, ax=ax, label='Safety OFF', color='red')
    ax.legend()

    ax = fig.add_subplot(gs[1, 0])
    plot_safety_violations(tracker_safe, ax=ax, label='Safety ON', color='blue')
    plot_safety_violations(tracker_unsafe, ax=ax, label='Safety OFF', color='red')
    ax.legend()

    ax = fig.add_subplot(gs[1, 1])
    plot_episode_reward(tracker_safe, ax=ax, label='Safety ON', color='blue')
    plot_episode_reward(tracker_unsafe, ax=ax, label='Safety OFF', color='red')
    ax.legend()

    ax = fig.add_subplot(gs[1, 2])
    plot_episode_reward(tracker_safe, ax=ax, label='Safety ON', color='blue')
    plot_episode_reward(tracker_unsafe, ax=ax, label='Safety OFF', color='red')
    ax.legend()

    fig.suptitle(f'Safety Filter Comparison — {scenario_name}',
                 fontsize=16, fontweight='bold')
    fig.savefig(os.path.join(save_dir, f'{prefix}dashboard.png'), dpi=150)
    if not show:
        plt.close(fig)

    # Print summary statistics
    print("\n=== Comparison Summary ===")
    print(f"{'Metric':<30} {'Safety ON':>12} {'Safety OFF':>12}")
    print("-" * 56)
    print(f"{'Total Episodes':<30} {len(tracker_safe.episode_rewards):>12} {len(tracker_unsafe.episode_rewards):>12}")
    print(f"{'Mean Return (last 20)':<30} {np.mean(tracker_safe.episode_rewards[-20:]):>12.1f} {np.mean(tracker_unsafe.episode_rewards[-20:]):>12.1f}")
    print(f"{'Safety Violations':<30} {tracker_safe.get_safety_violation_count():>12} {tracker_unsafe.get_safety_violation_count():>12}")
    print(f"{'Success Rate (last 20)':<30} {tracker_safe.get_success_rate(20)*100:>11.1f}% {tracker_unsafe.get_success_rate(20)*100:>11.1f}%")
    mean_dist_s = np.mean(tracker_safe.episode_goal_distances[-20:])
    mean_dist_u = np.mean(tracker_unsafe.episode_goal_distances[-20:])
    print(f"{'Mean Final Dist (last 20)':<30} {mean_dist_s:>12.3f} {mean_dist_u:>12.3f}")

    print(f"\nComparison plots saved to {save_dir}/")
