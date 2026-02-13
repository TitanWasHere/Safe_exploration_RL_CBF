"""
Metrics tracking for Safe MBRL experiments.
Records per-step and per-episode data for analysis and plotting.
Implements the same metrics used in Cohen & Belta (2021) for 1:1 validation.
"""
import numpy as np
import pickle
import os


class MetricsTracker:
    """
    Tracks all metrics needed to reproduce the paper's figures:
    - Episode return / integral cost J(x0)
    - Safety violations (collisions + out-of-bounds)
    - Parameter estimation error ||theta_hat - theta_true||
    - CBF value h(x) traces
    - Bellman error delta convergence
    - Critic/Actor weight norms ||Wc||, ||Wa||
    - State trajectories for 2D phase plots
    """

    def __init__(self):
        # Per-episode aggregates
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_reasons = []
        self.episode_min_h = []
        self.episode_theta_errors = []
        self.episode_mean_bellman_errors = []
        self.episode_wc_norms = []
        self.episode_wa_norms = []
        self.episode_goal_distances = []
        self.episode_success = []
        self.episode_cumulative_cost = []
        self.episode_obstacle_count = []

        # Selected full traces for detailed analysis
        self.stored_trajectories = []      # list of (ep_idx, np.array of states)
        self.stored_h_traces = []          # list of (ep_idx, np.array of h values)
        self.stored_bellman_traces = []    # list of (ep_idx, np.array of bellman errors)
        self.stored_action_traces = []    # list of (ep_idx, np.array of actions)
        self.stored_obstacles = []        # list of (ep_idx, list of obstacle dicts)

        # Internal per-episode buffers
        self._trajectory = []
        self._h_values = []
        self._bellman_errors = []
        self._actions = []
        self._costs = []
        self._step_count = 0

    def start_episode(self):
        """Call at the beginning of each episode."""
        self._trajectory = []
        self._h_values = []
        self._bellman_errors = []
        self._actions = []
        self._costs = []
        self._step_count = 0

    def record_step(self, state, action=None, h_val=None, bellman_error=None, cost=None):
        """Call after each environment step."""
        self._trajectory.append(np.array(state, dtype=np.float32).copy())
        if action is not None:
            self._actions.append(np.array(action, dtype=np.float32).copy())
        if h_val is not None:
            self._h_values.append(float(h_val))
        if bellman_error is not None:
            self._bellman_errors.append(float(bellman_error))
        if cost is not None:
            self._costs.append(float(cost))
        self._step_count += 1

    def end_episode(self, episode_idx, reward, dist, reason,
                    theta_error=None, wc=None, wa=None,
                    store_trajectory=False, obstacles=None,
                    goal_dist_threshold=0.35, num_obstacles=None):
        """Call at the end of each episode to commit metrics."""
        self.episode_rewards.append(float(reward))
        self.episode_goal_distances.append(float(dist))
        self.episode_lengths.append(self._step_count)
        self.episode_reasons.append(reason)
        self.episode_success.append(1 if dist < goal_dist_threshold else 0)

        # Obstacle curriculum tracking
        if num_obstacles is not None:
            self.episode_obstacle_count.append(int(num_obstacles))

        # Average cost per step — avoids rewarding early crashes.
        # An agent that crashes at step 5 has high avg cost (it was near
        # an obstacle), while one that carefully reaches the goal in 500
        # steps accumulates low per-step cost.
        if self._costs and self._step_count > 0:
            self.episode_cumulative_cost.append(
                float(np.sum(self._costs) / self._step_count))
        elif self._step_count > 0:
            self.episode_cumulative_cost.append(
                float(-reward / self._step_count))
        else:
            self.episode_cumulative_cost.append(0.0)

        # Min CBF value (safety metric)
        if self._h_values:
            self.episode_min_h.append(float(min(self._h_values)))
        else:
            self.episode_min_h.append(float('inf'))

        # Parameter estimation error
        if theta_error is not None:
            self.episode_theta_errors.append(float(theta_error))

        # Mean Bellman error
        if self._bellman_errors:
            self.episode_mean_bellman_errors.append(float(np.mean(np.abs(self._bellman_errors))))

        # Weight norms
        if wc is not None:
            self.episode_wc_norms.append(float(np.linalg.norm(wc)))
        if wa is not None:
            self.episode_wa_norms.append(float(np.linalg.norm(wa)))

        # Store full trajectory for selected episodes
        if store_trajectory and self._trajectory:
            traj_arr = np.array(self._trajectory)
            self.stored_trajectories.append((episode_idx, traj_arr))
            if self._h_values:
                self.stored_h_traces.append((episode_idx, np.array(self._h_values)))
            if self._bellman_errors:
                self.stored_bellman_traces.append((episode_idx, np.array(self._bellman_errors)))
            if self._actions:
                self.stored_action_traces.append((episode_idx, np.array(self._actions)))
            if obstacles is not None:
                self.stored_obstacles.append((episode_idx, obstacles))

    def get_safety_violation_count(self):
        return sum(1 for r in self.episode_reasons
                   if "collision" in r or "out_of_bounds" in r)

    def get_cumulative_violations(self):
        """Cumulative safety violation count over episodes."""
        violations = [1 if ("collision" in r or "out_of_bounds" in r) else 0
                      for r in self.episode_reasons]
        return np.cumsum(violations).tolist()

    def get_success_rate(self, last_n=None):
        if not self.episode_success:
            return 0.0
        if last_n is not None:
            return float(np.mean(self.episode_success[-last_n:]))
        return float(np.mean(self.episode_success))

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        data = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Metrics saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        tracker = cls()
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        for key, val in data.items():
            if hasattr(tracker, key):
                setattr(tracker, key, val)
        return tracker
