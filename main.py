import numpy as np
import argparse
import os
import pickle

# Import your modules
from src.custom_env import ObstacleEnv
from controllers.rl_algo import SafeMBRL
from controllers.learning_dyn import LearnerDynamics
from controllers.safety_filter import CasadiSafetyFilter
from src.scenarios import SingleIntegratorSystem, UnderactuatedSystem, UnicycleSystem
from src.basis_functions import PaperStaF, PolynomialBasis, PolarBasis
from analysis.metrics import MetricsTracker
from analysis.plots import generate_all_plots, generate_comparison_plots

# --- CONFIGURATION CONSTANTS ---
WEIGHTS_DIR = "weights"
METRICS_DIR = "metrics"
PLOTS_DIR = "plots"
GOAL_DISTANCE = 0.35


def get_scenario(name):
    """Factory to create the scenario object with paper-defined constants."""
    if name == "single_integrator":
        return SingleIntegratorSystem()
    elif name == "underactuated":
        return UnderactuatedSystem()
    elif name == "unicycle":
        return UnicycleSystem()
    else:
        raise ValueError(f"Unknown scenario: {name}")


def get_basis(name, state_dim):
    """Factory for basis functions."""
    if name == "paper_staf":
        assert state_dim == 2, "PaperStaF requires state_dim == 2"
        return PaperStaF(state_dim)
    elif name == "polynomial":
        return PolynomialBasis(state_dim, degree=2)
    elif name == "polar":
        assert state_dim >= 3, "PolarBasis requires state_dim >= 3 (e.g., unicycle)"
        return PolarBasis(state_dim, degree=2)
    else:
        raise ValueError(f"Unknown basis: {name}")


def get_auto_filename(scenario_name, basis_name, safety_on=True):
    """Generates standard filename: weights/scenario_basis[_nosafety].pkl"""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    suffix = "" if safety_on else "_nosafety"
    return os.path.join(WEIGHTS_DIR, f"{scenario_name}_{basis_name}{suffix}.pkl")


def get_metrics_filename(scenario_name, basis_name, safety_on=True):
    """Generates metrics filename."""
    os.makedirs(METRICS_DIR, exist_ok=True)
    suffix = "" if safety_on else "_nosafety"
    return os.path.join(METRICS_DIR, f"{scenario_name}_{basis_name}{suffix}.pkl")


def save_agent(agent, filepath):
    """Save weights and minimal config."""
    data = {
        'Wc': agent.Wc,
        'Wa': agent.Wa,
        'theta_hat': agent.dyn_learner.theta_hat,
        'Gamma': agent.Gamma
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved model to: {filepath}")


def load_agent(agent, filepath):
    """Load weights if they exist."""
    if not os.path.exists(filepath):
        print(f"No weights found at {filepath}. Starting from scratch.")
        return False

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    # Simple dimension check
    if len(data['Wc']) != len(agent.Wc):
        print(f"Dimension mismatch (Saved: {len(data['Wc'])}, Current: {len(agent.Wc)}). Ignored.")
        return False

    agent.Wc = data['Wc']
    agent.Wa = data['Wa']
    agent.Gamma = data.get('Gamma', agent.Gamma)
    agent.dyn_learner.theta_hat = data['theta_hat']
    print(f"Loaded model from: {filepath}")
    return True


def run_episode(env, agent, max_steps=500, noise=0.0, seed=None,
                metrics=None, episode_idx=0, store_trajectory=False,
                num_obstacles=None):
    """Runs one episode loop with optional metrics tracking."""
    obs, _ = env.reset(seed=seed)
    agent.dyn_learner.reset_window()
    agent.reset_episode()

    if metrics is not None:
        metrics.start_episode()

    total_reward = 0
    reason = "Timeout"
    dist_to_goal = 100.0

    for _ in range(max_steps):
        state = obs[:env.scenario.state_dim]
        goal = obs[env.scenario.state_dim:]

        # 1. Get Action (Agent -> Actor -> Safety Filter)
        action = agent.get_action(obs)

        # 2. Add Exploration Noise (Training Only)
        if noise > 0:
            action += np.random.normal(0, noise, size=action.shape)
            action = np.clip(action, -env.scenario.action_max, env.scenario.action_max)

        # 3. Step
        next_obs, reward, terminated, truncated, info = env.step(action)

        # 4. Update/Learn
        delta = agent.update(state, action, next_obs[:env.scenario.state_dim], env.dt, goal)

        total_reward += reward

        # 5. Record step metrics
        if metrics is not None:
            x_error = env.scenario.get_error_state(state, goal)
            cost = float(x_error.T @ env.scenario.Q @ x_error
                         + action.T @ env.scenario.R @ action)
            metrics.record_step(
                state=state,
                action=action,
                h_val=agent._last_h_val,
                bellman_error=float(delta) if delta is not None else None,
                cost=cost
            )

        obs = next_obs

        # Check success
        state = obs[:env.scenario.state_dim]
        goal = obs[env.scenario.state_dim:]
        dist_to_goal = np.linalg.norm(state[:2] - goal[:2])
        if dist_to_goal < 0.2:
            reason = "Goal Reached"
            terminated = True

        if terminated or truncated:
            reason = info.get("reason", reason)
            break

    # End-of-episode metrics
    if metrics is not None:
        theta_error = None
        if env.scenario.theta_dim > 0 and hasattr(env.scenario, 'theta_true'):
            theta_error = float(np.linalg.norm(
                agent.dyn_learner.theta_hat - env.scenario.theta_true))
        metrics.end_episode(
            episode_idx=episode_idx,
            reward=total_reward,
            dist=dist_to_goal,
            reason=reason,
            theta_error=theta_error,
            wc=agent.Wc.copy(),
            wa=agent.Wa.copy(),
            store_trajectory=store_trajectory,
            obstacles=[o.copy() for o in env.scenario.obstacles] if store_trajectory else None,
            goal_dist_threshold=GOAL_DISTANCE,
            num_obstacles=num_obstacles
        )

    return total_reward, dist_to_goal, reason


def train(args, scenario, basis, env, bounds, safety_on=True):
    """Train the agent and return metrics."""
    safety_filter = CasadiSafetyFilter(scenario=scenario, workspace_bounds=bounds)
    learner = LearnerDynamics(scenario=scenario)
    agent = SafeMBRL(learner, safety_filter, scenario, basis,
                     use_safety_filter=safety_on)

    filepath = get_auto_filename(args.scenario, args.basis, safety_on)
    metrics_path = get_metrics_filename(args.scenario, args.basis, safety_on)
    tracker = MetricsTracker()

    tag = "Safety ON" if safety_on else "Safety OFF"

    # Decide which episodes to store full trajectories for
    store_eps = set(list(range(5)) + list(range(max(0, args.episodes - 5), args.episodes)))

    # Obstacle curriculum state
    current_obstacles = args.start_obstacles
    curriculum_window = args.curriculum_window
    eps_at_current_level = 0          # reset when obstacles increase

    # max_bumps = episodes // window 
    _max_bumps = max(1, args.episodes // curriculum_window)
    _gap = args.max_obstacles - args.start_obstacles
    curriculum_step = max(1, -(-_gap // _max_bumps))

    print(f"\n--- Training {args.scenario} / {args.basis} [{tag}] ---")
    print(f"    Obstacle curriculum: {args.start_obstacles} → {args.max_obstacles} "
          f"(step={curriculum_step}, window={args.curriculum_window}, threshold=80%)")

    try:
        for ep in range(args.episodes):
            # Decay noise: 0.3 -> 0.0
            noise = max(0.0, 0.3 * (1.0 - ep / args.episodes))

            reward, dist, reason = run_episode(
                env, agent,
                max_steps=args.max_steps,
                noise=noise,
                seed=args.seed + ep,
                metrics=tracker,
                episode_idx=ep,
                store_trajectory=(ep in store_eps),
                num_obstacles=current_obstacles
            )
            eps_at_current_level += 1

            # Obstacle Curriculum
            if (eps_at_current_level >= curriculum_window
                    and current_obstacles < args.max_obstacles):
                recent_success = tracker.get_success_rate(last_n=curriculum_window)
                if recent_success >= 0.8:
                    prev = current_obstacles
                    current_obstacles = min(current_obstacles + curriculum_step,
                                            args.max_obstacles)
                    env.max_obstacles = current_obstacles
                    eps_at_current_level = 0      # reset counter
                    print(f"  ** Curriculum: obstacles {prev} → "
                          f"{current_obstacles} (success rate {recent_success*100:.0f}% "
                          f"over last {curriculum_window} eps)")

            # Print progress
            if scenario.theta_dim > 0 and hasattr(scenario, 'theta_true'):
                theta_error = np.linalg.norm(agent.dyn_learner.theta_hat - scenario.theta_true)
                print(f"Ep {ep+1:3d} | R: {reward:7.1f} | Dist: {dist:.2f} | "
                      f"Obs: {current_obstacles} | "
                      f"Theta Err: {theta_error:.4f} | {reason} [{tag}]")
            else:
                print(f"Ep {ep+1:3d} | R: {reward:7.1f} | Dist: {dist:.2f} | "
                      f"Obs: {current_obstacles} | {reason} [{tag}]")

    except KeyboardInterrupt:
        print("\nStopping early...")

    save_agent(agent, filepath)
    tracker.save(metrics_path)

    return tracker, agent


def main():
    parser = argparse.ArgumentParser(description="Safe MBRL with CBF")
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'plot', 'compare'])
    parser.add_argument('--scenario', type=str, default='single_integrator',
                        choices=['single_integrator', 'underactuated', 'unicycle'])
    parser.add_argument('--basis', type=str, default='paper_staf',
                        choices=['paper_staf', 'polynomial', 'polar'])
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--start_obstacles', type=int, default=1,
                        help='Starting number of obstacles (curriculum)')
    parser.add_argument('--max_obstacles', type=int, default=10,
                        help='Maximum obstacles the curriculum can reach')
    parser.add_argument('--curriculum_window', type=int, default=20,
                        help='Number of episodes to evaluate success rate for curriculum')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--no-safety', action='store_true',
                        help='Disable the CBF safety filter')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots after training/eval (off by default)')

    args = parser.parse_args()

    # 1. Setup Architecture
    scenario = get_scenario(args.scenario)
    basis = get_basis(name=args.basis, state_dim=scenario.state_dim)

    env = ObstacleEnv(
        scenario=scenario,
        render_mode="human" if args.render else None,
        goal_distance=GOAL_DISTANCE,
        max_obstacles=args.start_obstacles
    )

    bounds = {
        'x_min': env.area_bounds[0][0], 'x_max': env.area_bounds[0][1],
        'y_min': env.area_bounds[1][0], 'y_max': env.area_bounds[1][1],
    }

    # 2. Mode Selection
    if args.mode == 'train':
        safety_on = not args.no_safety
        tracker, agent = train(args, scenario, basis, env, bounds, safety_on=safety_on)

        if args.plot:
            safety_tag = "safe" if safety_on else "nosafety"
            plot_dir = os.path.join(PLOTS_DIR, f"{args.scenario}_{args.basis}_{safety_tag}")
            generate_all_plots(
                tracker, save_dir=plot_dir,
                scenario_name=args.scenario, basis_name=args.basis,
                area_bounds=env.area_bounds
            )

    elif args.mode == 'eval':
        safety_on = not args.no_safety
        filepath = get_auto_filename(args.scenario, args.basis, safety_on)

        safety_filter = CasadiSafetyFilter(scenario=scenario, workspace_bounds=bounds)
        learner = LearnerDynamics(scenario=scenario)
        agent = SafeMBRL(learner, safety_filter, scenario, basis,
                         use_safety_filter=safety_on)

        if not load_agent(agent, filepath):
            env.close()
            return

        tag = "Safety ON" if safety_on else "Safety OFF"
        print(f"\n--- Evaluating {args.scenario} / {args.basis} [{tag}] ---")

        tracker = MetricsTracker()
        success = 0
        for ep in range(args.episodes):
            reward, dist, reason = run_episode(
                env, agent,
                max_steps=args.max_steps,
                noise=0.0,
                seed=args.seed + ep,
                metrics=tracker,
                episode_idx=ep,
                store_trajectory=(ep < 10)
            )
            if dist < GOAL_DISTANCE:
                success += 1
            print(f"Eval {ep+1} | {reason} | Dist: {dist:.2f} [{tag}]")

        print(f"\nSuccess Rate: {success}/{args.episodes} "
              f"({100*success/args.episodes:.1f}%)")
        print(f"Safety Violations: {tracker.get_safety_violation_count()}")

        # Save eval metrics and generate plots
        metrics_path = get_metrics_filename(args.scenario, args.basis, safety_on)
        tracker.save(metrics_path.replace('.pkl', '_eval.pkl'))

        if args.plot:
            safety_tag = "safe" if safety_on else "nosafety"
            plot_dir = os.path.join(PLOTS_DIR, f"{args.scenario}_{args.basis}_{safety_tag}_eval")
            generate_all_plots(
                tracker, save_dir=plot_dir,
                scenario_name=args.scenario, basis_name=args.basis,
                area_bounds=env.area_bounds
            )

    elif args.mode == 'plot':
        # Regenerate plots from saved metrics
        safety_on = not args.no_safety
        metrics_path = get_metrics_filename(args.scenario, args.basis, safety_on)
        if not os.path.exists(metrics_path):
            print(f"No metrics found at {metrics_path}. Train first.")
            env.close()
            return
        tracker = MetricsTracker.load(metrics_path)
        safety_tag = "safe" if safety_on else "nosafety"
        plot_dir = os.path.join(PLOTS_DIR, f"{args.scenario}_{args.basis}_{safety_tag}")
        generate_all_plots(
            tracker, save_dir=plot_dir,
            scenario_name=args.scenario, basis_name=args.basis,
            area_bounds=env.area_bounds
        )

    elif args.mode == 'compare':
        # Train with safety ON and OFF, then generate comparison plots
        print("=" * 60)
        print("COMPARISON MODE: Training with Safety ON and OFF")
        print("=" * 60)

        tracker_safe, _ = train(args, scenario, basis, env, bounds, safety_on=True)

        # Need fresh basis for second run
        basis2 = get_basis(name=args.basis, state_dim=scenario.state_dim)
        tracker_unsafe, _ = train(args, scenario, basis2, env, bounds, safety_on=False)

        if args.plot:
            # Generate comparison plots
            plot_dir = os.path.join(PLOTS_DIR, f"{args.scenario}_{args.basis}_comparison")
            generate_comparison_plots(
                tracker_safe, tracker_unsafe,
                save_dir=plot_dir,
                scenario_name=f"{args.scenario}_{args.basis}",
                area_bounds=env.area_bounds
            )

            # Also generate individual plots
            for tracker, tag in [(tracker_safe, "safe"), (tracker_unsafe, "nosafety")]:
                ind_dir = os.path.join(PLOTS_DIR, f"{args.scenario}_{args.basis}_{tag}")
                generate_all_plots(
                    tracker, save_dir=ind_dir,
                    scenario_name=args.scenario, basis_name=args.basis,
                    area_bounds=env.area_bounds
                )

    env.close()


if __name__ == "__main__":
    main()