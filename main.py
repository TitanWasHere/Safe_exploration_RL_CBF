import numpy as np
import argparse
import os
import pickle

# Import your modules
from src.custom_env import ObstacleEnv
from controllers.rl_algo import SafeMBRL
from controllers.learning_dyn import LearnerDynamics
from controllers.safety_filter import CasadiSafetyFilter
from src.scenarios import SingleIntegratorSystem, UnderactuatedSystem
from src.basis_functions import PaperStaF

# --- CONFIGURATION CONSTANTS ---
WEIGHTS_DIR = "weights"
DEFAULT_NUM_KERNELS = 5  # Fixed complexity for RBF/StaF
GOAL_DISTANCE = 0.35

def get_scenario(name):
    """Factory to create the scenario object with paper-defined constants."""
    if name == "single_integrator":
        return SingleIntegratorSystem()
    elif name == "underactuated":
        return UnderactuatedSystem()
    else:
        raise ValueError(f"Unknown scenario: {name}")

def get_basis(name, state_dim):
    """Factory for basis functions. Defaults to PolyStaF as per paper."""
    # The paper uses L=3 polynomial StaF kernels
    if name == "paper_staf":
        return PaperStaF(state_dim)
    else:
        raise ValueError(f"Unknown scenario: {name}")

def get_auto_filename(scenario_name):
    """Generates standard filename: weights/scenario_staf.pkl"""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    return os.path.join(WEIGHTS_DIR, f"{scenario_name}_staf.pkl")

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
    print(f"✓ Saved model to: {filepath}")

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

def run_episode(env, agent, max_steps=500, noise=0.0, seed=None):
    """Runs one episode loop."""
    obs, _ = env.reset(seed=seed)
    agent.dyn_learner.reset_window()
    
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
        agent.update(state, action, next_obs[:env.scenario.state_dim], env.dt, goal)

        total_reward += reward
        obs = next_obs
        
        # Check success (use updated state after step)
        state = obs[:env.scenario.state_dim]
        goal = obs[env.scenario.state_dim:]
        dist_to_goal = np.linalg.norm(state[:2] - goal[:2])
        if dist_to_goal < 0.2:
            reason = "Goal Reached"
            terminated = True
            
        if terminated or truncated:
            reason = info.get("reason", reason)
            break
            
    return total_reward, dist_to_goal, reason

def main():
    parser = argparse.ArgumentParser(description="Safe MBRL with CBF")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--scenario', type=str, default='single_integrator', choices=['single_integrator', 'underactuated'])
    parser.add_argument('--basis', type=str, default='paper_staf', choices=['paper_staf'])
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--max_obstacles', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', action='store_true')
    
    args = parser.parse_args()
    
    # 1. Setup Architecture
    scenario = get_scenario(args.scenario)
    basis = get_basis(name=args.basis, state_dim=scenario.state_dim)
    filepath = get_auto_filename(args.scenario)
    
    env = ObstacleEnv(scenario=scenario, render_mode="human" if args.render else None, goal_distance=GOAL_DISTANCE, max_obstacles=args.max_obstacles)
    
    # 2. Initialize Components
    bounds = {
        'x_min': env.area_bounds[0][0],
        'x_max': env.area_bounds[0][1],
        'y_min': env.area_bounds[1][0],
        'y_max': env.area_bounds[1][1],
    }
    safety_filter = CasadiSafetyFilter(scenario=scenario, workspace_bounds=bounds)
    learner = LearnerDynamics(scenario=scenario)
    agent = SafeMBRL(learner, safety_filter, scenario, basis)
    
    # 3. Mode Selection
    if args.mode == 'train':
        print(f"--- Training {args.scenario} {args.basis} ---")
        history = []
        
        try:
            for ep in range(args.episodes):
                # Decay noise: 0.3 -> 0.0
                noise = max(0.0, 0.3 * (1.0 - ep / args.episodes))
                
                reward, dist, reason = run_episode(
                    env,
                    agent,
                    max_steps=args.max_steps,
                    noise=noise,
                    seed=args.seed + ep
                )
                history.append(reward)
                
                theta_mag = np.linalg.norm(agent.dyn_learner.theta_hat)
                print(f"Ep {ep+1:3d} | R: {reward:7.1f} | Dist: {dist:.2f} | Theta: {theta_mag:.2f} | {reason}")
                
        except KeyboardInterrupt:
            print("\nStopping early...")
            
        save_agent(agent, filepath)
        
    elif args.mode == 'eval':
        print(f"--- Evaluating {args.scenario} {args.basis} ---")
        if not load_agent(agent, filepath):
            return
        
        success = 0
        for ep in range(args.episodes):
            reward, dist, reason = run_episode(
                env,
                agent,
                max_steps=args.max_steps,
                noise=0.0,
                seed=args.seed + ep
            )
            if dist < GOAL_DISTANCE: 
                success += 1
            print(f"Eval {ep+1} | {reason} | Dist: {dist:.2f}")
            
        print(f"Success Rate: {success}/{args.episodes}")

    env.close()

if __name__ == "__main__":
    main()