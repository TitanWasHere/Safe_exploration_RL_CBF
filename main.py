import numpy as np
import time
import matplotlib.pyplot as plt
from src.custom_env import ObstacleEnv
from controllers.rl_algo import SafeMBRL
from controllers.learning_dyn import LearnerDynamics
from controllers.safety_filter import CasadiSafetyFilter

def main():
    SCENARIO = "single_integrator"
    DT = 0.02
    MAX_STEPS = 400
    NUM_EPISODES = 50
    
    Q = np.eye(2) * 1.0
    R = np.eye(2) * 1.0
    
    env = ObstacleEnv(scenario_name=SCENARIO, goal_mode="fixed", render_mode="rgb_array", dt=DT)
    
    dyn_learner = LearnerDynamics(scenario_type=SCENARIO, learning_rate=0.5)
    safety_filter = CasadiSafetyFilter(scenario_type=SCENARIO, c_b=3.0)
    
    agent = SafeMBRL(
        dyn_learner=dyn_learner,
        safety_filter=safety_filter,
        Q=Q, 
        R=R, 
        num_kernels=3
    )
    
    theta_true = np.array([-0.6, -1.0, 1.0])

    history_reward = []
    print(f"Starting Training: {NUM_EPISODES} episodes")

    try:
        for ep in range(NUM_EPISODES):
            obs, _ = env.reset()
            agent.dyn_learner.reset_window() 
            episode_reward = 0
            
            for step in range(MAX_STEPS):
                state = obs[:2].astype(np.float64)
                goal = obs[2:].astype(np.float64)
                
                nearby_obstacles = env.scenario.get_near_obstacle(state, search_radius=1.5)
                g_x = agent.safety_filter.f_g(state).full()
                action = agent.get_action(state, nearby_obstacles, goal, g_x=g_x)
                
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_state = next_obs[:2].astype(np.float64)
                episode_reward += reward
                
                state_rel = state - goal
                next_state_rel = next_state - goal
                be_error = agent.update(state_rel, action, next_state_rel, DT, g_x=g_x)
                
                obs = next_obs
                
                dist_to_goal = np.linalg.norm(next_state - goal)
                if dist_to_goal < 0.1:
                    print(f"  Goal reached at step {step}!")
                    break
                
                if terminated or truncated:
                    break
            
            theta_est = agent.dyn_learner.theta_hat
            theta_error = np.linalg.norm(theta_est - theta_true)
            history_reward.append(episode_reward)
            final_dist = np.linalg.norm(state - goal)
            
            print(f"Ep {ep+1:2d} | Reward: {episode_reward:7.1f} | "
                  f"Error Theta: {theta_error:6.4f} | "
                  f"Final Dist: {final_dist:5.3f} | Final Pos: [{state[0]:.2f}, {state[1]:.2f}] | "
                  )
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()