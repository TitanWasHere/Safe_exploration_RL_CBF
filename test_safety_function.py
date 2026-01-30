import numpy as np
import time
import matplotlib.pyplot as plt
from src.custom_env import ObstacleEnv
from controllers.safety_filter import CasadiSafetyFilter
from controllers.learning_dyn import LearnerDynamics

def get_nominal_action(scenario, obs):
    """Simple P-controller to drive the robot towards the goal."""
    goal_x, goal_y = obs[-2], obs[-1]
    
    if scenario == "single_integrator":
        rx, ry = obs[0], obs[1]
        K = 3.0
        return np.array([K * (goal_x - rx), K * (goal_y - ry)])
    
    elif scenario == "unicycle":
        rx, ry, theta = obs[0], obs[1], obs[2]
        target_angle = np.arctan2(goal_y - ry, goal_x - rx)
        angle_err = (target_angle - theta + np.pi) % (2*np.pi) - np.pi
        return np.array([1.0, 4.0 * angle_err])

    return np.zeros(2)

def main():
    # 1. Configuration
    SCENARIO = "unicycle" # Matches the paper's complex f(x)
    env = ObstacleEnv(scenario_name=SCENARIO, goal_mode="random", render_mode="human")
    
    # 2. Setup Safety Filter (to ensure we don't crash while learning)
    safety_filter = CasadiSafetyFilter(
        scenario_type=SCENARIO, 
        robot_radius=env.scenario.robot_radius,
        c_b=15.0
    )
    
    # 3. Setup the Learner
    # We use a history_size of 50 to store informative data points for ICL
    learner = LearnerDynamics(scenario_type=SCENARIO, learning_rate=0.8, history_size=100)

    # Define "True Theta" for comparison (from scenarios.py)
    if SCENARIO == "single_integrator":
        true_theta = np.array([-0.6, -1.0, 1.0]) # [x1_coeff, x2_coeff, x1_cube_coeff]
    else:
        true_theta = np.array([-0.1]) # y-drift for unicycle

    # Data logging for plotting
    theta_history = []
    error_history = []

    print(f"\n--- TESTING SYSTEM IDENTIFICATION (ICL) ---")
    print(f"Target Parameters (True Theta): {true_theta}")
    
    obs, _ = env.reset()
    
    for step in range(1, 1001):
        x_old = env.state.copy()
        
        # A. Control logic
        u_nom = get_nominal_action(SCENARIO, obs)
        nearby_obs = env.scenario.get_near_obstacle(x_old)
        u_safe, _ = safety_filter.get_safe_action(obs, u_nom, nearby_obs)
        
        # B. Step Environment
        obs, reward, terminated, truncated, info = env.step(u_safe)
        x_new = env.state.copy()
        
        # C. Update the Learner (The new function)
        g_val = env.scenario.get_g(x_old)
        learner.update(x_old, x_new, u_safe, env.dt, g_val)
        
        # D. Log progress
        theta_hat = learner.theta_hat.copy()
        theta_history.append(theta_hat)
        error = np.linalg.norm(true_theta - theta_hat)
        error_history.append(error)
        
        if step % 50 == 0:
            print(f"Step {step} | Current Estimate: {np.round(theta_hat, 3)} | Error: {error:.4f}")

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    # E. Plotting the results
    plt.figure(figsize=(12, 5))
    
    # Parameter convergence
    plt.subplot(1, 2, 1)
    theta_history = np.array(theta_history)
    for i in range(len(true_theta)):
        plt.plot(theta_history[:, i], label=f'Estimated $\\theta_{i+1}$')
        plt.axhline(y=true_theta[i], color='r', linestyle='--', alpha=0.5)
    plt.title("Parameter Convergence (ICL)")
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.legend()

    # Estimation Error
    plt.subplot(1, 2, 2)
    plt.plot(error_history, color='black')
    plt.yscale('log')
    plt.title("Estimation Error (Norm)")
    plt.xlabel("Steps")
    plt.ylabel("Error (log scale)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()