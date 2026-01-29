import numpy as np
import time
import argparse

# Generated test to check if the functions works correctly


# Import your modules
# Assuming your file structure is:
# project/
#   envs/safe_env.py
#   controllers/safety_filter.py
from src.custom_env import ObstacleEnv
from controllers.safety_filter import CasadiSafetyFilter

def get_nominal_action(scenario, obs):
    """
    Standard 'Unsafe' Controllers (Go-To-Goal).
    These blindly drive towards the goal, ignoring obstacles.
    """
    # Parse Observation based on scenario
    # Single/Uni: obs = [x, y, ..., goal_x, goal_y]
    # Double:     obs = [x, y, vx, vy, goal_x, goal_y]
    
    goal_x = obs[-2]
    goal_y = obs[-1]
    
    if scenario == "single_integrator":
        # State: [x, y]
        rx, ry = obs[0], obs[1]
        
        # P-Controller
        dx = goal_x - rx
        dy = goal_y - ry
        
        # Simple gain
        return np.array([1.0 * dx, 1.0 * dy])

    elif scenario == "double_integrator":
        # State: [x, y, vx, vy]
        rx, ry = obs[0], obs[1]
        vx, vy = obs[2], obs[3]
        
        # PD-Controller
        dx = goal_x - rx
        dy = goal_y - ry
        
        Kp = 2.0
        Kd = 1.5
        
        ax = Kp * dx - Kd * vx
        ay = Kp * dy - Kd * vy
        
        return np.array([ax, ay])

    elif scenario == "unicycle":
        # State: [x, y, theta]
        rx, ry, theta = obs[0], obs[1], obs[2]
        
        dx = goal_x - rx
        dy = goal_y - ry
        
        # Pure Pursuit Logic
        target_angle = np.arctan2(dy, dx)
        angle_err = (target_angle - theta + np.pi) % (2*np.pi) - np.pi
        
        v = 1.0 # Constant speed
        omega = 4.0 * angle_err # Proportional turning
        
        return np.array([v, omega])

    return np.zeros(2)

def main(scenario_name="unicycle", use_safety=True):
    # 1. Setup Environment
    # We use 'fixed' goal mode so the scenario is reproducible
    env = ObstacleEnv(scenario_name=scenario_name, goal_mode="fixed", render_mode="human")
    
    # 2. Setup Safety Filter
    # Note: c_b is the gain. 
    # - Higher (20.0) = Stronger reaction, harder braking.
    # - Lower (1.0) = Smoother, but might crash if too fast.
    safety_filter = CasadiSafetyFilter(
        scenario_type=scenario_name, 
        robot_radius=env.scenario.robot_radius,
        c_b=20.0 
    )
    
    print(f"\n--- STARTING TEST: {scenario_name} ---")
    print(f"Safety Filter: {'ENABLED' if use_safety else 'DISABLED'}")
    print("Goal: Reach the Green Circle.")
    print("Hazard: Avoid the Red Circle.")
    
    obs, info = env.reset()
    
    for t in range(1000): # Run for max 1000 steps
        
        # A. Get Nominal Action (The "Pilot" who wants to go to goal)
        u_nom = get_nominal_action(scenario_name, obs)
        
        # B. Apply Safety Filter (The "Co-Pilot" who prevents crashes)
        if use_safety:
            # 1. Get obstacles relative to current state
            # Note: We must pass the raw state (env.state) to getting obstacles logic
            # if your env exposes it. Or calculate from obs.
            # The env.scenario helper uses env.state logic usually.
            # Let's use the env's helper:
            nearby_obs = env.scenario.get_near_obstacle(env.state)
            
            # 2. Compute Safe Action
            u_applied, min_h = safety_filter.get_safe_action(obs, u_nom, nearby_obs)
            
            # Optional: Visual Debugging
            # If the filter changed the action significantly, print it
            diff = np.linalg.norm(u_applied - u_nom)
            if diff > 0.1:
                pass
                # print(f"Step {t}: Safety Active! h={min_h:.2f} | Correction: {diff:.2f}")
        else:
            u_applied = u_nom

        # C. Step Environment
        obs, reward, terminated, truncated, info = env.step(u_applied)
        
        # D. Check Result
        if terminated:
            print(f"\nEpisode Ended at Step {t}")
            if reward > 0:
                print(">>> SUCCESS: Goal Reached! <<<")
            else:
                print(">>> FAILURE: CRASHED into Obstacle! <<<")
            break
            
        # Slow down slightly for human viewing
        # time.sleep(0.01)

    # Keep window open for a moment
    time.sleep(1.0)
    env.close()

if __name__ == "__main__":
    # Change these variables to test different configs
    
    # Options: "single_integrator", "double_integrator", "unicycle"
    SCENARIO = "unicycle" 
    
    # Options: True, False
    SAFETY_ON = True 
    
    main(SCENARIO, SAFETY_ON)