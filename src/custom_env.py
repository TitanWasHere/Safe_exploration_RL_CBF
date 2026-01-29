"""
File defines an custom environment with obstacles used for safe RL testing.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from src.scenarios import *

class ObstacleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, scenario_name="single_integrator", goal_mode="fixed", render_mode=None, dt=0.01):
        super().__init__()
        self.render_mode = render_mode
        self.dt = dt
        self.goal_mode = goal_mode # "fixed" or "random"

        # 1. Select Scenario
        if scenario_name == "single_integrator":
            self.scenario = SingleIntegrator()
        elif scenario_name == "double_integrator":
            self.scenario = DoubleIntegrator()
        elif scenario_name == "unicycle":
            self.scenario = Unicycle()
        else:
            raise ValueError("Unknown scenario")

        # 2. Define Obstacles in the Environment
        self.scenario.obstacles = [
            {'x': 0.0, 'y': 0.0, 'r': 1.0},
            {'x': -2.0, 'y': 2.0, 'r': 0.7},
            {'x': 2.0, 'y': -1.5, 'r': 0.6}
        ]

        # 3. Define Goal State
        self.goal_pos = np.array([1.5, 2.0], dtype=np.float32)

        # 4. Spaces
        # Observation = [State_Vector, Goal_X, Goal_Y]
        # the observation space is what the agent sees, so it includes the goal position and the state vector of the robot
        obs_dim = self.scenario.state_dim + 2 

        # space.box creates a continuous space with given bounds
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Action space depends on scenario dynamics
        # the action space represent the control inputs the agent can apply to the robot
        self.action_space = spaces.Box(
            low=-self.scenario.action_max, 
            high=self.scenario.action_max,
            shape=(self.scenario.action_dim,), dtype=np.float32
        )

        # Rendering
        self.window = None
        self.clock = None
        self.scale = 40

    def reset(self, seed=None, options=None):
        """
        reset the environment to an initial state and returns an initial observation.
        """
        super().reset(seed=seed)
        
        # 1. Randomize Goal if needed
        if self.goal_mode == "random":
            self.goal_pos = self._generate_valid_goal()
        else:
            self.goal_pos = np.array([1.5, 2.0], dtype=np.float32)

        # 2. Reset Robot State
        self.state = self.scenario.get_initial_state()
        
        # 3. Construct Observation
        obs = self._get_obs()
        
        return obs, {}

    def step(self, action):
        """
        Take an action in the environment and return the result.
        """

        # 1. Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 2. Physics Integration (Move the robot)
        self.state = self._rk4_step(self.state, action, self.dt)
        
        # 3. Calculate metrics
        curr_pos = self.state[:2]
        dist_to_goal = np.linalg.norm(curr_pos - self.goal_pos)
        min_h = self.scenario.get_h(self.state)
        is_safe = bool(min_h >= 0)
        
        # 4. Base Reward (Distance penalty + Control effort)
        # This encourages moving toward goal and being efficient
        reward = -dist_to_goal - 0.05 * np.linalg.norm(action)**2
        
        terminated = False
        truncated = False
        
        # 5. Collision Logic
        if not is_safe:
            terminated = True   # Stop the episode
            reward = -500.0     # Big penalty for crashing
            
            info = {"is_safe": False, "reason": "collision", "min_h": min_h}
            return self._get_obs(), reward, terminated, truncated, info

        
        # 6. Goal Reached Logic
        if dist_to_goal < 0.2:
            terminated = True   # Stop the episode
            reward += 200.0     # Big bonus for success
            
            info = {"is_safe": True, "reason": "goal_reached", "min_h": min_h}
            return self._get_obs(), reward, terminated, truncated, info

        # 7. Standard Info
        info = {
            "is_safe": True, 
            "min_h": min_h, 
            "dist_goal": dist_to_goal
        }
        
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        # Concatenate State + Goal Position
        return np.concatenate([self.state, self.goal_pos]).astype(np.float32)

    def _generate_valid_goal(self):
        """Generates a random goal that is not inside an obstacle."""
        while True:

            proposal = np.random.uniform(-4, 4, size=2).astype(np.float32)
            
            # Check collision with obstacles
            valid = True
            for obs in self.scenario.obstacles:
                dist = np.linalg.norm(proposal - np.array([obs['x'], obs['y']]))
                if dist < obs['r'] + 0.2: # Buffer
                    valid = False
                    break
            
            if valid:
                return proposal

    def _rk4_step(self, x, u, dt):
        """
        Runge-Kutta 4th order integration step. Used to have more accurate physics simulation.
        """
        k1 = self.scenario.dynamics(x, u)
        k2 = self.scenario.dynamics(x + 0.5 * dt * k1, u)
        k3 = self.scenario.dynamics(x + 0.5 * dt * k2, u)
        k4 = self.scenario.dynamics(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((600, 600))
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((600, 600))
        canvas.fill((255, 255, 255))
        
        def to_pix(pos):
            return int(pos[0] * self.scale + 300), int(-pos[1] * self.scale + 300)

        # 1. Draw Obstacles (Red)
        for obs in self.scenario.obstacles:
            pos = to_pix([obs['x'], obs['y']])
            rad = int(obs['r'] * self.scale)
            pygame.draw.circle(canvas, (200, 50, 50), pos, rad)
            pygame.draw.circle(canvas, (100, 0, 0), pos, rad, 1)

        # 2. Draw Goal (Green)
        goal_pix = to_pix(self.goal_pos)
        goal_rad = int(0.2 * self.scale) 
        pygame.draw.circle(canvas, (50, 200, 50), goal_pix, goal_rad)

        # 3. Draw Robot (Blue)
        robot_pix = to_pix(self.state[:2])
        robot_rad_pix = int(self.scenario.robot_radius * self.scale)
        # Body (Light Blue)
        pygame.draw.circle(canvas, (100, 100, 255), robot_pix, robot_rad_pix)
        # Collision Boundary (Dark Blue Outline)
        pygame.draw.circle(canvas, (0, 0, 150), robot_pix, robot_rad_pix, 2)
        # Center Point (Small Black Dot for precision)
        pygame.draw.circle(canvas, (0, 0, 0), robot_pix, 2)
        
        # If Unicycle, draw heading line
        if isinstance(self.scenario, Unicycle):
            theta = self.state[2]
            end_x = self.state[0] + 0.5 * np.cos(theta)
            end_y = self.state[1] + 0.5 * np.sin(theta)
            pygame.draw.line(canvas, (0,0,0), robot_pix, to_pix([end_x, end_y]), 2)

        self.window.blit(canvas, (0, 0))
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()

# Some simple test code to verify the environment works as expected
if __name__ == "__main__":
    import time
    
    # Use Fixed Goal mode
    # mode: single_integrator, double_integrator, unicycle
    SCENARIO = "unicycle"
    env = ObstacleEnv(scenario_name=SCENARIO, goal_mode="fixed", render_mode="human")
    
    # Reset
    obs, info = env.reset()
    print("Test Started. Robot should crash into the obstacle at (0,0).")

    for step in range(1000):
        
        # Simple logic to aim at obstacle (0,0)
        target_angle = np.arctan2(0.0 - obs[1], 0.0 - obs[0])
        current_angle = obs[2]
        angle_err = (target_angle - current_angle + np.pi) % (2*np.pi) - np.pi
        
        if SCENARIO == "single_integrator":
            # Indices:
            # obs[0], obs[1] = Position (x, y)
            # obs[2], obs[3] = Goal (gx, gy)

            # 1. Vector to goal
            dx = obs[2] - obs[0] # Goal_x - x
            dy = obs[3] - obs[1] # Goal_y - y
            
            # 2. Normalize and scale to max speed
            gain = 1.0
            action = np.array([gain * dx, gain * dy])
        
        elif SCENARIO == "double_integrator":
            # Indices:
            # obs[0], obs[1] = Position (x, y)
            # obs[2], obs[3] = Velocity (vx, vy)
            # obs[4], obs[5] = Goal (gx, gy)

            # 1. Calculate Errors
            pos_error_x = obs[4] - obs[0]
            pos_error_y = obs[5] - obs[1]
            
            vel_x = obs[2]
            vel_y = obs[3]

            # 2. PD Controller Gains
            Kp = 2.0 
            Kd = 1.5 

            # 3. Compute Acceleration
            # u = Kp * error - Kd * velocity
            u_x = Kp * pos_error_x - Kd * vel_x
            u_y = Kp * pos_error_y - Kd * vel_y

            action = np.array([u_x, u_y])
            
        elif SCENARIO == "unicycle":
            # 1. Unicycle Logic
            # Indices:
            # obs[0], obs[1] = Position (x, y)
            # obs[2] = Orientation (theta)
            # obs[3], obs[4] = Goal (gx, gy)

            dx = obs[3] - obs[0]
            dy = obs[4] - obs[1]
            target_angle = np.arctan2(dy, dx)
            current_angle = obs[2]
            
            # Find shortest angle difference
            angle_err = (target_angle - current_angle + np.pi) % (2*np.pi) - np.pi
            
            v = 1.0
            omega = 3.0 * angle_err

            # action is simply move with fixed velocity and correct heading
            action = np.array([v, omega])
            
        # 2. Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 3. Check Termination
        if terminated:
            print(f"Episode Ended at Step {step}")
            print(f"Final Reward: {reward}")
            if reward < -100:
                print("Result: CRASHED (Success test)")
            else:
                print("Result: REACHED GOAL")
            
            time.sleep(1)
            break
            
    env.close()