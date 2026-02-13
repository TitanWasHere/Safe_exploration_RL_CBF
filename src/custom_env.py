"""
Custom environment used to test the rl algorithm.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class ObstacleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self, 
        scenario, 
        goal_mode="fixed", 
        render_mode=None, 
        dt=0.01,
        goal_distance = 0.25,
        max_obstacles=5,               # Maximum number of obstacles
        area_bounds=((-6, 6), (-6, 6)) # (min_x, max_x), (min_y, max_y)
    ):
        super().__init__()
        self.render_mode = render_mode
        self.dt = dt
        self.goal_mode = goal_mode 
        self.max_obstacles = max_obstacles
        self.area_bounds = area_bounds
        self.goal_distance = goal_distance

        # Scenario
        self.scenario = scenario

        # Goal is at the Origin — dimension matches scenario.goal_dim
        self.goal_dim = getattr(scenario, 'goal_dim', 2)
        self.goal_pos = np.zeros(self.goal_dim, dtype=np.float32)

        # OBSERVATION SPACE:
        # State (n) + Goal (goal_dim)
        self.obs_state_dim = self.scenario.state_dim + self.goal_dim 

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_state_dim,), dtype=np.float32
        )
        
        # ACTION SPACE:
        # +- max action defined in the scenario
        self.action_space = spaces.Box(
            low=-self.scenario.action_max, 
            high=self.scenario.action_max,
            shape=(self.scenario.action_dim,), dtype=np.float32
        )

        self.window = None
        self.clock = None
        self.scale = 50

    def reset(self, seed=None):
        """
        Reset the environment with the given seed
        """
        super().reset(seed=seed)
        
        # 1. Randomize Start Position
        boundary_margin = 0.5  # Safety margin from boundaries
        valid_start = False
        while not valid_start:
            x_rand = self.np_random.uniform(
                self.area_bounds[0][0] + boundary_margin, 
                self.area_bounds[0][1] - boundary_margin
            )
            y_rand = self.np_random.uniform(
                self.area_bounds[1][0] + boundary_margin, 
                self.area_bounds[1][1] - boundary_margin
            )
            dist_to_goal = np.linalg.norm([x_rand, y_rand])
            
            # Ensure we start at least some units away from goal
            if dist_to_goal > 2.0:
                valid_start = True
                
        self.state = np.zeros(self.scenario.state_dim, dtype=np.float32)
        self.state[0] = x_rand
        self.state[1] = y_rand
        # Initialize extra state dimensions
        self.state = self.scenario.init_extra_state(self.state, self.np_random)
        
        # 2. Randomize Obstacles
        # generate between 1 and max_obstacles in a random fashion
        num_obs = self.np_random.integers(self.max_obstacles//2, self.max_obstacles + 1)
        self.scenario.obstacles = []
        
        generated_count = 0
        attempts = 0
        max_attempts = 100 # Prevent infinite loops
        obstacle_margin = 0.2  # Keep obstacles away from boundaries
        
        # Try over and over until you find a good solution for the obstacles generation
        while generated_count < num_obs and attempts < max_attempts:
            
            orad = self.np_random.uniform(0.3, 0.8) # Radius between 0.3 and 0.8
            
            # Ensure obstacle stays within bounds
            ox = self.np_random.uniform(
                self.area_bounds[0][0] + orad + obstacle_margin, 
                self.area_bounds[0][1] - orad - obstacle_margin
            )
            oy = self.np_random.uniform(
                self.area_bounds[1][0] + orad + obstacle_margin, 
                self.area_bounds[1][1] - orad - obstacle_margin
            )
            
            # Distance checks
            d_goal = np.linalg.norm([ox, oy])                                  # Dist to goal (0,0)
            d_start = np.linalg.norm([ox - self.state[0], oy - self.state[1]]) # Dist to start
            
            # Ensure obstacle is not covering Goal or Start
            # Radius + safety margin
            if d_goal > (orad + 0.5) and d_start > (orad + 0.5):
                self.scenario.obstacles.append({'x': ox, 'y': oy, 'r': orad})
                generated_count += 1
            
            attempts += 1

        # Reset Goal (Fixed at origin)
        self.goal_pos = np.zeros(self.goal_dim, dtype=np.float32)
        
        return self._get_obs(), {}

    def step(self, action):
        """
        Simulate a single step of the ewnvironment
        """

        # Clip action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Integrate Dynamics
        self.state = self._rk4_step(self.state, action, self.dt)
        
        curr_pos = self.state[:2] # x, y coordinates
        goal_xy = self.goal_pos
        
        dist_to_goal = np.linalg.norm(curr_pos - goal_xy)
        
        # Safety Check
        has_crashed = self.scenario.check_collision(self.state)

        # Out-of-bound check
        out_of_bounds = False
        if self.state[0] - self.scenario.robot_radius < self.area_bounds[0][0] or self.state[0] + self.scenario.robot_radius > self.area_bounds[0][1] or \
           self.state[1] - self.scenario.robot_radius < self.area_bounds[1][0] or self.state[1] + self.scenario.robot_radius > self.area_bounds[1][1]:
            out_of_bounds = True
        
        # Reward
        x_error = self.scenario.get_error_state(self.state, self.goal_pos)
        cost_state = x_error.T @ self.scenario.Q @ x_error
        cost_action = action.T @ self.scenario.R @ action
        reward = - (cost_state + cost_action)
        
        terminated = False
        truncated = False
        
        # 3. Collision Logic
        if has_crashed:
            terminated = True
            info = {"is_safe": False, "reason": "collision_with_obstacle"}
            return self._get_obs(), reward, terminated, truncated, info

        if out_of_bounds:
            terminated = True
            info = {"is_safe": False, "reason": "out_of_bounds"}
            return self._get_obs(), reward, terminated, truncated, info
        
        # 4. Success Logic
        if dist_to_goal < self.goal_distance:
            terminated = True
            info = {"is_safe": True, "reason": "goal_reached"}
            return self._get_obs(), reward, terminated, truncated, info

        info = {"is_safe": True, "dist_goal": dist_to_goal}
        
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        """
        get observation of the environment
        in this case only the state of the system and the goal position w.r.t. the world frame
        """
        return np.concatenate([self.state, self.goal_pos]).astype(np.float32)

    def _rk4_step(self, x, u, dt):
        """
        dynamics integration
        """
        k1 = self.scenario.dynamics(x, u)
        k2 = self.scenario.dynamics(x + 0.5 * dt * k1, u)
        k3 = self.scenario.dynamics(x + 0.5 * dt * k2, u)
        k4 = self.scenario.dynamics(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def _render_frame(self):
        """
        render frame function to visualize the environment
        """
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((600, 600))
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((600, 600))
        canvas.fill((255, 255, 255))
        
        def to_pix(pos):
            return int(pos[0] * self.scale + 300), int(-pos[1] * self.scale + 300)

        # Draw Boundaries
        x_min_pix = to_pix([self.area_bounds[0][0], 0])[0]
        x_max_pix = to_pix([self.area_bounds[0][1], 0])[0]
        y_min_pix = to_pix([0, self.area_bounds[1][0]])[1]
        y_max_pix = to_pix([0, self.area_bounds[1][1]])[1]
        
        # Draw boundary rectangle (red lines)
        pygame.draw.rect(canvas, (255, 0, 0), 
                        (x_min_pix, y_max_pix, x_max_pix - x_min_pix, y_min_pix - y_max_pix), 
                        3)  # 3 pixel width

        # Draw Goal
        goal_pix = to_pix(self.goal_pos[:2])
        # Draw goal zone (0.25 unit threshold as light green circle)
        pygame.draw.circle(canvas, (150, 255, 150), goal_pix, int(0.25 * self.scale), 2)
        # Draw goal center
        pygame.draw.circle(canvas, (50, 200, 50), goal_pix, 8) # Green Goal center

        # Draw Obstacles
        for obs in self.scenario.obstacles:
            pos = to_pix([obs['x'], obs['y']])
            rad = int(obs['r'] * self.scale)
            # Make sure radius is at least 1 pixel
            if rad < 1: rad = 1
            pygame.draw.circle(canvas, (200, 50, 50), pos, rad) # Red Obstacles

        # Draw Robot with correct radius
        robot_pix = to_pix(self.state[:2])
        robot_radius_pixels = int(self.scenario.robot_radius * self.scale)
        pygame.draw.circle(canvas, (100, 100, 255), robot_pix, robot_radius_pixels) # Blue Robot

        # Draw heading indicator for systems with theta (state_dim > 2)
        if self.scenario.state_dim > 2:
            heading = self.state[2]
            arrow_len = self.scenario.robot_radius * 2.5
            end_x = self.state[0] - arrow_len * np.cos(heading)
            end_y = self.state[1] - arrow_len * np.sin(heading)
            end_pix = to_pix([end_x, end_y])
            pygame.draw.line(canvas, (0, 0, 0), robot_pix, end_pix, 3)

        self.window.blit(canvas, (0, 0))
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """
        close the rendered environment
        """
        if self.window is not None:
            pygame.quit()