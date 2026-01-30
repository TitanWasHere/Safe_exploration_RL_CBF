"""
File in which we define different robot scenarios you can use in the custom env for safe RL.
"""

import numpy as np
from abc import ABC, abstractmethod

class ScenarioStrategy(ABC):
    def __init__(self):
        self.state_dim = None # State dimension
        self.action_dim = None # Action dimension
        self.action_max = 10.0 # Default max action
        self.obstacles = [] # List of obstacles, each defined as a dict with keys 'x', 'y', 'r'
        self.robot_radius = 0.2 # Default robot radius

    # Abstract methods needed for dynamics definition
    @abstractmethod
    def get_initial_state(self): pass

    @abstractmethod
    def get_f(self, x): pass

    @abstractmethod
    def get_g(self, x): pass

    # General dynamics method
    def dynamics(self, x, u):
        return self.get_f(x) + self.get_g(x) @ u

    # Function to check closest obstacle safety margin
    def get_h(self, x):
        """
        Returns the safety margin of the closest obstacle.
        h(x) = dist_to_obs^2 - (robot_radius + obstacle_radius)^2
        """
        if not self.obstacles:
            return 100.0  # No obstacles, always safe

        min_h = float('inf')

        # position of the robot
        px, py = x[0], x[1]

        for obs in self.obstacles:
            # h_i = ||x - c_i||^2 - (r_robot + r_obs)^2
            dist_sq = (px - obs['x'])**2 + (py - obs['y'])**2
            h_i = dist_sq - (self.robot_radius + obs['r'])**2
            
            if h_i < min_h:
                min_h = h_i
        
        return min_h
    
    def get_near_obstacle(self, x, search_radius=2.0):
        """
        Returns the nearest obstacle within a certain search radius.
        If no obstacle is found, returns None.
        """
        px, py = x[0], x[1]
        nearest_obs = []
        
        for obs in self.obstacles:
            dist_sq = (px - obs['x'])**2 + (py - obs['y'])**2
            if dist_sq < (search_radius + obs['r'] + self.robot_radius)**2:
                nearest_obs.append(obs)
        
        # Sort by squared distance (closest first)
        nearest_obs.sort(key=lambda o: (px - o['x'])**2 + (py - o['y'])**2)

        return nearest_obs

class SingleIntegrator(ScenarioStrategy):
    """
    Simple 2D single integrator robot. The robot can move in any direction directly.
    we have direct control over velocity.
    """
    def __init__(self):
        super().__init__()
        self.state_dim = 2
        self.action_dim = 2
    
    def get_initial_state(self):
        # x, y
        return np.array([-3.0, -1.0], dtype=np.float32)

    def get_f(self, x):
        return np.array([-0.6 * x[0] - x[1], x[0]**3], dtype=np.float32)

    def get_g(self, x):
        return np.eye(2, dtype=np.float32)

class DoubleIntegrator(ScenarioStrategy):
    """
    Simple 2D double integrator robot. The robot has position and velocity states.
    The control inputs are accelerations in x and y directions.
    """
    def __init__(self):
        super().__init__()
        self.state_dim = 4
        self.action_dim = 2
        
    def get_initial_state(self):
        # x, y, vx, vy
        return np.array([-3.0, -1.0, 0.0, 0.0], dtype=np.float32)

    def get_f(self, x):
        drag = 0.5
        return np.array([x[2], x[3], -drag*x[2], -drag*x[3]], dtype=np.float32)

    def get_g(self, x):
        g = np.zeros((4, 2), dtype=np.float32)
        g[2, 0] = 1.0
        g[3, 1] = 1.0
        return g

class Unicycle(ScenarioStrategy):
    """
    Simple 2D unicycle robot. The robot has position and orientation states. The control inputs are linear and angular velocities.
    As usual the unicycle is subject to non-holonomic constraints.
    """

    # def sym

    def __init__(self):
        super().__init__()
        self.state_dim = 3
        self.action_dim = 2
        self.action_max = 2.0 # Lower max action for unicycle
        
    def get_initial_state(self):
        # x, y, theta
        return np.array([-3.0, -1.0, 0.0], dtype=np.float32)

    def get_f(self, x):
        return np.array([0.0, -0.1, 0.0], dtype=np.float32) # Drift in y

    def get_g(self, x):
        theta = x[2]
        return np.array([
            [np.cos(theta), 0],
            [np.sin(theta), 0],
            [0,             1]
        ], dtype=np.float32)