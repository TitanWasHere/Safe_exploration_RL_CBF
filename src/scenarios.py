"""
File in which we define different robot scenarios you can use in the custom env for safe RL.
"""

import numpy as np
import casadi as ca
from abc import ABC, abstractmethod

class ScenarioStrategy(ABC):
    def __init__(self):
        self.state_dim = None    # State dimension (n)
        self.theta_dim = None    # Theta dimension used for online dynamics learning
        self.action_dim = None   # Action dimension (m)
        self.goal_dim = 2        # Goal dimension (position target, always 2 for x,y)
        self.action_max = 10.0   # Default max action
        self.obstacles = []      # List of obstacles, each defined as a dict with keys 'x', 'y', 'r'
        self.robot_radius = 0.15 # Reduced from 0.2 to fit through tighter spaces
        self.Q = np.eye(2)       # Default error penalty weights (overridden per scenario)
        self.R = np.eye(2)       # Default cost penalty weights (overridden per scenario)
        self.c_b = 1.0           # Safeguarding gain used in CBF

    @abstractmethod
    def get_f(self, x): pass

    @abstractmethod
    def get_g(self, x): pass

    @abstractmethod
    def get_casadi_model(self):
        """
        Returns the CasADi symbolic variables and expressions for this robot.
        Returns:
            x_sym (ca.SX): State vector symbol
            g_sym (ca.SX): Input matrix symbol
            h_obs_func (function): Function that takes (x, obs_x, obs_y, obs_r) and returns symbolic h
        """
        pass

    @abstractmethod
    def get_regressor(self):
        """
        Return the regressor matrix for dynamics online learning
        """
        pass

    @abstractmethod
    def get_kinematics(self):
        """
        Return the know kinematics part of the system independent of parameters
        """
        pass

    # General dynamics method
    def dynamics(self, x, u):
        return self.get_f(x) + self.get_g(x) @ u

    def get_error_state(self, state, goal_pos):
        """
        Compute error state for value function approximation.
        Default: subtract goal from first goal_dim components, keep remaining.
        For state=[x,y,...], goal=[gx,gy]: error=[x-gx, y-gy, ...]
        """
        error = state.copy().astype(np.float64)
        gdim = min(len(goal_pos), self.state_dim)
        error[:gdim] -= goal_pos[:gdim]
        return error

    def init_extra_state(self, state, rng):
        """
        Initialize non-position state components during env reset.
        Override in subclasses for systems with extra states (e.g., heading).
        """
        return state

    def check_collision(self, x):
        """
        Binary check: Returns True if the robot has crashed into ANY obstacle.
        Used strictly for environment termination, not for control gradients.
        """
        if not self.obstacles:
            return False

        px, py = x[0], x[1]
        
        for obs in self.obstacles:
            # Calculate distance squared
            dist_sq = (px - obs['x'])**2 + (py - obs['y'])**2
            
            # Crash condition: dist < (r_robot + r_obs)
            # We use squared values to avoid expensive sqrt() calls
            min_safe_dist_sq = (self.robot_radius + obs['r'])**2
            
            # If we are closer than the physical limit, we crashed.
            if dist_sq <= min_safe_dist_sq:
                return True
                
        return False

    
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

class SingleIntegratorSystem(ScenarioStrategy):
    """
    First example made, Single integrator system (controls over velocity) with no drift
    State: [px, py] (2D position)
    Control: [vx, vy] (velocity)
    Dynamics: x_dot = u (pure velocity control)
    """
    def __init__(self):
        super().__init__()
        self.state_dim = 2
        self.theta_dim = 0  # No uncertain parameters for single integrator
        self.action_dim = 2
        self.action_max = 5.0
        # for Q and R use default
        self.c_b = 0.01
        self.theta_true = np.array([], dtype=np.float32)  # No true parameters
    
    def get_f(self, x):
        return np.array([0,0], dtype=np.float32)

    def get_g(self, x):
        return np.eye(2, dtype=np.float32)
    
    def get_casadi_model(self):
        """
        Returns CasADi symbolic model for single integrator.
        """
        # State: [px, py]
        x_sym = ca.SX.sym('x', 2)
        
        # Drift: f(x) = 0
        f_sym = ca.SX.zeros(2)
        
        # Control matrix: g(x) = I_2
        g_sym = ca.SX.eye(2)
        
        # Obstacle barrier function h(x) = ||pos - obs||^2 - safe_dist^2
        obs_x = ca.SX.sym('obs_x')
        obs_y = ca.SX.sym('obs_y')
        obs_r = ca.SX.sym('obs_r')
        safe_dist = obs_r + self.robot_radius
        
        h_obs = (x_sym[0] - obs_x)**2 + (x_sym[1] - obs_y)**2 - safe_dist**2
        
        return {
            'x_sym': x_sym,
            'f_sym': f_sym,
            'g_sym': g_sym,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'obs_params': (obs_x, obs_y, obs_r),
            'h_obs': h_obs,
            'pos_indices': (0, 1),  # indices for px, py in state vector
        }
    
    def get_kinematics(self, x):
        # Single integrator has no drift dynamics
        return np.zeros_like(x)
    
    def get_regressor(self, x):
        # No uncertain parameters, return empty regressor
        return np.zeros((self.state_dim, self.theta_dim), dtype=np.float32)


class UnderactuatedSystem(ScenarioStrategy):
    """
    Second example made, single integrator but underactuated with drift
    State: [x1, x2]
    Control: u (scalar)
    Dynamics: x_dot_1 = -0.6x_1 - x_2, x_dot_2 = x_1^3 + x_2·u
    """

    def __init__(self):
        super().__init__()
        self.state_dim = 2
        self.theta_dim = 3
        self.action_dim = 1
        self.action_max = 5.0
        self.R = np.array([[1.0]])  # Scalar control cost
        # default Q
        self.c_b = 1.0
        # True parameters: theta1=-0.6, theta2=-1.0, theta3=1.0
        self.theta_true = np.array([-0.6, -1.0, 1.0], dtype=np.float32)
    
    def get_f(self, x):
        return np.array([-0.6 * x[0] - x[1], x[0]**3], dtype=np.float32)

    def get_g(self, x):
        return np.array([[0], [x[1]]], dtype=np.float32)
    
    def get_casadi_model(self):
        """
        Returns CasADi symbolic model for underactuated system.
        x_dot = f(x) + g(x)u where f has nonlinear drift and g depends on x_2
        """
        # State: [x1, x2]
        x_sym = ca.SX.sym('x', 2)
        
        # Drift: f(x) = [-0.6*x1 - x2, x1^3]
        f_sym = ca.vertcat(-0.6 * x_sym[0] - x_sym[1], x_sym[0]**3)
        
        # Control matrix: g(x) = [0; x2] (underactuated)
        g_sym = ca.vertcat(0, x_sym[1])
        
        # Obstacle barrier function
        obs_x = ca.SX.sym('obs_x')
        obs_y = ca.SX.sym('obs_y')
        obs_r = ca.SX.sym('obs_r')
        safe_dist = obs_r + self.robot_radius
        
        h_obs = (x_sym[0] - obs_x)**2 + (x_sym[1] - obs_y)**2 - safe_dist**2
        
        return {
            'x_sym': x_sym,
            'f_sym': f_sym,
            'g_sym': g_sym,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'obs_params': (obs_x, obs_y, obs_r),
            'h_obs': h_obs,
            'pos_indices': (0, 1),
        }
    
    def get_kinematics(self, x):
        # This system has no "known" kinematic part independent of parameters
        return np.zeros_like(x)
    
    def get_regressor(self, x):
        # The Y matrix for f(x) = [-0.6x1 - x2, x1^3]
        # Matches Paper Example 1
        Y = np.zeros((self.state_dim, self.theta_dim), dtype=np.float32)
        Y[0, 0] = x[0]    # theta1 (-0.6)
        Y[0, 1] = x[1]    # theta2 (-1.0)
        Y[1, 2] = x[0]**3 # theta3 (1.0)
        return Y


class UnicycleSystem(ScenarioStrategy):
    """
    Unicycle (differential-drive) robot with uncertain drag.
    State: [x, y, theta]  (position + heading)
    Control: [v, omega]   (linear velocity, angular velocity)
    
    True dynamics:
        dx/dt  = v * cos(theta) + theta1 * x   (drag on x)
        dy/dt  = v * sin(theta) + theta2 * y   (drag on y)
        dth/dt = omega          + theta3 * theta (heading damping)
    
    Known kinematics (in g(x)*u): [[cos(th), 0], [sin(th), 0], [0, 1]]
    Uncertain drift: Y(x)*theta  where Y = diag(x, y, theta)
    """

    def __init__(self):
        super().__init__()
        self.state_dim = 3
        self.theta_dim = 3
        self.action_dim = 2
        self.action_max = 5.0
        self.goal_dim = 2  # Position-only goal (x, y)

        # Cost matrices: penalize position heavily, heading lightly
        self.Q = np.diag([1.0, 1.0, 0.1])
        self.R = np.diag([0.5, 0.5])
        self.c_b = 0.5

        # True uncertain parameters (small damping)
        self.theta_true = np.array([-0.1, -0.1, -0.05], dtype=np.float32)

    def get_f(self, x):
        """True drift: uncertain drag terms."""
        return np.array([
            self.theta_true[0] * x[0],
            self.theta_true[1] * x[1],
            self.theta_true[2] * x[2]
        ], dtype=np.float32)

    def get_g(self, x):
        """Known input matrix: standard unicycle kinematics."""
        theta = x[2]
        return np.array([
            [np.cos(theta), 0.0],
            [np.sin(theta), 0.0],
            [0.0,           1.0]
        ], dtype=np.float32)

    def get_casadi_model(self):
        """CasADi symbolic model for CBF safety filter."""
        x_sym = ca.SX.sym('x', 3)
        theta_s = x_sym[2]

        # Drift (true dynamics)
        f_sym = ca.vertcat(
            self.theta_true[0] * x_sym[0],
            self.theta_true[1] * x_sym[1],
            self.theta_true[2] * x_sym[2]
        )

        # Input matrix
        g_sym = ca.vertcat(
            ca.horzcat(ca.cos(theta_s), 0),
            ca.horzcat(ca.sin(theta_s), 0),
            ca.horzcat(0, 1)
        )

        # Obstacle barrier (position-only)
        obs_x = ca.SX.sym('obs_x')
        obs_y = ca.SX.sym('obs_y')
        obs_r = ca.SX.sym('obs_r')
        safe_dist = obs_r + self.robot_radius

        h_obs = (x_sym[0] - obs_x)**2 + (x_sym[1] - obs_y)**2 - safe_dist**2

        return {
            'x_sym': x_sym,
            'f_sym': f_sym,
            'g_sym': g_sym,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'obs_params': (obs_x, obs_y, obs_r),
            'h_obs': h_obs,
            'pos_indices': (0, 1),
        }

    def get_error_state(self, state, goal_pos):
        """
        Error state for unicycle: [x-gx, y-gy, theta].
        Heading stays as-is since goal heading is implicitly zero.
        """
        error = state.copy().astype(np.float64)
        error[0] -= goal_pos[0]
        error[1] -= goal_pos[1]
        # theta remains unchanged (target heading = 0)
        return error

    def init_extra_state(self, state, rng):
        """Randomize initial heading."""
        state[2] = rng.uniform(-np.pi, np.pi)
        return state

    def get_kinematics(self, x):
        """No known drift (all drift is uncertain via regressor)."""
        return np.zeros(self.state_dim, dtype=np.float32)

    def get_regressor(self, x):
        """
        Regressor Y(x) such that f(x) = Y(x) * theta.
        Y = diag(x, y, theta)
        """
        Y = np.zeros((self.state_dim, self.theta_dim), dtype=np.float32)
        Y[0, 0] = x[0]
        Y[1, 1] = x[1]
        Y[2, 2] = x[2]
        return Y