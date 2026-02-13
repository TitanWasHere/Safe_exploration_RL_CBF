"""
This file contains the class and methods definition for a classical CBF used to ensure safety
"""
import casadi as ca
import numpy as np

class CasadiSafetyFilter:
    """
    Generic Safety Filter that adapts to any scenario provided.
    """
    def __init__(self, scenario, workspace_bounds=None):
        self.scenario = scenario
        self.robot_radius = scenario.robot_radius
        self.c_b = scenario.c_b
        self.state_dim = scenario.state_dim
        
        # Ensure workspace bounds exists
        if workspace_bounds is None:
            bounds_square = 20.0
            self.workspace_bounds = {
                'x_min': -bounds_square/2, 'x_max': bounds_square/2, 
                'y_min': -bounds_square/2, 'y_max': bounds_square/2
            }
        else:
            self.workspace_bounds = workspace_bounds

        # Initialize the casadi symbolic solver
        self._setup_casadi()

    def _setup_casadi(self):
        # Get the physics from the scenario
        model = self.scenario.get_casadi_model()
        
        x = model['x_sym']
        g_sym = model['g_sym']
        h_obs_eqn = model['h_obs']
        obs_x, obs_y, obs_r = model['obs_params']
        
        # Extract position indices for wall checks (px, py)
        idx_x, idx_y = model['pos_indices']
        px, py = x[idx_x], x[idx_y]
        
        # Define Goal Symbols (for B(0) calculation)
        goal_x = ca.SX.sym('goal_x')
        goal_y = ca.SX.sym('goal_y')
        
        # Build barrier function
        # Calculate h at current state (h_obs) and at goal (h_obs_0)
        
        x_goal_val = ca.SX.zeros(self.state_dim)
        x_goal_val[idx_x] = goal_x
        x_goal_val[idx_y] = goal_y
        
        # Substitute x with x_goal in the h equation
        h_obs_0 = ca.substitute(h_obs_eqn, x, x_goal_val)

        # Clamp to prevent division by zero
        b_x = ca.fmin(1.0 / ca.fmax(h_obs_eqn, 0.001), 100.0)
        b_0 = ca.fmin(1.0 / ca.fmax(h_obs_0, 0.001), 100.0)

        diff = b_x - b_0
        B_obs = ca.fmax(diff, 0)**2
        
        # Gradient of B with respect to state x
        grad_B_obs = ca.jacobian(B_obs, x)
        
        # Create Function
        self.f_grad_B = ca.Function('f_grad_B', 
                                    [x, obs_x, obs_y, obs_r, goal_x, goal_y], 
                                    [grad_B_obs, h_obs_eqn])
        self.f_g = ca.Function('f_g', [x], [g_sym])

        # Static walls
        walls = []
        walls.append((px - self.workspace_bounds['x_min']) - self.robot_radius) # Left
        walls.append((self.workspace_bounds['x_max'] - px) - self.robot_radius) # Right
        walls.append((py - self.workspace_bounds['y_min']) - self.robot_radius) # Bottom
        walls.append((self.workspace_bounds['y_max'] - py) - self.robot_radius) # Top

        total_grad_bounds = 0
        min_h_bounds = ca.SX(100.0)

        for h_wall in walls:
            # b(x)
            b_w = 1.0 / ca.fmax(h_wall, 0.001)
            
            # b(0) - value at goal
            h_wall_0 = ca.substitute(h_wall, x, x_goal_val)
            b_w0 = 1.0 / ca.fmax(h_wall_0, 0.001)
            
            B_w = (b_w - b_w0)**2
            total_grad_bounds += ca.jacobian(B_w, x)
            min_h_bounds = ca.fmin(min_h_bounds, h_wall)

        self.f_grad_bounds = ca.Function('f_grad_bounds',
                                         [x, goal_x, goal_y],
                                         [total_grad_bounds, min_h_bounds])

    def get_safe_action(self, obs, u_nom, nearby_obstacles, R_inv=None, action_max=50.0):
            """
            Compute safe action respecting both CBF and action limits.
            
            Args:
                obs: observation [state, goal_x, goal_y]
                u_nom: nominal (unconstrained) control
                nearby_obstacles: list of nearby obstacles
                R_inv: inverse of control cost matrix
                action_max: maximum action magnitude (DEFAULT: 50.0)
            """
            gx = obs[self.state_dim]
            gy = obs[self.state_dim + 1]
            state = obs[:self.state_dim]

            if R_inv is None:
                R_inv = np.eye(u_nom.shape[0])

            total_grad_B = np.zeros_like(state)
            min_h_val = 100.0

            # Compute distance to goal for adaptive behavior
            goal_pos = np.array([gx, gy])
            robot_pos = state[:2]
            dist_to_goal = np.linalg.norm(robot_pos - goal_pos)
            
            # Accumulate gradient from dynamic obstacles
            for o in nearby_obstacles:
                grad_i, h_val = self.f_grad_B(state, o['x'], o['y'], o['r'], gx, gy)
                grad_i = np.array(grad_i).flatten()
                total_grad_B += grad_i
                if h_val < min_h_val:
                    min_h_val = float(h_val)
            
            # Accumulate gradient from static obstacles (walls)
            grad_bounds, h_bounds = self.f_grad_bounds(state, gx, gy)
            grad_bounds = np.array(grad_bounds).flatten()
            
            total_grad_B += grad_bounds
            if h_bounds < min_h_val:
                min_h_val = float(h_bounds)

            g_val = np.array(self.f_g(state))
            
            # Adaptive correction strength: reduce safety margin when very close to goal
            # This allows more aggressive maneuvering in the final approach
            adaptive_c_b = self.c_b
            if dist_to_goal < 1.0:
                # Reduce safety barrier more aggressively near goal
                adaptive_c_b = self.c_b * (0.7 * dist_to_goal)
            
            # Standard LCBF Control Law with action limit constraint
            u_corr = -0.5 * adaptive_c_b * (R_inv @ (g_val.T @ total_grad_B))
            u_safe = u_nom + u_corr
            
            # Clip to action limits
            u_safe = np.clip(u_safe, -action_max, action_max)
            
            return u_safe, min_h_val