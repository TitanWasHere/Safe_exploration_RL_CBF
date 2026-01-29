"""
CBF Safety filter implementation using CasADi for symbolic differentiation.
Implements barrier functions for different robot dynamics:
- Single Integrator
- Unicycle
- Double Integrator
"""

import casadi as ca
import numpy as np

class CasadiSafetyFilter:
    def __init__(self, scenario_type="unicycle", robot_radius=0.3, c_b=10.0):
        self.scenario_type = scenario_type
        self.robot_radius = robot_radius
        self.c_b = c_b

        # setupt the casadi functions
        self._setup_casadi()

    def _setup_casadi(self):
        """
        Setup CasADi functions for barrier function gradients and dynamics matrix g(x).
        """
        # Define Symbolic Variables
        
        # Dynamic inputs for one obstacle
        # we define the equations for a single obstacle, and will sum over multiple obstacles later
        obs_x = ca.SX.sym('obs_x')
        obs_y = ca.SX.sym('obs_y')
        obs_r = ca.SX.sym('obs_r')

        goal_x = ca.SX.sym('goal_x')
        goal_y = ca.SX.sym('goal_y')
        
        # State definitions
        if self.scenario_type == "single_integrator":
            x = ca.SX.sym('x', 2) # [x, y]
            px, py = x[0], x[1]
            
            # Dynamics matrices g(x)
            g_sym = ca.SX.eye(2)
            
            # Barrier Function
            # h = ||p - obs||^2 - (r_rob + r_obs)^2
            safe_dist = obs_r + self.robot_radius
            h = (px - obs_x)**2 + (py - obs_y)**2 - safe_dist**2

        elif self.scenario_type == "double_integrator":
            x = ca.SX.sym('x', 4) # [x, y, vx, vy]
            px, py, vx, vy = x[0], x[1], x[2], x[3]
            
            g_sym = ca.SX.zeros(4, 2)
            g_sym[2,0] = 1; g_sym[3,1] = 1
            
            # Position Barrier
            safe_dist = obs_r + self.robot_radius
            h_pos = (px - obs_x)**2 + (py - obs_y)**2 - safe_dist**2
            
            # Relative Degree 1 Extension
            # given that we control acceleration we need to extend the barrier function to 
            # consider velocity as well
            # h_new = h_pos + gamma * h_dot
            gamma = 1.0 # tuning parameter, determines how early we react to velocity, and high value means we react earlier
            h_dot = 2*(px - obs_x)*vx + 2*(py - obs_y)*vy
            h = h_pos + gamma * h_dot

        elif self.scenario_type == "unicycle":
            x = ca.SX.sym('x', 3) # [x, y, theta]
            px, py, theta = x[0], x[1], x[2]
            
            # Unicycle g(x)
            g_sym = ca.vertcat(
                ca.horzcat(ca.cos(theta), 0),
                ca.horzcat(ca.sin(theta), 0),
                ca.horzcat(0,             1)
            )
            
            safe_dist = obs_r + self.robot_radius
            h = (px - obs_x)**2 + (py - obs_y)**2 - safe_dist**2

        else:
            raise ValueError("Unknown Scenario")

        # Compute Gradient of One Barrier

        # Define b(x) (Standard CBF)
        # Clamp h to be slightly positive to prevent DivisionByZero during initialization
        h_clamped = ca.fmax(h, 0.001)
        b_x = 1.0 / h_clamped
        
        # Define b(0) (Barrier at the goal)
        if self.scenario_type == "single_integrator":
            dist_sq_at_0 = (goal_x - obs_x)**2 + (goal_y - obs_y)**2
            h_at_0 = dist_sq_at_0 - safe_dist**2
        elif self.scenario_type == "unicycle":
            dist_sq_at_0 = (goal_x - obs_x)**2 + (goal_y - obs_y)**2
            h_at_0 = dist_sq_at_0 - safe_dist**2
        elif self.scenario_type == "double_integrator":
            h_pos_0 = (goal_x - obs_x)**2 + (goal_y - obs_y)**2 - safe_dist**2
            h_dot_0 = 0
            h_at_0 = h_pos_0 + gamma * h_dot_0

        # Clamp h(0) to match logic
        b_0 = 1.0 / ca.fmax(h_at_0, 0.001)

        # Construct LCBF B(x) = (b(x) - b(0))^2 
        B_i = (b_x - b_0)**2

        # Gradient w.r.t state x
        grad_B_i = ca.jacobian(B_i, x)

        # We calculate the Gradient of B and the Matrix g(x)
        # We will do the summation and multiplication in Python
        # take symbolic graph and compile it into numerical functions
        self.f_grad_B = ca.Function('f_grad_B', 
                                    [x, obs_x, obs_y, obs_r, goal_x, goal_y], 
                                    [grad_B_i, h])
        
        self.f_g = ca.Function('f_g', [x], [g_sym])

    def get_safe_action(self, obs, u_nom, nearby_obstacles):
        """
        Sum over multiple obstacles.
        u = u_nom - 0.5 * c_b * g(x).T * (sum(grad_B_i)).T
        """

        gx = obs[-2]
        gy = obs[-1]

        # Parse Observation
        if self.scenario_type == "double_integrator":
            state = obs[:4]
        elif self.scenario_type == "unicycle":
            state = obs[:3]
        else:
            state = obs[:2]

        # Accumulate Gradients from all nearby obstacles
        total_grad_B = np.zeros_like(state)
        min_h_val = 100.0

        for o in nearby_obstacles:
            # Call CasADi for this specific obstacle
            grad_i, h_val = self.f_grad_B(state, o['x'], o['y'], o['r'], gx, gy)
            
            # Convert to numpy (CasADi returns DM/SX)
            grad_i = np.array(grad_i).flatten()
            
            # B(x) = sum(B_i(x)) => grad B = sum(grad B_i)
            total_grad_B += grad_i
            
            if h_val < min_h_val:
                min_h_val = float(h_val)

        # Compute g(x) for current state
        g_val = np.array(self.f_g(state))

        # Apply Control Law
        # Correction = -0.5 * c_b * g^T * total_grad^T
        # Shapes: g.T is (2, n), grad is (n,)
        u_corr = -0.5 * self.c_b * (g_val.T @ total_grad_B)
        
        u_safe = u_nom + u_corr

        return u_safe, min_h_val