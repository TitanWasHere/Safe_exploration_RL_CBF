import casadi as ca
import numpy as np

class CasadiSafetyFilter:
    def __init__(self, scenario_type="unicycle", robot_radius=0.3, c_b=10.0, workspace_bounds=None):
        self.scenario_type = scenario_type
        self.robot_radius = robot_radius
        self.c_b = c_b
        
        if workspace_bounds is None:
            bounds_square = 20.0
            self.workspace_bounds = {'x_min': -bounds_square/2, 'x_max': bounds_square/2, 'y_min': -bounds_square/2, 'y_max': bounds_square/2}
        else:
            self.workspace_bounds = workspace_bounds

        self._setup_casadi()

    def _setup_casadi(self):
        obs_x = ca.SX.sym('obs_x')
        obs_y = ca.SX.sym('obs_y')
        obs_r = ca.SX.sym('obs_r')
        goal_x = ca.SX.sym('goal_x')
        goal_y = ca.SX.sym('goal_y')
        
        if self.scenario_type == "single_integrator":
            x = ca.SX.sym('x', 2)
            px, py = x[0], x[1]
            g_sym = ca.SX.eye(2)
            safe_dist = obs_r + self.robot_radius
            h = (px - obs_x)**2 + (py - obs_y)**2 - safe_dist**2


        elif self.scenario_type == "double_integrator":
            x = ca.SX.sym('x', 4)
            px, py, vx, vy = x[0], x[1], x[2], x[3]
            g_sym = ca.SX.zeros(4, 2)
            g_sym[2,0] = 1; g_sym[3,1] = 1
            safe_dist = obs_r + self.robot_radius
            h_pos = (px - obs_x)**2 + (py - obs_y)**2 - safe_dist**2
            gamma = 1.0
            h_dot = 2*(px - obs_x)*vx + 2*(py - obs_y)*vy
            h = h_pos + gamma * h_dot

        elif self.scenario_type == "unicycle":
            x = ca.SX.sym('x', 3)
            px, py, theta = x[0], x[1], x[2]
            g_sym = ca.vertcat(
                ca.horzcat(ca.cos(theta), 0),
                ca.horzcat(ca.sin(theta), 0),
                ca.horzcat(0,             1)
            )
            safe_dist = obs_r + self.robot_radius
            h = (px - obs_x)**2 + (py - obs_y)**2 - safe_dist**2

        else:
            raise ValueError("Unknown Scenario")

        h_clamped = ca.fmax(h, 0.001)
        b_x = 1.0 / h_clamped
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
        gx = obs[-2]
        gy = obs[-1]

        if self.scenario_type == "double_integrator":
            state = obs[:4]
        elif self.scenario_type == "unicycle":
            state = obs[:3]
        else:
            state = obs[:2]

        total_grad_B = np.zeros_like(state)
        min_h_val = 100.0

        for o in nearby_obstacles:
            grad_i, h_val = self.f_grad_B(state, o['x'], o['y'], o['r'], gx, gy)
            grad_i = np.array(grad_i).flatten()
            total_grad_B += grad_i
            
            if h_val < min_h_val:
                min_h_val = float(h_val)
        
        px, py = state[0], state[1]
        boundary_margin = 0.3
        
        if px < self.workspace_bounds['x_min'] + boundary_margin:
            violation = self.workspace_bounds['x_min'] + boundary_margin - px
            total_grad_B[0] -= 1000.0 * violation * (1.0 + violation)
        elif px > self.workspace_bounds['x_max'] - boundary_margin:
            violation = px - (self.workspace_bounds['x_max'] - boundary_margin)
            total_grad_B[0] += 1000.0 * violation * (1.0 + violation)
        
        if py < self.workspace_bounds['y_min'] + boundary_margin:
            violation = self.workspace_bounds['y_min'] + boundary_margin - py
            total_grad_B[1] -= 1000.0 * violation * (1.0 + violation)
        elif py > self.workspace_bounds['y_max'] - boundary_margin:
            violation = py - (self.workspace_bounds['y_max'] - boundary_margin)
            total_grad_B[1] += 1000.0 * violation * (1.0 + violation)

        g_val = np.array(self.f_g(state))
        u_corr = -0.5 * self.c_b * (g_val.T @ total_grad_B)
        
        u_safe = u_nom + u_corr

        return u_safe, min_h_val