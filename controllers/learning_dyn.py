import numpy as np

class LearnerDynamics:
    def __init__(self, scenario, learning_rate=0.5, history_size=50, 
                 integration_window=10):
        self.scenario = scenario
        self.gamma = learning_rate
        self.integration_window = integration_window
        
        self.state_dim = self.scenario.state_dim
        self.theta_dim = self.scenario.theta_dim

        self.theta_hat = np.zeros(self.theta_dim, dtype=np.float32)
        
        self.history_size = history_size
        self.buffer = []
        self.trajectory_window = []


    def predict_f(self, x):
            f_kin = self.scenario.get_kinematics(x)
            Y = self.scenario.get_regressor(x)
            return f_kin + Y @ self.theta_hat

    def reset_window(self):
        self.trajectory_window = []

    def update(self, x_old, x_new, u, dt, g_matrix):
        
        self.trajectory_window.append({
            'x_old': x_old.copy(),
            'x_new': x_new.copy(),
            'u': u.copy(),
            'dt': dt,
            'g': g_matrix.copy()
        })
        
        if len(self.trajectory_window) > self.integration_window:
            self.trajectory_window.pop(0)
        
        if len(self.trajectory_window) >= self.integration_window:
            # Composite Adaptation / Concurrent Learning Integral Form
            
            integral_Y = np.zeros((self.state_dim, self.theta_dim))
            integral_gu = np.zeros(self.state_dim)
            integral_f_kin = np.zeros(self.state_dim)
            
            for item in self.trajectory_window:
                x_i = item['x_old']
                u_i = item['u']
                dt_i = item['dt']
                g_i = item['g']
                
                Y_i = self.scenario.get_regressor(x_i)
                f_kin_i = self.scenario.get_kinematics(x_i)
                gu_i = (g_i @ u_i).flatten()
                
                integral_Y += Y_i * dt_i
                integral_gu += gu_i * dt_i
                integral_f_kin += f_kin_i * dt_i
            
            x_start = self.trajectory_window[0]['x_old']
            x_end = self.trajectory_window[-1]['x_new']
            
            delta_x = x_end - x_start
            target = delta_x - integral_gu - integral_f_kin
            
            # Store the integral regressor and target for batch learning
            if len(self.buffer) < self.history_size:
                self.buffer.append((integral_Y, target))
            else:
                self.buffer.pop(0)
                self.buffer.append((integral_Y, target))
        else:
            # Fallback to instantaneous update if window not full
            Y = self.scenario.get_regressor(x_old)
            gu_term = (g_matrix @ u).flatten()
            f_kin = self.scenario.get_kinematics(x_old)
            target = (x_new - x_old) - (gu_term * dt) - (f_kin * dt)
            regressor_dt = Y * dt
            
            if len(self.buffer) < self.history_size:
                self.buffer.append((regressor_dt, target))
            else:
                self.buffer.pop(0)
                self.buffer.append((regressor_dt, target))

        # Batch gradient update on all stored data (concurrent learning)
        update_grad = np.zeros_like(self.theta_hat)
        for Y_stored, target_stored in self.buffer:
            # Residual: error = target - Y @ theta
            error = target_stored - (Y_stored @ self.theta_hat)
            # Gradient: dL/dtheta = -Y.T @ error
            update_grad += Y_stored.T @ error
        
        # Normalize by buffer size for stability
        if len(self.buffer) > 0:
            update_grad /= len(self.buffer)
        
        # Gradient clipping for numerical stability
        grad_norm = np.linalg.norm(update_grad)
        if grad_norm > 10.0:
            update_grad = update_grad * (10.0 / grad_norm)
        
        # Parameter update
        self.theta_hat += self.gamma * update_grad
        
        # Bounds on parameter estimates (prevent unbounded growth)
        self.theta_hat = np.clip(self.theta_hat, -20.0, 20.0)