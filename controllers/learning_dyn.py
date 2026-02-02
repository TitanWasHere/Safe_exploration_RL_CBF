import numpy as np

class LearnerDynamics:
    def __init__(self, scenario_type="single_integrator", learning_rate=0.5, history_size=50, 
                 integration_window=10):
        self.scenario_type = scenario_type
        self.gamma = learning_rate
        self.integration_window = integration_window
        
        if scenario_type == "single_integrator":
            self.theta_dim = 3
            self.state_dim = 2
        elif scenario_type == "double_integrator":
            self.theta_dim = 2
            self.state_dim = 4
        elif scenario_type == "unicycle":
            self.theta_dim = 1
            self.state_dim = 3
        else:
            raise ValueError("Unknown scenario")

        self.theta_hat = np.zeros(self.theta_dim, dtype=np.float32)
        self.history_size = history_size
        self.buffer = []
        self.trajectory_window = []

    def get_regressor(self, x):
        if self.scenario_type == "single_integrator":
            Y = np.zeros((2, 3))
            Y[0, 0] = x[0]
            Y[0, 1] = x[1]
            Y[1, 2] = x[0]**3
            return Y
        elif self.scenario_type == "double_integrator":
            Y = np.zeros((4, 2))
            Y[0, 0] = x[2]
            Y[1, 1] = x[3]
            Y[2, 0] = -x[2]
            Y[3, 1] = -x[3]
            return Y
        elif self.scenario_type == "unicycle":
            Y = np.zeros((3, 1))
            Y[1, 0] = 1.0
            return Y

    def predict_f(self, x):
        Y = self.get_regressor(x)
        return Y @ self.theta_hat

    def reset_window(self):
        self.trajectory_window = []

    def update(self, x_old, x_new, u, dt, g_matrix):
        self.trajectory_window.append((x_old.copy(), u.copy(), dt, g_matrix.copy()))
        
        if len(self.trajectory_window) > self.integration_window:
            self.trajectory_window.pop(0)
        
        if len(self.trajectory_window) >= self.integration_window:
            integral_Y = np.zeros((self.state_dim, self.theta_dim))
            integral_gu = np.zeros(self.state_dim)
            
            for i in range(len(self.trajectory_window)):
                x_i, u_i, dt_i, g_i = self.trajectory_window[i]
                Y_i = self.get_regressor(x_i)
                gu_i = (g_i @ u_i).flatten()
                integral_Y += Y_i * dt_i
                integral_gu += gu_i * dt_i
            
            x_start = self.trajectory_window[0][0]
            x_end = self.trajectory_window[-1][0]
            delta_x = x_end - x_start
            target = delta_x - integral_gu
            
            if len(self.buffer) < self.history_size:
                self.buffer.append((integral_Y, target))
            else:
                self.buffer.pop(0)
                self.buffer.append((integral_Y, target))
        else:
            Y = self.get_regressor(x_old)
            gu_term = (g_matrix @ u).flatten()
            target = (x_new - x_old) - (gu_term * dt)
            regressor_dt = Y * dt
            
            if len(self.buffer) < self.history_size:
                self.buffer.append((regressor_dt, target))
            else:
                self.buffer.pop(0)
                self.buffer.append((regressor_dt, target))

        update_grad = np.zeros_like(self.theta_hat)
        for Y_i, target_i in self.buffer:
            error = target_i - (Y_i @ self.theta_hat)
            update_grad += Y_i.T @ error
        
        update_grad = np.clip(update_grad, -10.0, 10.0)
        self.theta_hat += self.gamma * update_grad
        # Soft bounds on parameters (prevents NaN propagation)
        self.theta_hat = np.clip(self.theta_hat, -20.0, 20.0)