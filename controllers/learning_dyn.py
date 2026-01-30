import numpy as np

class LearnerDynamics:
    """
    Implements Section 4.3 (System Identification) of the paper.
    Uses Integral Concurrent Learning (ICL) to estimate the unknown drift f(x) = Y(x)θ.
    """
    def __init__(self, scenario_type="single_integrator", learning_rate=0.5, history_size=50):
        self.scenario_type = scenario_type
        self.gamma = learning_rate  # Adaptation gain
        
        # 1. Initialize Basis Functions and Parameter Dimensions
        if scenario_type == "single_integrator":
            # Paper Eq (Sec 6): f(x) = [-0.6x1 - x2, x1^3]
            # Basis Y(x) is (2 x 3), theta is (3 x 1)
            self.theta_dim = 3
            self.state_dim = 2
        elif scenario_type == "double_integrator":
            self.theta_dim = 2 # learning drag coefficients
            self.state_dim = 4
        elif scenario_type == "unicycle":
            self.theta_dim = 1 # learning the drift constant in y
            self.state_dim = 3
        else:
            raise ValueError("Unknown scenario")

        # Parameter estimate (theta_hat)
        self.theta_hat = np.zeros(self.theta_dim, dtype=np.float32)
        
        # Concurrent Learning Buffer (to store historical data for convergence)
        self.history_size = history_size
        self.buffer = [] # Stores tuples of (integral_Y, delta_x_minus_integral_gu)

    def get_regressor(self, x):
        """
        Defines Y(x) such that f(x) = Y(x) @ theta.
        Based on Section 6: Nonlinear System example.
        """
        if self.scenario_type == "single_integrator":
            # Y(x) = [[x1, x2, 0], [0, 0, x1^3]]
            # x = [x1, x2]
            Y = np.zeros((2, 3))
            Y[0, 0] = x[0]
            Y[0, 1] = x[1]
            Y[1, 2] = x[0]**3
            return Y
        
        elif self.scenario_type == "double_integrator":
            # Learning drag for vx and vy
            Y = np.zeros((4, 2))
            Y[0, 0] = x[2] # kinematic link
            Y[1, 1] = x[3]
            Y[2, 0] = -x[2] # drag x
            Y[3, 1] = -x[3] # drag y
            return Y

        elif self.scenario_type == "unicycle":
            # Learning a drift constant in y
            Y = np.zeros((3, 1))
            Y[1, 0] = 1.0 # Constant bias in y-dot
            return Y

    def predict_f(self, x):
        """Returns the current estimate of the drift: f_hat = Y(x) @ theta_hat"""
        Y = self.get_regressor(x)
        return Y @ self.theta_hat

    def update(self, x_old, x_new, u, dt, g_matrix):
        """
        Implements the parameter identification update law.
        Uses the integral form to avoid needing acceleration (x_dot).
        """
        # 1. Calculate Integrals (approximate over one dt)
        # In a real ICL implementation, you might integrate over a window T.
        # Here we use the Euler approximation for the step.
        Y = self.get_regressor(x_old)
        
        # Predicted change from known control part: integral of g(x)u
        gu_term = g_matrix @ u
        
        # The relationship: x_new - x_old = integral(Y*theta) + integral(g*u)
        # Therefore: (x_new - x_old - gu*dt) = (Y*dt) @ theta
        target = (x_new - x_old) - (gu_term * dt)
        regressor_dt = Y * dt
        
        # 2. Update Buffer for Concurrent Learning
        # This ensures convergence without Persistence of Excitation (PE)
        if len(self.buffer) < self.history_size:
            self.buffer.append((regressor_dt, target))
        else:
            # Replace oldest if new data is "sufficiently different" (simplified)
            self.buffer.pop(0)
            self.buffer.append((regressor_dt, target))

        # 3. Gradient Descent Update Law (Ref [29])
        # d(theta_hat)/dt = Gamma * sum( Y_i.T @ (target_i - Y_i @ theta_hat) )
        update_grad = np.zeros_like(self.theta_hat)
        for Y_i, target_i in self.buffer:
            error = target_i - (Y_i @ self.theta_hat)
            update_grad += Y_i.T @ error
            
        self.theta_hat += self.gamma * update_grad