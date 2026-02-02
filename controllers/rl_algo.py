import numpy as np

class SafeMBRL:
    def __init__(self, dyn_learner, safety_filter,
                 Q=np.eye(2), R=np.eye(2),
                 num_kernels=3):
        
        self.dyn_learner = dyn_learner
        self.safety_filter = safety_filter
        
        self.Q = Q
        self.R = R
        self.R_inv = np.linalg.inv(R)
        self.state_dim = self.dyn_learner.state_dim
        
        self.L = num_kernels
        self.Wc = np.ones(self.L) * 0.5
        self.Wa = np.ones(self.L) * 0.5
        self.Gamma = np.eye(self.L) * 100.0
        
        self.kc1 = 0.1
        self.kc2 = 1.0
        self.ka1 = 1.0
        self.ka2 = 0.1
        self.beta = 0.001
        self.gamma_c = 1.0
        
        self.N_extrapolate = 5 

    def get_staf_kernels(self, x):
        """State-Aware Function kernels: phi_i(x) = x^T * c_i(x)."""
        norm_x_sq = np.dot(x, x)
        nu = norm_x_sq / (norm_x_sq + 1.0)
        
        phi = np.zeros(self.L)
        for i in range(self.L):
            angle = 2 * np.pi * i / self.L
            di = np.array([np.cos(angle), np.sin(angle)])
            ci = x + nu * di
            phi[i] = np.dot(x, ci)
        
        grad_phi = np.zeros((self.L, self.state_dim))
        for i in range(self.L):
            angle = 2 * np.pi * i / self.L
            di = np.array([np.cos(angle), np.sin(angle)])
            ci = x + nu * di
            grad_phi[i, :] = ci
            
        return phi, grad_phi

    def compute_reward(self, x, u):
        """r(x, u) = x^T Q x + u^T R u (Eq. 10)"""
        return x.T @ self.Q @ x + u.T @ self.R @ u

    def get_action(self, x, obstacles, goal, g_x=None):
        phi, grad_phi = self.get_staf_kernels(x)
        if g_x is None:
            g_x = self.safety_filter.f_g(x).full()
        
        grad_V_hat = self.Wa @ grad_phi
        u_nom = -0.5 * self.R_inv @ (g_x.T @ grad_V_hat)
        u_nom = u_nom.flatten()

        obs_for_filter = np.concatenate([x, goal])
        u_safe, _ = self.safety_filter.get_safe_action(obs_for_filter, u_nom, obstacles)
        
        return u_safe

    def update(self, x, u, x_next, dt, g_x=None):
        x = x.reshape(-1, 1)
        u = u.reshape(-1, 1)
        x_next = x_next.reshape(-1, 1)

        if g_x is None:
            g_x = self.safety_filter.f_g(x).full()
        
        self.dyn_learner.update(x.flatten(), x_next.flatten(), u.flatten(), dt, g_x)
        f_hat = self.dyn_learner.predict_f(x.flatten()).reshape(-1, 1)

        phi, grad_phi = self.get_staf_kernels(x.flatten())
        
        f_hat_flat = f_hat.flatten()
        gu_flat = (g_x @ u).flatten()
        omega = grad_phi @ (f_hat_flat + gu_flat)
        
        reward = self.compute_reward(x, u)
        delta_t = float(reward + self.Wc @ omega)

        sum_omega_delta = np.zeros(self.L)
        sum_lambda = np.zeros((self.L, self.L))
        
        for _ in range(self.N_extrapolate):
            x_rand_offset = np.random.uniform(-0.5, 0.5, size=(self.state_dim, 1))
            x_rand = x + x_rand_offset
            x_rand_flat = x_rand.flatten()
            
            phi_i, grad_phi_i = self.get_staf_kernels(x_rand_flat)
            g_i = g_x
            f_i = self.dyn_learner.predict_f(x_rand_flat)
            
            grad_V_i = self.Wa @ grad_phi_i
            u_i = -0.5 * self.R_inv @ (g_i.T @ grad_V_i)
            u_i = u_i.flatten()
            
            f_i_flat = f_i.flatten() if f_i.ndim > 1 else f_i
            gu_i_flat = (g_i @ u_i).flatten()
            omega_i = grad_phi_i @ (f_i_flat + gu_i_flat)
            rho_i = 1.0 + self.gamma_c * np.dot(omega_i, omega_i)
            reward_i = self.compute_reward(x_rand, u_i.reshape(-1, 1))
            delta_i = float(reward_i + self.Wc @ omega_i)
            
            sum_omega_delta += (omega_i / rho_i**2) * delta_i
            sum_lambda += (np.outer(omega_i, omega_i) / rho_i**2)

        rho_t = 1.0 + self.gamma_c * np.dot(omega, omega)
        
        lambda_t = np.outer(omega, omega) / (rho_t**2)
        dot_Gamma = self.beta * self.Gamma - self.Gamma @ (self.kc1 * lambda_t + (self.kc2 / self.N_extrapolate) * sum_lambda) @ self.Gamma
        self.Gamma += dot_Gamma * dt
        self.Gamma = np.clip(self.Gamma, 0.1, 1000.0)
        
        dot_Wc = -self.Gamma @ (self.kc1 * (omega / rho_t**2) * delta_t + (self.kc2 / self.N_extrapolate) * sum_omega_delta)
        self.Wc += dot_Wc * dt
        self.Wc = np.clip(self.Wc, -100.0, 100.0)
        
        G_phi = grad_phi @ g_x @ self.R_inv @ g_x.T @ grad_phi.T
        omega_weighted = (self.Wa @ omega) * (self.Wc @ omega)
        term_actor = (self.kc1 / (4 * rho_t**2)) * omega_weighted * (G_phi @ self.Wa)
        dot_Wa = -self.ka1 * (self.Wa - self.Wc) - self.ka2 * self.Wa + term_actor
        self.Wa += dot_Wa * dt
        self.Wa = np.clip(self.Wa, -100.0, 100.0)

        return delta_t