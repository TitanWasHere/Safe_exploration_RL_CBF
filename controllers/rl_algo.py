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
        self.gamma_discount = 0.98
        self.N_extrapolate = 1 

    def get_staf_kernels(self, x):
        norm_x_sq = np.dot(x, x)
        nu = norm_x_sq / (norm_x_sq + 1.0)
        
        phi = np.zeros(self.L)
        for i in range(self.L):
            angle = 2 * np.pi * i / self.L
            di = np.array([np.cos(angle), np.sin(angle)])
            ci = x + nu * di
            phi[i] = np.dot(x, ci)
        
        grad_phi = np.zeros((self.L, self.state_dim))
        d_nu_dx = (2.0 * x) / ((norm_x_sq + 1.0)**2)
        
        for i in range(self.L):
            angle = 2 * np.pi * i / self.L
            di = np.array([np.cos(angle), np.sin(angle)])
            ci = x + nu * di
            dc_i_dx = np.eye(self.state_dim) + np.outer(di, d_nu_dx)
            grad_phi[i, :] = ci + x @ dc_i_dx
            
        return phi, grad_phi

    def compute_reward(self, x, u):
        return -(x.T @ self.Q @ x + u.T @ self.R @ u)

    def get_action(self, x, obstacles, goal, g_x=None):
        x_rel = x - goal
        phi, grad_phi = self.get_staf_kernels(x_rel)
        
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
        V_current = self.Wc @ phi
        delta_t = float(reward - V_current)

        sum_omega_delta = np.zeros(self.L)
        sum_lambda = np.zeros((self.L, self.L))
        sum_actor_term = np.zeros(self.L)
        
        for _ in range(self.N_extrapolate):
            norm_x_sq = np.dot(x.flatten(), x.flatten())
            nu = norm_x_sq / (norm_x_sq + 1.0)
            x_rand_offset = np.random.uniform(-nu, nu, size=(self.state_dim, 1))
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
            V_i = self.Wc @ phi_i
            delta_i = float(reward_i - V_i)
            
            sum_omega_delta += (omega_i / rho_i**2) * delta_i
            sum_lambda += (np.outer(omega_i, omega_i) / rho_i**2)
            G_phi_i = grad_phi_i @ g_i @ self.R_inv @ g_i.T @ grad_phi_i.T
            actor_contrib = (G_phi_i.T @ self.Wa) * np.dot(omega_i, self.Wc)
            sum_actor_term += actor_contrib / rho_i**2

        rho_t = 1.0 + self.gamma_c * np.dot(omega, omega)
        lambda_t = np.outer(omega, omega) / (rho_t**2)
        dot_Gamma = self.beta * self.Gamma - self.Gamma @ (self.kc1 * lambda_t + (self.kc2 / self.N_extrapolate) * sum_lambda) @ self.Gamma
        self.Gamma += dot_Gamma * dt
        self.Gamma = np.maximum(self.Gamma, 0.01 * np.eye(self.L))
        
        dot_Wc = -self.Gamma @ (self.kc1 * (omega / rho_t**2) * delta_t + (self.kc2 / self.N_extrapolate) * sum_omega_delta)
        dot_Wc = np.clip(dot_Wc, -5.0, 5.0)
        self.Wc += dot_Wc * dt
        
        G_phi = grad_phi @ g_x @ self.R_inv @ g_x.T @ grad_phi.T
        omega_wc = np.dot(omega, self.Wc)
        term_actor_real = (self.kc1 / (4 * rho_t**2)) * omega_wc * (G_phi.T @ self.Wa)
        term_actor_sim = (self.kc2 / (4 * self.N_extrapolate)) * sum_actor_term
        
        dot_Wa = -self.ka1 * (self.Wa - self.Wc) - self.ka2 * self.Wa + term_actor_real + term_actor_sim
        dot_Wa = np.clip(dot_Wa, -10.0, 10.0)
        
        self.Wa += dot_Wa * dt
        W_norm = np.linalg.norm(self.Wa)
        if W_norm > 50.0:
            self.Wa = self.Wa * (50.0 / W_norm)

        return delta_t