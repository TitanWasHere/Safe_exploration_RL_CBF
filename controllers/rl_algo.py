import numpy as np

class RLSafeActorCritic:
    """
    Implements Equations 22, 23, and 24 from Cohen & Belta.
    Updates Critic weights (Wc) and Actor weights (Wa) online.
    """
    def __init__(self, state_dim, action_dim, n_basis=3):
        self.kc1, self.kc2 = 0.1, 1.0
        self.ka1, self.ka2 = 1.0, 0.1
        self.beta_c = 0.005
        self.gamma_c = 1.0
        
        self.Wc = np.ones(n_basis) * 0.5
        self.Wa = np.ones(n_basis) * 0.5
        
        self.Gamma = np.eye(n_basis) * 100.0
        
        self.history = [] 
        self.max_history = 50

    def get_basis(self, x):
        """
        State-following (StaF) kernels or simple polynomials.
        Paper uses x^T * ci for Section 6.
        """
        return np.array([x[0]**2, x[0]*x[1], x[1]**2])

    def get_action(self, x, g_matrix):
        """
        Returns the nominal RL action: u = -0.5 * R^-1 * g^T * grad_phi^T * Wa
        (Simplified version of Eq 15b)
        """
        phi_grad = self._get_phi_gradient(x)
        u_nom = -0.5 * (phi_grad @ g_matrix).T @ self.Wa
        return u_nom

    def update(self, x, u, r, x_next, f_hat, g_matrix):
        """
        Implements Eq 22-24: Concurrent Learning Updates
        """
        phi = self.get_basis(x)
        phi_grad = self._get_phi_gradient(x)
        
        omega = phi_grad @ (f_hat + g_matrix @ u)
        rho_sq = 1 + self.gamma_c * (omega @ omega)
        
        delta = r + self.Wc @ omega
        
        if len(self.history) < self.max_history:
            self.history.append((omega, delta))

        Lambda = (np.outer(omega, omega)) / (rho_sq**2)
        Lambda_sum = np.zeros_like(Lambda)
        for om_i, _ in self.history:
            Lambda_sum += np.outer(om_i, om_i)
        
        d_Gamma = self.beta_c * self.Gamma - self.Gamma @ (self.kc1 * Lambda + (self.kc2/self.max_history) * Lambda_sum) @ self.Gamma
        self.Gamma += d_Gamma * 0.01

        sum_concurrent = np.zeros_like(self.Wc)
        for om_i, del_i in self.history:
            sum_concurrent += om_i * del_i
            
        dWc = -self.Gamma @ (self.kc1 * (omega * delta / rho_sq) + (self.kc2/self.max_history) * sum_concurrent)
        self.Wc += dWc * 0.01

        dWa = -self.ka1 * (self.Wa - self.Wc) - self.ka2 * self.Wa
        self.Wa += dWa * 0.01

    def _get_phi_gradient(self, x):
        return np.array([
            [2*x[0], 0],
            [x[1],   x[0]],
            [0,      2*x[1]]
        ])