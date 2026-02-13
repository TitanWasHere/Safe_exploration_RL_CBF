import numpy as np

class SafeMBRL:
    def __init__(self, dyn_learner, safety_filter, scenario, basis_strategy,
                 use_safety_filter=True):
        """
        Args:
            basis_strategy (BasisStrategy): An instance of RBFBasis, PolynomialBasis, etc.
            use_safety_filter (bool): If False, the CBF safety filter is bypassed.
        """
        self.dyn_learner = dyn_learner
        self.safety_filter = safety_filter
        self.scenario = scenario
        self.use_safety_filter = use_safety_filter
        
        # 1. Plug-and-Play Basis
        self.basis = basis_strategy
        self.L = self.basis.output_dim  # Auto-detect dimension
        
        # Load Parameters
        self.Q = scenario.Q
        self.R = scenario.R
        self.R_inv = np.linalg.inv(self.R)
        self.state_dim = scenario.state_dim
        self.action_dim = scenario.action_dim
        self.action_max = scenario.action_max
        self.goal_dim = getattr(scenario, 'goal_dim', 2)
        
        # Hyperparameters
        self.kc1 = 0.1    
        self.kc2 = 1.0     
        self.beta = 0.001  
        self.gamma_c = 1.0 
        self.ka1 = 1.0     
        self.ka2 = 0.1    
        self.N_extrapolate = 10 

        # Paper init
        self.static = True
        self.weight_value = 0.5 

        # Weights Initialization
        self._initialize_weights_quadratic(static=self.static, static_value=self.weight_value)
        self.Gamma = np.eye(self.L) * 100.0

        # Local Minima Escape
        self._pos_history = []
        self._stuck_window = 60       # steps to look back
        self._stuck_threshold = 0.15  # displacement threshold
        self._escape_direction = None
        self._escape_steps = 0
        self._escape_duration = 40    # steps per escape burst
        self._escape_magnitude = 2.0  # perturbation strength
        self._last_h_val = None       # store last CBF value for metrics

    def _initialize_weights_quadratic(self, static=False, static_value=0.0):
        """Pre-trains Wc to look like the cost function x.T @ Q @ x or initialize static weights"""
        if static:
            self.Wc = np.full(self.L, static_value)
            self.Wa = np.full(self.L, static_value)
            return
        
        self.Wc = np.zeros(self.L)
        X_samples = []
        y_samples = []
        
        # Train on the specific basis provided
        for _ in range(500):
            x_rel = np.random.uniform(-2, 2, self.state_dim)
            phi, _ = self.basis.evaluate(x_rel)
            target_val = x_rel.T @ self.Q @ x_rel
            X_samples.append(phi)
            y_samples.append(target_val)
            
        X = np.array(X_samples)
        y = np.array(y_samples)
        self.Wc = np.linalg.pinv(X.T @ X + 0.1*np.eye(self.L)) @ X.T @ y
        self.Wa = self.Wc.copy()

    def get_action(self, obs):
        state = obs[:self.state_dim]
        goal = obs[self.state_dim:]

        x_rel = self.scenario.get_error_state(state, goal)

        phi, grad_phi = self.basis.evaluate(x_rel, goal=None, center_state=None)

        grad_V = grad_phi.T @ self.Wa 
        g_x = self.scenario.get_g(state) 

        u_nom = -0.5 * self.R_inv @ (g_x.T @ grad_V)
        u_nom = u_nom.flatten()

        # Local Minima Escape
        dist_to_goal = np.linalg.norm(state[:2] - goal[:2])
        self._pos_history.append(state[:2].copy())
        if len(self._pos_history) > self._stuck_window:
            self._pos_history.pop(0)

        if self._escape_steps <= 0 and len(self._pos_history) >= self._stuck_window:
            displacement = np.linalg.norm(
                self._pos_history[-1] - self._pos_history[0])
            if displacement < self._stuck_threshold and dist_to_goal > 0.5:
                # if stuck: pick a persistent random direction in action space
                direction = np.random.randn(self.action_dim)
                norm = np.linalg.norm(direction)
                if norm > 1e-8:
                    direction /= norm
                self._escape_direction = direction * self._escape_magnitude
                self._escape_steps = self._escape_duration

        if self._escape_steps > 0 and self._escape_direction is not None:
            u_nom += self._escape_direction
            self._escape_steps -= 1

        # Safety Filter
        if self.use_safety_filter:
            nearby_obs = self.scenario.get_near_obstacle(state)
            u_safe, h_val = self.safety_filter.get_safe_action(
                obs, u_nom, nearby_obs, self.R_inv, self.action_max
            )
            self._last_h_val = float(h_val)
        else:
            u_safe = np.clip(u_nom, -self.action_max, self.action_max)
            self._last_h_val = None

        return u_safe

    def compute_cost(self, x_rel, u):
        return x_rel.T @ self.Q @ x_rel + u.T @ self.R @ u

    def reset_episode(self):
        """Reset per-episode state (call at start of each episode)."""
        self._pos_history = []
        self._escape_steps = 0
        self._escape_direction = None
        self._last_h_val = None

    def update(self, x_abs, u, x_next_abs, dt, goal):
        x_abs = x_abs.flatten()
        u = u.flatten()

        x_rel = self.scenario.get_error_state(x_abs, goal[:self.state_dim])

        g_x = self.scenario.get_g(x_abs) 
        self.dyn_learner.update(x_abs, x_next_abs, u, dt, g_x)

        f_hat = self.dyn_learner.predict_f(x_abs)
        phi, grad_phi = self.basis.evaluate(x_rel, goal=None, center_state=None)

        omega = grad_phi @ (f_hat + g_x @ u)
        rho_sq = 1.0 + self.gamma_c * np.dot(omega, omega)
        
        cost = self.compute_cost(x_rel, u)

        V_dot = np.dot(self.Wc, omega)
        delta = cost + V_dot
        
        G_R = g_x @ self.R_inv @ g_x.T 
        G_phi = grad_phi @ G_R @ grad_phi.T

        # Simulation Loop
        sum_omega_delta = np.zeros(self.L)
        sum_lambda = np.zeros((self.L, self.L))
        sum_actor_grad_term = np.zeros(self.L)
        
        for _ in range(self.N_extrapolate):
            x_sim_rel = x_rel + np.random.uniform(-0.5, 0.5, self.state_dim)
            
            phi_i, grad_phi_i = self.basis.evaluate(x_sim_rel, goal=None, center_state=x_rel)

            grad_V_i = grad_phi_i.T @ self.Wa
            u_i = -0.5 * self.R_inv @ (g_x.T @ grad_V_i)
            
            omega_i = grad_phi_i @ (f_hat + g_x @ u_i)
            rho_sq_i = 1.0 + self.gamma_c * np.dot(omega_i, omega_i)
            
            cost_i = self.compute_cost(x_sim_rel, u_i)
            V_dot_i = np.dot(self.Wc, omega_i)
            delta_i = cost_i + V_dot_i
            
            sum_omega_delta += (omega_i / rho_sq_i) * delta_i
            sum_lambda += np.outer(omega_i, omega_i) / rho_sq_i
            
            G_phi_i = grad_phi_i @ G_R @ grad_phi_i.T
            w_dot_omega = np.dot(self.Wc, omega_i)
            G_Wa = G_phi_i @ self.Wa
            
            sum_actor_grad_term += (G_Wa * w_dot_omega) / rho_sq_i

        # Average the sums
        avg_omega_delta = sum_omega_delta / self.N_extrapolate
        avg_lambda = sum_lambda / self.N_extrapolate
        avg_actor_term = sum_actor_grad_term / self.N_extrapolate

        # Weight Updates
        
        # A. Gain Matrix Gamma Update (Eq. 23)
        # dGamma = beta*Gamma - Gamma * (kc1*lambda + kc2*lambda_avg) * Gamma
        lambda_curr = np.outer(omega, omega) / rho_sq
        
        M = self.kc1 * lambda_curr + self.kc2 * avg_lambda
        dot_Gamma = self.beta * self.Gamma - self.Gamma @ M @ self.Gamma
        
        self.Gamma += dot_Gamma * dt
        
        # Numerical Safety for Gamma
        self.Gamma = 0.5 * (self.Gamma + self.Gamma.T)
        eig_vals = np.linalg.eigvalsh(self.Gamma)
        if np.min(eig_vals) < 0.01:
            self.Gamma += 0.01 * np.eye(self.L)
        self.Gamma = np.clip(self.Gamma, 0.01, 1000.0)

        # B. Critic Update (Eq. 22)
        # dWc = -Gamma * (kc1 * omega * delta / rho^2 + kc2 * avg_omega_delta)
        term_1 = (self.kc1 * delta / rho_sq) * omega
        term_2 = self.kc2 * avg_omega_delta
        
        dot_Wc = -self.Gamma @ (term_1 + term_2)
        self.Wc += dot_Wc * dt
        
        # C. Actor Update (Eq. 24)
        # dWa = -ka1(Wa - Wc) - ka2*Wa + (kc1/4)*Term_Curr + (kc2/4)*Term_Sim
        
        # Current Experience Term
        w_dot_omega_curr = np.dot(self.Wc, omega)
        G_Wa_curr = G_phi @ self.Wa
        actor_term_curr = (G_Wa_curr * w_dot_omega_curr) / rho_sq
        
        term_actor_1 = (self.kc1 / 4.0) * actor_term_curr
        term_actor_2 = (self.kc2 / 4.0) * avg_actor_term
        
        dot_Wa = -self.ka1 * (self.Wa - self.Wc) \
                 -self.ka2 * self.Wa \
                 + term_actor_1 \
                 + term_actor_2
                 
        self.Wa += dot_Wa * dt
        
        # Stability Clips
        self.Wc = np.clip(self.Wc, -200, 200)
        self.Wa = np.clip(self.Wa, -200, 200)
        
        return delta