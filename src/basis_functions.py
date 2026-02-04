import numpy as np
from abc import ABC, abstractmethod

class BasisStrategy(ABC):
    """
    Abstract Base Class for Value Function Approximation.
    V(x) = W^T * phi(x)
    """
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.L = 0 # Output dimension (number of weights)

    @abstractmethod
    def evaluate(self, state, goal=None, center_state=None):
        """
        Computes features and gradients.
        Args:
            state: Global state vector (x, y, ...)
            goal: Global goal vector (gx, gy, ...) - Needed for StaF
            center_state: (Optional) The center of the approximation (x in the paper notation)
                          If None, assumes state is both center and eval point.
        Returns:
            phi (L,): Feature vector
            grad_phi (L, state_dim): Gradient w.r.t state
        """
        pass
    
    @property
    def output_dim(self):
        return self.L


class PaperStaF(BasisStrategy):
    """
    Exact implementation of the StaF kernel from Section 6 (Numerical Examples) 
    of Cohen & Belta (2021).
    
    Paper Specifications:
        - Basis size: L = 3
        - Kernel type: Polynomial kernel phi_i(y, c_i) = y.T @ c_i
        - Center placement: Vertices of an equilateral triangle centered at x.
        - Center dynamics: c_i(x) = x + nu(x) * d_i
        - Mixing function: nu(x) = (x.T @ x) / (x.T @ x + 1)
    """
    def __init__(self, state_dim, radius=1.0, num_kernel=3):
        super().__init__(state_dim)
        # The paper example is explicitly for a 2D system (Section 6, "x in R^2")
        assert state_dim == 2, "Paper example is designed for 2D systems."
        self.L = num_kernel
        
        # Define vertices of an equilateral triangle (d_i)
        # Angles: 0, 120, 240 degrees
        angles = np.deg2rad([0, 120, 240])
        self.d_vectors = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles)
        ]) # Shape (3, 2)

    def evaluate(self, state, goal=None, center_state=None):
        """
        Evaluate the basis function phi(y, c(x)) and its gradient w.r.t y.
        
        In the paper's notation:
        - state corresponds to 'y' (the evaluation point)
        - center_state corresponds to 'x' (the trajectory point defining the centers)
        """
        if goal is None: goal = np.zeros_like(state)
        
        # 1. Transform to relative error coordinates
        # x_eval = y - x_goal (The point where we evaluate the basis)
        x_eval = state[:2] - goal[:2]
        
        # Determine the center-generating state (x)
        if center_state is not None:
            # We are evaluating at a neighbor point y (x_eval), but centers are fixed at x (x_center)
            # This is used for the "Simulation of Experience"
            x_center = center_state[:2] - goal[:2]
        else:
            # We are evaluating at x itself (y = x)
            x_center = x_eval
            
        # 2. Compute Mixing Function nu(x)
        # nu(x) = (x'x) / (x'x + 1)
        norm_sq = np.dot(x_center, x_center)
        nu = norm_sq / (norm_sq + 1.0)
        
        # Gradient of nu w.r.t x_center (needed for chain rule when y=x)
        # d(nu)/dx = 2x / (x'x + 1)^2
        grad_nu = (2 * x_center) / ((norm_sq + 1.0)**2)

        phi = np.zeros(self.L)
        grad_phi = np.zeros((self.L, self.state_dim))

        for i in range(self.L):
            d_i = self.d_vectors[i]
            
            # 3. Compute Centers c_i(x)
            # c_i(x) = x_center + nu(x_center) * d_i
            c_i = x_center + nu * d_i
            
            # 4. Compute Basis Value phi_i
            # Paper: phi_i(y, c(x)) = y.T @ c_i(x)
            phi[i] = np.dot(x_eval, c_i)
            
            # 5. Compute Gradient w.r.t evaluation state (y)
            # We need d(phi_i)/dy
            
            if center_state is None:
                # Case A: y = x (Evaluating at the current trajectory point)
                # phi_i(x) = x.T @ (x + nu(x)*d_i) = x.T@x + nu(x)*(x.T@d_i)
                
                # Term 1: d(x.T@x)/dx = 2x
                term1 = 2 * x_eval
                
                # Term 2: d(nu(x)*(x.T@d_i))/dx
                # Product rule: nu(x)*d_i + (x.T@d_i) * grad_nu
                term2 = nu * d_i
                term3 = np.dot(d_i, x_eval) * grad_nu
                
                grad_phi[i, :2] = term1 + term2 + term3
            else:
                # Case B: y != x (Simulation of Experience)
                # phi_i(y) = y.T @ c_i(x)
                # Here c_i(x) is constant with respect to y.
                # d(phi_i)/dy = c_i(x)
                grad_phi[i, :2] = c_i

        return phi, grad_phi