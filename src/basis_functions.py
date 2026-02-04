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


class PolynomialBasis(BasisStrategy):
    """
    Polynomial basis functions up to a specified degree for n-dimensional state.
    
    Examples:
        2D, degree=2: [1, x, y, x^2, xy, y^2] (L=6)
        3D, degree=2: [1, x, y, z, x^2, xy, xz, y^2, yz, z^2] (L=10)
        2D, degree=3: [1, x, y, x^2, xy, y^2, x^3, x^2*y, xy^2, y^3] (L=10)
    """
    def __init__(self, state_dim, degree=2):
        super().__init__(state_dim)
        self.degree = degree
        
        # Generate all monomial indices (exponent vectors) for n-dimensional case
        # A monomial is x1^i1 * x2^i2 * ... * xn^in where i1 + i2 + ... + in <= degree
        self.monomial_indices = self._generate_monomial_indices(state_dim, degree)
        self.L = len(self.monomial_indices)

    def _generate_monomial_indices(self, n_dim, max_degree):
        """
        Generate all monomial exponent vectors for n dimensions up to max_degree.
        Uses recursive generation to find all (i1, i2, ..., in) where sum(ij) <= max_degree
        """
        def generate_recursive(n, max_sum, current=[]):
            if n == 1:
                for i in range(max_sum + 1):
                    yield current + [i]
            else:
                for i in range(max_sum + 1):
                    yield from generate_recursive(n - 1, max_sum - i, current + [i])
        
        monomials = list(generate_recursive(n_dim, max_degree))
        return monomials

    def evaluate(self, state, goal=None, center_state=None):
        """
        Evaluate polynomial basis and its gradient for n-dimensional state.
        
        Args:
            state: Global state [x1, x2, ..., xn]
            goal: Global goal [gx1, gx2, ..., gxn]
            center_state: (Optional) Reference point for evaluation
            
        Returns:
            phi (L,): Polynomial feature vector
            grad_phi (L, state_dim): Gradient w.r.t state
        """
        if goal is None:
            goal = np.zeros(self.state_dim)
        
        # Transform to relative coordinates
        x_rel = state[:self.state_dim] - goal[:self.state_dim]
        
        phi = np.zeros(self.L)
        grad_phi = np.zeros((self.L, self.state_dim))
        
        for idx, exponents in enumerate(self.monomial_indices):
            # Compute monomial: x1^i1 * x2^i2 * ... * xn^in
            monomial_val = 1.0
            for dim, exp in enumerate(exponents):
                monomial_val *= x_rel[dim] ** exp
            phi[idx] = monomial_val
            
            # Compute gradient for each dimension
            # d(x1^i1 * x2^i2 * ... * xn^in) / dxj = ij * x1^i1 * ... * xj^(ij-1) * ... * xn^in
            for dim in range(self.state_dim):
                if exponents[dim] > 0:
                    grad_val = exponents[dim]
                    for d, exp in enumerate(exponents):
                        if d == dim:
                            grad_val *= x_rel[d] ** (exp - 1)
                        else:
                            grad_val *= x_rel[d] ** exp
                    grad_phi[idx, dim] = grad_val
        
        return phi, grad_phi