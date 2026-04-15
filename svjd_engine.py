import numpy as np
from scipy.integrate import quad
from numba import njit, complex128, float64
import cmath

class SVJDModel:
    """
    Stochastic Volatility Jump Diffusion (SVJD) model for financial spreads.
    Implements the Heston stochastic volatility framework augmented with 
    a compound Poisson jump component (Bates 1996).
    
    Dynamics:
        dS = mu*S*dt + sqrt(V)*S*dW1 + (exp(J) - 1)*S*dN(lambda)
        dV = kappa*(theta - V)*dt + sigma_v*sqrt(V)*dW2
        dW1*dW2 = rho*dt
        J ~ N(mu_J, sigma_J^2)
    """

    def __init__(self, mu, kappa, theta, sigma_v, rho, lambda_jump, mu_j, sigma_j):
        """
        Initialize the SVJD model parameters.
        
        Args:
            mu (float): Annualized drift of the spread.
            kappa (float): Mean-reversion speed of variance.
            theta (float): Long-term mean of variance.
            sigma_v (float): Volatility of variance (vol-of-vol).
            rho (float): Correlation between spread and variance shocks.
            lambda_jump (float): Poisson jump intensity.
            mu_j (float): Mean of the jump size distribution (log-scale).
            sigma_j (float): Standard deviation of the jump size distribution.
        """
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.lambda_jump = lambda_jump
        self.mu_j = mu_j
        self.sigma_j = sigma_j

    def characteristic_function(self, u, S, V, T):
        """
        Compute the characteristic function phi(u) for the SVJD process.
        Uses the Albrecher (2007) stable formulation of the Heston CF.
        """
        return _svjd_cf_kernel(u, S, V, T, self.mu, self.kappa, self.theta, 
                               self.sigma_v, self.rho, self.lambda_jump, 
                               self.mu_j, self.sigma_j)

    def price_spread(self, S, V, T, K, option_type='call'):
        """
        Price a European option on the spread using the Lewis (2001) Fourier kernel.
        
        Args:
            S (float): Current level of the spread.
            V (float): Current variance state.
            T (float): Time to maturity in years.
            K (float): Strike price.
            option_type (str): 'call' or 'put'.
            
        Returns:
            float: Theoretical fair value of the spread option.
        """
        # Adjusted drift for risk-neutral pricing if mu is treated as r
        # For a spread, we often assume mu is the risk-neutral drift.
        # m is the expected jump size %
        m = np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1
        
        # Integrand for Lewis formula
        def integrand(u):
            # Transform u to u - i/2 for the Lewis kernel
            phi = _svjd_cf_kernel(u - 0.5j, S, V, T, self.mu, self.kappa, self.theta, 
                                  self.sigma_v, self.rho, self.lambda_jump, 
                                  self.mu_j, self.sigma_j)
            
            weight = 1.0 / (u**2 + 0.25)
            # Re[exp(-iu * ln(K/S)) * phi(u - i/2)] * weight
            res = (cmath.exp(-1j * u * np.log(K / S)) * phi).real * weight
            return res

        integral, _ = quad(integrand, 0, 100, limit=100)
        
        # Lewis formula for Call: S - sqrt(S*K)/pi * integral
        # Note: This specific form assumes r=0 or drift is absorbed in phi.
        # If r > 0, we apply discount e^{-rT}.
        price = S - (np.sqrt(S * K) / np.pi) * integral
        
        if option_type.lower() == 'put':
            # Put-Call Parity: C - P = S - K*exp(-rT)
            # Assuming mu = r for the parity check
            price = price - S + K
            
        return max(0.0, price)

@njit(complex128(complex128, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64))
def _svjd_cf_kernel(u, S, V, T, mu, kappa, theta, sigma_v, rho, lambda_jump, mu_j, sigma_j):
    """Numba-accelerated core characteristic function calculation."""
    # Drift adjustment for jumps to ensure martingale property under risk-neutral
    m = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    drift_rn = mu - lambda_jump * m
    
    # Heston components (Stable form)
    # d = sqrt((rho*sigma_v*i*u - kappa)^2 - sigma_v^2*(-i*u - u^2))
    # Note: -i*u - u^2 is the same as i*u + u^2 in the sqrt
    xi = kappa - rho * sigma_v * 1j * u
    d = cmath.sqrt(xi**2 + sigma_v**2 * (1j * u + u**2))
    
    g = (xi - d) / (xi + d)
    
    # exponent terms
    D = ((xi - d) / sigma_v**2) * ((1.0 - cmath.exp(-d * T)) / (1.0 - g * cmath.exp(-d * T)))
    C = (kappa * theta / sigma_v**2) * ((xi - d) * T - 2.0 * cmath.log((1.0 - g * cmath.exp(-d * T)) / (1.0 - g)))
    
    # Heston CF part
    log_s = np.log(S)
    phi_heston = cmath.exp(C + D * V + 1j * u * (log_s + drift_rn * T))
    
    # Jump components (Bates 1996)
    # exp(lambda*T * (exp(i*u*mu_j - 0.5*sigma_j^2*u^2) - 1))
    phi_jump = cmath.exp(lambda_jump * T * (cmath.exp(1j * u * mu_j - 0.5 * sigma_j**2 * u**2) - 1.0))
    
    return phi_heston * phi_jump

class SVJDKalmanFilter:
    """
    Unscented Kalman Filter (UKF) for latent state estimation in SVJD models.
    States: [V_t, lambda_t]
    Observable: Log-returns of the spread price.
    """
    def __init__(self, Q, R, alpha=1e-3, beta=2.0, kappa_sut=0.0):
        """
        Initialize the UKF.
        
        Args:
            Q (ndarray): Process noise covariance matrix (2x2).
            R (float): Observation noise variance (scalar for log-returns).
            alpha, beta, kappa_sut: SUT scaling parameters.
        """
        self.Q = Q
        self.R = R
        self.alpha = alpha
        self.beta = beta
        self.kappa_sut = kappa_sut
        
        self.x = None # State mean [V, lambda]
        self.P = None # State covariance
        self.n = 2    # Dimension of state

    def initialize(self, V0, lambda0, P0):
        """Set initial state and covariance."""
        self.x = np.array([V0, lambda0])
        self.P = P0

    def _generate_sigma_points(self):
        """Generate sigma points using SUT."""
        n = self.n
        lmbda = self.alpha**2 * (n + self.kappa_sut) - n
        
        # Matrix square root
        try:
            U = np.linalg.cholesky((n + lmbda) * self.P)
        except np.linalg.LinAlgError:
            # Fallback for non-PSD if numerical jitter occurs
            U = np.linalg.cholesky((n + lmbda) * self.P + 1e-9 * np.eye(n))
            
        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = self.x
        for i in range(n):
            sigmas[i+1] = self.x + U[:, i]
            sigmas[i+1+n] = self.x - U[:, i]
            
        # Weights
        wm = np.zeros(2 * n + 1)
        wc = np.zeros(2 * n + 1)
        wm[0] = lmbda / (n + lmbda)
        wc[0] = wm[0] + (1 - self.alpha**2 + self.beta)
        wm[1:] = 1.0 / (2 * (n + lmbda))
        wc[1:] = 1.0 / (2 * (n + lmbda))
        
        return sigmas, wm, wc

    def predict(self, model, dt):
        """
        Predict step: Propagate state mean and covariance forward in time.
        Uses Euler-Maruyama discretization for the state dynamics.
        """
        sigmas, wm, wc = self._generate_sigma_points()
        
        # Propagate through transition function f(x)
        # V_t+1 = V_t + kappa*(theta - V_t)*dt
        # lambda_t+1 = lambda_t (assuming random walk/persistence)
        sigmas_pred = np.zeros_like(sigmas)
        for i in range(len(sigmas)):
            V, lmbda = sigmas[i]
            V_new = V + model.kappa * (model.theta - V) * dt
            V_new = max(V_new, 1e-6) # Feller condition / positivity
            lambda_new = lmbda # Simplest persistence
            sigmas_pred[i] = [V_new, lambda_new]
            
        # Calculate predicted mean and covariance
        self.x = np.sum(wm[:, np.newaxis] * sigmas_pred, axis=0)
        self.P = self.Q * dt
        for i in range(len(sigmas_pred)):
            diff = (sigmas_pred[i] - self.x)[:, np.newaxis]
            self.P += wc[i] * (diff @ diff.T)

    def update(self, observation, model, dt):
        """
        Update step: Refine state estimate using the observed log-return.
        """
        sigmas, wm, wc = self._generate_sigma_points()
        
        # Transform through measurement function h(x)
        # Expected log-return: r_t = (mu - lambda*m - 0.5*V)*dt
        m = np.exp(model.mu_j + 0.5 * model.sigma_j**2) - 1
        y_sigmas = np.zeros(len(sigmas))
        for i in range(len(sigmas)):
            V, lmbda = sigmas[i]
            y_sigmas[i] = (model.mu - lmbda * m - 0.5 * V) * dt
            
        # Mean predicted measurement
        y_mean = np.sum(wm * y_sigmas)
        
        # Measurement covariance S_cov
        S_cov = self.R
        for i in range(len(y_sigmas)):
            diff_y = y_sigmas[i] - y_mean
            S_cov += wc[i] * diff_y**2
            
        # Cross-covariance P_xy
        P_xy = np.zeros(self.n)
        for i in range(len(sigmas)):
            diff_x = sigmas[i] - self.x
            diff_y = y_sigmas[i] - y_mean
            P_xy += wc[i] * diff_x * diff_y
            
        # Kalman gain
        K = P_xy / S_cov
        
        # Update mean and covariance
        self.x = self.x + K * (observation - y_mean)
        self.P = self.P - S_cov * (K[:, np.newaxis] @ K[np.newaxis, :])

    def get_state(self):
        """Return the posterior state estimates."""
        return self.x[0], self.x[1]
