import numpy as np
from svjd_engine import SVJDModel, SVJDKalmanFilter

def verify_pricing():
    print("--- Pricing Verification ---")
    # Parameters for realistic Heston: mu=0.0, kappa=2.0, theta=0.04 (vol=0.2), sigma_v=0.1, rho=-0.5, lambda=0
    model = SVJDModel(mu=0.0, kappa=2.0, theta=0.04, sigma_v=0.1, rho=-0.5, 
                      lambda_jump=0.0, mu_j=0.0, sigma_j=0.0)
    
    S, V, T, K = 1.0, 0.04, 1.0, 1.0
    price = model.price_spread(S, V, T, K)
    print(f"SVJD Price (Heston ATM, S=1): {price:.6f}")
    # BS price for vol=0.2, T=1, S=1 is approx 0.0796
    
    # Simple Bates: mu=0, jumps only
    model_jumps = SVJDModel(mu=0.0, kappa=2.0, theta=0.04, sigma_v=0.1, rho=-0.5, 
                            lambda_jump=1.0, mu_j=-0.02, sigma_j=0.1)
    price_jumps = model_jumps.price_spread(S, V, T, K)
    print(f"SVJD Price (With Jumps, S=1): {price_jumps:.6f}")

def verify_filter():
    print("\n--- Filter Verification ---")
    # Setup filter
    Q = np.diag([1e-6, 1e-4]) # V and lambda noise
    R = 1e-4 # Observation noise
    kf = SVJDKalmanFilter(Q, R)
    
    model = SVJDModel(mu=0.05, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, 
                      lambda_jump=2.0, mu_j=-0.05, sigma_j=0.1)
    
    # Initialize at true values
    P0 = np.eye(2) * 1e-3
    kf.initialize(V0=0.04, lambda0=2.0, P0=P0)
    
    # Simulate a few steps
    dt = 1/252
    np.random.seed(42)
    
    true_V = 0.04
    true_L = 2.0
    
    est_Vs = []
    
    for _ in range(10):
        # Generate observation
        # log_return approx (mu - 0.5*V)*dt + sqrt(V*dt)*eps
        obs = (model.mu - 0.5 * true_V) * dt + np.sqrt(true_V * dt) * np.random.normal()
        
        kf.predict(model, dt)
        kf.update(obs, model, dt)
        
        v_est, l_est = kf.get_state()
        est_Vs.append(v_est)
        print(f"True V: {true_V:.4f}, Est V: {v_est:.4f}, Est Lambda: {l_est:.4f}")

if __name__ == "__main__":
    verify_pricing()
    verify_filter()
