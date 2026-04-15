import numpy as np
from dataclasses import dataclass
from collections import deque
from typing import Optional, Dict
from svjd_engine import SVJDModel, SVJDKalmanFilter

@dataclass(frozen=True)
class SignalResult:
    """
    Encapsulates the output of the volatility dislocation signal logic.
    """
    timestamp: float
    signal_direction: int  # 1 for Long Vol, -1 for Short Vol, 0 for Neutral
    z_score: float
    implied_vol: float
    realized_vol: float
    fair_value_spread: float
    roll_yield: float

class RealizedVolEstimator:
    """
    Implements multi-flavor realized volatility estimation (Parkinson, Yang-Zhang).
    Maintains a rolling window of OHLC data.
    """
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.ohlc_window = deque(maxlen=window_size)
        
    def update(self, open_p: float, high_p: float, low_p: float, close_p: float, prev_close: float):
        """
        Add a new bar to the estimator.
        Prices must be in natural units (levels).
        """
        if prev_close <= 0: return
        
        # Log components for YZ and Parkinson
        o = np.log(open_p / prev_close)
        u = np.log(high_p / open_p)
        d = np.log(low_p / open_p)
        c = np.log(close_p / open_p)
        
        # Parkinson individual range
        p_range = np.log(high_p / low_p)**2
        
        # RS individual component
        rs_comp = u * (u - c) + d * (d - c)
        
        self.ohlc_window.append({
            'o': o, 'u': u, 'd': d, 'c': c,
            'p_range': p_range, 'rs_comp': rs_comp,
            'log_ret_cc': np.log(close_p / prev_close) # for close-to-close component
        })

    def get_rv(self, weight_yz: float = 0.5) -> float:
        """
        Compute the weighted blend of Yang-Zhang and Parkinson estimators.
        Returns annualized realized volatility.
        """
        if len(self.ohlc_window) < self.window_size:
            return 0.0
            
        n = len(self.ohlc_window)
        
        # 1. Parkinson Estimator
        # sum(ln(H/L)^2) / (4 * ln(2) * n)
        sum_p_range = sum(bar['p_range'] for bar in self.ohlc_window)
        var_park = sum_p_range / (4.0 * np.log(2.0) * n)
        
        # 2. Yang-Zhang Estimator
        # Components: Sigma_O (Open), Sigma_C (Close), Sigma_RS (Rogers-Satchell)
        obs_o = [bar['o'] for bar in self.ohlc_window]
        obs_c = [bar['c'] for bar in self.ohlc_window]
        
        var_o = np.var(obs_o, ddof=1)
        var_c = np.var(obs_c, ddof=1)
        var_rs = sum(bar['rs_comp'] for bar in self.ohlc_window) / n
        
        # weighting k
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        var_yz = var_o + k * var_c + (1 - k) * var_rs
        
        # Blend and Annualize (assumes 252 bars per year)
        var_blend = weight_yz * var_yz + (1.0 - weight_yz) * var_park
        return np.sqrt(var_blend * 252.0)

class VolDislocationSignal:
    """
    Monitors the gap between SVJD-implied volatility and realized volatility.
    Generates trading signals using a rolling Z-score mechanism.
    """
    def __init__(self, 
                 ukf: SVJDKalmanFilter, 
                 model: SVJDModel,
                 rv_window: int = 20,
                 z_lookback: int = 60,
                 z_entry: float = 2.0,
                 z_exit: float = 0.5):
        self.ukf = ukf
        self.model = model
        self.rv_est = RealizedVolEstimator(window_size=rv_window)
        
        self.z_lookback = z_lookback
        self.z_entry = z_entry
        self.z_exit = z_exit
        
        # Welford state for vol dislocation (delta)
        self.delta_buffer = deque(maxlen=z_lookback)
        self.mean_delta = 0.0
        self.m2_delta = 0.0
        self.count_delta = 0
        
        self.current_signal = 0 # 0=Neutral, 1=Long Vol, -1=Short Vol

    def _update_welford(self, delta: float):
        """
        Incremental update for rolling mean and variance using Welford.
        Handles window removal first, then addition.
        """
        if len(self.delta_buffer) == self.z_lookback:
            # Remove oldest
            old_val = self.delta_buffer[0]
            if self.count_delta > 1:
                old_mean = self.mean_delta
                self.mean_delta = (self.count_delta * self.mean_delta - old_val) / (self.count_delta - 1)
                self.m2_delta -= (old_val - old_mean) * (old_val - self.mean_delta)
                self.count_delta -= 1
            else:
                self.mean_delta = 0.0
                self.m2_delta = 0.0
                self.count_delta = 0
                
        # Add newest
        self.delta_buffer.append(delta)
        self.count_delta += 1
        diff = delta - self.mean_delta
        self.mean_delta += diff / self.count_delta
        self.m2_delta += diff * (delta - self.mean_delta)

    def compute(self, tick_data: Dict[str, float]) -> Optional[SignalResult]:
        """
        Process a new tick and update signal state.
        
        tick_data expected fields: 
            'open_spread', 'high_spread', 'low_spread', 'close_spread', 'prev_close_spread',
            'wti_front', 'brent_front', 'wti_back', 'brent_back', 'dte', 'r_rate', 
            'wti_conv_yield', 'brent_conv_yield', 'timestamp'
        """
        # 1. Update Realized Vol
        self.rv_est.update(
            tick_data['open_spread'], tick_data['high_spread'], 
            tick_data['low_spread'], tick_data['close_spread'],
            tick_data['prev_close_spread']
        )
        rv = self.rv_est.get_rv()
        if rv <= 0: return None
        
        # 2. Extract Implied Vol from UKF
        v_hat, _ = self.ukf.get_state()
        iv = np.sqrt(max(1e-12, v_hat) * 252.0)
        
        # 3. Compute Dislocation and rolling stats
        delta = iv - rv
        self._update_welford(delta)
        
        if self.count_delta < self.z_lookback:
            return None
            
        std_delta = np.sqrt(max(1e-12, self.m2_delta / (self.count_delta - 1)))
        z_score = (delta - self.mean_delta) / std_delta
        
        # 4. Fair Value & Roll Yield Logic
        T = tick_data['dte'] / 365.25
        r = tick_data['r_rate']
        c_wti = tick_data['wti_conv_yield']
        c_brent = tick_data['brent_conv_yield']
        
        # Cost of Carry Adjusted Fair Value: F_spread = (F_WTI * e^(r-c_WTI)*T) - (F_Brent * e^(r-c_Brent)*T)
        fv_wti = tick_data['wti_front'] * np.exp((r - c_wti) * T)
        fv_brent = tick_data['brent_front'] * np.exp((r - c_brent) * T)
        fair_value = fv_wti - fv_brent
        
        # Roll Yield: (F_back - F_front) / F_front * (1 / days_to_expiry)
        # Note: We aggregate RY for the spread as a delta between WTI and Brent roll yields or just use WTI side
        # For simplicity, following user formula literally on WTI side as proxy or composite
        ry_wti = (tick_data['wti_back'] - tick_data['wti_front']) / tick_data['wti_front'] * (1 / max(1, tick_data['dte']))
        
        # 5. Signal State Machine
        if self.current_signal == 0:
            if z_score < -self.z_entry:
                self.current_signal = 1
            elif z_score > self.z_entry:
                self.current_signal = -1
        else:
            # Exit logic
            if abs(z_score) < self.z_exit:
                self.current_signal = 0
                
        return SignalResult(
            timestamp=tick_data['timestamp'],
            signal_direction=self.current_signal,
            z_score=z_score,
            implied_vol=iv,
            realized_vol=rv,
            fair_value_spread=fair_value,
            roll_yield=ry_wti
        )
