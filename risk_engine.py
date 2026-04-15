import numpy as np
import json
import time
from dataclasses import dataclass, asdict
from collections import deque
from typing import List, Dict, Optional, Any
from signal_engine import SignalResult

@dataclass(frozen=True)
class PositionOrder:
    """
    Defines a risk-adjusted position order.
    """
    timestamp: float
    signal_id: str
    quantity: float
    direction: int
    status: str
    iv: float
    rv: float
    send: bool

@dataclass(frozen=True)
class RiskStatus:
    """
    Current health status of the Risk Engine.
    """
    timestamp: float
    equity: float
    hwm: float
    drawdown: float
    is_halted: bool
    total_exposure: float
    effective_multiplier: float

class DrawdownBreachError(Exception):
    """
    Custom exception raised when hard drawdown limits are breached.
    """
    def __init__(self, severity: str, timestamp: float, equity: float, drawdown: float):
        self.severity = severity
        self.timestamp = timestamp
        self.equity = equity
        self.drawdown = drawdown
        super().__init__(f"{severity} Drawdown Limit Breached: {drawdown:.2%}")

class KellyEstimator:
    """
    Estimates optimal bet sizing using historical trade performance 
    and a Bayesian Beta-conjugate prior for win probability.
    """
    def __init__(self, alpha_init: float = 1.0, beta_init: float = 1.0, window_size: int = 50):
        self.alpha = alpha_init
        self.beta = beta_init
        self.pnl_history = deque(maxlen=window_size)
    
    def update(self, outcome_pnl: float):
        """
        Update estimator state with a realized P&L outcome.
        """
        if outcome_pnl > 0:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        self.pnl_history.append(outcome_pnl)
    
    def get_kelly_fraction(self) -> float:
        """
        Compute the optimal Kelly fraction f*.
        """
        p = self.alpha / (self.alpha + self.beta)
        
        wins = [x for x in self.pnl_history if x > 0]
        losses = [abs(x) for x in self.pnl_history if x <= 0]
        
        # Payoff ratio b: avg_win / avg_loss
        if not losses:
            b = 1.0 if not wins else 5.0 # Reasonable cap if no losses
        else:
            avg_win = np.mean(wins) if wins else 0.0
            avg_loss = np.mean(losses)
            b = avg_win / avg_loss if avg_loss > 0 else 1.0
            
        if b <= 0: return 0.0
        
        # f* = [p(b + 1) - 1] / b
        f_star = (p * (b + 1) - 1.0) / b
        return max(0.0, f_star)

    def get_state(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "pnl_history": list(self.pnl_history)
        }
    
    def set_state(self, state: Dict[str, Any]):
        self.alpha = state.get("alpha", 1.0)
        self.beta = state.get("beta", 1.0)
        self.pnl_history = deque(state.get("pnl_history", []), maxlen=self.pnl_history.maxlen)

class RiskEngine:
    """
    Production-grade risk engine for WTI-Brent spread volatility arbitrage.
    Implements fractional Kelly, drawdown circuit breakers, and vol-scaled sizing.
    """
    def __init__(self, 
                 initial_equity: float,
                 f_kelly: float = 0.20,
                 t_horizon: float = 0.1, # 10% of a year
                 soft_limit: float = 0.03,
                 hard_limit: float = 0.06,
                 sigma_max: float = 0.01):
        self.equity = initial_equity
        self.hwm = initial_equity
        self.f_kelly = f_kelly
        self.t_horizon = t_horizon
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit
        self.sigma_max = sigma_max
        self.is_halted = False
        
        self.kelly = KellyEstimator()
        self.breach_logs = []

    def check_drawdown(self, current_equity: float) -> float:
        """
        Compute current drawdown and maintain high-water mark.
        """
        if self.is_halted:
            return 1.0
            
        self.equity = current_equity
        if current_equity > self.hwm:
            self.hwm = current_equity
            
        drawdown = (self.hwm - self.equity) / self.hwm
        
        if drawdown >= self.hard_limit:
            self.is_halted = True
            log_entry = {
                "type": "HARD_BREACH",
                "timestamp": time.time(),
                "equity": current_equity,
                "drawdown": drawdown
            }
            self.breach_logs.append(log_entry)
            raise DrawdownBreachError("HARD", log_entry["timestamp"], current_equity, drawdown)
            
        if drawdown >= self.soft_limit:
            if not self.breach_logs or self.breach_logs[-1]["type"] != "SOFT_BREACH":
                self.breach_logs.append({
                    "type": "SOFT_BREACH",
                    "timestamp": time.time(),
                    "equity": current_equity,
                    "drawdown": drawdown
                })
        
        return drawdown

    def size_position(self, 
                      signal: SignalResult, 
                      equity: float, 
                      rv: float,
                      spread_price: float,
                      signal_id: str = "SVJD_VOL_ARB") -> PositionOrder:
        """
        Compute risk-adjusted position notional.
        """
        dd = self.check_drawdown(equity)
        
        if self.is_halted:
            return self._zero_order(signal_id, "STRATEGY_HALTED", signal)

        # 1. Effective Sizing Multiplier (Soft Limit adjustment)
        multiplier = 1.0
        if dd >= self.soft_limit:
            multiplier = 0.5
            
        # 2. Fractional Kelly Calculation
        f_star = self.kelly.get_kelly_fraction()
        f_eff = self.f_kelly * f_star * multiplier
        
        # 3. Volatility-Scaled Notional
        # N = (f_eff * Equity) / (RV * sqrt(T_horizon) * S_spread)
        vol_buffer = max(1e-6, rv * np.sqrt(self.t_horizon))
        # spread_price must be the contract unit price (denominator)
        div = vol_buffer * abs(spread_price)
        
        if div <= 0:
            return self._zero_order(signal_id, "INVALID_PRICE_OR_VOL", signal)
            
        quantity = (f_eff * self.equity) / div
        
        # 4. Correlation/Exposure Adjustment
        # For a single active signal, ensure w^2 * sig^2 <= sig_max^2
        # Which is equivalent to ensuring weight <= sig_max / rv_daily
        rv_daily = rv / np.sqrt(252)
        weight = (quantity * abs(spread_price)) / self.equity
        max_weight = self.sigma_max / max(1e-12, rv_daily)
        
        if weight > max_weight:
            quantity = (max_weight * self.equity) / abs(spread_price)
            status = "MAX_VARIANCE_CAPPED"
        else:
            status = "ACTIVE"
            
        return PositionOrder(
            timestamp=time.time(),
            signal_id=signal_id,
            quantity=float(quantity),
            direction=signal.signal_direction,
            status=status,
            iv=signal.implied_vol,
            rv=signal.realized_vol,
            send=True if quantity > 0 else False
        )

    def _zero_order(self, signal_id: str, reason: str, signal: SignalResult) -> PositionOrder:
        return PositionOrder(
            timestamp=time.time(),
            signal_id=signal_id,
            quantity=0.0,
            direction=0,
            status=reason,
            iv=signal.implied_vol,
            rv=signal.realized_vol,
            send=False
        )

    def get_status(self) -> RiskStatus:
        """Return current risk metrics."""
        dd = (self.hwm - self.equity) / self.hwm
        return RiskStatus(
            timestamp=time.time(),
            equity=self.equity,
            hwm=self.hwm,
            drawdown=dd,
            is_halted=self.is_halted,
            total_exposure=0.0, # Placeholder for aggregator
            effective_multiplier=0.5 if dd >= self.soft_limit else 1.0
        )

    def serialize_state(self) -> str:
        """Return a JSON string representing current engine state."""
        state = {
            "equity": self.equity,
            "hwm": self.hwm,
            "is_halted": self.is_halted,
            "breach_logs": self.breach_logs,
            "kelly_state": self.kelly.get_state()
        }
        return json.dumps(state, indent=4)

    def load_state(self, json_str: str):
        """Restore engine state from JSON string."""
        state = json.loads(json_str)
        self.equity = state.get("equity", self.equity)
        self.hwm = state.get("hwm", self.hwm)
        self.is_halted = state.get("is_halted", False)
        self.breach_logs = state.get("breach_logs", [])
        if "kelly_state" in state:
            self.kelly.set_state(state["kelly_state"])
