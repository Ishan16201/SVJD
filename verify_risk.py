from risk_engine import RiskEngine, KellyEstimator, DrawdownBreachError
from signal_engine import SignalResult
import numpy as np

def test_risk_logic():
    print("--- Testing Risk Engine Implementation ---")
    
    # Initial Setup
    initial_equity = 100000.0
    risk = RiskEngine(initial_equity=initial_equity)
    
    # Mock Signal
    sig = SignalResult(
        timestamp=0.0,
        signal_direction=1,
        z_score=-2.5,
        implied_vol=0.3, # 30%
        realized_vol=0.2, # 20%
        fair_value_spread=0.0,
        roll_yield=0.0
    )
    
    # 1. Test Kelly and Initial Sizing
    # Initially Kelly is 0 because no pnl_history
    order = risk.size_position(sig, equity=initial_equity, rv=0.2, spread_price=1.0)
    print(f"Initial Order (no hist): Q={order.quantity}, Status={order.status}")
    
    # Add some wins to Kelly
    for _ in range(10):
        risk.kelly.update(100.0)
    
    order_kelly = risk.size_position(sig, equity=initial_equity, rv=0.2, spread_price=1.0)
    print(f"Post-Win Order: Q={order_kelly.quantity}, Status={order_kelly.status}")
    
    # 2. Test Drawdown Circuit Breakers
    print("\n--- Testing Drawdown Breakers ---")
    # Soft breach at 3%
    equity_soft = initial_equity * 0.965 # 3.5% drawdown
    order_soft = risk.size_position(sig, equity=equity_soft, rv=0.2, spread_price=1.0)
    print(f"Soft Breach Order (3.5% DD): Q={order_soft.quantity}, Status={order_soft.status}")
    
    # Hard breach at 6%
    equity_hard = initial_equity * 0.93 # 7% drawdown
    try:
        print("Attempting size during hard breach...")
        risk.size_position(sig, equity=equity_hard, rv=0.2, spread_price=1.0)
    except DrawdownBreachError as e:
        print(f"Caught Expected Exception: {e}")
        
    status = risk.get_status()
    print(f"Risk Halted: {status.is_halted}")

    # 3. Test Serialization
    print("\n--- Testing Serialization ---")
    json_state = risk.serialize_state()
    print("State Serialized Successfully.")
    
    new_risk = RiskEngine(initial_equity=100000)
    new_risk.load_state(json_state)
    print(f"State Restored. Halted={new_risk.is_halted}, HWM={new_risk.hwm}")

if __name__ == "__main__":
    test_risk_logic()
