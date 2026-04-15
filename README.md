# SVJD Volatility Arbitrage Framework

A production-grade algorithmic trading system for WTI–Brent crude oil spreads, utilizing a **Stochastic Volatility Jump Diffusion (SVJD)** model with Unscented Kalman Filtering and Bayesian risk management.

## 🚀 Overview

This framework identifies and trades volatility dislocations in oil spreads by combining the **Bates (1996)** stochastic volatility model with a high-performance execution bridge to MetaTrader 5. It is designed for sub-millisecond hot-loop execution and features a fully containerized, fault-tolerant architecture.

### Key Mathematical Pillars
*   **SVJD Dynamics**: Heston stochastic volatility augmented with a compound Poisson jump process.
*   **Pricing Kernel**: Fourier-based pricing using the **Lewis (2001)** formulation and **Albrecher (2007)** stable characteristic functions.
*   **State Estimation**: Dual-state **Unscented Kalman Filter (UKF)** for real-time tracking of latent variance ($V_t$) and jump intensity ($\lambda_t$).
*   **Risk Control**: **Fractional Kelly Criterion** with a Bayesian Beta-conjugate prior for win rate estimation and hierarchical drawdown circuit breakers.

---

## 🏗 Project Structure

| Component | Responsibility |
| :--- | :--- |
| `svjd_engine.py` | Stochastic model dynamics, UKF, and CF pricing kernel. |
| `signal_engine.py` | Realized volatility estimators (Yang-Zhang/Parkinson) and signal generation. |
| `risk_engine.py` | Position sizing, Bayesian probability tracking, and drawdown controls. |
| `mt5_bridge.py` | Asynchronous connection management and order routing to MetaTrader 5. |
| `main.py` | Hot-loop orchestration, nanosecond timing, and health monitoring. |
| `Dockerfile` | Multi-stage production build for Dockerized deployment. |

---

## 🛠 Installation & Setup

### Local Development (macOS/Linux)
The system includes a local virtual environment and library stubs to satisfy IDE static analysis for the Windows-only MetaTrader 5 library.

1.  **Initialize Environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt  # Or manually install numpy, scipy, numba, aiohttp, pandas
    ```

2.  **IDE Configuration**:
    Set your Python interpreter to `./.venv/bin/python`. The local stubs `MetaTrader5.py` and `silicon_metatrader5.py` will clear any import warnings.

### Production Deployment (Docker)
The system is designed to run inside a Docker container bridged to a host-side Silicon MT5 terminal.

1.  **Build Engine**:
    ```bash
    docker build -t svjd-arb-system .
    ```

2.  **Launch Strategy**:
    ```bash
    docker run -d \
      --name svjd_bot \
      -e MT5_LOGIN=YOUR_ID \
      -e MT5_PASSWORD=YOUR_PW \
      -e MT5_SERVER=YOUR_SERVER \
      --add-host=host.docker.internal:host-gateway \
      svjd-arb-system
    ```

---

## 📊 Monitoring & Introspection

The strategy exposes a lightweight `aiohttp` server on port `8080` for real-time system audit.

*   **Health Status**: `GET /health` returns uptime, connectivity, equity, and drawdown metrics.
*   **Internal State**: `GET /state` returns a full JSON snapshot of the UKF state ($V_hat, \lambda_hat$) and current risk parameters.

---

## 🚦 Verification

Run the localized verification suite to test the mathematical and risk kernels:

*   **Risk Logic**: `python3 verify_risk.py`
*   **Stochastic Engine**: `python3 verify_svjd.py`

---

## 📝 Configuration (Environment Variables)

| Variable | Description | Default |
| :--- | :--- | :--- |
| `MT5_LOGIN` | MetaTrader 5 Account Number | (Required) |
| `MT5_PASSWORD` | MetaTrader 5 Investor/Master Password | (Required) |
| `MT5_SERVER` | Broker Server Address | (Required) |
| `KELLY_FRACTION` | Fraction of Kelly sizing to apply | `0.20` |
| `DRAWDOWN_HARD_LIMIT`| Percentage drawdown to trigger strategy halt | `0.06` |
| `LOG_LEVEL` | Logging verbosity (INFO, DEBUG, ERROR) | `INFO` |

---

## ⚖️ Disclaimer

*Trading financial instruments involves significant risk. This software is provided for educational and research purposes. Past performance is not indicative of future results.*
