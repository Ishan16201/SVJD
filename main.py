import asyncio
import os
import time
import json
import logging
import signal
from enum import Enum
from datetime import datetime
from typing import Dict, Any, Optional
from aiohttp import web
import numpy as np

# Internal module imports
from svjd_engine import SVJDModel, SVJDKalmanFilter
from signal_engine import VolDislocationSignal, SignalResult
from risk_engine import RiskEngine, PositionOrder, DrawdownBreachError
from mt5_bridge import MT5Connection, MT5DataFeed, MT5OrderRouter, TickData

# 1. Structured Logging Configuration
class EventType(Enum):
    TICK_RECEIVED = "TICK_RECEIVED"
    UKF_UPDATE = "UKF_UPDATE"
    SIGNAL_COMPUTED = "SIGNAL_COMPUTED"
    ORDER_DISPATCHED = "ORDER_DISPATCHED"
    ORDER_FILLED = "ORDER_FILLED"
    TICK_DROP = "TICK_DROP"
    LATENCY_BREACH = "LATENCY_BREACH"
    DRAWDOWN_BREACH = "DRAWDOWN_BREACH"
    RECONNECT = "RECONNECT"
    SHUTDOWN = "SHUTDOWN"

class JsonFormatter(logging.Formatter):
    """
    Custom logging formatter that outputs records as single-line JSON objects.
    """
    def format(self, record):
        log_entry = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "module": record.module,
            "event_type": getattr(record, "event_type", "GENERIC"),
            "latency_ns": getattr(record, "latency_ns", None),
            "z_score": getattr(record, "z_score", None),
            "signal_direction": getattr(record, "signal_direction", None),
            "position_size": getattr(record, "position_size", None),
            "equity": getattr(record, "equity", None),
            "drawdown_pct": getattr(record, "drawdown_pct", None),
            "message": record.getMessage()
        }
        return json.dumps({k: v for k, v in log_entry.items() if v is not None})

def setup_logger(log_file="logs/arb_strategy.log"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    
    # Json Formatter
    formatter = JsonFormatter()
    
    # Stdout handler
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)
    
    # Rotating File handler
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(log_file, maxBytes=100*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    return logging.getLogger("SVJDOrchestrator")

logger = setup_logger()

# 2. Orchestration Coroutines
async def tick_ingestion_loop(feed: MT5DataFeed, queue: asyncio.Queue, connection: MT5Connection):
    """
    Asynchronously reads ticks from the MT5 data feed and manages the latest-tick queue.
    """
    logger.info("Starting tick ingestion loop.")
    feed_queue = feed.output_queue
    
    while True:
        try:
            # Wait for tick from feed
            tick: TickData = await feed_queue.get()
            
            # Non-blocking put to the processed queue (maxsize=1)
            try:
                # If queue is full, drop the old tick and log it
                if queue.full():
                    try:
                        queue.get_nowait()
                        logger.warning("Stale tick dropped from processing queue.", 
                                       extra={"event_type": EventType.TICK_DROP.value})
                    except asyncio.QueueEmpty:
                        pass
                
                queue.put_nowait(tick)
                logger.debug(f"Tick received for {tick.symbol}", 
                             extra={"event_type": EventType.TICK_RECEIVED.value})
            except Exception as e:
                logger.error(f"Error in queue management: {e}")
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Critical error in tick ingestion: {e}")
            await asyncio.sleep(1)

async def signal_computation_loop(queue: asyncio.Queue, 
                                 ukf: SVJDKalmanFilter, 
                                 signal_engine: VolDislocationSignal, 
                                 risk_engine: RiskEngine, 
                                 router: MT5OrderRouter,
                                 model: SVJDModel):
    """
    High-performance hot loop for tick-to-trade execution.
    Target latency: < 1ms total. Breach warning at 500µs per stage.
    """
    logger.info("Starting signal computation loop.")
    
    last_prices = {"XTIUSD": 0.0, "XBRUSD": 0.0}
    prev_prices = {"XTIUSD": 0.0, "XBRUSD": 0.0}
    
    while True:
        try:
            tick: TickData = await queue.get()
            start_total = time.perf_counter_ns()
            
            # Step 1: Data Preprocessing
            prev_prices[tick.symbol] = last_prices[tick.symbol]
            last_prices[tick.symbol] = (tick.bid + tick.ask) / 2.0
            
            if last_prices["XTIUSD"] == 0 or last_prices["XBRUSD"] == 0:
                queue.task_done()
                continue
                
            spread = last_prices["XTIUSD"] - last_prices["XBRUSD"]
            prev_spread = prev_prices["XTIUSD"] - prev_prices["XBRUSD"]
            dt = 1/252 # Daily assumption
            
            # Step 2: UKF Update
            t1 = time.perf_counter_ns()
            ukf.update(spread - prev_spread, model, dt)
            t2 = time.perf_counter_ns()
            ukf_latency = t2 - t1
            
            if ukf_latency > 500000:
                logger.warning("UKF stage latency breach", extra={"event_type": EventType.LATENCY_BREACH.value, "latency_ns": ukf_latency})
            
            # Step 3: Signal Computation
            tick_data = {
                'open_spread': spread, 'high_spread': spread, 'low_spread': spread, 'close_spread': spread,
                'prev_close_spread': prev_spread if prev_spread != 0 else spread,
                'wti_front': last_prices["XTIUSD"], 'brent_front': last_prices["XBRUSD"],
                'wti_back': last_prices["XTIUSD"], 'brent_back': last_prices["XBRUSD"],
                'dte': 30, 'r_rate': 0.05, 'wti_conv_yield': 0.02, 'brent_conv_yield': 0.02,
                'timestamp': tick.timestamp_utc
            }
            
            t3 = time.perf_counter_ns()
            result: Optional[SignalResult] = signal_engine.compute(tick_data)
            t4 = time.perf_counter_ns()
            sig_latency = t4 - t3
            
            if sig_latency > 500000:
                logger.warning("Signal stage latency breach", extra={"event_type": EventType.LATENCY_BREACH.value, "latency_ns": sig_latency})
            
            # Step 4: Risk Sizing
            if result:
                t5 = time.perf_counter_ns()
                order: PositionOrder = risk_engine.size_position(result, risk_engine.equity, result.realized_vol, spread)
                t6 = time.perf_counter_ns()
                risk_latency = t6 - t5
                
                if risk_latency > 500000:
                    logger.warning("Risk stage latency breach", extra={"event_type": EventType.LATENCY_BREACH.value, "latency_ns": risk_latency})
                
                # Step 5: Order Dispatch
                if order.send:
                    t7 = time.perf_counter_ns()
                    dispatch_result = await router.dispatch(order)
                    t8 = time.perf_counter_ns()
                    dispatch_latency = t8 - t7
                    
                    logger.info(f"Order dispatched for {order.signal_id}", 
                                extra={
                                    "event_type": EventType.ORDER_DISPATCHED.value,
                                    "latency_ns": dispatch_latency,
                                    "z_score": result.z_score,
                                    "signal_direction": order.direction,
                                    "position_size": order.quantity,
                                    "equity": risk_engine.equity,
                                    "drawdown_pct": (risk_engine.hwm - risk_engine.equity) / risk_engine.hwm
                                })
            
            total_latency = time.perf_counter_ns() - start_total
            if total_latency > 1000000:
                logger.warning("Critical path latency breach", extra={"event_type": EventType.LATENCY_BREACH.value, "latency_ns": total_latency})
                
            queue.task_done()
            
        except DrawdownBreachError as e:
            logger.error(f"Risk Circuit Breaker Tripped: {e}", extra={"event_type": EventType.DRAWDOWN_BREACH.value, "equity": e.equity, "drawdown_pct": e.drawdown})
            # Strategy halted, the RiskEngine.is_halted flag will prevent further orders
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in computation loop: {e}")

# 3. Health & State Monitoring Server
class MonitoringServer:
    def __init__(self, risk_engine: RiskEngine, ukf: SVJDKalmanFilter, signal_engine: VolDislocationSignal):
        self.risk = risk_engine
        self.ukf = ukf
        self.signal = signal_engine
        self.start_time = time.time()
        self.last_tick_time = 0.0

    async def get_health(self, request):
        uptime = time.time() - self.start_time
        status = self.risk.get_status()
        return web.json_response({
            "status": "ok" if not status.is_halted else "halted",
            "uptime_s": uptime,
            "last_tick_age_ms": (time.time() - self.last_tick_time) * 1000,
            "equity": status.equity,
            "drawdown_pct": status.drawdown
        })

    async def get_state(self, request):
        v_hat, lambda_hat = self.ukf.get_state()
        return web.json_response({
            "ukf": {"v_hat": v_hat, "lambda_hat": lambda_hat},
            "risk": json.loads(self.risk.serialize_state()),
            "timestamp": time.time()
        })

async def run_health_server(monitor: MonitoringServer, port: int = 8080):
    app = web.Application()
    app.router.add_get('/health', monitor.get_health)
    app.router.add_get('/state', monitor.get_state)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    logger.info(f"Health and state server listening on port {port}")

# 4. Entry Point & Lifecycle Management
async def main():
    # Load and validate configuration
    try:
        mt5_login = int(os.environ["MT5_LOGIN"])
        mt5_password = os.environ["MT5_PASSWORD"]
        mt5_server = os.environ["MT5_SERVER"]
        kelly_fraction = float(os.environ.get("KELLY_FRACTION", 0.20))
        hard_limit = float(os.environ.get("DRAWDOWN_HARD_LIMIT", 0.06))
    except (KeyError, ValueError) as e:
        logger.error(f"Configuration error: Missing or invalid environment variable {e}")
        return

    # Initialize Components
    model = SVJDModel(mu=0.0, kappa=2.0, theta=0.04, sigma_v=0.1, rho=-0.5, 
                      lambda_jump=0.0, mu_j=0.0, sigma_j=0.0)
    
    ukf = SVJDKalmanFilter(Q=np.eye(2) * 1e-6, R=1e-4)
    ukf.initialize(V0=0.04, lambda0=0.0, P0=np.eye(2)*1e-4)
    
    signal_engine = VolDislocationSignal(ukf, model)
    risk_engine = RiskEngine(initial_equity=100000.0, f_kelly=kelly_fraction, hard_limit=hard_limit)
    router = MT5OrderRouter()
    
    connection = MT5Connection()
    tick_queue = asyncio.Queue(maxsize=1)
    feed_handler = MT5DataFeed(asyncio.Queue()) # Bridge's internal queue
    
    monitor = MonitoringServer(risk_engine, ukf, signal_engine)
    
    # 5. Signal Handlers for Graceful Shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def handle_signal():
        logger.info("Shutdown signal received.", extra={"event_type": EventType.SHUTDOWN.value})
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    # Launch Tasks
    tasks = [
        asyncio.create_task(connection.connect()),
        asyncio.create_task(feed_handler.start()),
        asyncio.create_task(tick_ingestion_loop(feed_handler, tick_queue, connection)),
        asyncio.create_task(signal_computation_loop(tick_queue, ukf, signal_engine, risk_engine, router, model)),
        asyncio.create_task(run_health_server(monitor, 8080))
    ]

    # Wait for stop signal
    await stop_event.wait()

    # Graceful Shutdown Sequence
    logger.info("Executing graceful shutdown...")
    
    # Cancel all open orders and close positions
    await router.cancel_all()
    
    # Cancel all background tasks
    for task in tasks:
        task.cancel()
    
    await asyncio.gather(*tasks, return_exceptions=True)
    
    connection.stop()
    logger.info("Shutdown complete.", extra={"event_type": EventType.SHUTDOWN.value})

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
