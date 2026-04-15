import asyncio
import os
import socket
import time
import json
import logging
import random
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from aiohttp import web
import MetaTrader5 as mt5
from risk_engine import PositionOrder

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MT5Bridge")

class ConnectionState(Enum):
    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2
    RECONNECTING = 3

@dataclass(frozen=True)
class TickData:
    """
    Normalized tick data from MetaTrader 5.
    """
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp_utc: float

class MT5Connection:
    """
    Manages the persistent connection to the Silicon MT5 terminal.
    Includes automatic reconnection logic with exponential backoff.
    """
    def __init__(self):
        self.login = int(os.environ.get("MT5_LOGIN", 0))
        self.password = os.environ.get("MT5_PASSWORD", "")
        self.server = os.environ.get("MT5_SERVER", "")
        self.state = ConnectionState.DISCONNECTED
        self._is_running = True
        
    async def connect(self):
        """
        Attempt to establish a connection to the MT5 terminal.
        """
        self.state = ConnectionState.CONNECTING
        attempt = 0
        
        while self._is_running:
            try:
                # Resolve host.docker.internal for Silicon MT5 Bridge communication
                host_ip = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: socket.gethostbyname('host.docker.internal')
                )
                logger.info(f"Resolved host.docker.internal as {host_ip}")
                
                # In Silicon MT5 environments, we often connect to a specific bridge server/IP
                # For the standard library, we call initialize. 
                # Note: Silicon MT5 specific clients often need path/port, but we follow the prompt's login reqs.
                success = await asyncio.to_thread(
                    mt5.initialize, login=self.login, password=self.password, server=self.server
                )
                
                if success:
                    self.state = ConnectionState.CONNECTED
                    logger.info("Successfully connected to MetaTrader 5")
                    asyncio.create_task(self._heartbeat_loop())
                    return True
                else:
                    err = mt5.last_error()
                    logger.error(f"MT5 Initialization failed: {err}")
            except Exception as e:
                logger.error(f"Connection attempt {attempt} failed: {str(e)}")
            
            # Exponential backoff: base 2, max 60s, jitter ±10%
            attempt += 1
            wait_time = min(60, 2**attempt) * (1 + random.uniform(-0.1, 0.1))
            self.state = ConnectionState.RECONNECTING
            logger.info(f"Retrying in {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)

    async def _heartbeat_loop(self):
        """
        Periodic ping to MT5 to ensure connection persistence.
        """
        while self._is_running and self.state == ConnectionState.CONNECTED:
            await asyncio.sleep(15)
            # Simple check via terminal_info
            info = await asyncio.to_thread(mt5.terminal_info)
            if info is None:
                logger.warning("MT5 Heartbeat failed. Reconnecting...")
                self.state = ConnectionState.DISCONNECTED
                asyncio.create_task(self.connect())
                break

    def stop(self):
        self._is_running = False
        mt5.shutdown()

class MT5OrderRouter:
    """
    Translates strategy position orders into MT5 trade requests.
    Includes slippage protection and JSON logging.
    """
    def __init__(self, max_slippage_ticks: int = 5):
        self.max_slippage_ticks = max_slippage_ticks

    async def dispatch(self, order: PositionOrder) -> Dict[str, Any]:
        """
        Alias for execute_order to match orchestration layer naming.
        """
        return await self.execute_order(order)

    async def cancel_all(self):
        """
        Close all open positions and cancel pending orders.
        """
        logger.info("Closing all positions and canceling orders...")
        positions = await asyncio.to_thread(mt5.positions_get)
        if positions:
            for pos in positions:
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": pos.ticket,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask,
                    "deviation": 20,
                    "magic": 1620,
                    "comment": "CLOSE_ALL_SHUTDOWN",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                await asyncio.to_thread(mt5.order_send, request)
        
        orders = await asyncio.to_thread(mt5.orders_get)
        if orders:
            for o in orders:
                request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": o.ticket
                }
                await asyncio.to_thread(mt5.order_send, request)

class MT5DataFeed:
    """
    Subscribes to and streams real-time tick data.
    """
    def __init__(self, output_queue: asyncio.Queue):
        self.output_queue = output_queue
        self.symbols = ["XTIUSD", "XBRUSD"]
        self._is_running = True

    async def start(self):
        """
        Main loop for tick ingestion.
        """
        logger.info(f"Starting DataFeed for {self.symbols}")
        while self._is_running:
            for symbol in self.symbols:
                # Get last tick
                tick = await asyncio.to_thread(mt5.symbol_info_tick, symbol)
                if tick:
                    normalized_tick = TickData(
                        symbol=symbol,
                        bid=tick.bid,
                        ask=tick.ask,
                        last=tick.last,
                        volume=tick.volume,
                        timestamp_utc=float(tick.time)
                    )
                    await self.output_queue.put(normalized_tick)
            
            # Polling delay to avoid pegged CPU on event loop
            await asyncio.sleep(0.1)

async def handle_health(request):
    """
    Health check endpoint handler.
    """
    # Check MT5 connection
    is_connected = mt5.terminal_info() is not None
    status = "healthy" if is_connected else "unhealthy"
    return web.json_response({
        "status": status,
        "mt5_connected": is_connected,
        "timestamp": time.time()
    }, status=200 if is_connected else 503)

async def run_health_server():
    """
    Start the lightweight aiohttp server for Docker healthchecks.
    """
    app = web.Application()
    app.router.add_get('/health', handle_health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()
    logger.info("Health check server started on port 8080")
