# Mock implementation of MetaTrader5 for IDE static analysis on macOS
# This file allows the IDE to resolve imports without the library being installed.

def initialize(*args, **kwargs): return True
def shutdown(): pass
def last_error(): return (0, "Success")
def terminal_info(): return type('Info', (), {'connected': True})()
def symbol_info(symbol): return type('Symbol', (), {'name': symbol})()
def symbol_info_tick(symbol): return type('Tick', (), {'bid': 0.0, 'ask': 0.0, 'last': 0.0, 'volume': 0.0, 'time': 0})()
def order_send(request): return type('Result', (), {'retcode': 10009, 'order': 12345})()
def positions_get(*args, **kwargs): return []
def orders_get(*args, **kwargs): return []

# Enums
TRADE_ACTION_DEAL = 1
TRADE_ACTION_REMOVE = 2
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
ORDER_TIME_GTC = 0
ORDER_FILLING_IOC = 1
TRADE_RETCODE_DONE = 10009
