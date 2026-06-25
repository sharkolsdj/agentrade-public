"""
utils/mt4_lock.py
Global lock to coordinate access to the MT4 DWX ZeroMQ connector.

Prevents race conditions between:
- Heartbeat loop (every 30s)
- Position monitor (every 120s)
- Historical data requests

Usage:
```python
from utils.mt4_lock import MT4_GET_TRADES_LOCK
async with MT4_GET_TRADES_LOCK:
    connector._DWX_MTX_GET_ALL_OPEN_TRADES_()
    # ... wait for response
```
"""

import asyncio

# Global lock for GET_ALL_OPEN_TRADES — prevents race conditions
MT4_GET_TRADES_LOCK = asyncio.Lock()

# Global lock for historical data requests — serializes HIST_REQUEST
MT4_HIST_DATA_LOCK = asyncio.Lock()

# Global lock for order execution
MT4_EXECUTION_LOCK = asyncio.Lock()
