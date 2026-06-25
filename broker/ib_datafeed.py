"""
broker/ib_datafeed.py — Dedicated IB datafeed service for micro futures.

Purpose (v3.3):
    Provide historical OHLCV data from the **native IB future** (MES/MGC/MCL/6E),
    real-time with real volume, replacing MT4 CFD proxies for the 4 micros.

Key design decisions:
    - Dedicated **persistent connection** on a dedicated clientId — does not collide
      with executor or monitor/reconcile clientIds. Can run while the system is active.
    - Dedicated thread with a **dedicated event loop**: ib_insync lives entirely on
      that loop, avoiding cross-loop conflicts with the scheduler.
    - `get_historical_bars(asset, tf, bars)` is **synchronous** and **thread-safe**:
      marshals the coroutine onto the dedicated loop via run_coroutine_threadsafe(),
      callable from any thread or loop (e.g. DataCache via asyncio.to_thread).
    - `reqHistoricalDataAsync` with `whatToShow='TRADES'`, `useRTH=False`
      (full electronic session → real volume), `formatDate=1`.
    - IB pacing (≤6 requests/2s) + request serialization on the loop.
    - **Lazy reconnect** on drop: the next fetch retries the connection.

ContFuture from ib_connector is reused (handles MES→CME, MGC→COMEX,
MCL→NYMEX, 6E→EUR/CME as continuous futures with automatic rollover).

Returns a DataFrame with Open/High/Low/Close/Volume columns and datetime index,
in the same format as mt4_datafeed/yfinance (compatible with all agents).

Connection parameters are read from environment variables.
"""

from __future__ import annotations

import asyncio
import math
import os
import threading
import time
from typing import Optional

import pandas as pd
from loguru import logger

from broker.ib_connector import build_contract


# ── Python 3.14: eventkit (imported by ib_insync) calls get_event_loop()
#    at import time; without an event loop in the current thread it raises
#    RuntimeError. Ensure a loop exists BEFORE importing ib_insync.
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# ── Timeframe → IB barSizeSetting ────────────────────────────────────────────
_BAR_SIZE: dict[str, str] = {
    "M5":  "5 mins",
    "M15": "15 mins",
    "M30": "30 mins",
    "H1":  "1 hour",
    "H4":  "4 hours",
    "D1":  "1 day",
    "W1":  "1 week",
}

# Minutes per bar (for intraday duration calculation)
_TF_MINUTES: dict[str, int] = {"H4": 240, "H1": 60, "M30": 30, "M15": 15, "M5": 5}

# Dedicated datafeed clientId (separate from executor and monitor)
_DATAFEED_CLIENT_ID = int(os.getenv("IB_CLIENT_ID_DATA", "96"))

# Pool of clientIds reserved for the datafeed, used in cascade on reconnect.
# After IB daily reset (error 1100/1102), the Gateway keeps the previous clientId
# "in use" for ~30-40 min → connectAsync on the same ID fails with error 326.
# Cycling through free IDs reconnects immediately.
_DATAFEED_CLIENT_IDS = [
    int(os.getenv("IB_CLIENT_ID_DATA",   "96")),
    int(os.getenv("IB_CLIENT_ID_DATA_1", "95")),
    int(os.getenv("IB_CLIENT_ID_DATA_2", "94")),
]


def _duration_str(tf: str, bars: int) -> str:
    """
    Compute durationStr using only valid IB units (S/D/W/Y).
    For intraday, converts to CALENDAR days with margin to cover
    weekends/holidays (bars only form during market hours).
    The caller slices to `bars` after download.
    """
    if tf == "W1":
        years = max(1, math.ceil(bars / 52))
        return f"{years} Y"
    if tf == "D1":
        return f"{bars} D" if bars <= 365 else f"{math.ceil(bars / 365)} Y"
    per_bar = _TF_MINUTES.get(tf)
    if per_bar is None:
        return f"{bars} D"
    total_min = bars * per_bar
    days      = max(1, math.ceil(total_min / (60 * 24)))
    # ~1.6× margin + 1 for session gaps (futures ~23h/day, 5 days/week)
    days      = math.ceil(days * 1.6) + 1
    return f"{days} D"


class IBDataFeed:
    """Persistent, synchronous, thread-safe IB datafeed (module singleton)."""

    def __init__(self):
        self._host      = os.getenv("IB_HOST", "127.0.0.1")
        self._port      = int(os.getenv("IB_PORT", "7497"))
        self._client_id = _DATAFEED_CLIENT_ID

        self._ib    = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread]        = None
        self._ready      = threading.Event()   # set when loop + first connection attempt done
        self._start_lock = threading.Lock()    # serializes start() across threads

        self._connected  = False
        self._req_lock: Optional[asyncio.Lock] = None
        self._req_times: list[float]           = []

    # ── Properties ─────────────────────────────────────────────────────────────
    @property
    def connected(self) -> bool:
        try:
            return bool(self._ib is not None and self._ib.isConnected())
        except Exception:
            return False

    # ── Startup / dedicated loop ───────────────────────────────────────────────
    def start(self) -> bool:
        """
        Start (idempotent) the dedicated thread+loop and attempt connection.
        Returns True if connected. Call from scheduler at startup.
        """
        self._ensure_thread()
        self._ready.wait(timeout=20)
        return self.connected

    def _ensure_thread(self) -> None:
        """Create and start the loop thread if not already alive."""
        with self._start_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._ready.clear()
            self._thread = threading.Thread(
                target=self._thread_main, name="IBDataFeed", daemon=True
            )
            self._thread.start()

    def _thread_main(self) -> None:
        """Thread entry point: create loop + IB(), connect, then run_forever()."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        try:
            from ib_insync import IB
            self._ib = IB()
            self._ib.errorEvent       += self._on_error
            self._ib.disconnectedEvent += self._on_disconnect
        except Exception as e:
            logger.error(f"[IBDataFeed] ib_insync init failed: {e}")
            self._ready.set()
            return

        try:
            loop.run_until_complete(self._connect_async())
        except Exception as e:
            logger.warning(f"[IBDataFeed] Initial connection failed: {e} — will retry on demand")
        finally:
            self._ready.set()

        try:
            loop.run_forever()
        except Exception as e:
            logger.error(f"[IBDataFeed] Loop terminated: {e}")

    async def _connect_async(self) -> bool:
        if self._req_lock is None:
            self._req_lock = asyncio.Lock()
        if self._ib is not None and self._ib.isConnected():
            self._connected = True
            return True
        await self._ib.connectAsync(
            self._host, self._port, clientId=self._client_id, timeout=15
        )
        self._connected = self._ib.isConnected()
        if self._connected:
            logger.success(
                f"[IBDataFeed] Connected to {self._host}:{self._port} "
                f"(clientId={self._client_id})"
            )
        return self._connected

    async def _ensure_connected_async(self) -> bool:
        if self._ib is not None and self._ib.isConnected():
            return True
        if self._req_lock is None:
            self._req_lock = asyncio.Lock()
        # Cycle through clientId pool on reconnect to avoid IB error 326
        for cid in _DATAFEED_CLIENT_IDS:
            try:
                try:
                    if self._ib is not None and self._ib.isConnected():
                        self._ib.disconnect()
                except Exception:
                    pass
                await asyncio.sleep(0.3)
                await self._ib.connectAsync(
                    self._host, self._port, clientId=cid, timeout=15
                )
                if self._ib.isConnected():
                    self._client_id = cid
                    self._connected = True
                    logger.success(f"[IBDataFeed] Reconnected (clientId={cid})")
                    return True
            except Exception as e:
                logger.debug(f"[IBDataFeed] Reconnect clientId={cid} failed: {e}")
                continue
        self._connected = False
        return False

    # ── Event handlers ─────────────────────────────────────────────────────────
    def _on_disconnect(self) -> None:
        self._connected = False
        logger.warning("[IBDataFeed] Disconnected from IB — will reconnect on next fetch")

    def _on_error(self, *args) -> None:
        try:
            code = args[1] if len(args) > 1 else None
            msg  = args[2] if len(args) > 2 else ""
            # ignore informational messages (farm OK / connection)
            if code not in (2104, 2106, 2107, 2119, 2158):
                logger.debug(f"[IBDataFeed] IB error {code}: {msg}")
        except Exception:
            pass

    # ── IB pacing (executed under _req_lock, single coro at a time) ───────────
    async def _pace(self) -> None:
        now = time.monotonic()
        self._req_times = [t for t in self._req_times if now - t < 2.0]
        if len(self._req_times) >= 6:
            wait = 2.0 - (now - self._req_times[0]) + 0.05
            if wait > 0:
                logger.debug(f"[IBDataFeed] pacing: waiting {wait:.2f}s")
                await asyncio.sleep(wait)
            now = time.monotonic()
            self._req_times = [t for t in self._req_times if now - t < 2.0]
        self._req_times.append(now)

    # ── Public API: synchronous thread-safe historical data ───────────────────
    def get_historical_bars(self, asset: str, tf: str,
                            bars: int = 300) -> Optional[pd.DataFrame]:
        """
        Download `bars` bars for `asset`/`tf` from IB native futures. Synchronous and thread-safe.
        Returns DataFrame Open/High/Low/Close/Volume (datetime index) or None.
        """
        if tf not in _BAR_SIZE:
            logger.debug(f"[IBDataFeed] Unsupported timeframe: {tf}")
            return None
        self._ensure_thread()
        if self._loop is None or not self._loop.is_running():
            self._ready.wait(timeout=20)
        if self._loop is None:
            return None
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._get_bars_async(asset, tf, bars), self._loop
            )
            return fut.result(timeout=35)
        except Exception as e:
            logger.debug(f"[IBDataFeed] {asset} {tf}: {e}")
            return None

    async def _get_bars_async(self, asset: str, tf: str,
                              bars: int) -> Optional[pd.DataFrame]:
        if not await self._ensure_connected_async():
            return None
        bar_size = _BAR_SIZE[tf]
        if self._req_lock is None:
            self._req_lock = asyncio.Lock()
        async with self._req_lock:
            await self._pace()
            try:
                contract = build_contract(asset)
            except Exception as e:
                logger.debug(f"[IBDataFeed] build_contract {asset}: {e}")
                return None
            await self._ib.qualifyContractsAsync(contract)
            data = await self._ib.reqHistoricalDataAsync(
                contract=contract,
                endDateTime="",
                durationStr=_duration_str(tf, bars),
                barSizeSetting=bar_size,
                whatToShow="TRADES",   # real volume (futures)
                useRTH=False,          # full electronic session
                formatDate=1,
                keepUpToDate=False,
            )

        if not data:
            logger.debug(f"[IBDataFeed] No bars for {asset} {tf}")
            return None

        rows = []
        for bar in data:
            try:
                vol = float(bar.volume)
                vol = int(vol) if vol >= 0 else 0
            except Exception:
                vol = 0
            rows.append({
                "Open":   float(bar.open),
                "High":   float(bar.high),
                "Low":    float(bar.low),
                "Close":  float(bar.close),
                "Volume": vol,
            })
        df       = pd.DataFrame(rows)
        df.index = pd.to_datetime([bar.date for bar in data])
        if len(df) > bars:
            df = df.tail(bars)
        logger.debug(f"[IBDataFeed] {asset} {tf}: {len(df)} bars from IB (futures)")
        return df

    # ── Clean shutdown (for scheduler shutdown) ────────────────────────────────
    def stop(self) -> None:
        try:
            if self._loop is not None and self._loop.is_running():
                def _shutdown():
                    try:
                        if self._ib is not None and self._ib.isConnected():
                            self._ib.disconnect()
                    finally:
                        self._loop.stop()
                self._loop.call_soon_threadsafe(_shutdown)
        except Exception as e:
            logger.debug(f"[IBDataFeed] stop: {e}")


# Module singleton — import this wherever IB datafeed is needed.
# Does not connect at import time: call ib_datafeed.start() (scheduler)
# or simply get_historical_bars(...) (lazy start).
ib_datafeed = IBDataFeed()
