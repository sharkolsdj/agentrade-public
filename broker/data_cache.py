"""
broker/data_cache.py  —  v3.1
Centralized OHLCV cache for AgenTrade v3.

v3.1 changes:
  - MT4 DWX ZeroMQ added as primary source for ALL assets
  - tvdatafeed as secondary fallback for M5, M15, M30
  - yfinance remains as last fallback for all timeframes
  - Unified hierarchy: MT4 DWX → tvdatafeed → yfinance

Data hierarchy v3.1:
  ALL ASSETS, ALL TF → MT4 DWX ZeroMQ (primary)
                     → tvdatafeed (fallback for M5/M15/M30)
                     → yfinance (last fallback)

MT4 DWX setup (one-time):
  - MT4 with DWX ZeroMQ EA active
  - Ports 32768/32769/32770 open (see .env.example)
  - Demo/live account configured
"""

import asyncio
import time
from datetime import datetime
from typing import Optional
from loguru import logger

import pandas as pd
import yfinance as yf
import threading as _threading

# Global lock to serialize yfinance downloads — not thread-safe under concurrent
# asyncio.to_thread calls, causing cross-asset data contamination.
_YF_DOWNLOAD_LOCK = _threading.Lock()

from broker.mt4_datafeed import mt4_datafeed
from broker.ib_datafeed   import ib_datafeed

# Assets whose data comes from IB native futures (never from MT4 CFD)
IB_MICROS = {"MES", "MGC", "MCL", "6E"}

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

ALL_ASSETS = [
    "EURUSD", "GBPUSD", "USDJPY", "GBPJPY", "USDCAD",
    "EURAUD", "EURGBP", "NZDJPY", "EURCHF",
    "XAUUSD", "XAGUSD", "MGC",
    "BTCUSD", "ETHUSD",
    "MES", "MCL", "NAS100", "6E",
    "NZDUSD", "AUDJPY", "CHFJPY",
    "GER40", "US2000", "DJ30",
    "SP500", "USOUSD",
]

TIMEFRAMES = ["W1", "D1", "H4", "H1", "M30", "M15", "M5"]

# Priority assets for reduced startup loading (avoids MT4 thundering herd)
STARTUP_PRIORITY_ASSETS = [
    "EURUSD", "GBPUSD", "XAUUSD", "USDJPY",
    "GBPJPY", "EURGBP",
    "MES", "MGC", "MCL", "6E",
]
STARTUP_PRIORITY_TIMEFRAMES = ["D1", "H4", "H1", "M15", "M5"]

FULL_STARTUP_LOADING = False
FULL_LOAD_STRATEGY   = "sequential_waves"

# Differentiated refresh intervals per timeframe group.
# Loop runs every CACHE_REFRESH_INTERVAL (5 min); cycle counter decides which
# timeframes to update — avoids 126 MT4 requests every 5 min.
#
#   every_n=1  → every cycle (5 min)
#   every_n=2  → every 2 cycles (10 min)
#   every_n=6  → every 6 cycles (30 min)
REFRESH_GROUPS = {
    "FAST":   {"tfs": ["M5", "M15", "M30"], "every_n": 1},  # 5 min — entry triggers
    "MEDIUM": {"tfs": ["H1"],               "every_n": 2},  # 10 min — intraday structure
    "HIGH":   {"tfs": ["H4"],               "every_n": 6},  # 30 min — daily bias
    "SLOW":   {"tfs": ["D1", "W1"],         "every_n": 6},  # 30 min — ICT levels (PDH/PDL, weekly)
}

TV_TIMEFRAMES = {"M5", "M15", "M30"}
YF_TIMEFRAMES = {"W1", "D1", "H4", "H1"}

YFINANCE_MAP = {
    "EURUSD": "EURUSD=X",  "GBPUSD": "GBPUSD=X",  "USDJPY": "USDJPY=X",
    "GBPJPY": "GBPJPY=X",  "USDCAD": "USDCAD=X",  "EURAUD": "EURAUD=X",
    "EURGBP": "EURGBP=X",  "NZDJPY": "NZDJPY=X",  "EURCHF": "EURCHF=X",
    "XAUUSD": "GC=F",      "MGC":    "GC=F",
    "XAGUSD": "SI=F",
    "BTCUSD": "BTC-USD",   "ETHUSD": "ETH-USD",
    "MES":    "ES=F",
    "MCL":    "CL=F",      "NAS100": "NQ=F",
    "6E":     "6E=F",
    "NZDUSD": "NZDUSD=X",  "AUDJPY": "AUDJPY=X",  "CHFJPY": "CHFJPY=X",
    "GER40":  "^GDAXI",    "US2000": "^RUT",       "DJ30":   "^DJI",
    "SP500":  "ES=F",      "USOUSD": "CL=F",
}

YF_PARAMS = {
    "W1":  ("2y",  "1wk"),
    "D1":  ("1y",  "1d"),
    "H4":  ("60d", "4h"),
    "H1":  ("30d", "1h"),
    "M30": ("10d", "30m"),
    "M15": ("10d", "15m"),
    "M5":  ("5d",  "5m"),
}

# TradingView symbol mapping: (symbol, exchange)
TV_SYMBOL_MAP = {
    "EURUSD": ("EURUSD", "FX_IDC"),
    "GBPUSD": ("GBPUSD", "FX_IDC"),
    "USDJPY": ("USDJPY", "FX_IDC"),
    "GBPJPY": ("GBPJPY", "FX_IDC"),
    "USDCAD": ("USDCAD", "FX_IDC"),
    "EURAUD": ("EURAUD", "FX_IDC"),
    "EURGBP": ("EURGBP", "FX_IDC"),
    "NZDJPY": ("NZDJPY", "FX_IDC"),
    "EURCHF": ("EURCHF", "FX_IDC"),
    "XAUUSD": ("XAUUSD", "OANDA"),
    "XAGUSD": ("XAGUSD", "OANDA"),
    "MGC":    ("GC1!",   "COMEX"),
    "BTCUSD": ("BTCUSD", "COINBASE"),
    "ETHUSD": ("ETHUSD", "COINBASE"),
    "MES":    ("ES1!",   "CME"),
    "MCL":    ("CL1!",   "NYMEX"),
    "NAS100": ("NQ1!",   "CME"),
    "6E":     ("6E1!",   "CME"),
    "NZDUSD": ("NZDUSD", "FX_IDC"),
    "AUDJPY": ("AUDJPY", "FX_IDC"),
    "CHFJPY": ("CHFJPY", "FX_IDC"),
    # Indices and USOUSD: MT4 primary + yfinance fallback
}

TV_INTERVAL_MAP = {"M5": "5", "M15": "15", "M30": "30"}

TV_BARS = {"M5": 500, "M15": 300, "M30": 200}

CACHE_REFRESH_INTERVAL = 5 * 60   # 5 minutes

MAX_AGE_MINUTES = 10

# Per-timeframe freshness thresholds (minutes).
# A series is "fresh" if last saved within one candle duration (small margin).
TF_FRESHNESS_MIN = {
    "M5": 6, "M15": 16, "M30": 32, "H1": 65, "H4": 245, "D1": 1450, "W1": 10100,
}


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Normalize DataFrame: MultiIndex → flat, dedup, lowercase cols, tz strip."""
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]
    rename = {
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df.dropna(subset=["Close"]) if "Close" in df.columns else None


def _normalize_tv(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Normalize DataFrame from tvdatafeed (already lowercase columns)."""
    if df is None or df.empty:
        return None
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]
    rename = {
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df.dropna(subset=["Close"]) if "Close" in df.columns else None


# ─────────────────────────────────────────────────────────────────────────────
# TvDatafeed Manager
# ─────────────────────────────────────────────────────────────────────────────

class TvDatafeedManager:
    """
    Manages TradingView connection via tvdatafeed.
    Singleton — connection created once.
    Free Basic account is sufficient for OHLCV download.

    Setup:
      pip install tvdatafeed-enhanced
      Anonymous access — works for forex/crypto (no credentials needed)
    """

    def __init__(self):
        self._tv          = None
        self._available   = False
        self._initialized = False

    def initialize(self) -> bool:
        """
        Attempt to initialize tvdatafeed.
        Returns True if available, False if not installed or error.
        """
        if self._initialized:
            return self._available

        try:
            from tvDatafeed import TvDatafeed, Interval
            # Anonymous access — credentials cause rate limits/CAPTCHA
            self._tv        = TvDatafeed()
            self._available = True
            logger.info("[tvdatafeed] Connected in anonymous mode")

        except ImportError:
            logger.warning(
                "[tvdatafeed] Library not installed — "
                "install with: pip install tvdatafeed-enhanced\n"
                "Falling back to yfinance for M5/M15/M30"
            )
            self._available = False
        except Exception as e:
            logger.warning(f"[tvdatafeed] Initialization error: {e} — fallback to yfinance")
            self._available = False

        self._initialized = True
        return self._available

    def get_bars(self, asset: str, tf: str, n_bars: int = 300) -> Optional[pd.DataFrame]:
        """
        Download n_bars bars for asset and timeframe from TradingView.

        Args:
            asset:  AgenTrade asset name (e.g. 'EURUSD')
            tf:     timeframe ('M5' | 'M15' | 'M30')
            n_bars: number of bars to download

        Returns:
            Normalized DataFrame or None on error
        """
        if not self._available or self._tv is None:
            return None

        tv_info = TV_SYMBOL_MAP.get(asset)
        if tv_info is None:
            logger.debug(f"[tvdatafeed] {asset} not in TV_SYMBOL_MAP — skip")
            return None

        interval_str = TV_INTERVAL_MAP.get(tf)
        if interval_str is None:
            return None

        symbol, exchange = tv_info

        try:
            from tvDatafeed import Interval
            interval_map = {
                "5":  Interval.in_5_minute,
                "15": Interval.in_15_minute,
                "30": Interval.in_30_minute,
            }
            interval = interval_map.get(interval_str)
            if interval is None:
                return None

            df = self._tv.get_hist(
                symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars,
            )
            df = _normalize_tv(df)
            if df is not None and len(df) > 0:
                logger.debug(f"[tvdatafeed] {asset} {tf}: {len(df)} bars")
            return df

        except Exception as e:
            logger.debug(f"[tvdatafeed] {asset} {tf}: {e}")
            return None


# Module singleton
_tv_manager = TvDatafeedManager()


# ─────────────────────────────────────────────────────────────────────────────
# DataCache v3
# ─────────────────────────────────────────────────────────────────────────────

class DataCache:
    """
    Centralized OHLCV cache for all assets across 7 timeframes.
    v3.1: MT4 DWX ZeroMQ as primary source for ALL assets.
    Fallback: tvdatafeed for M5/M15/M30, yfinance for all timeframes.
    Background refresh every 5 minutes with differentiated refresh groups.
    """

    def __init__(self):
        self._cache:       dict[str, dict[str, pd.DataFrame]] = {}
        self._last_update: dict[str, dict[str, datetime]]     = {}
        self._running     = False
        self._initialized = False
        import threading
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, asset: str, tf: str) -> Optional[pd.DataFrame]:
        """Return DataFrame from cache. None if not available."""
        return self._cache.get(asset, {}).get(tf)

    async def get_or_fetch(self, asset: str, tf: str) -> Optional[pd.DataFrame]:
        """
        On-demand loading.
        Returns from cache if available, otherwise fetches immediately.
        Used when data was not preloaded at startup.
        """
        cached = self.get(asset, tf)
        if cached is not None:
            return cached
        logger.info(f"[DataCache] On-demand loading: {asset} {tf}")
        await self._fetch(asset, tf)
        return self.get(asset, tf)

    def is_fresh(self, asset: str, tf: str) -> bool:
        """True if updated within the per-TF freshness threshold."""
        ts = self._last_update.get(asset, {}).get(tf)
        if ts is None:
            return False
        max_min = TF_FRESHNESS_MIN.get(tf, MAX_AGE_MINUTES)
        return (datetime.now() - ts).total_seconds() < max_min * 60

    async def get_fresh(self, asset: str, tf: str) -> Optional[pd.DataFrame]:
        """
        v3.4 — Freshness-aware read for decision time.
        If fresh (within per-TF threshold) → return cache.
        If stale → force ONE refetch of the single asset/tf, then return.
        Fail-safe: if refetch fails, return last cached data anyway.
        """
        cached = self.get(asset, tf)
        if cached is not None and self.is_fresh(asset, tf):
            return cached
        _ts  = self._last_update.get(asset, {}).get(tf)
        _age = f"{(datetime.now() - _ts).total_seconds()/60:.1f}min" if _ts else "n/a"
        _why = "stale" if cached is not None else "missing"
        logger.info(f"[DataCache] 🔄 get_fresh {asset} {tf}: {_why} (age {_age}) → on-demand refetch")
        try:
            await self._fetch(asset, tf)
        except Exception as e:
            logger.debug(f"[DataCache] get_fresh refetch {asset} {tf}: {e}")
        fresh = self.get(asset, tf)
        return fresh if fresh is not None else cached

    def status(self) -> dict:
        total = sum(
            1 for a in self._cache
            for tf in self._cache[a]
            if self._cache[a][tf] is not None
        )
        return {"total": total, "assets": len(self._cache)}

    def tv_available(self) -> bool:
        """True if tvdatafeed is available and connected."""
        return _tv_manager._available

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def set_mt4_connector(self, connector) -> None:
        """Configure shared MT4 connector (EA1 execution) to avoid ClientID conflicts."""
        if connector is not None:
            mt4_datafeed.set_connector(connector)
            logger.info("[DataCache] MT4 EA1 execution connector configured")
        else:
            logger.warning("[DataCache] Invalid MT4 connector")

    def set_data_connectors(self, forex=None, crypto=None, metals=None) -> None:
        """
        Configure the 3 separate DWX connectors for multi-EA datafeed.
        Call after set_mt4_connector during startup.
        """
        mt4_datafeed.set_data_connectors(forex=forex, crypto=crypto, metals=metals)
        logger.info("[DataCache] Multi-EA data connectors configured")

    async def start(self) -> None:
        """Initial load + start background refresh loop."""
        if self._initialized:
            return
        self._running = True
        logger.info("[DataCache] Starting — initial OHLCV data load...")

        tv_ok = _tv_manager.initialize()
        if tv_ok:
            logger.info("[DataCache] tvdatafeed active — M5/M15/M30 from TradingView")
        else:
            logger.warning("[DataCache] tvdatafeed not available — fallback to yfinance for all TFs")

        await self._refresh_all(startup_mode=True)
        self._initialized = True
        st = self.status()
        logger.info(f"[DataCache] Ready — {st['total']} series across {st['assets']} assets")
        asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False

    # ── Loop ──────────────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        """
        Background refresh loop with differentiated frequency per timeframe.
        Every cycle (5 min) updates only timeframes whose every_n divides the cycle count.
        Counter resets every 72 cycles (6 hours) to avoid overflow.
        """
        cycle = 0
        while self._running:
            await asyncio.sleep(CACHE_REFRESH_INTERVAL)
            cycle += 1

            tfs_this_cycle = []
            for group_name, group in REFRESH_GROUPS.items():
                if cycle % group["every_n"] == 0 or cycle == 1:
                    tfs_this_cycle.extend(group["tfs"])

            if tfs_this_cycle:
                active_groups = [
                    name for name, g in REFRESH_GROUPS.items()
                    if cycle % g["every_n"] == 0 or cycle == 1
                ]
                logger.debug(
                    f"[DataCache] Cycle {cycle} — groups: {active_groups} | "
                    f"TFs: {tfs_this_cycle} | "
                    f"requests: {len(ALL_ASSETS) * len(tfs_this_cycle)}"
                )
                await self._refresh_selected(tfs_this_cycle)

            if cycle >= 72:
                cycle = 0

    async def _refresh_selected(self, timeframes: list) -> None:
        """Update only specified timeframes for all assets."""
        tasks = [self._fetch(a, tf) for a in ALL_ASSETS for tf in timeframes]

        if len(tasks) > 20:
            batch_size = 10
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                await asyncio.gather(*batch, return_exceptions=True)
                if i + batch_size < len(tasks):
                    await asyncio.sleep(0.5)
        else:
            await asyncio.gather(*tasks, return_exceptions=True)

        st = self.status()
        logger.debug(f"[DataCache] Refresh complete — {st['total']} series in cache")

    async def _refresh_all(self, startup_mode=False) -> None:
        """
        Refresh OHLCV data.
        startup_mode=True: load only priority assets/timeframes (avoids MT4 thundering herd)
        startup_mode=False: load all assets/timeframes (normal behavior)
        """
        if startup_mode and not FULL_STARTUP_LOADING:
            assets     = STARTUP_PRIORITY_ASSETS
            timeframes = STARTUP_PRIORITY_TIMEFRAMES
            logger.info(
                f"[DataCache] Startup mode: reduced loading "
                f"{len(assets)} assets × {len(timeframes)} TFs = {len(assets)*len(timeframes)} requests"
            )
        elif startup_mode and FULL_STARTUP_LOADING:
            logger.info(f"[DataCache] Full startup loading: 126 combinations using {FULL_LOAD_STRATEGY}")
            await self._full_load_strategy()
            return
        else:
            assets     = ALL_ASSETS
            timeframes = TIMEFRAMES

        tasks = [self._fetch(a, tf) for a in assets for tf in timeframes]

        if len(tasks) > 20:
            batch_size = 10
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                await asyncio.gather(*batch, return_exceptions=True)
                if i + batch_size < len(tasks):
                    await asyncio.sleep(0.5)
        else:
            await asyncio.gather(*tasks, return_exceptions=True)

        st = self.status()
        logger.debug(f"[DataCache] Updated — {st['total']} series")

    async def _full_load_strategy(self) -> None:
        """Load all 126 combinations using selected strategy."""
        start_time   = time.time()
        loaded_count = 0

        if FULL_LOAD_STRATEGY == "sequential_waves":
            for i, asset in enumerate(ALL_ASSETS):
                logger.info(f"[DataCache] Wave {i+1}/{len(ALL_ASSETS)}: {asset}")
                for tf in TIMEFRAMES:
                    await self._fetch(asset, tf)
                    loaded_count += 1
                    await asyncio.sleep(0.3)
                if i < len(ALL_ASSETS) - 1:
                    await asyncio.sleep(1.0)

        elif FULL_LOAD_STRATEGY == "small_batches":
            all_combinations = [(a, tf) for a in ALL_ASSETS for tf in TIMEFRAMES]
            batch_size       = 3
            for i in range(0, len(all_combinations), batch_size):
                batch = all_combinations[i:i + batch_size]
                tasks = [self._fetch(asset, tf) for asset, tf in batch]
                await asyncio.gather(*tasks, return_exceptions=True)
                loaded_count += len(batch)
                if i + batch_size < len(all_combinations):
                    await asyncio.sleep(0.8)

        elif FULL_LOAD_STRATEGY == "priority_cascade":
            priority_groups = [
                ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"],
                ["XAUUSD", "XAGUSD", "GBPJPY", "EURJPY"],
                ["USDCAD", "AUDUSD", "NZDJPY", "EURGBP"],
                ["EURAUD", "EURCHF", "MGC"],
                ["BTCUSD", "ETHUSD", "MES", "MCL", "NAS100", "6E"],
            ]
            for tier, assets in enumerate(priority_groups, 1):
                logger.info(f"[DataCache] Tier {tier}: {assets}")
                for asset in assets:
                    tasks = [self._fetch(asset, tf) for tf in TIMEFRAMES]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    loaded_count += len(TIMEFRAMES)
                    await asyncio.sleep(0.5)
                if tier < len(priority_groups):
                    await asyncio.sleep(2.0)

        duration = time.time() - start_time
        logger.info(f"[DataCache] Full load completed: {loaded_count}/126 in {duration:.1f}s")

    async def _fetch(self, asset: str, tf: str) -> None:
        """
        Fetch single asset/tf.
        Strategy v3.1:
          1. IB native futures for IB_MICROS (MES/MGC/MCL/6E)
          2. MT4 DWX ZeroMQ (primary for all other assets)
          3. tvdatafeed (fallback for M5/M15/M30)
          4. yfinance (last fallback for all timeframes)
        """
        try:
            # ── Stage 1: IB native futures for micro assets ───────────────────
            if asset in IB_MICROS:
                try:
                    ib_df = await asyncio.to_thread(
                        ib_datafeed.get_historical_bars, asset, tf, 300
                    )
                    if ib_df is not None and len(ib_df) > 0:
                        logger.debug(f"[DataCache] {asset} {tf}: {len(ib_df)} bars from IB (futures)")
                        self._save(asset, tf, ib_df)
                        return
                    logger.debug(f"[DataCache] IB {asset} {tf}: empty — fallback tv/yf futures")
                except Exception as e:
                    logger.debug(f"[DataCache] IB {asset} {tf}: {e} — fallback tv/yf futures")

            # ── Stage 2: MT4 DWX ZeroMQ (primary, excluded for micros) ───────
            try:
                if mt4_datafeed._initialized and asset not in IB_MICROS:
                    from broker.mt4_datafeed import MT4_EXECUTION_BUSY
                    if MT4_EXECUTION_BUSY.is_set():
                        logger.debug(f"[DataCache] MT4 skip {asset} {tf} — execution in progress")
                    else:
                        df = await mt4_datafeed.get_historical_bars(asset, tf, bars=300)
                        if df is not None and len(df) > 0:
                            logger.debug(f"[DataCache] {asset} {tf}: {len(df)} bars from MT4 DWX")
                            self._save(asset, tf, df)
                            return
                logger.debug(f"[DataCache] MT4 DWX fallback → tvdatafeed/yfinance for {asset} {tf}")
            except Exception as e:
                logger.debug(f"[DataCache] MT4 DWX error {asset} {tf}: {e} — fallback")

            # ── Stage 3: tvdatafeed for low timeframes ────────────────────────
            if tf in TV_TIMEFRAMES and _tv_manager._available:
                n_bars = TV_BARS.get(tf, 300)
                df     = await asyncio.to_thread(_tv_manager.get_bars, asset, tf, n_bars)
                if df is not None and len(df) > 0:
                    logger.debug(f"[DataCache] {asset} {tf}: {len(df)} bars from tvdatafeed")
                    self._save(asset, tf, df)
                    return

            # ── Stage 4: yfinance (last fallback for all timeframes) ──────────
            ticker = YFINANCE_MAP.get(asset)
            if not ticker:
                return

            yf_params = YF_PARAMS.get(tf)
            if not yf_params:
                return

            period, interval = yf_params

            def _yf_safe():
                with _YF_DOWNLOAD_LOCK:
                    return yf.download(
                        ticker, period=period, interval=interval,
                        progress=False, auto_adjust=True,
                    )

            df = await asyncio.to_thread(_yf_safe)
            df = _normalize(df)
            if df is None or df.empty:
                return

            logger.debug(f"[DataCache] {asset} {tf}: {len(df)} bars from yfinance (last fallback)")
            self._save(asset, tf, df)

        except Exception as e:
            logger.debug(f"[DataCache] Error {asset} {tf}: {e}")

    def _save(self, asset: str, tf: str, df: pd.DataFrame) -> None:
        """
        Save DataFrame to cache with timestamp.
        v3.2: relative price validation — discards anomalous data (>60% change
        vs existing cache) without relying on absolute ranges.
        """
        with self._lock:
            existing = self._cache.get(asset, {}).get(tf)
            if existing is not None and not existing.empty and "Close" in existing.columns:
                try:
                    old_price = float(existing["Close"].iloc[-1])
                    new_price = float(df["Close"].iloc[-1])
                    if old_price > 0 and new_price > 0:
                        change = abs(new_price - old_price) / old_price
                        if change > 0.60:
                            logger.warning(
                                f"[DataCache] {asset} {tf}: anomalous price discarded "
                                f"({old_price:.4f}→{new_price:.4f}, Δ={change:.0%}) "
                                f"— possible data contamination"
                            )
                            return
                except Exception:
                    pass

            if asset not in self._cache:
                self._cache[asset]       = {}
                self._last_update[asset] = {}
            self._cache[asset][tf]       = df
            self._last_update[asset][tf] = datetime.now()


# Module singleton
data_cache = DataCache()
