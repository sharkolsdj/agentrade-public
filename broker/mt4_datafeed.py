"""
broker/mt4_datafeed.py
MT4 Historical Data Feed via DWX ZeroMQ.
Primary data source for all assets before tvdatafeed and yfinance.

Architecture:
    - Three separate DWX EAs (EA2 forex, EA3 crypto, EA4 metals+indices)
      each connected on their own ZeroMQ ports.
    - MT4 symbol map translates AgenTrade asset names to broker-specific symbols.
    - `get_historical_bars()` sends a HIST_REQUEST to the EA and polls
      _History_DB until the response arrives, with a per-connector busy flag
      to avoid concurrent request collisions.

ZeroMQ ports are read from environment variables. See .env.example.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger
import pandas as pd

import threading as _threading

# threading.Lock is event-loop-agnostic, safe across asyncio contexts in Python 3.12+
_MT4_HIST_SEND_LOCK = _threading.Lock()

# Global flag to block DataCache historical requests during MT4 order execution.
# Prevents HIST_DATA responses from overwriting EXECUTION responses in the shared
# DWX buffer. Set before sending an order, clear after confirmation.
MT4_EXECUTION_BUSY = _threading.Event()


class MT4DataFeed:
    """
    MT4 historical datafeed via DWX ZeroMQ.
    Supports all AgenTrade v3 assets.
    Uses a shared connector to avoid ClientID conflicts.
    """

    # ── Asset routing — multi-EA ──────────────────────────────────────────────
    FOREX_ASSETS = {
        "EURUSD", "GBPUSD", "USDJPY", "GBPJPY", "EURJPY", "USDCHF",
        "AUDUSD", "USDCAD", "EURAUD", "EURGBP", "NZDJPY", "EURCHF",
        "NZDUSD", "AUDJPY", "CHFJPY",
    }
    CRYPTO_ASSETS = {
        "BTCUSD", "ETHUSD",
    }
    METALS_INDICES_ASSETS = {
        "XAUUSD", "XAGUSD", "NAS100",
        "GER40", "US2000", "DJ30", "SP500", "USOUSD",
        # MES/MGC/MCL/6E removed — micro futures use IB native datafeed
    }

    def __init__(self, shared_connector=None):
        self.dwx           = shared_connector   # EA1: execution connector (TradeExecutor)
        self._initialized  = bool(shared_connector)
        self._last_symbol  = None
        self._request_timeout = 2.5
        self._connector_busy: dict = {}
        self._forex_connector  = None
        self._crypto_connector = None
        self._metals_connector = None

    # ── Asset → MT4 broker symbol mapping ────────────────────────────────────
    MT4_SYMBOL_MAP = {
        # Forex Major
        "EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "USDJPY": "USDJPY",
        "GBPJPY": "GBPJPY", "EURJPY": "EURJPY", "USDCHF": "USDCHF",
        "AUDUSD": "AUDUSD", "USDCAD": "USDCAD", "EURAUD": "EURAUD",
        "EURGBP": "EURGBP", "NZDJPY": "NZDJPY", "EURCHF": "EURCHF",
        # Metals
        "XAUUSD": "XAUUSD",  # Gold
        "XAGUSD": "XAGUSD",  # Silver
        # Crypto
        "BTCUSD": "BTCUSD", "ETHUSD": "ETHUSD",
        # Additional forex
        "NZDUSD": "NZDUSD", "AUDJPY": "AUDJPY", "CHFJPY": "CHFJPY",
        # Index CFDs (broker-specific suffixes — replace with your broker's naming)
        "GER40":  "GER40+",
        "US2000": "US2000+",
        "DJ30":   "DJ30+",
        "SP500":  "SP500ft+",  # S&P500 futures CFD
        "USOUSD": "USOUSD",    # WTI spot CFD
        "NAS100": "NAS100+",   # NAS100 Cash
        # MES/MCL/6E/MGC removed — micro futures use IB native datafeed
    }

    # AgenTrade timeframe → MT4 numeric timeframe
    MT4_TIMEFRAME_MAP = {
        "M1":  1,
        "M5":  5,
        "M15": 15,
        "M30": 30,
        "H1":  60,
        "H4":  240,
        "D1":  1440,
        "W1":  10080,
    }

    # ── Initialization ────────────────────────────────────────────────────────

    def set_connector(self, connector) -> bool:
        """
        Set the shared DWX connector from TradeExecutor.
        Avoids ClientID conflicts by using a single connector.
        """
        if connector is not None:
            self.dwx          = connector
            self._initialized = True
            logger.info("[MT4DataFeed] Shared connector configured")
            return True
        else:
            logger.error("[MT4DataFeed] Invalid shared connector")
            return False

    def set_data_connectors(self, forex=None, crypto=None, metals=None) -> None:
        """
        Configure the 3 separate DWX connectors for the datafeed.
        EA2 → forex, EA3 → crypto, EA4 → metals+indices.
        """
        if forex:
            self._forex_connector  = forex
            logger.info("[MT4DataFeed] EA2 Forex connector configured")
        if crypto:
            self._crypto_connector = crypto
            logger.info("[MT4DataFeed] EA3 Crypto connector configured")
        if metals:
            self._metals_connector = metals
            logger.info("[MT4DataFeed] EA4 Metals+Indices connector configured")

    # EA routing: forex 8 / crypto 7 / metals 7
    # The 4 micros (MES/MGC/MCL/6E) are NOT here — datafeed via IB native
    EA_ROUTING = {
        # EA2 (forex)
        "EURUSD": "forex", "GBPUSD": "forex", "USDJPY": "forex", "GBPJPY": "forex",
        "USDCAD": "forex", "EURAUD": "forex", "EURGBP": "forex", "NZDJPY": "forex",
        # EA3 (crypto+misc)
        "BTCUSD": "crypto", "ETHUSD": "crypto",
        "EURCHF": "crypto", "NZDUSD": "crypto", "AUDJPY": "crypto", "CHFJPY": "crypto",
        "USOUSD": "crypto",
        # EA4 (metals+indices)
        "XAUUSD": "metals", "XAGUSD": "metals", "NAS100": "metals",
        "GER40":  "metals", "US2000": "metals", "DJ30":   "metals", "SP500":  "metals",
    }

    def _get_data_connector(self, asset: str):
        """
        Return the data EA connector for the asset.
        Falls back to EA1 execution connector if multi-EA not configured.
        """
        _conn = {
            "forex":  self._forex_connector,
            "crypto": self._crypto_connector,
            "metals": self._metals_connector,
        }
        _k = self.EA_ROUTING.get(asset)
        if _k is not None and _conn.get(_k) is not None:
            return _conn[_k]

        if asset in self.FOREX_ASSETS  and self._forex_connector  is not None:
            return self._forex_connector
        if asset in self.CRYPTO_ASSETS and self._crypto_connector is not None:
            return self._crypto_connector
        if asset in self.METALS_INDICES_ASSETS and self._metals_connector is not None:
            return self._metals_connector
        # Fallback to EA1 (backward compatible if multi-EA not yet configured)
        return self.dwx

    # ── Historical Data ───────────────────────────────────────────────────────

    async def get_historical_bars(
        self,
        asset: str,
        timeframe: str,
        bars: int = 300,
    ) -> Optional[pd.DataFrame]:
        """
        Download historical data from MT4 via DWX ZeroMQ.

        Args:
            asset:     AgenTrade asset name (e.g. 'EURUSD')
            timeframe: timeframe ('M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1')
            bars:      number of bars to download (default 300)

        Returns:
            DataFrame with Open/High/Low/Close/Volume columns or None on failure
        """
        connector = self._get_data_connector(asset)
        if connector is None:
            logger.debug(f"[MT4DataFeed] {asset} unavailable — no connector configured")
            return None
        if not self._initialized and connector is self.dwx:
            logger.debug(f"[MT4DataFeed] {asset} unavailable — connector not configured")
            return None

        mt4_symbol = self.MT4_SYMBOL_MAP.get(asset)
        if not mt4_symbol:
            logger.debug(f"[MT4DataFeed] {asset} not in MT4_SYMBOL_MAP")
            return None

        mt4_timeframe = self.MT4_TIMEFRAME_MAP.get(timeframe)
        if not mt4_timeframe:
            logger.error(f"[MT4DataFeed] Unsupported timeframe {timeframe}")
            return None

        try:
            end_time = datetime.now()
            if timeframe == "M5":
                start_time = end_time - timedelta(days=max(2, bars * 5 // 1440))
            elif timeframe == "M15":
                start_time = end_time - timedelta(days=max(2, bars * 15 // 1440))
            elif timeframe == "M30":
                start_time = end_time - timedelta(days=max(3, bars * 30 // 1440))
            elif timeframe == "H1":
                start_time = end_time - timedelta(days=max(5, bars // 24))
            elif timeframe == "H4":
                start_time = end_time - timedelta(days=max(10, bars * 4 // 24))
            elif timeframe == "D1":
                start_time = end_time - timedelta(days=max(30, bars))
            elif timeframe == "W1":
                start_time = end_time - timedelta(days=max(90, bars * 7))
            else:
                start_time = end_time - timedelta(days=30)

            start_str = start_time.strftime('%Y.%m.%d %H:%M:%S')
            end_str   = end_time.strftime('%Y.%m.%d %H:%M:%S')

            logger.debug(f"[MT4DataFeed] Request {mt4_symbol} {timeframe}({mt4_timeframe}): {start_str} → {end_str}")

            # Per-connector busy flag — avoids concurrent request collisions.
            # dict check+set is atomic in asyncio single-thread between awaits.
            _cid = id(connector)
            while self._connector_busy.get(_cid, False):
                await asyncio.sleep(0.01)
            self._connector_busy[_cid] = True
            try:
                # 1. Clear _History_DB before sending (avoids stale data)
                history_key = f"{mt4_symbol}_{timeframe}"
                if hasattr(connector, '_History_DB') and history_key in connector._History_DB:
                    del connector._History_DB[history_key]

                # 2. Send request
                with _MT4_HIST_SEND_LOCK:
                    connector._DWX_MTX_SEND_HIST_REQUEST_(
                        _symbol=mt4_symbol,
                        _timeframe=mt4_timeframe,
                        _start=start_str,
                        _end=end_str,
                    )
                await asyncio.sleep(0.05)  # 50ms DWX EA buffer

                # 3. Wait for response — connector busy, no concurrent requests
                symbol_key = f"{mt4_symbol}_{timeframe}"
                start_wait = time.time()
                timeout    = self._request_timeout

                while (time.time() - start_wait) < timeout:
                    await asyncio.sleep(0.1)
                    if (hasattr(connector, '_History_DB') and
                        symbol_key in connector._History_DB and
                        connector._History_DB[symbol_key]):

                        hist_data = connector._History_DB[symbol_key]
                        df = self._convert_to_dataframe(hist_data, asset, timeframe)

                        if df is not None and len(df) > 0:
                            logger.success(f"[MT4DataFeed] {asset} {timeframe}: {len(df)} bars from MT4")
                            return df
                        break

                logger.warning(f"[MT4DataFeed] {asset} {timeframe}: timeout or no data received")
                return None
            finally:
                self._connector_busy[_cid] = False

        except Exception as e:
            logger.error(f"[MT4DataFeed] Error {asset} {timeframe}: {e}")
            return None

    def _convert_to_dataframe(
        self, hist_data: list, asset: str, timeframe: str
    ) -> Optional[pd.DataFrame]:
        """
        Convert DWX data to a pandas DataFrame compatible with yfinance format.

        DWX format:
            [{'time': EPOCH, 'open': O, 'high': H, 'low': L,
              'close': C, 'tick_volume': V, 'spread': S, 'real_volume': RV}]
        """
        try:
            if not hist_data or len(hist_data) == 0:
                return None

            df_data = []
            for bar in hist_data:
                try:
                    if 'time' not in bar:
                        continue
                    df_data.append({
                        'Open':  float(bar.get('open',  0)),
                        'High':  float(bar.get('high',  0)),
                        'Low':   float(bar.get('low',   0)),
                        'Close': float(bar.get('close', 0)),
                        # Use real_volume if > 0 (exchange), else tick_volume (forex/CFD)
                        'Volume': int((bar.get('real_volume') or bar.get('tick_volume') or 0)),
                    })
                except (ValueError, KeyError, TypeError) as e:
                    logger.debug(f"[MT4DataFeed] Skip malformed bar: {e}")
                    continue

            if not df_data:
                return None

            df = pd.DataFrame(df_data)

            timestamps = []
            for bar in hist_data[:len(df_data)]:
                try:
                    timestamps.append(pd.to_datetime(bar['time'], unit='s'))
                except Exception:
                    timestamps.append(pd.Timestamp.now())

            df.index = pd.DatetimeIndex(timestamps)
            df = df[~df.index.duplicated(keep='last')].sort_index()
            df = df[(df['Close'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Open'] > 0)]

            if len(df) > 0:
                logger.debug(f"[MT4DataFeed] DataFrame created: {len(df)} valid bars")
                return df
            else:
                logger.warning("[MT4DataFeed] No valid bars after filtering")
                return None

        except Exception as e:
            logger.error(f"[MT4DataFeed] DataFrame conversion error: {e}")
            return None

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self):
        """Close DWX connection."""
        if self.dwx and hasattr(self.dwx, '_ACTIVE'):
            self.dwx._ACTIVE  = False
            logger.info("[MT4DataFeed] DWX connection closed")
        self._initialized = False


# Module singleton — configured dynamically by scheduler
mt4_datafeed = MT4DataFeed(shared_connector=None)
