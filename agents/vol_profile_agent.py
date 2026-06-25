"""
agents/vol_profile_agent.py  —  v3.1
VolProfileAgent — Deterministic Volume Profile agent for AgenTrade v3.

Cost: $0 — no LLM calls, pure pandas/numpy computation.

v3 features (from Wyckoff 2.0 + volumetric analysis):
  - Naked VPOC: prior session VPOCs not yet revisited (priority targets)
  - VPOC Migration: VPOC shift in trade direction → continuation signal
  - SOS/SOW Bar: trigger candle (wide range + extreme close + high volume)
  - Rule of 80%: price re-enters VA after leaving → target opposite extreme
  - b/P Distribution: profile shape (accumulation vs distribution)
  - Real weekly VWAP: no longer fixed 40 bars
  - Value Area Gate v3: price inside VA → setup INVALID (no trade)
  - LVN/HVN in TP path: LVN = clear runway, HVN = resistance → reduces TP
  - Session weight: London/NY levels weighted higher than Asian
  - DVPOC: developing POC — migration monitoring
  - Adaptive row size per asset (more granular for futures)
  - Value Area 68% (first standard deviation, Wyckoff 2.0)

Role in v3: Mandatory Layer 2 gate.
  - Price inside Value Area → setup_quality = INVALID (sole hard block)
  - LVN ahead → confirms clear runway (+2 quality)
  - HVN ahead → reduces TP to HVN level
  - POC aligned with OB/FVG (±0.5%) → +2 quality
  - VWAP in direction → +1 quality (institutional confirmation)
"""

import asyncio
import threading
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger


try:
    from agents.base_agent import BaseAgent, AgentResult
except ImportError:
    from base_agent import BaseAgent, AgentResult

try:
    from utils.score_converter import to_score_0_100
except ImportError:
    def to_score_0_100(raw: float, direction: str, raw_max: float) -> float:
        clamped = max(-raw_max, min(raw_max, raw))
        if direction == "BUY":
            return round(50.0 + (clamped / raw_max) * 50.0, 1)
        return round(50.0 + (-clamped / raw_max) * 50.0, 1)

# Global shared lock — imported from utils.yfinance_lock.
# Shared with technical_agent and entry_layer to serialize all yfinance
# downloads across modules (fixes cross-module data mixing).
try:
    from utils.yfinance_lock import YF_DOWNLOAD_LOCK as _YF_DOWNLOAD_LOCK
except ImportError:
    _YF_DOWNLOAD_LOCK = threading.Lock()


# ── yfinance tickers ──────────────────────────────────────────────────────────

YFINANCE_TICKERS = {
    "EURUSD": "EURUSD=X",  "GBPUSD": "GBPUSD=X",  "USDJPY": "USDJPY=X",
    "GBPJPY": "GBPJPY=X",  "XAUUSD": "GC=F",       "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",   "EURAUD": "EURAUD=X",    "USDCAD": "USDCAD=X",
    "EURGBP": "EURGBP=X",  "NZDJPY": "NZDJPY=X",   "EURCHF": "EURCHF=X",
    "MGC":    "GC=F",      "MES":    "ES=F",
    "XAGUSD": "SI=F",      "6E":     "6E=F",         "MCL":    "CL=F",
    "NAS100": "NQ=F",
    "NZDUSD": "NZDUSD=X",  "AUDJPY": "AUDJPY=X",  "CHFJPY": "CHFJPY=X",
    "GER40":  "^GDAXI",    "US2000": "^RUT",      "DJ30":   "^DJI",
    "USOUSD": "CL=F",
    "SP500":  "ES=F",
}

# Alternative tickers for interval="1d" — indices/spot instead of expiring futures.
# NQ=F and 6E=F do not have reliable D1 history on free yfinance (expire every 3 months).
# VP D1 fix: use stable proxies with unlimited daily history.
YFINANCE_D1_TICKERS: dict[str, str] = {
    "NAS100": "^NDX",      # Nasdaq 100 Index  — stable D1 on yfinance
    "MES":    "^GSPC",     # S&P 500 Index     — stable D1 on yfinance
    "SP500":  "^GSPC",     # S&P 500 Index     — stable D1 (CFD S&P)
    "MGC":    "GC=F",      # Gold continuous   — already works for D1
    "6E":     "EURUSD=X",  # EUR/USD spot      — stable D1 on yfinance
    "MCL":    "CL=F",      # Crude Oil         — already works for D1
}

# Adaptive row size per asset (in price units)
ROW_SIZE_H1 = {
    "XAUUSD": 1.0,    "MGC":    1.0,     # Gold: $1/oz on H1
    "XAGUSD": 0.05,                       # Silver: 5 cents/oz
    "MCL":    0.10,                       # Crude Oil: 10 cents/barrel
    "MES":    2.0,                        # E-mini S&P: 2 points
    "NAS100": 10.0,                       # Nasdaq: 10 points
    "6E":     0.0002,                     # Euro FX: 2 pips
    "EURUSD": 0.0005, "GBPUSD": 0.0005,  # Forex: 5 pips
    "USDJPY": 0.05,   "GBPJPY": 0.05,
    "USDCAD": 0.0005, "EURAUD": 0.0005,
    "EURGBP": 0.0003, "NZDJPY": 0.05,
    "EURCHF": 0.0003,
    "BTCUSD": 50.0,   "ETHUSD": 2.0,
}

# Number of bars for VP (depends on timeframe)
VP_BARS = 110  # ~5 days H1

# London/NY sessions (UTC) — for session-weighted levels
LONDON_OPEN_UTC  = (8,  12)   # 08:00-12:00 UTC
NY_OPEN_UTC      = (13, 20)   # 13:00-20:00 UTC
ASIA_UTC         = (1,  8)    # 01:00-08:00 UTC

VOL_RAW_MAX = 10.0  # v3: extended range for more components


def _fmt(price: float, asset: str) -> str:
    """Format a price with the correct decimal places for the asset."""
    if asset in ("XAUUSD", "MGC", "XAGUSD", "MCL"):
        return f"{price:.2f}"
    elif asset in ("MES", "NAS100"):
        return f"{price:.2f}"
    elif asset in ("BTCUSD",):
        return f"{price:.1f}"
    elif asset in ("ETHUSD",):
        return f"{price:.2f}"
    elif "JPY" in asset:
        return f"{price:.3f}"
    else:
        return f"{price:.5f}"


# ═══════════════════════════════════════════════════════════════════════════════
# VOLUME ANALYZER v3
# ═══════════════════════════════════════════════════════════════════════════════

class VolumeAnalyzer:
    """
    Computes full Volume Profile from OHLCV data.
    v3: Naked VPOC, VPOC Migration, SOS/SOW bar, Rule of 80%,
        b/P Distribution, weekly VWAP, session-weighted levels.
    """

    # Asset groups for waterfall routing
    MT4_ASSETS = {
        "EURUSD", "GBPUSD", "USDJPY", "GBPJPY", "USDCAD",
        "EURAUD", "EURGBP", "NZDJPY", "EURCHF",
        "XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD",
        "NZDUSD", "AUDJPY", "CHFJPY",
        "USOUSD",
        "SP500",
        "GER40", "US2000", "DJ30",
    }
    IB_ASSETS  = {"MGC", "MES", "MCL", "6E", "NAS100"}
    IB_MICROS  = {"MGC", "MES", "MCL", "6E"}

    def _convert_to_bars(self, period: str, interval: str) -> int:
        """Convert yfinance period/interval to number of bars for broker APIs."""
        period_map   = {"1d": 1, "5d": 5, "10d": 10, "30d": 30, "60d": 60, "1y": 365}
        interval_map = {"5m": 1/12, "15m": 0.25, "30m": 0.5, "1h": 1, "4h": 4, "1d": 24}
        days          = period_map.get(period, 10)
        hours_per_bar = interval_map.get(interval, 1)
        return int(days * 24 / hours_per_bar)

    def _yf_to_broker_timeframe(self, interval: str) -> str:
        """Convert yfinance interval to broker timeframe string."""
        mapping = {"5m": "M5", "15m": "M15", "30m": "M30", "1h": "H1", "4h": "H4", "1d": "D1"}
        return mapping.get(interval, "H1")

    def fetch_ohlcv(self, asset: str, period: str = "10d",
                    interval: str = "1h") -> Optional[pd.DataFrame]:
        """Download OHLCV data with waterfall: MT4/IB → yfinance fallback."""
        try:
            try:
                asyncio.get_running_loop()
                return self._fetch_ohlcv_sync(asset, period, interval)
            except RuntimeError:
                return asyncio.run(self._fetch_ohlcv_async(asset, period, interval))
        except Exception as e:
            logger.error(f"[VolProfile] {asset} waterfall error: {e}")
            return self._fetch_ohlcv_sync(asset, period, interval)

    def _fetch_ohlcv_sync(self, asset: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Sync fallback implementation — goes directly to yfinance."""
        ticker = YFINANCE_TICKERS.get(asset)
        if not ticker:
            logger.warning(f"[VolProfile] No ticker found for {asset}")
            return None

        try:
            logger.debug(f"[VolProfile] {asset} {interval}: sync fallback to yfinance")
            with _YF_DOWNLOAD_LOCK:
                df = yf.download(ticker, period=period, interval=interval,
                                 progress=False, auto_adjust=True)

            if df.empty:
                logger.error(f"[VolProfile] {asset}: no data from yfinance")
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            else:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

            df = df.loc[:, ~df.columns.duplicated()]
            df.columns = [c.lower() for c in df.columns]
            available = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            df = df[available].dropna()
            df.columns = [c.capitalize() if c != "volume" else "Volume" for c in df.columns]

            for req in ["Open", "High", "Low", "Close", "Volume"]:
                if req not in df.columns:
                    df[req] = 0.0

            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            logger.success(f"[VolProfile] {asset}: {len(df)} bars from yfinance (sync)")
            return df

        except Exception as e:
            logger.error(f"[VolProfile] {asset}: sync yfinance error: {e}")
            return None

    async def _fetch_ohlcv_async(self, asset: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Async waterfall implementation."""

        try:
            from broker.mt4_datafeed import mt4_datafeed
        except Exception:
            mt4_datafeed = None

        try:
            from broker.ib_connector import ib_connector
        except Exception:
            ib_connector = None

        try:
            from broker.ib_datafeed import ib_datafeed
        except Exception:
            ib_datafeed = None

        timeframe = self._yf_to_broker_timeframe(interval)
        bars      = self._convert_to_bars(period, interval)

        # === WATERFALL STAGE 1: MT4 for MT4_ASSETS ===
        if asset in self.MT4_ASSETS:
            try:
                logger.debug(f"[VolProfile] {asset} {interval}: trying MT4")
                if mt4_datafeed is not None and mt4_datafeed._initialized:
                    df = await mt4_datafeed.get_historical_bars(asset, timeframe, bars)
                    if df is not None and len(df) > 0:
                        logger.success(f"[VolProfile] {asset}: {len(df)} bars from MT4")
                        return self._normalize_broker_df(df)
                logger.warning(f"[VolProfile] {asset}: MT4 failed, trying yfinance")
            except Exception as e:
                logger.warning(f"[VolProfile] {asset}: MT4 error {e}, trying yfinance")

        # === WATERFALL STAGE 2: IB for IB_ASSETS ===
        elif asset in self.IB_ASSETS:
            if asset in self.IB_MICROS and ib_datafeed is not None:
                try:
                    logger.debug(f"[VolProfile] {asset} {interval}: trying IB datafeed (futures)")
                    df = await asyncio.to_thread(
                        ib_datafeed.get_historical_bars, asset, timeframe, bars
                    )
                    if df is not None and len(df) > 0:
                        logger.success(f"[VolProfile] {asset}: {len(df)} bars from IB (futures)")
                        return self._normalize_broker_df(df)
                    logger.warning(f"[VolProfile] {asset}: IB datafeed empty, trying yfinance")
                except Exception as e:
                    logger.warning(f"[VolProfile] {asset}: IB datafeed error {e}, trying yfinance")
            else:
                try:
                    logger.debug(f"[VolProfile] {asset} {interval}: trying IB")
                    if ib_connector is not None and ib_connector.connected:
                        df = await ib_connector.get_historical_bars(asset, timeframe, bars)
                        if df is not None and len(df) > 0:
                            logger.success(f"[VolProfile] {asset}: {len(df)} bars from IB")
                            return self._normalize_broker_df(df)
                    logger.warning(f"[VolProfile] {asset}: IB failed, trying yfinance")
                except Exception as e:
                    logger.warning(f"[VolProfile] {asset}: IB error {e}, trying yfinance")

        # === WATERFALL STAGE 3: DataCache ===
        try:
            from broker.data_cache import data_cache as _dc
            df = await _dc.get_or_fetch(asset, timeframe)
            if df is not None and not df.empty:
                logger.success(f"[VolProfile] {asset}: {len(df)} bars from DataCache ({timeframe})")
                return self._normalize_broker_df(df)
        except Exception as e:
            logger.debug(f"[VolProfile] {asset}: DataCache fallback error {e}")

        # === WATERFALL STAGE 4: yfinance (last resort) ===
        if interval == "1d" and asset in YFINANCE_D1_TICKERS:
            ticker = YFINANCE_D1_TICKERS[asset]
        else:
            ticker = YFINANCE_TICKERS.get(asset)
        if not ticker:
            logger.warning(f"[VolProfile] No ticker found for {asset}")
            return None

        try:
            logger.debug(f"[VolProfile] {asset} {interval}: trying yfinance")
            with _YF_DOWNLOAD_LOCK:
                df = await asyncio.to_thread(
                    yf.download, ticker, period=period, interval=interval,
                    progress=False, auto_adjust=True
                )

            if df.empty:
                logger.error(f"[VolProfile] {asset}: all sources failed")
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            else:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

            df = df.loc[:, ~df.columns.duplicated()]
            df.columns = [c.lower() for c in df.columns]
            available = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            df = df[available].dropna()
            df.columns = [c.capitalize() if c != "volume" else "Volume" for c in df.columns]

            for req in ["Open", "High", "Low", "Close", "Volume"]:
                if req not in df.columns:
                    df[req] = 0.0

            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            logger.success(f"[VolProfile] {asset}: {len(df)} bars from yfinance")
            return df

        except Exception as e:
            logger.error(f"[VolProfile] {asset}: all sources failed: {e}")
            return None

    def _normalize_broker_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize broker DataFrame for compatibility with existing code."""
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    def _get_row_size(self, asset: str, price_range: float) -> float:
        """
        Adaptive row size per asset.
        The profile should show 20-60 rows.
        Too few → reduce row_size. Too many → increase.
        """
        base = ROW_SIZE_H1.get(asset, price_range / 40)
        return max(base, price_range / 60)  # never below price_range/60

    def _bar_hour_utc(self, bar_index) -> int:
        """Extract UTC hour from bar index (datetime or int)."""
        try:
            if hasattr(bar_index, 'hour'):
                return bar_index.hour
        except Exception:
            pass
        return 12  # default: neutral hour

    def _volume_weight(self, hour_utc: int) -> float:
        """
        Session-based volume weight.
        London/NY generate 70-80% of GC volume.
        Levels formed in London/NY are weighted 3× Asian levels.
        """
        if LONDON_OPEN_UTC[0] <= hour_utc < LONDON_OPEN_UTC[1]:
            return 3.0   # London open — maximum reliability
        elif NY_OPEN_UTC[0] <= hour_utc < NY_OPEN_UTC[1]:
            return 3.0   # NY session — maximum reliability
        elif ASIA_UTC[0] <= hour_utc < ASIA_UTC[1]:
            return 1.0   # Asia — normal weight
        else:
            return 1.5   # Overlap and other hours

    def calculate_volume_profile(self, df: pd.DataFrame,
                                 asset: str = "EURUSD",
                                 use_weighted: bool = True) -> dict:
        """
        Calculate Volume Profile with session weighting and adaptive row size.
        Value Area: 68% (first standard deviation, Wyckoff 2.0).

        Returns: POC, VAH, VAL, HVN, LVN, b/P distribution, DVPOC
        """
        if df is None or len(df) < 5:
            return {}

        prices  = df["Close"].values
        volumes = df["Volume"].values.astype(float)

        # Fallback for OTC forex (volume=0)
        if volumes.sum() == 0 or np.mean(volumes) < 1:
            ranges    = (df["High"] - df["Low"]).values
            avg_range = np.mean(ranges)
            volumes   = ranges / (avg_range + 1e-10) * 10000

        price_min = float(np.min(df["Low"].values))
        price_max = float(np.max(df["High"].values))
        if price_max <= price_min:
            return {}

        row_size   = self._get_row_size(asset, price_max - price_min)
        n_bins     = max(20, min(80, int((price_max - price_min) / row_size)))
        bins       = np.linspace(price_min, price_max, n_bins + 1)
        bin_volume = np.zeros(n_bins)

        for i in range(len(df)):
            lo  = float(df["Low"].iloc[i])
            hi  = float(df["High"].iloc[i])
            vol = float(volumes[i])
            if hi <= lo or vol <= 0:
                continue

            hour_utc = self._bar_hour_utc(df.index[i])
            weight   = self._volume_weight(hour_utc) if use_weighted else 1.0

            for b in range(n_bins):
                b_lo    = bins[b]
                b_hi    = bins[b + 1]
                overlap = max(0, min(hi, b_hi) - max(lo, b_lo))
                if overlap > 0:
                    bin_volume[b] += vol * weight * (overlap / (hi - lo))

        poc_idx   = int(np.argmax(bin_volume))
        poc_price = float((bins[poc_idx] + bins[poc_idx + 1]) / 2)
        total_vol = float(bin_volume.sum())

        # Value Area 68% (Wyckoff 2.0 — first standard deviation)
        va_target = total_vol * 0.68
        va_vol    = bin_volume[poc_idx]
        lo_idx    = poc_idx
        hi_idx    = poc_idx

        while va_vol < va_target and (lo_idx > 0 or hi_idx < n_bins - 1):
            add_lo = bin_volume[lo_idx - 1] if lo_idx > 0 else 0
            add_hi = bin_volume[hi_idx + 1] if hi_idx < n_bins - 1 else 0
            if add_lo >= add_hi and lo_idx > 0:
                lo_idx -= 1
                va_vol += add_lo
            elif hi_idx < n_bins - 1:
                hi_idx += 1
                va_vol += add_hi
            else:
                break

        vah = float((bins[hi_idx] + bins[hi_idx + 1]) / 2)
        val = float((bins[lo_idx] + bins[lo_idx + 1]) / 2)

        # HVN and LVN — significant peaks and valleys in the profile
        vol_mean   = float(np.mean(bin_volume[bin_volume > 0]))
        vol_std    = float(np.std(bin_volume[bin_volume > 0]))
        hvn_levels = []
        lvn_levels = []
        for b in range(n_bins):
            level = float((bins[b] + bins[b + 1]) / 2)
            if bin_volume[b] > vol_mean + 0.8 * vol_std:
                hvn_levels.append(level)
            elif bin_volume[b] < vol_mean - 0.5 * vol_std and bin_volume[b] > 0:
                lvn_levels.append(level)

        # b/P Distribution — profile shape
        # b: volume concentrated in lower half → accumulation
        # P: volume concentrated in upper half → distribution
        vol_lower = float(np.sum(bin_volume[:n_bins // 2]))
        vol_upper = float(np.sum(bin_volume[n_bins // 2:]))
        if vol_lower + vol_upper > 0:
            lower_pct = vol_lower / (vol_lower + vol_upper)
        else:
            lower_pct = 0.5
        if lower_pct >= 0.60:
            distribution_shape = "b"          # accumulation — bullish
        elif lower_pct <= 0.40:
            distribution_shape = "P"          # distribution — bearish
        else:
            distribution_shape = "balanced"

        # DVPOC — developing POC (last 12 bars of current session)
        dvpoc = poc_price  # default = full POC
        if len(df) >= 24:
            half_bins = np.zeros(n_bins)
            for i in range(len(df) - 12, len(df)):
                lo  = float(df["Low"].iloc[i])
                hi  = float(df["High"].iloc[i])
                vol = float(volumes[i])
                if hi <= lo or vol <= 0:
                    continue
                for b in range(n_bins):
                    b_lo    = bins[b]
                    b_hi    = bins[b + 1]
                    overlap = max(0, min(hi, b_hi) - max(lo, b_lo))
                    if overlap > 0:
                        half_bins[b] += vol * (overlap / (hi - lo))
            if half_bins.sum() > 0:
                dvpoc_idx = int(np.argmax(half_bins))
                dvpoc     = float((bins[dvpoc_idx] + bins[dvpoc_idx + 1]) / 2)

        # VPOC migration: if DVPOC has shifted relative to full POC
        vpoc_migration = "none"
        migration_size = dvpoc - poc_price
        if abs(migration_size) > row_size * 0.5:
            vpoc_migration = "up" if migration_size > 0 else "down"

        return {
            "poc":                round(poc_price, 6),
            "vah":                round(vah, 6),
            "val":                round(val, 6),
            "hvn_levels":         [round(l, 6) for l in hvn_levels[:6]],
            "lvn_levels":         [round(l, 6) for l in lvn_levels[:6]],
            "total_volume":       round(total_vol, 0),
            "price_range":        [round(price_min, 6), round(price_max, 6)],
            "distribution_shape": distribution_shape,  # b | P | balanced
            "lower_vol_pct":      round(lower_pct, 3),
            "dvpoc":              round(dvpoc, 6),
            "vpoc_migration":     vpoc_migration,       # up | down | none
            "n_bins":             n_bins,
            "row_size":           round(row_size, 6),
        }

    def calculate_session_profiles(self, df: pd.DataFrame,
                                   asset: str = "EURUSD",
                                   n_sessions: int = 5) -> list:
        """
        Calculate VP for the last N separate sessions.
        Each session ≈ 24 H1 bars.
        Used to identify Naked VPOCs.

        Returns: list of dicts {session_start, poc, vah, val, visited}
        """
        if df is None or len(df) < 24:
            return []

        sessions         = []
        bars_per_session = 24  # H1
        current_price    = float(df["Close"].iloc[-1])
        n_bars           = len(df)

        for s in range(n_sessions):
            end_idx   = n_bars - s * bars_per_session
            start_idx = max(0, end_idx - bars_per_session)
            if end_idx <= start_idx:
                break
            session_df = df.iloc[start_idx:end_idx]
            if len(session_df) < 5:
                continue

            prof = self.calculate_volume_profile(session_df, asset, use_weighted=False)
            if not prof:
                continue

            session_poc = prof["poc"]
            if end_idx < n_bars:
                subsequent_df = df.iloc[end_idx:]
                price_high    = float(subsequent_df["High"].max())
                price_low     = float(subsequent_df["Low"].min())
                visited       = price_low <= session_poc <= price_high
            else:
                visited = False

            sessions.append({
                "session_idx":   s,
                "session_start": str(df.index[start_idx]),
                "poc":           session_poc,
                "vah":           prof["vah"],
                "val":           prof["val"],
                "visited":       visited,
            })

        return sessions

    def get_naked_vpoc(self, sessions: list, current_price: float,
                       direction: str) -> list:
        """
        Identify Naked VPOCs: prior session POCs not yet revisited.
        High probability of being visited — use as TP targets.
        Priority: nearest to current price in the trade direction.

        Returns: list ordered by distance from price
        """
        naked = []
        for s in sessions[1:]:  # exclude current session
            if not s.get("visited", True):
                poc  = s["poc"]
                dist = abs(current_price - poc)
                if direction == "BUY" and poc > current_price:
                    naked.append({"poc": poc, "distance": dist, "session": s["session_start"]})
                elif direction == "SELL" and poc < current_price:
                    naked.append({"poc": poc, "distance": dist, "session": s["session_start"]})
                elif direction == "":
                    naked.append({"poc": poc, "distance": dist, "session": s["session_start"]})

        return sorted(naked, key=lambda x: x["distance"])[:3]

    def calculate_vwap_weekly(self, df: pd.DataFrame) -> Optional[float]:
        """
        Real weekly VWAP.
        The most respected institutional level on Gold.
        Above = structural long bias. Below = structural short bias.
        Computed over the last 5×24=120 H1 bars (1 trading week).
        """
        if df is None or len(df) < 5:
            return None
        weekly    = df.tail(min(120, len(df)))
        typ_price = (weekly["High"] + weekly["Low"] + weekly["Close"]) / 3
        volumes   = weekly["Volume"].values.astype(float)
        if volumes.sum() == 0 or np.mean(volumes) < 1:
            return float(typ_price.iloc[-1])
        return round(float((typ_price * volumes).sum() / volumes.sum()), 6)

    def calculate_vwap_daily(self, df: pd.DataFrame) -> Optional[float]:
        """Session VWAP (last 24 H1 bars = 1 day)."""
        if df is None or len(df) < 5:
            return None
        daily     = df.tail(min(24, len(df)))
        typ_price = (daily["High"] + daily["Low"] + daily["Close"]) / 3
        volumes   = daily["Volume"].values.astype(float)
        if volumes.sum() == 0 or np.mean(volumes) < 1:
            return float(typ_price.iloc[-1])
        return round(float((typ_price * volumes).sum() / volumes.sum()), 6)

    def calculate_delta_volume(self, df: pd.DataFrame) -> dict:
        """
        Approximated Delta Volume with divergence detection.
        buy_vol  = vol × (Close-Low)  / (High-Low)
        sell_vol = vol × (High-Close) / (High-Low)
        delta    = buy_vol - sell_vol
        """
        if df is None or len(df) < 10:
            return {}

        closes  = df["Close"].values.astype(float)
        highs   = df["High"].values.astype(float)
        lows    = df["Low"].values.astype(float)
        volumes = df["Volume"].values.astype(float)

        if volumes.sum() == 0 or np.mean(volumes) < 1:
            avg_range = np.mean(highs - lows)
            volumes   = (highs - lows) / (avg_range + 1e-10) * 10000

        ranges   = np.where(highs - lows == 0, 1e-10, highs - lows)
        buy_vol  = volumes * (closes - lows)  / ranges
        sell_vol = volumes * (highs  - closes) / ranges
        delta    = buy_vol - sell_vol

        cum_delta_20 = float(np.sum(delta[-20:]))
        cum_delta_50 = float(np.sum(delta[-50:])) if len(delta) >= 50 else float(np.sum(delta))
        price_now    = float(closes[-1])
        price_20ago  = float(closes[-20]) if len(closes) >= 20 else float(closes[0])
        price_change = price_now - price_20ago

        divergence = "none"
        if price_change > 0 and cum_delta_20 < 0:
            divergence = "bearish_divergence"
        elif price_change < 0 and cum_delta_20 > 0:
            divergence = "bullish_divergence"

        recent_buy     = float(np.sum(buy_vol[-10:]))
        recent_sell    = float(np.sum(sell_vol[-10:]))
        buy_sell_ratio = recent_buy / recent_sell if recent_sell > 0 else 1.0

        return {
            "delta_last_bar":      round(float(delta[-1]), 0),
            "cumulative_delta_20": round(cum_delta_20, 0),
            "cumulative_delta_50": round(cum_delta_50, 0),
            "divergence":          divergence,
            "buy_sell_ratio_10":   round(buy_sell_ratio, 3),
            "price_change_20":     round(price_change, 6),
        }

    def calculate_relative_volume(self, df: pd.DataFrame) -> float:
        """Relative volume: current bar vs 20-bar average."""
        if df is None or len(df) < 21:
            return 1.0
        volumes     = df["Volume"].values.astype(float)
        current_vol = volumes[-1]
        avg_vol     = np.mean(volumes[-21:-1])
        if avg_vol == 0:
            return 1.0
        return round(current_vol / avg_vol, 2)

    def detect_sos_sow_bar(self, df: pd.DataFrame, direction: str) -> dict:
        """
        Detect SOS bar (long) or SOW bar (short) — entry trigger.
        Wide range candle, close at the correct extreme, high volume.

        SOS bar: wide bullish range + close in upper third + volume > avg
        SOW bar: wide bearish range + close in lower third + volume > avg
        """
        if df is None or len(df) < 21:
            return {"detected": False, "type": "none"}

        last         = df.iloc[-1]
        candle_range = float(last["High"]) - float(last["Low"])
        close        = float(last["Close"])
        open_        = float(last["Open"])
        high         = float(last["High"])
        low          = float(last["Low"])

        avg_range = float((df["High"] - df["Low"]).tail(20).mean())
        avg_vol   = float(df["Volume"].tail(20).mean())
        curr_vol  = float(last["Volume"])
        if avg_vol == 0:
            avg_vol = 1

        wide_range = candle_range > avg_range * 1.3
        high_vol   = curr_vol > avg_vol * 1.2

        if direction == "BUY":
            upper_third = low + candle_range * 0.67
            is_sos = (close > open_ and close >= upper_third and wide_range and high_vol)
            return {
                "detected":    is_sos,
                "type":        "SOS" if is_sos else "none",
                "close_pct":   round((close - low) / candle_range * 100, 1) if candle_range > 0 else 50,
                "vol_ratio":   round(curr_vol / avg_vol, 2),
                "range_ratio": round(candle_range / avg_range, 2) if avg_range > 0 else 1.0,
            }
        else:
            lower_third = high - candle_range * 0.67
            is_sow = (close < open_ and close <= lower_third and wide_range and high_vol)
            return {
                "detected":    is_sow,
                "type":        "SOW" if is_sow else "none",
                "close_pct":   round((close - low) / candle_range * 100, 1) if candle_range > 0 else 50,
                "vol_ratio":   round(curr_vol / avg_vol, 2),
                "range_ratio": round(candle_range / avg_range, 2) if avg_range > 0 else 1.0,
            }

    def check_rule_80(self, df: pd.DataFrame, profile: dict) -> dict:
        """
        Rule of 80%: if price re-enters the VA after leaving it,
        there is ~80% probability of reaching the opposite extreme.
        One of the most reliable setups on Gold.

        Analyzes the last 5 bars to detect the re-entry.
        """
        if not profile or df is None or len(df) < 5:
            return {"active": False, "direction": "none"}

        vah = profile.get("vah", 0)
        val = profile.get("val", 0)
        if vah == val:
            return {"active": False, "direction": "none"}

        recent        = df.tail(5)
        closes        = recent["Close"].values.astype(float)
        highs         = recent["High"].values.astype(float)
        lows          = recent["Low"].values.astype(float)
        current_price = closes[-1]
        in_va_now     = val <= current_price <= vah

        was_above_vah = any(lows[i] > vah   for i in range(max(0, len(lows)-4),   len(lows)-1))
        was_below_val = any(highs[i] < val  for i in range(max(0, len(highs)-4), len(highs)-1))

        if in_va_now and was_above_vah:
            return {"active": True, "direction": "SELL",
                    "target": val, "reason": "Rule 80%: re-entry from above VAH → target VAL"}
        elif in_va_now and was_below_val:
            return {"active": True, "direction": "BUY",
                    "target": vah, "reason": "Rule 80%: re-entry from below VAL → target VAH"}

        return {"active": False, "direction": "none"}

    def check_lvn_hvn_path(self, current_price: float, direction: str,
                            profile: dict, sl_pips: float = 0,
                            pip_size: float = 0.0001) -> dict:
        """
        Analyze LVN/HVN in the path toward TP.
          LVN ahead → clear runway → confirms move (+2 quality)
          HVN ahead → resistance → reduces TP to HVN level

        Returns: path_clear, blocking_hvn, nearest_lvn, tp_reduction_pct
        """
        if not profile:
            return {"path_clear": True, "blocking_hvn": None,
                    "nearest_lvn": None, "tp_reduction_pct": 0}

        hvn_levels = profile.get("hvn_levels", [])
        lvn_levels = profile.get("lvn_levels", [])

        if direction == "BUY":
            hvn_above    = [h for h in hvn_levels if h > current_price * 1.001]
            lvn_above    = [l for l in lvn_levels if l > current_price * 1.001]
            blocking_hvn = min(hvn_above) if hvn_above else None
            nearest_lvn  = min(lvn_above) if lvn_above else None
        else:
            hvn_below    = [h for h in hvn_levels if h < current_price * 0.999]
            lvn_below    = [l for l in lvn_levels if l < current_price * 0.999]
            blocking_hvn = max(hvn_below) if hvn_below else None
            nearest_lvn  = max(lvn_below) if lvn_below else None

        tp_reduction_pct = 0
        if blocking_hvn is not None:
            dist_to_hvn = abs(current_price - blocking_hvn)
            if sl_pips > 0:
                sl_price = sl_pips * pip_size
                if dist_to_hvn < sl_price * 2:
                    tp_reduction_pct = 50
                elif dist_to_hvn < sl_price * 3:
                    tp_reduction_pct = 30
                else:
                    tp_reduction_pct = 15

        path_clear = (blocking_hvn is None) and (nearest_lvn is not None)

        return {
            "path_clear":       path_clear,
            "blocking_hvn":     round(blocking_hvn, 6) if blocking_hvn else None,
            "nearest_lvn":      round(nearest_lvn, 6) if nearest_lvn else None,
            "tp_reduction_pct": tp_reduction_pct,
        }

    def check_poc_iob_fvg_alignment(self, poc: float, ob_high: float = 0,
                                     ob_low: float = 0, fvg_high: float = 0,
                                     fvg_low: float = 0) -> bool:
        """
        Check if POC aligns with OB or FVG (±0.5%).
        POC + OB/FVG aligned → strong confluence (+2 quality).
        """
        tolerance = poc * 0.005  # 0.5%

        if ob_high > 0 and ob_low > 0:
            ob_mid = (ob_high + ob_low) / 2
            if abs(poc - ob_mid) <= tolerance:
                return True

        if fvg_high > 0 and fvg_low > 0:
            fvg_mid = (fvg_high + fvg_low) / 2
            if abs(poc - fvg_mid) <= tolerance:
                return True

        return False

    def check_poor_high_low(self, df: pd.DataFrame, profile: dict) -> dict:
        """
        Poor High / Poor Low (Exhausted Auction).
        A poor high/low is a range extreme reached with thin volume,
        without a complete auction → WEAK level, likely to be revisited.

        Identification:
          - Long wick (> 60% of candle range) in the last session
          - Volume at the extreme < 20% of average → incomplete auction
          - Poor high above: do not use as TP
          - Poor low below: do not use as SL
        """
        result = {"poor_high": None, "poor_low": None,
                  "poor_high_active": False, "poor_low_active": False}

        if df is None or len(df) < 10:
            return result

        recent  = df.tail(20)
        avg_vol = float(recent["Volume"].mean()) if "Volume" in recent.columns else 0
        if avg_vol <= 0:
            return result

        period_high = float(recent["High"].max())
        period_low  = float(recent["Low"].min())

        high_candle = recent.loc[recent["High"] == period_high].iloc[-1]
        rng         = float(high_candle["High"]) - float(high_candle["Low"])
        wick_top    = float(high_candle["High"]) - max(float(high_candle["Open"]),
                                                       float(high_candle["Close"]))
        vol_at_high = float(high_candle.get("Volume", avg_vol))

        if rng > 0 and (wick_top / rng) > 0.60 and vol_at_high < avg_vol * 0.20:
            result["poor_high"]        = round(period_high, 5)
            result["poor_high_active"] = True

        low_candle = recent.loc[recent["Low"] == period_low].iloc[-1]
        rng        = float(low_candle["High"]) - float(low_candle["Low"])
        wick_bot   = min(float(low_candle["Open"]),
                         float(low_candle["Close"])) - float(low_candle["Low"])
        vol_at_low = float(low_candle.get("Volume", avg_vol))

        if rng > 0 and (wick_bot / rng) > 0.60 and vol_at_low < avg_vol * 0.20:
            result["poor_low"]        = round(period_low, 5)
            result["poor_low_active"] = True

        return result

    def check_market_context(self, df: pd.DataFrame, profile: dict) -> dict:
        """
        Range vs Trending context + Inversion/Continuation principles.

        Wyckoff 2.0 logic:
          VA_width / total_range:
            > 70% → RANGING market (cautious on breakouts)
            < 40% → TRENDING market (favor continuation)
            40-70% → neutral

        Continuation Principle:
          Price closes outside VA with volume > avg → likely continuation

        Inversion Principle:
          Price re-enters VA after breakout → target opposite side
          (related to Rule of 80% already implemented)
        """
        result = {
            "market_context":      "neutral",
            "va_width_pct":        0.0,
            "continuation_signal": False,
            "inversion_signal":    False,
            "context_note":        "",
        }

        if df is None or len(df) < 20 or profile is None:
            return result

        vah = profile.get("vah", 0)
        val = profile.get("val", 0)
        if vah <= 0 or val <= 0:
            return result

        recent      = df.tail(20)
        period_high = float(recent["High"].max())
        period_low  = float(recent["Low"].min())
        total_range = period_high - period_low
        va_width    = vah - val

        if total_range <= 0:
            return result

        va_pct                 = (va_width / total_range) * 100
        result["va_width_pct"] = round(va_pct, 1)

        if va_pct > 70:
            result["market_context"] = "range"
            result["context_note"]   = f"VA={va_pct:.0f}% of range → RANGING market"
        elif va_pct < 40:
            result["market_context"] = "trending"
            result["context_note"]   = f"VA={va_pct:.0f}% of range → TRENDING market"
        else:
            result["market_context"] = "neutral"
            result["context_note"]   = f"VA={va_pct:.0f}% of range → NEUTRAL context"

        current_price = float(df["Close"].iloc[-1])
        avg_vol       = float(recent["Volume"].mean()) if "Volume" in recent.columns else 0
        last_vol      = float(df["Volume"].iloc[-1])   if "Volume" in df.columns else 0
        above_va      = current_price > vah
        below_va      = current_price < val
        high_volume   = last_vol > avg_vol * 1.2 if avg_vol > 0 else False

        if (above_va or below_va) and high_volume:
            result["continuation_signal"] = True

        prev_prices = df["Close"].iloc[-5:-1]
        was_outside = any(p > vah or p < val for p in prev_prices)
        now_inside  = val <= current_price <= vah
        if was_outside and now_inside:
            result["inversion_signal"] = True

        return result

    def check_trend_health(self, sessions: list, profile: dict,
                           direction: str) -> dict:
        """
        Trend health via Volume Profile.

        Healthy trend indicators (in trade direction):
          - VPOC shifts in direction (up for BUY, down for SELL)
          - b profile (BUY) or P profile (SELL) → correct accumulation/distribution
          - VAH/VAL expanding in direction

        Deterioration indicators:
          - VPOC migrating against direction
          - Opposite profile shape
          - VA narrowing → market losing momentum
        """
        result = {"trend_healthy": False, "trend_score": 0, "trend_notes": []}

        if not profile:
            return result

        score = 0
        notes = []

        vpoc_migration = profile.get("vpoc_migration", "none")
        if direction == "BUY" and vpoc_migration == "up":
            score += 1
            notes.append("VPOC migrating up ✅")
        elif direction == "SELL" and vpoc_migration == "down":
            score += 1
            notes.append("VPOC migrating down ✅")
        elif vpoc_migration != "none":
            notes.append("VPOC migrating against direction ⚠️")

        dist_shape = profile.get("distribution_shape", "balanced")
        if direction == "BUY" and dist_shape == "b":
            score += 1
            notes.append("b profile (accumulation) ✅")
        elif direction == "SELL" and dist_shape == "P":
            score += 1
            notes.append("P profile (distribution) ✅")
        elif dist_shape != "balanced":
            notes.append(f"Profile {dist_shape} opposite to direction ⚠️")

        if len(sessions) >= 2:
            vpocs = [s.get("poc", 0) for s in sessions[-3:] if s.get("poc", 0) > 0]
            if len(vpocs) >= 2:
                vpoc_trend_up = all(vpocs[i] >= vpocs[i-1] for i in range(1, len(vpocs)))
                vpoc_trend_dn = all(vpocs[i] <= vpocs[i-1] for i in range(1, len(vpocs)))
                if direction == "BUY" and vpoc_trend_up:
                    score += 1
                    notes.append("Session VPOCs rising ✅")
                elif direction == "SELL" and vpoc_trend_dn:
                    score += 1
                    notes.append("Session VPOCs falling ✅")

        result["trend_score"]   = score
        result["trend_healthy"] = score >= 2
        result["trend_notes"]   = notes
        return result

    def check_multiTF_vpoc_confluence(
        self, poc_h1: float, poc_m15: float = 0,
        poc_d1: float = 0, current_price: float = 0,
        direction: str = "BUY",
        poc_m5: float = 0,
        poc_h4: float = 0,  # legacy alias, ignored
    ) -> dict:
        """
        Multi-Timeframe VP Confluence (v3.1 intraday).

        D1+H1+M15+M5 POC alignment → high-quality intraday entry.

        D1:  weekly/monthly POC — long-term institutional level
        H1:  current session POC — daily equilibrium
        M15: entry-level POC — volume concentration over last few hours
        M5:  timing POC — recent intraday volume (volatile but precise)

        Condition: ≥2 POCs within 0.3% → confluence active → +2 quality.
        """
        result = {
            "mtf_confluence":      False,
            "mtf_confluence_zone": None,
            "mtf_tfs_aligned":     [],
            "mtf_note":            "",
        }

        pocs = {}
        if poc_h1  > 0: pocs["H1"]  = poc_h1
        if poc_m15 > 0: pocs["M15"] = poc_m15
        if poc_d1  > 0: pocs["D1"]  = poc_d1
        if poc_m5  > 0: pocs["M5"]  = poc_m5

        if len(pocs) < 2:
            return result

        poc_values = list(pocs.values())
        ref        = poc_values[0]
        tolerance  = ref * 0.003  # 0.3%

        aligned = {tf: p for tf, p in pocs.items() if abs(p - ref) <= tolerance}
        if len(aligned) >= 2:
            zone = round(sum(aligned.values()) / len(aligned), 5)
            dist = abs(current_price - zone) / zone * 100 if zone > 0 else 99
            result["mtf_confluence"]      = True
            result["mtf_confluence_zone"] = zone
            result["mtf_tfs_aligned"]     = list(aligned.keys())
            result["mtf_note"] = (
                f"Confluent POCs {'+'.join(aligned.keys())} @ {zone:.5f} "
                f"(dist {dist:.2f}%)"
            )

        return result

    def check_composite_profile(self, sessions: list, direction: str) -> dict:
        """
        Composite Profile (aggregated volume across multiple sessions).

        Shows where volume concentrated over the last 3-5 sessions →
        institutional reference levels.

          - Composite POC = multi-session equilibrium level
          - Price below composite POC (BUY) → discount zone → favorable
          - Price above composite POC (SELL) → premium zone → favorable
        """
        result = {
            "composite_poc":      0.0,
            "composite_vah":      0.0,
            "composite_val":      0.0,
            "price_vs_composite": "neutral",
            "composite_note":     "",
        }

        if len(sessions) < 2:
            return result

        total_vol = sum(s.get("total_volume", 1) for s in sessions)
        if total_vol <= 0:
            return result

        wpoc     = sum(s.get("poc", 0) * s.get("total_volume", 1)
                       for s in sessions) / total_vol
        comp_vah = max(s.get("vah", 0) for s in sessions if s.get("vah", 0) > 0)
        comp_val = min(s.get("val", 99999) for s in sessions if s.get("val", 0) > 0)

        if wpoc <= 0:
            return result

        result["composite_poc"] = round(wpoc, 5)
        result["composite_vah"] = round(comp_vah, 5) if comp_vah > 0 else 0
        result["composite_val"] = round(comp_val, 5) if comp_val < 99999 else 0
        result["composite_note"] = (
            f"Composite POC {wpoc:.5f} | "
            f"cVAH {comp_vah:.5f} | cVAL {comp_val:.5f}"
        )
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# VOL PROFILE AGENT v3
# ═══════════════════════════════════════════════════════════════════════════════

class VolProfileAgent(BaseAgent):
    """
    VolProfileAgent v3 — Mandatory Layer 2 gate.

    Primary role in v3:
      1. Value Area Gate: price inside VA → setup INVALID (no trade)
      2. LVN/HVN in path: LVN = confirmation, HVN = reduces TP
      3. POC vs ICT: aligned with OB/FVG → +2 quality
      4. Institutional VWAP: directional confirmation +1 quality
      5. Naked VPOC: priority TP target
      6. SOS/SOW bar: entry trigger on M15/H1
      7. b/P Distribution: structural bias
      8. Rule of 80%: high-probability setup on VA re-entry
    """

    AGENT_NAME  = "VolProfileAgent"
    MODEL       = "deterministic_v3"
    SCORE_RANGE = (0, 100)

    def __init__(self):
        self.name            = self.AGENT_NAME
        self.model           = self.MODEL
        self.volume_analyzer = VolumeAnalyzer()
        logger.info(f"[{self.name}] Initialized — deterministic v3 (cost $0)")

    async def collect_data(self, context: dict) -> dict:
        """
        Download OHLCV data for the asset.
        v3.1: D1 + H1 + M15 + M5 — optimized for intraday.
          D1  → daily context (weekly/monthly POC)
          H1  → current session (VA gate + main VP)
          M15 → entry-level VP (last 3 sessions, for MTF confluence)
          M5  → precise entry timing (current day only)
        """
        asset = context.get("asset", "EURUSD")

        # H1 main — VA gate + session VP (10 days for naked VPOC)
        df_h1 = await asyncio.to_thread(
            self.volume_analyzer.fetch_ohlcv, asset, "10d", "1h"
        )
        # M15 for intraday MTF confluence (3 days ≈ 288 bars)
        df_m15 = None
        try:
            df_m15 = await asyncio.to_thread(
                self.volume_analyzer.fetch_ohlcv, asset, "3d", "15m"
            )
        except Exception:
            pass

        # M5 for precise entry timing (1 day ≈ 288 bars)
        df_m5 = None
        try:
            df_m5 = await asyncio.to_thread(
                self.volume_analyzer.fetch_ohlcv, asset, "1d", "5m"
            )
        except Exception:
            pass

        # D1 for daily context (1 year)
        df_d1 = None
        try:
            df_d1 = await asyncio.to_thread(
                self.volume_analyzer.fetch_ohlcv, asset, "1y", "1d"
            )
        except Exception:
            pass

        return {"df": df_h1, "df_m15": df_m15, "df_m5": df_m5, "df_d1": df_d1, "asset": asset}

    async def analyze(self, data: dict, context: dict) -> dict:
        """
        Compute full v3 Volume Profile.
        Primary output: setup_quality_contribution for Layer 2.
        """
        asset     = data.get("asset", "EURUSD")
        direction = context.get("direction", "BUY")
        df        = data.get("df")
        df_h1     = df

        ob_high  = float(context.get("ob_high",  0))
        ob_low   = float(context.get("ob_low",   0))
        fvg_high = float(context.get("fvg_high", 0))
        fvg_low  = float(context.get("fvg_low",  0))
        sl_pips  = float(context.get("sl_pips",  0))

        if df is None or df.empty:
            logger.warning(f"[{self.name}] No data for {asset}")
            return self._empty_result(asset, direction)

        current_price = float(df["Close"].iloc[-1])

        # ── 1. Main Volume Profile ────────────────────────────────────────────
        profile = self.volume_analyzer.calculate_volume_profile(df, asset)
        if not profile:
            return self._empty_result(asset, direction)

        poc             = profile["poc"]
        vah             = profile["vah"]
        val             = profile["val"]
        hvn             = profile.get("hvn_levels", [])
        lvn             = profile.get("lvn_levels", [])
        dist_shape      = profile.get("distribution_shape", "balanced")
        vpoc_migration  = profile.get("vpoc_migration", "none")

        # ── 2. Value Area Gate v3 — sole hard block ───────────────────────────
        inside_va = val <= current_price <= vah
        if inside_va:
            vwap_weekly = self.volume_analyzer.calculate_vwap_weekly(df)
            sessions    = self.volume_analyzer.calculate_session_profiles(df, asset, n_sessions=5)
            logger.info(
                f"[{self.name}] {asset} — INSIDE VALUE AREA "
                f"(price={current_price:.5f} VA={val:.5f}-{vah:.5f}) → INVALID"
            )
            return {
                "score":              50.0,
                "summary":            f"{asset}: price INSIDE Value Area — no edge",
                "bull_case":          "Wait for price to leave VA for a valid setup",
                "bear_case":          "Price at equilibrium — avoid direct entries",
                "confidence":         "high",
                "details":            f"VA={val:.5f}-{vah:.5f} POC={poc:.5f}",
                "inside_va":          True,
                "setup_invalid":      True,
                "quality_points":     0,
                "quality_notes":      ["Price inside Value Area — INVALID"],
                "poc":                poc, "vah": vah, "val": val,
                "hvn_levels":         profile.get("hvn_levels", []),
                "lvn_levels":         profile.get("lvn_levels", []),
                "naked_vpoc_targets": [],
                "nearest_nvpoc":      None,
                "blocking_hvn":       None,
                "tp_reduction_pct":   0,
                "path_clear":         True,
                "poc_aligned":        False,
                "vwap_weekly":        vwap_weekly,
                "vwap_daily":         None,
                "vwap_confirms":      False,
                "distribution_shape": dist_shape,
                "vpoc_migration":     vpoc_migration,
                "sos_sow_bar":        {"detected": False, "type": "none"},
                "rule_80":            {"active": False, "direction": "none"},
                "delta":              {},
                "rel_vol":            1.0,
                "profile":            profile,
                "raw_score":          0.0,
            }

        # ── 3. Weekly and daily VWAP ──────────────────────────────────────────
        vwap_weekly = self.volume_analyzer.calculate_vwap_weekly(df)
        vwap_daily  = self.volume_analyzer.calculate_vwap_daily(df)

        vwap_confirms = False
        vwap_signal   = "VWAP not available"
        if vwap_weekly:
            if direction == "BUY" and current_price > vwap_weekly:
                vwap_confirms = True
                vwap_signal   = f"Above weekly VWAP ({vwap_weekly:.5f}) → long bias ✅"
            elif direction == "SELL" and current_price < vwap_weekly:
                vwap_confirms = True
                vwap_signal   = f"Below weekly VWAP ({vwap_weekly:.5f}) → short bias ✅"
            elif direction == "BUY":
                vwap_signal = f"Below weekly VWAP ({vwap_weekly:.5f}) → against bias ⚠️"
            else:
                vwap_signal = f"Above weekly VWAP ({vwap_weekly:.5f}) → against bias ⚠️"

        # ── 4. Delta Volume ───────────────────────────────────────────────────
        delta   = self.volume_analyzer.calculate_delta_volume(df)
        rel_vol = self.volume_analyzer.calculate_relative_volume(df)
        div     = delta.get("divergence", "none")

        # ── 5. Sessions and Naked VPOC ────────────────────────────────────────
        sessions    = self.volume_analyzer.calculate_session_profiles(df, asset, n_sessions=5)
        naked_vpocs = self.volume_analyzer.get_naked_vpoc(sessions, current_price, direction)

        # ── 6. SOS/SOW bar ────────────────────────────────────────────────────
        trigger_bar = self.volume_analyzer.detect_sos_sow_bar(df, direction)

        # ── 7. Rule of 80% ────────────────────────────────────────────────────
        rule_80 = self.volume_analyzer.check_rule_80(df, profile)

        # ── 8. LVN/HVN in TP path ────────────────────────────────────────────
        from agents.strategy_rules import pip_size as get_pip_size
        pip_sz    = get_pip_size(asset)
        path_info = self.volume_analyzer.check_lvn_hvn_path(
            current_price, direction, profile, sl_pips, pip_sz
        )

        # ── 9. POC vs OB/FVG alignment ───────────────────────────────────────
        poc_aligned = self.volume_analyzer.check_poc_iob_fvg_alignment(
            poc, ob_high, ob_low, fvg_high, fvg_low
        )

        # ── 9b. Additional concepts ───────────────────────────────────────────

        poor             = self.volume_analyzer.check_poor_high_low(df_h1, profile)
        poor_high        = poor.get("poor_high")
        poor_low         = poor.get("poor_low")
        poor_high_active = poor.get("poor_high_active", False)
        poor_low_active  = poor.get("poor_low_active", False)

        mkt_ctx             = self.volume_analyzer.check_market_context(df_h1, profile)
        market_context      = mkt_ctx.get("market_context", "neutral")
        va_width_pct        = mkt_ctx.get("va_width_pct", 0.0)
        continuation_signal = mkt_ctx.get("continuation_signal", False)
        inversion_signal    = mkt_ctx.get("inversion_signal", False)
        context_note        = mkt_ctx.get("context_note", "")

        trend_h       = self.volume_analyzer.check_trend_health(sessions, profile, direction)
        trend_healthy = trend_h.get("trend_healthy", False)
        trend_score   = trend_h.get("trend_score", 0)
        trend_notes   = trend_h.get("trend_notes", [])

        composite    = self.volume_analyzer.check_composite_profile(sessions, direction)
        comp_poc     = composite.get("composite_poc", 0)
        comp_vah     = composite.get("composite_vah", 0)
        comp_val     = composite.get("composite_val", 0)

        price_vs_comp = "neutral"
        if comp_poc > 0:
            if direction == "BUY" and current_price < comp_poc:
                price_vs_comp = "discount"
            elif direction == "SELL" and current_price > comp_poc:
                price_vs_comp = "premium"

        # Multi-TF VP confluence (D1 + H1 + M15 + M5)
        poc_m15 = poc_m5 = poc_d1 = 0.0
        vah_m15 = val_m15 = vah_m5 = val_m5 = 0.0
        if data.get("df_m15") is not None:
            try:
                prof_m15 = self.volume_analyzer.calculate_volume_profile(data["df_m15"], asset)
                poc_m15  = prof_m15.get("poc", 0)
                vah_m15  = prof_m15.get("vah", 0)
                val_m15  = prof_m15.get("val", 0)
            except Exception:
                pass
        if data.get("df_m5") is not None:
            try:
                prof_m5 = self.volume_analyzer.calculate_volume_profile(data["df_m5"], asset)
                poc_m5  = prof_m5.get("poc", 0)
                vah_m5  = prof_m5.get("vah", 0)
                val_m5  = prof_m5.get("val", 0)
            except Exception:
                pass
        if data.get("df_d1") is not None:
            try:
                prof_d1 = self.volume_analyzer.calculate_volume_profile(data["df_d1"], asset)
                poc_d1  = prof_d1.get("poc", 0)
            except Exception:
                pass

        mtf = self.volume_analyzer.check_multiTF_vpoc_confluence(
            poc_h1=poc, poc_m15=poc_m15, poc_d1=poc_d1,
            poc_m5=poc_m5, current_price=current_price, direction=direction
        )
        mtf_confluence      = mtf.get("mtf_confluence", False)
        mtf_confluence_zone = mtf.get("mtf_confluence_zone")
        mtf_tfs_aligned     = mtf.get("mtf_tfs_aligned", [])
        mtf_note            = mtf.get("mtf_note", "")

        logger.debug(
            f"[{self.name}] {asset} {direction} | "
            f"Context={market_context} VA={va_width_pct:.0f}% | "
            f"Trend={'✅' if trend_healthy else '⚠️'} ({trend_score}/3) | "
            f"MTF={'✅' if mtf_confluence else '—'}"
        )

        # ── 10. Quality points for Layer 2 (0-8 pts) ─────────────────────────
        quality_points = 0
        quality_notes  = []

        if direction == "BUY":
            if current_price < val:
                quality_points += 2
                quality_notes.append(f"Below VAL ({val:.5f}) — accumulation +2")
            elif current_price < poc:
                quality_points += 1
                quality_notes.append(f"Below POC ({poc:.5f}) +1")
        else:
            if current_price > vah:
                quality_points += 2
                quality_notes.append(f"Above VAH ({vah:.5f}) — distribution +2")
            elif current_price > poc:
                quality_points += 1
                quality_notes.append(f"Above POC ({poc:.5f}) +1")

        if vwap_confirms:
            quality_points += 1
            quality_notes.append("Weekly VWAP confirms +1")
        elif vwap_weekly:
            quality_points -= 1
            quality_notes.append("Weekly VWAP against direction -1")

        if vwap_weekly and poc:
            if direction == "BUY":
                vwap_poc_agree = current_price > vwap_weekly and current_price > poc
            else:
                vwap_poc_agree = current_price < vwap_weekly and current_price < poc
            if vwap_poc_agree:
                quality_points += 1
                quality_notes.append("VWAP and POC agree in direction +1")

        if path_info.get("path_clear") and path_info.get("nearest_lvn"):
            quality_points += 2
            quality_notes.append(f"LVN ahead ({path_info['nearest_lvn']:.5f}) — clear runway +2")

        if poc_aligned:
            quality_points += 2
            quality_notes.append("POC aligned with OB/FVG — strong confluence +2")

        if (direction == "BUY" and dist_shape == "b") or \
           (direction == "SELL" and dist_shape == "P"):
            quality_points += 1
            quality_notes.append(f"Distribution {dist_shape} confirms {direction} +1")

        if (direction == "BUY" and vpoc_migration == "up") or \
           (direction == "SELL" and vpoc_migration == "down"):
            quality_points += 1
            quality_notes.append(f"VPOC migrating in {direction} direction +1")

        if (direction == "BUY" and div == "bullish_divergence") or \
           (direction == "SELL" and div == "bearish_divergence"):
            quality_points += 1
            quality_notes.append(f"Delta divergence {direction} +1")
        elif (direction == "BUY" and div == "bearish_divergence") or \
             (direction == "SELL" and div == "bullish_divergence"):
            quality_points -= 1
            quality_notes.append("Delta divergence against direction -1")

        if rule_80.get("active") and rule_80.get("direction") == direction:
            quality_points += 2
            quality_notes.append(f"Rule 80%: {rule_80['reason']} +2")

        if trend_healthy:
            quality_points += 1
            quality_notes.append(f"Healthy trend (score {trend_score}/3) +1")
        elif trend_score == 0 and vpoc_migration not in ("none",):
            quality_points -= 1
            quality_notes.append("Unhealthy trend -1")

        if mtf_confluence:
            quality_points += 2
            quality_notes.append(f"MTF VP confluence {'+'.join(mtf_tfs_aligned)} +2")

        if poc_m5 > 0 and vah_m5 > 0 and val_m5 > 0:
            if direction == "BUY" and current_price < val_m5:
                quality_points += 1
                quality_notes.append(f"Below M5 VAL ({val_m5:.5f}) — BUY momentum confirmed +1")
            elif direction == "SELL" and current_price > vah_m5:
                quality_points += 1
                quality_notes.append(f"Above M5 VAH ({vah_m5:.5f}) — SELL momentum confirmed +1")

        if price_vs_comp in ("discount", "premium"):
            quality_points += 1
            quality_notes.append(f"Price vs Composite POC: {price_vs_comp} +1")

        if market_context == "range":
            quality_points -= 1
            quality_notes.append("RANGING market (VA>70%) -1")

        if continuation_signal and market_context == "trending":
            quality_points += 1
            quality_notes.append("Continuation signal (breakout+volume) +1")

        if direction == "BUY" and poor_high_active and poor_high:
            quality_points -= 1
            quality_notes.append(f"Poor High @ {poor_high:.5f} — weak TP -1")
        if direction == "SELL" and poor_low_active and poor_low:
            quality_points -= 1
            quality_notes.append(f"Poor Low @ {poor_low:.5f} — weak TP -1")

        quality_points = max(0, min(9, quality_points))

        # ── 11. Score 0-100 (for consensus v2 compatibility) ─────────────────
        poc_score = 0.0
        if direction == "BUY":
            if current_price < val:
                poc_score = +4.0
            elif current_price < poc:
                poc_score = +2.0
            elif current_price > vah:
                poc_score = -3.0
        else:
            if current_price > vah:
                poc_score = +4.0
            elif current_price > poc:
                poc_score = +2.0
            elif current_price < val:
                poc_score = -3.0

        delta_score = 0.0
        bsr = delta.get("buy_sell_ratio_10", 1.0)
        if div == "bullish_divergence":
            delta_score = +3.0 if direction == "BUY" else -3.0
        elif div == "bearish_divergence":
            delta_score = -3.0 if direction == "BUY" else +3.0
        else:
            if bsr > 1.2:
                delta_score = +1.5 if direction == "BUY" else -1.5
            elif bsr < 0.8:
                delta_score = -1.5 if direction == "BUY" else +1.5

        vwap_score = 0.0
        if vwap_weekly:
            if vwap_confirms:
                vwap_score = +1.5
                if direction == "BUY" and current_price > poc:
                    vwap_score = +2.0
                elif direction == "SELL" and current_price < poc:
                    vwap_score = +2.0
            else:
                vwap_score = -1.5
        raw_score   = round(max(-VOL_RAW_MAX, min(VOL_RAW_MAX,
                            poc_score + delta_score + vwap_score)), 1)
        score_0_100 = to_score_0_100(raw_score, direction, VOL_RAW_MAX)

        if quality_points >= 5:
            confidence = "high"
        elif quality_points >= 3:
            confidence = "medium"
        else:
            confidence = "low"

        # ── 12. Summary message ───────────────────────────────────────────────
        nvpoc_str = ""
        if naked_vpocs:
            nearest_nv = naked_vpocs[0]
            nvpoc_str  = f" | Naked VPOC target: {nearest_nv['poc']:.5f}"

        summary = (
            f"{asset}: VP {score_0_100:.0f}/100 | "
            f"POC={_fmt(poc, asset)} VA={_fmt(val, asset)}-{_fmt(vah, asset)} | "
            f"Q={quality_points}/8 | dist={dist_shape}"
            f"{nvpoc_str}"
        )

        bull_case = (
            f"Price below VAL ({_fmt(val, asset)}) — accumulation zone"
            if current_price < val else
            f"Above POC ({_fmt(poc, asset)}) — bullish momentum"
        )
        if dist_shape == "b":
            bull_case += " | b distribution (accumulation)"
        if naked_vpocs and direction == "BUY":
            bull_case += f" | NVPOC target {_fmt(naked_vpocs[0]['poc'], asset)}"

        bear_case = (
            f"Price above VAH ({_fmt(vah, asset)}) — distribution zone"
            if current_price > vah else
            f"Below POC ({_fmt(poc, asset)}) — bearish pressure"
        )
        if dist_shape == "P":
            bear_case += " | P distribution (distribution)"
        if path_info.get("blocking_hvn"):
            bear_case += f" | Blocking HVN {_fmt(path_info['blocking_hvn'], asset)}"

        details = (
            f"=== VOL PROFILE AGENT v3 — {asset} ===\n"
            f"POC={_fmt(poc, asset)} | VAH={_fmt(vah, asset)} | VAL={_fmt(val, asset)}\n"
            f"Distribution: {dist_shape} | VPOC migration: {vpoc_migration}\n"
            f"Weekly VWAP: {f'{vwap_weekly:.5f}' if vwap_weekly else 'N/A'} — {vwap_signal}\n"
            f"Delta: {div} | BSR: {bsr:.3f}\n"
            f"SOS/SOW bar: {trigger_bar['type']} (vol×{trigger_bar.get('vol_ratio',1):.1f})\n"
            f"Rule 80%: {rule_80.get('reason', 'not active')}\n"
            f"Naked VPOC: {[_fmt(n['poc'], asset) for n in naked_vpocs]}\n"
            f"LVN path: {'CLEAR' if path_info['path_clear'] else 'BLOCKED'} "
            f"Blocking HVN: {_fmt(path_info['blocking_hvn'], asset) if path_info.get('blocking_hvn') else 'none'}\n"
            f"POC vs OB/FVG: {'✅ ALIGNED' if poc_aligned else 'not aligned'}\n"
            f"Quality points: {quality_points}/8\n"
            f"Quality notes: {' | '.join(quality_notes)}"
        )

        logger.info(
            f"[{self.name}] Completed — score: {score_0_100:.1f}/100 "
            f"dir: {direction} Q={quality_points}/8 conf: {confidence}"
        )

        return {
            "score":               score_0_100,
            "summary":             summary,
            "bull_case":           bull_case,
            "bear_case":           bear_case,
            "confidence":          confidence,
            "details":             details,
            "raw_score":           raw_score,
            "inside_va":           False,
            "setup_invalid":       False,
            "quality_points":      quality_points,
            "quality_notes":       quality_notes,
            "poc":                 poc,
            "vah":                 vah,
            "val":                 val,
            "hvn_levels":          hvn,
            "lvn_levels":          lvn,
            "naked_vpoc_targets":  naked_vpocs,
            "nearest_nvpoc":       naked_vpocs[0]["poc"] if naked_vpocs else None,
            "blocking_hvn":        path_info.get("blocking_hvn"),
            "tp_reduction_pct":    path_info.get("tp_reduction_pct", 0),
            "path_clear":          path_info.get("path_clear", True),
            "poc_aligned":         poc_aligned,
            "vwap_weekly":         vwap_weekly,
            "vwap_daily":          vwap_daily,
            "vwap_confirms":       vwap_confirms,
            "distribution_shape":  dist_shape,
            "vpoc_migration":      vpoc_migration,
            "sos_sow_bar":         trigger_bar,
            "rule_80":             rule_80,
            "delta":               delta,
            "rel_vol":             rel_vol,
            "profile":             profile,
            "market_context":      market_context,
            "va_width_pct":        va_width_pct,
            "continuation_signal": continuation_signal,
            "inversion_signal":    inversion_signal,
            "trend_healthy":       trend_healthy,
            "trend_score":         trend_score,
            "trend_notes":         trend_notes,
            "poor_high":           poor_high,
            "poor_low":            poor_low,
            "poor_high_active":    poor_high_active,
            "poor_low_active":     poor_low_active,
            "composite_poc":       comp_poc,
            "composite_vah":       comp_vah,
            "composite_val":       comp_val,
            "price_vs_composite":  price_vs_comp,
            "mtf_confluence":      mtf_confluence,
            "mtf_confluence_zone": mtf_confluence_zone,
            "mtf_tfs_aligned":     mtf_tfs_aligned,
            "mtf_note":            mtf_note,
            "poc_m15":             poc_m15,
            "vah_m15":             vah_m15,
            "val_m15":             val_m15,
            "poc_m5":              poc_m5,
            "vah_m5":              vah_m5,
            "val_m5":              val_m5,
            "poc_d1":              poc_d1,
        }

    def _empty_result(self, asset: str, direction: str) -> dict:
        return {
            "score": 50.0, "summary": f"{asset}: VP data unavailable",
            "bull_case": "", "bear_case": "", "confidence": "low",
            "details": "Fetch failed", "inside_va": False,
            "setup_invalid": False, "quality_points": 0,
            "quality_notes": [], "poc": 0, "vah": 0, "val": 0,
            "hvn_levels": [], "lvn_levels": [], "naked_vpoc_targets": [],
            "nearest_nvpoc": None, "blocking_hvn": None,
            "tp_reduction_pct": 0, "path_clear": True, "poc_aligned": False,
            "vwap_weekly": None, "vwap_daily": None, "vwap_confirms": False,
            "distribution_shape": "balanced", "vpoc_migration": "none",
            "sos_sow_bar": {"detected": False, "type": "none"},
            "rule_80": {"active": False, "direction": "none"},
            "delta": {}, "rel_vol": 1.0, "profile": {},
        }

    async def run(self, context: dict) -> AgentResult:
        """Override to pass direction to AgentResult."""
        direction = context.get("direction", "BUY")
        asset     = context.get("asset", "EURUSD")
        logger.info(f"[{self.name}] Starting — {asset} {direction}")

        try:
            data   = await self.collect_data(context)
            result = await self.analyze(data, context)
            score  = max(0.0, min(100.0, float(result.get("score", 50))))

            agent_result = AgentResult(
                agent=self.name,
                score=score,
                direction=direction,
                summary=result.get("summary", ""),
                bull_case=result.get("bull_case", ""),
                bear_case=result.get("bear_case", ""),
                confidence=result.get("confidence", "low"),
                details=result.get("details", ""),
                raw_data={k: v for k, v in result.items()
                          if k not in ("score", "summary", "bull_case",
                                       "bear_case", "confidence", "details")},
            )
            return agent_result

        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return AgentResult(
                agent=self.name, score=50.0, direction=direction,
                summary=f"Error: {str(e)[:80]}",
                bull_case="", bear_case="",
                confidence="low", details="", error=str(e),
            )


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    asset     = sys.argv[1] if len(sys.argv) > 1 else "MGC"
    direction = sys.argv[2] if len(sys.argv) > 2 else "BUY"

    print(f"\n{'='*65}")
    print(f"  VolProfileAgent v3 — Test: {asset} {direction}")
    print(f"{'='*65}")

    agent  = VolProfileAgent()
    result = asyncio.run(agent.run({"asset": asset, "direction": direction}))

    print(f"\n  Score:          {result.score:.1f}/100")
    print(f"  Confidence:     {result.confidence}")
    print(f"  Summary:        {result.summary}")
    raw = result.raw_data or {}
    print(f"\n  Gate v3:")
    print(f"    Inside VA:     {raw.get('inside_va', False)}")
    print(f"    Setup INVALID: {raw.get('setup_invalid', False)}")
    print(f"    Quality pts:   {raw.get('quality_points', 0)}/8")
    print(f"    Path clear:    {raw.get('path_clear', True)}")
    bhvn = raw.get('blocking_hvn')
    print(f"    Blocking HVN:  {_fmt(bhvn, asset) if bhvn else 'none'}")
    print(f"    TP reduction:  {raw.get('tp_reduction_pct', 0)}%")
    poc_v  = raw.get('poc', 0)
    vah_v  = raw.get('vah', 0)
    val_v  = raw.get('val', 0)
    vwap_w = raw.get('vwap_weekly')
    nvpoc  = raw.get('nearest_nvpoc')
    print(f"\n  Levels:")
    print(f"    POC:          {_fmt(poc_v, asset) if poc_v else 'N/A'}")
    print(f"    VAH:          {_fmt(vah_v, asset) if vah_v else 'N/A'}")
    print(f"    VAL:          {_fmt(val_v, asset) if val_v else 'N/A'}")
    print(f"    VWAP weekly:  {f'{vwap_w:.5f}' if vwap_w else 'N/A'}")
    print(f"    Naked VPOC:   {[_fmt(n['poc'], asset) for n in raw.get('naked_vpoc_targets', [])]}")
    print(f"\n  Analysis:")
    print(f"    Distribution: {raw.get('distribution_shape', 'N/A')}")
    print(f"    VPOC migr.:   {raw.get('vpoc_migration', 'none')}")
    print(f"    SOS/SOW:      {raw.get('sos_sow_bar', {}).get('type', 'none')}")
    print(f"    Rule 80%:     {raw.get('rule_80', {}).get('reason', 'not active')}")
    print(f"    POC vs ICT:   {'✅ aligned' if raw.get('poc_aligned') else 'no'}")
    print(f"    VWAP conf:    {'✅' if raw.get('vwap_confirms') else '❌'}")
    print(f"\n  Quality notes:")
    for note in raw.get("quality_notes", []):
        print(f"    • {note}")
    print(f"\n{'='*65}")
    print(f"  ✅ VolProfileAgent v3 — cost $0")
    print(f"{'='*65}\n")
