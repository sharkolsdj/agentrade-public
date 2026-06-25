"""
orchestrator/pre_filter.py
Fast pre-filter — runs locally without API calls.
Two separate scores per asset/direction:

TECHNICAL SCORE (0-6):
  1. EMA20 D1 — directional bias
  2. ATR H4   — sufficient volatility
  3. H4 candle body — directional signal
  4. RSI M15 — extreme momentum
  5. Wick signal M15 — possible liquidity sweep
  6. Key level proximity H4

ICT SCORE (0-12) — 6 criteria × 2 timeframes (2=both, 1=one only):
  1. Daily Bias     D1 + H4
  2. Valid FVG      H1 + M15
  3. Order Block    H4 + H1
  4. BOS/CHoCH      H1 + M15
  5. Liq. Sweep     M15 + M5
  6. OTE 62-79%     H4 + H1

LOGIC A: passes if technical >= 4/6 OR ict >= 5/12
"""

import asyncio
from datetime import datetime, timezone, time as dtime
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

try:
    from broker.data_cache import data_cache as _data_cache
    _CACHE_AVAILABLE = True
except ImportError:
    _data_cache = None
    _CACHE_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Assets and tickers
# ─────────────────────────────────────────────────────────────────────────────
YFINANCE_MAP: dict[str, str] = {
    "EURUSD": "EURUSD=X",  "GBPUSD": "GBPUSD=X",  "USDJPY": "USDJPY=X",
    "GBPJPY": "GBPJPY=X",  "USDCAD": "USDCAD=X",  "EURAUD": "EURAUD=X",
    "EURGBP": "EURGBP=X",  "NZDJPY": "NZDJPY=X",  "EURCHF": "EURCHF=X",
    "XAUUSD": "GC=F",      "MGC":    "GC=F",       "XAGUSD": "SI=F",
    "BTCUSD": "BTC-USD",   "ETHUSD": "ETH-USD",
    "MES":    "ES=F",
    "MCL":    "CL=F",      "NAS100": "NQ=F",
    "6E":     "EURUSD=X",
    "NZDUSD": "NZDUSD=X",  "AUDJPY": "AUDJPY=X",  "CHFJPY": "CHFJPY=X",
    "GER40":  "^GDAXI",    "US2000": "^RUT",       "DJ30":   "^DJI",
    "USOUSD": "CL=F",      "SP500":  "ES=F",
}

CRYPTO_ASSETS  = {"BTCUSD", "ETHUSD"}
FUTURES_ASSETS = {"MES", "MGC", "MCL", "NAS100", "6E"}

# Pre-filter parameters
MIN_CRITERIA       = 4     # minimum criteria satisfied out of 6
ATR_MIN_PIPS       = 15    # minimum volatility in pips (below → flat market)
BODY_RATIO_MIN     = 0.45  # minimum H4 candle body (% of range)
RSI_BUY_THRESHOLD  = 40    # RSI M15 < threshold → possible BUY (extreme momentum)
RSI_SELL_THRESHOLD = 60    # RSI M15 > threshold → possible SELL
WICK_RATIO_MIN     = 0.35  # minimum wick relative to range (possible sweep)
KEY_LEVEL_PCT      = 0.15  # % within which price is "near" a key level

ICT_MIN_SCORE = 5    # out of 12 — passes if >= 5/12
FVG_MIN_PIPS  = 3    # minimum FVG size in pips to be valid


# ─────────────────────────────────────────────────────────────────────────────
# Market schedule
# ─────────────────────────────────────────────────────────────────────────────
class MarketSchedule:
    """Determines the active session and polling interval."""

    # (start_time, end_time, session_name, interval_minutes)
    SESSIONS = [
        (dtime(7, 30),  dtime(9, 30),  "LONDON_OPEN",   10),
        (dtime(9, 30),  dtime(12, 0),  "LONDON_MID",    15),
        (dtime(12, 0),  dtime(13, 0),  "LONDON_NY_PRE", 10),
        (dtime(13, 0),  dtime(15, 0),  "NY_OPEN",       10),
        (dtime(15, 0),  dtime(16, 30), "NY_MID",        15),
        (dtime(16, 30), dtime(18, 0),  "LONDON_CLOSE",  10),
        (dtime(18, 0),  dtime(21, 0),  "NY_LATE",       30),
        (dtime(3, 0),   dtime(7, 30),  "ASIAN",         30),
    ]
    DEAD_ZONE = (dtime(21, 0), dtime(3, 0))   # 21:00 - 03:00 → 60 min

    @classmethod
    def current(cls, now: Optional[datetime] = None) -> tuple[str, int]:
        """
        Return (session_name, interval_minutes) for the current time.
        Uses Italian time (UTC+2 summer, UTC+1 winter).
        """
        if now is None:
            now = datetime.now()
        t       = now.time()
        weekday = now.weekday()   # 0=Monday, 5=Saturday, 6=Sunday

        if weekday == 5:
            return "WEEKEND_SAT", 3600
        if weekday == 6 and t < dtime(23, 0):
            return "WEEKEND_SUN", 3600

        dz_start, dz_end = cls.DEAD_ZONE
        if t >= dz_start or t < dz_end:
            return "DEAD_ZONE", 60

        for start, end, name, interval in cls.SESSIONS:
            if start <= t < end:
                return name, interval

        return "UNKNOWN", 30

    @classmethod
    def is_kill_zone(cls, now: Optional[datetime] = None) -> bool:
        session, _ = cls.current(now)
        return session in {"LONDON_OPEN", "NY_OPEN", "LONDON_CLOSE", "LONDON_NY_PRE"}

    @classmethod
    def skip_asset(cls, asset: str, now: Optional[datetime] = None) -> bool:
        """True if the asset should be skipped at this time."""
        if now is None:
            now = datetime.now()
        weekday = now.weekday()
        if asset in CRYPTO_ASSETS:
            return False
        if weekday == 5:
            return True
        if weekday == 6 and now.time() < dtime(23, 0):
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FilterSignal:
    """Pre-filter result for a single asset/direction."""
    asset:      str
    direction:  str          # "BUY" | "SELL"
    score:      int          # 0-6 technical criteria satisfied
    passed:     bool         # True if passes Logic A
    criteria:   dict[str, bool] = field(default_factory=dict)
    reason:     str = ""
    session:    str = ""
    kill_zone:  bool = False
    ict_score:  int = 0      # 0-12 ICT score (6 criteria × 2 TF)
    pass_reason: str = ""    # "technical" | "ict" | "both"
    atr_ratio:      float = 1.0
    suggested_mode: str   = "intraday"


class PreFilter:
    """
    Scans all assets in a few seconds using only yfinance + pandas.
    Zero expensive API calls.
    """

    def __init__(self):
        self._cache: dict[str, tuple] = {}
        self._cache_ttl_minutes = 14

    # ─── Data fetch with cache ────────────────────────────────────────────────

    async def _fetch(self, asset: str) -> tuple:
        """
        Fetch D1, H4, H1, M15, M5 for technical + ICT criteria.
        Uses data_cache if available, otherwise falls back to yfinance directly.
        Returns: (df_d1, df_h4, df_h1, df_m15, df_m5)
        """
        empty = pd.DataFrame()

        if _CACHE_AVAILABLE and _data_cache and _data_cache._initialized:
            def _c(v):
                if v is None or v.empty:
                    return empty
                df = v.copy()
                df.columns = [c.lower() for c in df.columns]
                return df
            df_d1  = _c(_data_cache.get(asset, "D1"))
            df_h4  = _c(_data_cache.get(asset, "H4"))
            df_h1  = _c(_data_cache.get(asset, "H1"))
            df_m15 = _c(_data_cache.get(asset, "M15"))
            df_m5  = _c(_data_cache.get(asset, "M5"))
            # Tolerant gate: don't zero out the whole asset if only D1 is missing.
            # Individual ICT/tech criteria handle empty TFs (df.empty → skip).
            if not df_h1.empty and not df_m15.empty:
                return df_d1, df_h4, df_h1, df_m15, df_m5
            return empty, empty, empty, empty, empty

        yf_ticker = YFINANCE_MAP.get(asset)
        if not yf_ticker:
            return empty, empty, empty, empty, empty

        if asset in self._cache:
            cached_time, df_d1, df_h4, df_m15 = self._cache[asset]
            if (datetime.now() - cached_time).seconds / 60 < self._cache_ttl_minutes:
                return df_d1, df_h4, empty, df_m15, empty

        try:
            from utils.yfinance_lock import YF_DOWNLOAD_LOCK as _YF_LOCK

            def _dl(period, interval):
                with _YF_LOCK:
                    return yf.download(
                        yf_ticker, period=period, interval=interval,
                        progress=False, auto_adjust=True
                    )

            df_d1  = await asyncio.to_thread(_dl, "60d", "1d")
            df_h4  = await asyncio.to_thread(_dl, "30d", "4h")
            df_m15 = await asyncio.to_thread(_dl, "5d",  "15m")

            def _flatten(df):
                if df.empty:
                    return df
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.loc[:, ~df.columns.duplicated()]
                df.columns = [c.lower() for c in df.columns]
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                return df

            df_d1  = _flatten(df_d1)
            df_h4  = _flatten(df_h4)
            df_m15 = _flatten(df_m15)
            self._cache[asset] = (datetime.now(), df_d1, df_h4, df_m15)
            return df_d1, df_h4, empty, df_m15, empty

        except Exception as e:
            logger.warning(f"[PreFilter] Fetch error {asset}: {e}")
            return empty, empty, empty, empty, empty

    # ─── Technical criteria ───────────────────────────────────────────────────

    def _criterion_ema20(self, df_d1: pd.DataFrame, direction: str) -> bool:
        """C1: EMA20 D1 — directional bias."""
        if len(df_d1) < 21:
            return False
        close = float(df_d1["close"].iloc[-1])
        ema20 = float(df_d1["close"].ewm(span=20, adjust=False).mean().iloc[-1])
        return close > ema20 if direction == "BUY" else close < ema20

    def _criterion_atr(self, df_h4: pd.DataFrame) -> bool:
        """C2: Minimum ATR H4 — market not flat."""
        if len(df_h4) < 14:
            return False
        high  = df_h4["high"]
        low   = df_h4["low"]
        close = df_h4["close"]
        tr    = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr14            = float(tr.rolling(14).mean().iloc[-1])
        atr_normalized   = atr14 * 10000  # approximate pip conversion for forex
        return atr_normalized > ATR_MIN_PIPS

    def _criterion_candle_body(self, df_h4: pd.DataFrame, direction: str) -> bool:
        """C3: H4 candle body — clean directional signal."""
        if len(df_h4) < 2:
            return False
        last      = df_h4.iloc[-1]
        o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
        rng       = h - l
        if rng == 0:
            return False
        body      = abs(c - o)
        if body / rng < BODY_RATIO_MIN:
            return False
        return (c > o) if direction == "BUY" else (c < o)

    def _criterion_rsi(self, df_m15: pd.DataFrame, direction: str) -> bool:
        """C4: RSI M15 — extreme momentum."""
        if len(df_m15) < 15:
            return False
        delta   = df_m15["close"].diff()
        gain    = delta.clip(lower=0).rolling(14).mean()
        loss    = (-delta.clip(upper=0)).rolling(14).mean()
        rs      = gain / loss.replace(0, float("nan"))
        rsi_val = float((100 - (100 / (1 + rs))).iloc[-1])
        return rsi_val < RSI_BUY_THRESHOLD if direction == "BUY" else rsi_val > RSI_SELL_THRESHOLD

    def _criterion_wick(self, df_m15: pd.DataFrame, direction: str) -> bool:
        """C5: Wick signal M15 — possible liquidity sweep."""
        if len(df_m15) < 3:
            return False
        last  = df_m15.iloc[-1]
        o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
        rng = h - l
        if rng == 0:
            return False
        if direction == "BUY":
            return (min(o, c) - l) / rng > WICK_RATIO_MIN
        else:
            return (h - max(o, c)) / rng > WICK_RATIO_MIN

    def _criterion_key_level(self, df_m15: pd.DataFrame, df_h4: pd.DataFrame, direction: str) -> bool:
        """C6: Proximity to key level — recent swing high/low on H4."""
        if len(df_m15) < 2 or len(df_h4) < 20:
            return False
        current_price = float(df_m15["close"].iloc[-1])
        swing_high    = float(df_h4["high"].tail(20).nlargest(3).iloc[-1])
        swing_low     = float(df_h4["low"].tail(20).nsmallest(3).iloc[-1])
        target_level  = swing_low if direction == "BUY" else swing_high
        price_range   = swing_high - swing_low
        if price_range == 0:
            return False
        return abs(current_price - target_level) / price_range < KEY_LEVEL_PCT

    # ─── ICT criteria (score 0-2 per criterion, max 12 total) ────────────────

    def _ict_daily_bias(self, df_d1: pd.DataFrame, df_h4: pd.DataFrame, direction: str) -> int:
        """ICT1: Daily Bias — EMA20 + swing structure on D1 and H4. Score 0-2."""
        score = 0
        for df, min_len in [(df_d1, 21), (df_h4, 21)]:
            if df.empty or len(df) < min_len:
                continue
            col = "close" if "close" in df.columns else "Close"
            if col not in df.columns:
                continue
            close = float(df[col].iloc[-1])
            ema20 = float(df[col].ewm(span=20, adjust=False).mean().iloc[-1])
            if direction == "BUY" and close > ema20:
                score += 1
            elif direction == "SELL" and close < ema20:
                score += 1
        return score

    def _ict_fvg(self, df_h1: pd.DataFrame, df_m15: pd.DataFrame, direction: str) -> int:
        """ICT2: Valid unfilled Fair Value Gap on H1 and M15. Score 0-2."""
        score = 0
        for df in [df_h1, df_m15]:
            if df.empty or len(df) < 5:
                continue
            hi_col = "High" if "High" in df.columns else "high"
            lo_col = "Low"  if "Low"  in df.columns else "low"
            cl_col = "Close" if "Close" in df.columns else "close"
            if hi_col not in df.columns:
                continue
            highs  = df[hi_col].values
            lows   = df[lo_col].values
            closes = df[cl_col].values
            found  = False
            for i in range(max(2, len(df) - 20), len(df) - 1):
                if direction == "BUY":
                    if lows[i] > highs[i - 2]:
                        if closes[-1] > highs[i - 2]:
                            found = True
                            break
                else:
                    if highs[i] < lows[i - 2]:
                        if closes[-1] < lows[i - 2]:
                            found = True
                            break
            if found:
                score += 1
        return score

    def _ict_order_block(self, df_h4: pd.DataFrame, df_h1: pd.DataFrame, direction: str) -> int:
        """ICT3: Valid (unmitigated) Order Block on H4 and H1. Score 0-2."""
        score = 0
        for df in [df_h4, df_h1]:
            if df.empty or len(df) < 10:
                continue
            op_col = "Open"  if "Open"  in df.columns else "open"
            hi_col = "High"  if "High"  in df.columns else "high"
            lo_col = "Low"   if "Low"   in df.columns else "low"
            cl_col = "Close" if "Close" in df.columns else "close"
            if op_col not in df.columns:
                continue
            opens  = df[op_col].values
            highs  = df[hi_col].values
            lows   = df[lo_col].values
            closes = df[cl_col].values
            current = closes[-1]
            found   = False
            for i in range(max(1, len(df) - 15), len(df) - 2):
                body = abs(closes[i] - opens[i])
                rng  = highs[i] - lows[i]
                if rng == 0:
                    continue
                if direction == "BUY":
                    if closes[i] < opens[i] and body / rng > 0.5:
                        if lows[i] < current < highs[i] * 1.02:
                            found = True
                            break
                else:
                    if closes[i] > opens[i] and body / rng > 0.5:
                        if lows[i] * 0.98 < current < highs[i]:
                            found = True
                            break
            if found:
                score += 1
        return score

    def _ict_bos(self, df_h1: pd.DataFrame, df_m15: pd.DataFrame, direction: str) -> int:
        """ICT4: BOS or CHoCH on H1 and M15. Score 0-2."""
        score = 0
        for df in [df_h1, df_m15]:
            if df.empty or len(df) < 20:
                continue
            hi_col = "High"  if "High"  in df.columns else "high"
            lo_col = "Low"   if "Low"   in df.columns else "low"
            cl_col = "Close" if "Close" in df.columns else "close"
            if hi_col not in df.columns:
                continue
            highs  = df[hi_col].values
            lows   = df[lo_col].values
            closes = df[cl_col].values
            n        = len(closes)
            lookback = min(15, n - 3)
            recent_high = float(highs[n - lookback - 2:n - 2].max())
            recent_low  = float(lows[n - lookback - 2:n - 2].min())
            last_close  = float(closes[-1])
            if direction == "BUY" and last_close > recent_high:
                score += 1
            elif direction == "SELL" and last_close < recent_low:
                score += 1
        return score

    def _ict_sweep(self, df_m15: pd.DataFrame, df_m5: pd.DataFrame, direction: str) -> int:
        """ICT5: Liquidity Sweep on M15 and M5 (spike + return). Score 0-2."""
        score = 0
        for df in [df_m15, df_m5]:
            if df.empty or len(df) < 5:
                continue
            hi_col = "High"  if "High"  in df.columns else "high"
            lo_col = "Low"   if "Low"   in df.columns else "low"
            cl_col = "Close" if "Close" in df.columns else "close"
            if hi_col not in df.columns:
                continue
            highs  = df[hi_col].values
            lows   = df[lo_col].values
            closes = df[cl_col].values
            n      = len(closes)
            if n < 5:
                continue
            ref_high  = float(highs[n - 6:n - 2].max())
            ref_low   = float(lows[n - 6:n - 2].min())
            last_high  = float(highs[-1])
            last_low   = float(lows[-1])
            last_close = float(closes[-1])
            if direction == "BUY":
                if last_low < ref_low and last_close > ref_low:
                    score += 1
            else:
                if last_high > ref_high and last_close < ref_high:
                    score += 1
        return score

    def _ict_ote(self, df_h4: pd.DataFrame, df_h1: pd.DataFrame, direction: str) -> int:
        """ICT6: OTE — price in Fibonacci 62-79% zone on H4 and H1. Score 0-2."""
        score = 0
        for df in [df_h4, df_h1]:
            if df.empty or len(df) < 20:
                continue
            hi_col = "High"  if "High"  in df.columns else "high"
            lo_col = "Low"   if "Low"   in df.columns else "low"
            cl_col = "Close" if "Close" in df.columns else "close"
            if hi_col not in df.columns:
                continue
            highs  = df[hi_col].values
            lows   = df[lo_col].values
            closes = df[cl_col].values
            n           = len(closes)
            swing_high  = float(highs[n - 20:].max())
            swing_low   = float(lows[n - 20:].min())
            rng         = swing_high - swing_low
            if rng <= 0:
                continue
            current = float(closes[-1])
            if direction == "BUY":
                ote_high = swing_high - rng * 0.618
                ote_low  = swing_high - rng * 0.786
                if ote_low <= current <= ote_high:
                    score += 1
            else:
                ote_low  = swing_low + rng * 0.618
                ote_high = swing_low + rng * 0.786
                if ote_low <= current <= ote_high:
                    score += 1
        return score

    # ─── Single asset analysis ────────────────────────────────────────────────

    async def analyze_asset(
        self, asset: str, direction: str, session: str, kill_zone: bool
    ) -> FilterSignal:
        """
        Analyze a single asset/direction.
        Computes technical score (0-6) and ICT score (0-12).
        Logic A: passes if technical >= 4/6 OR ict >= 5/12.
        """
        df_d1, df_h4, df_h1, df_m15, df_m5 = await self._fetch(asset)

        if df_d1.empty or df_h4.empty:
            return FilterSignal(
                asset=asset, direction=direction, score=0, passed=False,
                reason="Data not available", session=session,
                kill_zone=kill_zone, ict_score=0,
            )

        # ── Technical score (0-6) ─────────────────────────────────────────────
        criteria = {
            "ema20_bias":     self._criterion_ema20(df_d1, direction),
            "atr_volatility": self._criterion_atr(df_h4),
            "candle_body":    self._criterion_candle_body(df_h4, direction),
            "rsi_extreme":    self._criterion_rsi(df_m15, direction) if not df_m15.empty else False,
            "wick_sweep":     self._criterion_wick(df_m15, direction) if not df_m15.empty else False,
            "key_level":      self._criterion_key_level(df_m15, df_h4, direction) if not df_m15.empty else False,
        }
        tech_score = sum(criteria.values())

        tech_threshold = MIN_CRITERIA - 1 if kill_zone else MIN_CRITERIA
        tech_pass = tech_score >= tech_threshold

        # ── ICT score (0-12) ──────────────────────────────────────────────────
        ict_scores = {
            "daily_bias":  self._ict_daily_bias(df_d1, df_h4, direction),
            "fvg":         self._ict_fvg(df_h1, df_m15, direction),
            "order_block": self._ict_order_block(df_h4, df_h1, direction),
            "bos_choch":   self._ict_bos(df_h1, df_m15, direction),
            "liq_sweep":   self._ict_sweep(df_m15, df_m5, direction),
            "ote":         self._ict_ote(df_h4, df_h1, direction),
        }
        ict_score = sum(ict_scores.values())
        ict_pass  = ict_score >= ICT_MIN_SCORE

        # ── Logic A: passes if technical OR ict ───────────────────────────────
        passed = tech_pass or ict_pass

        if tech_pass and ict_pass:
            pass_reason = "both"
        elif tech_pass:
            pass_reason = "technical"
        elif ict_pass:
            pass_reason = "ict"
        else:
            pass_reason = ""

        tech_active = [k for k, v in criteria.items() if v]
        ict_active  = [f"{k}:{v}" for k, v in ict_scores.items() if v > 0]
        reason = (
            f"tech={tech_score}/6 [{', '.join(tech_active)}] | "
            f"ict={ict_score}/12 [{', '.join(ict_active)}]"
        ) if passed else (
            f"tech={tech_score}/6 ict={ict_score}/12 — below threshold"
        )

        # ── ATR ratio + suggested mode ────────────────────────────────────────
        atr_ratio      = 1.0
        suggested_mode = "intraday"
        try:
            from orchestrator.atr_regime import get_atr_ratio, detect_mode, should_skip_workflow
            if not df_h1.empty and len(df_h1) >= 21:
                atr_info       = get_atr_ratio(df_h1)
                atr_ratio      = atr_info.get("ratio", 1.0)
                suggested_mode = detect_mode(atr_info, session)

                if should_skip_workflow(atr_info, kill_zone) and passed:
                    return FilterSignal(
                        asset=asset, direction=direction,
                        score=tech_score, passed=False,
                        criteria=criteria,
                        reason=(
                            f"ATR ratio {atr_ratio:.2f}× < 0.5× outside KZ — "
                            f"flat market (tech={tech_score}/6 ict={ict_score}/12 OK but skip)"
                        ),
                        session=session, kill_zone=kill_zone,
                        ict_score=ict_score, pass_reason="",
                        atr_ratio=atr_ratio, suggested_mode=suggested_mode,
                    )
        except Exception as e:
            logger.debug(f"[PreFilter] ATR check error {asset}: {e}")

        return FilterSignal(
            asset=asset, direction=direction,
            score=tech_score, passed=passed,
            criteria=criteria, reason=reason,
            session=session, kill_zone=kill_zone,
            ict_score=ict_score, pass_reason=pass_reason,
            atr_ratio=atr_ratio, suggested_mode=suggested_mode,
        )

    # ─── Full scan ────────────────────────────────────────────────────────────

    async def scan_asset(self, asset: str, direction: str) -> list[FilterSignal]:
        """Scan a single asset/direction — used by the rapid scan loop."""
        now       = datetime.now()
        session, _= MarketSchedule.current(now)
        kill_zone = MarketSchedule.is_kill_zone(now)
        result    = await self.analyze_asset(asset, direction, session, kill_zone)
        return [result] if result.passed else []

    async def scan_all(self, skip_assets: set = None) -> list[FilterSignal]:
        """
        Scan all assets for BUY and SELL.
        Returns only signals that passed the filter.
        Typical runtime: 3-8 seconds.

        Args:
            skip_assets: set of assets to skip (e.g. assets with open trades)
        """
        now         = datetime.now()
        session, _  = MarketSchedule.current(now)
        kill_zone   = MarketSchedule.is_kill_zone(now)
        skip_assets = skip_assets or set()

        logger.info(
            f"[PreFilter] Scan — session: {session} | kill zone: {kill_zone}"
            + (f" | skip: {skip_assets}" if skip_assets else "")
        )

        # Block scan during weekend DEAD_ZONE: markets are closed,
        # data is stale from Friday, ICT signals would be false positives.
        weekday = now.weekday()
        if session == "DEAD_ZONE" and weekday in (5, 6):
            logger.info(
                f"[PreFilter] Weekend DEAD_ZONE (day={weekday}) — "
                f"scan suspended (markets closed, stale data)"
            )
            return []

        tasks = []
        for asset in YFINANCE_MAP.keys():
            if MarketSchedule.skip_asset(asset, now):
                continue
            if asset in skip_assets:
                logger.debug(f"[PreFilter] {asset} — open trade, skip analysis")
                continue
            for direction in ("BUY", "SELL"):
                tasks.append(self.analyze_asset(asset, direction, session, kill_zone))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        signals = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"[PreFilter] Error: {r}")
                continue
            if r.passed:
                signals.append(r)
                logger.info(
                    f"[PreFilter] ✅ {r.asset} {r.direction} | "
                    f"tech={r.score}/6 ict={r.ict_score}/12 | "
                    f"ATR={r.atr_ratio:.2f}× mode={r.suggested_mode} | "
                    f"via={r.pass_reason} | {r.reason[:60]}"
                )

        logger.info(f"[PreFilter] Scan complete: {len(signals)} signals from {len(tasks)} analyses")
        return signals
