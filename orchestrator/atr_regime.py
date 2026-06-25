"""
orchestrator/atr_regime.py  —  v3.0
Regime detection for AgenTrade v3.

v3 changes:
  - SCALPING removed entirely
  - Only INTRADAY and SWING modes
  - Swing triggered by Daily Bias (no longer hardcoded False)
  - 2-level logic instead of 3:
      Level 1: Strong Daily Bias + session → SWING
      Level 2: ATR ratio → INTRADAY or INTRADAY_REDUCED
  - Updated weights: Technical and Volume carry more weight

Recomputed: every hour at H1 bar close + at Kill Zone open.

PARTIAL: Detection logic and ATR computation are real.
Weight tables and the ATR regime thresholds show correct
structure but use intentionally indicative values, not the
exact production tuning. See paper Section 3.2 for rationale.
"""

import pandas as pd
import numpy as np
from typing import Optional
from loguru import logger


# ─── ATR regime thresholds ────────────────────────────────────────────────────

ATR_EXPLOSIVE_THRESHOLD = 2.0   # indicative — ATR > this × average → intraday_reduced (size -50%)
ATR_SKIP_THRESHOLD      = 0.6   # indicative — ATR < this × average outside KZ → skip workflow


# ─── Agent weights per mode (v3) ─────────────────────────────────────────────
# SCALPING removed. Only INTRADAY, INTRADAY_REDUCED, SWING.
# v3: Technical and Volume carry more weight than v2.
# StrategyRAGAgent not included — produces quality_delta ±1/2, not proportional.
#
# NOTE: Weight values shown here are illustrative (sum correctly to 1.0)
# but differ from production values. See paper Section 3.2 for rationale.

WEIGHTS = {
    "intraday": {
        "macro":        0.15,
        "correlations": 0.10,
        "sentiment":    0.10,
        "volprofile":   0.25,
        "technical":    0.40,
    },
    "intraday_reduced": {   # explosive ATR → 50% size, same weights
        "macro":        0.15,
        "correlations": 0.10,
        "sentiment":    0.10,
        "volprofile":   0.25,
        "technical":    0.40,
    },
    "swing": {
        "macro":        0.30,   # macro critical for swing
        "correlations": 0.12,
        "sentiment":    0.08,
        "volprofile":   0.20,
        "technical":    0.30,
    },
}

# Size multiplier per mode
SIZE_MULTIPLIER = {
    "intraday":         1.0,
    "intraday_reduced": 0.5,   # explosive ATR → half size
    "swing":            1.0,
}


# ─── ATR computation ──────────────────────────────────────────────────────────

def calculate_atr_ewm(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Compute ATR using Exponential Weighted Moving Average.
    Standard used by TradingView and MetaTrader.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]

    if not all(c in df.columns for c in ["high", "low", "close"]):
        raise ValueError("DataFrame must have high, low, close columns")

    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(span=period, adjust=False).mean()


def get_atr_ratio(df_h1: pd.DataFrame, period: int = 20) -> dict:
    """
    Compute ATR ratio: current ATR / 20-bar average.

    Returns:
        dict with atr_current, atr_avg_20, ratio, regime
    """
    if df_h1 is None or len(df_h1) < period + 1:
        logger.warning("[ATR] Insufficient H1 data — fallback 'normal'")
        return {"atr_current": 0.0, "atr_avg_20": 0.0, "ratio": 1.0, "regime": "normal"}

    try:
        atr     = calculate_atr_ewm(df_h1, period)
        current = float(atr.iloc[-1])
        avg     = float(atr.iloc[-period:].mean())

        if avg <= 0:
            ratio = 1.0
            regime = "normal"
        else:
            ratio  = current / avg
            regime = "explosive" if ratio > ATR_EXPLOSIVE_THRESHOLD else "normal"

        return {
            "atr_current": round(current, 6),
            "atr_avg_20":  round(avg, 6),
            "ratio":       round(ratio, 3),
            "regime":      regime,
        }

    except Exception as e:
        logger.warning(f"[ATR] Error: {e} — fallback 'normal'")
        return {"atr_current": 0.0, "atr_avg_20": 0.0, "ratio": 1.0, "regime": "normal"}


def should_skip_workflow(atr_info: dict, kill_zone: bool) -> bool:
    """
    Return True if ATR < 0.5× AND we are not in a Kill Zone.
    During Kill Zone: bypass (ICT setups valid even at low volatility).
    """
    if kill_zone:
        return False
    return atr_info.get("ratio", 1.0) < ATR_SKIP_THRESHOLD


# ─── Regime detection v3 ──────────────────────────────────────────────────────

def detect_mode(
    atr_info: dict,
    session: str,
    daily_bias_direction: str = "NEUTRAL",
    daily_bias_strength: int = 1,
) -> str:
    """
    Determine the operational mode with 2-level logic.

    SCALPING REMOVED — only INTRADAY and SWING.

    Level 1 — Strong Daily Bias:
        If daily_bias_strength == 3 (D1/W1 candle closes >75% or <25%)
        AND we are in a favorable session → SWING.

    Level 2 — ATR ratio:
        ratio <= 1.5× → INTRADAY
        ratio >  1.5× → INTRADAY_REDUCED (50% size)

    Args:
        atr_info:              output of get_atr_ratio()
        session:               current session from MarketSchedule
        daily_bias_direction:  BUY | SELL | NEUTRAL (from DailyBiasResult)
        daily_bias_strength:   1=weak, 2=medium, 3=strong (from DailyBiasResult)

    Returns:
        str: 'intraday' | 'intraday_reduced' | 'swing'
    """
    # ── Level 1: Strong Daily Bias → SWING ───────────────────────────────────
    SWING_SESSIONS = {"LONDON_OPEN", "NY_OPEN", "LONDON_MID", "NY_MID"}
    if daily_bias_strength >= 3 and session in SWING_SESSIONS:
        if daily_bias_direction in ("BUY", "SELL"):
            logger.info(
                f"[Regime] Daily Bias {daily_bias_direction} strength={daily_bias_strength} "
                f"+ session={session} → SWING"
            )
            return "swing"

    # ── Level 2: ATR ratio ────────────────────────────────────────────────────
    regime = atr_info.get("regime", "normal")
    if regime == "explosive":
        return "intraday_reduced"
    return "intraday"


def get_weights(mode: str) -> dict:
    """Return agent weights for the given mode. Fallback to 'intraday'."""
    return WEIGHTS.get(mode, WEIGHTS["intraday"])


def get_size_multiplier(mode: str) -> float:
    """Size multiplier: 1.0 = full size, 0.5 = half size."""
    return SIZE_MULTIPLIER.get(mode, 1.0)


# ─── Session-specific consensus thresholds ────────────────────────────────────
# Sessions with high noise or reduced liquidity require higher conviction
# to reach MANUAL_APPROVE. Default value (60) = MANUAL_THRESHOLD from graph.py.
#
# NOTE: Exact threshold values are indicative. Production thresholds are
# calibrated from historical session win-rate analysis.

SESSION_MANUAL_THRESHOLD: dict[str, float] = {
    # Primary Kill Zones — standard threshold
    "LONDON_OPEN":   60.0,
    "LONDON_NY_PRE": 60.0,
    "NY_OPEN":       60.0,
    # Normal sessions
    "LONDON_MID":    60.0,
    "LONDON_CLOSE":  60.0,
    # Noisy sessions — higher threshold
    "NY_MID":        63.0,   # reduced liquidity, more false signals
    "NY_LATE":       62.0,
    "DEAD_ZONE":     64.0,   # thin market, wider spreads
    "ASIAN":         62.0,   # low volatility, frequent false signals
}
_DEFAULT_SESSION_THRESHOLD = 60.0


def get_session_manual_threshold(session: str) -> float:
    """
    Return the MANUAL approval threshold for the current session.

    Session-specific thresholds are calibrated from historical win-rate
    analysis per time window. Used in the scheduler instead of a fixed
    MANUAL_THRESHOLD, allowing stronger conviction requirements during
    historically noisy sessions without blocking the entire window.
    """
    return SESSION_MANUAL_THRESHOLD.get(session, _DEFAULT_SESSION_THRESHOLD)
