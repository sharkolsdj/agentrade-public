"""
agents/technical_agent.py  —  v3.0
TechnicalAgent — Multi-timeframe technical analysis, FULLY DETERMINISTIC.

v3 changes vs v2:
  ✅ REMOVED: GPT-4o / gpt-4o-mini (zero LLM cost)
  ✅ ADDED:   M30 and M5 via tvdatafeed (+ yfinance fallback)
  ✅ ADDED:   Deterministic SL from OB/sweep (strategy_rules v3.1)
  ✅ ADDED:   Deterministic TP from S/D zones + Naked VPOC
  ✅ ADDED:   Layer 2 score (0-16pt) for 3-layer v3 pipeline
  ✅ ADDED:   All new v3.1 concepts in scoring
  ✅ ADDED:   SMT Divergence on correlated pairs
  ✅ ADDED:   PDH/PDL, PWH/PWL, Asian Range for DOL and Judas Swing

Architecture v3 — Layer 2:
  Input:  signal from Layer 1 (rr_modifier from MacroAgent)
  Output: score 0-16pt + structural SL/TP levels
  Thresholds:
    0-3   → INVALID  (no trade)
    4-7   → WEAK     (MANUAL review)
    8-11  → GOOD     (MANUAL or AUTO if other agents OK)
    12-16 → STRONG   (AUTO if global score ≥ 85)

PARTIAL: Interface, data collection waterfall, and scoring architecture are real.
The `strategy_rules` module (imported at production time) contains proprietary
ICT/SMC detection logic and is intentionally excluded from this public repository.
The stub below preserves the interface contract without exposing strategy logic.
See paper Section 3.5b for design rationale.

Output AgentResult:
  score: 0-100 (converted from Layer 2 raw 0-16pt)
  raw_data: dict with all structural levels, confluences, score breakdown
"""

from __future__ import annotations

import asyncio
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from typing import Optional
from loguru import logger

try:
    from agents.base_agent import BaseAgent, AgentResult
except ImportError:
    from base_agent import BaseAgent, AgentResult

try:
    from utils.yfinance_lock import YF_DOWNLOAD_LOCK as _YF_DOWNLOAD_LOCK
except ImportError:
    import threading
    _YF_DOWNLOAD_LOCK = threading.Lock()

# strategy_rules: proprietary ICT/SMC detection module.
# Contains the full BOS/CHoCH/OB/FVG/MSS/Judas/MMXM detection logic.
# Not included in this public repository — see paper Section 3.5b.
_STRATEGY_RULES_AVAILABLE = False
try:
    from agents.strategy_rules import (
        analyze_all_strategies, StrategyRulesResult, pip_size,
        find_swing_points, is_monday,
    )
    _STRATEGY_RULES_AVAILABLE = True
except ImportError:
    logger.warning("[TechnicalAgent] strategy_rules not available — using stub scoring")

    def pip_size(asset: str) -> float:
        """Stub pip_size — returns default pip size per asset."""
        sizes = {
            "EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01,
            "GBPJPY": 0.01, "XAUUSD": 0.10, "MGC": 0.10,
            "BTCUSD": 1.0, "MES": 0.25, "MCL": 0.01, "NAS100": 0.25,
        }
        return sizes.get(asset.upper(), 0.0001)

    def is_monday() -> bool:
        return datetime.now().weekday() == 0


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

YFINANCE_TICKERS: dict[str, str] = {
    "EURUSD": "EURUSD=X",  "GBPUSD": "GBPUSD=X",  "USDJPY": "USDJPY=X",
    "GBPJPY": "GBPJPY=X",  "USDCAD": "USDCAD=X",  "EURAUD": "EURAUD=X",
    "EURGBP": "EURGBP=X",  "NZDJPY": "NZDJPY=X",  "EURCHF": "EURCHF=X",
    "XAUUSD": "GC=F",      "MGC":    "GC=F",
    "XAGUSD": "SI=F",
    "BTCUSD": "BTC-USD",   "ETHUSD": "ETH-USD",
    "MES":    "ES=F",
    "MCL":    "CL=F",      "NAS100": "NQ=F",
    "6E":     "EURUSD=X",
    "NZDUSD": "NZDUSD=X",  "AUDJPY": "AUDJPY=X",  "CHFJPY": "CHFJPY=X",
    "GER40":  "^GDAXI",    "US2000": "^RUT",       "DJ30":   "^DJI",
    "SP500":  "ES=F",      "USOUSD": "CL=F",
}

INSTRUMENT_TYPE: dict[str, str] = {
    "MES": "IB_MICRO", "MGC": "IB_MICRO", "MCL": "IB_MICRO", "6E": "IB_MICRO",
}

# Kill Zone windows (UTC) for session detection
KILL_ZONES = {
    "LONDON_OPEN":   (7, 9),
    "NY_OPEN":       (12, 14),
    "NY_CLOSE":      (19, 21),
    "ASIAN_SESSION": (1, 5),
}


class TechnicalAgent(BaseAgent):
    """
    TechnicalAgent v3 — Fully deterministic, zero LLM cost.

    PARTIAL: The scoring architecture, data collection, and interface are real.
    The strategy_rules module (which contains the actual BOS/OB/FVG/MSS
    detection logic) is intentionally excluded from this public repository.
    When strategy_rules is not available, a stub scoring path is used
    that returns neutral results and clearly labels them as such.
    """

    AGENT_NAME  = "TechnicalAgent"
    MODEL       = "deterministic_v3"
    SCORE_RANGE = (0, 100)

    def __init__(self):
        self.name  = self.AGENT_NAME
        self.model = self.MODEL
        logger.info(f"[{self.name}] Initialized — deterministic v3 | "
                    f"strategy_rules: {'available' if _STRATEGY_RULES_AVAILABLE else 'stub'}")

    # ─── collect_data ─────────────────────────────────────────────────────────

    async def collect_data(self, context: dict) -> dict:
        """
        Download OHLCV data from DataCache waterfall (MT4 DWX → tvdatafeed → yfinance).

        Timeframes:
          W1, D1, H4, H1 → DataCache (broker data or yfinance)
          M30, M15, M5   → DataCache (tvdatafeed primary, yfinance fallback)
        """
        asset     = str(context.get("asset", "EURUSD")).upper()
        direction = str(context.get("direction", "BUY")).upper()

        price_data: dict[str, pd.DataFrame] = {}

        # Waterfall: DataCache → yfinance direct fallback
        try:
            from broker.data_cache import data_cache
            tfs = ["W1", "D1", "H4", "H1", "M30", "M15", "M5"]
            tasks = [data_cache.get_or_fetch(asset, tf) for tf in tfs]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for tf, df in zip(tfs, results):
                if isinstance(df, pd.DataFrame) and not df.empty:
                    price_data[tf] = df
        except Exception as e:
            logger.warning(f"[{self.name}] DataCache error for {asset}: {e} — falling back to yfinance")

        # Direct yfinance fallback for missing timeframes
        ticker = YFINANCE_TICKERS.get(asset)
        if ticker:
            tf_params = [
                ("W1",  "2y",  "1wk"),
                ("D1",  "1y",  "1d"),
                ("H4",  "60d", "4h"),
                ("H1",  "30d", "1h"),
                ("M30", "10d", "30m"),
                ("M15", "10d", "15m"),
                ("M5",  "5d",  "5m"),
            ]
            for tf_name, period, interval in tf_params:
                if tf_name in price_data:
                    continue
                try:
                    with _YF_DOWNLOAD_LOCK:
                        df = await asyncio.to_thread(
                            yf.download, ticker,
                            period=period, interval=interval,
                            progress=False, auto_adjust=True,
                        )
                    if df is None or df.empty:
                        continue
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df.loc[:, ~df.columns.duplicated()]
                    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                    price_data[tf_name] = df[cols].copy()
                except Exception as e:
                    logger.warning(f"[{self.name}] yfinance {asset} {tf_name}: {e}")

        logger.info(f"[{self.name}] {asset}: {list(price_data.keys())} timeframes loaded")
        return {
            "asset":      asset,
            "direction":  direction,
            "price_data": price_data,
            "pip_size":   pip_size(asset),
        }

    # ─── analyze ──────────────────────────────────────────────────────────────

    async def analyze(self, data: dict, context: dict) -> dict:
        """
        Layer 2 scoring pipeline.

        If strategy_rules is available: uses full ICT/SMC chain analysis.
        If not (this public version): returns stub results clearly labeled.
        """
        asset      = data["asset"]
        direction  = data["direction"]
        price_data = data["price_data"]
        pip_sz     = data["pip_size"]

        instrument_type = INSTRUMENT_TYPE.get(asset, "MT4_CFD")

        # Kill zone detection
        now_utc   = datetime.now(timezone.utc)
        kill_zone = None
        for kz_name, (h_start, h_end) in KILL_ZONES.items():
            if h_start <= now_utc.hour < h_end:
                kill_zone = kz_name
                break

        if not _STRATEGY_RULES_AVAILABLE:
            return self._stub_analyze(asset, direction, price_data, pip_sz,
                                      instrument_type, kill_zone)

        # ── Full strategy_rules analysis (production path) ────────────────────
        try:
            strategy_result: StrategyRulesResult = await asyncio.to_thread(
                analyze_all_strategies, asset, direction, price_data, pip_sz
            )
            return self._build_result_from_strategy(
                asset, direction, strategy_result, price_data, pip_sz,
                instrument_type, kill_zone
            )
        except Exception as e:
            logger.error(f"[{self.name}] strategy_rules error for {asset}: {e}")
            return self._stub_analyze(asset, direction, price_data, pip_sz,
                                      instrument_type, kill_zone)

    def _build_result_from_strategy(
        self,
        asset: str,
        direction: str,
        sr,   # StrategyRulesResult
        price_data: dict,
        pip_sz: float,
        instrument_type: str,
        kill_zone: Optional[str],
    ) -> dict:
        """
        Build AgentResult output from StrategyRulesResult.

        # Strategy rules internals intentionally omitted.
        # Production: reads sr.db, sr.mss, sr.smc, sr.ote, sr.fvg, etc.
        # to compute Layer 2 score and extract SL/TP prices.
        """
        # Layer 2 raw score (0-16pt)
        layer2_raw     = getattr(sr, "layer2_score",      0)
        signal_quality = getattr(sr, "signal_quality",    "INVALID")
        confluences    = getattr(sr, "active_confluences", [])
        counter_conf   = getattr(sr, "counter_confluences", [])
        sl_pips        = getattr(sr, "sl_pips",           0.0)
        sl_price       = getattr(sr, "sl_price",          0.0)
        sl_source      = getattr(sr, "sl_source",         "")
        tp1_pips       = getattr(sr, "tp1_pips",          0.0)
        tp1_price      = getattr(sr, "tp1_price",         0.0)
        tp1_source     = getattr(sr, "tp1_source",        "")
        tp2_pips       = getattr(sr, "tp2_pips",          0.0)
        tp2_price      = getattr(sr, "tp2_price",         0.0)
        tp2_source     = getattr(sr, "tp2_source",        "")
        rr1            = getattr(sr, "rr1",               0.0)
        rr2            = getattr(sr, "rr2",               0.0)
        atr_pips       = getattr(sr, "atr_pips_m15",      0.0)

        score_0_100 = self._layer2_to_score(layer2_raw)
        summary = (
            f"{asset} {direction}: L2={layer2_raw:.1f}/16 [{signal_quality}] "
            f"SL={sl_pips:.0f}pip TP1={tp1_pips:.0f}pip R:R={rr1:.1f}"
        )

        return {
            "score":        score_0_100,
            "summary":      summary,
            "bull_case":    " | ".join(confluences[:3]),
            "bear_case":    " | ".join(counter_conf[:2]),
            "confidence":   ("high"   if signal_quality == "STRONG" else
                             "medium" if signal_quality == "GOOD"   else "low"),
            "details":      f"[TechnicalAgent v3] {asset} {direction} — L2={layer2_raw:.1f}",
            # Extra → raw_data
            "layer2_raw":         layer2_raw,
            "signal_quality":     signal_quality,
            "active_confluences": confluences,
            "counter_confluences": counter_conf,
            "instrument_type":    instrument_type,
            "kill_zone":          kill_zone,
            "strategy":           sr,    # in-memory only — not serialized by LangGraph
            "agent_detail": {
                "levels": {
                    "sl_pips":    sl_pips,  "sl_price":  sl_price,
                    "sl_source":  sl_source,
                    "tp1_pips":   tp1_pips, "tp1_price": tp1_price,
                    "tp1_source": tp1_source,
                    "tp2_pips":   tp2_pips, "tp2_price": tp2_price,
                    "tp2_source": tp2_source,
                    "atr_pips":   atr_pips,
                },
            },
            "tp_data": {
                "tp1": tp1_price, "tp1_source": tp1_source, "rr1": rr1,
                "tp2": tp2_price, "tp2_source": tp2_source, "rr2": rr2,
            },
            "atr_pips":       atr_pips,
            "sl_pips":        sl_pips,
            "tp1_pips":       tp1_pips,
            "layer2_factors": confluences,
        }

    def _stub_analyze(
        self,
        asset: str,
        direction: str,
        price_data: dict,
        pip_sz: float,
        instrument_type: str,
        kill_zone: Optional[str],
    ) -> dict:
        """
        Stub analysis path — used when strategy_rules is not available.

        Returns a clearly-labeled neutral result. The interface contract
        (all expected raw_data keys) is preserved so the orchestrator
        can process the output without errors.
        """
        # Basic ATR computation from M15 data (the one deterministic thing we can do)
        atr_pips = 0.0
        df_m15   = price_data.get("M15")
        if df_m15 is not None and len(df_m15) >= 14:
            try:
                highs  = df_m15["High"].values[-20:]
                lows   = df_m15["Low"].values[-20:]
                closes = df_m15["Close"].values[-20:]
                tr     = np.maximum(highs[1:] - lows[1:],
                          np.maximum(abs(highs[1:] - closes[:-1]),
                                     abs(lows[1:]  - closes[:-1])))
                atr_pips = float(np.mean(tr)) / pip_sz
            except Exception:
                pass

        layer2_raw    = 0.0
        signal_quality = "INVALID"
        summary = (
            f"[stub] {asset} {direction}: strategy_rules not available. "
            f"ATR≈{atr_pips:.0f}pip. See paper Section 3.5b."
        )

        return {
            "score":              50.0,     # neutral fallback
            "summary":            summary,
            "bull_case":          "[stub — strategy_rules not available]",
            "bear_case":          "[stub — strategy_rules not available]",
            "confidence":         "low",
            "details":            f"[TechnicalAgent stub] {asset} {direction} | ATR={atr_pips:.0f}pip",
            "layer2_raw":         layer2_raw,
            "signal_quality":     signal_quality,
            "active_confluences": [],
            "counter_confluences":[],
            "instrument_type":    instrument_type,
            "kill_zone":          kill_zone,
            "strategy":           None,
            "agent_detail": {
                "levels": {
                    "sl_pips": 0, "sl_price": 0, "sl_source": "stub",
                    "tp1_pips": 0, "tp1_price": 0, "tp1_source": "stub",
                    "tp2_pips": 0, "tp2_price": 0, "tp2_source": "stub",
                    "atr_pips": atr_pips,
                },
            },
            "tp_data": {
                "tp1": 0, "tp1_source": "stub", "rr1": 0,
                "tp2": 0, "tp2_source": "stub", "rr2": 0,
            },
            "atr_pips":       atr_pips,
            "sl_pips":        0.0,
            "tp1_pips":       0.0,
            "layer2_factors": [],
        }

    def _layer2_to_score(self, layer2_raw: float) -> float:
        """
        Convert Layer 2 raw score (0-16pt) to AgentResult score (0-100).
        Thresholds:
          0-3   → INVALID  (maps to 0-20)
          4-7   → WEAK     (maps to 20-50)
          8-11  → GOOD     (maps to 50-80)
          12-16 → STRONG   (maps to 80-100)
        """
        if layer2_raw <= 3:
            return round(layer2_raw / 3 * 20, 1)
        elif layer2_raw <= 7:
            return round(20 + (layer2_raw - 3) / 4 * 30, 1)
        elif layer2_raw <= 11:
            return round(50 + (layer2_raw - 7) / 4 * 30, 1)
        else:
            return round(80 + (layer2_raw - 11) / 5 * 20, 1)

    # ─── run ──────────────────────────────────────────────────────────────────

    async def run(self, context: dict) -> AgentResult:
        """Main method — collect data + analyze."""
        asset     = context.get("asset",     "EURUSD")
        direction = context.get("direction", "BUY")
        logger.info(f"[{self.name}] Starting — {asset} {direction}")

        try:
            data   = await self.collect_data(context)
            result = await self.analyze(data, context)

            score_0_100    = max(0.0, min(100.0, float(result.get("score", 50))))
            signal_quality = result.get("signal_quality", "INVALID")
            layer2_raw     = result.get("layer2_raw", 0)

            summary = result.get("summary", f"{asset} {direction}: L2={layer2_raw:.1f}/16 [{signal_quality}]")
            bull_case = " | ".join(result.get("active_confluences",  [])[:3]) or "[stub]"
            bear_case = " | ".join(result.get("counter_confluences", [])[:2]) or "No opposing signals"

            logger.info(
                f"[TechnicalAgent v3] {asset} {direction} → "
                f"L2={layer2_raw:.1f}/16 [{signal_quality}] "
                f"score={score_0_100:.1f}/100"
            )

            return AgentResult(
                agent      = self.AGENT_NAME,
                score      = score_0_100,
                direction  = direction,
                summary    = summary,
                bull_case  = bull_case,
                bear_case  = bear_case,
                confidence = ("high"   if signal_quality == "STRONG" else
                              "medium" if signal_quality == "GOOD"   else "low"),
                details    = result.get("details", ""),
                raw_data   = result,
            )

        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
            return AgentResult(
                agent=self.AGENT_NAME, score=50.0, direction=direction,
                summary=f"Error: {str(e)[:80]}",
                bull_case="", bear_case="",
                confidence="low", details=str(e),
                raw_data={"layer2_raw": 0, "signal_quality": "INVALID",
                          "active_confluences": [], "error": str(e)},
            )
