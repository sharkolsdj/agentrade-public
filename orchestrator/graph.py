"""
orchestrator/graph.py
LangGraph workflow v2 — connects 8 agents in sequential/parallel stages.
Manages weighted 0-100 consensus and decides when to invoke RiskManagerAgent.

Pipeline v2:
  account_state_node      ← reads equity, drawdown, open positions
    ↓
  regime_detection_node   ← NEW: ATR ratio → mode (intraday/swing) + weights
    ↓
  direction_lock_node     ← NEW: COT extreme check → rr_modifier contribution
    ↓
  [MacroAgent]            ┐
  [SentimentAgent]        ├── parallel (score 0-100 each)
  [VolProfileAgent]       ┘  ← NEW: replaces CotVolumeAgent
    ↓
  [TechnicalAgent]        ← sequential (produces sl_pips for StrategyRAGAgent)
    ↓
  [CorrelationsAgent]     ← with divergence_veto flag
    ↓
  consensus_check_node    ← weighted score 0-100 with mode-specific weights
    ↓
  [StrategyRAGAgent]      ← quality_delta ±2/±1/0
    ↓
  [RiskManagerAgent]      ← thresholds: AUTO≥85, MANUAL 56-84
    ↓
  telegram_dispatch_node
    ↓
  END
"""

import asyncio
import os
from datetime import datetime, timezone
from typing import Literal

from loguru import logger

# LangGraph
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not available — using sequential mode")

from orchestrator.state import TradingState

# Agents
from agents.macro_agent        import MacroAgent
from agents.sentiment_agent    import SentimentAgent
from agents.technical_agent    import TechnicalAgent
from agents.vol_profile_agent  import VolProfileAgent
from agents.correlations_agent import CorrelationsAgent
from agents.strategy_rag_agent import StrategyRAGAgent
from agents.risk_manager_agent import RiskManagerAgent, AUTO_EXECUTE_THRESHOLD, MANUAL_THRESHOLD
from orchestrator.atr_regime   import (
    detect_mode, get_weights, get_size_multiplier,
    get_atr_ratio, get_session_manual_threshold,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PRELIMINARY_THRESHOLD = 35.0   # weighted score < 35 → NO_TRADE
VETO_PENALTY          = 10.0   # penalty if CorrelationsAgent veto active


def _to_0_100(score: float, score_range: tuple) -> float:
    """
    Convert a score from an arbitrary range to 0-100.
    score_range: (min, max) of the original range.
    0 in the original range → 50 in 0-100 (neutral).
    Guard: nan/inf → 50.0 (neutral).
    """
    import math
    if score is None or (isinstance(score, float) and (math.isnan(score) or math.isinf(score))):
        return 50.0
    lo, hi = score_range
    if score >= 0:
        return min(100.0, 50.0 + (score / hi) * 50.0) if hi > 0 else 50.0
    else:
        return max(0.0, 50.0 + (score / abs(lo)) * 50.0) if lo < 0 else 50.0


def compute_rr_modifier(
    macro_contribution: float = 0.0,
    cot_contribution: float = 0.0,
    correlation_contribution: float = 0.0,
    high_impact_news: bool = False,
) -> float:
    """
    v3: Compute rr_modifier for Layer 1.
    Formula: 0.75 + macro + cot + corr - 0.20 if high-impact news
    Clamp: 0.50 - 1.00 (never blocks, never exceeds 1.0)

    Examples:
      macro=+0.25 + cot=+0.20 + corr=+0.05 → 1.25 → clamped 1.00
      macro=-0.25 + cot=-0.20              → 0.30 → clamped 0.50
    """
    modifier = 0.75 + macro_contribution + cot_contribution + correlation_contribution
    if high_impact_news:
        modifier -= 0.20
    return round(max(0.50, min(1.00, modifier)), 3)


# Assets routed to IB Micro Futures (not MT4 CFD)
IB_MICRO_ASSETS = {"MES", "MGC", "MCL", "6E"}

# Index CFDs — cautious treatment (inside-VA hard-block in trend, never AUTO)
INDEX_ASSETS         = {"NAS100", "GER40", "US2000", "DJ30", "SP500"}
# SP500 and NAS100 excluded from hard-block — more liquid/robust indices.
# Inside-VA on SP500/NAS100 is not hard-blocked: treated as no-edge → MANUAL + tag
VA_HARDBLOCK_INDICES = INDEX_ASSETS - {"SP500", "NAS100"}
# ATR ratio threshold below which an inside-VA setup is considered "range-day"
RANGE_DAY_ATR_THR    = 0.8

# Agents in parallel stage v2
PARALLEL_AGENTS = ["MacroAgent", "SentimentAgent", "VolProfileAgent"]
# Agents in weighted consensus (5 main agents)
WEIGHTED_AGENTS = ["MacroAgent", "SentimentAgent", "VolProfileAgent",
                   "TechnicalAgent", "CorrelationsAgent"]

# ─────────────────────────────────────────────────────────────────────────────
# Agent singletons — initialized once
# ─────────────────────────────────────────────────────────────────────────────
_agents: dict = {}

# Per-asset mutex — prevents duplicate workflow on the same asset in rapid scans.
# If workflow for the same asset is already running (e.g. scan every 30min but
# previous workflow has not finished), we skip immediately.
# Set is cleared in finally → even if workflow crashes, it doesn't stay locked.
_executing_assets: set[str] = set()


def _get_agent(name: str):
    if name not in _agents:
        mapping = {
            "macro":        MacroAgent,
            "sentiment":    SentimentAgent,
            "technical":    TechnicalAgent,
            "vol_profile":  VolProfileAgent,
            "correlations": CorrelationsAgent,
            "strategy_rag": StrategyRAGAgent,
            "risk_manager": RiskManagerAgent,
        }
        _agents[name] = mapping[name]()
    return _agents[name]


# ─────────────────────────────────────────────────────────────────────────────
# Graph nodes
# ─────────────────────────────────────────────────────────────────────────────

async def account_state_node(state: TradingState) -> TradingState:
    """
    Read account state.
    TODO: connect to IB Gateway via ib_insync for live data.
    Currently uses static values or last saved values from PostgreSQL.

    Live connection example:
        from ib_insync import IB
        ib = IB()
        ib.connect(os.getenv('IB_HOST'), int(os.getenv('IB_PORT')), clientId=1)
        equity = float([a for a in ib.accountValues()
                        if a.tag == 'NetLiquidation'][0].value)
    """
    logger.info(f"[Orchestrator] account_state_node — {state['asset']}")
    from agents.risk_manager_agent import PAPER_EQUITY_USD
    return {
        **state,
        "equity":               PAPER_EQUITY_USD,
        "open_positions":       state.get("open_positions", []),
        "current_drawdown_pct": state.get("current_drawdown_pct", 0.0),
        "current_step":         "account_state",
    }


async def regime_detection_node(state: TradingState) -> TradingState:
    """
    v3 — Determine operational mode (intraday/swing).
    SCALPING REMOVED. Real Daily Bias replaces the hardcoded False.
    Monday Rule removed — higher consensus thresholds on Mondays
    are handled at the consensus check level.
    """
    asset = state["asset"]
    logger.info(f"[Orchestrator] regime_detection — {asset}")
    session = "LONDON_MID"
    try:
        import yfinance as yf
        import pandas as pd
        from orchestrator.atr_regime import get_atr_ratio, detect_mode, get_weights, get_size_multiplier, WEIGHTS
        from agents.strategy_rules import analyze_daily_bias
        from broker.data_cache import data_cache

        # ATR H1 from canonical datafeed (DataCache: MT4/IB→yf)
        df_h1 = await data_cache.get_fresh(asset, "H1")
        if df_h1 is None or getattr(df_h1, "empty", True):
            raise RuntimeError(f"H1 not available from DataCache for {asset}")
        df_h1 = df_h1.copy()
        df_h1.columns = [str(c).lower() for c in df_h1.columns]
        if df_h1.index.duplicated().any():
            df_h1 = df_h1[~df_h1.index.duplicated(keep="last")]
        if df_h1.index.tz is not None:
            df_h1.index = df_h1.index.tz_localize(None)
        atr_info = get_atr_ratio(df_h1)

        try:
            from orchestrator.scheduler import MarketSchedule
            session, _ = MarketSchedule.current()
        except Exception:
            pass

        # v3: Real Daily Bias from cache
        try:
            df_d1_bias = data_cache.get(asset, "D1")
            df_w1_bias = data_cache.get(asset, "W1")
            db_result  = analyze_daily_bias(df_d1_bias, df_w1_bias, state.get("direction", ""))
            daily_bias_dir = db_result.direction
            daily_bias_str = db_result.strength
        except Exception as e:
            logger.debug(f"[Orchestrator] daily_bias fallback: {e}")
            daily_bias_dir = "NEUTRAL"
            daily_bias_str = 1

        mode            = detect_mode(atr_info, session, daily_bias_dir, daily_bias_str)
        weights         = get_weights(mode)
        size_multiplier = get_size_multiplier(mode)
        logger.info(
            f"[Orchestrator] Regime: {mode.upper()} | ATR={atr_info['ratio']:.2f}x | "
            f"size={size_multiplier}x | DailyBias={daily_bias_dir}(strength={daily_bias_str})"
        )
    except Exception as e:
        logger.warning(f"[Orchestrator] regime_detection fallback: {e}")
        from orchestrator.atr_regime import WEIGHTS
        mode = "intraday"; weights = WEIGHTS["intraday"]; size_multiplier = 1.0
        atr_info = {"ratio": 1.0, "regime": "normal"}

    return {
        **state,
        "mode":             mode,
        "weights":          weights,
        "atr_ratio":        atr_info.get("ratio", 1.0),
        "atr_regime":       atr_info.get("regime", "normal"),
        "size_multiplier":  size_multiplier,
        "session":          session,
        "current_step":     "regime_detection",
    }


async def direction_lock_node(state: TradingState) -> TradingState:
    """
    v3 — COT no longer blocks trades.
    Becomes a contribution to rr_modifier (-0.20 / 0 / +0.20).
    Directional blocking is removed in v3 — only R:R modulation.
    """
    asset     = state["asset"]
    direction = state["direction"]
    logger.info(f"[Orchestrator] direction_lock — {asset} {direction}")
    try:
        macro_agent = _get_agent("macro")
        full        = await macro_agent.run_full()
        asset_data  = full["scores"].get(asset, {})
        cot_extreme = asset_data.get("cot_extreme", False)
        cot_bias    = asset_data.get("cot_bias", "neutral")
        cot_index   = float(asset_data.get("cot_index", 50.0))

        # v3: COT never blocks — contribution to rr_modifier
        cot_contribution = 0.0
        if cot_extreme:
            if direction == "BUY" and cot_bias == "bearish":
                cot_contribution = -0.20   # opposing bias → reduces R:R
            elif direction == "SELL" and cot_bias == "bullish":
                cot_contribution = -0.20
            elif direction == "BUY" and cot_bias == "bullish":
                cot_contribution = +0.20   # favorable bias → improves R:R
            elif direction == "SELL" and cot_bias == "bearish":
                cot_contribution = +0.20

        logger.info(
            f"[Orchestrator] Direction OK — COT {cot_bias} ({cot_index:.0f}%) "
            f"→ contribution={cot_contribution:+.2f} (no block in v3)"
        )
    except Exception as e:
        logger.warning(f"[Orchestrator] direction_lock error: {e}")
        cot_extreme = False; cot_bias = "neutral"; cot_index = 50.0; cot_contribution = 0.0

    return {
        **state,
        "cot_extreme":      cot_extreme,
        "cot_bias":         cot_bias,
        "cot_index":        cot_index,
        "cot_contribution": cot_contribution,
        "direction_locked": True,   # v3: always True — no block
        "abort_reason":     None,
        "skip_reason":      state.get("skip_reason", ""),
        "current_step":     "direction_lock",
    }


def should_continue_after_lock(state: TradingState) -> Literal["parallel", "end"]:
    """Router — abort if direction_lock has blocked."""
    return "end" if state.get("skip_reason") else "parallel"


async def parallel_agents_node(state: TradingState) -> TradingState:
    """
    MacroAgent + SentimentAgent + VolProfileAgent in parallel.
    These agents do not depend on each other.

    Note: VolProfileAgent uses yfinance which is NOT thread-safe with shared instances.
    A fresh instance is created per call (zero cost — no persistent state).
    """
    logger.info(f"[Orchestrator] parallel_agents — {state['asset']} {state['direction']}")

    context = {
        "asset":     state["asset"],
        "direction": state["direction"],
    }

    macro_agent = _get_agent("macro")
    sent_agent  = _get_agent("sentiment")
    vp_agent    = VolProfileAgent()  # fresh instance per call (yfinance thread-safety)

    import concurrent.futures
    import functools

    def _run_agent(agent, ctx):
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(agent.run(ctx))
        finally:
            new_loop.close()

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures_list = [
            loop.run_in_executor(executor, functools.partial(_run_agent, macro_agent, context)),
            loop.run_in_executor(executor, functools.partial(_run_agent, sent_agent,  context)),
            loop.run_in_executor(executor, functools.partial(_run_agent, vp_agent,    context)),
        ]
        results = await asyncio.gather(*futures_list, return_exceptions=True)

    agent_scores = dict(state.get("agent_scores", {}))

    names = ["MacroAgent", "SentimentAgent", "VolProfileAgent"]
    for name, result in zip(names, results):
        if isinstance(result, Exception):
            logger.error(f"[Orchestrator] {name} error: {result}")
            agent_scores[name] = {
                "score": 50.0, "confidence": "low",
                "summary": f"Error: {str(result)[:50]}",
                "bull_case": "", "bear_case": "",
            }
        else:
            # SentimentAgent uses SCORE_RANGE (-15,15) → convert to 0-100
            _score = _to_0_100(result.score, (-15, 15)) if name == "SentimentAgent" else result.score
            agent_scores[name] = {
                "score":      _score,
                "confidence": result.confidence,
                "summary":    result.summary,
                "bull_case":  result.bull_case,
                "bear_case":  result.bear_case,
            }
            logger.info(f"[Orchestrator] {name}: {_score:.1f}/100 | {result.confidence}")

    macro_r = results[0] if not isinstance(results[0], Exception) else None
    sent_r  = results[1] if not isinstance(results[1], Exception) else None
    vp_r    = results[2] if not isinstance(results[2], Exception) else None

    # v3: extract Layer 2 flags from VolProfileAgent raw_data
    vp_setup_invalid       = False
    vp_quality_points      = 0
    vp_inside_va           = False
    vp_naked_nvpoc         = None
    vp_blocking_hvn        = None
    vp_tp_reduction        = 0
    vp_path_clear          = True
    vp_market_context      = "neutral"
    vp_trend_healthy       = False
    vp_trend_score         = 0
    vp_poor_high           = None
    vp_poor_low            = None
    vp_composite_poc       = 0.0
    vp_price_vs_composite  = "neutral"
    vp_mtf_confluence      = False
    vp_mtf_zone            = None
    vp_continuation_signal = False
    vp_inversion_signal    = False

    if vp_r and vp_r.raw_data:
        rd = vp_r.raw_data
        vp_setup_invalid       = rd.get("setup_invalid",       False)
        vp_quality_points      = rd.get("quality_points",      0)
        vp_inside_va           = rd.get("inside_va",           False)
        vp_naked_nvpoc         = rd.get("nearest_nvpoc")
        vp_blocking_hvn        = rd.get("blocking_hvn")
        vp_tp_reduction        = rd.get("tp_reduction_pct",    0)
        vp_path_clear          = rd.get("path_clear",          True)
        vp_market_context      = rd.get("market_context",      "neutral")
        vp_trend_healthy       = rd.get("trend_healthy",        False)
        vp_trend_score         = rd.get("trend_score",          0)
        vp_poor_high           = rd.get("poor_high")
        vp_poor_low            = rd.get("poor_low")
        vp_composite_poc       = rd.get("composite_poc",        0.0)
        vp_price_vs_composite  = rd.get("price_vs_composite",  "neutral")
        vp_mtf_confluence      = rd.get("mtf_confluence",       False)
        vp_mtf_zone            = rd.get("mtf_confluence_zone")
        vp_continuation_signal = rd.get("continuation_signal", False)
        vp_inversion_signal    = rd.get("inversion_signal",    False)

    # v3: HIGH_IMPACT_NEWS from SentimentAgent raw_data
    high_impact_news = False
    if sent_r and sent_r.raw_data:
        high_impact_news = sent_r.raw_data.get("high_impact_news", False)

    return {
        **state,
        "agent_scores":        agent_scores,
        "macro_score":         macro_r.score if macro_r else 50.0,
        "sentiment_score":     sent_r.score  if sent_r  else 50.0,
        "volprofile_score":    vp_r.score    if vp_r    else 50.0,
        "vp_setup_invalid":    vp_setup_invalid,
        "vp_quality_points":   vp_quality_points,
        "vp_inside_va":        vp_inside_va,
        "vp_naked_nvpoc":      vp_naked_nvpoc,
        "vp_blocking_hvn":     vp_blocking_hvn,
        "vp_tp_reduction":     vp_tp_reduction,
        "vp_path_clear":       vp_path_clear,
        "vp_market_context":       vp_market_context,
        "vp_trend_healthy":        vp_trend_healthy,
        "vp_trend_score":          vp_trend_score,
        "vp_poor_high":            vp_poor_high,
        "vp_poor_low":             vp_poor_low,
        "vp_composite_poc":        vp_composite_poc,
        "vp_price_vs_composite":   vp_price_vs_composite,
        "vp_mtf_confluence":       vp_mtf_confluence,
        "vp_mtf_zone":             vp_mtf_zone,
        "vp_continuation_signal":  vp_continuation_signal,
        "vp_inversion_signal":     vp_inversion_signal,
        "high_impact_news":        high_impact_news,
        "current_step":            "parallel_agents",
    }


async def technical_agent_node(state: TradingState) -> TradingState:
    """
    TechnicalAgent — sequential because it produces sl_pips/tp_pips/atr_pips
    needed by StrategyRAGAgent and RiskManagerAgent.
    """
    logger.info(f"[Orchestrator] TechnicalAgent — {state['asset']} {state['direction']}")

    context = {"asset": state["asset"], "direction": state["direction"]}
    agent   = _get_agent("technical")
    result  = await agent.run(context)

    agent_scores = dict(state.get("agent_scores", {}))
    agent_scores["TechnicalAgent"] = {
        "score":      result.score,
        "confidence": result.confidence,
        "summary":    result.summary,
        "bull_case":  result.bull_case,
        "bear_case":  result.bear_case,
    }

    # Extract technical levels from raw_data (v3: real prices from S/D zones + VPOC)
    raw      = result.raw_data or {}
    levels_r = (raw.get("agent_detail") or {}).get("levels") or {}
    tp_data  = raw.get("tp_data") or {}

    # SL
    sl_pips   = _safe_float(levels_r.get("sl_pips")) or _safe_float(raw.get("sl_pips"))
    sl_price  = _safe_float(levels_r.get("sl_price"))
    sl_source = levels_r.get("sl_source", "")

    # TP in pips
    tp1_pips = _safe_float(levels_r.get("tp1_pips")) or _safe_float(raw.get("tp1_pips"))
    tp2_pips = _safe_float(levels_r.get("tp2_pips"))
    tp3_pips = _safe_float(levels_r.get("tp3_pips"))

    # Real TP prices — used by risk_manager and trade_executor
    tp1_price = _safe_float(tp_data.get("tp1") or levels_r.get("tp1_price"))
    tp2_price = _safe_float(tp_data.get("tp2") or levels_r.get("tp2_price"))
    tp3_price = _safe_float(tp_data.get("tp3") or levels_r.get("tp3_price"))

    # TP sources (for log and Telegram)
    tp1_source = tp_data.get("tp1_source", levels_r.get("tp1_source", ""))
    tp2_source = tp_data.get("tp2_source", levels_r.get("tp2_source", ""))
    tp3_source = tp_data.get("tp3_source", levels_r.get("tp3_source", ""))

    # Real R:R
    rr1 = _safe_float(tp_data.get("rr1"))
    rr2 = _safe_float(tp_data.get("rr2"))
    rr3 = _safe_float(tp_data.get("rr3"))

    # ATR
    atr_pips = _safe_float(raw.get("atr_pips")) or _safe_float(levels_r.get("atr_pips"))

    instrument_type = raw.get("instrument_type", "")

    # tp_pips = tp1_pips for backward compatibility
    tp_pips = tp1_pips
    if tp_pips <= 0 and sl_pips > 0:
        tp_pips = round(sl_pips * 1.5, 1)
        logger.debug(f"[Orchestrator] TP1 fallback: {sl_pips:.1f} × 1.5 = {tp_pips:.1f}pip")

    logger.info(
        f"[Orchestrator] TechnicalAgent: {result.score:+.1f} | "
        f"SL={sl_pips:.1f}pip [{sl_source}] | "
        f"TP1={tp1_pips:.1f}pip [{tp1_source}] R:R={rr1:.1f} | "
        f"TP2={tp2_pips:.1f}pip [{tp2_source}] R:R={rr2:.1f} | "
        f"ATR={atr_pips:.1f}pip"
    )

    return {
        **state,
        "agent_scores":       agent_scores,
        "sl_pips":            sl_pips,
        "sl_price":           sl_price,
        "sl_source":          sl_source,
        "tp_pips":            tp_pips,
        "tp1_pips":           tp1_pips,
        "tp2_pips":           tp2_pips,
        "tp3_pips":           tp3_pips,
        "tp1_price":          tp1_price,
        "tp2_price":          tp2_price,
        "tp3_price":          tp3_price,
        "tp1_source":         tp1_source,
        "tp2_source":         tp2_source,
        "tp3_source":         tp3_source,
        "rr1":                rr1,
        "rr2":                rr2,
        "rr3":                rr3,
        "atr_pips":           atr_pips,
        "instrument_type":    instrument_type,
        "technical_score":    result.score,
        "_technical_raw_data": result.raw_data or {},
        "current_step":       "technical_agent",
    }


async def correlations_agent_node(state: TradingState) -> TradingState:
    """CorrelationsAgent."""
    logger.info(f"[Orchestrator] CorrelationsAgent — {state['asset']}")

    context = {"asset": state["asset"], "direction": state["direction"]}
    agent   = _get_agent("correlations")
    result  = await agent.run(context)

    agent_scores   = dict(state.get("agent_scores", {}))
    corr_score_100 = _to_0_100(result.score, (-10, 10))

    # v3: CorrelationsAgent is direction-agnostic (100=bullish, 0=bearish,
    # ignores trade direction). For SELL, a bearish correlation SUPPORTS the
    # short → flip the score, otherwise SELL setups are systematically penalized.
    _corr_flipped = False
    if state.get("direction") == "SELL":
        corr_score_100 = 100.0 - corr_score_100
        _corr_flipped  = True

    agent_scores["CorrelationsAgent"] = {
        "score":      corr_score_100,
        "score_raw":  result.score,
        "confidence": result.confidence,
        "summary":    result.summary,
        "bull_case":  result.bull_case,
        "bear_case":  result.bear_case,
    }
    _flip_note = " [SELL-flip]" if _corr_flipped else ""
    logger.info(f"[Orchestrator] CorrelationsAgent: {result.score:+.1f} → {corr_score_100:.1f}/100{_flip_note}")

    corr_veto = result.raw_data.get("divergence_veto", False) if result.raw_data else False
    return {
        **state,
        "agent_scores":      agent_scores,
        "correlations_score": result.score,
        "correlations_veto":  corr_veto,
        "current_step":       "correlations_agent",
    }


def consensus_check_node(state: TradingState) -> TradingState:
    """
    v3: Layer 2 gate + weighted score 0-100.

    Check order:
      1. Volume Profile gate (INVALID if price inside Value Area)
      2. Weighted score < 35 → NO_TRADE
    """
    scores  = state.get("agent_scores", {})
    weights = state.get("weights", {
        "macro": 0.25, "correlations": 0.14, "sentiment": 0.15,
        "volprofile": 0.18, "technical": 0.28,
    })

    # ── Layer 2 Gate v3: Volume Profile inside-VA ─────────────────────────────
    # Price inside Value Area = market at equilibrium, no volumetric edge.
    # Policy v3:
    #   - INDEX in trend (ATR≥0.8, NOT range-day) → HARD BLOCK
    #   - other (forex/crypto, or index in range-day) → NOT blocked; stays marked
    #     as no-edge. Scheduler forces MANUAL (never AUTO) and adds "no edge" tag.
    if state.get("vp_setup_invalid", False):
        asset     = state.get("asset", "?")
        direction = state.get("direction", "?")
        _range_day_eligible = (
            state.get("vp_inside_va", False)
            and state.get("instrument_type", "") == "MT4_CFD"
            and state.get("atr_ratio", 1.0) < RANGE_DAY_ATR_THR
        )
        if asset in VA_HARDBLOCK_INDICES and not _range_day_eligible:
            reason = (
                f"VolProfile Gate [INDEX in trend]: price inside Value Area, "
                f"no edge → hard-block ({asset} {direction})"
            )
            logger.info(f"[Orchestrator] {reason}")
            return {
                **state,
                "consensus_score": 0.0,
                "final_score":     0.0,
                "skip_reason":     reason,
                "current_step":    "consensus_check",
            }
        logger.info(
            f"[Orchestrator] VolProfile inside-VA NOT blocking ({asset} {direction}) "
            f"→ no-edge, forcing MANUAL + tag downstream"
        )

    # Agent name → weight key mapping
    agent_weight_map = {
        "MacroAgent":        "macro",
        "SentimentAgent":    "sentiment",
        "VolProfileAgent":   "volprofile",
        "TechnicalAgent":    "technical",
        "CorrelationsAgent": "correlations",
    }

    # Weighted score 0-100
    weighted_score = 0.0
    for agent_name, weight_key in agent_weight_map.items():
        agent_score = _safe_float(scores.get(agent_name, {}).get("score", 50))
        weight      = weights.get(weight_key, 0.2)
        weighted_score += agent_score * weight
        logger.debug(f"[Orchestrator] {agent_name}: {agent_score:.1f} x {weight:.0%} = {agent_score*weight:.1f}")

    # CorrelationsAgent veto penalty (score < 25 = strongly opposing correlations)
    corr_veto = state.get("correlations_veto", False)
    if corr_veto:
        weighted_score -= VETO_PENALTY
        logger.info(f"[Orchestrator] CorrelationsAgent VETO — penalty -{VETO_PENALTY:.0f} points")

    weighted_score = round(max(0, min(100, weighted_score)), 1)

    _prelim = PRELIMINARY_THRESHOLD

    logger.info(
        f"[Orchestrator] Weighted score: {weighted_score:.1f}/100 "
        f"(NO_TRADE threshold: {_prelim:.0f}) | "
        f"mode: {state.get('mode','?')} | veto: {corr_veto}"
    )

    if weighted_score < _prelim:
        reason = f"Weighted score {weighted_score:.1f}/100 < {_prelim:.0f} — NO_TRADE"
        logger.info(f"[Orchestrator] {reason}")
        return {
            **state,
            "consensus_score": weighted_score,
            "final_score":     weighted_score,
            "skip_reason":     reason,
            "current_step":    "consensus_check",
        }

    # Session-specific manual threshold
    _session  = state.get("session", "LONDON_MID")
    _sess_thr = get_session_manual_threshold(_session)
    if weighted_score < _sess_thr:
        _reason = (
            f"Session threshold [{_session}]: score {weighted_score:.1f} "
            f"< {_sess_thr:.0f} — NO_TRADE"
        )
        logger.info(f"[Orchestrator] {_reason}")
        return {
            **state,
            "consensus_score": weighted_score,
            "final_score":     weighted_score,
            "skip_reason":     _reason,
            "current_step":    "consensus_check",
        }

    return {
        **state,
        "consensus_score": weighted_score,
        "current_step":    "consensus_check",
    }


def should_continue_after_consensus(state: TradingState) -> Literal["strategy_rag", "end"]:
    """LangGraph router — continue or terminate after consensus check."""
    return "end" if state.get("skip_reason") else "strategy_rag"


async def strategy_rag_node(state: TradingState) -> TradingState:
    """
    StrategyRAGAgent v3.
    Uses quality_delta ±2/±1/0 instead of flat bonus.
    Passes technical_raw_data in context to avoid re-computing ICT chain.
    """
    logger.info(f"[Orchestrator] StrategyRAGAgent — {state['asset']} {state['direction']}")

    tech_raw = state.get("_technical_raw_data", {})

    context = {
        "asset":               state["asset"],
        "direction":           state["direction"],
        "technical_raw_data":  tech_raw,
        "sl_pips":             state.get("sl_pips",    0),
        "sl_source":           state.get("sl_source",  ""),
        "tp1_pips":            state.get("tp1_pips",   0),
        "tp2_pips":            state.get("tp2_pips",   0),
        "tp1_source":          state.get("tp1_source", ""),
        "tp2_source":          state.get("tp2_source", ""),
        "tp1_price":           state.get("tp1_price",  0.0),
        "tp2_price":           state.get("tp2_price",  0.0),
        "rr1":                 state.get("rr1",        0),
        "rr2":                 state.get("rr2",        0),
        "atr_pips":            state.get("atr_pips",   0),
        "layer2_raw":          tech_raw.get("layer2_raw", 0),
        "signal_quality":      tech_raw.get("signal_quality",     ""),
        "active_confluences":  tech_raw.get("active_confluences", []),
        "layer2_factors":      tech_raw.get("layer2_factors",     []),
        "instrument_type":     state.get("instrument_type", ""),
    }

    agent  = _get_agent("strategy_rag")
    result = await agent.run(context)

    agent_scores = dict(state.get("agent_scores", {}))
    agent_scores["StrategyRAGAgent"] = {
        "score":      result.score,
        "confidence": result.confidence,
        "summary":    result.summary,
        "bull_case":  result.bull_case,
        "bear_case":  result.bear_case,
    }

    # v3: quality_delta ±2/±1/0 applied to consensus
    raw            = result.raw_data or {}
    quality_delta  = int(raw.get("quality_delta", 0))
    weighted_score = _safe_float(state.get("consensus_score", 0))

    # delta ±2 = ±4 points on consensus (0-100 scale)
    # delta ±1 = ±2 points on consensus
    delta_contribution = quality_delta * 2.0
    final_score = round(min(105, max(-5, weighted_score + delta_contribution)), 1)

    logger.info(
        f"[Orchestrator] StrategyRAGAgent: "
        f"Δ={quality_delta:+d} ({delta_contribution:+.0f}pt) | "
        f"ICT L{raw.get('ict_chain_level',0)}/4 | "
        f"weighted={weighted_score:.1f} → final={final_score:.1f}"
    )

    return {
        **state,
        "agent_scores":    agent_scores,
        "quality_delta":   quality_delta,
        "rag_bonus":       quality_delta,
        "consensus_score": final_score,
        "final_score":     final_score,
        "ict_chain_level": raw.get("ict_chain_level", 0),
        "setup_quality":   raw.get("setup_quality", "GOOD"),
        "is_counter_trend": raw.get("is_counter_trend", False),
        "current_step":    "strategy_rag",
    }


async def risk_manager_node(state: TradingState) -> TradingState:
    """RiskManagerAgent — final APPROVE/BLOCK/MODIFY decision."""
    logger.info(f"[Orchestrator] RiskManagerAgent — {state['asset']} {state['direction']}")

    _asset      = state["asset"]
    _instrument = "IB_MICRO" if _asset in IB_MICRO_ASSETS else "MT4_CFD"

    # v3: compute rr_modifier
    cot_contribution   = state.get("cot_contribution", 0.0)
    high_impact_news   = state.get("high_impact_news", False)
    macro_score_raw    = state.get("macro_score", 50.0)
    macro_contribution = round((macro_score_raw - 50.0) / 50.0 * 0.25, 3)
    rr_modifier        = compute_rr_modifier(
        macro_contribution=macro_contribution,
        cot_contribution=cot_contribution,
        high_impact_news=high_impact_news,
    )

    context = {
        "asset":                _asset,
        "direction":            state["direction"],
        "consensus_score":      state.get("consensus_score", 0),
        "agent_scores":         {k: v["score"] for k, v in state.get("agent_scores", {}).items()},
        "sl_pips":              state.get("sl_pips",     0),
        "sl_price":             state.get("sl_price",    0.0),
        "sl_source":            state.get("sl_source",   ""),
        "tp_pips":              state.get("tp_pips",     0),
        "tp1_pips":             state.get("tp1_pips",    0),
        "tp2_pips":             state.get("tp2_pips",    0),
        "tp3_pips":             state.get("tp3_pips",    0),
        "tp1_price":            state.get("tp1_price",   0.0),
        "tp2_price":            state.get("tp2_price",   0.0),
        "tp3_price":            state.get("tp3_price",   0.0),
        "tp1_source":           state.get("tp1_source",  ""),
        "tp2_source":           state.get("tp2_source",  ""),
        "tp3_source":           state.get("tp3_source",  ""),
        "rr1":                  state.get("rr1",         0.0),
        "rr2":                  state.get("rr2",         0.0),
        "rr3":                  state.get("rr3",         0.0),
        "atr_pips":             state.get("atr_pips",    0),
        "open_positions":       state.get("open_positions", []),
        "current_drawdown_pct": state.get("current_drawdown_pct", 0),
        "instrument":           _instrument,
        "rr_modifier":          rr_modifier,
        "vp_naked_nvpoc":       state.get("vp_naked_nvpoc"),
        "vp_blocking_hvn":      state.get("vp_blocking_hvn"),
        "vp_tp_reduction":      state.get("vp_tp_reduction", 0),
        "vp_path_clear":        state.get("vp_path_clear", True),
        "vp_quality_points":    state.get("vp_quality_points", 0),
        "high_impact_news":     state.get("high_impact_news", False),
        "vp_market_context":      state.get("vp_market_context",      "neutral"),
        "vp_trend_healthy":       state.get("vp_trend_healthy",        False),
        "vp_trend_score":         state.get("vp_trend_score",          0),
        "vp_poor_high":           state.get("vp_poor_high"),
        "vp_poor_low":            state.get("vp_poor_low"),
        "vp_composite_poc":       state.get("vp_composite_poc",        0.0),
        "vp_price_vs_composite":  state.get("vp_price_vs_composite",   "neutral"),
        "vp_mtf_confluence":      state.get("vp_mtf_confluence",       False),
        "vp_mtf_zone":            state.get("vp_mtf_zone"),
        "vp_continuation_signal": state.get("vp_continuation_signal",  False),
        "vp_inversion_signal":    state.get("vp_inversion_signal",     False),
        "atr_ratio":          state.get("atr_ratio",         1.0),
        "vp_inside_va":       state.get("vp_inside_va",      False),
        "ict_chain_level":    state.get("ict_chain_level",   0),
        "is_counter_trend":   state.get("is_counter_trend",  False),
        "setup_quality":      state.get("setup_quality",     "GOOD"),
    }

    agent  = _get_agent("risk_manager")
    result = await agent.run(context)
    raw    = result.raw_data or {}

    logger.info(
        f"[Orchestrator] RiskManagerAgent: {raw.get('decision','?')} | "
        f"Risk ${raw.get('sizing', {}).get('risk_usd', 0):.0f}"
    )

    return {
        **state,
        "risk_decision":    raw.get("decision", "BLOCK"),
        "telegram_message": raw.get("telegram_message", ""),
        "callback_id":      _extract_callback_id(raw.get("telegram_message", "")),
        "sizing":           raw.get("sizing", {}),
        "tp_levels":        raw.get("tp_levels", {}),
        "current_step":     "risk_manager",
    }


def _extract_callback_id(msg: str) -> str:
    """Extract callback_id from Telegram message."""
    import re
    match = re.search(r"ID: ([A-F0-9]{8})", msg)
    return match.group(1) if match else ""


def should_continue_after_risk(state: TradingState) -> Literal["telegram", "end"]:
    """Router — continue only if APPROVE or MODIFY."""
    decision = state.get("risk_decision", "BLOCK")
    if decision == "BLOCK":
        logger.info("[Orchestrator] Trade blocked — end of workflow")
        return "end"
    return "telegram"


async def telegram_dispatch_node(state: TradingState) -> TradingState:
    """
    Prepare the Telegram message for dispatch.
    Actual sending is handled by the scheduler's approval handler
    after wait_for_approval is active, to prevent double-send.
    """
    consensus = state.get("consensus_score", 0)
    is_auto   = abs(consensus) >= AUTO_EXECUTE_THRESHOLD

    logger.info(
        f"[Orchestrator] Telegram dispatch — "
        f"{'AUTO-EXECUTE' if is_auto else 'MANUAL APPROVE'} | "
        f"{state['asset']} {state['direction']}"
    )
    logger.debug(f"[Orchestrator] Telegram message ready for {state['asset']}")

    return {
        **state,
        "current_step":  "telegram_dispatched",
        "trade_executed": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    """
    Build the LangGraph graph with all nodes and edges.
    Returns the compiled graph ready for execution.
    """
    if not LANGGRAPH_AVAILABLE:
        logger.warning("LangGraph not available — using run_sequential()")
        return None

    graph = StateGraph(TradingState)

    graph.add_node("account_state",     account_state_node)
    graph.add_node("regime_detection",  regime_detection_node)
    graph.add_node("direction_lock",    direction_lock_node)
    graph.add_node("parallel_agents",   parallel_agents_node)
    graph.add_node("technical_agent",   technical_agent_node)
    graph.add_node("correlations",      correlations_agent_node)
    graph.add_node("consensus_check",   consensus_check_node)
    graph.add_node("strategy_rag",      strategy_rag_node)
    graph.add_node("risk_manager",      risk_manager_node)
    graph.add_node("telegram_dispatch", telegram_dispatch_node)

    graph.set_entry_point("account_state")
    graph.add_edge("account_state",    "regime_detection")
    graph.add_edge("regime_detection", "direction_lock")

    graph.add_conditional_edges(
        "direction_lock",
        should_continue_after_lock,
        {"parallel": "parallel_agents", "end": END}
    )

    graph.add_edge("parallel_agents", "technical_agent")
    graph.add_edge("technical_agent", "correlations")
    graph.add_edge("correlations",    "consensus_check")

    graph.add_conditional_edges(
        "consensus_check",
        should_continue_after_consensus,
        {"strategy_rag": "strategy_rag", "end": END}
    )
    graph.add_edge("strategy_rag", "risk_manager")

    graph.add_conditional_edges(
        "risk_manager",
        should_continue_after_risk,
        {"telegram": "telegram_dispatch", "end": END}
    )
    graph.add_edge("telegram_dispatch", END)

    return graph.compile()


# Singleton compiled graph — avoids rebuild on every workflow invocation.
# LangGraph compiled graphs are stateless (state is passed per invocation)
# so it is safe to reuse the same compiled instance for all workflows.
_compiled_graph = None


def _get_compiled_graph():
    """Return the compiled graph, building it only on the first call."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# ─────────────────────────────────────────────────────────────────────────────
# Sequential fallback (if LangGraph not available)
# ─────────────────────────────────────────────────────────────────────────────

async def run_sequential(state: TradingState) -> TradingState:
    """
    Run the workflow sequentially without LangGraph.
    Same result — useful as fallback when LangGraph is not installed.
    """
    state = await account_state_node(state)
    state = await regime_detection_node(state)
    state = await direction_lock_node(state)
    if state.get("skip_reason"):
        logger.info(f"[Orchestrator] Direction lock abort: {state['skip_reason']}")
        return state
    state = await parallel_agents_node(state)
    state = await technical_agent_node(state)
    state = await correlations_agent_node(state)
    state = consensus_check_node(state)

    if state.get("skip_reason"):
        logger.info(f"[Orchestrator] Stop: {state['skip_reason']}")
        return state

    state = await strategy_rag_node(state)

    consensus = state.get("consensus_score", 0)
    if abs(consensus) < MANUAL_THRESHOLD:
        state["skip_reason"] = f"Final consensus {consensus:+.1f} < ±{MANUAL_THRESHOLD:.0f}"
        logger.info(f"[Orchestrator] {state['skip_reason']}")
        return state

    state = await risk_manager_node(state)

    if state.get("risk_decision") == "BLOCK":
        return state

    state = await telegram_dispatch_node(state)
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val, default=0.0) -> float:
    """Robustly convert to float — handles Series, None, str, int."""
    if val is None:
        return default
    try:
        import pandas as pd
        if isinstance(val, pd.Series):
            val = val.iloc[0] if len(val) > 0 else default
        return float(val) or default
    except (TypeError, ValueError, IndexError):
        return default


async def run_workflow(asset: str, direction: str, **kwargs) -> TradingState:
    """
    Run the full workflow for an asset/direction.
    Called by the pre-filter when a candidate signal is found.

    Args:
        asset:     e.g. "EURUSD"
        direction: "BUY" | "SELL"
        **kwargs:  open_positions, current_drawdown_pct, etc.
    """
    # Per-asset mutex — prevents duplicate trade in rapid scans
    if asset in _executing_assets:
        logger.warning(
            f"[Orchestrator] {asset} already in active workflow — skip (anti-duplicate mutex)"
        )
        return {
            "asset": asset, "direction": direction,
            "skip_reason": f"Workflow {asset} already active — preventing duplicate trade",
            "trade_executed": False, "error": "", "consensus_score": 0.0,
            "final_score": 0.0, "risk_decision": "BLOCK",
        }

    _executing_assets.add(asset)
    try:
        return await _run_workflow_inner(asset, direction, **kwargs)
    finally:
        _executing_assets.discard(asset)


async def _run_workflow_inner(asset: str, direction: str, **kwargs) -> TradingState:
    """Workflow logic separated from mutex wrapper."""
    initial_state: TradingState = {
        "asset":                asset,
        "direction":            direction,
        "triggered_at":         datetime.now(timezone.utc),
        "open_positions":       kwargs.get("open_positions", []),
        "current_drawdown_pct": kwargs.get("current_drawdown_pct", 0.0),
        "agent_scores":         {},
        "consensus_score":      0.0,
        "final_score":          0.0,
        "sl_pips":              0.0,
        "tp_pips":              0.0,
        "atr_pips":             0.0,
        "mode":                 "intraday",
        "weights":              {},
        "atr_ratio":            1.0,
        "atr_regime":           "normal",
        "size_multiplier":      1.0,
        "cot_extreme":          False,
        "cot_bias":             "neutral",
        "cot_index":            50.0,
        "direction_locked":     True,
        "macro_score":          50.0,
        "sentiment_score":      50.0,
        "volprofile_score":     50.0,
        "technical_score":      50.0,
        "correlations_score":   50.0,
        "correlations_veto":    False,
        "rag_bonus":            0,
        "trade_executed":       False,
        "error":                "",
        "skip_reason":          "",
        "current_step":         "init",
        "sl_source":            "",
        "sl_price":             0.0,
        "tp1_pips":             0.0,
        "tp2_pips":             0.0,
        "tp3_pips":             0.0,
        "tp1_price":            0.0,
        "tp2_price":            0.0,
        "tp3_price":            0.0,
        "tp1_source":           "",
        "tp2_source":           "",
        "tp3_source":           "",
        "rr1":                  0.0,
        "rr2":                  0.0,
        "rr3":                  0.0,
        "instrument_type":      "",
        "quality_delta":        0,
        "ict_chain_level":      0,
        "setup_quality":        "GOOD",
        "_technical_raw_data":  {},
        "vp_market_context":       "neutral",
        "vp_trend_healthy":        False,
        "vp_trend_score":          0,
        "vp_poor_high":            None,
        "vp_poor_low":             None,
        "vp_composite_poc":        0.0,
        "vp_price_vs_composite":   "neutral",
        "vp_mtf_confluence":       False,
        "vp_mtf_zone":             None,
        "vp_continuation_signal":  False,
        "vp_inversion_signal":     False,
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"[Orchestrator] START — {asset} {direction}")
    logger.info(f"{'='*60}")

    compiled_graph = _get_compiled_graph()

    try:
        if compiled_graph:
            final_state = await compiled_graph.ainvoke(initial_state)
        else:
            final_state = await run_sequential(initial_state)
    except Exception as e:
        logger.error(f"[Orchestrator] Workflow error {asset} {direction}: {e}")
        final_state = {**initial_state, "error": str(e)}

    consensus = final_state.get("consensus_score", 0)
    decision  = final_state.get("risk_decision", "N/A")
    skip      = final_state.get("skip_reason", "")

    if skip:
        logger.info(f"[Orchestrator] END — {asset} {direction} | SKIP: {skip}")
    else:
        logger.info(
            f"[Orchestrator] END — {asset} {direction} | "
            f"consensus={consensus:+.1f} | decision={decision}"
        )

    return final_state
