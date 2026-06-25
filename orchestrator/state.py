"""
orchestrator/state.py
TradingState — shared object across all LangGraph nodes.
Travels through the graph accumulating results agent by agent.

v2.0 — added fields for:
  - Regime detection (mode, weights, ATR)
  - Direction lock (cot_extreme, direction_locked, abort_reason)
  - Score 0-100 per agent (replaces ± in consensus)
  - VolProfileAgent separate from COT
  - Size multiplier for intraday_reduced
"""

from __future__ import annotations
from typing import TypedDict, Optional
from datetime import datetime


class AgentScoreEntry(TypedDict):
    score:      float
    confidence: str    # high / medium / low
    summary:    str
    bull_case:  str
    bear_case:  str


class SizingInfo(TypedDict):
    lot_size:           float
    risk_usd:           float
    risk_pct:           float
    instrument:         str    # MT4_CFD | IB_MICRO
    risk_elevated:      bool
    risk_elevated_note: str
    modify_reason:      str


class TPLevels(TypedDict):
    type:       str    # MULTI_TP | SINGLE_TP
    sl_pips:    float
    tp1_pips:   float  # MULTI_TP
    tp2_pips:   float
    tp3_pips:   float
    tp_pips:    float  # SINGLE_TP
    rr_ratio:   float


class TradingState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────────
    asset:                  str       # e.g. "EURUSD"
    direction:              str       # "BUY" | "SELL"
    triggered_at:           datetime  # pre-filter activation timestamp
    session:                str       # "LONDON_OPEN" | "NY_OPEN" | etc.
    kill_zone_active:       bool

    # ── Pre-filter ───────────────────────────────────────────────────────────
    prefilter_passed:           bool
    prefilter_score:            int   # 0-6 criteria satisfied
    prefilter_reason:           str
    prefilter_ict_score:        int   # v2: 0-12 ICT score from pre-filter
    prefilter_suggested_mode:   str   # v2: 'scalping'|'intraday'|'swing' suggested

    # ── Account state (from IB Gateway or static) ────────────────────────────
    equity:                 float
    open_positions:         list      # [{"asset": str, "direction": str, ...}]
    current_drawdown_pct:   float

    # ── v2: Regime detection (populated by regime_detection_node) ─────────────
    mode:                   str       # 'intraday' | 'intraday_reduced' | 'swing'
    weights:                dict      # {'macro': 0.25, 'technical': 0.28, ...}
    atr_ratio:              float     # current H1 ATR / 20-bar average
    atr_regime:             str       # 'flat' | 'normal' | 'explosive'
    size_multiplier:        float     # 1.0 = full size, 0.5 = half (intraday_reduced)

    # ── v2: Direction lock (populated by direction_lock_node) ─────────────────
    cot_extreme:            bool      # True if COT Index >80 or <20
    cot_bias:               str       # 'bullish' | 'bearish' | 'neutral'
    cot_index:              float     # 0-100 (26-week formula)
    direction_locked:       bool      # True = direction confirmed, False = abort
    abort_reason:           Optional[str]  # abort reason if direction_locked=False

    # ── Agent results — v1 (kept for compatibility) ───────────────────────────
    agent_scores:           dict[str, AgentScoreEntry]
    consensus_score:        float     # v2: weighted score 0-100 of 5 main agents
    consensus_level:        str       # "NO_TRADE" | "MANUAL" | "AUTO_EXECUTE"

    # ── v2: Individual 0-100 scores per agent ────────────────────────────────
    # Each field corresponds to agent_score() result from score_converter.py
    macro_score:            float     # 0-100 MacroAgent (includes COT)
    sentiment_score:        float     # 0-100 SentimentAgent
    volprofile_score:       float     # 0-100 VolProfileAgent (v2: replaces CotVolumeAgent)
    technical_score:        float     # 0-100 TechnicalAgent ICT
    correlations_score:     float     # 0-100 CorrelationsAgent
    correlations_veto:      bool      # True if score < 25 → -10 points on final
    rag_bonus:              int       # 0 or +5 (flat bonus from StrategyRAGAgent)

    # ── v2: Aggregated final score ────────────────────────────────────────────
    final_score:            float     # 0-105: weighted_score + rag_bonus
    # Thresholds: <35=NO_TRADE, 35-55=BLOCK, 56-74=MANUAL 50%size,
    #             75-84=MANUAL 100%size, >=85=AUTO_EXECUTE

    # ── Technical data (from TechnicalAgent raw_data) ────────────────────────
    sl_pips:                float
    sl_price:               float     # v3: absolute SL price
    sl_source:              str       # v3: SL source (BB_HIGH, OB_LOW, etc.)
    tp_pips:                float     # = tp1_pips (compatibility)
    tp1_pips:               float     # v3: TP1 in pips (partial close target)
    tp2_pips:               float     # v3: TP2 in pips (runner target)
    tp3_pips:               float     # v3: TP3 in pips (extension)
    tp1_price:              float     # v3: absolute TP1 price
    tp2_price:              float     # v3: absolute TP2 price
    tp3_price:              float     # v3: absolute TP3 price
    tp1_source:             str       # v3: TP1 source (SUPPLY_ZONE, VPOC_M30, etc.)
    tp2_source:             str       # v3: TP2 source
    tp3_source:             str       # v3: TP3 source
    rr1:                    float     # v3: R:R for TP1
    rr2:                    float     # v3: R:R for TP2
    rr3:                    float     # v3: R:R for TP3
    atr_pips:               float
    entry_price:            float
    instrument_type:        str       # v3: MT4_CFD | IB_MICRO

    # ── v3: RAG data and setup quality ───────────────────────────────────────
    quality_delta:          int       # v3: -2/-1/0/+1/+2 from StrategyRAGAgent
    ict_chain_level:        int       # v3: 0-4 ICT chain level
    setup_quality:          str       # v3: INVALID/WEAK/GOOD/STRONG
    _technical_raw_data:    dict      # v3: TechnicalAgent raw_data (in-memory)

    # ── v3: Macro and COT contributions ──────────────────────────────────────
    cot_contribution:       float     # v3: COT contribution to rr_modifier
    high_impact_news:       bool      # v3: True if HIGH_IMPACT news imminent
    vp_naked_nvpoc:         Optional[float]   # v3: Naked VPOC from VolProfileAgent
    vp_blocking_hvn:        bool              # v3: HVN blocks the TP path
    vp_tp_reduction:        float             # v3: TP reduction % due to HVN
    vp_path_clear:          bool              # v3: path to TP is clear
    vp_quality_points:      int               # v3: VP quality points (0-8)
    # v3+: Group A new volumetric concepts
    vp_market_context:      str               # range | trending | neutral
    vp_trend_healthy:       bool              # True if trend healthy (score ≥2/3)
    vp_trend_score:         int               # 0-3
    vp_poor_high:           Optional[float]   # Active Poor High (weak TP)
    vp_poor_low:            Optional[float]   # Active Poor Low (weak TP)
    vp_composite_poc:       float             # multi-session composite POC
    vp_price_vs_composite:  str               # discount | premium | neutral
    vp_mtf_confluence:      bool              # True if H1/H4/D1 POCs aligned
    vp_mtf_zone:            Optional[float]   # multi-TF confluence price zone
    vp_continuation_signal: bool              # breakout with volume → continuation
    vp_inversion_signal:    bool              # VA re-entry → inversion
    vp_setup_invalid:       bool              # True = price inside Value Area (no edge)
    vp_inside_va:           bool              # True = price inside Value Area (range-day trigger)

    # ── Risk decision ─────────────────────────────────────────────────────────
    risk_decision:          str       # "APPROVE" | "BLOCK" | "MODIFY"
    telegram_message:       str
    callback_id:            str
    sizing:                 SizingInfo
    tp_levels:              TPLevels

    # ── Execution ─────────────────────────────────────────────────────────────
    trade_executed:         bool
    trade_id:               str       # broker order ID
    broker_used:            str       # "MT4" | "IB"
    execution_price:        float
    execution_timestamp:    datetime

    # ── Internal state ────────────────────────────────────────────────────────
    error:                  str
    skip_reason:            str
    current_step:           str
