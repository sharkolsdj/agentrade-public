"""
agents/risk_manager_agent.py
RiskManagerAgent — Agent 7
Output: APPROVE / BLOCK / MODIFY

Last gate before execution. Makes the final operational decision:
  - Verifies risk conditions (drawdown, open positions, correlations)
  - Computes position sizing based on equity and configured risk %
  - Structures TP/SL levels per broker (MT4 MULTI_TP vs IB SINGLE_TP)
  - Produces formatted Telegram notification with APPROVE/BLOCK/MODIFY
  - Fully deterministic in v3 (LLM removed — see paper Section 3.5c)

Does NOT produce a numeric score — produces an operational decision.
The consensus score is read from the context passed by the orchestrator.

Configuration via .env:
  RISK_PER_TRADE_PCT   = 1.0   # % equity per trade (default 1%)
  MAX_DRAWDOWN_PCT     = 10.0  # maximum tolerated drawdown (%)
  MAX_OPEN_POSITIONS   = 3     # maximum simultaneous open positions
  MANUAL_THRESHOLD     = 60.0  # minimum consensus for MANUAL APPROVE
  AUTO_EXECUTE_THRESHOLD = 85.0 # minimum consensus for AUTO-EXECUTE

Equity:
  PAPER_EQUITY_USD is read from env (PAPER_EQUITY_USD).
  For live trading, connect to IB Gateway via ib_insync:
    from ib_insync import IB
    ib = IB(); ib.connect(host, port, clientId=1)
    equity = ib.accountValues()  # → 'NetLiquidation'
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from agents.base_agent import BaseAgent, AgentResult
from utils.lot_spec import normalize_lot

# ─────────────────────────────────────────────────────────────────────────────
# Equity — read from environment (paper account or live)
# For live trading: replace with IB Gateway API call (ib_insync)
# ─────────────────────────────────────────────────────────────────────────────
PAPER_EQUITY_USD: float = float(os.getenv("PAPER_EQUITY_USD", "10000.0"))

# Risk thresholds (overridable via .env)
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT",   "1.0"))
MAX_DRAWDOWN_PCT   = float(os.getenv("MAX_DRAWDOWN_PCT",     "10.0"))
MAX_OPEN_MT4       = int(os.getenv("MAX_OPEN_MT4",           "8"))
MAX_OPEN_IB        = int(os.getenv("MAX_OPEN_IB",            "3"))

# Price display precision per asset (for Telegram messages)
_PRICE_DIGITS: dict[str, int] = {
    "EURUSD": 5, "GBPUSD": 5, "USDCAD": 5, "EURAUD": 5,
    "EURGBP": 5, "EURCHF": 5, "6E":     5,
    "USDJPY": 3, "GBPJPY": 3, "NZDJPY": 3,
    "XAUUSD": 2, "XAGUSD": 2, "MGC":    2, "MCL":  2,
    "MES":    2, "NAS100": 2, "RTY":    2,
    "BTCUSD": 1, "ETHUSD": 2,
    "SP500":  2, "USOUSD": 2,
}


def _fmt_price(asset: str, price: float) -> str:
    """Format a price with the correct decimal places for the asset."""
    digits = _PRICE_DIGITS.get(asset.upper(), 5)
    return f"{price:.{digits}f}"


# ── Consensus thresholds — three levels ──────────────────────────────────────
# |consensus| >= AUTO_EXECUTE_THRESHOLD → automatic trade (notification only)
# |consensus| >= MANUAL_THRESHOLD       → APPROVE/BLOCK/MODIFY notification
# |consensus| <  MANUAL_THRESHOLD       → NO_TRADE, no action
AUTO_EXECUTE_THRESHOLD = float(os.getenv("AUTO_EXECUTE_THRESHOLD", "85.0"))
MANUAL_THRESHOLD       = float(os.getenv("MANUAL_THRESHOLD",       "60.0"))

# IB Micro futures risk cap (USD per contract) — two levels:
#  - WARN (soft): trade passes but Telegram shows warning at the top
#  - MAX  (hard): trade is BLOCKED (1 contract is not fractional)
IB_MICRO_WARN_RISK_USD = float(os.getenv("IB_MICRO_WARN_RISK_USD", "300.0"))
IB_MICRO_MAX_RISK_USD  = float(os.getenv("IB_MICRO_MAX_RISK_USD",  "550.0"))

# ─────────────────────────────────────────────────────────────────────────────
# Pip value per standard lot (USD per pip per 1 lot)
# MT4 CFD: used for fractional lot sizing
# ─────────────────────────────────────────────────────────────────────────────
PIP_VALUE_PER_LOT: dict[str, float] = {
    # ── MT4 CFD — USD per pip per 1 standard lot ──────────────────────────
    "EURUSD": 10.0,  "GBPUSD": 10.0,
    "USDCAD": 7.5,
    "EURAUD": 6.5,
    "EURGBP": 12.5,
    "EURCHF": 11.0,
    "USDJPY": 6.5,   "GBPJPY": 6.5,   "NZDJPY": 6.5,
    "XAUUSD": 1.0,   # pip = $0.01, contract = 100oz → $1/pip/lot
    "XAGUSD": 50.0,
    "BTCUSD": 1.0,
    "ETHUSD": 0.10,
    # ── IB Micro Futures — USD per tick per 1 contract ────────────────────
    "MES":    1.25,
    "RTY":    0.50,
    "MGC":    1.00,
    "MCL":    1.00,
    "NAS100": 0.25,
    "6E":     0.625,
    # ── Additional forex ──
    "NZDUSD": 10.0, "AUDJPY": 6.5, "CHFJPY": 6.5,
    # ── Index CFDs (pip 0.25) ──
    "DJ30": 0.25, "US2000": 0.25, "GER40": 0.27,
    "SP500": 2.50,
    # ── Oil CFD ──
    "USOUSD": 1.0,
    # Default
    "DEFAULT": 10.0,
}

# Pip size per asset
PIP_SIZE: dict[str, float] = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "USDCAD": 0.0001,
    "EURAUD": 0.0001, "EURGBP": 0.0001, "EURCHF": 0.0001,
    "6E":     0.00005,
    "USDJPY": 0.01,   "GBPJPY": 0.01,   "NZDJPY": 0.01,
    "XAUUSD": 0.01,   "MGC":    0.10,
    "XAGUSD": 0.010,
    "BTCUSD": 1.0,    "ETHUSD": 0.10,
    "MES":    0.25,   "RTY":    0.10,
    "MCL":    0.010,  "NAS100": 0.25,
    "NZDUSD": 0.0001, "AUDJPY": 0.01,  "CHFJPY": 0.01,
    "GER40":  0.25,   "US2000": 0.25,  "DJ30":   0.25,  "SP500":  0.25,
    "USOUSD": 0.01,
}

# Minimum SL per asset — prevents stop hunt from spread and market noise.
# Values computed from: 1× typical spread + minimum historical M15 ATR.
# If TechnicalAgent produces an SL below these values, the trade is blocked.
# Rationale: SL < MIN_SL → entry is inside market noise → prevents duplicate
# signals on the same asset across consecutive scans.
# Values in pips (same unit as PIP_SIZE above).
MIN_SL_PIPS: dict[str, float] = {
    # ── Forex CFD (pip = 0.0001) ──────────────────────────────────────────────
    "EURUSD": 10.0,
    "GBPUSD": 12.0,
    "USDCAD": 10.0,
    "EURAUD": 15.0,
    "EURGBP": 15.0,
    "EURCHF": 12.0,
    # ── EUR/USD futures 6E (pip = 0.00005) ───────────────────────────────────
    "6E":     8.0,
    # ── JPY pairs (pip = 0.01) ────────────────────────────────────────────────
    "USDJPY": 12.0,
    "GBPJPY": 20.0,
    "NZDJPY": 15.0,
    # ── Metals (pip = 0.10 for XAUUSD/MGC, 0.010 for XAGUSD) ─────────────────
    "XAUUSD": 150.0,
    "MGC":    50.0,
    "XAGUSD": 30.0,
    # ── Crypto ────────────────────────────────────────────────────────────────
    "BTCUSD": 300.0,
    "ETHUSD": 50.0,
    # ── IB Micro Futures ──────────────────────────────────────────────────────
    "MES":    8.0,
    "MCL":    40.0,
    "NAS100": 30.0,
    "NZDUSD": 12.0, "AUDJPY": 18.0, "CHFJPY": 18.0,
    "GER40":  30.0, "US2000": 30.0, "DJ30":   30.0, "SP500": 30.0,
    "USOUSD": 40.0,
}

# Instrument routing per broker
INSTRUMENT_TYPE: dict[str, str] = {
    # MT4 CFD → fractional lots, partial-close lifecycle
    "EURUSD": "MT4_CFD",  "GBPUSD": "MT4_CFD",  "USDJPY": "MT4_CFD",
    "GBPJPY": "MT4_CFD",  "XAUUSD": "MT4_CFD",  "BTCUSD": "MT4_CFD",
    "ETHUSD": "MT4_CFD",  "EURAUD": "MT4_CFD",  "USDCAD": "MT4_CFD",
    "EURGBP": "MT4_CFD",  "NZDJPY": "MT4_CFD",  "EURCHF": "MT4_CFD",
    "XAGUSD": "MT4_CFD",
    # IB Micro Futures → 1 contract, single-TP lifecycle
    "MGC":    "IB_MICRO", "MES":    "IB_MICRO", "RTY":    "IB_MICRO",
    "MCL":    "IB_MICRO", "6E":     "IB_MICRO", "NAS100": "MT4_CFD",
    "NZDUSD": "MT4_CFD",  "AUDJPY": "MT4_CFD",  "CHFJPY": "MT4_CFD",
    "GER40":  "MT4_CFD",  "US2000": "MT4_CFD",  "DJ30":   "MT4_CFD",
    "SP500":  "MT4_CFD",  "USOUSD": "MT4_CFD",
}

# Correlated asset groups — avoids over-exposure on the same underlying
CORRELATED_GROUPS: list[set] = [
    {"EURUSD", "6E", "EURGBP", "EURCHF"},
    {"GBPUSD", "GBPJPY", "EURGBP"},
    {"USDJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY"},
    {"AUDUSD", "NZDUSD"},
    {"NAS100", "US2000", "DJ30", "SP500", "MES"},
    {"XAUUSD", "MGC", "XAGUSD"},
    {"BTCUSD", "ETHUSD"},
    {"MES", "RTY", "NAS100"},
    {"MCL", "USDCAD", "USOUSD"},
]

# Possible decisions
DECISION_APPROVE = "APPROVE"
DECISION_BLOCK   = "BLOCK"
DECISION_MODIFY  = "MODIFY"


# ─────────────────────────────────────────────────────────────────────────────
class RiskManagerAgent(BaseAgent):
    """
    Agent 7 — RiskManagerAgent.

    Last gate before trade execution.
    Produces: APPROVE / BLOCK / MODIFY with full operational parameters.

    Note: Internal implementation is intentionally omitted in this public
    showcase version. The interface, data flow, and design decisions are
    documented in the paper (see README for link).

    Published because policy code SHOULD be auditable — see paper Section 3.5c.

    Phases:
      1. Deterministic Risk Gate (drawdown, positions, correlations, threshold)
      2. Position Sizing (equity-based, 1% risk per trade)
      3. Structured TP/SL per broker (MULTI_TP MT4 vs SINGLE_TP IB)
      4. Final decision + rationale (fully deterministic in v3)
      5. Formatted Telegram message
    """

    AGENT_NAME  = "RiskManagerAgent"
    MODEL       = "deterministic_v3"
    SCORE_RANGE = (0, 0)   # does not produce a numeric score

    def __init__(self):
        super().__init__()

    # ─── collect_data ─────────────────────────────────────────────────────────

    async def collect_data(self, context: dict) -> dict:
        """
        Read account state and consensus from context.

        Args:
            context: {
                "asset": str,
                "direction": str,
                "consensus_score": float,          # total from 6 agents
                "agent_scores": dict,              # per-agent breakdown
                "sl_pips": float,                  # from TechnicalAgent raw_data
                "tp_pips": float,                  # TP1 from TechnicalAgent
                "open_positions": list[dict],      # current open positions
                "current_drawdown_pct": float,     # current drawdown %
                "atr_pips": float,                 # M15 ATR in pips (optional)
            }
        Returns:
            dict with account_state + trade_params
        """
        asset     = str(context.get("asset", "EURUSD")).upper()
        direction = str(context.get("direction", "BUY")).upper()

        equity = PAPER_EQUITY_USD

        consensus_score  = float(context.get("consensus_score", 0))
        agent_scores     = context.get("agent_scores", {})
        sl_pips          = float(context.get("sl_pips", 0))
        tp_pips          = float(context.get("tp_pips", 0))
        atr_pips         = float(context.get("atr_pips", 0))
        open_positions   = context.get("open_positions", [])
        current_drawdown = float(context.get("current_drawdown_pct", 0))

        if sl_pips <= 0 and atr_pips > 0:
            sl_pips = atr_pips * 1.5
            logger.info(f"[{self.name}] SL not available — using ATR proxy: {sl_pips:.1f}pip")

        if tp_pips <= 0 and sl_pips > 0:
            tp_pips = round(sl_pips * 2.0, 1)
            logger.info(f"[{self.name}] TP=0 — fallback 2.0×SL: {tp_pips:.1f}pip")

        instrument_type = INSTRUMENT_TYPE.get(asset, "MT4_CFD")
        pip_size        = PIP_SIZE.get(asset, 0.0001)
        pip_val         = PIP_VALUE_PER_LOT.get(asset, PIP_VALUE_PER_LOT["DEFAULT"])

        return {
            "asset":              asset,
            "direction":          direction,
            "equity":             equity,
            "consensus_score":    consensus_score,
            "agent_scores":       agent_scores,
            "sl_pips":            sl_pips,
            "sl_price":           float(context.get("sl_price",  0.0)),
            "sl_source":          str(context.get("sl_source",   "")),
            "tp_pips":            tp_pips,
            "tp1_pips":           float(context.get("tp1_pips",  0.0)),
            "tp2_pips":           float(context.get("tp2_pips",  0.0)),
            "tp3_pips":           float(context.get("tp3_pips",  0.0)),
            "tp1_price":          float(context.get("tp1_price", 0.0)),
            "tp2_price":          float(context.get("tp2_price", 0.0)),
            "tp3_price":          float(context.get("tp3_price", 0.0)),
            "tp1_source":         str(context.get("tp1_source",  "")),
            "tp2_source":         str(context.get("tp2_source",  "")),
            "tp3_source":         str(context.get("tp3_source",  "")),
            "rr1":                float(context.get("rr1", 0.0)),
            "rr2":                float(context.get("rr2", 0.0)),
            "rr3":                float(context.get("rr3", 0.0)),
            "atr_pips":           atr_pips,
            "open_positions":     open_positions,
            "current_drawdown":   current_drawdown,
            "instrument_type":    instrument_type,
            "pip_size":           pip_size,
            "pip_value_per_lot":  pip_val,
            # VP Group A context
            "vp_market_context":      str(context.get("vp_market_context",      "neutral")),
            "vp_trend_healthy":       bool(context.get("vp_trend_healthy",       False)),
            "vp_trend_score":         int(context.get("vp_trend_score",          0)),
            "vp_poor_high":           context.get("vp_poor_high"),
            "vp_poor_low":            context.get("vp_poor_low"),
            "vp_mtf_confluence":      bool(context.get("vp_mtf_confluence",      False)),
            "vp_price_vs_composite":  str(context.get("vp_price_vs_composite",   "neutral")),
            "vp_continuation_signal": bool(context.get("vp_continuation_signal", False)),
        }

    # ─── analyze ──────────────────────────────────────────────────────────────

    async def analyze(self, data: dict, context: dict) -> dict:
        """
        Full pipeline:
          1. Deterministic Risk Gate (drawdown, positions, consensus threshold)
          2. IB_MICRO cap check
          3. Position Sizing (MT4: fractional lot | IB: 1 contract)
          4. Structured TP/SL per broker
          5. Correlation check
          6. Final decision (fully deterministic in v3)
          7. Telegram message (auto-execute if ≥AUTO threshold, else manual)
        """
        asset          = data["asset"]
        direction      = data["direction"]
        equity         = data["equity"]
        consensus      = data["consensus_score"]
        sl_pips        = data["sl_pips"]
        instrument     = data["instrument_type"]
        open_positions = data["open_positions"]
        drawdown       = data["current_drawdown"]
        agent_scores   = data.get("agent_scores", {})

        # ── Phase 0: Deterministic veto for critical agents ───────────────────
        tech_score = float(agent_scores.get("TechnicalAgent", 100))
        vp_score   = float(agent_scores.get("VolProfileAgent", 100))

        if tech_score < 48:
            block_reason = (
                f"TechnicalAgent {tech_score:.1f}/100 < 48 — "
                f"ICT setup invalid (weak or absent BOS/OB/FVG)"
            )
            logger.info(f"[{self.name}] Deterministic BLOCK: {block_reason}")
            telegram_msg = self._format_telegram_block(asset, direction, consensus, block_reason)
            return self._build_output(
                decision=DECISION_BLOCK, asset=asset, direction=direction,
                consensus=consensus, reason=block_reason,
                telegram_msg=telegram_msg, sizing={}, tp_levels={},
            )

        # ── Phase 3: Range-day mode detection ────────────────────────────────
        _atr_ratio       = float(context.get("atr_ratio",       1.0))
        _vp_inside_va    = bool(context.get("vp_inside_va",     False))
        _ict_chain_level = int(context.get("ict_chain_level",   0))

        _RANGE_DAY_ATR_THR  = 0.8
        _RANGE_DAY_MIN_CONS = 65.0
        _RANGE_DAY_MIN_ICT  = 2
        _RANGE_DAY_SL_MULT  = 0.7
        _RANGE_DAY_TP_RR    = 1.8

        range_day_mode = (
            instrument == "MT4_CFD"          and
            _atr_ratio  < _RANGE_DAY_ATR_THR and
            _vp_inside_va
        )

        if range_day_mode:
            logger.info(
                f"[{self.name}] 🔵 RANGE-DAY MODE — {asset} {direction} | "
                f"ATR={_atr_ratio:.2f}x, inside_VA=True, ICT_L{_ict_chain_level}"
            )
            if abs(consensus) < _RANGE_DAY_MIN_CONS:
                block_reason = (
                    f"Range-day: consensus {consensus:.1f} < {_RANGE_DAY_MIN_CONS:.0f} "
                    f"(ATR={_atr_ratio:.2f}x, inside VA)"
                )
                logger.info(f"[{self.name}] Deterministic BLOCK: {block_reason}")
                telegram_msg = self._format_telegram_block(asset, direction, consensus, block_reason)
                return self._build_output(
                    decision=DECISION_BLOCK, asset=asset, direction=direction,
                    consensus=consensus, reason=block_reason,
                    telegram_msg=telegram_msg, sizing={}, tp_levels={},
                )
            if _ict_chain_level < _RANGE_DAY_MIN_ICT:
                block_reason = (
                    f"Range-day: ICT L{_ict_chain_level} < L{_RANGE_DAY_MIN_ICT} "
                    f"(BSL/SSL not identified — swing confirmation required)"
                )
                logger.info(f"[{self.name}] Deterministic BLOCK: {block_reason}")
                telegram_msg = self._format_telegram_block(asset, direction, consensus, block_reason)
                return self._build_output(
                    decision=DECISION_BLOCK, asset=asset, direction=direction,
                    consensus=consensus, reason=block_reason,
                    telegram_msg=telegram_msg, sizing={}, tp_levels={},
                )
            _atr_pips_val = float(data.get("atr_pips", 0))
            if _atr_pips_val > 0:
                _sl_rd  = round(_atr_pips_val * _RANGE_DAY_SL_MULT, 1)
                _sl_min = MIN_SL_PIPS.get(asset, 5.0)
                sl_pips = max(_sl_rd, _sl_min)
                logger.info(
                    f"[{self.name}] Range-day SL: 0.7×ATR={_sl_rd:.1f}pip "
                    f"(floor={_sl_min:.0f}) → sl_pips={sl_pips:.1f}"
                )
            data["range_day_mode"]  = True
            data["range_day_tp_rr"] = _RANGE_DAY_TP_RR

        else:
            if vp_score < 45:
                block_reason = (
                    f"VolProfileAgent {vp_score:.1f}/100 < 45 — "
                    f"price inside Value Area, no volumetric edge"
                )
                logger.info(f"[{self.name}] Deterministic BLOCK: {block_reason}")
                telegram_msg = self._format_telegram_block(asset, direction, consensus, block_reason)
                return self._build_output(
                    decision=DECISION_BLOCK, asset=asset, direction=direction,
                    consensus=consensus, reason=block_reason,
                    telegram_msg=telegram_msg, sizing={}, tp_levels={},
                )

        # ── Phase 1: Deterministic Risk Gate ─────────────────────────────────
        block_reason = self._risk_gate(
            asset, direction, consensus, sl_pips, open_positions, drawdown
        )

        if block_reason:
            logger.info(f"[{self.name}] Deterministic BLOCK: {block_reason}")
            telegram_msg = self._format_telegram_block(asset, direction, consensus, block_reason)
            return self._build_output(
                decision=DECISION_BLOCK,
                asset=asset, direction=direction, consensus=consensus,
                reason=block_reason, telegram_msg=telegram_msg,
                sizing={}, tp_levels={},
            )

        # ── Phase 2: Position Sizing ──────────────────────────────────────────
        sizing = self._calculate_sizing(
            asset, direction, equity, sl_pips, instrument, data["pip_value_per_lot"]
        )

        if sizing.get("block_reason"):
            block_reason = sizing["block_reason"]
            logger.info(f"[{self.name}] Sizing BLOCK: {block_reason}")
            telegram_msg = self._format_telegram_block(asset, direction, consensus, block_reason)
            return self._build_output(
                decision="BLOCK", asset=asset, direction=direction,
                consensus=consensus, reason=block_reason,
                telegram_msg=telegram_msg, sizing=sizing, tp_levels={},
            )

        # ── Phase 3: Structured TP/SL ─────────────────────────────────────────
        tp_levels = self._calculate_tp_levels(
            direction        = direction,
            sl_pips          = sl_pips,
            instrument       = instrument,
            lot_size         = sizing["lot_size"],
            tp1_price        = float(data.get("tp1_price",  0.0)),
            tp2_price        = float(data.get("tp2_price",  0.0)),
            tp3_price        = float(data.get("tp3_price",  0.0)),
            tp1_pips         = float(data.get("tp1_pips",   0.0)),
            tp2_pips         = float(data.get("tp2_pips",   0.0)),
            tp3_pips         = float(data.get("tp3_pips",   0.0)),
            tp1_source       = str(data.get("tp1_source",  "")),
            tp2_source       = str(data.get("tp2_source",  "")),
            tp3_source       = str(data.get("tp3_source",  "")),
            rr1              = float(data.get("rr1", 0.0)),
            rr2              = float(data.get("rr2", 0.0)),
            rr3              = float(data.get("rr3", 0.0)),
            range_day_mode   = data.get("range_day_mode",   False),
            range_day_tp_rr  = data.get("range_day_tp_rr",  1.8),
        )

        ib_risk_elevated = (
            instrument == "IB_MICRO"
            and sizing["risk_usd"] > IB_MICRO_MAX_RISK_USD
        )
        if ib_risk_elevated:
            sizing["risk_elevated"]      = True
            sizing["risk_elevated_note"] = (
                f"⚠️ Elevated risk: ${sizing['risk_usd']:.0f}/contract "
                f"(recommended threshold ${IB_MICRO_MAX_RISK_USD:.0f}) — "
                f"you decide if the setup justifies it"
            )

        logger.info(
            f"[{self.name}] {asset} {direction} | "
            f"Sizing: {float(sizing['lot_size']):.2f} {instrument} | "
            f"Risk: ${sizing['risk_usd']:.0f} ({sizing['risk_pct']:.1f}%)"
        )

        # ── Phase 4: Correlation risk check ───────────────────────────────────
        corr_warning = self._correlation_check(asset, open_positions)
        if corr_warning and corr_warning.startswith("BLOCK:"):
            logger.info(f"[{self.name}] Correlation BLOCK: {corr_warning}")
            telegram_msg = self._format_telegram_block(asset, direction, consensus, corr_warning)
            return self._build_output(
                decision=DECISION_BLOCK,
                asset=asset, direction=direction, consensus=consensus,
                reason=corr_warning, telegram_msg=telegram_msg,
                sizing={}, tp_levels={},
            )

        # ── Phase 5: Final decision — fully deterministic ────────────────────
        rr_modifier = float(data.get("rr_modifier", 1.0))
        rr_modifier = max(0.50, min(1.00, rr_modifier))

        if "tp_pips" in tp_levels:
            tp_levels["tp_pips"]  = round(tp_levels["tp_pips"] * rr_modifier, 1)
            tp_levels["rr_ratio"] = round(tp_levels["tp_pips"] / sl_pips, 2) if sl_pips > 0 else 1.5
        if "partial_pips" in tp_levels:
            tp_levels["partial_pips"] = round(tp_levels["partial_pips"] * rr_modifier, 1)
            tp_levels["runner_pips"]  = round(tp_levels["runner_pips"]  * rr_modifier, 1)
            tp_levels["rr_partial"]   = round(tp_levels["partial_pips"] / sl_pips, 2) if sl_pips > 0 else 2.0
            tp_levels["rr_runner"]    = round(tp_levels["runner_pips"]  / sl_pips, 2) if sl_pips > 0 else 3.0

        final_rr = tp_levels.get("rr_ratio") or tp_levels.get("rr_partial", 0)
        if final_rr > 0 and final_rr < 1.5:
            block_rr = (
                f"R:R {final_rr:.2f} < 1.5 after rr_modifier={rr_modifier:.2f} — "
                f"setup does not meet minimum R:R requirement"
            )
            logger.info(f"[{self.name}] R:R minimum BLOCK: {block_rr}")
            telegram_msg = self._format_telegram_block(asset, direction, consensus, block_rr)
            return self._build_output(
                decision=DECISION_BLOCK,
                asset=asset, direction=direction, consensus=consensus,
                reason=block_rr, telegram_msg=telegram_msg,
                sizing={}, tp_levels={},
            )

        atr_pips = data.get("atr_pips", 0)
        if atr_pips > 0 and sl_pips < atr_pips * 0.5:
            block_atr = (
                f"SL {sl_pips:.1f}pip < 0.5× ATR {atr_pips:.1f}pip — "
                f"SL inside M15 statistical noise (ATR floor)"
            )
            logger.info(f"[{self.name}] ATR floor BLOCK: {block_atr}")
            telegram_msg = self._format_telegram_block(asset, direction, consensus, block_atr)
            return self._build_output(
                decision=DECISION_BLOCK, asset=asset, direction=direction,
                consensus=consensus, reason=block_atr,
                telegram_msg=telegram_msg, sizing=sizing, tp_levels={},
            )

        # Adaptive sizing: STRONG → 1.25×, GOOD → 1.0×, WEAK → 0.75×
        _scale_size   = (instrument == "MT4_CFD")
        setup_quality = data.get("setup_quality", "GOOD")
        if _scale_size and setup_quality == "STRONG":
            sizing["lot_size"] = round(sizing["lot_size"] * 1.25 / 0.01) * 0.01
            sizing["risk_usd"] = round(sizing["lot_size"] * sizing["pip_value_per_lot"] * sl_pips, 2)
            sizing["quality_note"] = "STRONG setup → sizing 1.25×"
        elif _scale_size and setup_quality == "WEAK":
            sizing["lot_size"] = round(sizing["lot_size"] * 0.75 / 0.01) * 0.01
            sizing["lot_size"] = max(0.01, sizing["lot_size"])
            sizing["risk_usd"] = round(sizing["lot_size"] * sizing["pip_value_per_lot"] * sl_pips, 2)
            sizing["quality_note"] = "WEAK setup → sizing 0.75×"

        # VP Group A sizing adjustments
        vp_market_context = data.get("vp_market_context", "neutral")
        vp_trend_healthy  = data.get("vp_trend_healthy", False)
        vp_mtf_confluence = data.get("vp_mtf_confluence", False)
        vp_price_vs_comp  = data.get("vp_price_vs_composite", "neutral")
        vp_size_note      = []

        if _scale_size and vp_market_context == "range":
            sizing["lot_size"] = round(sizing["lot_size"] * 0.75 / 0.01) * 0.01
            sizing["lot_size"] = max(0.01, sizing["lot_size"])
            sizing["risk_usd"] = round(sizing["lot_size"] * sizing.get("pip_value_per_lot", 10) * sl_pips, 2)
            vp_size_note.append("Ranging market → sizing -25%")

        elif _scale_size and (vp_mtf_confluence and vp_trend_healthy and
              vp_price_vs_comp in ("discount", "premium")):
            sizing["lot_size"] = round(sizing["lot_size"] * 1.10 / 0.01) * 0.01
            sizing["risk_usd"] = round(sizing["lot_size"] * sizing.get("pip_value_per_lot", 10) * sl_pips, 2)
            vp_size_note.append("MTF+Trend+Composite confluent → sizing +10%")

        elif _scale_size and (not vp_trend_healthy and data.get("vp_trend_score", 0) == 0):
            sizing["lot_size"] = round(sizing["lot_size"] * 0.85 / 0.01) * 0.01
            sizing["lot_size"] = max(0.01, sizing["lot_size"])
            sizing["risk_usd"] = round(sizing["lot_size"] * sizing.get("pip_value_per_lot", 10) * sl_pips, 2)
            vp_size_note.append("Unhealthy VP trend → sizing -15%")

        if vp_size_note:
            existing           = sizing.get("quality_note", "")
            sizing["quality_note"] = (existing + " | " if existing else "") + " | ".join(vp_size_note)

        sizing["vp_market_context"] = vp_market_context
        sizing["vp_trend_healthy"]  = vp_trend_healthy
        sizing["vp_mtf_confluence"] = vp_mtf_confluence

        vp_poor_high = data.get("vp_poor_high")
        vp_poor_low  = data.get("vp_poor_low")
        if direction == "BUY" and vp_poor_high:
            sizing["poor_level_warning"] = f"⚠️ Poor High @ {_fmt_price(asset, vp_poor_high)} — weak TP level"
        elif direction == "SELL" and vp_poor_low:
            sizing["poor_level_warning"] = f"⚠️ Poor Low @ {_fmt_price(asset, vp_poor_low)} — weak TP level"

        if data.get("range_day_mode"):
            sizing["range_day_mode"] = True
            sizing["quality_note"]   = (
                (sizing.get("quality_note", "") + " | " if sizing.get("quality_note") else "") +
                f"Range-day: SL=0.7×ATR={sl_pips:.1f}pip, TP=1.8R"
            )

        # Normalize lot size to broker step (indices/oil have step 0.10)
        if instrument == "MT4_CFD":
            _lot_pre  = float(sizing.get("lot_size", 0))
            _lot_norm = normalize_lot(asset, _lot_pre)
            if abs(_lot_norm - _lot_pre) > 1e-9:
                sizing["lot_size"] = _lot_norm
                sizing["risk_usd"] = round(
                    _lot_norm * sizing.get("pip_value_per_lot", 10) * sl_pips, 2
                )
                logger.info(
                    f"[{self.name}] Lot normalized for {asset}: "
                    f"{_lot_pre:.2f} → {_lot_norm:.2f} (risk ${sizing['risk_usd']:.0f})"
                )

        decision  = DECISION_APPROVE
        rationale = (
            f"v3 deterministic: consensus {consensus:+.1f}, "
            f"rr_modifier={rr_modifier:.2f}, R:R={final_rr:.2f}, "
            f"setup={setup_quality}, risk=${sizing['risk_usd']:.0f}"
        )

        logger.info(
            f"[{self.name}] → APPROVE | rr_modifier={rr_modifier:.2f} "
            f"R:R={final_rr:.2f} | {setup_quality} sizing={float(sizing['lot_size']):.2f}"
        )

        telegram_msg = self._format_telegram_message(
            asset, direction, consensus, data["agent_scores"],
            sizing, tp_levels, sl_pips, instrument, decision, rationale,
            corr_warning
        )

        logger.info(f"[{self.name}] {asset} {direction} → {decision}")

        return self._build_output(
            decision=decision,
            asset=asset, direction=direction, consensus=consensus,
            reason=rationale, telegram_msg=telegram_msg,
            sizing=sizing, tp_levels=tp_levels,
        )

    # ─── Risk Gate ────────────────────────────────────────────────────────────

    def _risk_gate(
        self,
        asset: str,
        direction: str,
        consensus: float,
        sl_pips: float,
        open_positions: list,
        drawdown: float,
    ) -> Optional[str]:
        """
        Deterministic checks. Returns the block reason or None if OK.
        """
        # 1. Maximum drawdown
        if drawdown >= MAX_DRAWDOWN_PCT:
            return (
                f"Current drawdown {drawdown:.1f}% ≥ limit {MAX_DRAWDOWN_PCT:.0f}% — "
                f"trading suspended until recovery"
            )

        # 2. Maximum open positions — separate limit per broker
        instrument = INSTRUMENT_TYPE.get(asset, "MT4_CFD")
        mt4_count  = sum(1 for p in open_positions if p.get("broker", "MT4") == "MT4")
        ib_count   = sum(1 for p in open_positions if p.get("broker", "IB")  == "IB")
        if instrument == "IB_MICRO":
            if ib_count >= MAX_OPEN_IB:
                return f"IB position limit reached ({ib_count}/{MAX_OPEN_IB})"
        else:
            if mt4_count >= MAX_OPEN_MT4:
                return f"MT4 position limit reached ({mt4_count}/{MAX_OPEN_MT4})"

        # 3. Position already open on the same asset
        open_assets = [p.get("asset", "").upper() for p in open_positions]
        if asset in open_assets:
            return f"Position already open on {asset} — no double exposure"

        # 4. Consensus below minimum threshold
        from agents.strategy_rules import is_monday as _is_monday
        _manual = 62.0 if _is_monday() else MANUAL_THRESHOLD
        if abs(consensus) < _manual:
            return (
                f"Consensus {consensus:+.1f} below minimum threshold "
                f"(|{abs(consensus):.1f}| < {_manual:.0f}{'[MON]' if _is_monday() else ''}) — NO_TRADE"
            )

        # 5. SL not available
        if sl_pips <= 0:
            return "Stop loss cannot be computed — insufficient technical data"

        # 6. Minimum SL per asset — blocks if SL is inside market noise.
        min_sl = MIN_SL_PIPS.get(asset, 10.0)
        if sl_pips < min_sl:
            return (
                f"SL {sl_pips:.1f}pip < minimum {min_sl:.0f}pip for {asset} — "
                f"entry inside market noise (risk of stop hunt and spurious re-scan)"
            )

        return None  # all checks passed

    # ─── Position Sizing ──────────────────────────────────────────────────────

    def _calculate_sizing(
        self,
        asset: str,
        direction: str,
        equity: float,
        sl_pips: float,
        instrument: str,
        pip_value_per_lot: float,
    ) -> dict:
        """
        Compute lot size based on equity × RISK_PCT.

        MT4 CFD: fractional lot rounded to 0.01
        IB Micro: always 1 contract (verifies risk is acceptable)
        """
        risk_amount  = equity * (RISK_PER_TRADE_PCT / 100)
        risk_warning = ""

        if instrument == "IB_MICRO":
            n_contracts     = 2 if asset == "6E" else 1
            _risk_per_c     = sl_pips * pip_value_per_lot
            if n_contracts > 1 and n_contracts * _risk_per_c > IB_MICRO_MAX_RISK_USD:
                n_contracts = 1
            risk_1_contract = n_contracts * _risk_per_c
            risk_pct_1c     = (risk_1_contract / equity) * 100
            lot_size        = n_contracts
            actual_risk     = risk_1_contract
            actual_risk_pct = risk_pct_1c

            if risk_1_contract > IB_MICRO_MAX_RISK_USD:
                note = (
                    f"IB Micro: {n_contracts} contracts — risk ${risk_1_contract:.0f} "
                    f"EXCEEDS hard cap ${IB_MICRO_MAX_RISK_USD:.0f} — BLOCK"
                )
                return {
                    "lot_size":           0,
                    "risk_usd":           round(risk_1_contract, 2),
                    "risk_pct":           round(risk_pct_1c, 2),
                    "risk_per_trade_pct": RISK_PER_TRADE_PCT,
                    "equity":             equity,
                    "pip_value_per_lot":  pip_value_per_lot,
                    "note":               note,
                    "instrument":         instrument,
                    "block_reason":       f"IB Micro risk ${risk_1_contract:.0f} exceeds hard cap ${IB_MICRO_MAX_RISK_USD:.0f}",
                }
            elif risk_1_contract > IB_MICRO_WARN_RISK_USD:
                risk_warning = (
                    f"⚠️ *ABOVE SOFT CAP* — micro risk ${risk_1_contract:.0f} "
                    f"above ${IB_MICRO_WARN_RISK_USD:.0f} (hard ${IB_MICRO_MAX_RISK_USD:.0f})"
                )
                note = (
                    f"IB Micro: {n_contracts} contracts — risk ${risk_1_contract:.0f} "
                    f"({risk_pct_1c:.1f}% equity) — ABOVE SOFT CAP, approved with warning"
                )
            else:
                note = (
                    f"IB Micro: {n_contracts} contracts — risk ${risk_1_contract:.0f} "
                    f"({risk_pct_1c:.1f}% equity) — OK"
                )
        else:
            # MT4 CFD: fractional lot
            if sl_pips > 0 and pip_value_per_lot > 0:
                raw_lot  = risk_amount / (sl_pips * pip_value_per_lot)
                lot_size = max(0.01, round(round(raw_lot, 4) / 0.01) * 0.01)
                lot_size = round(lot_size, 2)
            else:
                lot_size = 0.01
            actual_risk     = lot_size * pip_value_per_lot * sl_pips
            actual_risk_pct = (actual_risk / equity) * 100
            note            = f"MT4 CFD: {lot_size:.2f} lot — risk {actual_risk_pct:.2f}%"

        return {
            "lot_size":           lot_size,
            "risk_usd":           round(actual_risk, 2),
            "risk_pct":           round(actual_risk_pct, 2),
            "risk_per_trade_pct": RISK_PER_TRADE_PCT,
            "equity":             equity,
            "pip_value_per_lot":  pip_value_per_lot,
            "note":               note,
            "instrument":         instrument,
            "risk_warning":       risk_warning,
        }

    # ─── Structured TP/SL ─────────────────────────────────────────────────────

    def _calculate_tp_levels(
        self,
        direction: str,
        sl_pips: float,
        instrument: str,
        lot_size: float,
        tp1_price:  float = 0.0,
        tp2_price:  float = 0.0,
        tp3_price:  float = 0.0,
        tp1_pips:   float = 0.0,
        tp2_pips:   float = 0.0,
        tp3_pips:   float = 0.0,
        tp1_source: str   = "",
        tp2_source: str   = "",
        tp3_source: str   = "",
        rr1: float = 0.0,
        rr2: float = 0.0,
        rr3: float = 0.0,
        range_day_mode:  bool  = False,
        range_day_tp_rr: float = 1.8,
    ) -> dict:
        """
        Compute structured TP levels per broker.

        v3: uses real prices from TechnicalAgent (S/D zones + VPOC) when available.
        Fallback formula (2×SL / 3×SL) only when real prices are not available.

        Range-day mode (SINGLE_TP_RANGE):
          - TP = 1.8×SL (single exit, 100%)
          - SL already overridden to 0.7×ATR in analyze()
          - No partial close, no runner

        MT4 CFD (SINGLE_TP_PARTIAL):
          - Fixed TP on order = tp3_price (runner) or 3×SL fallback
          - Partial close 60% when price reaches tp1_price or 2×SL
          - Runner 40% moves to BE toward tp2/tp3

        IB Micro (SINGLE_TP):
          - Single TP = tp2_price (main target) or 2.5×SL fallback
          - 1 fixed contract, no partial close
        """
        if range_day_mode and instrument == "MT4_CFD":
            tp_pips_rd = round(sl_pips * range_day_tp_rr, 1)
            return {
                "type":      "SINGLE_TP_RANGE",
                "tp_pips":   tp_pips_rd,
                "tp_price":  0.0,
                "sl_pips":   sl_pips,
                "rr_ratio":  round(range_day_tp_rr, 2),
                "trailing":  "none — full exit at TP",
                "close_pct": {"tp": 100},
                "tp_source": "Range-day 1.8R (SL=0.7×ATR)",
            }

        has_real_tp1 = tp1_pips > 0 and (rr1 >= 1.5 or tp1_price > 0)
        has_real_tp2 = tp2_pips > 0 and (rr2 >= 1.5 or tp2_price > 0)

        if instrument == "IB_MICRO":
            if has_real_tp2:
                tp_pips_real  = tp2_pips
                rr_real       = rr2
                tp_price_real = tp2_price
                source_note   = f"TP2 real [{tp2_source}]"
            elif has_real_tp1:
                tp_pips_real  = tp1_pips
                rr_real       = rr1
                tp_price_real = tp1_price
                source_note   = f"TP1 real [{tp1_source}]"
            else:
                tp_pips_real  = round(sl_pips * 2.5, 1)
                rr_real       = 2.5
                tp_price_real = 0.0
                source_note   = "2.5×SL fallback"

            return {
                "type":      "SINGLE_TP",
                "tp_pips":   tp_pips_real,
                "tp_price":  tp_price_real,
                "sl_pips":   sl_pips,
                "rr_ratio":  rr_real,
                "trailing":  "breakeven at 50% TP, then trailing 1×ATR",
                "close_pct": {"tp": 100},
                "tp_source": source_note,
            }
        else:
            # MT4 CFD: single order with partial close at TP1
            if has_real_tp1:
                partial_pips  = tp1_pips
                partial_price = tp1_price
                partial_rr    = rr1
                partial_src   = f"TP1 real [{tp1_source}]"
            else:
                partial_pips  = round(sl_pips * 2.0, 1)
                partial_price = 0.0
                partial_rr    = 2.0
                partial_src   = "2×SL fallback"

            if has_real_tp2:
                runner_pips  = tp2_pips
                runner_price = tp2_price
                runner_rr    = rr2
                runner_src   = f"TP2 real [{tp2_source}]"
            else:
                runner_pips  = round(sl_pips * 3.0, 1)
                runner_price = 0.0
                runner_rr    = 3.0
                runner_src   = "3×SL fallback"

            return {
                "type":           "SINGLE_TP_PARTIAL",
                "tp_pips":        runner_pips,
                "tp_price":       runner_price,
                "sl_pips":        sl_pips,
                "partial_pips":   partial_pips,
                "partial_price":  partial_price,
                "partial_pct":    60,
                "runner_pips":    runner_pips,
                "runner_price":   runner_price,
                "rr_partial":     partial_rr,
                "rr_runner":      runner_rr,
                "be_trigger":     f"BE after partial close at {partial_pips:.0f}pip",
                "partial_source": partial_src,
                "runner_source":  runner_src,
                "tp3_pips":       tp3_pips  if tp3_pips > 0  else 0.0,
                "tp3_price":      tp3_price if tp3_price > 0 else 0.0,
                "tp3_source":     tp3_source if tp3_pips > 0 else "",
            }

    # ─── Correlation check ────────────────────────────────────────────────────

    def _correlation_check(self, asset: str, open_positions: list) -> Optional[str]:
        """
        Check for open positions on correlated assets.
        BLOCK if ≥3 assets from the same group are already open.
        Warning if 1-2 correlated assets are open.
        """
        if not open_positions:
            return None

        open_assets     = {p.get("asset", "").upper() for p in open_positions}
        MAX_CORRELATED  = 3

        for group in CORRELATED_GROUPS:
            if asset in group:
                overlap = group & open_assets
                if len(overlap) >= MAX_CORRELATED:
                    return (
                        f"BLOCK: {len(overlap)} correlated assets already open "
                        f"{', '.join(sorted(overlap))} — "
                        f"max {MAX_CORRELATED} per group allowed"
                    )
                elif overlap:
                    return (
                        f"Correlated assets in position: {', '.join(sorted(overlap))} — "
                        f"partial exposure ({len(overlap)}/{MAX_CORRELATED})"
                    )
        return None

    # ─── Telegram formatting ──────────────────────────────────────────────────

    def _format_telegram_message(
        self,
        asset: str,
        direction: str,
        consensus: float,
        agent_scores: dict,
        sizing: dict,
        tp_levels: dict,
        sl_pips: float,
        instrument: str,
        decision: str,
        rationale: str,
        corr_warning: Optional[str],
    ) -> str:
        """
        Format the Telegram message with all operational parameters.
        Includes a unique callback_id for APPROVE/BLOCK/MODIFY tracking.
        """
        import uuid
        callback_id = str(uuid.uuid4())[:8].upper()

        icon_dir  = "📈" if direction == "BUY" else "📉"
        icon_dec  = {"APPROVE": "✅", "BLOCK": "🚫", "MODIFY": "⚠️"}.get(decision, "❓")

        abs_cons = abs(consensus)
        if abs_cons >= AUTO_EXECUTE_THRESHOLD:
            prob_label = "🚀 AUTO-EXECUTE"
        elif abs_cons >= MANUAL_THRESHOLD:
            prob_label = "✅ MANUAL APPROVE"
        else:
            prob_label = "⚠️ BORDERLINE"

        range_day_badge = "  🔵 *RANGE-DAY*" if sizing.get("range_day_mode") else ""
        lines = [f"{icon_dir} *NEW SETUP — {asset} {direction}*{range_day_badge}"]

        _risk_warn = sizing.get("risk_warning", "")
        if _risk_warn:
            lines.append(_risk_warn)
        lines += [
            "━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Consensus: *{consensus:+.1f}*  [{prob_label}]",
            "",
        ]

        if agent_scores:
            lines.append("*Agent breakdown:*")
            for agent, score in agent_scores.items():
                bar = "▓" * min(int(abs(float(score)) / 2), 10)
                lines.append(f"  {agent:<20} {float(score):>+6.1f}  {bar}")
            lines.append("")

        sl_price     = tp_levels.get("sl_price", 0)
        sl_price_str = f" @ {_fmt_price(asset, sl_price)}" if sl_price else ""
        lines.append("*Operational levels:*")
        lines.append(f"  🛑 SL:   {sl_pips:.1f} pip{sl_price_str}")

        if tp_levels.get("type") == "SINGLE_TP":
            tp_p         = tp_levels.get("tp_price", 0)
            tp_price_str = f" @ {_fmt_price(asset, tp_p)}" if tp_p else ""
            lines.append(f"  🎯 TP:   {tp_levels['tp_pips']:.1f} pip{tp_price_str}  R:R {tp_levels['rr_ratio']:.1f}  [{tp_levels.get('tp_source','')}]")
            lines.append(f"  🔄 BE:   {tp_levels.get('trailing', 'BE at 50% of TP')}")
        else:
            p_pip   = tp_levels.get("partial_pips", 0)
            p_price = tp_levels.get("partial_price", 0)
            p_rr    = tp_levels.get("rr_partial", 0)
            p_src   = tp_levels.get("partial_source", "")
            r_pip   = tp_levels.get("runner_pips", 0)
            r_price = tp_levels.get("runner_price", 0)
            r_rr    = tp_levels.get("rr_runner", 0)
            r_src   = tp_levels.get("runner_source", "")
            p_price_str = f" @ {_fmt_price(asset, p_price)}" if p_price else ""
            r_price_str = f" @ {_fmt_price(asset, r_price)}" if r_price else ""
            lines.append(f"  🎯 Partial (60%): {p_pip:.1f} pip{p_price_str}  R:R {p_rr:.1f}  [{p_src}]")
            lines.append(f"  🎯 Runner  (40%): {r_pip:.1f} pip{r_price_str}  R:R {r_rr:.1f}  [{r_src}]")
            lines.append(f"  📌 BE: SL → open price after partial close")

        lines.append("")
        lines.append("*Position sizing:*")
        lines.append(f"  📊 Lot:     {float(sizing['lot_size']):.2f} ({instrument})")
        lines.append(f"  💰 Risk:    ${sizing['risk_usd']:.0f} ({sizing['risk_pct']:.1f}% equity)")
        lines.append(f"  💼 Equity:  ${sizing['equity']:,.0f}")

        setup_quality = sizing.get("setup_quality", "")
        if setup_quality:
            quality_icon = {"STRONG": "🟢", "GOOD": "🟡", "WEAK": "🟠"}.get(setup_quality, "⚪")
            lines.append(f"  {quality_icon} Setup:  {setup_quality}")
        if sizing.get("quality_note"):
            lines.append(f"  📐 Sizing:  {sizing['quality_note']}")
        if sizing.get("risk_elevated"):
            lines.append(f"  🔴 {sizing['risk_elevated_note']}")
        if sizing.get("modify_reason"):
            lines.append(f"  ⚠️  Mod:    {sizing['modify_reason']}")

        vp_ctx   = sizing.get("vp_market_context", "")
        vp_trend = sizing.get("vp_trend_healthy", None)
        vp_mtf   = sizing.get("vp_mtf_confluence", False)
        vp_poor  = sizing.get("poor_level_warning", "")

        if vp_ctx and vp_ctx != "neutral":
            ctx_icon = "📊" if vp_ctx == "trending" else "〰️"
            lines.append(f"  {ctx_icon} VP:     Market {vp_ctx.upper()}")
        if vp_trend is not None:
            trend_icon = "✅" if vp_trend else "⚠️"
            lines.append(f"  {trend_icon} Trend:  {'healthy' if vp_trend else 'unhealthy'}")
        if vp_mtf:
            lines.append(f"  🎯 MTF VP: Multi-TF confluence active")
        if vp_poor:
            lines.append(f"  {vp_poor}")

        if corr_warning:
            lines.append(f"\n  ⚠️ {corr_warning}")

        lines.append("")
        lines.append(f"*RiskManager decision:* {icon_dec} {decision}")
        lines.append(f"_{rationale}_")
        lines.append("")
        lines.append(f"```ID: {callback_id}```")

        risk_elevated = sizing.get("risk_elevated", False)
        if abs(consensus) >= AUTO_EXECUTE_THRESHOLD and not risk_elevated:
            lines.append(f"🚀 *Trade executed automatically* (consensus ≥ {AUTO_EXECUTE_THRESHOLD:.0f})")
            lines.append("_No action required._")
        else:
            if risk_elevated:
                lines.append("🔴 *Manual approval required* — elevated risk, your call")
            lines.append("⏱ Timeout: 5 min → trade cancelled automatically")
            lines.append("")
            lines.append("[APPROVE ✅] [BLOCK 🚫] [MODIFY ✏️]")

        return "\n".join(lines)

    def _format_telegram_block(
        self, asset: str, direction: str, consensus: float, reason: str
    ) -> str:
        return (
            f"🚫 *TRADE BLOCKED — {asset} {direction}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Consensus: {consensus:+.1f}\n\n"
            f"*Reason:* {reason}\n\n"
            f"_RiskManagerAgent blocked the trade automatically._"
        )

    async def run(self, context: dict) -> AgentResult:
        """
        Override BaseAgent.run() to inject analyze() fields into raw_data.
        BaseAgent only puts collect_data() output into raw_data.
        """
        logger.info(
            f"[{self.name}] Starting analysis — asset: {context.get('asset')} "
            f"dir: {context.get('direction')}"
        )
        try:
            data   = await self.collect_data(context)
            result = await self.analyze(data, context)

            extra    = {
                k: v for k, v in result.items()
                if k not in ("score", "summary", "bull_case", "bear_case", "confidence", "details")
            }
            raw_data = {**data, **extra}
            raw_data.pop("agent_scores", None)

            agent_result = AgentResult(
                agent=self.name,
                score=0,
                summary=result.get("summary", ""),
                bull_case=result.get("bull_case", ""),
                bear_case=result.get("bear_case", ""),
                confidence=result.get("confidence", "medium"),
                details=result.get("details", ""),
                raw_data=raw_data,
            )
            logger.info(
                f"[{self.name}] Completed — "
                f"decision: {raw_data.get('decision','?')} "
                f"confidence: {agent_result.confidence}"
            )
            return agent_result

        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
            return AgentResult(
                agent=self.name, score=0,
                summary=f"Error: {str(e)[:50]}",
                bull_case="", bear_case="",
                confidence="low", details="", error=str(e),
            )

    # ─── Build output ─────────────────────────────────────────────────────────

    def _build_output(
        self,
        decision: str,
        asset: str,
        direction: str,
        consensus: float,
        reason: str,
        telegram_msg: str,
        sizing: dict,
        tp_levels: dict,
    ) -> dict:
        """Build the output dict for BaseAgent."""
        icon    = {"APPROVE": "✅", "BLOCK": "🚫", "MODIFY": "⚠️"}.get(decision, "❓")
        summary = f"{icon} {decision} | {asset} {direction} | Consensus {consensus:+.1f}"
        if sizing:
            summary += f" | {float(sizing.get('lot_size', 0)):.2f} lot | Risk {sizing.get('risk_pct','?'):.1f}%"

        details = (
            f"=== RISK MANAGER AGENT — {asset} {direction} ===\n"
            f"Decision    : {decision}\n"
            f"Consensus   : {consensus:+.1f}\n"
            f"Rationale   : {reason}\n\n"
            f"--- Sizing ---\n"
            f"{json.dumps(sizing, indent=2, default=str)}\n\n"
            f"--- TP Levels ---\n"
            f"{json.dumps(tp_levels, indent=2, default=str)}\n\n"
            f"--- Telegram Message ---\n"
            f"{telegram_msg}"
        )

        return {
            "score":            0,
            "summary":          summary,
            "bull_case":        f"Trade {decision}: {reason}",
            "bear_case":        f"Trade {decision}: {reason}",
            "confidence":       "high" if decision == DECISION_APPROVE else "medium",
            "details":          details,
            "decision":         decision,
            "telegram_message": telegram_msg,
            "sizing":           sizing,
            "tp_levels":        tp_levels,
            "consensus_score":  consensus,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint  →  python -m agents.risk_manager_agent
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    """
    Quick verification entrypoint for development.

    In production this agent is NEVER run directly.
    It is invoked by the LangGraph orchestrator with the real context
    from the preceding 6 agents:

        result = await risk_manager.run({
            "asset":               asset,
            "direction":           direction,
            "consensus_score":     total_score,
            "agent_scores":        {agent: score},
            "sl_pips":             sl_pips,
            "tp_pips":             tp_pips,
            "atr_pips":            atr_pips,
            "open_positions":      open_positions,
            "current_drawdown_pct": drawdown_pct,
        })

    Decision: APPROVE / BLOCK / MODIFY + ready-to-send Telegram message.
    """
    agent = RiskManagerAgent()

    print("\n" + "=" * 70)
    print("  RISK MANAGER AGENT — Ready")
    print(f"  Equity: ${PAPER_EQUITY_USD:,.0f} | Risk/trade: {RISK_PER_TRADE_PCT}%")
    print(f"  Max drawdown: {MAX_DRAWDOWN_PCT}% | MT4 positions: {MAX_OPEN_MT4} | IB positions: {MAX_OPEN_IB}")
    print(f"  Thresholds: auto-execute ±{AUTO_EXECUTE_THRESHOLD:.0f} | "
          f"manual ±{MANUAL_THRESHOLD:.0f} | IB cap ${IB_MICRO_MAX_RISK_USD:.0f}")
    print("=" * 70)
    print()
    print("  Initialized. Waiting for context from orchestrator.")
    print()
    print("  Production flow:")
    print("  Orchestrator → collect_data(context) → analyze() → AgentResult")
    print("    ↳ raw_data.decision:         APPROVE / BLOCK / MODIFY")
    print("    ↳ raw_data.telegram_message: ready-to-send message")
    print("    ↳ raw_data.sizing:           lot_size, risk_usd, risk_pct")
    print("    ↳ raw_data.tp_levels:        SINGLE_TP_PARTIAL (MT4) or SINGLE_TP (IB)")


if __name__ == "__main__":
    asyncio.run(main())
