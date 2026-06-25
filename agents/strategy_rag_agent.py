"""
agents/strategy_rag_agent.py  —  v3.0
StrategyRAGAgent — Agent 6

NOTE: Production implementation queries a Qdrant vector database populated
with ICT/SMC strategy documentation, then calls Claude for qualitative
evaluation of the setup. This stub demonstrates the modifier interface and
the full ICT chain logic. See paper Section 2.3 for design rationale.

v3 changes vs v2:
  - quality_delta ±1/2 instead of flat bonus
    → no longer adds a fixed score to consensus
    → produces a qualitative delta that graph.py applies to the consensus score
  - strategy_rules v3.1 integration
    → does not recompute the ICT chain from scratch
    → reads results already produced by TechnicalAgent from context
    → adds only the qualitative RAG + LLM evaluation
  - New v3.1 concepts in evaluation
    → MSS, MMXM, Judas Swing, BPR, SMT, LRLR, EQH/EQL, IPDA
  - Mini-debate removed
    → replaced by deterministic counter_signal_check
    → saves 1 LLM call per trade

Output v3:
  score:         -20/+20 (consensus compatibility)
  quality_delta: -2/-1/0/+1/+2 — delta applied to consensus score
                 +2: excellent ICT setup, KB strongly confirms
                 +1: good setup, KB confirms
                  0: neutral setup, no impact
                 -1: weak setup, KB does not confirm
                 -2: contradictory setup, strong opposing signals
"""

import asyncio
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger

from agents.base_agent import BaseAgent, AgentResult

# ─────────────────────────────────────────────────────────────────────────────
# yfinance tickers
# ─────────────────────────────────────────────────────────────────────────────
YFINANCE_MAP: dict[str, str] = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "GBPJPY": "GBPJPY=X", "USDCAD": "USDCAD=X", "EURAUD": "EURAUD=X",
    "EURGBP": "EURGBP=X", "NZDJPY": "NZDJPY=X", "EURCHF": "EURCHF=X",
    "XAUUSD": "GC=F",     "MGC":    "GC=F",
    "XAGUSD": "SI=F",
    "BTCUSD": "BTC-USD",  "ETHUSD": "ETH-USD",
    "MES":    "ES=F",
    "MCL":    "CL=F",     "NAS100": "NQ=F",
    "6E":     "EURUSD=X",
}

PIP_SIZE: dict[str, float] = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "USDCAD": 0.0001,
    "EURAUD": 0.0001, "EURGBP": 0.0001, "EURCHF": 0.0001,
    "6E":     0.00005,
    "USDJPY": 0.01,   "GBPJPY": 0.01,   "NZDJPY": 0.01,
    "XAUUSD": 0.10,   "MGC":    0.10,
    "XAGUSD": 0.010,
    "BTCUSD": 1.0,    "ETHUSD": 0.10,
    "MES":    0.25,
    "MCL":    0.010,  "NAS100": 0.25,
}

SCORE_MIN = -20
SCORE_MAX = +20
MIN_RR    =  1.5

# ICT chain deterministic thresholds
ICT_L1_SCORE = 5
ICT_L2_SCORE = 9
ICT_L3_SCORE = 13
ICT_L4_SCORE = 16

# quality_delta thresholds
QD_STRONG_POSITIVE = 16
QD_WEAK_POSITIVE   = 8
QD_WEAK_NEGATIVE   = -8
QD_STRONG_NEGATIVE = -16


# ─────────────────────────────────────────────────────────────────────────────
class StrategyRAGAgent(BaseAgent):
    """
    Agent 6 — StrategyRAGAgent v3.

    NOTE: Production implementation uses Qdrant vector database populated
    with ICT/SMC strategy documentation. This stub demonstrates the interface.

    In v3 the role changes slightly:
    - Does not recompute the ICT chain (already done by TechnicalAgent)
    - Reads ICT results from context (strategy results passed by graph.py)
    - Focuses on qualitative RAG + LLM evaluation
    - Produces quality_delta ±2/±1/0 instead of flat bonus

    Score: -20/+20 for consensus compatibility.
    quality_delta: extra field in raw_data used by graph.py.
    """

    AGENT_NAME  = "StrategyRAGAgent"
    MODEL       = "claude-haiku-4-5"
    SCORE_RANGE = (SCORE_MIN, SCORE_MAX)

    _rag_cache: dict = {}
    RAG_CACHE_TTL    = 30 * 60  # 30 minutes

    def __init__(self):
        super().__init__()
        # Production: self.client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # Production: self._qdrant = QdrantClient(url=os.getenv("QDRANT_URL"))

    # ─── collect_data ─────────────────────────────────────────────────────────

    async def collect_data(self, context: dict) -> dict:
        """
        v3: collects OHLCV data for local ICT chain (fallback when
        TechnicalAgent has not passed results in context).
        If strategy_rules results are already in context, uses them directly.
        """
        asset     = str(context.get("asset",     "EURUSD")).upper()
        direction = str(context.get("direction", "BUY")).upper()

        tech_raw         = context.get("technical_raw_data") or {}
        strategy_results = tech_raw.get("strategy")

        sl_pips    = float(context.get("sl_pips",    0))
        tp1_pips   = float(context.get("tp1_pips",   0))
        tp2_pips   = float(context.get("tp2_pips",   0))
        rr1        = float(context.get("rr1",        0))
        rr2        = float(context.get("rr2",        0))
        atr_pips   = float(context.get("atr_pips",   0))
        layer2_raw = float(context.get("layer2_raw", 0))

        price_data: dict[str, pd.DataFrame] = {}

        return {
            "asset":              asset,
            "direction":          direction,
            "price_data":         price_data,
            "pip_size":           PIP_SIZE.get(asset, 0.0001),
            "strategy_results":   strategy_results,
            "sl_pips":            sl_pips,
            "tp1_pips":           tp1_pips,
            "tp2_pips":           tp2_pips,
            "tp1_source":         str(context.get("tp1_source",  "")),
            "tp2_source":         str(context.get("tp2_source",  "")),
            "tp1_price":          float(context.get("tp1_price", 0.0)),
            "tp2_price":          float(context.get("tp2_price", 0.0)),
            "rr1":                rr1,
            "rr2":                rr2,
            "atr_pips":           atr_pips,
            "layer2_raw":         layer2_raw,
            "signal_quality":     str(context.get("signal_quality",     "")),
            "active_confluences": list(context.get("active_confluences", [])),
            "layer2_factors":     list(context.get("layer2_factors",     [])),
            "instrument_type":    str(context.get("instrument_type",     "")),
        }

    # ─── analyze ──────────────────────────────────────────────────────────────

    async def analyze(self, data: dict, context: dict) -> dict:
        """
        Pipeline v3:
          1. Build ICT profile from context (strategy_rules v3.1)
             or recompute local chain (fallback)
          2. Deterministic counter-signal check (replaces mini-debate)
          3. RAG retrieval from Qdrant (stubbed in this version)
          4. LLM qualitative evaluation (stubbed in this version)
          5. Compute quality_delta ±2/±1/0
        """
        asset            = data["asset"]
        direction        = data["direction"]
        pip_size         = data["pip_size"]
        strategy_results = data.get("strategy_results")

        # ── Phase 1: ICT profile ──────────────────────────────────────────────
        has_sr = (
            strategy_results is not None and
            hasattr(strategy_results, "db") and
            hasattr(strategy_results, "mss")
        )
        has_context_ict = bool(
            data.get("signal_quality") or
            data.get("active_confluences") or
            data.get("layer2_raw", 0) > 0
        )

        if has_sr:
            ict    = self._build_ict_from_strategy_rules(asset, direction, strategy_results, data)
            source = "v3.1-object"
        elif has_context_ict:
            ict    = self._build_ict_from_context(asset, direction, data)
            source = "v3.1-context"
        else:
            return self._abstain(asset, direction, "Insufficient data for ICT analysis")

        logger.info(
            f"[{self.name}] {asset} {direction} | "
            f"ICT L{ict['chain_level']}/4 | det={ict['det_score']:+d} | source={source}"
        )

        if ict["chain_level"] == 0:
            return self._abstain(asset, direction, ict["stop_reason"])

        # ── Phase 2: Deterministic counter-signal check ───────────────────────
        counter_penalty, counter_argument = self._counter_signal_check(direction, ict, data)

        # ── Phase 3 & 4: RAG + LLM (stubbed) ────────────────────────────────
        # Implementation intentionally omitted.
        # Production: query Qdrant with ICT concept embeddings, then call Claude
        # for qualitative evaluation against KB chunks.
        rag_chunks = []
        claude     = self._stub_llm_analysis(asset, direction, ict, data)

        # ── Phase 5: Final score + quality_delta ─────────────────────────────
        daily_bias_penalty = ict.get("daily_bias_penalty", 0)
        raw_score   = claude["score"] + counter_penalty + daily_bias_penalty
        final_score = float(max(SCORE_MIN, min(SCORE_MAX, raw_score)))
        if direction == "BUY":
            final_score = max(0.0, final_score)
        else:
            final_score = min(0.0, final_score)
        final_score = round(final_score, 1)

        total_penalty = counter_penalty + daily_bias_penalty
        quality_delta = self._compute_quality_delta(final_score, direction, ict, total_penalty)

        ct_flag = " [CT]" if ict.get("is_counter_trend", False) else ""
        logger.info(
            f"[{self.name}] {asset} {direction}{ct_flag} | "
            f"det={ict['det_score']:+d} llm={claude['score']:+d} "
            f"counter={counter_penalty:+d} → final={final_score:+.1f} "
            f"quality_delta={quality_delta:+d}"
        )

        chain_desc = self._chain_description(ict)
        abs_score  = abs(final_score)

        summary = (
            f"ICT L{ict['chain_level']}/4 | "
            f"{ict.get('setup_type','N/A')} | "
            f"RAG [stub] | "
            f"Δ={quality_delta:+d} score={final_score:+.1f}/±20"
        )

        if abs_score >= 14:
            bull_case = f"Complete ICT chain — {chain_desc}."
            bear_case = f"Invalidation: {claude.get('invalidation','OB break')}."
        elif abs_score >= 8:
            bull_case = f"Sweep + BOS confirmed. {claude.get('rationale','')[:80]}"
            bear_case = f"Entry not qualified. Missing: {ict.get('missing_elements','OB valid / R:R')}."
        else:
            bull_case = f"Daily Bias {direction} confirmed. Partial setup."
            bear_case = f"Incomplete chain: {ict.get('stop_reason','')}."

        details = (
            f"=== STRATEGY RAG AGENT v3 — {asset} {direction} ===\n"
            f"Final score   : {final_score:+.1f} / ±20\n"
            f"quality_delta : {quality_delta:+d} (consensus impact)\n"
            f"ICT Chain     : Level {ict['chain_level']}/4\n"
            f"Setup type    : {ict.get('setup_type','N/A')}\n"
            f"RAG chunks    : [stub — 0]\n"
            f"Counter [{counter_argument[:80]}]: {counter_penalty:+d}\n\n"
            f"--- ICT Chain breakdown ---\n"
            f"{chain_desc}"
        )

        return {
            "score":            final_score,
            "summary":          summary,
            "bull_case":        bull_case,
            "bear_case":        bear_case,
            "confidence":       claude.get("confidence", "medium"),
            "details":          details,
            "quality_delta":    quality_delta,
            "ict_chain_level":  ict["chain_level"],
            "det_score":        ict["det_score"],
            "setup_type":       ict.get("setup_type", "N/A"),
            "rag_chunks_count": 0,
            "ssl_level":        ict.get("ssl_level"),
            "bsl_level":        ict.get("bsl_level"),
            "ob_zone":          (ict.get("ob_high"), ict.get("ob_low")),
            "rr_ratio":         ict.get("rr_ratio"),
            "invalidation":     claude.get("invalidation", "N/A"),
            "counter_penalty":  counter_penalty,
            "counter_argument": counter_argument,
        }

    # ─── Stub LLM analysis ────────────────────────────────────────────────────

    def _stub_llm_analysis(self, asset: str, direction: str, ict: dict, data: dict) -> dict:
        """
        # Implementation intentionally omitted.
        # Production: calls Claude with ICT profile + Qdrant KB chunks for
        # qualitative evaluation. Returns score -20/+20 and rationale.
        # See paper Section 2.3 for design rationale.
        """
        det_score = ict.get("det_score", 0)
        # Stub: derive synthetic score from deterministic ICT chain level
        sign  = 1 if direction == "BUY" else -1
        score = sign * min(abs(det_score), SCORE_MAX)
        return {
            "score":            score,
            "confidence":       "medium",
            "setup_quality":    "[stub]",
            "rag_confirmation": "[stub — Qdrant not connected]",
            "invalidation":     "OB break",
            "rationale":        "[stub] Score derived from deterministic ICT chain.",
        }

    # ─── ICT profile from strategy_rules v3.1 ────────────────────────────────

    def _build_ict_from_strategy_rules(
        self, asset: str, direction: str, sr, data: dict
    ) -> dict:
        """
        Build ICT profile using results already computed by TechnicalAgent
        via strategy_rules v3.1.

        Maps v3.1 concepts to 4 ICT chain levels:
          L1 → Daily Bias
          L2 → BSL/SSL (EQH/EQL, DOL, LRLR)
          L3 → Sweep + BOS (Judas, MSS, MMXM)
          L4 → Qualified entry (OB, OTE, FVG, Silver Bullet, R:R)
        """
        result: dict = {"chain_level": 0, "det_score": 0, "stop_reason": ""}

        sl_pips  = data.get("sl_pips",  0)
        tp1_pips = data.get("tp1_pips", 0)
        rr1      = data.get("rr1",      0)

        # ── L1: Daily Bias ────────────────────────────────────────────────────
        db        = sr.db
        swing_str = sr.swing_str

        l1_ok = (db.valid and db.direction == direction and db.strength >= 1)
        is_counter_trend = db.valid and db.direction != direction and db.strength >= 2

        struct_aligned = (
            (direction == "BUY"  and swing_str.current_structure == "BULLISH_HH_HL") or
            (direction == "SELL" and swing_str.current_structure == "BEARISH_LH_LL")
        )

        if not l1_ok:
            if is_counter_trend:
                logger.info(f"[{self.name}] Counter-trend: {db.direction} vs {direction}, strength={db.strength}")
                result.update({
                    "chain_level":       1,
                    "det_score":         ICT_L1_SCORE - 2,
                    "is_counter_trend":  True,
                    "daily_bias_penalty": -2,
                })
            else:
                result["stop_reason"] = (
                    f"L1 failed: Daily Bias {db.direction} invalid "
                    f"(required {direction}) strength={db.strength}"
                )
                return result

        if not result.get("chain_level"):
            result.update({
                "chain_level":       1,
                "det_score":         ICT_L1_SCORE,
                "is_counter_trend":  False,
                "daily_bias_penalty": 0,
            })

        result.update({
            "daily_bias":       db.direction,
            "ema20_d1":         f"{db.close_pct:.0f}% close pct",
            "swing_structure":  swing_str.current_structure,
            "struct_aligned":   struct_aligned,
            "swing_high_d1":    swing_str.last_lth if swing_str.last_lth else "N/A",
            "swing_low_d1":     swing_str.last_ltl if swing_str.last_ltl else "N/A",
        })

        # ── L2: BSL/SSL identification ─────────────────────────────────────
        eqh_eql = sr.eqheql
        dol     = sr.dol
        hrlr    = sr.hrlr

        has_ssl = eqh_eql.eql_is_ssl and eqh_eql.nearest_eql > 0
        has_bsl = eqh_eql.eqh_is_bsl and eqh_eql.nearest_eqh > 0
        has_dol = dol.dol_price > 0
        l2_ok   = has_ssl or has_bsl or has_dol

        if not l2_ok:
            result["stop_reason"] = "L2 failed: no BSL/SSL/DOL identified"
            return result

        result.update({
            "chain_level":  2,
            "det_score":    ICT_L2_SCORE,
            "ssl_level":    eqh_eql.nearest_eql if has_ssl else None,
            "bsl_level":    eqh_eql.nearest_eqh if has_bsl else None,
            "dol_type":     dol.dol_type,
            "dol_price":    dol.dol_price,
            "lrlr_detected": hrlr.lrlr_detected,
            "lrlr_target":  hrlr.lrlr_target if hrlr.lrlr_detected else None,
            "liq_target":   (
                eqh_eql.nearest_eql if direction == "BUY" and has_ssl
                else eqh_eql.nearest_eqh if direction == "SELL" and has_bsl
                else dol.dol_price
            ),
        })

        # ── L3: Sweep + BOS ────────────────────────────────────────────────
        mss   = sr.mss
        judas = sr.judas
        smc   = sr.smc
        mmxm  = sr.mmxm
        bpr   = sr.bpr
        smt   = sr.smt

        l3_ok = (
            (mss.valid and mss.displacement_ok) or
            (judas.valid and judas.reversal_confirmed) or
            (smc.liquidity_swept and smc.bos_detected and smc.bos_direction == direction)
        )

        if not l3_ok:
            reasons = []
            if not mss.valid:   reasons.append("no MSS")
            if not judas.valid: reasons.append("no Judas Swing")
            if not smc.liquidity_swept: reasons.append("no liquidity sweep")
            result["stop_reason"] = f"L3 failed: {' | '.join(reasons)}"
            return result

        sweep_price = (
            mss.sweep_price if mss.valid
            else judas.sweep_level if judas.valid
            else smc.sweep_level if smc.liquidity_swept
            else 0
        )
        bos_level = (
            mss.prior_swing_broken if mss.valid
            else getattr(smc, "bos_level", 0) if smc.bos_detected
            else 0
        )

        setup_type_parts = []
        if judas.valid and judas.reversal_confirmed: setup_type_parts.append("Judas Swing")
        if mss.valid:   setup_type_parts.append(f"MSS {mss.direction}")
        if mmxm.valid:  setup_type_parts.append(f"MMXM {mmxm.model_type}")
        setup_type = " + ".join(setup_type_parts) if setup_type_parts else "Sweep + BOS"
        setup_type += f" {'▲' if direction=='BUY' else '▼'}"

        result.update({
            "chain_level":  3,
            "det_score":    ICT_L3_SCORE,
            "sweep_price":  round(sweep_price, 5) if sweep_price else None,
            "bos_level":    round(bos_level,   5) if bos_level   else None,
            "setup_type":   setup_type,
            "mss_valid":    mss.valid,
            "mss_fvg":      mss.fvg_formed,
            "judas_valid":  judas.valid,
            "judas_in_kz":  judas.in_kill_zone,
            "mmxm_valid":   mmxm.valid,
            "bpr_valid":    bpr.valid and bpr.price_in_bpr,
            "smt_valid":    smt.valid,
        })

        # ── L4: Qualified entry ────────────────────────────────────────────
        ob  = sr.smc
        ote = sr.ote
        fvg = sr.fvg
        sb  = sr.silver_bullet

        in_ob  = ob.ob_valid  and not ob.ob_mitigated
        in_ote = ote.in_ote_zone
        in_fvg = fvg.valid    and not fvg.gap_filled and fvg.price_in_gap
        in_sb  = sb.valid     and sb.window_active

        l4_ok = (in_ob or in_ote or in_fvg or in_sb) and rr1 >= MIN_RR

        if not l4_ok:
            missing = []
            if not (in_ob or in_ote or in_fvg or in_sb):
                entry_zones = []
                if not in_ob:  entry_zones.append("OB")
                if not in_ote: entry_zones.append("OTE")
                if not in_fvg: entry_zones.append("FVG")
                if not in_sb:  entry_zones.append("Silver Bullet")
                missing.append(f"price outside {'/'.join(entry_zones)}")
            if rr1 < MIN_RR:
                missing.append(f"R:R={rr1:.1f} < {MIN_RR}")
            result["missing_elements"] = " | ".join(missing)
            result.update({
                "ob_high":      ob.ob_high if ob.ob_valid else None,
                "ob_low":       ob.ob_low  if ob.ob_valid else None,
                "rr_ratio":     rr1,
                "in_ob_zone":   in_ob,
                "in_sb_window": sb.window_active if sb else False,
                "sb_window":    sb.window if sb and sb.window_active else None,
            })
            return result

        entry_types = []
        if in_ob:  entry_types.append("OB")
        if in_ote: entry_types.append("OTE")
        if in_fvg: entry_types.append("FVG")
        if in_sb:  entry_types.append(f"Silver Bullet {sb.window}")

        result.update({
            "chain_level":   4,
            "det_score":     ICT_L4_SCORE,
            "ob_high":       round(ob.ob_high, 5) if ob.ob_valid else None,
            "ob_low":        round(ob.ob_low,  5) if ob.ob_valid else None,
            "ob_mid":        round((ob.ob_high + ob.ob_low)/2, 5) if ob.ob_valid else None,
            "sl_pips":       round(sl_pips, 1),
            "tp_pips":       round(tp1_pips, 1),
            "rr_ratio":      round(rr1, 2),
            "in_ob_zone":    in_ob,
            "in_ote":        in_ote,
            "in_fvg":        in_fvg,
            "in_sb_window":  in_sb,
            "sb_window":     sb.window if in_sb else None,
            "setup_type":    f"{setup_type} + {'/'.join(entry_types)} entry",
        })
        return result

    # ─── ICT profile from extracted context (LangGraph pipeline mode) ─────────

    def _build_ict_from_context(self, asset: str, direction: str, data: dict) -> dict:
        """
        Build ICT profile from data already extracted in context by graph.py.
        Used when LangGraph serializes state and the StrategyRulesResult object
        is no longer available as a Python object.
        """
        result: dict = {"chain_level": 0, "det_score": 0, "stop_reason": ""}

        layer2_raw  = float(data.get("layer2_raw",       0))
        active_conf = data.get("active_confluences",     [])
        sl_pips     = float(data.get("sl_pips",          0))
        tp1_pips    = float(data.get("tp1_pips",         0))
        rr1         = float(data.get("rr1",              0))

        if layer2_raw <= 0 and not active_conf:
            result["stop_reason"] = "No ICT data available from TechnicalAgent"
            return result

        def has_conf(*patterns):
            return any(any(p in c for p in patterns) for c in active_conf)

        # L1
        result.update({
            "chain_level": 1, "det_score": ICT_L1_SCORE,
            "daily_bias": direction, "ema20_d1": f"layer2={layer2_raw:.1f}/16",
            "swing_structure": "",
        })

        # L2
        has_liq = has_conf("EQL", "EQH", "DOL", "BSL", "SSL", "LRLR")
        if not has_liq and layer2_raw < 4:
            result["stop_reason"] = f"L2: no liquidity identified (L2={layer2_raw:.1f})"
            return result

        result.update({
            "chain_level": 2, "det_score": ICT_L2_SCORE,
            "ssl_level": None, "bsl_level": None,
            "lrlr_detected": has_conf("LRLR"),
        })

        # L3
        has_sweep = has_conf("MSS", "Judas", "MMXM", "Sweep", "BOS")
        if not has_sweep:
            result["stop_reason"] = "L3: no sweep/BOS identified"
            return result

        setup_parts = []
        if has_conf("Judas"): setup_parts.append("Judas Swing")
        if has_conf("MSS"):   setup_parts.append("MSS")
        if has_conf("MMXM"):  setup_parts.append("MMXM")
        setup_type  = " + ".join(setup_parts) if setup_parts else "Sweep + BOS"
        setup_type += f" {'▲' if direction=='BUY' else '▼'}"

        result.update({
            "chain_level": 3, "det_score": ICT_L3_SCORE,
            "setup_type":  setup_type,
            "mss_valid":   has_conf("MSS"),  "judas_valid": has_conf("Judas"),
            "mmxm_valid":  has_conf("MMXM"), "bpr_valid":   has_conf("BPR"),
            "smt_valid":   has_conf("SMT"),  "sweep_price": None, "bos_level": None,
        })

        # L4
        has_entry = has_conf("OB", "OTE", "FVG", "Demand", "Supply", "Breaker", "BPR")
        l4_ok     = has_entry and rr1 >= MIN_RR

        if not l4_ok:
            missing = []
            if not has_entry: missing.append("no OB/OTE/FVG")
            if rr1 < MIN_RR:  missing.append(f"R:R={rr1:.1f} < {MIN_RR}")
            result["missing_elements"] = " | ".join(missing)
            result.update({"rr_ratio": rr1, "in_ob_zone": has_entry})
            return result

        result.update({
            "chain_level": 4, "det_score": ICT_L4_SCORE,
            "setup_type":  f"{setup_type} + entry",
            "rr_ratio":    rr1, "sl_pips": sl_pips, "tp_pips": tp1_pips,
            "in_ob_zone":  has_conf("OB", "Breaker"),
            "in_ote":      has_conf("OTE"), "in_fvg": has_conf("FVG"),
        })
        result["_source"] = "context_v3.1"
        return result

    # ─── Counter-signal check ─────────────────────────────────────────────────

    def _counter_signal_check(
        self, direction: str, ict: dict, data: dict
    ) -> tuple[int, str]:
        """
        Check for signals opposing the setup — deterministic version.
        Replaces the v2 mini-debate (which cost 1 extra LLM call per trade).

        Returns: (penalty, counter_argument_text)
          penalty: 0 / -2 / -3 / -5
        """
        penalty = 0
        reasons = []

        bos_dir = ict.get("bos_direction_from_sr")
        if bos_dir and bos_dir != direction:
            penalty -= 3
            reasons.append(f"BOS {bos_dir} vs {direction} required")

        struct = ict.get("swing_structure")
        if struct:
            if direction == "BUY"  and struct == "BEARISH_LH_LL":
                penalty -= 2
                reasons.append(f"Swing Structure {struct} against BUY")
            elif direction == "SELL" and struct == "BULLISH_HH_HL":
                penalty -= 2
                reasons.append(f"Swing Structure {struct} against SELL")

        if ict.get("hrlr_detected"):
            penalty -= 2
            reasons.append("HRLR: liquidity level already swept → high resistance")

        penalty = max(-5, penalty)
        if not reasons:
            counter_text     = "No significant opposing signals — aligned setup"
            counter_strength = "LOW"
        elif penalty <= -4:
            counter_strength = "HIGH"
            counter_text = (
                f"STRONG opposing argument: {' | '.join(reasons)}. "
                f"Setup shows signals opposing the proposed direction."
            )
        elif penalty <= -2:
            counter_strength = "MEDIUM"
            counter_text = (
                f"MEDIUM opposing argument: {' | '.join(reasons)}. "
                f"Resistance present, not invalidating if other confluences hold."
            )
        else:
            counter_strength = "LOW"
            counter_text = f"WEAK opposing argument: {' | '.join(reasons)}."

        logger.info(
            f"[{self.name}] Counter-signal [{counter_strength}] penalty={penalty:+d}: "
            f"{counter_text[:80]}"
        )
        return penalty, counter_text

    # ─── quality_delta ────────────────────────────────────────────────────────

    def _compute_quality_delta(
        self, final_score: float, direction: str, ict: dict, counter_penalty: int
    ) -> int:
        """
        Compute quality_delta ±2/±1/0.
        Applied by graph.py to the consensus score AFTER all agent votes.
        """
        abs_score   = abs(final_score)
        chain_level = ict.get("chain_level", 0)
        has_premium = (
            ict.get("mss_valid") or ict.get("mmxm_valid") or
            ict.get("judas_valid") or ict.get("bpr_valid") or ict.get("smt_valid")
        )

        if chain_level >= 4 and abs_score >= QD_STRONG_POSITIVE and counter_penalty >= 0 and has_premium:
            return +2
        elif chain_level >= 3 and abs_score >= QD_WEAK_POSITIVE and counter_penalty >= -2:
            return +1
        elif chain_level <= 1 or abs_score == 0:
            return 0
        elif counter_penalty <= -4 or abs_score < 5:
            return -2
        elif counter_penalty <= -2:
            return -1
        else:
            return 0

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _chain_description(self, ict: dict) -> str:
        level     = ict.get("chain_level", 0)
        direction = ict.get("daily_bias", "?")
        lines     = []

        if level >= 1:
            struct = ict.get("swing_structure", "")
            lines.append(
                f"  ✅ L1 Daily Bias {direction}: EMA={ict.get('ema20_d1','?')} "
                f"struct={struct or 'N/A'}"
            )
        else:
            lines.append(f"  ❌ L1: {ict.get('stop_reason','not confirmed')}")
            return "\n".join(lines)

        if level >= 2:
            ssl  = ict.get("ssl_level", "?")
            bsl  = ict.get("bsl_level", "?")
            lrlr = " [LRLR✅]" if ict.get("lrlr_detected") else ""
            lines.append(
                f"  ✅ L2 BSL/SSL: BSL={bsl} SSL={ssl} "
                f"DOL={ict.get('dol_type','?')}@{ict.get('dol_price','?')}{lrlr}"
            )
        else:
            lines.append(f"  ❌ L2: {ict.get('stop_reason','not identified')}")
            return "\n".join(lines)

        if level >= 3:
            premium = []
            if ict.get("judas_valid"): premium.append("Judas✅")
            if ict.get("mss_valid"):   premium.append("MSS✅")
            if ict.get("mmxm_valid"):  premium.append("MMXM✅")
            if ict.get("smt_valid"):   premium.append("SMT✅")
            lines.append(
                f"  ✅ L3 Sweep+BOS: sweep@{ict.get('sweep_price','?')} "
                f"→ BOS@{ict.get('bos_level','?')} "
                f"[{', '.join(premium) or 'standard'}]"
            )
        else:
            lines.append(f"  ❌ L3: {ict.get('stop_reason','not confirmed')}")
            return "\n".join(lines)

        if level >= 4:
            entry_type = []
            if ict.get("in_ob_zone"): entry_type.append("OB")
            if ict.get("in_ote"):     entry_type.append("OTE")
            if ict.get("in_fvg"):     entry_type.append("FVG")
            bpr = " [BPR✅]" if ict.get("bpr_valid") else ""
            lines.append(
                f"  ✅ L4 Entry: {'+'.join(entry_type) or 'N/A'}{bpr} "
                f"R:R={ict.get('rr_ratio','?')}"
            )
        else:
            lines.append(f"  🟡 L4: partial — {ict.get('missing_elements','')}")

        return "\n".join(lines)

    def _abstain(self, asset: str, direction: str, reason: str) -> dict:
        logger.info(f"[{self.name}] {asset} {direction} → ABSTAIN: {reason}")
        return {
            "score":            0,
            "summary":          f"Abstain — {reason}",
            "bull_case":        "No valid ICT setup.",
            "bear_case":        "No valid ICT setup.",
            "confidence":       "low",
            "details":          f"[{self.name}] {asset} {direction} — Abstain: {reason}",
            "quality_delta":    0,
            "ict_chain_level":  0,
            "det_score":        0,
            "setup_type":       "N/A",
            "rag_chunks_count": 0,
            "ssl_level":        None,
            "bsl_level":        None,
            "ob_zone":          (None, None),
            "rr_ratio":         None,
            "invalidation":     "N/A",
            "counter_penalty":  0,
            "counter_argument": "N/A",
        }

    async def run(self, context: dict) -> AgentResult:
        """Override BaseAgent.run() to put analyze() output into raw_data."""
        logger.info(
            f"[{self.name}] Starting analysis — asset: {context.get('asset')} "
            f"dir: {context.get('direction')}"
        )
        try:
            data   = await self.collect_data(context)
            result = await self.analyze(data, context)

            raw_data = {**data, **result}
            raw_data.pop("price_data",       None)
            raw_data.pop("strategy_results", None)

            # Convert ±20 → 0-100 for AgentResult compatibility
            score_raw = float(result.get("score", 0))
            score_100 = 50.0 + (score_raw / 20.0) * 50.0
            score_100 = max(0.0, min(100.0, score_100))

            agent_result = AgentResult(
                agent      = self.name,
                score      = score_100,
                direction  = context.get("direction", ""),
                summary    = result.get("summary",    ""),
                bull_case  = result.get("bull_case",  ""),
                bear_case  = result.get("bear_case",  ""),
                confidence = result.get("confidence", "medium"),
                details    = result.get("details",    ""),
                raw_data   = raw_data,
            )
            logger.info(
                f"[{self.name}] Completed — "
                f"score: {score_100:.1f}/100 | "
                f"quality_delta: {result.get('quality_delta', 0):+d} | "
                f"dir: {context.get('direction')} "
                f"confidence: {agent_result.confidence}"
            )
            return agent_result

        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
            return AgentResult(
                agent=self.name, score=50.0,
                direction=context.get("direction", ""),
                summary=f"Error: {e}", bull_case="", bear_case="",
                confidence="low", details=str(e),
                raw_data={"quality_delta": 0, "ict_chain_level": 0, "error": str(e)},
            )
