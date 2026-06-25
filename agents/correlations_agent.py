"""
agents/correlations_agent.py
CorrelationsAgent — Agent 5
Score: -10 / +10  (fully deterministic in v3)

Analyzes real-time correlations between the target asset and market monitors
to autonomously determine the direction suggested by the inter-market picture.

Score convention (consistent with all other agents):
  +10 → strong BULLISH signal on the asset
  -10 → strong BEARISH signal on the asset
    0 → neutral / mixed picture

Does NOT receive direction as input — determines direction autonomously
from correlations, like COT+VolumeAgent.

Data: yfinance H4 (30 days), 30-min inter-call cache.
v3: 100% deterministic (LLM removed — see PARTIAL note on _gpt4o_analysis).

PARTIAL: Correlation cluster definitions and deterministic scoring are real.
The _gpt4o_analysis method is stubbed. See paper Section 2.3.
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from utils.yfinance_lock import YF_DOWNLOAD_LOCK
from agents.base_agent import BaseAgent, AgentResult

# ─────────────────────────────────────────────────────────────────────────────
# Ticker mapping (internal name → yfinance ticker)
# ─────────────────────────────────────────────────────────────────────────────
YFINANCE_MAP: dict[str, str] = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "GBPJPY": "GBPJPY=X", "USDCAD": "USDCAD=X", "EURAUD": "EURAUD=X",
    "EURGBP": "EURGBP=X", "NZDJPY": "NZDJPY=X", "EURCHF": "EURCHF=X",
    "XAUUSD": "GC=F",     "XAGUSD": "SI=F",      "MGC":    "GC=F",
    "BTCUSD": "BTC-USD",  "ETHUSD": "ETH-USD",
    "MES":    "ES=F",
    "NAS100": "NQ=F",     "MCL":    "CL=F",      "USOUSD": "CL=F",
    "SP500":  "ES=F",
    "6E":     "EURUSD=X",
    # Monitors
    "USDX":   "DX-Y.NYB",
    "AUDUSD": "AUDUSD=X", "NZDUSD": "NZDUSD=X",  "GBPCAD": "GBPCAD=X",
    "NZDCHF": "NZDCHF=X", "AUDJPY": "AUDJPY=X",
    "CHFJPY": "CHFJPY=X",
    "GER40":  "^GDAXI",   "US2000": "^RUT",      "DJ30":   "^DJI",
}

# ─────────────────────────────────────────────────────────────────────────────
# Correlation matrix per asset
#
# Format: (monitor_name, direction_mult, strength)
#
# direction_mult defines the STRUCTURAL relationship between monitor and asset:
#   +1 = they move in the same direction
#        → if monitor rises, asset tends to rise → bullish signal
#   -1 = they move in opposite directions
#        → if monitor falls, asset tends to rise → bullish signal
#
# Example EURUSD:
#   USDX direction_mult=-1 → USDX falls → EURUSD rises → bullish EURUSD
#   GBPUSD direction_mult=+1 → GBPUSD rises → EURUSD rises → bullish EURUSD
#
# The final score is POSITIVE if monitors suggest upside on the asset,
# NEGATIVE if they suggest downside.
# ─────────────────────────────────────────────────────────────────────────────
CORRELATION_MATRIX: dict[str, list[tuple[str, int, str]]] = {
    "EURUSD": [
        ("USDX",   -1, "high"),    # USDX↓ → EURUSD↑
        ("GBPUSD", +1, "high"),    # GBPUSD↑ → EURUSD↑ (USD weak across board)
        ("AUDUSD", +1, "medium"),  # risk-on proxy
        ("NZDUSD", +1, "medium"),
    ],
    "GBPUSD": [
        ("USDX",   -1, "high"),
        ("EURUSD", +1, "high"),
        ("AUDUSD", +1, "medium"),
        ("GBPCAD", +1, "medium"),
    ],
    "USDJPY": [
        ("USDX",   +1, "high"),    # USDX↑ → USDJPY↑
        ("AUDJPY", +1, "medium"),  # risk appetite → JPY weakness
        ("NZDJPY", +1, "medium"),
        ("GBPJPY", +1, "medium"),
    ],
    "GBPJPY": [
        ("AUDJPY", +1, "high"),
        ("USDJPY", +1, "high"),
        ("USDX",   -1, "medium"),  # via GBP leg
    ],
    "EURAUD": [
        ("AUDUSD", -1, "high"),    # AUD↑ → EURAUD↓
        ("EURUSD", +1, "medium"),
        ("AUDJPY", -1, "medium"),  # AUD strength → EURAUD↓
    ],
    "USDCAD": [
        ("USDX",   +1, "high"),
        ("MCL",    -1, "high"),    # oil↑ → CAD↑ → USDCAD↓
        ("AUDUSD", -1, "medium"),
    ],
    "EURGBP": [
        ("EURUSD", +1, "high"),
        ("GBPUSD", -1, "high"),    # GBP↑ → EURGBP↓
    ],
    "NZDJPY": [
        ("AUDJPY", +1, "high"),
        ("NZDUSD", +1, "medium"),
        ("USDX",   -1, "medium"),
    ],
    "EURCHF": [
        ("EURUSD", +1, "high"),
        ("USDX",   -1, "medium"),
    ],
    "XAUUSD": [
        ("USDX",   -1, "high"),    # USDX↓ → gold↑
        ("XAGUSD", +1, "high"),    # silver↑ → gold↑
        ("EURUSD", +1, "medium"),  # USD proxy
    ],
    "XAGUSD": [
        ("USDX",   -1, "high"),
        ("XAUUSD", +1, "high"),
        ("MCL",    +1, "medium"),  # commodity complex
    ],
    "MGC": [
        ("USDX",   -1, "high"),    # same underlying as XAUUSD — GC=F
        ("XAGUSD", +1, "high"),
        ("EURUSD", +1, "medium"),
    ],
    "BTCUSD": [
        ("ETHUSD", +1, "high"),
        ("AUDUSD", +1, "low"),     # risk-on proxy
    ],
    "ETHUSD": [
        ("BTCUSD", +1, "high"),
        ("AUDUSD", +1, "low"),
    ],
    "MES": [
        ("AUDJPY", +1, "medium"),  # risk-on → equities↑
        ("USDX",   -1, "low"),
    ],
    "SP500": [
        ("DJ30",   +1, "high"),    # Dow: highly correlated with S&P
        ("AUDJPY", +1, "medium"),
        ("USDX",   -1, "low"),
    ],
    "MCL": [
        ("USDX",   -1, "high"),    # oil priced in USD
        ("AUDUSD", +1, "medium"),  # commodity currencies
        ("USDCAD", -1, "medium"),  # CAD↑ → USDCAD↓ → oil↑
    ],
    "USOUSD": [
        ("USDX",   -1, "high"),
        ("AUDUSD", +1, "medium"),
    ],
    "NAS100": [
        ("MES",    +1, "high"),
        ("BTCUSD", +1, "medium"),
        ("USDX",   -1, "low"),
    ],
    "6E": [
        ("USDX",   -1, "high"),    # same underlying as EURUSD — EURUSD=X
        ("GBPUSD", +1, "high"),
        ("AUDUSD", +1, "medium"),
        ("NZDUSD", +1, "medium"),
    ],
}

STRENGTH_WEIGHT: dict[str, float] = {
    "high":   1.0,
    "medium": 0.6,
    "low":    0.3,
}


# ─────────────────────────────────────────────────────────────────────────────
class CorrelationsAgent(BaseAgent):
    """
    Agent 5 — CorrelationsAgent.

    Analyzes inter-market correlations and autonomously determines
    whether the picture suggests a BULLISH or BEARISH bias on the asset.

    Score: +10 = strong bullish | -10 = strong bearish | 0 = neutral
    v3: 100% deterministic (LLM removed).
    Does NOT use direction as input — produces an independent signal.

    PARTIAL: cluster definitions and deterministic scoring logic are real.
    The _gpt4o_analysis method stub is kept for interface completeness.
    """

    AGENT_NAME  = "CorrelationsAgent"
    MODEL       = "gpt-4o-mini"   # retained for interface; not called in v3
    SCORE_RANGE = (-10, 10)

    def __init__(self):
        super().__init__()
        self._cache: dict[str, pd.DataFrame] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_sec: int = 30 * 60   # 30 minutes

    # ─── collect_data ─────────────────────────────────────────────────────────

    async def collect_data(self, context: dict) -> dict:
        """
        Fetch H4 data for target asset + all monitors in the matrix.

        Args:
            context: {"asset": str, ...}  — direction NOT used
        Returns:
            {"asset": str, "price_data": {name: DataFrame}}
        """
        asset    = str(context.get("asset", "EURUSD")).upper()
        monitors = CORRELATION_MATRIX.get(asset, [])
        tickers_needed = {asset} | {m[0] for m in monitors} | {"USDX"}

        now = datetime.now(timezone.utc)
        cache_valid = (
            self._cache_time is not None
            and (now - self._cache_time).total_seconds() < self._cache_ttl_sec
        )

        price_data: dict[str, pd.DataFrame] = {}
        to_fetch: list[str] = []

        for t in tickers_needed:
            if cache_valid and t in self._cache:
                price_data[t] = self._cache[t]
            else:
                to_fetch.append(t)

        if to_fetch:
            logger.info(f"[{self.name}] Fetching yfinance: {', '.join(to_fetch)}")
            fetched = await asyncio.to_thread(self._fetch_yfinance_batch, to_fetch)
            price_data.update(fetched)
            self._cache.update(fetched)
            self._cache_time = now

        return {"asset": asset, "price_data": price_data}

    def _fetch_yfinance_batch(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        import time as _time
        result: dict[str, pd.DataFrame] = {}
        for name in tickers:
            yf_ticker = YFINANCE_MAP.get(name)
            if not yf_ticker:
                logger.warning(f"[{self.name}] Ticker not in map: {name}")
                continue
            df = None
            for _attempt in range(2):
                try:
                    with YF_DOWNLOAD_LOCK:
                        df = yf.download(
                            yf_ticker, period="30d", interval="4h",
                            progress=False, auto_adjust=True,
                        )
                    break
                except Exception as e:
                    if _attempt == 0 and "dictionary changed size" in str(e):
                        logger.warning(f"[{self.name}] Race condition on {name} — retry in 0.4s")
                        _time.sleep(0.4)
                        continue
                    logger.error(f"[{self.name}] Fetch error {name}: {e}")
                    df = None
                    break
            if df is None or df.empty:
                if df is not None:
                    logger.warning(f"[{self.name}] Empty data: {name} ({yf_ticker})")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]
            df.columns = [c.lower() for c in df.columns]
            cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            result[name] = df[cols].copy()
            logger.debug(f"[{self.name}] {name}: {len(df)} candles H4")
        return result

    # ─── analyze ──────────────────────────────────────────────────────────────

    async def analyze(self, data: dict, context: dict) -> dict:
        """
        Autonomously determine directional bias on the asset by observing
        movements of correlated monitors.

        Logic:
          For each monitor → monitor_chg × direction_mult = directional contribution
          If contribution > 0 → bullish signal on asset
          If contribution < 0 → bearish signal on asset

        Example USDJPY:
          USDX direction_mult=+1, USDX falls (-) → negative contribution → bearish USDJPY
          AUDJPY direction_mult=+1, AUDJPY falls (-) → negative contribution → bearish USDJPY
          → negative final score = bearish USDJPY ✅
        """
        asset      = data.get("asset", str(context.get("asset", "EURUSD")).upper())
        price_data = data.get("price_data", {})
        monitors   = CORRELATION_MATRIX.get(asset, [])

        if not monitors:
            return self._empty_result("Asset not in correlation matrix")

        asset_df = price_data.get(asset)
        if asset_df is None or len(asset_df) < 20:
            return self._empty_result("Insufficient data for target asset")

        # ── Deterministic calculations ─────────────────────────────────────────
        monitor_results: list[dict] = []
        summary_lines:  list[str]  = []

        for monitor_name, dir_mult, strength in monitors:
            monitor_df = price_data.get(monitor_name)
            if monitor_df is None or len(monitor_df) < 20:
                summary_lines.append(f"  ⚠️  {monitor_name}: DATA MISSING")
                continue

            close_col = "close" if "close" in asset_df.columns else "Close"
            merged = (
                asset_df[close_col].rename("asset").to_frame()
                .join(monitor_df[close_col].rename("monitor"), how="inner")
                .dropna()
            )
            if len(merged) < 20:
                summary_lines.append(f"  ⚠️  {monitor_name}: INSUFFICIENT OVERLAP ({len(merged)}p)")
                continue

            n = len(merged)

            # Rolling correlation
            _corr_20_raw = merged["asset"].rolling(20).corr(merged["monitor"]).iloc[-1]
            _corr_60_raw = (merged["asset"].rolling(min(60, n)).corr(merged["monitor"]).iloc[-1]
                            if n >= 40 else _corr_20_raw)
            if np.isnan(_corr_20_raw) or np.isnan(_corr_60_raw):
                summary_lines.append(f"  ⚠️  {monitor_name}: CORRELATION NaN — skip")
                continue
            corr_20 = float(_corr_20_raw)
            corr_60 = float(_corr_60_raw)

            # Monitor % change over 5p and 20p H4
            _m_last = merged["monitor"].iloc[-1]
            _m_5    = merged["monitor"].iloc[-5]
            _m_20   = merged["monitor"].iloc[-20]
            if np.isnan(_m_last) or np.isnan(_m_5) or np.isnan(_m_20) or _m_5 == 0 or _m_20 == 0:
                summary_lines.append(f"  ⚠️  {monitor_name}: NaN/ZERO prices — skip")
                continue
            monitor_chg5  = float((_m_last - _m_5)  / _m_5  * 100)
            monitor_chg20 = float((_m_last - _m_20) / _m_20 * 100)

            # Directional contribution on asset:
            # monitor_chg × direction_mult > 0 → bullish on asset
            signal_5p  = monitor_chg5  * dir_mult
            signal_20p = monitor_chg20 * dir_mult

            # Magnitude-weighted scoring via tanh (self-calibrating across assets)
            monitor_pct_returns = merged["monitor"].pct_change().dropna() * 100
            _std_raw = monitor_pct_returns.rolling(20).std().iloc[-1]
            if np.isnan(_std_raw):
                summary_lines.append(f"  ⚠️  {monitor_name}: STD NaN — skip")
                continue
            std_1p      = max(float(_std_raw), 1e-6)
            std_5p_vol  = std_1p * (5  ** 0.5)
            std_20p_vol = std_1p * (20 ** 0.5)

            norm_5p  = float(np.tanh(signal_5p  / std_5p_vol))
            norm_20p = float(np.tanh(signal_20p / std_20p_vol))
            if np.isnan(norm_5p) or np.isnan(norm_20p):
                summary_lines.append(f"  ⚠️  {monitor_name}: TANH NaN — skip")
                continue

            # 60% short-term (5p) + 40% medium-term (20p)
            combined      = 0.6 * norm_5p + 0.4 * norm_20p
            w             = STRENGTH_WEIGHT[strength]
            partial_score = w * combined
            both_agree    = (signal_5p > 0) == (signal_20p > 0)

            bias  = "BULLISH" if partial_score > 0 else ("BEARISH" if partial_score < 0 else "NEUTRAL")
            emoji = "📈" if partial_score > 0 else ("📉" if partial_score < 0 else "➡️")

            monitor_results.append({
                "monitor":        monitor_name,
                "direction_mult": dir_mult,
                "strength":       strength,
                "corr_20":        round(corr_20, 3),
                "corr_60":        round(corr_60, 3),
                "corr_delta":     round(corr_20 - corr_60, 3),
                "monitor_chg5p":  round(monitor_chg5, 4),
                "monitor_chg20p": round(monitor_chg20, 4),
                "signal_5p":      round(signal_5p, 4),
                "signal_20p":     round(signal_20p, 4),
                "both_agree":     both_agree,
                "partial_score":  round(partial_score, 3),
                "bias":           bias,
            })

            summary_lines.append(
                f"  {emoji} {monitor_name} [{strength}] → {bias} | "
                f"Mon chg: 5p={monitor_chg5:+.3f}% 20p={monitor_chg20:+.3f}% | "
                f"Corr20={corr_20:+.2f} Corr60={corr_60:+.2f} (Δ={corr_20-corr_60:+.2f})"
            )

        # ── Deterministic score normalization ─────────────────────────────────
        if monitor_results:
            raw_score      = sum(r["partial_score"] for r in monitor_results)
            max_possible   = sum(STRENGTH_WEIGHT[r["strength"]] for r in monitor_results)
            det_score      = (raw_score / max_possible) * 8.0 if max_possible > 0 else 0.0
            bullish_count  = sum(1 for r in monitor_results if r["partial_score"] > 0)
            bearish_count  = sum(1 for r in monitor_results if r["partial_score"] < 0)
            total_monitors = len(monitor_results)
        else:
            det_score = 0.0
            bullish_count = bearish_count = total_monitors = 0

        if np.isnan(det_score):
            logger.warning(f"[{self.name}] {asset}: det_score=nan — reset to 0.0")
            det_score = 0.0

        # ── Final score — 100% deterministic (v3: LLM removed) ───────────────
        final_score = round(
            max(float(self.SCORE_RANGE[0]), min(float(self.SCORE_RANGE[1]),
            det_score * (10.0 / 8.0))), 1
        )

        # v3: contribution for rr_modifier Layer 1 (-0.10 to +0.10)
        corr_contribution = round(max(-0.10, min(0.10, final_score / 10.0 * 0.10)), 3)

        # Divergence alert: correlated assets moving in opposite direction
        divergence_alert = False
        if monitor_results:
            agree_count  = (sum(1 for r in monitor_results if r["partial_score"] > 0)
                            if final_score > 0
                            else sum(1 for r in monitor_results if r["partial_score"] < 0))
            if (len(monitor_results) - agree_count) > agree_count:
                divergence_alert     = True
                corr_contribution    = round(corr_contribution - 0.05, 3)

        logger.info(
            f"[{self.name}] {asset} | "
            f"Bull:{bullish_count} Bear:{bearish_count}/{total_monitors} | "
            f"Det: {det_score:+.2f} → Final={final_score:+.1f} | "
            f"contribution={corr_contribution:+.3f}"
        )

        overall_bias   = "BULLISH" if final_score > 0 else ("BEARISH" if final_score < 0 else "NEUTRAL")
        corr_breakdown = "\n".join(summary_lines)

        summary = (
            f"Bias: {overall_bias} | "
            f"Bull:{bullish_count} Bear:{bearish_count}/{total_monitors} monitors | "
            f"contribution={corr_contribution:+.3f}"
        )
        bull_case = f"Bullish signals from {bullish_count}/{total_monitors} monitors (det={det_score:+.2f})"
        bear_case = f"Bearish signals from {bearish_count}/{total_monitors} monitors (det={det_score:+.2f})"

        details = (
            f"=== CORRELATIONS AGENT v3 — {asset} ===\n"
            f"Final score     : {final_score:+.1f} / 10  → {overall_bias}\n"
            f"Det (100%)      : {det_score:+.2f}\n"
            f"contribution    : {corr_contribution:+.3f}\n"
            f"Divergence alert: {divergence_alert}\n"
            f"Monitors        : {bullish_count} bullish / {bearish_count} bearish / {total_monitors} total\n\n"
            f"--- Monitor breakdown ---\n"
            f"{corr_breakdown}"
        )

        return {
            "score":              final_score,
            "summary":            summary,
            "bull_case":          bull_case,
            "bear_case":          bear_case,
            "confidence":         "high" if total_monitors >= 2 else "low",
            "details":            details,
            "det_score":          round(det_score, 2),
            "qual_score":         0,       # v3: LLM removed
            "bullish_count":      bullish_count,
            "bearish_count":      bearish_count,
            "total_monitors":     total_monitors,
            "overall_bias":       overall_bias,
            "monitor_results":    monitor_results,
            "correlation_regime": "deterministic_v3",
            "corr_contribution":  corr_contribution,
            "divergence_alert":   divergence_alert,
            "divergence_veto":    divergence_alert,   # backward compatibility
        }

    # ─── GPT-4o (stubbed — removed in v3) ────────────────────────────────────

    async def _gpt4o_analysis(
        self, asset: str, monitor_results: list[dict],
        summary_lines: list[str], det_score: float,
    ) -> dict:
        """
        # Implementation intentionally omitted.
        # Production (v2): GPT-4o interpreted the correlation picture and
        # confirmed or corrected the deterministic directional bias.
        # v3: removed entirely — 100% deterministic scoring.
        # See paper Section 3.5c for rationale.
        """
        return {
            "score": 0, "confidence": "LOW",
            "correlation_regime": "MIXED",
            "top_bullish_signal": "none",
            "top_bearish_signal": "none",
            "rationale": "[stub — LLM removed in v3]",
        }

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _empty_result(self, reason: str) -> dict:
        return {
            "score": 0, "summary": reason,
            "bull_case": "Data not available.",
            "bear_case": "Data not available.",
            "confidence": "low", "details": f"[{self.name}] {reason}",
            "det_score": 0.0, "qual_score": 0,
            "bullish_count": 0, "bearish_count": 0, "total_monitors": 0,
            "overall_bias": "NEUTRAL", "monitor_results": [],
            "correlation_regime": "N/A",
            "corr_contribution": 0.0,
            "divergence_alert": False,
            "divergence_veto": False,
        }

    async def run(self, context: dict) -> AgentResult:
        """Override BaseAgent.run() to inject analyze() fields into raw_data."""
        logger.info(
            f"[{self.name}] Starting analysis — asset: {context.get('asset')} "
            f"dir: {context.get('direction')}"
        )
        try:
            data   = await self.collect_data(context)
            result = await self.analyze(data, context)

            score = max(self.SCORE_RANGE[0], min(self.SCORE_RANGE[1], result.get("score", 0)))

            extra_fields = {
                k: v for k, v in result.items()
                if k not in ("score", "summary", "bull_case", "bear_case", "confidence", "details")
            }
            raw_data = {**data, **extra_fields}
            raw_data.pop("price_data", None)

            agent_result = AgentResult(
                agent=self.name,
                score=score,
                summary=result.get("summary", ""),
                bull_case=result.get("bull_case", ""),
                bear_case=result.get("bear_case", ""),
                confidence=result.get("confidence", "medium"),
                details=result.get("details", ""),
                raw_data=raw_data,
            )

            logger.info(f"[{self.name}] Completed — score: {score:+.1f} confidence: {agent_result.confidence}")
            return agent_result

        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
            return AgentResult(
                agent=self.name, score=0,
                summary=f"Analysis error: {str(e)[:50]}",
                bull_case="", bear_case="",
                confidence="low", details="", error=str(e),
            )

    async def run_full(self, assets: Optional[list[str]] = None) -> dict[str, AgentResult]:
        """Full analysis across all assets for briefing/dashboard."""
        if assets is None:
            assets = list(CORRELATION_MATRIX.keys())
        logger.info(f"[{self.name}] run_full() — {len(assets)} assets")
        results: dict[str, AgentResult] = {}
        for asset in assets:
            results[asset] = await self.run({"asset": asset})
            r = results[asset]
            bias = r.raw_data.get("overall_bias", "?")
            logger.info(f"[{self.name}] {asset} → {r.score:+.1f} ({bias})")
            await asyncio.sleep(0.4)
        return results
