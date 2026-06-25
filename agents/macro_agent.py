"""
agents/macro_agent.py v2
MacroAgent — Complete macroeconomic analysis for all tradable assets.

Architecture (three sub-agents):
  - MacroDataCollector:      fetches FRED, ForexFactory, Reuters, central bank RSS
  - MacroDeterministicScorer: computes rule-based scores (60% of total)
  - COTAnalyzer:             processes CFTC positioning data (20% of total)
  - Claude qualitative:      narrative adjustment from central bank communications
                             (20% of total — stubbed in this version)

PARTIAL: Data collection, deterministic scoring, FRED series selection, COT
computation, and caching architecture are real. The Claude qualitative prompt
and its internal scoring rules are intentionally omitted.
See paper Section 3.3 for design rationale.

Output: score per asset + ranking for trade selection.

Sources:
  - FRED API (US + international macro data)
  - ForexFactory RSS (forex economic calendar)
  - Reuters RSS (breaking macro news)
  - yfinance (commodity prices: oil, copper, gold)
  - Central bank RSS: FED, ECB, BOE, BOJ, RBA, RBNZ, BOC, SNB
"""
import os
import json
import re
import asyncio
import feedparser
import yfinance as yf
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional
from bs4 import BeautifulSoup
from loguru import logger

try:
    from agents.web_context import fetch_all_web_context as _fetch_web_ctx
    _WEB_CTX_AVAILABLE = True
except ImportError:
    _WEB_CTX_AVAILABLE = False

import anthropic

try:
    from agents.base_agent import BaseAgent, AgentResult
except ImportError:
    from base_agent import BaseAgent, AgentResult

import zipfile
import io
from pathlib import Path
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════
# ASSET AND CURRENCY CONFIGURATION
# ══════════════════════════════════════════════════════════════

CURRENCY_CONFIG = {
    "USD": {
        "bank":      "FED",
        "bank_name": "Federal Reserve",
        "rss":       "https://www.federalreserve.gov/feeds/press_all.xml",
        "fred_rate": "FEDFUNDS",
        "fred_cpi":  "CPIAUCSL",
        "fred_unemp":"UNRATE",
        "fred_gdp":  "GDP",
        "yf_bond":   "^TNX",
    },
    "EUR": {
        "bank":      "ECB",
        "bank_name": "European Central Bank",
        "rss":       "https://www.ecb.europa.eu/rss/fst.html",
        "fred_rate": "ECBDFR",
        "fred_cpi":  "CP0000EZ19M086NEST",
        "yf_bond":   "DE10YT=RR",
    },
    "GBP": {
        "bank":      "BOE",
        "bank_name": "Bank of England",
        "rss":       "https://www.bankofengland.co.uk/rss/news",
        "fred_rate": "BOERUKM",
        "yf_bond":   "GBGB10YD=RR",
    },
    "JPY": {
        "bank":      "BOJ",
        "bank_name": "Bank of Japan",
        "rss":       "https://www.boj.or.jp/en/rss/",
        "fred_rate": "IRSTCI01JPM156N",
        "fred_cpi":  "JPNCPIALLMINMEI",
    },
    "AUD": {
        "bank":      "RBA",
        "bank_name": "Reserve Bank of Australia",
        "rss":       "https://www.rba.gov.au/rss/rss-cb-speeches.xml",
        "fred_rate": "IRSTCI01AUM156N",
        "yf_commodity": "HG=F",
    },
    "NZD": {
        "bank":      "RBNZ",
        "bank_name": "Reserve Bank of New Zealand",
        "rss":       "https://www.rbnz.govt.nz/hub/news/feed",
        "fred_rate": "IRSTCI01NZM156N",
    },
    "CAD": {
        "bank":      "BOC",
        "bank_name": "Bank of Canada",
        "rss":       "https://www.bankofcanada.ca/feed/",
        "fred_rate": "IRSTCI01CAM156N",
        "yf_commodity": "CL=F",
    },
    "CHF": {
        "bank":      "SNB",
        "bank_name": "Swiss National Bank",
        "rss":       "https://www.snb.ch/en/publications/news/news-releases/feed",
        "fred_rate": "IRSTCI01CHM156N",
    },
}

ASSET_CONFIG = {
    "EURUSD": {"tier": 1, "base": "EUR", "quote": "USD", "type": "forex",
               "driver": "ECB/FED rate differential + Eurozone vs USA inflation",
               "key_data": ["FEDFUNDS", "ECBDFR", "CPIAUCSL", "DTWEXBGS"]},
    "GBPUSD": {"tier": 1, "base": "GBP", "quote": "USD", "type": "forex",
               "driver": "BOE vs FED + persistent UK inflation",
               "key_data": ["FEDFUNDS", "BOERUKM", "CPIAUCSL"]},
    "USDJPY": {"tier": 1, "base": "USD", "quote": "JPY", "type": "forex",
               "driver": "USD/JP rate differential + ultra-accommodative BOJ policy",
               "key_data": ["FEDFUNDS", "IRSTCI01JPM156N", "DGS10"]},
    "GBPJPY": {"tier": 1, "base": "GBP", "quote": "JPY", "type": "forex",
               "driver": "Cross EUR/JPY — risk appetite + BOE vs BOJ",
               "key_data": ["BOERUKM", "IRSTCI01JPM156N"]},
    "XAUUSD": {"tier": 1, "base": "XAU", "quote": "USD", "type": "commodity",
               "driver": "US real rates (T10Y2Y) + inverse DXY + risk-off",
               "key_data": ["DTWEXBGS", "T10Y2Y", "FEDFUNDS", "VIXCLS"]},
    "BTCUSD": {"tier": 1, "base": "BTC", "quote": "USD", "type": "crypto",
               "driver": "Risk appetite + FED liquidity + tech sentiment",
               "key_data": ["FEDFUNDS", "VIXCLS", "M2SL"]},
    "ETHUSD": {"tier": 1, "base": "ETH", "quote": "USD", "type": "crypto",
               "driver": "Correlated with BTC + DeFi activity",
               "key_data": ["FEDFUNDS", "VIXCLS"]},
    "EURAUD": {"tier": 1, "base": "EUR", "quote": "AUD", "type": "forex",
               "driver": "ECB vs RBA + commodity cycle (copper, minerals)",
               "key_data": ["ECBDFR", "IRSTCI01AUM156N"]},
    "USDCAD": {"tier": 1, "base": "USD", "quote": "CAD", "type": "forex",
               "driver": "FED vs BOC + crude oil price (WTI)",
               "key_data": ["FEDFUNDS", "IRSTCI01CAM156N"]},
    "EURGBP": {"tier": 1, "base": "EUR", "quote": "GBP", "type": "forex",
               "driver": "ECB vs BOE + UK post-Brexit policy",
               "key_data": ["ECBDFR", "BOERUKM"]},
    "NZDJPY": {"tier": 1, "base": "NZD", "quote": "JPY", "type": "forex",
               "driver": "Risk-on/off barometer + RBNZ vs BOJ",
               "key_data": ["IRSTCI01NZM156N", "IRSTCI01JPM156N", "VIXCLS"]},
    "EURCHF": {"tier": 1, "base": "EUR", "quote": "CHF", "type": "forex",
               "driver": "ECB vs SNB + safe-haven CHF in crisis",
               "key_data": ["ECBDFR", "IRSTCI01CHM156N", "VIXCLS"]},
    "MGC":    {"tier": 1, "base": "XAU", "quote": "USD", "type": "futures",
               "driver": "Identical to XAUUSD — IB micro futures",
               "key_data": ["DTWEXBGS", "T10Y2Y", "FEDFUNDS", "VIXCLS"]},
    "MES":    {"tier": 1, "base": "SP500", "quote": "USD", "type": "futures",
               "driver": "S&P500 earnings + FED policy + US growth",
               "key_data": ["FEDFUNDS", "CPIAUCSL", "UNRATE", "T10Y2Y"]},
    "XAGUSD": {"tier": 2, "base": "XAG", "quote": "USD", "type": "commodity",
               "driver": "Correlated with gold + industrial demand",
               "key_data": ["DTWEXBGS", "T10Y2Y"]},
    "6E":     {"tier": 2, "base": "EUR", "quote": "USD", "type": "futures",
               "driver": "Identical to EURUSD — CME futures",
               "key_data": ["FEDFUNDS", "ECBDFR", "CPIAUCSL"]},
    "MCL":    {"tier": 2, "base": "OIL", "quote": "USD", "type": "futures",
               "driver": "OPEC+ supply + global demand + USD",
               "key_data": ["DTWEXBGS"]},
    "NAS100": {"tier": 2, "base": "NASDAQ", "quote": "USD", "type": "futures",
               "driver": "Tech earnings + FED rates + growth vs value",
               "key_data": ["FEDFUNDS", "DGS10", "VIXCLS"]},
    "NZDUSD": {"tier": 2, "base": "NZD", "quote": "USD", "type": "forex",
               "driver": "RBNZ vs FED + commodity/risk sentiment",
               "key_data": ["FEDFUNDS"]},
    "AUDJPY": {"tier": 2, "base": "AUD", "quote": "JPY", "type": "forex",
               "driver": "Risk appetite + RBA vs BOJ",
               "key_data": ["IRSTCI01JPM156N", "VIXCLS"]},
    "CHFJPY": {"tier": 2, "base": "CHF", "quote": "JPY", "type": "forex",
               "driver": "SNB vs BOJ + safe-haven flows",
               "key_data": ["IRSTCI01JPM156N"]},
    "GER40":  {"tier": 2, "base": "DAX", "quote": "EUR", "type": "futures",
               "driver": "ECB rates + Eurozone growth + risk appetite",
               "key_data": ["ECBDFR", "VIXCLS"]},
    "US2000": {"tier": 2, "base": "RUSSELL", "quote": "USD", "type": "futures",
               "driver": "Small-cap + FED rates + US domestic economy",
               "key_data": ["FEDFUNDS", "DGS10", "VIXCLS"]},
    "DJ30":   {"tier": 2, "base": "DOW", "quote": "USD", "type": "futures",
               "driver": "Large-cap value + FED rates + risk appetite",
               "key_data": ["FEDFUNDS", "DGS10", "VIXCLS"]},
}

FRED_CORE_SERIES = {
    "DTWEXBGS":  "Dollar Index (DXY)",
    "VIXCLS":    "VIX — market volatility",
    "T10Y2Y":    "10Y-2Y spread (yield curve)",
    "FEDFUNDS":  "Fed Funds Rate",
    "CPIAUCSL":  "US CPI",
    "UNRATE":    "US Unemployment",
    "DGS10":     "Treasury 10Y yield",
    "M2SL":      "M2 Money Supply",
}

FRED_EXTENDED_SERIES = {
    "ECBDFR":              "ECB Deposit Facility Rate",
    "BOERUKM":             "BOE Bank Rate",
    "IRSTCI01JPM156N":     "BOJ Policy Rate",
    "IRSTCI01AUM156N":     "RBA Cash Rate",
    "IRSTCI01CAM156N":     "BOC Overnight Rate",
    "IRSTCI01CHM156N":     "SNB Policy Rate",
    "IRSTCI01NZM156N":     "RBNZ OCR",
    "CP0000EZ19M086NEST":  "Eurozone HICP",
    "JPNCPIALLMINMEI":     "Japan CPI",
}

COMMODITY_TICKERS = {
    "CL=F":  "WTI Crude Oil",
    "HG=F":  "Copper (AUD/China proxy)",
    "GC=F":  "Gold spot",
    "SI=F":  "Silver spot",
}

# ══════════════════════════════════════════════════════════════
# CLAUDE SYSTEM PROMPT — qualitative narrative interpretation
# Implementation intentionally omitted.
# Production: prompts Claude with central bank communications and
# news feeds to produce per-asset narrative adjustments.
# ══════════════════════════════════════════════════════════════

MACRO_QUALITATIVE_PROMPT = """[STUB — implementation intentionally omitted]
See paper Section 3.3 for design rationale.
Production: Senior macro analyst prompt evaluating central bank narrative,
Reuters news, and ForexFactory calendar to adjust deterministic scores.
"""

# ══════════════════════════════════════════════════════════════
# COT — CFTC DATA
# ══════════════════════════════════════════════════════════════

COT_CACHE_DIR  = Path(__file__).parent.parent / "data" / "cot_cache"
COT_CACHE_DAYS = 7
MACRO_CACHE_SECONDS = 24 * 3600

_BASE_DATA_DIR        = Path(__file__).parent.parent / "data"
MACRO_CACHE_PATH      = _BASE_DATA_DIR / "macro_daily_cache.json"
MACRO_QUAL_CACHE_PATH = _BASE_DATA_DIR / "macro_qual_cache.json"
COT_PROCESSED_CACHE_PATH = _BASE_DATA_DIR / "cot_cache" / "cot_processed.json"

CFTC_FIN_URL = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
CFTC_DIS_URL = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"

CFTC_CODES = {
    "USD": "098662", "EUR": "099741", "GBP": "096742", "JPY": "097741",
    "CAD": "090741", "AUD": "232741", "NZD": "112741", "CHF": "092741",
    "MES": "13874A", "NAS": "20974+",
    "XAU": "088691", "XAG": "084691", "OIL": "067651", "BTC": "133741",
}

CFTC_ASSET_MAP = {
    "USD": {
        "bullish_assets": ["USDJPY", "USDCAD"],
        "bearish_assets": ["EURUSD", "GBPUSD", "EURAUD", "EURGBP", "NZDJPY",
                           "EURCHF", "XAUUSD", "MGC", "XAGUSD", "BTCUSD", "ETHUSD",
                           "6E", "MES", "NAS100"],
        "weight": 0.5,
    },
    "EUR": {"bullish_assets": ["EURUSD", "EURGBP", "EURAUD", "EURCHF", "6E"], "bearish_assets": []},
    "GBP": {"bullish_assets": ["GBPUSD", "GBPJPY"], "bearish_assets": ["EURGBP"]},
    "JPY": {"bullish_assets": [], "bearish_assets": ["USDJPY", "GBPJPY", "NZDJPY"], "inverted": True},
    "CAD": {"bullish_assets": [], "bearish_assets": ["USDCAD"], "inverted": True},
    "AUD": {"bullish_assets": [], "bearish_assets": ["EURAUD"], "inverted": True},
    "NZD": {"bullish_assets": ["NZDJPY"], "bearish_assets": []},
    "CHF": {"bullish_assets": [], "bearish_assets": ["EURCHF"], "inverted": True},
    "MES": {"bullish_assets": ["MES"], "bearish_assets": []},
    "NAS": {"bullish_assets": ["NAS100"], "bearish_assets": []},
    "XAU": {"bullish_assets": ["MGC"], "bearish_assets": []},
    "XAG": {"bullish_assets": ["XAGUSD"], "bearish_assets": []},
    "OIL": {"bullish_assets": ["MCL"], "bearish_assets": []},
    "BTC": {"bullish_assets": ["BTCUSD"], "bearish_assets": []},
}


class CFTCDataFetcher:
    """Downloads and caches CFTC weekly CSV files to disk."""

    def __init__(self):
        COT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, report_type: str, year: int) -> Path:
        return COT_CACHE_DIR / f"cot_{report_type}_{year}.csv"

    def _is_cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        age_days = (datetime.now().timestamp() - path.stat().st_mtime) / 86400
        return age_days < COT_CACHE_DAYS

    def _download_zip(self, url: str) -> Optional[pd.DataFrame]:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                logger.warning(f"[COT] HTTP {resp.status_code} for {url}")
                return None
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                csvs = [n for n in z.namelist() if n.endswith(".csv") or n.endswith(".txt")]
                if not csvs:
                    return None
                with z.open(csvs[0]) as f:
                    df = pd.read_csv(f, low_memory=False)
            logger.info(f"[COT] Downloaded {url} → {len(df)} rows")
            return df
        except Exception as e:
            logger.warning(f"[COT] Download error {url}: {e}")
            return None

    def get_financial_data(self, year: int = None) -> Optional[pd.DataFrame]:
        year  = year or datetime.now().year
        cache = self._cache_path("fin", year)
        if not self._is_cache_valid(cache):
            df = self._download_zip(CFTC_FIN_URL.format(year=year))
            if df is None:
                df = self._download_zip(CFTC_FIN_URL.format(year=year-1))
            if df is not None:
                df.to_csv(cache, index=False)
        df = pd.read_csv(cache, low_memory=False) if cache.exists() else None
        cache_prev = self._cache_path("fin", year - 1)
        if not self._is_cache_valid(cache_prev):
            df_prev = self._download_zip(CFTC_FIN_URL.format(year=year-1))
            if df_prev is not None:
                df_prev.to_csv(cache_prev, index=False)
        if cache_prev.exists():
            df_prev = pd.read_csv(cache_prev, low_memory=False)
            df = pd.concat([df, df_prev], ignore_index=True) if df is not None else df_prev
        return df

    def get_disaggregated_data(self, year: int = None) -> Optional[pd.DataFrame]:
        year  = year or datetime.now().year
        cache = self._cache_path("dis", year)
        if not self._is_cache_valid(cache):
            df = self._download_zip(CFTC_DIS_URL.format(year=year))
            if df is None:
                df = self._download_zip(CFTC_DIS_URL.format(year=year-1))
            if df is not None:
                df.to_csv(cache, index=False)
        df = pd.read_csv(cache, low_memory=False) if cache.exists() else None
        cache_prev = self._cache_path("dis", year - 1)
        if not self._is_cache_valid(cache_prev):
            df_prev = self._download_zip(CFTC_DIS_URL.format(year=year-1))
            if df_prev is not None:
                df_prev.to_csv(cache_prev, index=False)
        if cache_prev.exists():
            df_prev = pd.read_csv(cache_prev, low_memory=False)
            df = pd.concat([df, df_prev], ignore_index=True) if df is not None else df_prev
        return df


class COTAnalyzer:
    """Computes COT metrics — extracted from cot_volume_agent.py."""

    LONG_COLS  = ["Lev_Money_Positions_Long_All", "M_Money_Positions_Long_All",
                  "NonComm_Positions_Long_All", "Noncommercial Long", "Non-Commercial Long"]
    SHORT_COLS = ["Lev_Money_Positions_Short_All", "M_Money_Positions_Short_All",
                  "NonComm_Positions_Short_All", "Noncommercial Short", "Non-Commercial Short"]
    CFTC_CODE_COLS = ["CFTC_Contract_Market_Code", "CFTC_Market_Code",
                      "CFTC_Commodity_Code", "Market_Code"]
    DATE_COLS  = ["Report_Date_as_YYYY-MM-DD", "As_of_Date_In_Form_YYMMDD", "Report Date", "Date"]
    NAME_COLS  = ["Market_and_Exchange_Names", "Market and Exchange Names", "Commodity"]

    def _find_col(self, df: pd.DataFrame, candidates: list) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        df_cols_lower = {col.lower(): col for col in df.columns}
        for c in candidates:
            if c.lower() in df_cols_lower:
                return df_cols_lower[c.lower()]
        for c in candidates:
            needle  = c.lower().replace(" ", "_")
            matches = [col for col in df.columns if needle in col.lower().replace(" ", "_")]
            if matches:
                return matches[0]
        return None

    def extract_positioning(self, df: pd.DataFrame, cftc_code: str) -> Optional[dict]:
        if df is None or df.empty:
            return None
        code_col  = self._find_col(df, self.CFTC_CODE_COLS)
        long_col  = self._find_col(df, self.LONG_COLS)
        short_col = self._find_col(df, self.SHORT_COLS)
        date_col  = self._find_col(df, self.DATE_COLS)
        name_col  = self._find_col(df, self.NAME_COLS)
        if not all([code_col, long_col, short_col]):
            return None
        mask = df[code_col].astype(str).str.strip() == str(cftc_code).strip()
        rows = df[mask].copy()
        if rows.empty:
            return None
        if date_col:
            try:
                rows[date_col] = pd.to_datetime(rows[date_col], errors="coerce")
                rows = rows.sort_values(date_col, ascending=False)
            except Exception:
                pass
        rows[long_col]  = pd.to_numeric(rows[long_col],  errors="coerce").fillna(0)
        rows[short_col] = pd.to_numeric(rows[short_col], errors="coerce").fillna(0)
        rows["net"]     = rows[long_col] - rows[short_col]
        history    = rows["net"].values[:52]
        current    = float(history[0]) if len(history) > 0 else 0
        prev_week  = float(history[1]) if len(history) > 1 else current
        if len(history) >= 2:
            min_val    = float(np.min(history))
            max_val    = float(np.max(history))
            percentile = (current - min_val) / (max_val - min_val) * 100 if max_val != min_val else 50.0
        else:
            percentile = 50.0
        delta_weekly   = current - prev_week
        extreme_signal = ("extreme_long" if percentile > 85
                          else "extreme_short" if percentile < 15 else "none")
        trend_4w       = float(history[0]) - float(history[3]) if len(history) >= 4 else 0.0
        instrument_name = str(rows[name_col].iloc[0])[:50] if name_col and not rows.empty else ""
        return {
            "cftc_code": cftc_code, "instrument": instrument_name,
            "net_current": round(current, 0), "net_prev": round(prev_week, 0),
            "percentile_52w": round(percentile, 1), "delta_weekly": round(delta_weekly, 0),
            "trend_4w": round(trend_4w, 0), "extreme_signal": extreme_signal,
            "history_weeks": len(history),
        }

    def extract_by_name(self, df: pd.DataFrame, keyword: str) -> Optional[dict]:
        if df is None or df.empty:
            return None
        name_col = self._find_col(df, self.NAME_COLS)
        if not name_col:
            return None
        mask = df[name_col].astype(str).str.upper().str.contains(keyword.upper())
        if not mask.any():
            return None
        code_col = self._find_col(df, self.CFTC_CODE_COLS)
        if code_col:
            first_code = str(df[mask][code_col].iloc[0]).strip()
            return self.extract_positioning(df, first_code)
        return None

    def calculate_cot_score(self, positioning: dict) -> float:
        if not positioning:
            return 0.0
        pct   = positioning["percentile_52w"]
        delta = positioning["delta_weekly"]
        trend = positioning["trend_4w"]
        net   = positioning["net_current"]
        if pct >= 85:   pct_score = -6.0
        elif pct >= 70: pct_score = -3.0
        elif pct >= 55: pct_score = +2.0
        elif pct >= 45: pct_score =  0.0
        elif pct >= 30: pct_score = -2.0
        elif pct >= 15: pct_score = +3.0
        else:           pct_score = +6.0
        delta_pct   = delta / abs(net) if abs(net) > 1000 else delta / 10000 if abs(net) > 0 else 0
        delta_score = max(-3, min(3, delta_pct * 30))
        trend_pct   = trend / abs(net) if abs(net) > 1000 else trend / 10000 if abs(net) > 0 else 0
        trend_score = max(-3, min(3, trend_pct * 15))
        return round(max(-12, min(12, pct_score + delta_score + trend_score)), 1)


class MacroDataCollector:
    """Collects all required macro data."""

    def __init__(self):
        self._fred = None

    def _get_fred(self):
        if self._fred is None:
            try:
                from fredapi import Fred
                self._fred = Fred(api_key=os.getenv("FRED_API_KEY"))
            except Exception as e:
                logger.warning(f"FRED not available: {e}")
        return self._fred

    def fetch_fred_series(self, series_id: str, periods: int = 4) -> Optional[dict]:
        """Download a FRED series — last N values."""
        fred = self._get_fred()
        if not fred:
            return None
        try:
            series = fred.get_series(series_id).dropna()
            if len(series) == 0:
                return None
            recent   = series.tail(periods)
            latest   = float(recent.iloc[-1])
            prev     = float(recent.iloc[-2]) if len(recent) > 1 else latest
            prev2    = float(recent.iloc[-3]) if len(recent) > 2 else prev
            trend_3m = "rising" if latest > prev2 else "falling" if latest < prev2 else "stable"
            return {
                "id":       series_id,
                "latest":   round(latest, 4),
                "prev":     round(prev, 4),
                "delta":    round(latest - prev, 4),
                "trend_3m": trend_3m,
                "date":     str(recent.index[-1].date()),
            }
        except Exception as e:
            logger.warning(f"FRED {series_id}: {e}")
            return None

    def fetch_all_fred(self) -> dict:
        """Download all core + extended FRED series."""
        data       = {}
        all_series = {**FRED_CORE_SERIES, **FRED_EXTENDED_SERIES}
        for sid, name in all_series.items():
            result = self.fetch_fred_series(sid)
            if result:
                result["name"] = name
                data[sid] = result
        logger.info(f"FRED: {len(data)}/{len(all_series)} series downloaded")
        return data

    def fetch_commodities(self) -> dict:
        """Download commodity prices from yfinance."""
        from utils.yfinance_lock import YF_DOWNLOAD_LOCK
        data = {}
        for ticker, name in COMMODITY_TICKERS.items():
            try:
                with YF_DOWNLOAD_LOCK:
                    tk   = yf.Ticker(ticker)
                    hist = tk.history(period="5d")
                if hist.empty:
                    continue
                latest = float(hist["Close"].iloc[-1])
                prev   = float(hist["Close"].iloc[-2]) if len(hist) > 1 else latest
                data[ticker] = {
                    "name":      name,
                    "latest":    round(latest, 4),
                    "prev":      round(prev, 4),
                    "delta":     round(latest - prev, 4),
                    "delta_pct": round((latest - prev) / prev * 100, 2),
                    "trend":     "rising" if latest > prev else "falling",
                }
            except Exception as e:
                logger.warning(f"yfinance {ticker}: {e}")
        logger.info(f"Commodities: {len(data)} tickers downloaded")
        return data

    def fetch_forexfactory_calendar(self) -> list:
        """Download economic calendar from ForexFactory with cascade fallback."""
        events = []

        try:
            url     = "https://www.forexfactory.com/calendar"
            headers = {
                "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                   "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
                "Accept":          "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer":         "https://www.google.com/",
            }
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                rows = soup.select("tr.calendar__row") or soup.select("tr[data-event-id]")
                for row in rows[:40]:
                    currency_el = row.select_one(".calendar__currency")
                    currency    = currency_el.text.strip() if currency_el else ""
                    title_el    = row.select_one(".calendar__event-title") or row.select_one(".event")
                    title       = title_el.text.strip() if title_el else ""
                    impact      = "high" if any(w in title.upper() for w in
                                  ["RATE", "DECISION", "CPI", "NFP", "GDP", "FOMC",
                                   "EMPLOYMENT", "INFLATION", "PMI", "MINUTES"]) else "medium"
                    time_el     = row.select_one(".calendar__time")
                    time_str    = time_el.text.strip() if time_el else ""
                    if title and currency:
                        events.append({
                            "title":    f"[{currency}] {title}",
                            "time":     time_str,
                            "impact":   impact,
                            "currency": currency,
                            "source":   "ForexFactory",
                        })
                if events:
                    logger.info(f"ForexFactory scraping: {len(events)} events found")
        except Exception as e:
            logger.warning(f"ForexFactory scraping: {e}")

        if not events:
            try:
                feed = feedparser.parse("https://www.myfxbook.com/rss/forex-economic-calendar-events.xml")
                for entry in feed.entries[:20]:
                    title  = entry.get("title", "")
                    impact = "high" if any(w in title.upper() for w in
                             ["RATE", "CPI", "GDP", "NFP", "PMI", "FOMC"]) else "medium"
                    events.append({"title": title, "time": entry.get("published", ""),
                                   "impact": impact, "source": "myfxbook"})
                if events:
                    logger.info(f"myfxbook: {len(events)} events found")
            except Exception as e:
                logger.warning(f"myfxbook RSS: {e}")

        if not events:
            logger.warning("Economic calendar: all feeds unavailable")
        return events[:30]

    def fetch_reuters_news(self) -> list:
        """Download breaking macro news from Reuters + Yahoo Finance RSS."""
        news     = []
        rss_urls = [
            "https://feeds.reuters.com/reuters/businessNews",
            "https://feeds.reuters.com/reuters/economicNews",
            "https://finance.yahoo.com/rss/topstories",
            "https://feeds.marketwatch.com/marketwatch/topstories/",
        ]
        for url in rss_urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]:
                    news.append({
                        "title":   entry.get("title", ""),
                        "summary": entry.get("summary", "")[:200],
                        "date":    entry.get("published", ""),
                        "source":  "Reuters",
                    })
            except Exception as e:
                logger.warning(f"Reuters RSS {url}: {e}")
        logger.info(f"Reuters: {len(news)} news items downloaded")
        return news[:10]

    def fetch_central_bank_news(self) -> dict:
        """Download RSS from all central banks."""
        all_news = {}
        for currency, config in CURRENCY_CONFIG.items():
            bank = config["bank"]
            url  = config.get("rss")
            if not url:
                continue
            try:
                feed    = feedparser.parse(url)
                entries = []
                for entry in feed.entries[:3]:
                    entries.append({
                        "title":   entry.get("title", ""),
                        "summary": entry.get("summary", "")[:150],
                        "date":    entry.get("published", ""),
                    })
                if entries:
                    all_news[bank] = entries
            except Exception as e:
                logger.warning(f"RSS {bank}: {e}")
        logger.info(f"Central banks: {len(all_news)} feeds downloaded")
        return all_news


# ══════════════════════════════════════════════════════════════
# DETERMINISTIC SCORER — 60% of total
# ══════════════════════════════════════════════════════════════

class MacroDeterministicScorer:
    """
    Computes deterministic score per asset based on fixed calibrated rules.
    Output range: -12 / +12.
    """

    def score_all_assets(self, fred: dict, commodities: dict, calendar: list) -> dict:
        """Compute deterministic score for all assets."""
        scores = {}

        dxy       = self._get(fred, "DTWEXBGS")
        vix       = self._get(fred, "VIXCLS")
        t10y2y    = self._get(fred, "T10Y2Y")
        fedfunds  = self._get(fred, "FEDFUNDS")
        cpi_usa   = self._get(fred, "CPIAUCSL")
        unemp_usa = self._get(fred, "UNRATE")
        dgs10     = self._get(fred, "DGS10")
        m2        = self._get(fred, "M2SL")
        ecb_rate  = self._get(fred, "ECBDFR")
        boe_rate  = self._get(fred, "BOERUKM")
        boj_rate  = self._get(fred, "IRSTCI01JPM156N")
        rba_rate  = self._get(fred, "IRSTCI01AUM156N")
        boc_rate  = self._get(fred, "IRSTCI01CAM156N")
        snb_rate  = self._get(fred, "IRSTCI01CHM156N")
        rbnz_rate = self._get(fred, "IRSTCI01NZM156N")

        oil    = self._get_commodity(commodities, "CL=F")
        copper = self._get_commodity(commodities, "HG=F")

        cal_penalty = self._calendar_penalty(calendar)

        usd_bias = self._usd_bias(dxy, fedfunds, cpi_usa, unemp_usa, dgs10)
        eur_bias = self._eur_bias(ecb_rate, fedfunds)
        gbp_bias = self._gbp_bias(boe_rate, fedfunds)
        jpy_bias = self._jpy_bias(boj_rate, fedfunds, vix)
        aud_bias = self._aud_bias(rba_rate, copper, vix)
        nzd_bias = self._nzd_bias(rbnz_rate, vix)
        cad_bias = self._cad_bias(boc_rate, oil)
        chf_bias = self._chf_bias(snb_rate, vix)
        risk_on  = self._risk_appetite(vix, t10y2y, dgs10)

        for asset, config in ASSET_CONFIG.items():
            s, factors = 0.0, []

            if asset == "EURUSD":    s, factors = self._score_pair(eur_bias, usd_bias, "EUR", "USD")
            elif asset == "GBPUSD":  s, factors = self._score_pair(gbp_bias, usd_bias, "GBP", "USD")
            elif asset == "USDJPY":  s, factors = self._score_pair(usd_bias, jpy_bias, "USD", "JPY")
            elif asset == "GBPJPY":  s, factors = self._score_pair(gbp_bias, jpy_bias, "GBP", "JPY")
            elif asset in ("XAUUSD", "MGC"):   s, factors = self._score_gold(dxy, t10y2y, vix, risk_on)
            elif asset in ("BTCUSD", "ETHUSD"): s, factors = self._score_crypto(risk_on, fedfunds, m2)
            elif asset == "EURAUD":  s, factors = self._score_pair(eur_bias, aud_bias, "EUR", "AUD")
            elif asset == "USDCAD":  s, factors = self._score_pair(usd_bias, cad_bias, "USD", "CAD")
            elif asset == "EURGBP":  s, factors = self._score_pair(eur_bias, gbp_bias, "EUR", "GBP")
            elif asset == "NZDJPY":  s, factors = self._score_pair(nzd_bias, jpy_bias, "NZD", "JPY")
            elif asset == "EURCHF":  s, factors = self._score_pair(eur_bias, chf_bias, "EUR", "CHF")
            elif asset in ("MES", "NAS100", "GER40", "US2000", "DJ30"):
                s, factors = self._score_equities(fedfunds, t10y2y, vix, risk_on, unemp_usa)
            elif asset == "XAGUSD":  s, factors = self._score_silver(dxy, t10y2y, copper, risk_on)
            elif asset == "MCL":     s, factors = self._score_oil(oil, usd_bias, risk_on)
            elif asset == "6E":      s, factors = self._score_pair(eur_bias, usd_bias, "EUR", "USD")
            elif asset == "NZDUSD":  s, factors = self._score_pair(nzd_bias, usd_bias, "NZD", "USD")
            elif asset == "AUDJPY":  s, factors = self._score_pair(aud_bias, jpy_bias, "AUD", "JPY")
            elif asset == "CHFJPY":  s, factors = self._score_pair(chf_bias, jpy_bias, "CHF", "JPY")

            asset_cal = cal_penalty.get(asset, cal_penalty.get("ALL", 0))
            if asset_cal != 0:
                factors.append(f"Upcoming event: {asset_cal:+.0f}")
            s += asset_cal
            s = max(-12, min(12, s))

            bull_factors = [f for f in factors if any(k in f.lower() for k in
                            ["strong", "rising", "bullish", "positive", "risk-on", "cut"])]
            bear_factors = [f for f in factors if any(k in f.lower() for k in
                            ["weak", "falling", "bearish", "pressure", "risk-off", "hawkish"])]

            if s > 0 and not bull_factors:
                bull_factors = [config["driver"][:80]]
            if s < 0 and not bear_factors:
                bear_factors = [f for f in factors if f]

            bull_case = "; ".join(bull_factors[:2]) if bull_factors else config["driver"][:80]
            bear_case = "; ".join(bear_factors[:2]) if bear_factors else f"Negative bias pressure ({s:+.1f})"

            scores[asset] = {
                "deterministic_score": round(s, 1),
                "factors":   factors,
                "bull_case": bull_case,
                "bear_case": bear_case,
                "tier":      config["tier"],
                "type":      config["type"],
                "driver":    config["driver"],
            }

        return scores

    def _usd_bias(self, dxy, fedfunds, cpi, unemp, dgs10) -> float:
        score = 0.0
        if dxy:
            if dxy["delta"] > 0.3:    score += 3.0
            elif dxy["delta"] > 0:    score += 1.5
            elif dxy["delta"] < -0.3: score -= 3.0
            elif dxy["delta"] < 0:    score -= 1.5
        if fedfunds:
            if fedfunds["trend_3m"] == "rising":  score += 2.0
            elif fedfunds["trend_3m"] == "falling": score -= 2.0
        if cpi:
            if cpi["trend_3m"] == "rising":  score += 1.0
            elif cpi["trend_3m"] == "falling": score -= 1.0
        if unemp:
            if unemp["trend_3m"] == "falling": score += 1.0
            elif unemp["trend_3m"] == "rising":  score -= 1.0
        return max(-6, min(6, score))

    def _eur_bias(self, ecb_rate, fedfunds) -> float:
        score = 0.0
        if ecb_rate and fedfunds:
            diff = ecb_rate["latest"] - fedfunds["latest"]
            if diff > 0.5:    score += 2.0
            elif diff > 0:    score += 0.5
            elif diff < -0.5: score -= 2.0
            elif diff < 0:    score -= 0.5
            if ecb_rate["trend_3m"] == "rising":  score += 1.5
            elif ecb_rate["trend_3m"] == "falling": score -= 1.5
        return max(-4, min(4, score))

    def _gbp_bias(self, boe_rate, fedfunds) -> float:
        score = 0.0
        if boe_rate and fedfunds:
            diff = boe_rate["latest"] - fedfunds["latest"]
            if diff > 0.5:    score += 2.0
            elif diff > 0:    score += 0.5
            elif diff < -0.5: score -= 2.0
            elif diff < 0:    score -= 0.5
            if boe_rate["trend_3m"] == "rising":  score += 1.5
            elif boe_rate["trend_3m"] == "falling": score -= 1.5
        return max(-4, min(4, score))

    def _jpy_bias(self, boj_rate, fedfunds, vix) -> float:
        score = 0.0
        if boj_rate and fedfunds:
            diff   = boj_rate["latest"] - fedfunds["latest"]
            score += max(-3, min(3, diff))
            if boj_rate["trend_3m"] == "rising":  score += 2.0
            elif boj_rate["trend_3m"] == "falling": score -= 1.0
        if vix:
            if vix["latest"] > 25:   score += 2.0
            elif vix["latest"] > 20: score += 0.5
        return max(-5, min(5, score))

    def _aud_bias(self, rba_rate, copper, vix) -> float:
        score = 0.0
        if rba_rate:
            if rba_rate["trend_3m"] == "rising":  score += 1.5
            elif rba_rate["trend_3m"] == "falling": score -= 1.5
        if copper:
            if copper["delta_pct"] > 1:    score += 2.0
            elif copper["delta_pct"] > 0:  score += 0.5
            elif copper["delta_pct"] < -1: score -= 2.0
            elif copper["delta_pct"] < 0:  score -= 0.5
        if vix and vix["latest"] > 25:
            score -= 1.5
        return max(-4, min(4, score))

    def _nzd_bias(self, rbnz_rate, vix) -> float:
        score = 0.0
        if rbnz_rate:
            if rbnz_rate["trend_3m"] == "rising":  score += 1.5
            elif rbnz_rate["trend_3m"] == "falling": score -= 1.5
        if vix and vix["latest"] > 25:
            score -= 1.5
        return max(-3, min(3, score))

    def _cad_bias(self, boc_rate, oil) -> float:
        score = 0.0
        if boc_rate:
            if boc_rate["trend_3m"] == "rising":  score += 1.5
            elif boc_rate["trend_3m"] == "falling": score -= 1.5
        if oil:
            if oil["delta_pct"] > 2:    score += 3.0
            elif oil["delta_pct"] > 0:  score += 1.0
            elif oil["delta_pct"] < -2: score -= 3.0
            elif oil["delta_pct"] < 0:  score -= 1.0
        return max(-4, min(4, score))

    def _chf_bias(self, snb_rate, vix) -> float:
        score = 0.0
        if snb_rate and snb_rate["trend_3m"] == "rising":
            score += 1.0
        if vix:
            if vix["latest"] > 30:   score += 3.0
            elif vix["latest"] > 25: score += 1.5
        return max(-2, min(4, score))

    def _risk_appetite(self, vix, t10y2y, dgs10) -> float:
        score = 0.0
        if vix:
            if vix["latest"] < 15:    score += 3.0
            elif vix["latest"] < 20:  score += 1.0
            elif vix["latest"] > 30:  score -= 3.0
            elif vix["latest"] > 25:  score -= 1.5
        if t10y2y:
            if t10y2y["latest"] > 0:      score += 1.0
            elif t10y2y["latest"] < -0.5: score -= 1.5
        return max(-5, min(5, score))

    def _score_pair(self, base_bias, quote_bias, base, quote):
        s = base_bias - quote_bias
        factors = []
        if base_bias > 0:  factors.append(f"{base} strong ({base_bias:+.1f})")
        if base_bias < 0:  factors.append(f"{base} weak ({base_bias:+.1f})")
        if quote_bias > 0: factors.append(f"{quote} strong ({quote_bias:+.1f})")
        if quote_bias < 0: factors.append(f"{quote} weak ({quote_bias:+.1f})")
        return s, factors

    def _score_gold(self, dxy, t10y2y, vix, risk_on):
        s = 0.0; factors = []
        if dxy:
            if dxy["delta"] < -0.3:  s += 4.0; factors.append("DXY falling → gold bullish")
            elif dxy["delta"] < 0:   s += 1.5
            elif dxy["delta"] > 0.3: s -= 4.0; factors.append("Strong DXY → gold bearish")
            elif dxy["delta"] > 0:   s -= 1.5
        if t10y2y:
            if t10y2y["latest"] < -0.3: s += 2.0; factors.append("Inverted curve → gold strong")
            elif t10y2y["latest"] < 0:  s += 0.5
        if vix:
            if vix["latest"] > 25: s += 2.0; factors.append("High VIX → safe-haven gold")
            elif vix["latest"] < 15: s -= 1.0
        return s, factors

    def _score_crypto(self, risk_on, fedfunds, m2):
        s = risk_on * 1.5; factors = []
        if risk_on > 0: factors.append(f"Risk-on ({risk_on:+.1f}) → crypto bullish")
        if risk_on < 0: factors.append(f"Risk-off ({risk_on:+.1f}) → crypto bearish")
        if fedfunds and fedfunds["trend_3m"] == "falling":
            s += 2.0; factors.append("FED cutting → liquidity → crypto strong")
        elif fedfunds and fedfunds["trend_3m"] == "rising":
            s -= 2.0; factors.append("FED hawkish → crypto pressure")
        if m2 and m2["trend_3m"] == "rising":
            s += 1.5; factors.append("M2 expanding → crypto bullish")
        return max(-10, min(10, s)), factors

    def _score_equities(self, fedfunds, t10y2y, vix, risk_on, unemp):
        s = risk_on * 1.2; factors = []
        if fedfunds and fedfunds["trend_3m"] == "falling":
            s += 3.0; factors.append("FED cutting → equity markets bullish")
        elif fedfunds and fedfunds["trend_3m"] == "rising":
            s -= 2.0; factors.append("FED hawkish → equity pressure")
        if t10y2y and t10y2y["latest"] < -0.5:
            s -= 1.5; factors.append("Inverted curve → recession risk")
        if unemp and unemp["trend_3m"] == "falling":
            s += 1.0; factors.append("Strong labor market → solid earnings")
        if vix and vix["latest"] < 15:
            s += 1.5; factors.append("Low VIX → positive sentiment")
        return max(-10, min(10, s)), factors

    def _score_silver(self, dxy, t10y2y, copper, risk_on):
        s = 0.0; factors = []
        if dxy and dxy["delta"] < 0:
            s += 2.0; factors.append("DXY falling → silver bullish")
        if copper and copper["delta_pct"] > 1:
            s += 2.0; factors.append("Copper strong → industrial demand → silver")
        if risk_on > 0:
            s += 1.0
        return max(-8, min(8, s)), factors

    def _score_oil(self, oil, usd_bias, risk_on):
        s = 0.0; factors = []
        if oil:
            if oil["delta_pct"] > 2:    s += 5.0; factors.append("Oil in strong uptrend")
            elif oil["delta_pct"] > 0:  s += 2.0
            elif oil["delta_pct"] < -2: s -= 5.0; factors.append("Oil in strong downtrend")
            elif oil["delta_pct"] < 0:  s -= 2.0
        if usd_bias > 0: s -= 1.0; factors.append("Strong USD → oil pressure")
        if risk_on < 0:  s -= 1.5; factors.append("Risk-off → oil demand falling")
        return max(-10, min(10, s)), factors

    def _calendar_penalty(self, events: list) -> dict:
        """Compute penalties for imminent high-impact events."""
        penalties = {}
        for event in events:
            if event.get("impact") != "high":
                continue
            title = event.get("title", "").upper()
            if any(w in title for w in ["FOMC", "FED", "RATE DECISION", "NFP", "CPI USA", "GDP USA"]):
                penalties["ALL"] = min(penalties.get("ALL", 0), -5)
            if any(w in title for w in ["ECB", "EUROZONE", "EUR"]):
                for a in ["EURUSD", "EURGBP", "EURAUD", "EURCHF", "6E"]:
                    penalties[a] = min(penalties.get(a, 0), -4)
            if any(w in title for w in ["BOE", "UK", "GBP"]):
                for a in ["GBPUSD", "EURGBP", "GBPJPY"]:
                    penalties[a] = min(penalties.get(a, 0), -4)
            if any(w in title for w in ["BOJ", "JAPAN", "JPY"]):
                for a in ["USDJPY", "GBPJPY", "NZDJPY"]:
                    penalties[a] = min(penalties.get(a, 0), -4)
        return penalties

    def _get(self, fred: dict, sid: str) -> Optional[dict]:
        return fred.get(sid)

    def _get_commodity(self, commodities: dict, ticker: str) -> Optional[dict]:
        return commodities.get(ticker)


# ══════════════════════════════════════════════════════════════
# MACRO AGENT
# ══════════════════════════════════════════════════════════════

SCHEDULED_HOURS_CET = [7]
CACHE_MAX_SECONDS   = 24 * 3600


class MacroAgent(BaseAgent):
    """
    MacroAgent v2 — complete macro analysis for all assets.

    Score per asset: deterministic (-12/+12) + qualitative (-8/+8) + COT (-12×0.2)
    Final range: -20 / +20

    Schedule:
    - 07:00 CET → run_full() complete (1×/day — 24h disk cache)
    - On-demand → uses disk cache (max 24h) when called outside window
    - force_refresh=True → ignores cache (use after NFP/CPI/Fed shock)

    PARTIAL: Full architecture is real. Claude prompt template and
    internal qualitative scoring rules are intentionally omitted.
    The deterministic scoring and COT computation are fully real.
    """

    AGENT_NAME  = "MacroAgent"
    MODEL       = "claude-sonnet-4-6"
    SCORE_RANGE = (-20, 20)

    _qual_cache:    Optional[dict]     = None
    _qual_cache_ts: Optional[datetime] = None
    QUAL_CACHE_TTL = 24 * 3600

    def __init__(self):
        super().__init__()
        # Production: self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.collector = MacroDataCollector()
        self.scorer    = MacroDeterministicScorer()
        self._cached_data = None
        self._cache_time  = None
        self._cot_cache:    Optional[dict]     = None
        self._cot_cache_ts: Optional[datetime] = None
        self.cftc_fetcher  = CFTCDataFetcher()
        self.cot_analyzer  = COTAnalyzer()
        MACRO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        MACRO_QUAL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._load_cache_from_disk()

    # ── Cache management ──────────────────────────────────────────────────────

    def _load_cache_from_disk(self):
        """Load macro and qualitative caches from disk on startup."""
        now = datetime.now(timezone.utc)
        if MACRO_CACHE_PATH.exists():
            try:
                with open(MACRO_CACHE_PATH, encoding="utf-8") as f:
                    saved = json.load(f)
                ts  = datetime.fromisoformat(saved.get("_cache_time", ""))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age = (now - ts).total_seconds()
                if age < MACRO_CACHE_SECONDS:
                    self._cached_data = saved.get("data")
                    self._cache_time  = ts
                    logger.info(f"[MacroAgent] Data cache restored from disk (age: {age/3600:.1f}h)")
            except Exception as e:
                logger.warning(f"[MacroAgent] Cannot load data cache from disk: {e}")

        if MACRO_QUAL_CACHE_PATH.exists():
            try:
                with open(MACRO_QUAL_CACHE_PATH, encoding="utf-8") as f:
                    saved = json.load(f)
                ts  = datetime.fromisoformat(saved.get("_cache_time", ""))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age = (now - ts).total_seconds()
                if age < MacroAgent.QUAL_CACHE_TTL:
                    MacroAgent._qual_cache    = saved.get("data")
                    MacroAgent._qual_cache_ts = ts
                    logger.info(f"[MacroAgent] Qualitative cache restored from disk (age: {age/3600:.1f}h)")
            except Exception as e:
                logger.warning(f"[MacroAgent] Cannot load qualitative cache from disk: {e}")

        if COT_PROCESSED_CACHE_PATH.exists():
            try:
                with open(COT_PROCESSED_CACHE_PATH, encoding="utf-8") as f:
                    saved = json.load(f)
                ts       = datetime.fromisoformat(saved.get("_cache_time", ""))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_days = (now - ts).total_seconds() / 86400
                if age_days < COT_CACHE_DAYS:
                    self._cot_cache    = saved.get("data")
                    self._cot_cache_ts = ts
                    logger.info(f"[MacroAgent] COT cache restored from disk (age: {age_days:.1f}d)")
            except Exception as e:
                logger.warning(f"[MacroAgent] Cannot load COT cache from disk: {e}")

    def _save_data_cache_to_disk(self):
        if not self._cached_data or not self._cache_time:
            return
        try:
            with open(MACRO_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump({"_cache_time": self._cache_time.isoformat(),
                           "data": self._cached_data}, f, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning(f"[MacroAgent] Cannot save data cache to disk: {e}")

    def _save_cot_cache_to_disk(self):
        if not self._cot_cache or not self._cot_cache_ts:
            return
        try:
            with open(COT_PROCESSED_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump({"_cache_time": self._cot_cache_ts.isoformat(),
                           "data": self._cot_cache}, f, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning(f"[MacroAgent] Cannot save COT cache to disk: {e}")

    def _save_qual_cache_to_disk(self):
        if not MacroAgent._qual_cache or not MacroAgent._qual_cache_ts:
            return
        try:
            with open(MACRO_QUAL_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump({"_cache_time": MacroAgent._qual_cache_ts.isoformat(),
                           "data": MacroAgent._qual_cache}, f, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning(f"[MacroAgent] Cannot save qualitative cache to disk: {e}")

    def should_run_now(self) -> bool:
        """Returns True if within the daily scheduled window: 07:00 CET ±15 min."""
        try:
            import pytz
            cet     = pytz.timezone("Europe/Rome")
            now_cet = datetime.now(cet)
            h, m    = now_cet.hour, now_cet.minute
            return abs((h * 60 + m) - (7 * 60 + 0)) <= 15
        except Exception:
            return False

    def _is_cache_valid(self) -> bool:
        if not self._cache_time:
            return False
        return (datetime.now(timezone.utc) - self._cache_time).total_seconds() < CACHE_MAX_SECONDS

    def _is_cot_cache_valid(self) -> bool:
        if self._cot_cache is None or self._cot_cache_ts is None:
            return False
        return (datetime.now(timezone.utc) - self._cot_cache_ts).total_seconds() / 86400 < COT_CACHE_DAYS

    def _get_cot_data(self) -> dict:
        """Return COT data for all currencies with 7-day cache."""
        if self._is_cot_cache_valid():
            return self._cot_cache

        logger.info("[MacroAgent] COT: updating CFTC data...")
        year   = datetime.now().year
        fin_df = self.cftc_fetcher.get_financial_data(year)
        dis_df = self.cftc_fetcher.get_disaggregated_data(year)

        result = {}
        for code in ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF", "MES", "NAS"]:
            cftc_code = CFTC_CODES.get(code)
            if not cftc_code:
                continue
            pos = self.cot_analyzer.extract_positioning(fin_df, cftc_code)
            if pos:
                result[code] = pos

        for code in ["XAU", "XAG", "OIL", "BTC"]:
            cftc_code = CFTC_CODES.get(code)
            if not cftc_code:
                continue
            if code == "BTC":
                pos = (self.cot_analyzer.extract_positioning(dis_df, cftc_code) or
                       self.cot_analyzer.extract_positioning(fin_df, cftc_code) or
                       self.cot_analyzer.extract_by_name(fin_df, "BITCOIN") or
                       self.cot_analyzer.extract_by_name(dis_df, "BITCOIN"))
            else:
                pos = self.cot_analyzer.extract_positioning(dis_df, cftc_code)
            if pos:
                result[code] = pos

        self._cot_cache    = result
        self._cot_cache_ts = datetime.now(timezone.utc)
        self._save_cot_cache_to_disk()
        logger.info(f"[MacroAgent] COT loaded for {len(result)} instruments: {list(result.keys())}")
        return result

    def _calc_cot_scores(self, cot_data: dict) -> dict:
        """Convert COT positioning per currency → score per asset (±12)."""
        currency_scores = {code: self.cot_analyzer.calculate_cot_score(pos)
                           for code, pos in cot_data.items()}

        all_assets   = list(ASSET_CONFIG.keys())
        asset_scores = {a: 0.0 for a in all_assets}

        for code, cur_score in currency_scores.items():
            asset_map   = CFTC_ASSET_MAP.get(code, {})
            is_inverted = asset_map.get("inverted", False)
            weight      = asset_map.get("weight", 1.0)
            for asset in asset_map.get("bullish_assets", []):
                if asset in asset_scores:
                    asset_scores[asset] += cur_score * weight * (-1 if is_inverted else 1)
            for asset in asset_map.get("bearish_assets", []):
                if asset in asset_scores:
                    asset_scores[asset] -= cur_score * weight * (-1 if is_inverted else 1)

        if "MGC" in asset_scores:
            asset_scores["XAUUSD"] = asset_scores["MGC"]

        return {a: round(max(-12, min(12, s)), 1) for a, s in asset_scores.items()}

    # ── Data collection ───────────────────────────────────────────────────────

    async def collect_data(self, context: dict = None, force_refresh: bool = False) -> dict:
        """Collect all macro data — cached for 24h on disk and in memory."""
        if not force_refresh and self._is_cache_valid() and self._cached_data:
            logger.info("[MacroAgent] Using cached data (TTL 24h)")
            return self._cached_data

        logger.info("[MacroAgent] Full macro data collection starting...")
        data = {
            "fred":        self.collector.fetch_all_fred(),
            "commodities": self.collector.fetch_commodities(),
            "calendar":    self.collector.fetch_forexfactory_calendar(),
            "reuters":     self.collector.fetch_reuters_news(),
            "banks":       self.collector.fetch_central_bank_news(),
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "web_context": await _fetch_web_ctx() if _WEB_CTX_AVAILABLE else "",
        }

        self._cached_data = data
        self._cache_time  = datetime.now(timezone.utc)
        self._save_data_cache_to_disk()
        return data

    # ── Qualitative analysis (Claude) ─────────────────────────────────────────

    async def analyze(self, data: dict, context: dict = None, force_refresh: bool = False) -> dict:
        """
        Claude qualitative analysis — narrative adjustment.

        NOTE: Implementation intentionally omitted.
        Production: calls Claude with central bank communications + news feeds
        to produce per-asset narrative adjustments (-8/+8) per asset.
        See paper Section 3.3 for design rationale.

        This stub returns neutral qualitative scores (0 for all assets),
        so the final score is driven entirely by the deterministic component.
        """
        now = datetime.now(timezone.utc)
        if (not force_refresh
                and MacroAgent._qual_cache is not None
                and MacroAgent._qual_cache_ts is not None
                and (now - MacroAgent._qual_cache_ts).total_seconds() < MacroAgent.QUAL_CACHE_TTL):
            logger.info("[MacroAgent] Qualitative: using 24h cache — no Claude call")
            return MacroAgent._qual_cache

        # Stub: return neutral qualitative scores
        # Production: calls Claude with MACRO_QUALITATIVE_PROMPT + data
        logger.info("[MacroAgent] Qualitative [stub]: returning neutral scores (0 for all assets)")
        stub_result = {
            "qualitative_scores": {
                asset: {"score": 0, "reasoning": "[stub — implementation omitted]"}
                for asset in ASSET_CONFIG
            },
            "macro_summary": "[stub] Qualitative analysis not available in this version.",
            "top_risk":      "[stub]",
            "market_regime": "neutral",
        }

        MacroAgent._qual_cache    = stub_result
        MacroAgent._qual_cache_ts = now
        self._save_qual_cache_to_disk()
        return stub_result

    # ── Full run — all assets ──────────────────────────────────────────────────

    async def run_full(self, force_refresh: bool = False) -> dict:
        """
        Complete macro analysis for all assets.
        Scheduled 1×/day at 07:00 CET — 24h disk cache.
        """
        logger.info("[MacroAgent] Starting complete macro analysis")

        data       = await self.collect_data(force_refresh=force_refresh)
        det_scores = self.scorer.score_all_assets(data["fred"], data["commodities"], data["calendar"])

        qual_result = await self.analyze(data, force_refresh=force_refresh)
        qual_scores = qual_result.get("qualitative_scores", {})

        cot_raw    = self._get_cot_data()
        cot_scores = self._calc_cot_scores(cot_raw)
        logger.info(f"[MacroAgent] COT scores computed for {len(cot_scores)} assets")

        # Score inheritance: MGC ← XAUUSD, 6E ← EURUSD
        for _futures, _cfd in (("MGC", "XAUUSD"), ("6E", "EURUSD")):
            if _cfd in qual_scores and qual_scores[_cfd]:
                _q = qual_scores[_cfd].copy()
                _q["reasoning"] = f"[≡{_cfd}] " + _q.get("reasoning", "")
                qual_scores[_futures] = _q
            if _cfd in cot_scores:
                cot_scores[_futures] = cot_scores[_cfd]

        # Combine: 50% deterministic + 30% qualitative + 20% COT
        final_scores = {}
        for asset in ASSET_CONFIG:
            det   = det_scores.get(asset, {}).get("deterministic_score", 0)
            qual  = qual_scores.get(asset, {}).get("score", 0)
            cot   = cot_scores.get(asset, 0)
            total = round(det + qual + (cot * 0.2), 1)
            total = max(-20, min(20, total))

            q_reasoning     = qual_scores.get(asset, {}).get("reasoning", "")
            det_info        = det_scores.get(asset, {})
            cot_positioning = cot_raw.get(asset[:3], cot_raw.get(asset, {}))
            cot_index       = float(cot_positioning.get("percentile_52w", 50.0)) if cot_positioning else 50.0
            cot_extreme     = cot_index > 80 or cot_index < 20
            cot_bias        = "bullish" if cot_index > 60 else "bearish" if cot_index < 40 else "neutral"

            macro_contribution = round(max(-0.25, min(0.25, total / 20.0 * 0.25)), 3)
            cot_contribution   = (+0.20 if cot_bias == "bullish"
                                  else -0.20 if cot_bias == "bearish" else 0.0) if cot_extreme else 0.0

            final_scores[asset] = {
                "score":              total,
                "deterministic":      det,
                "qualitative":        qual,
                "cot":                cot,
                "cot_index":          cot_index,
                "cot_extreme":        cot_extreme,
                "cot_bias":           cot_bias,
                "confidence":         "high" if abs(det) > 6 else "medium" if abs(det) > 3 else "low",
                "bull_case":          det_info.get("bull_case", ""),
                "bear_case":          det_info.get("bear_case", ""),
                "qualitative_note":   q_reasoning,
                "driver":             det_info.get("driver", ""),
                "tier":               det_info.get("tier", 2),
                "type":               det_info.get("type", "forex"),
                "factors":            det_info.get("factors", []),
                "cot_data":           cot_positioning,
                "macro_contribution": macro_contribution,
                "cot_contribution":   cot_contribution,
            }

        ranked   = sorted(final_scores.items(), key=lambda x: abs(x[1]["score"]), reverse=True)
        top_buy  = [(a, v) for a, v in ranked if v["score"] > 5][:5]
        top_sell = [(a, v) for a, v in ranked if v["score"] < -5][:5]

        result = {
            "scores":          final_scores,
            "ranking":         [(a, v["score"]) for a, v in ranked],
            "top_buy":         top_buy,
            "top_sell":        top_sell,
            "macro_summary":   qual_result.get("macro_summary", ""),
            "top_risk":        qual_result.get("top_risk", ""),
            "market_regime":   qual_result.get("market_regime", "neutral"),
            "timestamp":       data["timestamp"],
            "calendar_events": data["calendar"],
            "fred_snapshot": {
                sid: data["fred"][sid]["latest"]
                for sid in ["DTWEXBGS", "VIXCLS", "FEDFUNDS", "T10Y2Y"]
                if sid in data["fred"]
            },
        }

        logger.info(
            f"[MacroAgent] Completed — regime: {result['market_regime']} "
            f"| top BUY: {[a for a, _ in top_buy[:3]]} "
            f"| top SELL: {[a for a, _ in top_sell[:3]]}"
        )
        return result

    # ── Single asset run — for orchestrator ──────────────────────────────────

    async def run(self, context: dict) -> AgentResult:
        """Single asset analysis — called by orchestrator."""
        asset     = context.get("asset",     "EURUSD")
        direction = context.get("direction", "BUY")

        full       = await self.run_full()
        asset_data = full["scores"].get(asset, {})
        raw_score  = asset_data.get("score", 0)

        try:
            from utils.score_converter import to_score_0_100
        except ImportError:
            def to_score_0_100(raw, direction, raw_max):
                clamped = max(-raw_max, min(raw_max, raw))
                return round(50.0 + (clamped / raw_max) * 50.0 if direction == "BUY"
                             else 50.0 + (-clamped / raw_max) * 50.0, 1)

        score_0_100 = to_score_0_100(raw_score, direction, raw_max=20.0)

        cot_data   = asset_data.get("cot_data", {})
        cot_index  = float(cot_data.get("percentile_52w", 50.0))
        cot_extreme = cot_index > 80 or cot_index < 20
        cot_bias   = "bullish" if cot_index > 60 else "bearish" if cot_index < 40 else "neutral"

        return AgentResult(
            agent=self.AGENT_NAME,
            score=score_0_100,
            direction=direction,
            summary=(
                f"{asset}: macro {score_0_100:.0f}/100 | "
                f"{full['market_regime']} | raw={raw_score:+.1f}"
            ),
            bull_case=asset_data.get("bull_case", ""),
            bear_case=asset_data.get("bear_case", ""),
            confidence=asset_data.get("confidence", "medium"),
            details=(
                f"Det: {asset_data.get('deterministic', 0):+.1f} | "
                f"Qual: {asset_data.get('qualitative', 0):+.1f} | "
                f"COT: {cot_index:.0f}% | "
                f"Driver: {asset_data.get('driver', '')[:60]}"
            ),
            raw_data={
                **full,
                "cot_extreme": cot_extreme,
                "cot_bias":    cot_bias,
                "cot_index":   cot_index,
                "raw_score":   raw_score,
            },
        )
