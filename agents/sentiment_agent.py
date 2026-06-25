"""
agents/sentiment_agent.py
SentimentAgent v2 — Real-time news sentiment analysis for all 19 assets.

NOTE: Production implementation uses GPT-4o-mini for news analysis and a
deterministic calendar parser (actual vs. forecast comparison). This stub
returns synthetic data to demonstrate the interface.
See paper Section 2.1 for design rationale.

v2 additions vs v1:
  - New RSS sources: ForexLive, Investing.com, Reuters alternative
  - ForexFactory calendar scraping: upcoming events with actual vs. estimate
  - Deterministic actual/estimate logic in Python (not only GPT-4o-mini)
  - Updated 2025-2026 keywords: tariff, trade war, recession, sanctions

Data sources:
  - FXStreet, Yahoo Finance, MarketWatch, Milano Finanza
  - ForexLive RSS
  - Investing.com RSS
  - Reuters RSS (alternative URL)
  - ForexFactory calendar HTML scraping (actual vs. estimate)

Model: GPT-4o-mini — stubbed in this version
"""
import os
import asyncio
import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from typing import Optional
from loguru import logger

try:
    from agents.base_agent import BaseAgent, AgentResult
except ImportError:
    from base_agent import BaseAgent, AgentResult

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

OPERATING_HOURS_START  = 8    # 08:00 CET
OPERATING_HOURS_END    = 23   # 23:00 CET
SCREENING_INTERVAL_MIN = 15
PROCESSED_NEWS_TTL_MIN = 120
THRESHOLD_WARNING      = -5
THRESHOLD_EMERGENCY    = -10

# ── RSS Sources ────────────────────────────────────────────────────────────────
RSS_SOURCES = {
    "FXStreet": [
        "https://www.fxstreet.com/rss/news",
        "https://www.fxstreet.com/rss/analysis",
    ],
    "Yahoo Finance": [
        "https://finance.yahoo.com/rss/topstories",
        "https://finance.yahoo.com/rss/headline?s=EURUSD=X",
    ],
    "MarketWatch": [
        "https://feeds.marketwatch.com/marketwatch/topstories/",
        "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    ],
    "ForexLive": [
        "https://www.forexlive.com/feed/news",
        "https://www.forexlive.com/feed/centralbank",
    ],
    "Investing.com": [
        "https://www.investing.com/rss/news_285.rss",   # Forex news
        "https://www.investing.com/rss/news_301.rss",   # Commodities
    ],
}

# ── Currency keyword mapping ────────────────────────────────────────────────
CURRENCY_KEYWORDS = {
    "USD": [
        "fed", "federal reserve", "fomc", "powell", "dollar", "usd",
        "nfp", "non-farm", "us inflation", "us gdp", "us jobs",
        "treasury", "dollar index", "tariff", "trade war", "us recession",
    ],
    "EUR": [
        "ecb", "european central bank", "lagarde", "euro", "eur",
        "eurozone", "germany", "france", "eu tariff", "european recession",
    ],
    "GBP": [
        "boe", "bank of england", "bailey", "sterling", "gbp",
        "uk", "britain", "pound", "uk inflation", "uk gdp",
    ],
    "JPY": [
        "boj", "bank of japan", "ueda", "yen", "jpy",
        "japan", "yield curve control", "carry trade", "yen intervention",
    ],
    "XAU": [
        "gold", "xau", "precious metal", "safe haven",
        "geopolitical", "war", "conflict", "sanctions", "flight to safety",
    ],
    "BTC": [
        "bitcoin", "btc", "crypto", "cryptocurrency",
        "crypto regulation", "bitcoin etf", "digital asset",
    ],
    "OIL": [
        "oil price", "crude oil", "brent", "wti", "opec",
        "opec cut", "energy crisis", "petroleum",
    ],
}

HIGH_IMPACT_KEYWORDS = [
    "rate decision", "rate hike", "rate cut", "interest rate",
    "fomc", "ecb meeting", "boe meeting", "boj meeting",
    "non-farm payroll", "nfp", "cpi", "inflation data", "gdp",
    "tariff", "trade war", "trade deal", "sanctions",
    "flash crash", "bank failure", "banking crisis",
]


# ─────────────────────────────────────────────────────────────────────────────
# Economic Calendar Fetcher
# ─────────────────────────────────────────────────────────────────────────────

class EconomicCalendarFetcher:
    """
    Scrapes ForexFactory economic calendar with cascade fallbacks.
    Extracts actual/forecast/previous for deterministic score computation.
    The calendar is OPTIONAL: if all sources fail, the agent continues
    using RSS news only (not blocking).
    """

    FF_URL  = "https://www.forexfactory.com/calendar"

    def fetch_calendar(self) -> list:
        """Download economic calendar with cascade fallback."""
        events = self._fetch_forexfactory()
        if not events:
            logger.warning("[Calendar] ForexFactory: no events — trying Yahoo Finance")
            events = self._fetch_yahoo_calendar()
        if not events:
            logger.warning("[Calendar] All calendars unavailable — continuing without")
        else:
            logger.info(f"[Calendar] {len(events)} events loaded")
        return events

    def _fetch_forexfactory(self) -> list:
        """Scrape ForexFactory with Google Referer to avoid blocking."""
        events = []
        try:
            headers = {
                "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                   "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
                "Accept":          "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer":         "https://www.google.com/",
            }
            response = requests.get(self.FF_URL, headers=headers, timeout=15)
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
                    actual_el   = row.select_one(".calendar__actual")
                    forecast_el = row.select_one(".calendar__forecast")
                    prev_el     = row.select_one(".calendar__previous")
                    actual      = actual_el.text.strip()   if actual_el   else ""
                    forecast    = forecast_el.text.strip() if forecast_el else ""
                    previous    = prev_el.text.strip()     if prev_el     else ""
                    if title and currency:
                        events.append({
                            "title":    title,
                            "time":     time_str,
                            "impact":   impact,
                            "currency": currency.upper(),
                            "actual":   actual,
                            "forecast": forecast,
                            "previous": previous,
                            "source":   "ForexFactory",
                        })
                if events:
                    logger.info(f"[Calendar] ForexFactory: {len(events)} events")
        except Exception as e:
            logger.warning(f"[Calendar] ForexFactory error: {e}")
        return events

    def _fetch_yahoo_calendar(self) -> list:
        """Fallback: Yahoo Finance RSS filtered for high-impact macro keywords."""
        events = []
        try:
            for url in ["https://finance.yahoo.com/rss/topstories"]:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:
                    title = entry.get("title", "")
                    if any(w in title.upper() for w in
                           ["FED", "ECB", "BOE", "BOJ", "RATE", "CPI",
                            "INFLATION", "NFP", "GDP", "TARIFF"]):
                        events.append({
                            "title":    title,
                            "time":     entry.get("published", "")[:16],
                            "impact":   "high",
                            "currency": "MULTI",
                            "actual":   "", "forecast": "", "previous": "",
                            "source":   "Yahoo Finance",
                        })
        except Exception as e:
            logger.warning(f"[Calendar] Yahoo Finance RSS: {e}")
        return events[:20]

    def parse_value(self, value_str: str) -> Optional[float]:
        """Convert strings like '0.3%', '180K', '-2.1B' to float."""
        if not value_str or value_str in ("", "-", "—", "N/A"):
            return None
        try:
            s = value_str.strip().replace(",", "").replace("%", "")
            multiplier = 1.0
            if s.upper().endswith("K"):   multiplier = 1_000;           s = s[:-1]
            elif s.upper().endswith("M"): multiplier = 1_000_000;       s = s[:-1]
            elif s.upper().endswith("B"): multiplier = 1_000_000_000;   s = s[:-1]
            return float(s) * multiplier
        except ValueError:
            return None

    def calculate_deterministic_scores(self, events: list) -> dict:
        """
        Compute deterministic sub-scores from actual vs. estimate comparison.
        This is the v2 core refinement: numeric comparison in Python,
        not left entirely to LLM interpretation.

        Logic:
          - Actual > Forecast → bullish for that currency
          - Actual < Forecast → bearish for that currency
          - Larger relative difference → higher score magnitude
          - Only HIGH impact events are considered
        """
        CURRENCY_TO_ASSETS = {
            "USD": ["EURUSD", "GBPUSD", "USDJPY", "GBPJPY", "USDCAD",
                    "XAUUSD", "MGC", "MES", "NAS100", "MCL", "BTCUSD", "ETHUSD", "6E"],
            "EUR": ["EURUSD", "EURGBP", "EURAUD", "EURCHF", "6E", "GER40"],
            "GBP": ["GBPUSD", "EURGBP", "GBPJPY"],
            "JPY": ["USDJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY"],
            "CAD": ["USDCAD", "MCL"],
            "CHF": ["EURCHF", "CHFJPY"],
        }
        scores = {}

        for event in events:
            if event.get("impact") not in ("high", "medium"):
                continue
            actual_str   = event.get("actual",   "")
            forecast_str = event.get("forecast", "")
            if not actual_str or actual_str in ("", "-", "—"):
                continue

            actual   = self.parse_value(actual_str)
            forecast = self.parse_value(forecast_str)
            if actual is None or forecast is None:
                continue

            diff_pct = (actual - forecast) / abs(forecast) if forecast != 0 else (
                1.0 if actual > 0 else -1.0 if actual < 0 else 0.0
            )
            if abs(diff_pct) > 0.20:   base_score = 6.0
            elif abs(diff_pct) > 0.10: base_score = 4.0
            elif abs(diff_pct) > 0.05: base_score = 2.5
            else:                       base_score = 1.0

            direction = 1 if actual > forecast else -1
            score     = direction * base_score

            title_lower = event.get("title", "").lower()
            if any(k in title_lower for k in ["unemployment", "jobless"]):
                score = -score

            currency = event.get("currency", "")
            assets   = CURRENCY_TO_ASSETS.get(currency, [])
            for asset in assets:
                usd_second = ["USDJPY", "USDCAD"]
                if currency == "USD":
                    asset_score = score if asset in usd_second else -score
                else:
                    asset_score = score
                scores[asset] = scores.get(asset, 0) + asset_score

        return {
            asset: max(-8, min(8, round(score, 1)))
            for asset, score in scores.items()
        }

    def get_imminent_events(self, events: list, hours_ahead: int = 4) -> list:
        """Return only genuinely imminent scheduled economic events."""
        REAL_CURRENCIES    = {"USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"}
        ECONOMIC_KEYWORDS  = [
            "rate", "cpi", "gdp", "nfp", "pmi", "inflation", "employment",
            "payroll", "fomc", "ecb", "boe", "boj", "rba", "decision", "pce",
        ]
        imminent = []
        for event in events:
            if event.get("source", "") not in ("ForexFactory",):
                continue
            if event.get("currency", "") not in REAL_CURRENCIES:
                continue
            if event.get("actual", "") not in ("", "-", "—", "N/A"):
                continue
            if event.get("impact") not in ("high", "medium"):
                continue
            title_lower = event.get("title", "").lower()
            if not any(k in title_lower for k in ECONOMIC_KEYWORDS):
                continue
            imminent.append({
                "title":           event["title"],
                "time_str":        event.get("time", ""),
                "currencies":      [event.get("currency", "")],
                "expected_impact": event["impact"],
                "source":          event.get("source", "ForexFactory"),
            })
        return imminent[:5]


# ─────────────────────────────────────────────────────────────────────────────
# News Collector
# ─────────────────────────────────────────────────────────────────────────────

class NewsCollector:

    def fetch_rss(self, source_name: str, urls: list) -> list:
        import hashlib
        news = []
        for url in urls:
            try:
                headers  = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(url, headers=headers, timeout=8)
                feed     = feedparser.parse(response.content)
                for entry in feed.entries[:10]:
                    title   = entry.get("title", "").strip()
                    summary = entry.get("summary", "")[:300].strip()
                    if not title:
                        continue
                    news.append({
                        "hash":    hashlib.md5(title.encode()).hexdigest(),
                        "title":   title,
                        "summary": summary,
                        "date":    entry.get("published", ""),
                        "source":  source_name,
                        "url":     entry.get("link", ""),
                    })
            except Exception as e:
                logger.warning(f"RSS {source_name} {url}: {e}")
        return news

    def fetch_all(self) -> list:
        all_news = []
        for source, urls in RSS_SOURCES.items():
            news = self.fetch_rss(source, urls)
            all_news.extend(news)
            if news:
                logger.info(f"  {source}: {len(news)} news items")
        seen   = set()
        unique = []
        for n in all_news:
            if n["hash"] not in seen:
                seen.add(n["hash"])
                unique.append(n)
        logger.info(f"Total news: {len(unique)} unique out of {len(all_news)}")
        return unique

    def detect_currencies(self, title: str, summary: str) -> list:
        text  = (title + " " + summary).lower()
        found = [c for c, kws in CURRENCY_KEYWORDS.items() if any(k in text for k in kws)]
        return found if found else ["MULTI"]

    def is_high_impact(self, title: str, summary: str) -> bool:
        text = (title + " " + summary).lower()
        return any(k in text for k in HIGH_IMPACT_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate Filter
# ─────────────────────────────────────────────────────────────────────────────

class DuplicateFilter:

    def __init__(self):
        self._processed: dict = {}

    def filter_new(self, news_list: list) -> list:
        now    = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=PROCESSED_NEWS_TTL_MIN)
        self._processed = {h: t for h, t in self._processed.items() if t > cutoff}
        return [n for n in news_list if n["hash"] not in self._processed]

    def mark_processed(self, news_list: list):
        now = datetime.now(timezone.utc)
        for n in news_list:
            self._processed[n["hash"]] = now


# ─────────────────────────────────────────────────────────────────────────────
# SentimentAgent v2 — STUB
# ─────────────────────────────────────────────────────────────────────────────

class SentimentAgent(BaseAgent):
    """
    NOTE: Production implementation uses GPT-4o-mini for news analysis
    and a deterministic calendar parser. This stub demonstrates the interface
    and preserves the RSS collection + ForexFactory scraping logic.
    The LLM scoring call is replaced with synthetic data.
    See paper Section 2.1 for design rationale.
    """

    AGENT_NAME  = "SentimentAgent"
    MODEL       = "gpt-4o-mini"
    SCORE_RANGE = (-15, 15)

    def __init__(self):
        super().__init__()
        # Production: self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collector   = NewsCollector()
        self.dup_filter  = DuplicateFilter()
        self.calendar    = EconomicCalendarFetcher()
        self._last_scores: dict            = {}
        self._last_run: Optional[datetime] = None
        self._calendar_cache: list         = []
        self._calendar_cache_time: Optional[datetime] = None

    def is_operating_hours(self) -> bool:
        try:
            import pytz
            cet = pytz.timezone("Europe/Rome")
            h   = datetime.now(cet).hour
            return OPERATING_HOURS_START <= h < OPERATING_HOURS_END
        except Exception:
            return True

    def _get_calendar_events(self) -> list:
        """Load calendar with 1-hour cache."""
        now = datetime.now(timezone.utc)
        if (self._calendar_cache_time is None or
                (now - self._calendar_cache_time).total_seconds() > 3600):
            logger.info("[SentimentAgent] Updating economic calendar...")
            self._calendar_cache      = self.calendar.fetch_calendar()
            self._calendar_cache_time = now
        return self._calendar_cache

    async def collect_data(self, context: dict = None) -> dict:
        logger.info("[SentimentAgent] Collecting news...")
        all_news  = self.collector.fetch_all()
        new_news  = self.dup_filter.filter_new(all_news)
        logger.info(
            f"[SentimentAgent] {len(new_news)} new items out of {len(all_news)} total "
            f"({len(all_news) - len(new_news)} duplicates filtered)"
        )
        calendar_events = self._get_calendar_events()
        return {
            "new_news":  new_news,
            "all_news":  all_news,
            "calendar":  calendar_events,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def analyze(self, data: dict, context: dict = None) -> dict:
        """
        NOTE: Production implementation calls GPT-4o-mini for news analysis.
        This stub returns synthetic neutral scores and uses only the
        deterministic calendar scoring (actual vs. forecast).
        """
        new_news        = data.get("new_news",  [])
        calendar_events = data.get("calendar",  [])

        # Deterministic calendar scores (real implementation — no LLM needed)
        det_calendar_scores = {}
        imminent_calendar   = []
        if calendar_events:
            det_calendar_scores = self.calendar.calculate_deterministic_scores(calendar_events)
            imminent_calendar   = self.calendar.get_imminent_events(calendar_events)
            if det_calendar_scores:
                active = {a: s for a, s in det_calendar_scores.items() if s != 0}
                logger.info(f"[SentimentAgent] Calendar scores: {active}")

        # NOTE: LLM scoring call intentionally omitted.
        # Production: calls GPT-4o-mini with news headlines → scores per asset.
        # Stub: return calendar-only scores, zeroed for assets without calendar data.
        scores = {a: det_calendar_scores.get(a, 0.0) for a in self._get_all_assets()}
        scores = self._post_process_scores(scores)

        self.dup_filter.mark_processed(new_news)

        logger.info(
            f"[SentimentAgent] [STUB] Completed — calendar-only scores | "
            f"{len(det_calendar_scores)} assets from calendar"
        )

        high_impact_news  = False
        high_impact_event = None
        for event in imminent_calendar:
            if event.get("expected_impact") in ("high", "medium"):
                high_impact_news  = True
                high_impact_event = event.get("title", "upcoming macro event")
                break

        return {
            "scores":           scores,
            "top_news":         [],
            "imminent_events":  imminent_calendar,
            "market_mood":      "neutral",
            "summary":          f"[stub] Calendar-only: {len(det_calendar_scores)} assets updated",
            "calendar_scores":  det_calendar_scores,
            "high_impact_news": high_impact_news,
            "high_impact_event": high_impact_event,
        }

    def _post_process_scores(self, scores: dict) -> dict:
        """Propagate scores to correlated assets."""
        scores["6E"]  = scores.get("EURUSD", 0)
        scores["MGC"] = scores.get("XAUUSD", 0)
        if scores.get("XAUUSD", 0) != 0:
            scores["XAGUSD"] = round(scores["XAUUSD"] * 0.7, 1)
        eur_bias = scores.get("EURUSD", 0)
        aud_bias = scores.get("AUDUSD", 0)
        if eur_bias != 0 or aud_bias != 0:
            scores["EURAUD"] = round((eur_bias - aud_bias) / 2, 1)
        scores.pop("AUDUSD", None)
        for asset in self._get_all_assets():
            if asset not in scores:
                scores[asset] = 0
        return scores

    async def run_full(self) -> dict:
        if not self.is_operating_hours():
            logger.info("[SentimentAgent] Outside operating hours (08-23 CET) — skip")
            return {
                "scores":           self._last_scores or {a: 0 for a in self._get_all_assets()},
                "flags":            [],
                "top_news":         [],
                "imminent_events":  [],
                "market_mood":      "neutral",
                "summary":          "Outside operating hours",
                "new_news_count":   0,
                "timestamp":        "",
                "calendar_events":  0,
                "high_impact_news": False,
                "skipped":          True,
            }

        logger.info("[SentimentAgent] Starting news screening...")
        data     = await self.collect_data()
        analysis = await self.analyze(data)

        scores = {
            asset: max(-15, min(15, float(analysis.get("scores", {}).get(asset, 0))))
            for asset in self._get_all_assets()
        }

        self._last_scores = scores
        self._last_run    = datetime.now(timezone.utc)

        imminent         = analysis.get("imminent_events", [])
        high_impact_news = analysis.get("high_impact_news", False)

        return {
            "scores":            scores,
            "flags":             [],
            "top_news":          [],
            "imminent_events":   imminent,
            "market_mood":       analysis.get("market_mood", "neutral"),
            "summary":           analysis.get("summary", ""),
            "new_news_count":    len(data["new_news"]),
            "timestamp":         data["timestamp"],
            "calendar_events":   len(data.get("calendar", [])),
            "high_impact_news":  high_impact_news,
            "high_impact_event": analysis.get("high_impact_event"),
        }

    async def run(self, context: dict) -> AgentResult:
        asset     = context.get("asset",     "EURUSD")
        direction = context.get("direction", "BUY")

        full      = await self.run_full()
        raw_score = full["scores"].get(asset, 0)
        final     = raw_score if direction == "BUY" else -raw_score
        final     = max(-15, min(15, final))

        return AgentResult(
            agent=self.AGENT_NAME,
            score=final,
            direction=direction,
            summary=f"{asset}: sentiment {raw_score:+.0f} | {full['summary'][:60]}",
            bull_case=f"Mood: {full['market_mood']}" if raw_score >= 0 else f"Sentiment {raw_score:+.0f}",
            bear_case=f"Sentiment {raw_score:+.0f}" if raw_score < 0 else f"Mood: {full['market_mood']}",
            confidence="high" if abs(raw_score) > 8 else "medium" if abs(raw_score) > 4 else "low",
            details=(
                f"News: {full['new_news_count']} | "
                f"Calendar: {full.get('calendar_events', 0)} events | "
                f"Mood: {full['market_mood']}"
            ),
            raw_data=full,
        )

    def _get_all_assets(self) -> list:
        return [
            "EURUSD", "GBPUSD", "USDJPY", "GBPJPY",
            "XAUUSD", "BTCUSD", "ETHUSD", "EURAUD",
            "USDCAD", "EURGBP", "NZDJPY", "EURCHF",
            "MGC", "MES", "XAGUSD", "6E", "MCL", "NAS100",
        ]
