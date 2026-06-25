"""
agents/base_agent.py
Common base class for all agents in the AgenTrade system.
Defines the standard interface expected by the orchestrator.

v2.0 — changes:
  - AgentResult.score: now always 0-100 (was ± per agent)
  - AgentResult.direction: added — the analyzed direction (BUY|SELL)
  - SCORE_RANGE: updated to (0, 100)
  - to_telegram_line(): updated for 0-100 score
  - run(): removed ± clamp — each agent already delivers score 0-100
"""
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from loguru import logger


@dataclass
class AgentResult:
    """Standardized result from every agent."""
    agent:      str                   # Agent name
    score:      float                 # Score 0-100 (v2 — was ± per agent)
    summary:    str                   # Short text for Telegram notification
    bull_case:  str                   # Main bullish argument
    bear_case:  str                   # Main bearish argument
    confidence: str                   # high | medium | low
    details:    str                   # Detailed analysis
    direction:  str = ""              # v2: "BUY" | "SELL" — analyzed direction
    raw_data:   dict = field(default_factory=dict)  # Raw collected data
    timestamp:  datetime = field(default_factory=datetime.utcnow)
    error:      Optional[str] = None  # Error message if any

    def is_valid(self) -> bool:
        return self.error is None

    def to_telegram_line(self) -> str:
        """
        Formatted line for the Telegram notification.
        v2: score 0-100 — icon based on confidence in the direction.
          > 60 = supports direction (📈 for BUY, 📉 for SELL)
          40-60 = neutral (➡️)
          < 40 = contradicts direction
        """
        if self.score > 60:
            icon = "📈" if self.direction == "BUY" else "📉"
        elif self.score < 40:
            icon = "📉" if self.direction == "BUY" else "📈"
        else:
            icon = "➡️"
        return f"  {self.agent}: {icon} {self.score:.0f}/100  ({self.summary})"


class BaseAgent:
    """
    Base class for all AgenTrade agents.

    Each specialized agent extends this class and implements:
    - collect_data(): collects raw data from sources
    - analyze(): calls the LLM with the system prompt

    v2: each agent applies utils.score_converter.agent_score()
    in its own output section before returning the result.
    BaseAgent no longer performs ± clamping — it receives scores already
    in the 0-100 range.
    """

    # Override in each subclass
    AGENT_NAME  = "BaseAgent"
    MODEL       = "claude-sonnet-4-6"
    SCORE_RANGE = (0, 100)   # v2: all agents use 0-100

    def __init__(self):
        self.name  = self.AGENT_NAME
        self.model = self.MODEL
        logger.info(f"[{self.name}] Initialized — model: {self.model}")

    async def collect_data(self, context: dict) -> dict:
        """
        Collect raw data from the agent's specific sources.
        Must be implemented in each subclass.

        Args:
            context: dict with asset, direction, timeframe, etc.
        Returns:
            dict with structured data ready for analysis
        """
        raise NotImplementedError(f"{self.name}: collect_data() not implemented")

    async def analyze(self, data: dict, context: dict) -> dict:
        """
        Call the LLM with the system prompt and collected data.
        Must be implemented in each subclass.

        Args:
            data:    output of collect_data()
            context: same context passed to collect_data()
        Returns:
            dict with score (0-100), summary, bull_case, bear_case, details
        """
        raise NotImplementedError(f"{self.name}: analyze() not implemented")

    async def run(self, context: dict) -> AgentResult:
        """
        Main method — orchestrates collect_data + analyze.
        Should not be overridden in subclasses except in special cases.

        Args:
            context: {
                'asset':     'EURUSD',
                'direction': 'BUY',   # BUY | SELL
                'timeframe': 'H4',
                'tier':      1,
                'timestamp': datetime
            }
        Returns:
            AgentResult with score 0-100, direction, and Telegram data
        """
        asset     = context.get("asset", "")
        direction = context.get("direction", "")
        logger.info(f"[{self.name}] Starting analysis — asset: {asset} dir: {direction}")

        try:
            # Collect data
            data = await self.collect_data(context)

            # Analyze — each agent applies score_converter internally
            result = await self.analyze(data, context)

            # v2: score already 0-100 — safety clamp
            score = max(0.0, min(100.0, float(result.get("score", 50))))

            agent_result = AgentResult(
                agent=self.name,
                score=score,
                direction=direction,
                summary=result.get("summary", ""),
                bull_case=result.get("bull_case", ""),
                bear_case=result.get("bear_case", ""),
                confidence=result.get("confidence", "medium"),
                details=result.get("details", ""),
                raw_data=data,
            )

            logger.info(
                f"[{self.name}] Completed — score: {score:.1f}/100 "
                f"dir: {direction} confidence: {agent_result.confidence}"
            )
            return agent_result

        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
            return AgentResult(
                agent=self.name,
                score=50.0,       # v2: fallback to neutral (was 0 in v1)
                direction=direction,
                summary=f"Analysis error: {str(e)[:50]}",
                bull_case="",
                bear_case="",
                confidence="low",
                details="",
                error=str(e),
            )

    def _clamp_score(self, score: float) -> float:
        """
        Safety clamp to 0-100 range.
        v2: kept for compatibility but range updated.
        """
        return max(0.0, min(100.0, score))
