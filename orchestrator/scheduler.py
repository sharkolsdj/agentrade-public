"""
orchestrator/scheduler.py
Main loop of the AgenTrade system.

Operation:
  1. Checks schedule (active session, polling interval)
  2. Runs pre-filter across all assets
  3. For each candidate signal → launches the full workflow
  4. Handles APPROVE/BLOCK/MODIFY via Telegram
  5. Executes approved trades
  6. Logs everything to PostgreSQL

Launch: python -m orchestrator.scheduler

PARTIAL: Scan loop, session gating, and pre-filter integration are real.
Internal modules (TradeExecutor, PositionTracker, DBLogger, EntryLayer,
TelegramDispatcher) are not included in this public repository.
"""

import asyncio
import os
import signal
from datetime import datetime, timezone
from typing import Optional

from loguru import logger
from dotenv import load_dotenv

from orchestrator.pre_filter import PreFilter, MarketSchedule, FilterSignal
from orchestrator.graph      import run_workflow, AUTO_EXECUTE_THRESHOLD, MANUAL_THRESHOLD
from orchestrator.atr_regime import get_session_manual_threshold
from broker.data_cache       import data_cache
from broker.ib_datafeed      import ib_datafeed
from agents.macro_agent      import MacroAgent

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MAX_CONCURRENT_WORKFLOWS = 5

# Per-session rapid scan intervals (seconds)
RAPID_INTERVALS: dict[str, int] = {
    "LONDON_OPEN":   3 * 60,
    "LONDON_NY_PRE": 3 * 60,
    "NY_OPEN":       3 * 60,
    "LONDON_CLOSE":  3 * 60,
    "LONDON_MID":    5 * 60,
    "NY_MID":        5 * 60,
    "ASIAN":        10 * 60,
    "NY_LATE":      15 * 60,
    "DEAD_ZONE":    60 * 60,
}


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class AgenTradeScheduler:
    """
    Main AgenTrade scheduler.
    Manages the scan loop, session gating, and workflow dispatch.

    Production internal modules not included in this public version:
      - TradeExecutor: broker order submission (MT4 DWX + IB Gateway)
      - PositionTracker: live position lifecycle (partial close, BE, trailing)
      - TelegramDispatcher: sends notifications and waits for APPROVE/BLOCK
      - DBLogger: PostgreSQL trade logging
      - EntryLayer: M5 entry refinement before execution
    """

    def __init__(self):
        self.pre_filter    = PreFilter()
        self.macro_agent   = MacroAgent()
        self._running      = False
        self._semaphore    = asyncio.Semaphore(MAX_CONCURRENT_WORKFLOWS)
        self._active_workflows: set[str] = set()

        # Stubs for production modules not in this public version
        self._trade_executor   = None   # TradeExecutor — not published
        self._position_tracker = None   # PositionTracker — not published
        self._telegram         = None   # TelegramDispatcher — not published
        self._db_logger        = None   # DBLogger — not published

    # ── Startup ───────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialize all subsystems and start the scan loop."""
        logger.info("=" * 60)
        logger.info("  AgenTrade — Starting")
        logger.info("=" * 60)

        # Start DataCache (background OHLCV refresh)
        await data_cache.start()
        logger.info("[Scheduler] DataCache started")

        # Start IB datafeed for micro futures (lazy — connects on first request)
        ib_datafeed.start()
        logger.info("[Scheduler] IB datafeed initialized")

        # Pre-warm MacroAgent cache (07:00 CET daily run)
        if self.macro_agent.should_run_now():
            logger.info("[Scheduler] Pre-warming MacroAgent cache (07:00 window)")
            try:
                await self.macro_agent.run_full()
            except Exception as e:
                logger.warning(f"[Scheduler] MacroAgent pre-warm failed: {e}")

        # Register signal handlers for clean shutdown
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_event_loop().add_signal_handler(
                sig, lambda: asyncio.create_task(self.stop())
            )

        self._running = True
        logger.info("[Scheduler] System ready — starting scan loop")
        await self._scan_loop()

    async def stop(self) -> None:
        """Graceful shutdown."""
        logger.info("[Scheduler] Shutdown requested")
        self._running = False
        await data_cache.stop()
        logger.info("[Scheduler] Shutdown complete")

    # ── Scan loop ─────────────────────────────────────────────────────────────

    async def _scan_loop(self) -> None:
        """
        Main scan loop.
        Polls pre-filter at session-appropriate intervals.
        Launches full workflows for candidate signals.
        """
        while self._running:
            try:
                session, interval_min = MarketSchedule.current()
                interval_sec = RAPID_INTERVALS.get(session, interval_min * 60)

                logger.info(
                    f"[Scheduler] Session: {session} | "
                    f"Next scan in {interval_sec // 60}m {interval_sec % 60}s"
                )

                if session in ("WEEKEND_SAT", "WEEKEND_SUN"):
                    await asyncio.sleep(interval_sec)
                    continue

                # Pre-filter: scan all assets
                skip_assets = self._get_open_assets()
                signals     = await self.pre_filter.scan_all(skip_assets=skip_assets)

                if signals:
                    logger.info(f"[Scheduler] {len(signals)} candidate signals → launching workflows")
                    workflow_tasks = [
                        self._run_workflow_gated(signal)
                        for signal in signals
                    ]
                    await asyncio.gather(*workflow_tasks, return_exceptions=True)
                else:
                    logger.debug(f"[Scheduler] No signals — next scan in {interval_sec}s")

                # MacroAgent daily pre-warm check
                if self.macro_agent.should_run_now():
                    asyncio.create_task(self._prewarm_macro())

                await asyncio.sleep(interval_sec)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Scheduler] Scan loop error: {e}")
                await asyncio.sleep(60)

    # ── Workflow dispatch ─────────────────────────────────────────────────────

    async def _run_workflow_gated(self, signal: FilterSignal) -> None:
        """
        Run a workflow with concurrency control.
        Skips if max concurrent workflows reached or asset already active.
        """
        asset_dir = f"{signal.asset}_{signal.direction}"

        if asset_dir in self._active_workflows:
            logger.debug(f"[Scheduler] {asset_dir} already active — skip")
            return

        async with self._semaphore:
            self._active_workflows.add(asset_dir)
            try:
                await self._run_full_workflow(signal)
            finally:
                self._active_workflows.discard(asset_dir)

    async def _run_full_workflow(self, signal: FilterSignal) -> None:
        """
        Run the complete LangGraph workflow for a candidate signal.

        Production flow:
          1. run_workflow() → LangGraph pipeline (7 agents)
          2. If APPROVE → EntryLayer M5 refinement (not published)
          3. TradeExecutor → broker order (not published)
          4. TelegramDispatcher → notification + approval wait (not published)
          5. DBLogger → PostgreSQL log (not published)
        """
        session         = MarketSchedule.current()[0]
        session_thr     = get_session_manual_threshold(session)
        open_positions  = self._get_open_positions()
        drawdown        = self._get_current_drawdown()

        logger.info(
            f"[Scheduler] ▶ Workflow: {signal.asset} {signal.direction} | "
            f"session={session} | tech={signal.score}/6 ict={signal.ict_score}/12"
        )

        try:
            final_state = await run_workflow(
                asset                = signal.asset,
                direction            = signal.direction,
                open_positions       = open_positions,
                current_drawdown_pct = drawdown,
            )
        except Exception as e:
            logger.error(f"[Scheduler] Workflow error {signal.asset}: {e}")
            return

        skip_reason   = final_state.get("skip_reason", "")
        risk_decision = final_state.get("risk_decision", "BLOCK")
        consensus     = final_state.get("consensus_score", 0)

        if skip_reason:
            logger.info(f"[Scheduler] {signal.asset} {signal.direction} → SKIP: {skip_reason}")
            return

        if risk_decision == "BLOCK":
            logger.info(f"[Scheduler] {signal.asset} {signal.direction} → BLOCKED by RiskManager")
            # Production: send block notification to Telegram
            return

        if abs(consensus) < session_thr:
            logger.info(
                f"[Scheduler] {signal.asset} {signal.direction} → "
                f"NO_TRADE: consensus {consensus:+.1f} < session threshold {session_thr}"
            )
            return

        if risk_decision in ("APPROVE", "MODIFY"):
            logger.info(
                f"[Scheduler] {signal.asset} {signal.direction} → "
                f"{risk_decision} | consensus={consensus:+.1f}"
            )

            if abs(consensus) >= AUTO_EXECUTE_THRESHOLD:
                # Production: auto-execute trade + send Telegram notification
                logger.info(f"[Scheduler] AUTO-EXECUTE: {signal.asset} {signal.direction}")
                # self._trade_executor.execute(final_state)      # not published
                # self._telegram.send_auto(final_state)          # not published
            else:
                # Production: send Telegram notification + wait for manual APPROVE
                logger.info(f"[Scheduler] MANUAL APPROVE: {signal.asset} {signal.direction}")
                # approved = await self._telegram.wait_for_approval(final_state)  # not published
                # if approved: self._trade_executor.execute(final_state)          # not published

    # ── Account state helpers ─────────────────────────────────────────────────

    def _get_open_positions(self) -> list:
        """
        Return current open positions.
        Production: reads from PositionTracker (not published).
        Stub: returns empty list.
        """
        if self._position_tracker is not None:
            return self._position_tracker.get_open_positions()
        return []

    def _get_open_assets(self) -> set:
        """Return set of assets with open positions (to skip in pre-filter)."""
        positions = self._get_open_positions()
        return {p.get("asset", "") for p in positions}

    def _get_current_drawdown(self) -> float:
        """
        Return current drawdown percentage.
        Production: computed from DBLogger equity curve (not published).
        Stub: returns 0.0.
        """
        return 0.0

    async def _prewarm_macro(self) -> None:
        """Pre-warm MacroAgent cache (runs once daily at 07:00 CET)."""
        try:
            logger.info("[Scheduler] Pre-warming MacroAgent (daily run)")
            await self.macro_agent.run_full(force_refresh=True)
        except Exception as e:
            logger.warning(f"[Scheduler] MacroAgent pre-warm error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    scheduler = AgenTradeScheduler()
    await scheduler.start()


if __name__ == "__main__":
    asyncio.run(main())
