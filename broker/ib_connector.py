"""
broker/ib_connector.py
Interactive Brokers connector via ib_insync.
Handles orders for Forex, Metals, and Indices.

Key design decision: ContFuture (continuous futures with automatic rollover)
used for micro futures (MES/MGC/MCL/6E) instead of hardcoded conId.
qualifyContracts() resolves the front-month contract automatically.
See paper Section 3.7 for rationale.

Connection parameters are read from environment variables.
Copy .env.example to .env and fill in your values.
"""
import asyncio
import os
import time
from typing import Optional, Dict, Any, Tuple
from loguru import logger
import pandas as pd
from ib_insync import (
    IB, Contract, Forex, Future, ContFuture,
    MarketOrder, LimitOrder, StopOrder,
    BracketOrder, util
)


# ── Symbol → IB Contract mapping ─────────────────────────────────────────────
def build_contract(asset: str) -> Contract:
    """Build the correct IB contract for each asset."""

    # Forex pairs
    forex_map = {
        "EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"),
        "USDJPY": ("USD", "JPY"), "GBPJPY": ("GBP", "JPY"),
        "EURJPY": ("EUR", "JPY"), "USDCHF": ("USD", "CHF"),
        "AUDUSD": ("AUD", "USD"), "USDCAD": ("USD", "CAD"),
    }
    # Metals (Forex pairs in IB)
    metals_map = {
        "XAUUSD": ("XAU", "USD"),  # Gold
        "XAGUSD": ("XAG", "USD"),  # Silver
    }
    # Indices and micro futures
    futures_map = {
        "ES":  ("ES",  "CME"),    # S&P 500 E-mini
        "NQ":  ("NQ",  "CME"),    # Nasdaq E-mini
        "DAX": ("DAX", "EUREX"),
        "MES": ("ES",  "CME"),    # DATA: FULL E-mini S&P500 (full volume) — exec on MES micro
        "MGC": ("GC",  "COMEX"),  # DATA: FULL Gold (full volume) — exec on MGC micro
        "MCL": ("CL",  "NYMEX"),  # DATA: FULL WTI (full volume) — exec on MCL micro
        "6E":  ("EUR", "CME"),    # Euro FX
    }

    if asset in forex_map:
        base, quote = forex_map[asset]
        return Forex(f"{base}{quote}")

    if asset in metals_map:
        base, quote = metals_map[asset]
        return Forex(f"{base}{quote}")

    if asset in futures_map:
        symbol, exchange = futures_map[asset]
        # ContFuture = continuous contract (automatic rollover)
        # qualifyContracts() resolves the current front-month automatically.
        # This eliminates the need to hardcode conId values that expire every quarter.
        contract = ContFuture(symbol, exchange=exchange)
        return contract

    raise ValueError(f"Asset not supported: {asset}")


class IBConnector:
    """
    Interactive Brokers connector.
    Supports Forex, Precious Metals, and Futures.
    """

    def __init__(self):
        self.ib = IB()
        self.connected = False
        self._reconnect_attempts = 0
        self._max_reconnects = 5

    async def connect(self) -> bool:
        """Connect to IB Gateway or TWS."""
        try:
            await self.ib.connectAsync(
                host=os.getenv("IB_HOST", "127.0.0.1"),
                port=int(os.getenv("IB_PORT", "7497")),
                clientId=int(os.getenv("IB_CLIENT_ID_EXEC", "11")),
                timeout=20,
            )
            self.connected = True
            self._reconnect_attempts = 0
            account = os.getenv("IB_ACCOUNT") or self.ib.managedAccounts()[0]
            logger.info(f"IB connected. Account: {account}")

            self.ib.disconnectedEvent += self._on_disconnect
            return True

        except Exception as e:
            logger.error(f"IB connection failed: {e}")
            self.connected = False
            return False

    def _on_disconnect(self):
        logger.warning("IB disconnected! Attempting reconnect...")
        self.connected = False
        asyncio.create_task(self._reconnect())

    async def _reconnect(self):
        """Reconnect with exponential backoff."""
        while self._reconnect_attempts < self._max_reconnects:
            self._reconnect_attempts += 1
            wait_sec = 2 ** self._reconnect_attempts
            logger.info(f"IB reconnect in {wait_sec}s (attempt {self._reconnect_attempts})")
            await asyncio.sleep(wait_sec)
            if await self.connect():
                return
        logger.critical("IB: unable to reconnect after 5 attempts!")

    async def get_account_summary(self) -> Dict[str, Any]:
        """Retrieve account summary (balance, margin, etc.)."""
        if not self.connected:
            return {}
        account_values = await self.ib.reqAccountSummaryAsync()
        summary = {}
        for av in account_values:
            summary[av.tag] = {"value": av.value, "currency": av.currency}
        return summary

    async def get_balance(self) -> float:
        """Retrieve net account balance in USD."""
        summary = await self.get_account_summary()
        net_liq = summary.get("NetLiquidation", {})
        return float(net_liq.get("value", 0))

    async def get_open_positions(self) -> list:
        """Retrieve all open positions."""
        if not self.connected:
            return []
        positions = await self.ib.reqPositionsAsync()
        result = []
        for pos in positions:
            result.append({
                "asset":    pos.contract.symbol,
                "quantity": pos.position,
                "avg_cost": pos.avgCost,
                "contract": pos.contract,
            })
        return result

    async def calculate_position_size(
        self,
        asset: str,
        entry_price: float,
        stop_loss: float,
        risk_pct: float = None,
    ) -> float:
        """
        Compute position size based on risk percentage of equity.
        Fixed risk in % of capital.
        """
        risk_pct    = risk_pct or float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))
        balance     = await self.get_balance()
        risk_amount = balance * (risk_pct / 100)

        sl_distance = abs(entry_price - stop_loss)
        if sl_distance == 0:
            return 0.01

        if asset in ["ES", "NQ"]:
            multiplier = 50 if asset == "ES" else 20
            size = round(risk_amount / (sl_distance * multiplier))
            return max(1, size)
        elif asset in ["XAUUSD"]:
            size = round(risk_amount / (sl_distance * 100), 2)
            return max(0.01, size)
        else:
            pip_size  = 0.0001 if "JPY" not in asset else 0.01
            pip_value = pip_size * 100000
            sl_pips   = sl_distance / pip_size
            size      = round(risk_amount / (sl_pips * (pip_value / 1000)), 2)
            return max(0.01, min(size, 10.0))

    async def place_order(
        self,
        asset: str,
        direction: str,        # "BUY" or "SELL"
        quantity: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        order_type: str = "MKT",  # "MKT" or "LMT"
    ) -> Optional[Dict]:
        """
        Place a bracket order (entry + SL + TP).
        Returns order details dict.

        Broker-asymmetric lifecycle (see paper Section 3.1):
        IB Micro uses SINGLE_TP + BE-to-entry at 50% TP.
        MT4 uses SINGLE_TP_PARTIAL with Python-side monitor for partial close.
        """
        if not self.connected:
            logger.error("IB not connected, cannot send order")
            return None

        try:
            contract = build_contract(asset)
            await self.ib.qualifyContractsAsync(contract)

            action       = "BUY"  if direction == "BUY"  else "SELL"
            close_action = "SELL" if direction == "BUY"  else "BUY"

            if order_type == "MKT":
                parent_order = MarketOrder(action=action, totalQuantity=quantity)
            else:
                parent_order = LimitOrder(
                    action=action, totalQuantity=quantity, lmtPrice=entry_price
                )

            sl_order = StopOrder(
                action=close_action, totalQuantity=quantity, stopPrice=stop_loss,
            )
            sl_order.parentId = parent_order.orderId

            tp_order = LimitOrder(
                action=close_action, totalQuantity=quantity, lmtPrice=take_profit,
            )
            tp_order.parentId = parent_order.orderId
            tp_order.transmit = True  # transmit all at once

            parent_trade = self.ib.placeOrder(contract, parent_order)
            sl_trade     = self.ib.placeOrder(contract, sl_order)
            tp_trade     = self.ib.placeOrder(contract, tp_order)

            await asyncio.sleep(1)  # wait for confirmation

            logger.success(
                f"IB order sent: {direction} {quantity} {asset} "
                f"@ {entry_price} | SL: {stop_loss} | TP: {take_profit}"
            )

            return {
                "broker":       "IB",
                "order_id":     str(parent_trade.order.orderId),
                "asset":        asset,
                "direction":    direction,
                "quantity":     quantity,
                "entry_price":  entry_price,
                "stop_loss":    stop_loss,
                "take_profit":  take_profit,
                "status":       "SUBMITTED",
                "ib_trade":     parent_trade,
            }

        except Exception as e:
            logger.error(f"IB order failed for {asset}: {e}")
            return None

    async def close_position(self, asset: str, quantity: float = None) -> bool:
        """Close an existing position (market order)."""
        positions = await self.get_open_positions()
        for pos in positions:
            if pos["asset"] == asset:
                qty      = abs(quantity or pos["quantity"])
                action   = "SELL" if pos["quantity"] > 0 else "BUY"
                contract = pos["contract"]
                order    = MarketOrder(action=action, totalQuantity=qty)
                self.ib.placeOrder(contract, order)
                logger.info(f"Position {asset} closed ({qty} units)")
                return True
        logger.warning(f"No open position for {asset}")
        return False

    async def get_historical_bars(
        self,
        asset: str,
        timeframe: str,
        bars: int = 200,
    ) -> Optional[pd.DataFrame]:
        """
        Download OHLCV historical data from IB using reqHistoricalDataAsync.

        Args:
            asset:     symbol (MES, MGC, MCL, 6E, EURUSD, XAUUSD, etc.)
            timeframe: M5, M15, M30, H1, H4, D1
            bars:      number of bars to download (default 200)

        Returns:
            DataFrame with Open/High/Low/Close/Volume columns or None on failure
        """
        if not self.connected:
            logger.error("IB not connected, cannot download historical data")
            return None

        try:
            timeframe_map = {
                "M5":  "5 mins",
                "M15": "15 mins",
                "M30": "30 mins",
                "H1":  "1 hour",
                "H4":  "4 hours",
                "D1":  "1 day",
            }

            if timeframe not in timeframe_map:
                logger.error(f"Unsupported timeframe: {timeframe}")
                return None

            bar_size = timeframe_map[timeframe]

            # IB pacing: max 6 requests/2s for timeframe <= M30
            if timeframe in ["M5", "M15", "M30"]:
                if not hasattr(self, '_last_hist_request'):
                    self._last_hist_request  = 0
                    self._hist_request_count = 0
                current_time = time.time()
                if current_time - self._last_hist_request < 2:
                    if self._hist_request_count >= 6:
                        wait_time = 2 - (current_time - self._last_hist_request)
                        logger.info(f"IB pacing: waiting {wait_time:.1f}s")
                        await asyncio.sleep(wait_time)
                        self._hist_request_count = 0
                        self._last_hist_request  = time.time()
                    else:
                        self._hist_request_count += 1
                else:
                    self._hist_request_count = 1
                    self._last_hist_request  = current_time

            contract = build_contract(asset)
            await self.ib.qualifyContractsAsync(contract)

            if timeframe == "D1":
                duration = f"{bars} D"
            elif timeframe in ["H1", "H4"]:
                hours = bars * (1 if timeframe == "H1" else 4)
                duration = f"{hours // 24} D" if hours > 24 else f"{hours} H"
            else:
                minutes = bars * int(timeframe[1:])
                if minutes <= 60:
                    duration = f"{minutes} M"
                elif minutes <= 24 * 60:
                    duration = f"{minutes // 60} H"
                else:
                    duration = f"{minutes // 60 // 24} D"

            logger.debug(f"IB hist request: {asset} {timeframe} {bars} bars ({duration})")

            hist_data = await self.ib.reqHistoricalDataAsync(
                contract=contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
                keepUpToDate=False,
            )

            if not hist_data:
                logger.warning(f"No historical data received for {asset} {timeframe}")
                return None

            df_data = []
            for bar in hist_data:
                df_data.append({
                    "Open":   float(bar.open),
                    "High":   float(bar.high),
                    "Low":    float(bar.low),
                    "Close":  float(bar.close),
                    "Volume": int(bar.volume) if bar.volume != -1 else 0,
                })

            df       = pd.DataFrame(df_data)
            df.index = pd.to_datetime([bar.date for bar in hist_data])

            logger.success(f"IB data downloaded: {asset} {timeframe} - {len(df)} bars")
            return df

        except Exception as e:
            logger.error(f"IB get_historical_bars failed for {asset} {timeframe}: {e}")
            return None

    def disconnect(self):
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("IB disconnected")


# Singleton
ib_connector = IBConnector()
