"""
broker/base_broker.py
Abstract broker interface for AgenTrade.
Both MT4 (DWX ZeroMQ) and IB (ib_insync) implement this interface.

Key design decision (see paper Section 3.1):
The broker-asymmetric lifecycle is modeled explicitly here rather than
abstracted away. MT4 supports partial close at the broker level;
IB Micro Futures trade in whole contracts. The abstract interface
reflects this through the `supports_partial_close` property.
"""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class BaseBroker(ABC):
    """
    Abstract base class for all AgenTrade broker connectors.

    Subclasses: MT4Broker (DWX ZeroMQ), IBConnector (ib_insync)
    """

    # ── Connection ────────────────────────────────────────────────────────────

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the broker.
        Returns True if connected successfully.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Close the broker connection cleanly."""

    @property
    @abstractmethod
    def connected(self) -> bool:
        """True if currently connected to the broker."""

    # ── Account ───────────────────────────────────────────────────────────────

    @abstractmethod
    async def get_balance(self) -> float:
        """Return net account balance in USD."""

    @abstractmethod
    async def get_open_positions(self) -> list:
        """
        Return all open positions.
        Each position is a dict with at minimum:
            {
                "asset":     str,    # e.g. "EURUSD"
                "direction": str,    # "BUY" | "SELL"
                "quantity":  float,  # lots (MT4) or contracts (IB)
                "ticket":    str,    # broker order ID
                "broker":    str,    # "MT4" | "IB"
            }
        """

    # ── Order execution ───────────────────────────────────────────────────────

    @abstractmethod
    async def place_order(
        self,
        asset:       str,
        direction:   str,      # "BUY" | "SELL"
        quantity:    float,
        entry_price: float,
        stop_loss:   float,
        take_profit: float,
        order_type:  str = "MKT",
    ) -> Optional[dict]:
        """
        Place a bracket order (entry + SL + TP).

        Returns a dict with order details on success, None on failure:
            {
                "broker":      "MT4" | "IB",
                "order_id":    str,
                "asset":       str,
                "direction":   str,
                "quantity":    float,
                "entry_price": float,
                "stop_loss":   float,
                "take_profit": float,
                "status":      "SUBMITTED" | "FILLED" | "REJECTED",
            }
        """

    @abstractmethod
    async def close_position(self, asset: str, quantity: float = None) -> bool:
        """
        Close an existing position (market order).
        quantity=None → close the full position.
        Returns True on success.
        """

    # ── Historical data ───────────────────────────────────────────────────────

    @abstractmethod
    async def get_historical_bars(
        self,
        asset:     str,
        timeframe: str,    # "M5" | "M15" | "M30" | "H1" | "H4" | "D1"
        bars:      int = 200,
    ) -> Optional[pd.DataFrame]:
        """
        Download OHLCV historical data.
        Returns DataFrame with Open/High/Low/Close/Volume columns (datetime index),
        or None on failure.
        """

    # ── Broker-specific capabilities ──────────────────────────────────────────

    @property
    def supports_partial_close(self) -> bool:
        """
        True if the broker supports partial position close at the broker level.

        MT4 CFD: True — partial close is a native broker operation.
        IB Micro Futures: False — contracts are not fractional; the lifecycle
        uses breakeven shift instead.

        This property drives the SINGLE_TP_PARTIAL vs SINGLE_TP lifecycle
        selection in RiskManagerAgent._calculate_tp_levels().
        See paper Section 3.1 for rationale.
        """
        return True

    @property
    def broker_name(self) -> str:
        """Human-readable broker name."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.broker_name}(connected={self.connected})"
