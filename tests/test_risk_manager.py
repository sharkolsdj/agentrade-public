"""
tests/test_risk_manager.py
Tests for RiskManagerAgent deterministic gates (agents/risk_manager_agent.py).

The paper argues (Section 3.5c) that policy code SHOULD be auditable —
publishing the RiskManagerAgent proves the point. These tests verify
every deterministic gate in isolation.
"""

import pytest
from agents.risk_manager_agent import (
    RiskManagerAgent,
    DECISION_BLOCK,
    DECISION_APPROVE,
    MIN_SL_PIPS,
    MAX_DRAWDOWN_PCT,
    MAX_OPEN_MT4,
    MAX_OPEN_IB,
    MANUAL_THRESHOLD,
    normalize_lot,
)


@pytest.fixture
def agent():
    return RiskManagerAgent()


@pytest.fixture
def base_context():
    """Minimal context that passes all risk gates."""
    return {
        "asset":                "EURUSD",
        "direction":            "BUY",
        "consensus_score":      75.0,
        "agent_scores":         {
            "TechnicalAgent":   75.0,
            "VolProfileAgent":  65.0,
        },
        "sl_pips":              20.0,
        "tp_pips":              40.0,
        "atr_pips":             15.0,
        "open_positions":       [],
        "current_drawdown_pct": 0.0,
    }


class TestRiskGate:

    def test_drawdown_block(self, agent):
        block = agent._risk_gate(
            "EURUSD", "BUY", 75.0, 20.0, [], MAX_DRAWDOWN_PCT + 0.1
        )
        assert block is not None
        assert "drawdown" in block.lower()

    def test_drawdown_ok(self, agent):
        block = agent._risk_gate(
            "EURUSD", "BUY", 75.0, 20.0, [], MAX_DRAWDOWN_PCT - 0.1
        )
        assert block is None

    def test_max_open_mt4_block(self, agent):
        positions = [{"asset": f"ASSET{i}", "broker": "MT4"} for i in range(MAX_OPEN_MT4)]
        block = agent._risk_gate("EURUSD", "BUY", 75.0, 20.0, positions, 0.0)
        assert block is not None

    def test_max_open_ib_block(self, agent):
        positions = [{"asset": f"ASSET{i}", "broker": "IB"} for i in range(MAX_OPEN_IB)]
        block = agent._risk_gate("MGC", "BUY", 75.0, 20.0, positions, 0.0)
        assert block is not None

    def test_duplicate_asset_block(self, agent):
        positions = [{"asset": "EURUSD", "broker": "MT4"}]
        block = agent._risk_gate("EURUSD", "BUY", 75.0, 20.0, positions, 0.0)
        assert block is not None
        assert "EURUSD" in block

    def test_below_threshold_block(self, agent):
        block = agent._risk_gate(
            "EURUSD", "BUY", MANUAL_THRESHOLD - 5.0, 20.0, [], 0.0
        )
        assert block is not None

    def test_zero_sl_block(self, agent):
        block = agent._risk_gate("EURUSD", "BUY", 75.0, 0.0, [], 0.0)
        assert block is not None

    def test_sl_below_minimum_block(self, agent):
        min_sl = MIN_SL_PIPS.get("EURUSD", 10.0)
        block  = agent._risk_gate("EURUSD", "BUY", 75.0, min_sl - 1.0, [], 0.0)
        assert block is not None

    def test_sl_at_minimum_passes(self, agent):
        min_sl = MIN_SL_PIPS.get("EURUSD", 10.0)
        block  = agent._risk_gate("EURUSD", "BUY", 75.0, min_sl, [], 0.0)
        assert block is None

    def test_all_ok_returns_none(self, agent):
        block = agent._risk_gate("EURUSD", "BUY", 75.0, 20.0, [], 0.0)
        assert block is None


class TestMinSlPips:

    def test_gold_higher_minimum(self):
        assert MIN_SL_PIPS["XAUUSD"] > MIN_SL_PIPS["EURUSD"]

    def test_jpy_pairs_higher_minimum(self):
        assert MIN_SL_PIPS["GBPJPY"] > MIN_SL_PIPS["EURUSD"]

    def test_crypto_highest_minimum(self):
        assert MIN_SL_PIPS["BTCUSD"] > MIN_SL_PIPS["XAUUSD"]

    def test_all_minimums_positive(self):
        assert all(v > 0 for v in MIN_SL_PIPS.values())


class TestSizing:

    def test_mt4_lot_above_zero(self, agent):
        sizing = agent._calculate_sizing("EURUSD", "BUY", 10000.0, 20.0, "MT4_CFD", 10.0)
        assert sizing["lot_size"] > 0

    def test_ib_micro_returns_contract(self, agent):
        sizing = agent._calculate_sizing("MES", "BUY", 10000.0, 8.0, "IB_MICRO", 1.25)
        assert sizing["lot_size"] >= 1

    def test_risk_pct_within_bounds(self, agent):
        sizing = agent._calculate_sizing("EURUSD", "BUY", 10000.0, 20.0, "MT4_CFD", 10.0)
        assert sizing["risk_pct"] <= 2.5   # reasonable upper bound

    def test_risk_usd_consistent(self, agent):
        sizing = agent._calculate_sizing("EURUSD", "BUY", 10000.0, 20.0, "MT4_CFD", 10.0)
        expected = sizing["lot_size"] * 10.0 * 20.0
        assert abs(sizing["risk_usd"] - expected) < 1.0


class TestTpLevels:

    def test_mt4_produces_partial_type(self, agent):
        levels = agent._calculate_tp_levels(
            direction="BUY", sl_pips=20.0, instrument="MT4_CFD", lot_size=0.10,
        )
        assert levels["type"] in ("SINGLE_TP_PARTIAL", "SINGLE_TP_RANGE")

    def test_ib_produces_single_type(self, agent):
        levels = agent._calculate_tp_levels(
            direction="BUY", sl_pips=8.0, instrument="IB_MICRO", lot_size=1,
        )
        assert levels["type"] == "SINGLE_TP"

    def test_range_day_produces_range_type(self, agent):
        levels = agent._calculate_tp_levels(
            direction="BUY", sl_pips=10.0, instrument="MT4_CFD", lot_size=0.10,
            range_day_mode=True, range_day_tp_rr=1.8,
        )
        assert levels["type"] == "SINGLE_TP_RANGE"

    def test_partial_pips_above_sl(self, agent):
        levels = agent._calculate_tp_levels(
            direction="BUY", sl_pips=20.0, instrument="MT4_CFD", lot_size=0.10,
        )
        assert levels.get("partial_pips", levels.get("tp_pips", 0)) > 20.0

    def test_rr_ratio_positive(self, agent):
        levels = agent._calculate_tp_levels(
            direction="BUY", sl_pips=20.0, instrument="IB_MICRO", lot_size=1,
        )
        assert levels.get("rr_ratio", 0) > 0


class TestCorrelationCheck:

    def test_no_positions_returns_none(self, agent):
        assert agent._correlation_check("EURUSD", []) is None

    def test_non_correlated_asset_returns_none(self, agent):
        positions = [{"asset": "BTCUSD"}]
        result = agent._correlation_check("EURUSD", positions)
        # BTCUSD and EURUSD are not in the same correlated group
        assert result is None or "BLOCK" not in result

    def test_correlated_asset_returns_warning(self, agent):
        positions = [{"asset": "GBPUSD"}]
        result = agent._correlation_check("EURUSD", positions)
        # EURUSD and GBPUSD are correlated — should return at least a warning
        assert result is not None
