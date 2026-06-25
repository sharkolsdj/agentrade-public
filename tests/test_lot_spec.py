"""
tests/test_lot_spec.py
Tests for lot normalisation (utils/lot_spec.py).

Broker-specific lot rules: indices and oil CFDs on STARTRADER have
Volume Min = 0.10 and Volume Step = 0.10. An arbitrary lot like 0.29
is rejected by the broker. normalize_lot() rounds to the nearest valid
multiple and enforces the minimum.
"""

import pytest
from utils.lot_spec import normalize_lot, can_partial, get_lot_spec, LOT_SPEC


class TestGetLotSpec:

    def test_forex_default(self):
        min_lot, step = get_lot_spec("EURUSD")
        assert min_lot == 0.01
        assert step == 0.01

    def test_index_cfd_min(self):
        min_lot, step = get_lot_spec("NAS100")
        assert min_lot == 0.10
        assert step == 0.10

    def test_ger40_step(self):
        _, step = get_lot_spec("GER40")
        assert step == 0.10

    def test_unknown_asset_returns_default(self):
        min_lot, step = get_lot_spec("UNKNOWN_ASSET")
        assert min_lot == 0.01
        assert step == 0.01


class TestNormalizeLot:

    # ── Forex (step 0.01) ────────────────────────────────────────────────────

    def test_forex_round_up(self):
        assert normalize_lot("EURUSD", 0.156) == 0.16

    def test_forex_round_down(self):
        assert normalize_lot("EURUSD", 0.154) == 0.15

    def test_forex_exact(self):
        assert normalize_lot("EURUSD", 0.15) == 0.15

    def test_forex_min_enforced(self):
        assert normalize_lot("EURUSD", 0.001) == 0.01

    def test_forex_zero_returns_zero(self):
        assert normalize_lot("EURUSD", 0.0) == 0.0

    def test_forex_none_returns_zero(self):
        assert normalize_lot("EURUSD", None) == 0.0

    # ── Index CFD (step 0.10) ────────────────────────────────────────────────

    def test_index_round_nearest_step(self):
        """0.29 should round to 0.30 (nearest multiple of 0.10)."""
        assert normalize_lot("DJ30", 0.29) == 0.30

    def test_index_round_down(self):
        """0.42 should round to 0.40."""
        assert normalize_lot("DJ30", 0.42) == 0.40

    def test_index_exactly_half_rounds_up(self):
        """0.25 → half-up → 0.30."""
        assert normalize_lot("DJ30", 0.25) == 0.30

    def test_index_min_enforced(self):
        """0.05 is below min 0.10 → raised to 0.10."""
        assert normalize_lot("NAS100", 0.05) == 0.10

    def test_index_exact_multiple(self):
        assert normalize_lot("GER40", 0.30) == 0.30

    def test_index_large_lot(self):
        assert normalize_lot("SP500", 1.25) == 1.30

    # ── Parametric ───────────────────────────────────────────────────────────

    @pytest.mark.parametrize("asset", list(LOT_SPEC.keys()))
    def test_result_is_valid_multiple(self, asset):
        """For all configured assets, result must be a valid multiple of step."""
        _, step = get_lot_spec(asset)
        step_c = round(step * 100)
        for raw in (0.07, 0.15, 0.23, 0.42, 1.0):
            result = normalize_lot(asset, raw)
            result_c = round(result * 100)
            assert result_c % step_c == 0, (
                f"{asset}: normalize_lot({raw}) = {result} — not a multiple of step={step}"
            )


class TestCanPartial:

    def test_forex_small_cannot_partial(self):
        """0.01 lot: partial would be 0.006 → below min → cannot partial."""
        assert can_partial("EURUSD", 0.01) is False

    def test_forex_large_can_partial(self):
        assert can_partial("EURUSD", 0.10) is True

    def test_index_minimum_cannot_partial(self):
        """0.10 lot (minimum for indices): runner would be 0.0 → cannot partial."""
        assert can_partial("NAS100", 0.10) is False

    def test_index_double_minimum_can_partial(self):
        """0.20 lot: partial=0.10, runner=0.10 → both >= min → can partial."""
        assert can_partial("DJ30", 0.20) is True

    def test_index_triple_minimum_can_partial(self):
        assert can_partial("GER40", 0.30) is True

    def test_zero_lot_cannot_partial(self):
        assert can_partial("EURUSD", 0.0) is False

    def test_none_lot_cannot_partial(self):
        assert can_partial("EURUSD", None) is False
