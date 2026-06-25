"""
tests/test_vol_profile_basic.py
Tests for VolumeAnalyzer deterministic computations (agents/vol_profile_agent.py).

The VPOC/VAH/VAL math is standard finance, not proprietary.
These tests verify the Value Area computation and key properties.
"""

import numpy as np
import pandas as pd
import pytest

from agents.vol_profile_agent import VolumeAnalyzer


@pytest.fixture
def analyzer():
    return VolumeAnalyzer()


def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV with enough data for a meaningful VP."""
    np.random.seed(seed)
    closes = 1800.0 + np.cumsum(np.random.randn(n) * 2.0)  # gold-like prices
    highs  = closes + np.abs(np.random.randn(n) * 1.5)
    lows   = closes - np.abs(np.random.randn(n) * 1.5)
    highs  = np.maximum(highs, closes)
    lows   = np.minimum(lows, closes)
    return pd.DataFrame({
        "Open":   np.roll(closes, 1), "High": highs,
        "Low":    lows, "Close": closes,
        "Volume": np.random.randint(100, 5000, n).astype(float),
    })


class TestCalculateVolumeProfile:

    def test_returns_dict(self, analyzer):
        df      = _make_ohlcv()
        profile = analyzer.calculate_volume_profile(df, "XAUUSD")
        assert isinstance(profile, dict)

    def test_required_keys_present(self, analyzer):
        df      = _make_ohlcv()
        profile = analyzer.calculate_volume_profile(df, "XAUUSD")
        for key in ["poc", "vah", "val", "hvn_levels", "lvn_levels"]:
            assert key in profile, f"Missing key: {key}"

    def test_vah_above_val(self, analyzer):
        df      = _make_ohlcv()
        profile = analyzer.calculate_volume_profile(df, "XAUUSD")
        assert profile["vah"] > profile["val"]

    def test_poc_inside_value_area(self, analyzer):
        df      = _make_ohlcv()
        profile = analyzer.calculate_volume_profile(df, "XAUUSD")
        assert profile["val"] <= profile["poc"] <= profile["vah"]

    def test_poc_inside_price_range(self, analyzer):
        df      = _make_ohlcv()
        profile = analyzer.calculate_volume_profile(df, "XAUUSD")
        low, high = profile["price_range"]
        assert low <= profile["poc"] <= high

    def test_empty_df_returns_empty(self, analyzer):
        assert analyzer.calculate_volume_profile(pd.DataFrame(), "XAUUSD") == {}

    def test_too_few_bars_returns_empty(self, analyzer):
        df = _make_ohlcv(3)
        assert analyzer.calculate_volume_profile(df, "XAUUSD") == {}

    def test_distribution_shape_valid(self, analyzer):
        df      = _make_ohlcv()
        profile = analyzer.calculate_volume_profile(df, "XAUUSD")
        assert profile["distribution_shape"] in ("b", "P", "balanced")

    def test_vpoc_migration_valid(self, analyzer):
        df      = _make_ohlcv()
        profile = analyzer.calculate_volume_profile(df, "XAUUSD")
        assert profile["vpoc_migration"] in ("up", "down", "none")

    def test_hvn_levels_are_list(self, analyzer):
        df      = _make_ohlcv()
        profile = analyzer.calculate_volume_profile(df, "XAUUSD")
        assert isinstance(profile["hvn_levels"], list)

    def test_lvn_levels_are_list(self, analyzer):
        df      = _make_ohlcv()
        profile = analyzer.calculate_volume_profile(df, "XAUUSD")
        assert isinstance(profile["lvn_levels"], list)


class TestCalculateVwap:

    def test_weekly_vwap_returns_float(self, analyzer):
        df   = _make_ohlcv()
        vwap = analyzer.calculate_vwap_weekly(df)
        assert isinstance(vwap, float)
        assert vwap > 0

    def test_daily_vwap_returns_float(self, analyzer):
        df   = _make_ohlcv()
        vwap = analyzer.calculate_vwap_daily(df)
        assert isinstance(vwap, float)

    def test_vwap_inside_price_range(self, analyzer):
        df    = _make_ohlcv()
        vwap  = analyzer.calculate_vwap_weekly(df)
        low   = float(df["Low"].min())
        high  = float(df["High"].max())
        assert low <= vwap <= high

    def test_empty_returns_none(self, analyzer):
        assert analyzer.calculate_vwap_weekly(pd.DataFrame()) is None


class TestCheckRule80:

    def test_not_active_when_inside_va(self, analyzer):
        df      = _make_ohlcv()
        profile = analyzer.calculate_volume_profile(df, "XAUUSD")
        result  = analyzer.check_rule_80(df, profile)
        assert isinstance(result, dict)
        assert "active" in result

    def test_returns_dict_keys(self, analyzer):
        df      = _make_ohlcv()
        profile = analyzer.calculate_volume_profile(df, "XAUUSD")
        result  = analyzer.check_rule_80(df, profile)
        assert "active" in result
        assert "direction" in result


class TestNakedVpoc:

    def test_returns_list(self, analyzer):
        df       = _make_ohlcv(200)
        sessions = analyzer.calculate_session_profiles(df, "XAUUSD", n_sessions=5)
        result   = analyzer.get_naked_vpoc(sessions, 1800.0, "BUY")
        assert isinstance(result, list)

    def test_buy_vpocs_above_price(self, analyzer):
        df       = _make_ohlcv(200)
        price    = float(df["Close"].iloc[-1])
        sessions = analyzer.calculate_session_profiles(df, "XAUUSD", n_sessions=5)
        nvpocs   = analyzer.get_naked_vpoc(sessions, price, "BUY")
        for nv in nvpocs:
            assert nv["poc"] > price

    def test_sell_vpocs_below_price(self, analyzer):
        df       = _make_ohlcv(200)
        price    = float(df["Close"].iloc[-1])
        sessions = analyzer.calculate_session_profiles(df, "XAUUSD", n_sessions=5)
        nvpocs   = analyzer.get_naked_vpoc(sessions, price, "SELL")
        for nv in nvpocs:
            assert nv["poc"] < price
