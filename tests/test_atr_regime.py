"""
tests/test_atr_regime.py
Tests for regime detection and ATR computation (orchestrator/atr_regime.py).
"""

import numpy as np
import pandas as pd
import pytest

from orchestrator.atr_regime import (
    calculate_atr_ewm,
    get_atr_ratio,
    detect_mode,
    get_weights,
    get_size_multiplier,
    get_session_manual_threshold,
    should_skip_workflow,
    ATR_EXPLOSIVE_THRESHOLD,
    ATR_SKIP_THRESHOLD,
    WEIGHTS,
    SIZE_MULTIPLIER,
)


def _make_df(n: int = 50, volatility: float = 0.001) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame for testing."""
    np.random.seed(42)
    closes = 1.1000 + np.cumsum(np.random.randn(n) * volatility)
    highs  = closes + np.abs(np.random.randn(n) * volatility)
    lows   = closes - np.abs(np.random.randn(n) * volatility)
    opens  = np.roll(closes, 1)
    opens[0] = closes[0]
    return pd.DataFrame({
        "Open":   opens, "High": highs, "Low": lows,
        "Close":  closes, "Volume": np.random.randint(100, 10000, n),
    })


class TestCalculateAtrEwm:

    def test_returns_series(self):
        df  = _make_df()
        atr = calculate_atr_ewm(df)
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(df)

    def test_all_positive(self):
        df  = _make_df()
        atr = calculate_atr_ewm(df).dropna()
        assert (atr > 0).all()

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="high, low, close"):
            calculate_atr_ewm(df)

    def test_lowercase_columns(self):
        df  = _make_df()
        df.columns = [c.lower() for c in df.columns]
        atr = calculate_atr_ewm(df)
        assert not atr.empty

    def test_high_volatility_produces_larger_atr(self):
        low_vol  = calculate_atr_ewm(_make_df(volatility=0.0001)).iloc[-1]
        high_vol = calculate_atr_ewm(_make_df(volatility=0.01)).iloc[-1]
        assert high_vol > low_vol


class TestGetAtrRatio:

    def test_normal_regime(self):
        df   = _make_df(50)
        info = get_atr_ratio(df)
        assert info["regime"] in ("normal", "explosive")
        assert info["ratio"] > 0

    def test_insufficient_data_returns_fallback(self):
        df   = _make_df(5)
        info = get_atr_ratio(df)
        assert info["regime"] == "normal"
        assert info["ratio"] == 1.0

    def test_none_returns_fallback(self):
        info = get_atr_ratio(None)
        assert info["regime"] == "normal"

    def test_explosive_detected(self):
        """Artificially spike the last bar's range to trigger explosive regime."""
        df = _make_df(50, volatility=0.0001)
        df.loc[df.index[-1], "High"] = df["High"].max() * 5.0
        df.loc[df.index[-1], "Low"]  = df["Low"].min()  * 0.5
        info = get_atr_ratio(df)
        assert info["ratio"] > ATR_EXPLOSIVE_THRESHOLD
        assert info["regime"] == "explosive"


class TestDetectMode:

    def test_normal_atr_intraday(self):
        atr_info = {"ratio": 1.0, "regime": "normal"}
        mode = detect_mode(atr_info, "LONDON_OPEN")
        assert mode == "intraday"

    def test_explosive_atr_intraday_reduced(self):
        atr_info = {"ratio": 2.0, "regime": "explosive"}
        mode = detect_mode(atr_info, "LONDON_OPEN")
        assert mode == "intraday_reduced"

    def test_strong_daily_bias_swing(self):
        atr_info = {"ratio": 1.0, "regime": "normal"}
        mode = detect_mode(atr_info, "LONDON_OPEN",
                           daily_bias_direction="BUY", daily_bias_strength=3)
        assert mode == "swing"

    def test_weak_daily_bias_not_swing(self):
        atr_info = {"ratio": 1.0, "regime": "normal"}
        mode = detect_mode(atr_info, "LONDON_OPEN",
                           daily_bias_direction="BUY", daily_bias_strength=1)
        assert mode == "intraday"

    def test_strong_bias_wrong_session_not_swing(self):
        atr_info = {"ratio": 1.0, "regime": "normal"}
        mode = detect_mode(atr_info, "DEAD_ZONE",
                           daily_bias_direction="BUY", daily_bias_strength=3)
        assert mode == "intraday"

    def test_neutral_bias_not_swing(self):
        atr_info = {"ratio": 1.0, "regime": "normal"}
        mode = detect_mode(atr_info, "LONDON_OPEN",
                           daily_bias_direction="NEUTRAL", daily_bias_strength=3)
        assert mode == "intraday"


class TestGetWeights:

    @pytest.mark.parametrize("mode", list(WEIGHTS.keys()))
    def test_weights_sum_to_one(self, mode):
        w = get_weights(mode)
        total = sum(w.values())
        assert abs(total - 1.0) < 1e-9, f"Mode {mode}: weights sum to {total}"

    def test_all_weights_positive(self):
        for mode, w in WEIGHTS.items():
            assert all(v > 0 for v in w.values()), f"Mode {mode} has non-positive weight"

    def test_unknown_mode_returns_intraday(self):
        w = get_weights("unknown_mode")
        assert w == WEIGHTS["intraday"]

    def test_required_agents_present(self):
        required = {"macro", "sentiment", "volprofile", "technical", "correlations"}
        for mode, w in WEIGHTS.items():
            assert required == set(w.keys()), f"Mode {mode} missing agent weights"


class TestGetSizeMultiplier:

    def test_intraday_full_size(self):
        assert get_size_multiplier("intraday") == 1.0

    def test_intraday_reduced_half(self):
        assert get_size_multiplier("intraday_reduced") == 0.5

    def test_swing_full_size(self):
        assert get_size_multiplier("swing") == 1.0

    def test_unknown_returns_full(self):
        assert get_size_multiplier("unknown") == 1.0


class TestShouldSkipWorkflow:

    def test_low_atr_outside_kz_skip(self):
        atr_info = {"ratio": ATR_SKIP_THRESHOLD - 0.1, "regime": "normal"}
        assert should_skip_workflow(atr_info, kill_zone=False) is True

    def test_low_atr_inside_kz_no_skip(self):
        atr_info = {"ratio": ATR_SKIP_THRESHOLD - 0.1, "regime": "normal"}
        assert should_skip_workflow(atr_info, kill_zone=True) is False

    def test_normal_atr_no_skip(self):
        atr_info = {"ratio": 1.0, "regime": "normal"}
        assert should_skip_workflow(atr_info, kill_zone=False) is False


class TestSessionThreshold:

    def test_london_open_standard(self):
        assert get_session_manual_threshold("LONDON_OPEN") == 60.0

    def test_ny_open_standard(self):
        assert get_session_manual_threshold("NY_OPEN") == 60.0

    def test_dead_zone_higher(self):
        """Dead zone has a higher threshold than standard."""
        assert get_session_manual_threshold("DEAD_ZONE") > 60.0

    def test_unknown_session_default(self):
        assert get_session_manual_threshold("UNKNOWN_SESSION") == 60.0

    def test_all_thresholds_above_50(self):
        from orchestrator.atr_regime import SESSION_MANUAL_THRESHOLD
        for session, thr in SESSION_MANUAL_THRESHOLD.items():
            assert thr >= 50.0, f"Session {session} has threshold below 50"
