"""
tests/test_score_normalization.py
Tests for the score normalisation utility (utils/score_converter.py).

The _to_0_100 boundary is a core architectural decision (paper Section 3.4):
all agents return scores on different native ranges, and a single conversion
function normalises them before the weighted consensus is computed.
"""

import pytest
from utils.score_converter import to_score_0_100, agent_score, RAW_MAX


class TestToScore0100:

    def test_neutral_returns_50(self):
        assert to_score_0_100(0.0, "BUY",  20.0) == 50.0
        assert to_score_0_100(0.0, "SELL", 20.0) == 50.0

    def test_max_positive_buy_returns_100(self):
        assert to_score_0_100(20.0, "BUY", 20.0) == 100.0

    def test_max_negative_buy_returns_0(self):
        assert to_score_0_100(-20.0, "BUY", 20.0) == 0.0

    def test_max_negative_sell_returns_100(self):
        """Bearish raw score supports SELL → 100."""
        assert to_score_0_100(-20.0, "SELL", 20.0) == 100.0

    def test_max_positive_sell_returns_0(self):
        """Bullish raw score contradicts SELL → 0."""
        assert to_score_0_100(20.0, "SELL", 20.0) == 0.0

    def test_partial_bullish_buy(self):
        result = to_score_0_100(7.1, "BUY", 20.0)
        assert 67.0 <= result <= 68.0

    def test_partial_bullish_sell(self):
        """Same bullish raw score contradicts SELL → below 50."""
        result = to_score_0_100(7.1, "SELL", 20.0)
        assert 32.0 <= result <= 33.0

    def test_clamp_above_max(self):
        """Values above raw_max are clamped to 100."""
        assert to_score_0_100(999.0, "BUY", 20.0) == 100.0

    def test_clamp_below_min(self):
        assert to_score_0_100(-999.0, "BUY", 20.0) == 0.0

    def test_zero_raw_max_returns_50(self):
        assert to_score_0_100(5.0, "BUY", 0.0) == 50.0

    def test_symmetry_buy_sell(self):
        """BUY and SELL scores are symmetric around 50."""
        buy_score  = to_score_0_100(10.0, "BUY",  20.0)
        sell_score = to_score_0_100(10.0, "SELL", 20.0)
        assert abs((buy_score + sell_score) - 100.0) < 0.01

    def test_rounding_to_one_decimal(self):
        result = to_score_0_100(7.1, "BUY", 20.0)
        assert result == round(result, 1)


class TestAgentScore:

    def test_macro_agent_bullish_buy(self):
        score = agent_score("MacroAgent", 7.1, "BUY")
        assert 67.0 <= score <= 68.0

    def test_macro_agent_bullish_sell(self):
        score = agent_score("MacroAgent", 7.1, "SELL")
        assert 32.0 <= score <= 33.0

    def test_sentiment_agent_bearish_sell(self):
        """Bearish sentiment (-3.0) supports SELL → above 50."""
        score = agent_score("SentimentAgent", -3.0, "SELL")
        assert score > 50.0

    def test_correlations_agent_strong_bear_sell(self):
        score = agent_score("CorrelationsAgent", -8.0, "SELL")
        assert score > 85.0

    def test_correlations_agent_strong_bear_buy(self):
        """Strong bearish correlations veto BUY → below 15."""
        score = agent_score("CorrelationsAgent", -8.0, "BUY")
        assert score < 15.0

    def test_vol_profile_bullish(self):
        score = agent_score("VolProfileAgent", 4.0, "BUY")
        assert score == 75.0

    def test_unknown_agent_raises_key_error(self):
        with pytest.raises(KeyError):
            agent_score("UnknownAgent", 5.0, "BUY")

    def test_all_known_agents_present(self):
        known = {"MacroAgent", "SentimentAgent", "TechnicalAgent",
                 "CorrelationsAgent", "VolProfileAgent"}
        assert known == set(RAW_MAX.keys())

    @pytest.mark.parametrize("agent,raw_max", RAW_MAX.items())
    def test_max_raw_returns_100_buy(self, agent, raw_max):
        assert agent_score(agent, raw_max, "BUY") == 100.0

    @pytest.mark.parametrize("agent,raw_max", RAW_MAX.items())
    def test_min_raw_returns_0_buy(self, agent, raw_max):
        assert agent_score(agent, -raw_max, "BUY") == 0.0
