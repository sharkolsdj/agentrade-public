"""
utils/score_converter.py
Universal signed-score to 0-100 conversion function for AgenTrade.

Each agent internally produces a raw_score with sign:
  positive = bullish context
  negative = bearish context

This function converts the raw_score into a 0-100 score that answers:
"How confident am I that this trade in direction X is valid, from my
perspective?"

  score > 60 → context supports the direction
  score = 50 → neutral
  score < 40 → context contradicts the direction

Raw max per agent:
  MacroAgent:        20  (range -20/+20)
  SentimentAgent:    15  (range -15/+15)
  TechnicalAgent:    20  (range -20/+20)
  CorrelationsAgent: 10  (range -10/+10)
  VolProfileAgent:    8  (range -8/+8, deterministic)
  StrategyRAGAgent:  n/a (becomes flat bonus 0 or +5, does not use this function)
  RiskManagerAgent:  n/a (produces APPROVE/BLOCK/MODIFY, does not use this function)
"""


def to_score_0_100(raw: float, direction: str, raw_max: float) -> float:
    """
    Convert a signed raw score into a 0-100 score based on direction.

    Args:
        raw:       agent raw value (e.g. +7.1 or -3.6)
        direction: 'BUY' | 'SELL'
        raw_max:   absolute maximum of the raw score for that agent

    Returns:
        float 0-100 rounded to 1 decimal place

    Examples:
        MacroAgent raw=+7.1, direction=BUY
            → 50 + (7.1/20)*50 = 67.75  ✅ bullish macro supports BUY

        MacroAgent raw=+7.1, direction=SELL
            → 50 + (-7.1/20)*50 = 32.25  ✅ bullish macro does not support SELL

        SentimentAgent raw=-3.0, direction=SELL
            → 50 + (3.0/15)*50 = 60.0  ✅ bearish sentiment supports SELL

        CorrelationsAgent raw=-8.0, direction=SELL
            → 50 + (8.0/10)*50 = 90.0  ✅ strong bearish correlations support SELL

        Any agent raw=0.0, any direction
            → 50.0  ✅ neutral
    """
    if raw_max <= 0:
        return 50.0

    # Clamp to agent range
    clamped = max(-raw_max, min(raw_max, raw))

    if direction == "BUY":
        score = 50.0 + (clamped / raw_max) * 50.0
    else:  # SELL
        score = 50.0 + (-clamped / raw_max) * 50.0

    return round(score, 1)


# ─── Raw max per agent — importable by individual agents ──────────────────────

RAW_MAX = {
    "MacroAgent":        20.0,
    "SentimentAgent":    15.0,
    "TechnicalAgent":    20.0,
    "CorrelationsAgent": 10.0,
    "VolProfileAgent":    8.0,
}


def agent_score(agent_name: str, raw: float, direction: str) -> float:
    """
    Shortcut: convert raw score using the predefined raw_max for the agent.

    Args:
        agent_name: agent name (must be present in RAW_MAX)
        raw:        raw value
        direction:  'BUY' | 'SELL'

    Returns:
        float 0-100

    Raises:
        KeyError: if agent_name is not in RAW_MAX
    """
    raw_max = RAW_MAX[agent_name]
    return to_score_0_100(raw, direction, raw_max)


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  score_converter.py — Verification tests")
    print("=" * 60)

    test_cases = [
        # (agent, raw, direction, expected, description)
        ("MacroAgent",        +7.1,  "BUY",  67.8,  "bullish macro → supports BUY"),
        ("MacroAgent",        +7.1,  "SELL", 32.2,  "bullish macro → does not support SELL"),
        ("MacroAgent",        -7.1,  "SELL", 67.8,  "bearish macro → supports SELL"),
        ("MacroAgent",        -7.1,  "BUY",  32.2,  "bearish macro → does not support BUY"),
        ("MacroAgent",         0.0,  "BUY",  50.0,  "neutral → 50 for any direction"),
        ("MacroAgent",         0.0,  "SELL", 50.0,  "neutral → 50 for any direction"),
        ("MacroAgent",       +20.0,  "BUY",  100.0, "maximum bullish → 100"),
        ("MacroAgent",       -20.0,  "SELL", 100.0, "maximum bearish supports SELL → 100"),
        ("MacroAgent",       +20.0,  "SELL",  0.0,  "maximum bullish does not support SELL → 0"),
        ("SentimentAgent",    -3.0,  "SELL", 60.0,  "bearish sentiment supports SELL"),
        ("SentimentAgent",   +15.0,  "BUY",  100.0, "maximum bullish supports BUY"),
        ("CorrelationsAgent", -8.0,  "SELL", 90.0,  "strong bearish correlations"),
        ("CorrelationsAgent", -8.0,  "BUY",  10.0,  "bearish correlations → veto BUY"),
        ("TechnicalAgent",    +9.0,  "BUY",  72.5,  "ICT bullish supports BUY"),
        ("VolProfileAgent",   +4.0,  "BUY",  75.0,  "VP bullish supports BUY"),
        ("VolProfileAgent",   -4.0,  "SELL", 75.0,  "VP bearish supports SELL"),
    ]

    passed = 0
    failed = 0

    for agent, raw, direction, expected, desc in test_cases:
        result = agent_score(agent, raw, direction)
        ok = abs(result - expected) < 0.01
        icon = "✅" if ok else "❌"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {icon} {agent:<20} raw={raw:+6.1f} {direction:<5} "
              f"→ {result:6.2f}  (expected {expected:6.2f})  {desc}")

    print(f"\n{'─'*60}")
    print(f"  Result: {passed} passed, {failed} failed")

    if failed == 0:
        print("  ✅ All tests passed — score_converter.py ready")
    else:
        print("  ❌ Fix failing tests before proceeding")
    print("=" * 60 + "\n")
