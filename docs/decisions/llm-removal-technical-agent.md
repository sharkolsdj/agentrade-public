# Decision: LLM Removal from TechnicalAgent and RiskManagerAgent

## Context

AgenTrade v2 used GPT-4o-mini in two agents:
- `TechnicalAgent` — to evaluate ICT/SMC pattern quality
- `RiskManagerAgent` — to make the APPROVE/BLOCK/MODIFY decision

Both were consuming ~$15/month each and adding 3-8 seconds of latency per workflow.

## Decision

Remove the LLM from both agents in v3. Make them fully deterministic.

**TechnicalAgent v3**: ICT scoring is computed by `strategy_rules.py` — a deterministic Python module that checks each pattern condition with boolean logic. Either the condition is met or it is not.

**RiskManagerAgent v3**: All decisions are rule-based:
- Maximum drawdown gate: `drawdown >= MAX_DRAWDOWN_PCT -> BLOCK`
- Minimum SL gate: `sl_pips < MIN_SL_PIPS[asset] -> BLOCK`
- Position limit gate: `open_count >= MAX_OPEN -> BLOCK`
- R:R minimum gate: `rr < 1.5 -> BLOCK`
- Position sizing formula: `lot = equity x risk_pct / (sl_pips x pip_value)`

None of these require contextual interpretation.

## Rationale

The LLM was approximating the same rules already written — inconsistently. Deterministic code is auditable, repeatable, fast, and free. The LLM is retained only where genuine ambiguity exists: MacroAgent (central bank tone) and StrategyRAGAgent (qualitative setup evaluation).

## Consequences

- TechnicalAgent latency: 8.2s -> 1.1s average
- RiskManagerAgent latency: 4.1s -> 0.05s
- LLM cost for these two agents reduced by about $30 per month

## See Also

- `agents/technical_agent.py`
- `agents/risk_manager_agent.py`
- Paper Section 3.5c
