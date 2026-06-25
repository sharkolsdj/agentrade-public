# Decision: Score Normalisation at the Boundary

## Context

Each agent produces a score on a different native range:

| Agent | Native Range |
|---|---|
| MacroAgent | -20 / +20 |
| SentimentAgent | -15 / +15 |
| TechnicalAgent | 0 / 16 pt |
| VolProfileAgent | quality points |
| CorrelationsAgent | -10 / +10 |

Feeding raw values into a weighted sum without normalisation produces incorrect results.

## Decision

A single `to_score_0_100()` function converts every agent's raw score to 0-100 before the weighted consensus is computed. The conversion is direction-aware:

- BUY: `score = 50 + (raw / raw_max) * 50`
- SELL: `score = 50 + (-raw / raw_max) * 50`

Boundary: 0 = strongly contradicts direction, 50 = neutral, 100 = strongly supports.

## Rationale

Converting at the boundary keeps agents independent of the consensus formula. Agents adjust internal scoring without touching normalisation; consensus never needs to know each agent's range. Fixed formula tied to `raw_max` is deterministic and requires no historical state.

## Consequences

- `utils/score_converter.py` is the single source of truth for normalisation
- Each agent calls `agent_score(agent_name, raw, direction)` as its last step

## See Also

- `utils/score_converter.py`
- `tests/test_score_normalization.py`
- Paper Section 3.4
