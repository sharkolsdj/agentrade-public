# Decision: Broker-Asymmetric Position Lifecycle

## Context

AgenTrade routes trades to two brokers with fundamentally different execution capabilities:

- **MT4 CFD (STARTRADER)** — fractional lots, native partial close at the broker level, no contract expiry
- **IB Micro Futures (MES, MGC, MCL, 6E)** — whole contracts only, no fractional positions, quarterly rollover

## Decision

Model the lifecycle asymmetry explicitly rather than abstracting it away. Each instrument type has its own TP/SL structure produced by `RiskManagerAgent._calculate_tp_levels()`:

| Type | Lifecycle |
|---|---|
| `SINGLE_TP_PARTIAL` (MT4 CFD) | 60% closed at TP1, 40% runner holds to TP2. After partial close, SL moves to breakeven. |
| `SINGLE_TP` (IB Micro) | Single TP. At 50% of TP distance, SL moves to entry (breakeven). 1 contract — not fractional. |
| `SINGLE_TP_RANGE` (range-day mode) | Single exit at TP = 1.8×SL. No partial close, no runner. |

## Rationale

The alternative — a unified abstract lifecycle — would require awkward workarounds:

- MT4 "partial close" is a native broker operation sent as a separate order. IB has no equivalent.
- Faking partial close on IB by scaling down contract size fails because IB Micro contracts are indivisible (1 MES = 1 contract, not 0.6).
- Abstracting this difference hides a real operational distinction and makes debugging harder when trade behavior differs across brokers.

Explicit modeling makes the asymmetry visible in code: `instrument_type: "MT4_CFD" | "IB_MICRO"` drives branching in `_calculate_tp_levels()` and the position monitor.

## Consequences

- `RiskManagerAgent` produces different `tp_levels` dicts depending on `instrument_type`
- `PositionTracker` (not published) has separate lifecycle handlers per type
- `broker/base_broker.py` exposes `supports_partial_close` property to make the capability explicit at the interface level

## See Also

- `agents/risk_manager_agent.py` — `_calculate_tp_levels()`
- `broker/base_broker.py` — `supports_partial_close`
- Paper Section 3.1
