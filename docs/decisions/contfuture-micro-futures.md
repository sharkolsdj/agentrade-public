# Decision: ContFuture for IB Micro Futures (Automatic Rollover)

## Context

AgenTrade trades four IB Micro Futures: MES, MGC, MCL, 6E. A common production failure is hardcoding `conId` — the system works until quarterly rollover, then silently fails with "contract not found" errors.

## Decision

Use `ContFuture` from `ib_insync` instead of `Future` with a hardcoded `conId`:

```python
# WRONG — breaks every 3 months:
contract = Contract(conId=495512559)

# RIGHT — resolves to front-month at runtime:
contract = ContFuture("ES", exchange="CME")
await ib.qualifyContractsAsync(contract)
```

## Rationale

`ContFuture` resolves to the current front-month automatically when `qualifyContracts()` is called. No manual rollover procedure required. The `conId` bug that caused a critical incident in production (hardcoded MES conId expired) is permanently eliminated.

For data: fetch from ES (full E-mini) which has deeper liquidity and real volume. For execution: place orders on MES (micro, 1/10 the notional).

## Consequences

- No manual rollover procedure required
- `IBDataFeed` uses a dedicated clientId pool (96/95/94) separate from executor (11-14)

## See Also

- `broker/ib_connector.py` — `build_contract()`
- `broker/ib_datafeed.py` — `IBDataFeed`
- Paper Section 3.7
