"""
utils/lot_spec.py
Lot rules per symbol (minimum volume + step) and normalization.

Why this exists: some instruments on STARTRADER do not accept arbitrary
volume sizes. Indices and oil CFDs (NAS100, GER40, US2000, DJ30, SP500,
USOUSD) have Volume Min = 0.10 and Volume Step = 0.10 — a lot like 0.42
or 0.29 is REJECTED by the broker (invalid volume) and the order will not
open. Forex/crypto/metals use 0.01/0.01 (default).

This module is the SINGLE SOURCE of these rules. It is imported by:
  - agents/risk_manager_agent.py  (sizing → entry)
  - orchestrator/trade_executor.py (safety net at order submission)
  - orchestrator/position_tracker.py (partial close)
  - orchestrator/scheduler.py (lot modification -50%/-25% from Telegram)

No internal project imports → no risk of circular imports.
"""

# (volume_min, volume_step) per asset. Key = AgenTrade asset name.
# Verified from MT4 STARTRADER contract specifications.
DEFAULT_LOT_SPEC: tuple[float, float] = (0.01, 0.01)  # forex / crypto / metals

LOT_SPEC: dict[str, tuple[float, float]] = {
    # Index CFDs + Oil CFD → min 0.10, step 0.10
    "NAS100": (0.10, 0.10),
    "GER40":  (0.10, 0.10),
    "US2000": (0.10, 0.10),
    "DJ30":   (0.10, 0.10),
    "SP500":  (0.10, 0.10),
    "USOUSD": (0.10, 0.10),
}


def get_lot_spec(asset: str) -> tuple[float, float]:
    """Return (volume_min, volume_step) for the asset (default 0.01/0.01)."""
    return LOT_SPEC.get(asset, DEFAULT_LOT_SPEC)


def normalize_lot(asset: str, lot: float) -> float:
    """
    Round `lot` to a VALID volume for the broker:
      - rounds to the nearest multiple of the step (half-up):
          DJ30  0.42 → 0.40 | 0.29 → 0.30 | 0.25 → 0.30
      - enforces minimum volume (if result is below minimum → minimum)

    Computation in integer centesimals to avoid floating point errors.

    WARNING: if the computed lot is below the minimum, it is raised to the
    minimum — effective risk may exceed the target. The caller should log this.
    """
    if lot is None or lot <= 0:
        return 0.0
    min_lot, step = get_lot_spec(asset)
    step_c = int(round(step * 100))      # 10 (indices/oil) | 1 (default)
    min_c  = int(round(min_lot * 100))   # 10 | 1
    lot_c  = int(round(lot * 100))
    if step_c <= 0:
        step_c = 1
    # nearest multiple of step (half-up)
    n = (lot_c + step_c // 2) // step_c
    norm_c = n * step_c
    if norm_c < min_c:
        norm_c = min_c
    return norm_c / 100.0


def can_partial(asset: str, position_lots: float, fraction: float = 0.60) -> bool:
    """
    Return True if the position can be partially closed while respecting
    the step/minimum: both the closed portion and the runner must be
    >= minimum volume, and the closed portion must be < full position.

    Examples for indices (min/step 0.10):
      0.10 → partial 0.10, runner 0.00 → NOT partial-closeable
      0.20 → partial 0.10, runner 0.10 → OK
      0.30 → partial 0.20, runner 0.10 → OK
    """
    if position_lots is None or position_lots <= 0:
        return False
    min_lot, _ = get_lot_spec(asset)
    partial = normalize_lot(asset, position_lots * fraction)
    runner  = round(position_lots - partial, 2)
    return (partial >= min_lot) and (runner >= min_lot) and (partial < position_lots)
