"""Temporary Newsvendor logic.

Replace with `from src.policy import ...` once Member 1 delivers src/policy.py.
Keep the same function names/signatures to make the swap a one-line change.
"""
import numpy as np

from app.api.deps import QUANTILE_COLS, QUANTILE_LEVELS


def critical_ratio(cu: float, co: float) -> float:
    return cu / (cu + co)


def order_up_to(quantiles: dict[str, float], tau: float) -> float:
    """Interpolate the forecast quantile at level tau.

    Levels outside [0.10, 0.99] are clamped to the nearest available quantile.
    """
    levels = [QUANTILE_LEVELS[c] for c in QUANTILE_COLS]
    values = [quantiles[c] for c in QUANTILE_COLS]
    return float(np.interp(tau, levels, values))


def order_quantity(S: float, on_hand: float) -> float:
    return max(0.0, S - on_hand)


def expected_cost(S: float, y: float, cu: float, co: float) -> float:
    return cu * max(y - S, 0.0) + co * max(S - y, 0.0)