"""Newsvendor inventory policy.

Implements the critical-ratio (Newsvendor) ordering rule used throughout the
Supply-Chain forecasting project.

Public API
----------
critical_ratio(cu, co)      -> float
order_up_to(quantiles, tau) -> float
order_quantity(S, on_hand)  -> float
expected_cost(S, y, cu, co) -> float
"""
from __future__ import annotations

import numpy as np

# Quantile levels must stay in sync with QUANTILE_LEVELS in app/api/deps.py
_QUANTILE_LEVELS: dict[str, float] = {
    "q10": 0.10,
    "q25": 0.25,
    "q50": 0.50,
    "q75": 0.75,
    "q90": 0.90,
    "q95": 0.95,
    "q99": 0.99,
}
_QUANTILE_COLS: list[str] = list(_QUANTILE_LEVELS)


def critical_ratio(cu: float, co: float) -> float:
    """Return the Newsvendor critical ratio τ = cu / (cu + co).

    Parameters
    ----------
    cu : float
        Unit cost of under-stocking (lost-sale / back-order cost).
    co : float
        Unit cost of over-stocking (holding / obsolescence cost).

    Returns
    -------
    float
        Optimal service level τ in (0, 1).
    """
    if cu <= 0 or co <= 0:
        raise ValueError(f"cu and co must be > 0, got cu={cu}, co={co}")
    return cu / (cu + co)


def order_up_to(quantiles: dict[str, float], tau: float) -> float:
    """Interpolate the forecast CDF to find the order-up-to level S*.

    Uses linear interpolation across the provided quantile values.
    Levels outside [q10, q99] are clamped to the nearest available quantile.

    Parameters
    ----------
    quantiles : dict[str, float]
        Mapping of quantile column name -> value, e.g. {"q10": 2.0, ..., "q99": 45.0}.
    tau : float
        Target service level in (0, 1).

    Returns
    -------
    float
        Order-up-to level S*.
    """
    levels = [_QUANTILE_LEVELS[c] for c in _QUANTILE_COLS]
    values = [quantiles[c] for c in _QUANTILE_COLS]
    return float(np.interp(tau, levels, values))


def order_quantity(S: float, on_hand: float) -> float:
    """Return the quantity to order: max(0, S - on_hand).

    Parameters
    ----------
    S : float
        Order-up-to level from :func:`order_up_to`.
    on_hand : float
        Current on-hand inventory (>= 0).

    Returns
    -------
    float
        Units to order (0 when inventory already covers S).
    """
    return max(0.0, S - on_hand)


def expected_cost(S: float, y: float, cu: float, co: float) -> float:
    """Realised Newsvendor cost for a single period.

    Parameters
    ----------
    S : float
        Order-up-to level (units stocked).
    y : float
        Actual demand realised.
    cu, co : float
        Under- and over-stocking unit costs.

    Returns
    -------
    float
        cu * max(y - S, 0) + co * max(S - y, 0).
    """
    return cu * max(y - S, 0.0) + co * max(S - y, 0.0)

