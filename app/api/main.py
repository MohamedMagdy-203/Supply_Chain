"""M5 Probabilistic Forecasting API – main application.

Endpoints
---------
GET /health          – liveness + available models
GET /series          – list series IDs and origin weeks for a model
GET /forecast        – quantile forecast for one series / week
GET /inventory       – Newsvendor order-up-to decision for one series / week
GET /history         – weekly actuals for one series (from weekly.parquet)
GET /models          – model comparison metrics (from model_metrics.csv)
GET /orders          – batch Newsvendor decisions for all series (one origin week)
"""
from __future__ import annotations

import math

from fastapi import FastAPI, Query

import src.policy as policy
from app.api.deps import (
    DEFAULT_MODEL,
    QUANTILE_COLS,
    available_models,
    get_history,
    get_inventory_stock,
    get_metrics,
    get_predictions,
    get_row,
)
from app.api.schemas import (
    ForecastResponse,
    HealthResponse,
    HistoryPoint,
    HistoryResponse,
    InventoryResponse,
    ModelMetric,
    ModelsResponse,
    OrderItem,
    OrdersResponse,
    SeriesResponse,
)

app = FastAPI(
    title="M5 Probabilistic Forecasting API",
    description=(
        "Quantile demand forecasts and Newsvendor inventory decisions "
        "built on the M5 dataset.\n\n"
        "**Limitations**: forecasts cover origin weeks 252–275 (test split only). "
        "Real-time inference for future weeks is not implemented. "
        "Inventory balances are resolved by product and origin week; "
        "a manual override is available when needed."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Shared query parameter objects
# ---------------------------------------------------------------------------
MODEL_PARAM = Query(
    default=None,
    pattern=r"^[A-Za-z0-9_\-]+$",
    description="Model name (parquet file stem under predictions/). Defaults to lgbm.",
)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    """API liveness check."""
    return HealthResponse(
        status="ok",
        default_model=DEFAULT_MODEL,
        available_models=available_models(),
    )


# ---------------------------------------------------------------------------
# /series
# ---------------------------------------------------------------------------
@app.get("/series", response_model=SeriesResponse, tags=["Forecast"])
def series(model: str | None = MODEL_PARAM):
    """List all series IDs and available origin weeks for a model."""
    df = get_predictions(model)
    return SeriesResponse(
        model=model or DEFAULT_MODEL,
        series_ids=sorted(df["series_id"].unique().tolist()),
        origin_weeks=sorted(int(w) for w in df["origin_week"].unique()),
    )


# ---------------------------------------------------------------------------
# /forecast
# ---------------------------------------------------------------------------
@app.get("/forecast", response_model=ForecastResponse, tags=["Forecast"])
def forecast(
    series_id: str,
    origin_week: int = Query(ge=0),
    model: str | None = MODEL_PARAM,
):
    """Return the quantile forecast for a single series / origin week."""
    row = get_row(model, series_id, origin_week)
    return ForecastResponse(
        model=model or DEFAULT_MODEL,
        series_id=series_id,
        origin_week=origin_week,
        **{c: float(row[c]) for c in QUANTILE_COLS},
    )


# ---------------------------------------------------------------------------
# /inventory
# ---------------------------------------------------------------------------
@app.get("/inventory", response_model=InventoryResponse, tags=["Inventory"])
def inventory(
    series_id: str,
    origin_week: int = Query(ge=0),
    service_level: float | None = Query(
        default=None,
        gt=0,
        lt=1,
        description="Target service level τ. If omitted, τ = cu / (cu + co).",
    ),
    cu: float | None = Query(
        default=None,
        gt=0,
        description="Optional product-cost override for testing.",
    ),
    co: float | None = Query(
        default=None,
        gt=0,
        description="Optional product-cost override for testing.",
    ),
    on_hand: float | None = Query(
        default=None,
        ge=0,
        description="Optional manual override. If omitted, use the warehouse balance for this product and week.",
    ),
    model: str | None = MODEL_PARAM,
):
    """Compute the Newsvendor order-up-to level S* for a single series."""
    row = get_row(model, series_id, origin_week)
    quantiles = {c: float(row[c]) for c in QUANTILE_COLS}
    stock = get_inventory_stock(series_id, origin_week)
    resolved_on_hand = float(stock["available_stock"]) if on_hand is None else on_hand

    product_cu = float(stock["unit_shortage_cost"])
    product_co = float(stock["unit_holding_cost"])
    resolved_cu = cu if cu is not None else product_cu
    resolved_co = co if co is not None else product_co
    tau = (
        service_level
        if service_level is not None
        else policy.critical_ratio(resolved_cu, resolved_co)
    )
    S = policy.order_up_to(quantiles, tau)

    return InventoryResponse(
        model=model or DEFAULT_MODEL,
        series_id=series_id,
        origin_week=origin_week,
        tau=round(tau, 4),
        cu=round(resolved_cu, 2),
        co=round(resolved_co, 2),
        on_hand=round(resolved_on_hand, 3),
        reserved_stock=round(float(stock["reserved_stock"]), 3),
        available_stock=round(float(stock["available_stock"]), 3),
        unit_price=round(float(stock["unit_price"]), 2),
        unit_holding_cost=round(product_co, 2),
        unit_shortage_cost=round(product_cu, 2),
        inventory_source="warehouse" if on_hand is None else "manual override",
        order_up_to=round(S, 3),
        order_qty=round(policy.order_quantity(S, resolved_on_hand), 3),
    )


# ---------------------------------------------------------------------------
# /history
# ---------------------------------------------------------------------------
@app.get("/history", response_model=HistoryResponse, tags=["History"])
def history(
    series_id: str,
    up_to_week: int | None = Query(
        default=None,
        ge=0,
        description="Only return weeks with week_idx <= this value.",
    ),
    n_weeks: int | None = Query(
        default=None,
        ge=1,
        le=300,
        description="Return only the last N weeks (applied after up_to_week filter).",
    ),
):
    """Return weekly sales history for a single series from weekly.parquet."""
    df = get_history(series_id, up_to_week=up_to_week, n_weeks=n_weeks)

    points = [
        HistoryPoint(
            week_idx=int(row.week_idx),
            week_start=str(row.week_start.date()),
            y=int(row.y),
            sell_price=float(row.sell_price) if not math.isnan(row.sell_price) else None,
            split=str(row.split),
        )
        for row in df.itertuples()
    ]

    return HistoryResponse(
        series_id=series_id,
        n_weeks=len(points),
        history=points,
    )


# ---------------------------------------------------------------------------
# /models
# ---------------------------------------------------------------------------
@app.get("/models", response_model=ModelsResponse, tags=["Meta"])
def models(
    split: str | None = Query(
        default=None,
        pattern=r"^(val|test)$",
        description="Filter by split: 'val' or 'test'. Returns both if omitted.",
    ),
):
    """Return model comparison metrics from model_metrics.csv."""
    df = get_metrics()
    if split is not None:
        df = df[df["split"] == split]

    metrics = []
    for row in df.itertuples():
        metrics.append(
            ModelMetric(
                model=str(row.model),
                split=str(row.split),
                mae=float(row.mae),
                rmse=float(row.rmse),
                r2=float(row.r2),
                wape=float(row.wape),
                mean_pinball=float(row.mean_pinball),
                pinball_q10=float(row.pinball_q10) if hasattr(row, "pinball_q10") else None,
                pinball_q25=float(row.pinball_q25) if hasattr(row, "pinball_q25") else None,
                pinball_q50=float(row.pinball_q50) if hasattr(row, "pinball_q50") else None,
                pinball_q75=float(row.pinball_q75) if hasattr(row, "pinball_q75") else None,
                pinball_q90=float(row.pinball_q90) if hasattr(row, "pinball_q90") else None,
                pinball_q95=float(row.pinball_q95) if hasattr(row, "pinball_q95") else None,
                pinball_q99=float(row.pinball_q99) if hasattr(row, "pinball_q99") else None,
            )
        )

    return ModelsResponse(
        default_model=DEFAULT_MODEL,
        available_models=available_models(),
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# /orders
# ---------------------------------------------------------------------------
@app.get("/orders", response_model=OrdersResponse, tags=["Inventory"])
def orders(
    origin_week: int = Query(
        ge=252,
        le=275,
        description=(
            "Origin week for the batch order run (252–275). "
            "NOTE: This is a replay of the test period; "
            "live forecasting for future weeks is not implemented."
        ),
    ),
    service_level: float | None = Query(
        default=None,
        gt=0,
        lt=1,
        description="Global service level τ. If omitted, τ = cu / (cu + co).",
    ),
    cu: float | None = Query(
        default=None,
        gt=0,
        description="Optional uniform cost override for testing.",
    ),
    co: float | None = Query(
        default=None,
        gt=0,
        description="Optional uniform cost override for testing.",
    ),
    on_hand: float | None = Query(
        default=None,
        ge=0,
        description=(
            "Optional uniform override. If omitted, use the warehouse balance for each product and week."
        ),
    ),
    model: str | None = MODEL_PARAM,
):
    """Batch Newsvendor decisions for **all** series at a given origin week.

    Returns a list of (series_id, order_up_to, order_qty, q50, tau) for every
    series that has a forecast at `origin_week`. Suitable for feeding the
    weekly procurement email or the Inventory tab of the UI.

    **Limitations**
    - Forecasts only exist for origin weeks 252–275 (M5 test split).
    - Inventory is resolved per product and origin week.
    - This is a historical replay, not a live prediction.
    """
    df = get_predictions(model)
    week_df = df[df["origin_week"] == origin_week]
    if week_df.empty:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail=f"No forecasts found for origin_week={origin_week}. "
                   f"Available range: {int(df['origin_week'].min())}–{int(df['origin_week'].max())}.",
        )

    resolved_model = model or DEFAULT_MODEL

    order_items: list[OrderItem] = []
    cost_pairs: list[tuple[float, float]] = []
    for row in week_df.itertuples():
        quantiles = {c: float(getattr(row, c)) for c in QUANTILE_COLS}
        stock = get_inventory_stock(str(row.series_id), origin_week)
        product_cu = float(stock["unit_shortage_cost"])
        product_co = float(stock["unit_holding_cost"])
        resolved_cu = cu if cu is not None else product_cu
        resolved_co = co if co is not None else product_co
        tau = (
            service_level
            if service_level is not None
            else policy.critical_ratio(resolved_cu, resolved_co)
        )
        S = policy.order_up_to(quantiles, tau)
        resolved_on_hand = float(stock["available_stock"]) if on_hand is None else on_hand
        qty = policy.order_quantity(S, resolved_on_hand)
        cost_pairs.append((resolved_cu, resolved_co))
        order_items.append(
            OrderItem(
                series_id=str(row.series_id),
                order_up_to=round(S, 3),
                order_qty=round(qty, 3),
                q50=round(quantiles["q50"], 3),
                tau=round(tau, 4),
                current_stock=round(float(stock["current_stock"]), 3),
                reserved_stock=round(float(stock["reserved_stock"]), 3),
                available_stock=round(float(stock["available_stock"]), 3),
                unit_price=round(float(stock["unit_price"]), 2),
                unit_holding_cost=round(product_co, 2),
                unit_shortage_cost=round(product_cu, 2),
            )
        )

    # Sort by descending order quantity (highest-priority items first)
    order_items.sort(key=lambda x: x.order_qty, reverse=True)

    return OrdersResponse(
        model=resolved_model,
        origin_week=origin_week,
        tau=round(
            service_level
            if service_level is not None
            else sum(
                policy.critical_ratio(shortage, holding)
                for shortage, holding in cost_pairs
            ) / len(cost_pairs),
            4,
        ),
        cu=round(sum(pair[0] for pair in cost_pairs) / len(cost_pairs), 2),
        co=round(sum(pair[1] for pair in cost_pairs) / len(cost_pairs), 2),
        cost_source="uniform override" if cu is not None or co is not None else "product-specific",
        on_hand_source="warehouse" if on_hand is None else "manual override",
        n_series=len(order_items),
        orders=order_items,
    )
