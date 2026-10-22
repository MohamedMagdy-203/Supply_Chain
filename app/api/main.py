from fastapi import FastAPI, Query

from app.api import policy_tmp as policy
from app.api.deps import (
    DEFAULT_MODEL,
    QUANTILE_COLS,
    available_models,
    get_predictions,
    get_row,
)
from app.api.schemas import (
    ForecastResponse,
    HealthResponse,
    InventoryResponse,
    SeriesResponse,
)

app = FastAPI(
    title="M5 Probabilistic Forecasting API",
    description="Quantile demand forecasts and Newsvendor inventory decisions.",
    version="0.1.0",
)

MODEL_PARAM = Query(
    default=None,
    pattern=r"^[A-Za-z0-9_\-]+$",
    description="Model name (file name under predictions/, without .parquet).",
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        default_model=DEFAULT_MODEL,
        available_models=available_models(),
    )


@app.get("/series", response_model=SeriesResponse)
def series(model: str | None = MODEL_PARAM):
    df = get_predictions(model)
    return SeriesResponse(
        model=model or DEFAULT_MODEL,
        series_ids=sorted(df["series_id"].unique().tolist()),
        origin_weeks=sorted(int(w) for w in df["origin_week"].unique()),
    )


@app.get("/forecast", response_model=ForecastResponse)
def forecast(
    series_id: str,
    origin_week: int = Query(ge=0),
    model: str | None = MODEL_PARAM,
):
    row = get_row(model, series_id, origin_week)
    return ForecastResponse(
        model=model or DEFAULT_MODEL,
        series_id=series_id,
        origin_week=origin_week,
        **{c: float(row[c]) for c in QUANTILE_COLS},
    )


@app.get("/inventory", response_model=InventoryResponse)
def inventory(
    series_id: str,
    origin_week: int = Query(ge=0),
    service_level: float | None = Query(
        default=None, gt=0, lt=1,
        description="Target tau. If omitted, tau = cu / (cu + co).",
    ),
    cu: float = Query(default=0.30, gt=0, description="Stock-out cost per unit"),
    co: float = Query(default=0.10, gt=0, description="Overstock cost per unit"),
    on_hand: float = Query(default=0.0, ge=0),
    model: str | None = MODEL_PARAM,
):
    row = get_row(model, series_id, origin_week)
    quantiles = {c: float(row[c]) for c in QUANTILE_COLS}

    tau = service_level if service_level is not None else policy.critical_ratio(cu, co)
    S = policy.order_up_to(quantiles, tau)

    return InventoryResponse(
        model=model or DEFAULT_MODEL,
        series_id=series_id,
        origin_week=origin_week,
        tau=round(tau, 4),
        cu=cu,
        co=co,
        on_hand=on_hand,
        order_up_to=round(S, 3),
        order_qty=round(policy.order_quantity(S, on_hand), 3),
    )