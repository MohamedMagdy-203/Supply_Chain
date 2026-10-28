"""Pydantic response schemas for the M5 Forecasting API."""
from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Existing schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    default_model: str
    available_models: list[str]


class SeriesResponse(BaseModel):
    model: str
    series_ids: list[str]
    origin_weeks: list[int]


class ForecastResponse(BaseModel):
    model: str
    series_id: str
    origin_week: int
    q10: float
    q25: float
    q50: float
    q75: float
    q90: float
    q95: float
    q99: float


class InventoryResponse(BaseModel):
    model: str
    series_id: str
    origin_week: int
    tau: float = Field(description="Service level (critical ratio) used")
    cu: float
    co: float
    on_hand: float
    reserved_stock: float = 0
    available_stock: float = 0
    unit_price: float = Field(description="Product unit price assumption")
    unit_holding_cost: float = Field(description="Product-specific holding cost per unit")
    unit_shortage_cost: float = Field(description="Product-specific shortage cost per unit")
    inventory_source: str = "warehouse"
    order_up_to: float = Field(description="S, the target stock level")
    order_qty: float = Field(description="max(0, S - on_hand)")


# ---------------------------------------------------------------------------
# New: /history
# ---------------------------------------------------------------------------
class HistoryPoint(BaseModel):
    week_idx: int
    week_start: str = Field(description="ISO date of the Monday that opens the week")
    y: int = Field(description="Actual weekly sales units")
    sell_price: float | None = None
    split: str = Field(description="train / val / test")


class HistoryResponse(BaseModel):
    series_id: str
    n_weeks: int
    history: list[HistoryPoint]


# ---------------------------------------------------------------------------
# New: /models
# ---------------------------------------------------------------------------
class ModelMetric(BaseModel):
    model: str
    split: str
    mae: float
    rmse: float
    r2: float
    wape: float
    mean_pinball: float
    # per-quantile pinball scores
    pinball_q10: float | None = None
    pinball_q25: float | None = None
    pinball_q50: float | None = None
    pinball_q75: float | None = None
    pinball_q90: float | None = None
    pinball_q95: float | None = None
    pinball_q99: float | None = None


class ModelsResponse(BaseModel):
    default_model: str
    available_models: list[str]
    metrics: list[ModelMetric]


# ---------------------------------------------------------------------------
# New: /orders
# ---------------------------------------------------------------------------
class OrderItem(BaseModel):
    series_id: str
    order_up_to: float = Field(description="Newsvendor S* (target stock level)")
    order_qty: float = Field(description="max(0, S* - on_hand)")
    q50: float = Field(description="Median forecast demand")
    tau: float = Field(description="Service level used")
    current_stock: float = 0
    reserved_stock: float = 0
    available_stock: float = 0
    unit_price: float = Field(description="Product unit price assumption")
    unit_holding_cost: float = Field(description="Product-specific holding cost per unit")
    unit_shortage_cost: float = Field(description="Product-specific shortage cost per unit")


class OrdersResponse(BaseModel):
    model: str
    origin_week: int
    tau: float = Field(description="Average service level across products")
    cu: float = Field(description="Average shortage cost across products")
    co: float = Field(description="Average holding cost across products")
    cost_source: str = "product-specific"
    on_hand_source: str = Field(description="Source of the inventory balance used.")
    n_series: int
    orders: list[OrderItem]
