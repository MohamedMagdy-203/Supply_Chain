from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    default_model: str
    available_models: list[str]


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
    order_up_to: float = Field(description="S, the target stock level")
    order_qty: float = Field(description="max(0, S - on_hand)")


class SeriesResponse(BaseModel):
    model: str
    series_ids: list[str]
    origin_weeks: list[int]