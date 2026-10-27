# Deployment Guide

## Local URLs

After starting the project locally:

- Streamlit UI: http://localhost:8501
- FastAPI API: http://localhost:8000
- API health check: http://localhost:8000/health

The same UI URL is documented in `.env.example` as `UI_URL`.

## Docker Compose

From the repository root:

```powershell
docker compose up --build
```

Open the dashboard at <http://localhost:8501>.

To stop the services:

```powershell
docker compose down
```

The API container serves the prediction files, model metrics, weekly data, and
the per-product/per-week warehouse inventory file. The UI communicates with the
API through `API_URL=http://api:8000` inside the Compose network.

## Local development

Install the requirements, then start the services separately:

```powershell
uvicorn app.api.main:app --reload --port 8000
streamlit run app/ui/streamlit_app.py
```

For local Streamlit execution, the default API URL is
`http://localhost:8000`. Set `API_URL` when the API is hosted elsewhere.

## Environment configuration

Copy `.env.example` to `.env` and adjust values for the target environment.
Do not commit real webhook URLs or other secrets.

Important values include:

- `UI_URL`: browser URL used to open the Streamlit dashboard.
- `API_URL`: API base URL used by the UI.
- `DEFAULT_MODEL`: default forecast model, normally `lgbm`.
- `PREDICTIONS_DIR`: directory containing model prediction parquet files.
- `INVENTORY_PATH`: per-product/per-week warehouse inventory parquet file.
- `METRICS_PATH`: model benchmark CSV.

## Inventory and order decisions

The dashboard resolves inventory by product and origin week from the warehouse
balance file. It reports current stock, reserved stock, available stock, target
stock, and recommended order quantity.

The selected forecasting model changes demand estimates and order
recommendations, but the physical warehouse balance remains shared across
models. The API still supports an explicit `on_hand` override for testing.

## Current UI scope

The deployment is English-only and includes:

- Demand outlook and forecast ranges.
- Executive replenishment summary.
- Order recommendation and purchase priority.
- Service-level scenarios.
- Business explanation of demand, uncertainty, service target, and inventory.
- Validation and test model comparison.

SHAP feature-level charts are not displayed in the deployment. The `fake`
prediction model is not available in model discovery and its prediction file
has been removed.

## Validation

Run the API regression tests with:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
pytest tests/test_api.py -q
```

