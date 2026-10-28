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

## Make.com procurement email

Run the webhook script after the API is available:

```powershell
python scripts\push_orders_webhook.py --origin-week 275
```

The webhook payload contains all products in `all_products` and separates them
into:

- `replenishment_required`: expected demand is greater than available stock.
- `covered_by_stock`: available stock covers expected demand.

Each product row includes expected demand, available stock, shortfall or
surplus, recommended order quantity, unit price, and deterministic
product-specific holding and shortage costs. The same product costs drive the
API service level and order recommendation, the Streamlit product profile, and
the Make email payload.

The payload also includes:

- `email_body_text`: plain-text fallback.
- `email_body_html`: formatted email with KPI cards and separate tables for
  replenishment-required and stock-covered products.
- `report_sheet_name`: the weekly tab name, for example `Week 275`.
- `report_rows`: all product rows in a flat structure for Google Sheets.
- `excel_file_base64`: the complete Excel workbook encoded for Make file
  handling, with `excel_filename` and `excel_mime_type`.

In the Make Gmail/Outlook module, enable HTML content and map
`email_body_html` to the message body. Use `email_body_text` only as a fallback
for email providers that do not support HTML. The HTML email shows the top
items (controlled by `--top-n`, default 10) and reports how many additional
products are not displayed in each table. Set `UI_URL` to include a dashboard
link in the email.

### Full Excel and Google Sheets report

The webhook now carries all products, not just the email preview. For an Excel
attachment, add a Make Tools/Base64 decode step before the email module and map
the decoded `excel_file_base64` as the attachment data. Use
`excel_filename` as the attachment name and `excel_mime_type` as its MIME type.

For Google Sheets, use this Make flow:

1. Add `Google Sheets > Add a Sheet` after the webhook and map
   `report_sheet_name` to the sheet name.
2. Add `Google Sheets > Add a Row` and select the created sheet.
3. Add an Iterator before `Add a Row` over `report_rows`.
4. Map the row fields to the columns in `report_columns`.

The sheet columns are:
`Product`, `Coverage Status`, `Expected Demand`, `Available Stock`,
`Shortfall`, `Surplus`, `Recommended Order`, `Unit Price`,
`Holding Cost per Unit`, `Shortage Cost per Unit`, `Estimated Excess Cost`,
and `Estimated Shortage Exposure`.

The iterator will add all `report_row_count` products to the weekly tab, where
they can be filtered and shared independently of the email. If the scenario
can run more than once for the same week, add a duplicate check or delete the
existing tab before creating it.

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
