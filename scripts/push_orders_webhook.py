"""Push weekly replenishment orders to a Make.com Custom Webhook.

Architecture & Limitations
--------------------------
1. Push Model: This script fetches replenishment orders from the local FastAPI
   backend (`/orders?origin_week={week}`) and pushes the aggregated summary
   and top order lines to Make via HTTP POST.
2. Historical Replay: Forecasts cover test origin weeks 252–275. Default is week 275.
3. Inventory: The API resolves current, reserved, and available stock by product and
   origin week from the warehouse balance file.
4. Security: The webhook URL is loaded from the MAKE_WEBHOOK_URL environment variable
   or a local .env file. No real email addresses or secrets are hardcoded.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import html
import io
import json
import os
import sys

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_BASE = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_WEBHOOK = os.getenv("MAKE_WEBHOOK_URL", "")


def parse_args():
    parser = argparse.ArgumentParser(description="Push procurement orders to Make.com Webhook")
    parser.add_argument(
        "--origin-week",
        type=int,
        default=275,
        help="Origin week for the forecast (252–275, default: 275)",
    )
    parser.add_argument(
        "--api-url",
        default=API_BASE,
        help=f"FastAPI base URL (default: {API_BASE})",
    )
    parser.add_argument(
        "--webhook-url",
        default=DEFAULT_WEBHOOK,
        help="Make.com Custom Webhook URL (reads from MAKE_WEBHOOK_URL env if omitted)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top items to highlight in the email body (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the payload to stdout instead of sending HTTP POST",
    )
    return parser.parse_args()


def fetch_orders(api_url: str, origin_week: int) -> dict:
    url = f"{api_url.rstrip('/')}/orders"
    params = {"origin_week": origin_week}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying API at {url}: {e}", file=sys.stderr)
        sys.exit(1)


def enrich_order(item: dict) -> dict:
    """Add business-facing coverage and deterministic product cost estimates."""
    expected_demand = float(item["q50"])
    available_stock = float(item["available_stock"])
    shortfall = max(expected_demand - available_stock, 0.0)
    surplus = max(available_stock - expected_demand, 0.0)
    holding_cost = float(item["unit_holding_cost"])
    shortage_cost = float(item["unit_shortage_cost"])
    return {
        **item,
        "expected_demand": round(expected_demand, 2),
        "available_stock": round(available_stock, 2),
        "shortfall": round(shortfall, 2),
        "surplus": round(surplus, 2),
        "unit_price": round(float(item["unit_price"]), 2),
        "unit_holding_cost": round(holding_cost, 2),
        "unit_shortage_cost": round(shortage_cost, 2),
        "estimated_excess_cost": round(surplus * holding_cost, 2),
        "estimated_shortage_exposure": round(shortfall * shortage_cost, 2),
        "coverage_status": (
            "Replenishment Required"
            if shortfall > 0
            else "Covered by Available Stock"
        ),
    }


def _html_table(title: str, items: list[dict], required: bool, limit: int = 10) -> str:
    visible_items = items[:limit]
    if required:
        columns = (
            "<th>Product</th><th>Expected</th><th>Available</th>"
            "<th>Shortfall</th><th>Recommended Order</th>"
        )
    else:
        columns = (
            "<th>Product</th><th>Expected</th><th>Available</th>"
            "<th>Surplus</th><th>Excess Cost</th>"
        )

    rows = []
    for item in visible_items:
        product = html.escape(str(item["series_id"]))
        if required:
            values = (
                f"<td>{product}</td><td>{item['expected_demand']:,.1f}</td>"
                f"<td>{item['available_stock']:,.1f}</td><td>{item['shortfall']:,.1f}</td>"
                f"<td><strong>{item['order_qty']:,.1f}</strong></td>"
            )
        else:
            values = (
                f"<td>{product}</td><td>{item['expected_demand']:,.1f}</td>"
                f"<td>{item['available_stock']:,.1f}</td><td>{item['surplus']:,.1f}</td>"
                f"<td>${item['estimated_excess_cost']:,.2f}</td>"
            )
        rows.append(f"<tr>{values}</tr>")

    remaining = len(items) - len(visible_items)
    remaining_note = (
        f'<p class="muted">Showing {len(visible_items)} of {len(items):,} products.</p>'
        if remaining > 0
        else ""
    )
    return (
        f'<h2>{html.escape(title)}</h2>'
        '<table><thead><tr>'
        f"{columns}"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + remaining_note
    )


def build_html_email(
    *,
    origin_week: int,
    model: str,
    service_level: float,
    total_series: int,
    total_order_qty: float,
    replenishment_required: list[dict],
    covered_by_stock: list[dict],
    top_n: int,
    dashboard_url: str = "",
) -> str:
    generated_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    dashboard_link = (
        f'<p><a class="button" href="{html.escape(dashboard_url, quote=True)}">'
        "Open Supply Chain Dashboard</a></p>"
        if dashboard_url
        else ""
    )
    return f"""<!doctype html>
<html>
<body style="margin:0;background:#f4f7fb;font-family:Arial,sans-serif;color:#172033;">
<div style="max-width:900px;margin:0 auto;padding:24px;">
  <div style="background:#1428a0;color:white;padding:24px;border-radius:12px 12px 0 0;">
    <div style="font-size:13px;letter-spacing:1px;text-transform:uppercase;">Supply Chain Control Tower</div>
    <h1 style="margin:8px 0 4px;font-size:26px;">Weekly Procurement Recommendation</h1>
    <div style="opacity:.9;">Origin week {origin_week} | Model {html.escape(model)} | Generated {generated_at}</div>
  </div>
  <div style="background:white;padding:24px;border-radius:0 0 12px 12px;">
    <div style="display:flex;gap:12px;flex-wrap:wrap;">
      <div class="kpi"><span>Products to replenish</span><strong>{len(replenishment_required):,}</strong></div>
      <div class="kpi"><span>Covered by stock</span><strong>{len(covered_by_stock):,}</strong></div>
      <div class="kpi"><span>Recommended purchase</span><strong>{total_order_qty:,.1f} units</strong></div>
      <div class="kpi"><span>Service level</span><strong>{service_level:.1%}</strong></div>
    </div>
    <p style="margin-top:22px;">The inventory review covers <strong>{total_series:,} product series</strong>.
    Products below expected demand are listed first for procurement action.</p>
    {_html_table("Products Requiring Replenishment", replenishment_required, True, top_n)}
    {_html_table("Products Covered by Available Stock", covered_by_stock, False, top_n)}
    {dashboard_link}
    <p class="muted">Planning context: forecasts are historical replay estimates for the available test weeks.
    Product costs are synthetic, product-specific planning assumptions.</p>
  </div>
</div>
<style>
  h2 {{ margin:28px 0 10px;color:#1428a0;font-size:19px; }}
  table {{ width:100%;border-collapse:collapse;font-size:13px; }}
  th {{ background:#e9efff;color:#1428a0;text-align:left;padding:10px; }}
  td {{ border-bottom:1px solid #e5e9f2;padding:10px; }}
  .kpi {{ flex:1 1 180px;background:#f0f4ff;border-left:4px solid #1428a0;padding:12px; }}
  .kpi span {{ display:block;color:#63708a;font-size:12px; }}
  .kpi strong {{ display:block;color:#1428a0;font-size:20px;margin-top:5px; }}
  .muted {{ color:#697386;font-size:12px;line-height:1.5; }}
  .button {{ display:inline-block;background:#1428a0;color:white!important;padding:11px 16px;
    border-radius:6px;text-decoration:none;font-weight:bold; }}
</style>
</body>
</html>"""


REPORT_COLUMNS = [
    "Product",
    "Coverage Status",
    "Expected Demand",
    "Available Stock",
    "Shortfall",
    "Surplus",
    "Recommended Order",
    "Unit Price",
    "Holding Cost per Unit",
    "Shortage Cost per Unit",
    "Estimated Excess Cost",
    "Estimated Shortage Exposure",
]


def build_report_rows(orders: list[dict]) -> list[dict]:
    return [
        {
            "Product": item["series_id"],
            "Coverage Status": item["coverage_status"],
            "Expected Demand": item["expected_demand"],
            "Available Stock": item["available_stock"],
            "Shortfall": item["shortfall"],
            "Surplus": item["surplus"],
            "Recommended Order": item["order_qty"],
            "Unit Price": item["unit_price"],
            "Holding Cost per Unit": item["unit_holding_cost"],
            "Shortage Cost per Unit": item["unit_shortage_cost"],
            "Estimated Excess Cost": item["estimated_excess_cost"],
            "Estimated Shortage Exposure": item["estimated_shortage_exposure"],
        }
        for item in orders
    ]


def build_excel_base64(report_rows: list[dict], sheet_name: str) -> str:
    report_df = pd.DataFrame(report_rows, columns=REPORT_COLUMNS)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        report_df.to_excel(writer, index=False, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for column_cells in worksheet.columns:
            width = min(
                max(len(str(cell.value or "")) for cell in column_cells) + 2,
                 34,
            )
            worksheet.column_dimensions[column_cells[0].column_letter].width = width
    return base64.b64encode(output.getvalue()).decode("ascii")


def build_payload(orders_data: dict, top_n: int = 10) -> dict:
    orders = [enrich_order(item) for item in orders_data.get("orders", [])]
    origin_week = orders_data.get("origin_week")
    total_qty = sum(o["order_qty"] for o in orders)
    replenishment_required = [
        item for item in orders if item["coverage_status"] == "Replenishment Required"
    ]
    covered_by_stock = [
        item for item in orders if item["coverage_status"] == "Covered by Available Stock"
    ]
    replenishment_required.sort(
        key=lambda item: (item["order_qty"], item["shortfall"]),
        reverse=True,
    )
    covered_by_stock.sort(key=lambda item: item["surplus"], reverse=True)
    top_items = replenishment_required[:top_n]
    dashboard_url = os.getenv("UI_URL", "")
    report_rows = build_report_rows(orders)
    report_filename = f"procurement_report_week_{origin_week}.xlsx"
    report_sheet_name = f"Week {origin_week}"

    # Render clean summary text for email body module in Make
    summary_lines = [
        f"Weekly Procurement Order Recommendation (Origin Week: {origin_week})",
        f"Generated: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Total Active Product Series: {len(orders)}",
        f"Products Requiring Replenishment: {len(replenishment_required):,}",
        f"Products Covered by Available Stock: {len(covered_by_stock):,}",
        f"Total Recommended Purchase: {total_qty:,.1f} units",
        f"Target Service Level (Critical Ratio): {orders_data.get('tau', 0.75):.1%}",
        "",
        f"Top {len(top_items)} Items Requiring Replenishment:",
        "----------------------------------------------------------------------",
        f"{'Series ID':<30} | {'Expected':<10} | {'Available':<10} | {'Shortfall':<10} | {'Order':<10}",
        "----------------------------------------------------------------------",
    ]
    for itm in top_items:
        summary_lines.append(
            f"{itm['series_id']:<30} | {itm['expected_demand']:<10.1f} | "
            f"{itm['available_stock']:<10.1f} | {itm['shortfall']:<10.1f} | "
            f"{itm['order_qty']:<10.1f}"
        )

    disclaimer = (
        "\n[Planning Context]\n"
        "Forecasts are generated in historical replay mode for the available test weeks. "
        "Cost estimates are product-specific planning assumptions."
    )

    return {
        "event": "procurement_order_ready",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "origin_week": origin_week,
        "model": orders_data.get("model", "lgbm"),
        "service_level": orders_data.get("tau", 0.75),
        "total_series": len(orders),
        "total_order_qty": round(total_qty, 2),
        "top_items": top_items,
        "replenishment_required": replenishment_required,
        "covered_by_stock": covered_by_stock,
        "all_products": orders,
        "report_columns": REPORT_COLUMNS,
        "report_rows": report_rows,
        "report_row_count": len(report_rows),
        "report_sheet_name": report_sheet_name,
        "excel_filename": report_filename,
        "excel_mime_type": (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        "excel_file_base64": build_excel_base64(report_rows, report_sheet_name),
        "email_subject": f"[Procurement] Automated Weekly Orders (Week {origin_week}) - {total_qty:,.0f} units",
        "email_body_text": "\n".join(summary_lines) + disclaimer,
        "email_body_html": build_html_email(
            origin_week=origin_week,
            model=orders_data.get("model", "lgbm"),
            service_level=orders_data.get("tau", 0.75),
            total_series=len(orders),
            total_order_qty=total_qty,
            replenishment_required=replenishment_required,
            covered_by_stock=covered_by_stock,
            top_n=top_n,
            dashboard_url=dashboard_url,
        ),
        "orders_preview": orders[:50],
    }


def main():
    args = parse_args()
    print(f"Fetching orders for origin_week={args.origin_week} from {args.api_url}...")
    data = fetch_orders(args.api_url, args.origin_week)
    payload = build_payload(data, top_n=args.top_n)

    print(f"Prepared payload: {payload['total_series']} series, {payload['total_order_qty']} total units.")

    if args.dry_run or not args.webhook_url:
        print("\n=== DRY RUN / PREVIEW PAYLOAD ===")
        print(json.dumps(payload, indent=2))
        if not args.webhook_url:
            print("\nNOTE: No webhook URL provided. Set MAKE_WEBHOOK_URL or pass --webhook-url to send live.")
        return

    print(f"Pushing payload to Make.com Webhook ({args.webhook_url})...")
    try:
        resp = requests.post(args.webhook_url, json=payload, timeout=15)
        print(f"Response from Make: Status {resp.status_code} - {resp.text}")
        if resp.status_code == 200:
            print("Success! Webhook accepted the payload.")
        else:
            print("Webhook responded with non-200 code.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to push to webhook: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
