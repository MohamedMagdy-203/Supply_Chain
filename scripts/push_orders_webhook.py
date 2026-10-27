"""Push weekly replenishment orders to a Make.com Custom Webhook.

Architecture & Limitations
--------------------------
1. Push Model: This script fetches replenishment orders from the local FastAPI
   backend (`/orders?origin_week={week}`) and pushes the aggregated summary
   and top order lines to Make via HTTP POST.
2. Historical Replay: Forecasts cover test origin weeks 252–275. Default is week 275.
3. Simulated On-Hand: Real on-hand stock data is unavailable; on_hand=0 (or simulated)
   is used to compute required order quantity (S* - on_hand).
4. Security: The webhook URL is loaded from the MAKE_WEBHOOK_URL environment variable
   or a local .env file. No real email addresses or secrets are hardcoded.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

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
        print(f"❌ Error querying API at {url}: {e}", file=sys.stderr)
        sys.exit(1)


def build_payload(orders_data: dict, top_n: int = 10) -> dict:
    orders = orders_data.get("orders", [])
    origin_week = orders_data.get("origin_week")
    total_qty = sum(o["order_qty"] for o in orders)
    top_items = orders[:top_n]

    # Render clean summary text for email body module in Make
    summary_lines = [
        f"Weekly Procurement Order Recommendation (Origin Week: {origin_week})",
        f"Generated: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Total Active Product Series: {len(orders)}",
        f"Total Warehouse Units to Replenish: {total_qty:,.1f}",
        f"Target Service Level (Critical Ratio): {orders_data.get('tau', 0.75):.1%}",
        "",
        f"Top {len(top_items)} Items by Replenishment Priority:",
        "----------------------------------------------------------------------",
        f"{'Series ID':<30} | {'Median Dem (q50)':<16} | {'Order Target S*':<16} | {'Order Qty':<12}",
        "----------------------------------------------------------------------",
    ]
    for itm in top_items:
        summary_lines.append(
            f"{itm['series_id']:<30} | {itm['q50']:<16.1f} | {itm['order_up_to']:<16.1f} | {itm['order_qty']:<12.1f}"
        )

    disclaimer = (
        "\n[System Limitations & Simulation Notice]\n"
        "1. This order was generated in historical replay mode (Week 275 of test set).\n"
        "2. Real-time on-hand stock is unavailable; calculations assume simulated inventory level.\n"
        "3. Automated by M5 Supply Chain Forecasting & Newsvendor Policy Pipeline."
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
        "email_subject": f"[Procurement] Automated Weekly Orders (Week {origin_week}) - {total_qty:,.0f} units",
        "email_body_text": "\n".join(summary_lines) + disclaimer,
        "orders_preview": orders[:50],
    }


def main():
    args = parse_args()
    print(f"📦 Fetching orders for origin_week={args.origin_week} from {args.api_url}...")
    data = fetch_orders(args.api_url, args.origin_week)
    payload = build_payload(data, top_n=args.top_n)

    print(f"✅ Prepared payload: {payload['total_series']} series, {payload['total_order_qty']} total units.")

    if args.dry_run or not args.webhook_url:
        print("\n=== DRY RUN / PREVIEW PAYLOAD ===")
        print(json.dumps(payload, indent=2))
        if not args.webhook_url:
            print("\n💡 NOTE: No webhook URL provided. Set MAKE_WEBHOOK_URL or pass --webhook-url to send live.")
        return

    print(f"🚀 Pushing payload to Make.com Webhook ({args.webhook_url})...")
    try:
        resp = requests.post(args.webhook_url, json=payload, timeout=15)
        print(f"📡 Response from Make: Status {resp.status_code} - {resp.text}")
        if resp.status_code == 200:
            print("🎉 Success! Webhook accepted the payload.")
        else:
            print("⚠️ Webhook responded with non-200 code.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to push to webhook: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

