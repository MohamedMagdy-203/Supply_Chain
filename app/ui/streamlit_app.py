"""Streamlit User Interface for M5 Probabilistic Forecasting & Inventory Policy.

Architecture:
- The UI never loads models or data files directly.
- All data and decisions are fetched from the FastAPI backend.
- 5 Tabs: Forecast, Inventory, Scenarios, Why, Models.
"""
from __future__ import annotations

import os
import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration & API client
# ---------------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
SAMSUNG_LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "samsung_logo.png")
TOGETHER_LOGO_PATH = os.path.join(
    os.path.dirname(__file__),
    "assets",
    "together_for_tomorrow_logo.png",
)

st.set_page_config(
    page_title="Supply Chain Decision Dashboard",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --brand-blue: #1428A0;
        --brand-blue-dark: #0B1B70;
        --ink: #172033;
        --muted: #64748B;
        --surface: #FFFFFF;
        --background: #F4F7FC;
        --border: #DCE3F0;
    }
    .stApp { background: var(--background); color: var(--ink); }
    .block-container {
        padding-top: 0.65rem;
        padding-bottom: 0.75rem;
    }
    h1 { margin: 0.25rem 0 0.1rem !important; font-size: 1.65rem !important; }
    h2, h3 { margin-top: 0.45rem !important; margin-bottom: 0.3rem !important; }
    .stCaption { margin-bottom: 0.2rem !important; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: var(--brand-blue); }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: #FFFFFF;
        border-color: rgba(255,255,255,.35);
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] p {
        color: #64748B !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"],
    [data-testid="stSidebar"] [data-testid="stMetricValue"] div {
        color: #0B1B70 !important;
    }
    .product-cost-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 10px 14px;
        margin: 4px 0 10px;
        box-shadow: 0 2px 8px rgba(20,40,160,.06);
    }
    .product-cost-card .cost-label {
        color: #64748B !important;
        font-size: .78rem;
        line-height: 1.2;
    }
    .product-cost-card .cost-value {
        color: #0B1B70 !important;
        font-size: 1.15rem;
        font-weight: 700;
        line-height: 1.25;
        margin-top: 3px;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        margin-top: 0.25rem !important;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #0F1E73 !important;
        border-color: rgba(255,255,255,.35) !important;
    }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.30); }
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 9px 12px;
        box-shadow: 0 2px 8px rgba(20,40,160,.06);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--brand-blue-dark); }
    div[data-testid="stTabs"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 5px 8px 0;
        margin: 8px 0 20px;
    }
    div[data-testid="stTabs"] [role="tab"] {
        color: #334155 !important;
        font-size: 0.92rem;
        font-weight: 600;
        min-height: 46px;
        padding: 0 16px;
        border-radius: 8px 8px 0 0;
    }
    div[data-testid="stTabs"] [role="tab"]:hover {
        color: var(--brand-blue) !important;
        background: #EEF2FF;
    }
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--brand-blue) !important;
        background: #EEF2FF;
        border-bottom: 3px solid var(--brand-blue);
    }
    div[data-testid="stTabs"] [role="tab"] p {
        color: #334155 !important;
        font-weight: 600;
    }
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] p {
        color: var(--brand-blue) !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: var(--brand-blue);
        height: 3px;
    }
    .section-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 14px 16px;
        height: 100%;
    }
    .section-card .label {
        color: var(--muted);
        font-size: .82rem;
        margin-bottom: 4px;
    }
    .section-card .value {
        color: var(--brand-blue-dark);
        font-size: 1.25rem;
        font-weight: 700;
    }
    .business-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        overflow: hidden;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: #FFFFFF;
        color: var(--ink);
        font-size: .9rem;
    }
    .business-table th {
        background: var(--brand-blue);
        color: #FFFFFF;
        text-align: left;
        font-weight: 600;
        padding: 11px 14px;
    }
    .business-table td {
        border-top: 1px solid #E8EDF5;
        padding: 10px 14px;
        color: var(--ink);
    }
    .business-table tr:nth-child(even) td { background: #F7F9FD; }
    .business-table .best-row td {
        background: #EAF0FF;
        color: var(--brand-blue-dark);
        font-weight: 700;
    }
    .business-note {
        background: #EAF0FF;
        border-left: 4px solid var(--brand-blue);
        border-radius: 8px;
        color: var(--ink);
        padding: 8px 12px;
        margin: 6px 0;
    }
    .stDownloadButton > button {
        background: var(--brand-blue) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--brand-blue) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
    }
    .stDownloadButton > button:hover {
        background: var(--brand-blue-dark) !important;
        color: #FFFFFF !important;
        border-color: var(--brand-blue-dark) !important;
    }
    .stDownloadButton > button p { color: #FFFFFF !important; }
    .decision-card {
        background: var(--brand-blue);
        color: #FFFFFF;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 8px 0 18px;
        box-shadow: 0 5px 16px rgba(20,40,160,.18);
    }
    .decision-card h3 { color: #FFFFFF; margin: 0 0 6px; }
    .decision-card p { margin: 0; color: #E8ECFF; }
        .brand-bar {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: clamp(120px, 24vw, 420px);
            height: 68px;
            padding: 8px 20px;
            width: calc(100% - 64px);
            margin: -0.65rem 64px 1rem 0;
            background: #2040B3;
            border-radius: 0 0 10px 10px;
            box-shadow: 0 3px 12px rgba(20,40,160,.14);
            box-sizing: border-box;
        }
        .brand-bar img {
            display: block;
            height: 48px;
            width: auto;
            object-fit: contain;
        }
        .brand-bar .program-logo { height: 52px; }
        </style>
    """,
    unsafe_allow_html=True,
)


def format_units(value: float) -> str:
    return f"{value:,.0f} units"


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def split_series(series_id: str) -> tuple[str, str, str]:
    store, item = series_id.split("__", maxsplit=1)
    parts = item.split("_", maxsplit=2)
    department = parts[0] if parts else item
    product = parts[2] if len(parts) > 2 else item
    return store, department, product


def build_order_analytics(orders: list[dict]) -> pd.DataFrame:
    data = pd.DataFrame(orders)
    if data.empty:
        return data
    data[["store", "department", "product"]] = data["series_id"].apply(
        lambda value: pd.Series(split_series(value))
    )
    data["coverage_ratio"] = data["available_stock"] / data["q50"].replace(0, np.nan)
    data["uncertainty_ratio"] = (
        data["order_up_to"] - data["q50"]
    ).clip(lower=0) / data["q50"].replace(0, np.nan)
    data["priority"] = np.select(
        [
            data["available_stock"] < data["q50"] * 0.5,
            data["available_stock"] < data["q50"],
            data["order_qty"] > 0,
        ],
        ["Critical", "High", "Medium"],
        default="Covered",
    )
    return data


@st.cache_data(ttl=60)
def api_get(endpoint: str, params: dict | None = None) -> dict | list | None:
    """Helper to query the API with caching."""
    url = f"{API_URL}{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        st.error(f"API error ({resp.status_code}) on {endpoint}: {resp.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to FastAPI at {API_URL}. Is the API server running?")
        return None
    except Exception as e:
        st.error(f"Unexpected error calling {endpoint}: {e}")
        return None


# ---------------------------------------------------------------------------
# Header & API Status
# ---------------------------------------------------------------------------
def image_data_uri(path: str) -> str:
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


if os.path.exists(SAMSUNG_LOGO_PATH) and os.path.exists(TOGETHER_LOGO_PATH):
    st.markdown(
        f"""
        <div class="brand-bar">
            <img src="{image_data_uri(SAMSUNG_LOGO_PATH)}" alt="Samsung">
            <img class="program-logo" src="{image_data_uri(TOGETHER_LOGO_PATH)}"
                 alt="Together for Tomorrow: Enabling People">
        </div>
        """,
        unsafe_allow_html=True,
    )
st.title("Demand Forecasting & Inventory Decision Dashboard")
st.caption("Turning demand uncertainty into clear, cost-aware procurement decisions")

# Check Health
health = api_get("/health")
if not health:
    st.warning("The API is unavailable. Start FastAPI with: `uvicorn app.api.main:app --port 8000`")
    st.stop()

available_models = health.get("available_models", ["lgbm"])
default_model = health.get("default_model", "lgbm")

# ---------------------------------------------------------------------------
# Sidebar: Controls
# ---------------------------------------------------------------------------
st.sidebar.header("Business Controls")

# Model selection
selected_model = st.sidebar.selectbox(
    "Forecasting Model",
    options=available_models,
    index=available_models.index(default_model) if default_model in available_models else 0,
)

# Fetch series and origin weeks for selected model
series_meta = api_get("/series", params={"model": selected_model})
if not series_meta:
    st.stop()

series_ids = series_meta.get("series_ids", [])
origin_weeks = series_meta.get("origin_weeks", [])

if not series_ids or not origin_weeks:
    st.error("The API returned no products or weeks.")
    st.stop()

# Series selector
selected_series = st.sidebar.selectbox(
    "Select Product",
    options=series_ids,
    index=0,
    help="Select the store, department, and product series used for the decision.",
)
selected_store, selected_department, selected_product = split_series(selected_series)
st.sidebar.caption(f"Store: {selected_store}")
st.sidebar.caption(f"Department: {selected_department}")
st.sidebar.caption(f"Product: {selected_product}")

product_cost_data = api_get("/inventory", params={
    "series_id": selected_series,
    "origin_week": origin_weeks[-1],
    "model": selected_model,
})
if product_cost_data:
    calculated_tau = product_cost_data["tau"]
else:
    calculated_tau = 0.75

# Origin week selector (Replay of test split)
selected_week = st.sidebar.selectbox(
    "Last Actual Week",
    options=origin_weeks,
    index=len(origin_weeks) - 1,
    help="The forecast and order recommendation apply to the following week.",
)

custom_tau = st.sidebar.checkbox("Set a Custom Service Level", value=False)
if custom_tau:
    tau = st.sidebar.slider(
        "Custom Service Level",
        min_value=0.05,
        max_value=0.99,
        value=calculated_tau,
        step=0.01,
    )
else:
    tau = calculated_tau
    st.sidebar.info(f"Recommended service level: {calculated_tau:.2%}")

st.sidebar.info("Inventory is loaded by product and week from the warehouse balance.")

# Portfolio analytics are shared by the executive summary and procurement plan.
portfolio_orders_data = api_get("/orders", params={
    "origin_week": selected_week,
    "model": selected_model,
    **({"service_level": tau} if custom_tau else {}),
})

# ---------------------------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------------------------
tab_forecast, tab_inventory, tab_scenarios, tab_why, tab_models = st.tabs([
    "Demand Outlook",
    "Order Recommendation",
    "Service Scenarios",
    "Business Explanation",
    "Model Performance",
])

# ===========================================================================
# Tab 1: Forecast
# ===========================================================================
with tab_forecast:
    # Fetch Forecast & History
    fc_data = api_get("/forecast", params={
        "series_id": selected_series,
        "origin_week": selected_week,
        "model": selected_model,
    })
    hist_data = api_get("/history", params={
        "series_id": selected_series,
        "up_to_week": selected_week,
        "n_weeks": 30,
    })

    if fc_data and hist_data:
        points = hist_data.get("history", [])
        hist_df = pd.DataFrame(points)
        analytics_df = build_order_analytics(
            portfolio_orders_data.get("orders", []) if portfolio_orders_data else []
        )
        if not analytics_df.empty:
            st.markdown("### Executive Summary")
            products_requiring_order = int((analytics_df["order_qty"] > 0).sum())
            critical_products = int((analytics_df["priority"] == "Critical").sum())
            excess_products = int(
                (analytics_df["available_stock"] > analytics_df["order_up_to"]).sum()
            )
            total_recommended_order = float(analytics_df["order_qty"].sum())
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            summary_col1.metric("Products Requiring Replenishment", f"{products_requiring_order:,}")
            summary_col2.metric("Total Recommended Purchase", format_units(total_recommended_order))
            summary_col3.metric("Critical Stockout Risk", f"{critical_products:,}")
            summary_col4.metric("Products Above Target Stock", f"{excess_products:,}")
            st.markdown(
                f'<div class="business-note"><strong>Executive summary:</strong> '
                f'For week {selected_week + 1}, the portfolio requires replenishment for '
                f'{products_requiring_order:,} of {len(analytics_df):,} products. '
                f'The highest priority is the {critical_products:,} products with available stock '
                'below half of expected demand.</div>',
                unsafe_allow_html=True,
            )

        if product_cost_data:
            st.markdown("### Product Cost Profile")
            cost_col1, cost_col2, cost_col3 = st.columns(3)
            cost_cards = (
                ("Unit Price", "unit_price"),
                ("Holding Cost per Unit", "unit_holding_cost"),
                ("Shortage Cost per Unit", "unit_shortage_cost"),
            )
            for column, (label, key) in zip(
                (cost_col1, cost_col2, cost_col3),
                cost_cards,
            ):
                with column:
                    st.markdown(
                        f'<div class="product-cost-card">'
                        f'<div class="cost-label">{label}</div>'
                        f'<div class="cost-value">${product_cost_data[key]:.2f}</div>'
                        "</div>",
                        unsafe_allow_html=True,
                    )

        st.subheader(f"Demand Outlook: {selected_store} | {selected_department} | {selected_product}")
        st.caption(f"Store: {selected_store}  |  Department: {selected_department}  |  Product: {selected_product}")

        forecast_col1, forecast_col2, forecast_col3, forecast_col4 = st.columns(4)
        forecast_col1.metric("Expected Demand", format_units(fc_data["q50"]))
        forecast_col1.caption("Central estimate for next week.")
        forecast_col2.metric(
            "Expected Demand Range",
            f"[{fc_data['q10']:.0f}, {fc_data['q90']:.0f}] units",
        )
        forecast_col2.caption("Likely demand range for planning.")
        forecast_col3.metric(
            "Typical Demand Range",
            f"[{fc_data['q25']:.0f}, {fc_data['q75']:.0f}] units",
        )
        forecast_col3.caption("Most common demand range.")
        forecast_col4.metric("High-Demand Planning Case", format_units(fc_data["q99"]))
        forecast_col4.caption("Conservative capacity planning case.")

        st.markdown("#### Demand Trend and Forecast Range")
        fig, ax = plt.subplots(figsize=(11, 3.65))

        if not hist_df.empty:
            ax.plot(
                hist_df["week_idx"],
                hist_df["y"],
                label="Actual Demand",
                color="#3155B7",
                marker="o",
                markersize=3,
            )

        # Target week
        target_week = selected_week + 1
        ax.axvline(x=selected_week, color="#64748B", linestyle="--", alpha=0.8,
                   label=f"Last Actual Week ({selected_week})")

        # Uncertainty Fan
        ax.fill_between(
            [selected_week, target_week],
            [hist_df["y"].iloc[-1] if not hist_df.empty else fc_data["q50"], fc_data["q10"]],
            [hist_df["y"].iloc[-1] if not hist_df.empty else fc_data["q50"], fc_data["q90"]],
            color="#9FB6FF",
            alpha=0.25,
            label="Expected Range",
        )
        ax.fill_between(
            [selected_week, target_week],
            [hist_df["y"].iloc[-1] if not hist_df.empty else fc_data["q50"], fc_data["q25"]],
            [hist_df["y"].iloc[-1] if not hist_df.empty else fc_data["q50"], fc_data["q75"]],
            color="#3155B7",
            alpha=0.24,
            label="Typical Range",
        )
        ax.plot([selected_week, target_week],
                [hist_df["y"].iloc[-1] if not hist_df.empty else fc_data["q50"], fc_data["q50"]],
                color="#1428A0",
                linewidth=2.5,
                marker="s",
                label=f"Expected Demand ({fc_data['q50']:.0f} units)")

        ax.set_title(f"Demand Outlook for Week {target_week}", color="#172033", pad=14)
        ax.set_xlabel("Week")
        ax.set_ylabel("Demand (units)")
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, linestyle="-", alpha=0.12, color="#64748B")
        ax.spines[["top", "right"]].set_visible(False)
        fig.subplots_adjust(bottom=0.22, top=0.9)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=3,
            frameon=False,
            borderaxespad=0,
            columnspacing=1.5,
            handlelength=2.2,
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ===========================================================================
# Tab 2: Inventory Policy
# ===========================================================================
with tab_inventory:
    st.subheader("Order Recommendation")
    st.caption(
        "The warehouse balance is shared across models. Changing the forecast model changes demand estimates, "
        "not the physical stock position."
    )

    inv_params = {
        "series_id": selected_series,
        "origin_week": selected_week,
        "model": selected_model,
    }
    if custom_tau:
        inv_params["service_level"] = tau

    inv_data = api_get("/inventory", params=inv_params)

    if inv_data:
        order_qty = inv_data["order_qty"]
        decision_title = "Order recommended" if order_qty > 0 else "No order required"
        decision_message = (
            f"Procure {format_units(order_qty)} to reach the target stock level."
            if order_qty > 0 else "Current stock meets the target stock level."
        )
        st.markdown(
            f'<div class="decision-card"><h3>{decision_title}</h3><p>{decision_message}</p></div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Service Target", f"{inv_data['tau']:.0%}")
        c2.metric("Target Stock", format_units(inv_data["order_up_to"]))
        c3.metric("Available Stock", format_units(inv_data["available_stock"]))
        c4.metric("Recommended Order", format_units(order_qty))
        st.caption(
            f"Warehouse balance: {format_units(inv_data['on_hand'])} current stock "
            f"less {format_units(inv_data['reserved_stock'])} reserved stock."
        )
        st.info("The recommendation balances shortage risk, excess inventory, and the selected service target.")

    st.markdown("---")
    st.subheader("Weekly Procurement Plan")
    st.caption(f"Portfolio-level order recommendations for the week after {selected_week}.")

    orders_data = portfolio_orders_data

    if orders_data:
        orders_list = orders_data.get("orders", [])
        analytics_df = build_order_analytics(orders_list)
        total_order_qty = float(analytics_df["order_qty"].sum())
        total_series = len(analytics_df)

        col_b1, col_b2, col_b3 = st.columns(3)
        col_b1.metric("Products Processed", f"{total_series:,}")
        col_b2.metric("Total Recommended Order", format_units(total_order_qty))
        source = orders_data.get("on_hand_source", "warehouse")
        col_b3.metric("Stock Data", "Warehouse" if source == "warehouse" else source)

        display_orders = analytics_df[
            [
                "store",
                "department",
                "product",
                "priority",
                "q50",
                "available_stock",
                "order_up_to",
                "order_qty",
            ]
        ].head(10).copy()
        display_orders = display_orders.rename(columns={
            "q50": "expected_demand",
            "order_up_to": "target_stock",
            "order_qty": "recommended_order",
        })
        display_orders["target_stock"] = display_orders["target_stock"].round(0).astype(int)
        display_orders["recommended_order"] = display_orders["recommended_order"].round(0).astype(int)
        display_orders["expected_demand"] = display_orders["expected_demand"].round(0).astype(int)
        display_orders["available_stock"] = display_orders["available_stock"].round(0).astype(int)
        display_orders.columns = [
            "Store", "Department", "Product", "Priority", "Expected Demand",
            "Available Stock", "Target Stock", "Recommended Order",
        ]
        st.markdown(display_orders.to_html(
            index=False, classes="business-table", border=0, justify="left"
        ), unsafe_allow_html=True)
        st.caption("Showing the 10 highest recommended orders. Download the CSV for the full procurement plan.")
        st.markdown("### Purchase Priority and Risk")
        priority_order = ["Critical", "High", "Medium", "Covered"]
        priority_counts = analytics_df["priority"].value_counts().reindex(priority_order, fill_value=0)
        priority_col1, priority_col2 = st.columns([1.3, 1])
        with priority_col1:
            fig_priority, ax_priority = plt.subplots(figsize=(7, 2.8))
            bars = ax_priority.bar(
                priority_counts.index,
                priority_counts.values,
                color=["#B42318", "#D97706", "#3155B7", "#94A3B8"],
            )
            ax_priority.set_ylabel("Products")
            ax_priority.set_title("Portfolio Risk Distribution", loc="left", color="#172033", pad=12)
            ax_priority.set_facecolor("#FFFFFF")
            ax_priority.grid(axis="y", linestyle="-", alpha=0.12, color="#64748B")
            ax_priority.spines[["top", "right", "left"]].set_visible(False)
            ax_priority.tick_params(axis="y", length=0)
            for bar in bars:
                ax_priority.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(priority_counts.max(), 1) * 0.02,
                    f"{int(bar.get_height()):,}",
                    ha="center",
                    fontsize=9,
                )
            st.pyplot(fig_priority)
            plt.close(fig_priority)
        with priority_col2:
            st.markdown(
                '<div class="business-note"><strong>How to read this:</strong><br>'
                '<strong>Critical:</strong> available stock is below half of expected demand.<br>'
                '<strong>High:</strong> available stock is below expected demand.<br>'
                '<strong>Medium:</strong> replenishment is recommended to reach target stock.<br>'
                '<strong>Covered:</strong> current availability meets the target.</div>',
                unsafe_allow_html=True,
            )

        csv = analytics_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Order CSV",
            data=csv,
            file_name=f"weekly_orders_week_{selected_week}.csv",
            mime="text/csv",
        )

# ===========================================================================
# Tab 3: Scenarios & Sensitivity
# ===========================================================================
with tab_scenarios:
    st.subheader("Service Target Scenarios")
    st.caption("See how the recommended stock changes as the required probability of meeting demand increases.")

    if fc_data:
        tau_levels = np.linspace(0.10, 0.99, 50)
        # Quantile interpolation
        q_cols = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        q_vals = [fc_data[f"q{int(l*100):02d}"] for l in q_cols]

        S_curve = np.interp(tau_levels, q_cols, q_vals)

        fig_sc, ax_sc1 = plt.subplots(figsize=(10, 4.5))

        color = "tab:blue"
        ax_sc1.set_xlabel("Service Target")
        ax_sc1.set_ylabel("Target Stock (units)", color=color)
        ax_sc1.plot(tau_levels, S_curve, color="#1428A0", linewidth=2.5, label="Target Stock")
        ax_sc1.tick_params(axis="y", labelcolor=color)

        # Highlight current tau
        current_S = np.interp(tau, q_cols, q_vals)
        ax_sc1.scatter(
            [tau],
            [current_S],
            color="red",
            s=80,
            zorder=5,
            label=f"Current choice (service level={tau:.0%}, stock={current_S:.0f})",
        )

        ax_sc1.grid(True, linestyle=":", alpha=0.6)
        ax_sc1.set_title("Target Stock by Service Target", color="#172033", pad=14)
        ax_sc1.set_facecolor("#FFFFFF")
        ax_sc1.grid(True, linestyle="-", alpha=0.12, color="#64748B")
        ax_sc1.spines[["top", "right"]].set_visible(False)
        ax_sc1.legend(loc="upper left", frameon=False)

        st.pyplot(fig_sc)
        plt.close(fig_sc)
        scenario_df = pd.DataFrame({
            "Service Target": [0.50, 0.75, 0.90, 0.95],
            "Target Stock": [
                np.interp(level, q_cols, q_vals) for level in [0.50, 0.75, 0.90, 0.95]
            ],
        })
        scenario_df["Target Stock"] = scenario_df["Target Stock"].round(0).astype(int)
        scenario_df["Service Target"] = scenario_df["Service Target"].map(lambda value: f"{value:.0%}")
        st.markdown(scenario_df.to_html(
            index=False, classes="business-table", border=0, justify="left"
        ), unsafe_allow_html=True)
        st.markdown(
            f'<div class="business-note"><strong>Business trade-off:</strong> '
            f'raising the service target to 95% requires approximately '
            f'{scenario_df["Target Stock"].iloc[-1] - scenario_df["Target Stock"].iloc[1]:,} '
            f'more units than the current 75% target. This reduces shortage risk, but ties up more working capital.</div>',
            unsafe_allow_html=True,
        )

# ===========================================================================
# Tab 4: Business Explanation
# ===========================================================================
with tab_why:
    st.subheader("Business Decision Explanation")
    st.markdown(
        f"Business explanation for the {selected_model.upper()} forecast and the resulting purchasing decision."
    )

    if fc_data:
        expected_demand = float(fc_data["q50"])
        low_demand = float(fc_data["q10"])
        high_demand = float(fc_data["q90"])
        range_width = high_demand - low_demand
        uncertainty_ratio = range_width / expected_demand if expected_demand else 0.0
        risk_label = "High" if uncertainty_ratio >= 1.0 else "Moderate" if uncertainty_ratio >= 0.5 else "Low"
        driver_card_1, driver_card_2, driver_card_3 = st.columns(3)
        driver_card_1.metric("Forecast Model", selected_model.upper())
        driver_card_2.metric("Demand Volatility", risk_label)
        driver_card_3.metric("Expected Demand Range", f"{low_demand:.0f}–{high_demand:.0f} units")
        st.markdown(
            f'<div class="business-note"><strong>Business interpretation:</strong> '
            f'expected demand is {expected_demand:.0f} units, with a likely range of '
            f'{low_demand:.0f} to {high_demand:.0f} units. The {risk_label.lower()} '
            'volatility level should guide the buffer kept above expected demand.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("### What this means for the order")
        common_driver_df = pd.DataFrame({
            "Business Signal": [
                "Expected demand",
                "Demand uncertainty",
                "Service target",
                "Inventory position",
            ],
            "Decision Meaning": [
                f"Plan around {expected_demand:.0f} units for the next week.",
                f"Demand may move across a {range_width:.0f}-unit range.",
                f"Protect against shortage risk at the selected {tau:.0%} service target.",
                "Subtract available warehouse stock from the target stock.",
            ],
        })
        st.markdown(common_driver_df.to_html(
            index=False, classes="business-table", border=0, justify="left"
        ), unsafe_allow_html=True)


# ===========================================================================
# Tab 5: Models Comparison
# ===========================================================================
with tab_models:
    st.subheader("Model Performance")
    st.caption("Benchmark results across validation and test data. Lower error indicates stronger forecast performance.")

    metrics_data = api_get("/models")
    if metrics_data:
        metrics_list = metrics_data.get("metrics", [])
        if metrics_list:
            m_df = pd.DataFrame(metrics_list)
            test_df = m_df[m_df["split"] == "test"].sort_values("mean_pinball")
            if not test_df.empty:
                best = test_df.iloc[0]
                best_model = str(best["model"]).upper()
                card1, card2, card3 = st.columns(3)
                with card1:
                    st.markdown(
                        f'<div class="section-card"><div class="label">Recommended Model</div>'
                        f'<div class="value">{best_model}</div></div>',
                        unsafe_allow_html=True,
                    )
                with card2:
                    st.markdown(
                        f'<div class="section-card"><div class="label">Best Test Pinball Loss</div>'
                        f'<div class="value">{best["mean_pinball"]:.3f}</div></div>',
                        unsafe_allow_html=True,
                    )
                with card3:
                    st.markdown(
                        f'<div class="section-card"><div class="label">Test Models Reviewed</div>'
                        f'<div class="value">{len(test_df)}</div></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("### Test Set Forecast Accuracy")
                chart_col, insight_col = st.columns([1.7, 1])
                with chart_col:
                    fig_m, ax_m = plt.subplots(figsize=(8, 3.8))
                    bars = ax_m.barh(
                        test_df["model"].str.upper(),
                        test_df["mean_pinball"],
                        color=["#1428A0" if i == 0 else "#9FB6FF" for i in range(len(test_df))],
                    )
                    ax_m.invert_yaxis()
                    ax_m.set_xlabel("Mean Pinball Loss (lower is better)")
                    ax_m.set_title("Forecast Accuracy on Test Data", color="#172033", pad=14, loc="left")
                    ax_m.set_facecolor("#FFFFFF")
                    ax_m.grid(axis="x", linestyle="-", alpha=0.12, color="#64748B")
                    ax_m.spines[["top", "right", "left"]].set_visible(False)
                    ax_m.tick_params(axis="y", length=0)
                    for bar in bars:
                        ax_m.text(
                            bar.get_width() + max(test_df["mean_pinball"]) * 0.015,
                            bar.get_y() + bar.get_height() / 2,
                            f"{bar.get_width():.3f}",
                            va="center",
                            fontsize=9,
                            color="#172033",
                        )
                    st.pyplot(fig_m)
                    plt.close(fig_m)

                with insight_col:
                    improvement = 0.0
                    if len(test_df) > 1 and float(test_df.iloc[1]["mean_pinball"]) > 0:
                        improvement = (
                            float(test_df.iloc[1]["mean_pinball"]) - float(best["mean_pinball"])
                        ) / float(test_df.iloc[1]["mean_pinball"])
                    st.markdown(
                        '<div class="section-card"><div class="label">Business Interpretation</div>'
                        f'<div class="value" style="font-size:1.05rem">Use {best_model} for the current deployment</div>'
                        '<p style="color:#64748B;margin-top:10px">'
                        "It achieved the lowest error on the held-out test data. "
                        f"Forecast loss is {improvement:.1%} lower than the next-best model. "
                        "This supports more reliable safety-stock and purchasing decisions."
                        "</p></div>",
                        unsafe_allow_html=True,
                    )

                display_columns = ["model", "split", "mae", "rmse", "r2", "wape", "mean_pinball"]
                def render_benchmark_table(data: pd.DataFrame) -> None:
                    display_df = data[[c for c in display_columns if c in data.columns]].copy()
                    display_df = display_df.sort_values("mean_pinball")
                    display_df["model"] = display_df["model"].str.upper()
                    display_df["mae"] = display_df["mae"].map(lambda value: f"{value:.2f}")
                    display_df["rmse"] = display_df["rmse"].map(lambda value: f"{value:.2f}")
                    display_df["r2"] = display_df["r2"].map(lambda value: f"{value:.2f}")
                    display_df["wape"] = display_df["wape"].map(lambda value: f"{value:.1%}")
                    display_df["mean_pinball"] = display_df["mean_pinball"].map(lambda value: f"{value:.3f}")
                    display_df.columns = [
                        {
                            "model": "Model",
                            "split": "Dataset",
                            "mae": "MAE",
                            "rmse": "RMSE",
                            "r2": "R²",
                            "wape": "WAPE",
                            "mean_pinball": "Mean Pinball Loss",
                        }.get(column, column)
                        for column in display_df.columns
                    ]
                    st.markdown(display_df.to_html(
                        index=False, classes="business-table", border=0, justify="left"
                    ), unsafe_allow_html=True)

                st.markdown("### Test Set Benchmark")
                st.caption("Use this holdout result to choose the model for deployment.")
                render_benchmark_table(test_df)

                validation_df = m_df[m_df["split"] == "val"]
                if not validation_df.empty:
                    st.markdown("### Validation Set Benchmark")
                    st.caption("Use this result to compare model choices during development and tuning.")
                    render_benchmark_table(validation_df)
