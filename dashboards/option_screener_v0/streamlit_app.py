from __future__ import annotations

import os
import sys

# --- Make project root importable so "src" works ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ---------------------------------------------------

import datetime as dt
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from src.polygon_client import (
    compute_realized_vol,
    get_underlying_bars,
    get_underlying_last_price,
    get_nearest_expiration_chain,
)


# ---------------------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Volatility Alpha Engine – Option Screener V1",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------
# Sidebar – inputs
# ---------------------------------------------------------------------

st.sidebar.header("Screener Settings")
tickers_raw = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value="SPY, QQQ, TSLA, NVDA, AMD",
    help="Start with index ETFs + high-beta names to see where the action is.",
)
run_button = st.sidebar.button("Run Screener")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tip: Keep the list short (5–15 tickers) to avoid rate limits on free data tiers."
)

tickers: List[str] = [
    t.strip().upper() for t in tickers_raw.split(",") if t.strip()
]


# ---------------------------------------------------------------------
# Helper to pull metrics for a single ticker
# ---------------------------------------------------------------------


def _fetch_ticker_row(symbol: str) -> Dict[str, Any]:
    """
    Fetch metrics for one underlying:
    - Always tries to return price / day % / volume from yfinance
    - Tries Polygon for RV + nearest expiration, but falls back cleanly
    """
    symbol = symbol.upper()

    # --- 1. Base price / volume from yfinance (this MUST work or we fail loudly) ---
    hist = yf.download(symbol, period="5d", interval="1d", progress=False)
    if hist.empty or len(hist) < 2: # type: ignore
        raise RuntimeError(f"No price history for {symbol} from yfinance")

    last = hist.iloc[-1] # type: ignore
    prev = hist.iloc[-2] # type: ignore

    last_price = float(last["Close"])
    day_pct = float((last["Close"] / prev["Close"] - 1.0) * 100.0)
    volume = float(last["Volume"])

    # --- 2. Defaults for Polygon fields ---
    rv_20d: float = float("nan")
    rv_60d: float = float("nan")
    nearest_exp: str | None = None

    # --- 3. Polygon realized vol (soft-fail) ---
    try:
        bars = get_underlying_bars(symbol, days=90)
        rv_20d = compute_realized_vol(bars, window=20)
        rv_60d = compute_realized_vol(bars, window=60)
    except Exception as e:  # soft fail
        print(f"[WARN] RV failed for {symbol}: {e!r}")

    # --- 4. Polygon nearest expiration (soft-fail) ---
    try:
        exp, calls, puts = get_nearest_expiration_chain(symbol)
        nearest_exp = exp
    except Exception as e:
        print(f"[WARN] chain failed for {symbol}: {e!r}")

    # --- 5. Simple edge score V1 ---
    components: list[float] = [abs(day_pct)]
    if not np.isnan(rv_20d):
        components.append(rv_20d)

    edge_score = float(np.mean(components)) if components else float("nan")

    return {
        "ticker": symbol,
        "last_price": last_price,
        "day_pct": day_pct,
        "volume": volume,
        "rv_20d": rv_20d,
        "rv_60d": rv_60d,
        "edge_score": edge_score,
        "nearest_exp": nearest_exp,
    }



# ---------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------

st.title("Volatility Alpha Engine – Option Screener V1 (Polygon RV)")
st.markdown(
    """
V1: Screener with **Polygon-based realized volatility**, ATM implied volatility (future),
IV Rank, and a composite edge score.

This is your **daily volatility radar**:
- You type in a list of tickers  
- We pull price & volume with `yfinance` and realized vol from **Polygon/Massive**  
- We compute **20-day and 60-day realized volatility**  
- We rank names by a **Daily Edge Score V1**
"""
)

if not run_button:
    st.info("👈 Enter tickers and hit **Run Screener** to pull live data.")
    st.stop()

if not tickers:
    st.warning("Please enter at least one ticker.")
    st.stop()

rows: List[Dict[str, Any]] = []
for symbol in tickers:
    rows.append(_fetch_ticker_row(symbol))

df = pd.DataFrame(rows)

if df.empty:
    st.error("No data returned. Check your tickers and try again.")
    st.stop()

# Sort by edge_score descending
df_sorted = df.sort_values("edge_score", ascending=False).reset_index(drop=True)

# Nicely formatted view
display_df = df_sorted.copy()
display_df.index = display_df.index + 1  # 1-based ranking
display_df.rename(
    columns={
        "ticker": "Ticker",
        "last_price": "Last Price",
        "day_pct": "Day %",
        "volume": "Volume",
        "rv_20d": "RV 20d",
        "rv_60d": "RV 60d",
        "edge_score": "Daily Edge Score",
        "nearest_exp": "Nearest Exp",
    },
    inplace=True,
)

# Formatting helpers
def _fmt_pct(x: Any) -> str:
    return f"{x:,.2f}%" if pd.notna(x) else "–"


def _fmt_price(x: Any) -> str:
    return f"{x:,.2f}" if pd.notna(x) else "–"


def _fmt_int(x: Any) -> str:
    return f"{int(x):,}" if pd.notna(x) else "–"


st.subheader("Daily Edge Ranking (V1 – Polygon RV)")

st.dataframe(
    display_df.style.format(
        {
            "Last Price": _fmt_price,
            "Day %": _fmt_pct,
            "Volume": _fmt_int,
            "RV 20d": _fmt_pct,
            "RV 60d": _fmt_pct,
            "Daily Edge Score": _fmt_pct,
        }
    ),
    use_container_width=True,
)

st.markdown("### How to read this")
st.markdown(
    """
- **Last Price** – latest close from Yahoo Finance  
- **Day %** – today’s % move vs. prior close  
- **RV 20d / 60d** – annualized realized volatility over the last 20 / 60 trading days  
- **Daily Edge Score** – simple V1 composite of short-term move + realized vol  
- **Nearest Exp** – closest listed options expiration Massive/Polygon can see  

V2/V3 will add:
- ATM IV %, IV Rank %, and skew  
- Strategy filters (iron condors, strangles, directional plays)  
- Visual gauges and color-coded risk bands
"""
)

