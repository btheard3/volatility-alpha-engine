from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# --- Make project root importable so we can do "from src..." ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ----------------------------------------------------------------

from src.polygon_client import (  # type: ignore
    get_underlying_bars,
    compute_realized_vol,
)

# ----------------------------------------------------------------
# Streamlit page config
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Volatility Alpha Engine – Option Screener V1",
    layout="wide",
)

st.title("Volatility Alpha Engine – Option Screener V1 (Polygon RV)")
st.markdown(
    """
V1: Screener with Polygon-based realized volatility, ATM implied volatility (placeholder),
IV Rank (placeholder), and a composite edge score.

This is your **daily volatility radar**:

- You type in a list of tickers  
- We pull price/volume with `yfinance` and realized vol from **Polygon**  
- We compute a simple Daily Edge Score V1  
- We rank names by that edge score
"""
)

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


def _parse_ticker_input(raw: str) -> List[str]:
    """Split comma/space separated tickers, clean and dedupe."""
    if not raw:
        return []
    parts = [p.strip().upper() for p in raw.replace("\n", ",").split(",")]
    tickers = [p for p in parts if p]
    # keep order but drop duplicates
    seen = set()
    ordered: List[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def _fetch_ticker_row(symbol: str) -> Dict[str, Any]:
    """
    Fetch metrics for one underlying:
    - Always returns price / day % / volume from yfinance
    - Tries Polygon for RV 20d / 60d (soft-fail)
    - Nearest expiration currently disabled (to avoid 429 spam)
    """
    symbol = symbol.upper()

    # --- 1. Base price / volume from yfinance ---
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
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] RV failed for {symbol}: {e!r}")

    # --- 4. Nearest expiration / chain (temporarily disabled) ---
    # TODO: Re-enable when we add a “click to load options chain” UI
    # try:
    #     exp, calls, puts = get_nearest_expiration_chain(symbol)
    #     nearest_exp = exp
    # except Exception as e:
    #     print(f"[WARN] chain failed for {symbol}: {e!r}")

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


def _build_screener_table(tickers: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for t in tickers:
        try:
            row = _fetch_ticker_row(t)
            rows.append(row)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] failed for {t}: {e!r}")
            rows.append(
                {
                    "ticker": t.upper(),
                    "last_price": float("nan"),
                    "day_pct": float("nan"),
                    "volume": float("nan"),
                    "rv_20d": float("nan"),
                    "rv_60d": float("nan"),
                    "edge_score": float("nan"),
                    "nearest_exp": None,
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("edge_score", ascending=False).reset_index(drop=True)
    return df


def _format_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    def fmt_price(x: float) -> str:
        return "None" if np.isnan(x) else f"{x:.2f}"

    def fmt_pct(x: float) -> str:
        return "None" if np.isnan(x) else f"{x:+.2f}%"

    def fmt_pos_pct(x: float) -> str:
        return "None" if np.isnan(x) else f"{x:.2f}%"

    def fmt_vol(x: float) -> str:
        return "None" if np.isnan(x) else f"{x:,.0f}"

    display = pd.DataFrame(
        {
            "Ticker": df["ticker"],
            "Last Price": df["last_price"].map(fmt_price),
            "Day %": df["day_pct"].map(fmt_pct),
            "Volume": df["volume"].map(fmt_vol),
            "RV 20d": df["rv_20d"].map(fmt_pos_pct),
            "RV 60d": df["rv_60d"].map(fmt_pos_pct),
            "Daily Edge Score": df["edge_score"].map(fmt_pos_pct),
            "Nearest Exp": df["nearest_exp"].fillna("None"),
        }
    )
    display.index = np.arange(1, len(display) + 1) # type: ignore
    return display


# ----------------------------------------------------------------
# Sidebar – input
# ----------------------------------------------------------------

st.sidebar.header("Screener Settings")
raw_tickers = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value="SPY, QQQ, TSLA, NVDA, AMD",
    help="Enter stock tickers separated by commas.",
)

st.sidebar.markdown(
    """
Tip: Start with index ETFs + high-beta names to see where the action is.
"""
)

tickers = _parse_ticker_input(raw_tickers)

# ----------------------------------------------------------------
# Main table
# ----------------------------------------------------------------

if not tickers:
    st.info("Add at least one ticker in the sidebar to run the screener.")
else:
    st.subheader("Daily Edge Ranking (V1 – Polygon RV)")
    raw_df = _build_screener_table(tickers)
    display_df = _format_for_display(raw_df)
    st.dataframe(display_df, width="stretch")

    st.markdown(
        """
### How to read this

- **Last Price** – latest close from Yahoo Finance  
- **Day %** – today’s % move vs. prior close  
- **RV 20d / RV 60d** – annualized realized volatility from Polygon based on the last 20 / 60 trading days  
- **Daily Edge Score** – simple composite: average of |Day %| and RV 20d (for now)  
- **Nearest Exp** – placeholder until we re-enable the options chain lookup
"""
    )
