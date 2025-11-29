from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI

# ----------------------------------------------------------------
# ENV + OpenAI client
# ----------------------------------------------------------------

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------------------------------------------
# Make project root importable so we can do "from src..."
# ----------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.polygon_client import (  # type: ignore  # noqa: E402
    compute_realized_vol,
    get_underlying_bars,
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
V1: Screener with Polygon-based realized volatility and a simple Daily Edge Score.

This is your **daily volatility radar**:

- You type in a list of tickers  
- We pull price/volume with `yfinance` and realized vol from **Polygon**  
- We compute a simple Daily Edge Score  
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
    seen: set[str] = set()
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
    - Tries Polygon for RV 20d / 60d (soft-fail, uses cache in polygon_client)
    - Nearest expiration currently disabled (to avoid 429 spam)
    """
    symbol = symbol.upper()

    # --- 1. Base price / volume from yfinance ---
    hist = yf.download(symbol, period="5d", interval="1d", progress=False)

    if hist.empty or len(hist) < 2:  # type: ignore[truthy-function]
        raise RuntimeError(f"No price history for {symbol} from yfinance")

    last = hist.iloc[-1]  # type: ignore[index]
    prev = hist.iloc[-2]  # type: ignore[index]

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


def _styled_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Build a nicely formatted Styler with numeric formatting + day % coloring.
    """
    if df.empty:
        return df.style  # type: ignore[return-value]

    table_df = df.copy()

    # Rename columns for display
    table_df = table_df.rename(
        columns={
            "ticker": "Ticker",
            "last_price": "Last Price",
            "day_pct": "Day %",
            "volume": "Volume",
            "rv_20d": "RV 20d",
            "rv_60d": "RV 60d",
            "edge_score": "Daily Edge Score",
            "nearest_exp": "Nearest Exp",
        }
    )

    def fmt_price(x: float) -> str:
        return "None" if np.isnan(x) else f"{x:.2f}"

    def fmt_pct_signed(x: float) -> str:
        return "None" if np.isnan(x) else f"{x:+.2f}%"

    def fmt_pct_pos(x: float) -> str:
        return "None" if np.isnan(x) else f"{x:.2f}%"

    def fmt_vol(x: float) -> str:
        return "None" if np.isnan(x) else f"{x:,.0f}"

    styler = table_df.style.format(
        {
            "Last Price": fmt_price,
            "Day %": fmt_pct_signed,
            "Volume": fmt_vol,
            "RV 20d": fmt_pct_pos,
            "RV 60d": fmt_pct_pos,
            "Daily Edge Score": fmt_pct_pos,
        }  # type: ignore[arg-type]
    )

    # Color Day % text: green for positive, red for negative
    def color_day_pct(col: pd.Series) -> List[str]:
        styles: List[str] = []
        for v in col:
            if isinstance(v, (int, float)) and not np.isnan(v):
                if v > 0:
                    styles.append("color: #22c55e;")  # green-500
                elif v < 0:
                    styles.append("color: #ef4444;")  # red-500
                else:
                    styles.append("")
            else:
                styles.append("")
        return styles

    if "Day %" in styler.data.columns:  # type: ignore[attr-defined]
        styler = styler.apply(color_day_pct, subset=["Day %"])

    return styler


def generate_ai_insights(df: pd.DataFrame) -> str:
    """
    Take the ranking dataframe and return a human-friendly, hybrid
    (professional + beginner-friendly) explanation of today's volatility.
    """

    if df.empty:
        return "No data available for AI insights today."

    # If no key, keep the app from crashing
    if not os.getenv("OPENAI_API_KEY"):
        return (
            "AI insights are disabled. Add `OPENAI_API_KEY` to your `.env` file "
            "to see an automatic explanation of today's volatility and trade ideas."
        )

    cols_for_ai = [
        "ticker",
        "last_price",
        "day_pct",
        "volume",
        "rv_20d",
        "rv_60d",
        "edge_score",
    ]

    table = df[cols_for_ai].copy()

    # Rename to human-readable column names for the prompt
    table = table.rename(
        columns={
            "ticker": "Ticker",
            "last_price": "Last Price",
            "day_pct": "Day %",
            "volume": "Volume",
            "rv_20d": "RV 20d",
            "rv_60d": "RV 60d",
            "edge_score": "Daily Edge Score",
        }
    )

    csv_snapshot = table.to_csv(index=False)

    prompt = f"""
You are an options & volatility trading mentor explaining a daily volatility dashboard
to an intelligent but non-expert user.

The user sees a table with these columns:
- Ticker
- Last Price
- Day % (today's move vs prior close)
- Volume
- RV 20d (20-day realized volatility, annualized, in %)
- RV 60d (60-day realized volatility, annualized, in %)
- Daily Edge Score (a composite ranking based on move + volatility)

Here is today's data in CSV format:

{csv_snapshot}

Write a concise explanation with a **hybrid tone**:
- Professional enough for a hiring manager / PM
- Clear enough for a newer trader

Follow this structure:

1. **Big Picture (2–3 sentences)**
   - What does volatility look like across these names today?
   - Is the overall environment quiet, moderate, or hot?

2. **Watchlist Ideas (3–5 bullet points)**
   - For each bullet, name the ticker and explain *why* it stands out
     (e.g., high Daily Edge Score, big Day %, unusual RV20 vs RV60, etc.)
   - Keep each bullet to 1–2 sentences max.

3. **Risk & Context (2–3 short bullets)**
   - Mention things like: “these are volatile names”, “position size carefully”,
     or “broad market may be calm but single-name risk is high.”

Guidelines:
- Use plain language, no formulas.
- Do NOT output any code or tables.
- Keep the whole answer under 220 words.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clear, pragmatic trading mentor who explains "
                        "volatility and options in simple, professional language."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()  # type: ignore[return-value]
    except Exception as e:
        return (
            "AI insights are temporarily unavailable "
            f"({e.__class__.__name__}). The dashboard data is still live and accurate."
        )


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
# Main content
# ----------------------------------------------------------------

if not tickers:
    st.info("Add at least one ticker in the sidebar to run the screener.")
    st.stop()

# Build raw numeric table
raw_df = _build_screener_table(tickers)
if raw_df.empty:
    st.error("No data returned for the given tickers.")
    st.stop()

# ---------------- KPI cards row ----------------
total_names = len(raw_df)

valid_edge = raw_df["edge_score"].replace([np.inf, -np.inf], np.nan)
max_edge = float(valid_edge.max(skipna=True))
max_edge_row = raw_df.loc[valid_edge.idxmax()] if not np.isnan(max_edge) else None

valid_rv20 = raw_df["rv_20d"].replace([np.inf, -np.inf], np.nan)
avg_rv20 = float(valid_rv20.mean(skipna=True)) if not valid_rv20.empty else float("nan")

valid_day_pct = raw_df["day_pct"].replace([np.inf, -np.inf], np.nan)
if valid_day_pct.dropna().empty:
    biggest_move_row = None
else:
    biggest_move_row = raw_df.loc[valid_day_pct.abs().idxmax()]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Names Scanned", f"{total_names}")

with col2:
    if max_edge_row is not None:
        st.metric(
            "Highest Edge Score",
            f"{max_edge:.2f}%",
            help=f"Top name: {max_edge_row['ticker']}",
        )
    else:
        st.metric("Highest Edge Score", "None")

with col3:
    if not np.isnan(avg_rv20):
        st.metric("Avg RV 20d", f"{avg_rv20:.2f}%")
    else:
        st.metric("Avg RV 20d", "None")

with col4:
    if biggest_move_row is not None:
        st.metric(
            "Biggest Mover (Day %)",
            f"{biggest_move_row['day_pct']:+.2f}%",
            help=f"{biggest_move_row['ticker']}",
        )
    else:
        st.metric("Biggest Mover (Day %)", "None")

st.markdown("---")

# ---------------- Screener table ----------------
st.subheader("Daily Edge Ranking (V1 – Polygon RV)")

styled = _styled_table(raw_df)
st.dataframe(styled, width="stretch")

# ---------------- Charts row ----------------
st.markdown("### Volatility & Edge Overview")

chart_col1, chart_col2 = st.columns(2)

# Scatter: RV 20d vs Edge Score
with chart_col1:
    chart_data = raw_df.copy()
    chart_data = chart_data.replace([np.inf, -np.inf], np.nan)
    chart_data = chart_data.dropna(subset=["rv_20d", "edge_score"])

    if chart_data.empty:
        st.info("Not enough data to plot RV vs Edge Score.")
    else:
        scatter = (
            alt.Chart(chart_data)
            .mark_circle(size=80, opacity=0.9)
            .encode(
                x=alt.X("rv_20d", title="RV 20d (%)"),
                y=alt.Y("edge_score", title="Daily Edge Score (%)"),
                color=alt.condition(
                    "datum.day_pct >= 0",
                    alt.value("#22c55e"),  # green
                    alt.value("#ef4444"),  # red
                ),
                tooltip=[
                    "ticker",
                    alt.Tooltip("last_price", title="Last Price", format=".2f"),
                    alt.Tooltip("day_pct", title="Day %", format="+.2f"),
                    alt.Tooltip("rv_20d", title="RV 20d", format=".2f"),
                    alt.Tooltip("rv_60d", title="RV 60d", format=".2f"),
                    alt.Tooltip("edge_score", title="Edge Score", format=".2f"),
                ],
            )
            .properties(height=320)
        )

        st.altair_chart(scatter, use_container_width=True)

# Bar: Top 5 Edge Opportunities
with chart_col2:
    top5 = raw_df.copy()
    top5 = top5.replace([np.inf, -np.inf], np.nan)
    top5 = top5.dropna(subset=["edge_score"])
    top5 = top5.sort_values("edge_score", ascending=False).head(5)

    if top5.empty:
        st.info("Not enough data to plot Top Edge Opportunities.")
    else:
        bar = (
            alt.Chart(top5)
            .mark_bar()
            .encode(
                x=alt.X("ticker:N", title="Ticker"),
                y=alt.Y("edge_score:Q", title="Daily Edge Score (%)"),
                color=alt.value("#6366f1"),  # indigo-ish
                tooltip=[
                    "ticker",
                    alt.Tooltip("edge_score", title="Edge Score", format=".2f"),
                    alt.Tooltip("rv_20d", title="RV 20d", format=".2f"),
                    alt.Tooltip("day_pct", title="Day %", format="+.2f"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(bar, use_container_width=True)

# ---------------- AI Insights ----------------
st.markdown("---")
st.subheader("AI Volatility Insights (Beta)")

ai_text = generate_ai_insights(raw_df)
st.markdown(ai_text)

# ---------------- Explainer ----------------
st.markdown(
    """
### How to read this

- **Last Price** – latest close from Yahoo Finance  
- **Day %** – today’s % move vs. prior close (green = up, red = down)  
- **RV 20d / RV 60d** – annualized realized volatility from Polygon based on the last 20 / 60 trading days  
- **Daily Edge Score** – simple composite: average of |Day %| and RV 20d (for now)  

Next versions will add:

- IV Rank and ATM IV  
- Single-ticker deep dive with RV trend and options snapshot  
- Export / notebook links and strategy suggestions
"""
)