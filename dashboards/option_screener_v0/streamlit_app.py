from __future__ import annotations 

import os
import sys
from typing import Any, Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from openai import OpenAI
from datetime import datetime

# ---------------------------------------------------------------------
# Make project root importable so we can do "from src..."
# (Assumes this file lives in dashboards/option_screener_v1/)
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------
# Now safe to import from src/*
# ---------------------------------------------------------------------
from src.db_duckdb import (
    ensure_schema,
    upsert_screener_snapshot,
)
from src.polygon_client import (  # type: ignore  # noqa: E402
    compute_realized_vol,
    get_underlying_bars,
)

# ---------------------------------------------------------------------
# Streamlit page config + light CSS polish
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Volatility Alpha Engine – Option Screener V1",
    layout="wide",
)

# Center the main column and narrow it for a “phone scroll” feel
st.markdown(
    """
    <style>
    .block-container {
        max-width: 960px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
        margin: auto;
    }
    /* Make metrics slightly bolder / bigger */
    [data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Top-right personal badge (GitHub + LinkedIn)
st.markdown(
    """
    <style>
    .vae-top-links {
        position: fixed;
        top: 12px;
        right: 18px;
        font-size: 13px;
        z-index: 999;
        background: rgba(15, 23, 42, 0.85);
        padding: 4px 10px;
        border-radius: 999px;
        backdrop-filter: blur(6px);
    }
    .vae-top-links a {
        color: #9ca3af;
        text-decoration: none;
        margin-left: 6px;
        margin-right: 6px;
    }
    .vae-top-links a:hover {
        color: #e5e7eb;
        text-decoration: underline;
    }
    </style>
    <div class="vae-top-links">
        <span style="opacity:0.7;">Built by Brandon Theard ·</span>
        <a href="https://github.com/btheard3/volatility-alpha-engine" target="_blank">GitHub</a>|
        <a href="https://www.linkedin.com/in/brandon-theard-811b38131" target="_blank">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Sidebar – inputs, presets, and "Run Screener" button
# ---------------------------------------------------------------------
st.sidebar.header("Screener Settings")

# Simple session flags
if "has_run" not in st.session_state:
    st.session_state["has_run"] = False
if "last_run_ts" not in st.session_state:
    st.session_state["last_run_ts"] = None

recruiter_mode = st.sidebar.checkbox(
    "Recruiter / Demo Mode",
    value=False,
    help=(
        "Uses a fixed demo basket so you can show the app without worrying "
        "about bad ticker input or API hiccups mid-demo."
    ),
)

# Preset baskets
PRESET_BASKETS = {
    "Index & Tech (default)": ["SPY", "QQQ", "AMD", "TSLA"],
    "Mega-Cap Tech": ["AAPL", "MSFT", "META", "GOOGL", "NVDA"],
    "Volatility Watchlist": ["TSLA", "NVDA", "AMD", "SMCI", "COIN"],
    "S&P Sectors Mix": ["SPY", "XLF", "XLE", "XLY", "XLK"],
    "Custom only": [],
}

preset_label = st.sidebar.selectbox(
    "Preset basket",
    options=list(PRESET_BASKETS.keys()),
    index=0,
    help="Quick starting baskets for different use cases.",
)

# Common names list for multi-select
COMMON_NAMES = sorted(
    list(
        {
            *PRESET_BASKETS["Index & Tech (default)"],
            *PRESET_BASKETS["Mega-Cap Tech"],
            *PRESET_BASKETS["Volatility Watchlist"],
            *PRESET_BASKETS["S&P Sectors Mix"],
        }
    )
)

selected_from_list = st.sidebar.multiselect(
    "Pick from popular names",
    options=COMMON_NAMES,
    default=PRESET_BASKETS["Index & Tech (default)"],
    help="Layer on top of the preset basket or build your own mini-universe.",
)

extra_raw = st.sidebar.text_input(
    "Extra tickers (comma-separated)",
    value="",
    help="Add any other stocks here, e.g. SMCI, COIN.",
)


def _parse_ticker_input(raw: str) -> List[str]:
    """Split comma/space separated tickers, clean and dedupe."""
    if not raw:
        return []
    parts = [p.strip().upper() for p in raw.replace("\n", ",").split(",")]
    tickers = [p for p in parts if p]
    seen: set[str] = set()
    ordered: List[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


st.sidebar.markdown(
    """
Tip: Start with index ETFs + high-beta names to see where the action is.
"""
)

run_clicked = st.sidebar.button("Run Screener", type="primary")

# ---------------------------------------------------------------------
# Build final ticker list based on mode & inputs
# ---------------------------------------------------------------------
if recruiter_mode:
    # Ignore sidebar selections and use a fixed, safe basket
    base = PRESET_BASKETS["Index & Tech (default)"]
    tickers = list(base)
else:
    base = PRESET_BASKETS[preset_label]
    tickers: List[str] = list(base)

    # layer in multi-select names
    for t in selected_from_list:
        if t not in tickers:
            tickers.append(t)

    # layer in extra text input names
    extras = _parse_ticker_input(extra_raw)
    for t in extras:
        if t not in tickers:
            tickers.append(t)

# Show how many names will be scanned
st.sidebar.caption(f"📊 You’ll screen **{len(tickers)}** ticker(s) on the next run.")

# Update session flag / timestamp when the button is clicked
if run_clicked:
    st.session_state["has_run"] = True
    st.session_state["last_run_ts"] = datetime.now()

# Tiny API/status footer
if st.session_state["last_run_ts"] is not None:
    ts = st.session_state["last_run_ts"].strftime("%Y-%m-%d %H:%M")
    st.sidebar.caption(f"✅ Last run: {ts} (local time)")
else:
    st.sidebar.caption("⏱️ No runs yet this session.")

# ---------------------------------------------------------------------
# Early exits & welcome state
# ---------------------------------------------------------------------
if not st.session_state["has_run"]:
    st.subheader("Welcome – get today’s volatility snapshot in 3 quick steps")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(
            """
**1. Choose a basket**  
Pick a preset (e.g. *Index & Tech*) and layer in a few favourites.
"""
        )
    with col_b:
        st.markdown(
            """
**2. Add any extras**  
Type tickers into **Extra tickers** if they aren’t in the list.
"""
        )
    with col_c:
        st.markdown(
            """
**3. Run Screener**  
Click **Run Screener** to pull prices, realized vol, and edge scores.
"""
        )

    st.info(
        "Once you run the screener, you’ll see rankings, charts, and trade ideas "
        "based on today’s moves and volatility.",
        icon="✨",
    )
    st.stop()

if not tickers:
    st.info("No tickers selected. Add at least one ticker in the sidebar.", icon="⚠️")
    st.stop()

# ---------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------
title_col, badge_col = st.columns([0.85, 0.15])

with title_col:
    st.title("Volatility Alpha Engine – Option Screener V1 (Polygon RV)")

    st.success(
        "This is a daily volatility screener that ranks tickers by a "
        "composite edge score and highlights where a simple RL-style policy would "
        "prefer to take risk.",
        icon="✅",
    )

    st.markdown(
        """
V1 screener with Polygon-based realized volatility and a simple Daily Edge Score.

This is your **daily volatility radar**:

- You type in a list of tickers  
- We pull price/volume with `yfinance` and realized vol from **Polygon**  
- We compute a simple **Daily Edge Score**  
- We rank names by that edge score
"""
    )

with badge_col:
    if recruiter_mode:
        st.markdown(
            "<div style='margin-top:1rem; padding:0.4rem 0.8rem; "
            "border-radius:999px; background-color:#1d4ed8; color:white; "
            "font-size:0.8rem; text-align:center;'>Demo Mode</div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------
def _fetch_ticker_row(symbol: str) -> Dict[str, Any]:
    """
    Fetch metrics for one underlying:
    - Always returns price / day % / volume from yfinance
    - Tries Polygon for RV 20d / 60d (soft-fail)
    """
    symbol = symbol.upper()

    # --- 1. Base price / volume from yfinance ---
    hist = yf.download(symbol, period="5d", interval="1d", progress=False)
    if hist.empty or len(hist) < 2:  # type: ignore[arg-type]
        raise RuntimeError(f"No price history for {symbol} from yfinance")

    last = hist.iloc[-1]  # type: ignore[index]
    prev = hist.iloc[-2]  # type: ignore[index]

    last_price = float(last["Close"])
    day_pct = float((last["Close"] / prev["Close"] - 1.0) * 100.0)
    volume = float(last["Volume"])

    # --- 2. Defaults for Polygon fields ---
    rv_20d: float = float("nan")
    rv_60d: float = float("nan")
    nearest_exp: Optional[str] = None

    # --- 3. Polygon realized vol (soft-fail) ---
    try:
        bars = get_underlying_bars(symbol, days=90)
        rv_20d = compute_realized_vol(bars, window=20)
        rv_60d = compute_realized_vol(bars, window=60)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] RV failed for {symbol}: {e!r}")

    # --- 4. Simple edge score V1 ---
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


def classify_vol_env(avg_rv20: float) -> tuple[str, str]:
    """
    Turn average 20d realized volatility into a simple regime label + explanation.

    This is used in the 'Today at a Glance' card so a non-quant can follow the story.
    """
    if np.isnan(avg_rv20):
        return "Unknown", "We don’t have enough volatility data yet."

    if avg_rv20 < 15:
        return (
            "Calm",
            "Index-style moves. Edges mostly come from single-name gaps rather than broad chaos.",
        )

    if avg_rv20 < 25:
        return (
            "Normal",
            "Typical tape. There’s movement, but nothing out of control.",
        )

    return (
        "Hot",
        "Tape is whippy. Expect bigger swings and favour defined-risk structures.",
    )


def _classify_vol_env(avg_rv20: float) -> str:
    if np.isnan(avg_rv20):
        return "Unknown"
    if avg_rv20 < 12:
        return "Calm"
    if avg_rv20 < 25:
        return "Normal"
    return "Hot"


def _generate_today_brief(
    total_names: int,
    avg_rv20: float,
    max_edge_row: Optional[pd.Series],
    biggest_move_row: Optional[pd.Series],
) -> str:
    env = _classify_vol_env(avg_rv20)
    parts: List[str] = []

    # Vol environment
    if env == "Calm":
        parts.append(
            "Volatility is **calm** overall. Expect smaller daily swings and slower setups."
        )
    elif env == "Normal":
        parts.append(
            "Volatility is in a **normal** range. Moves are tradable but not extreme."
        )
    elif env == "Hot":
        parts.append(
            "Volatility is **hot**. Expect bigger swings and whippier moves across this basket."
        )
    else:
        parts.append(
            "Volatility environment is unclear due to limited data, so position size carefully."
        )

    # Edge / best idea
    if max_edge_row is not None and not np.isnan(max_edge_row["edge_score"]):
        parts.append(
            f"**Top edge name:** {max_edge_row['ticker']} "
            f"with a Daily Edge Score around {max_edge_row['edge_score']:.1f}%."
        )

    # Biggest mover
    if biggest_move_row is not None and not np.isnan(biggest_move_row["day_pct"]):
        direction = "up" if biggest_move_row["day_pct"] > 0 else "down"
        parts.append(
            f"**Biggest mover:** {biggest_move_row['ticker']} at "
            f"{biggest_move_row['day_pct']:+.2f}% on the day ({direction})."
        )

    # Portfolio size note
    parts.append(
        f"This scan covers **{total_names} names**. Treat this as a radar, not a trade recommendation."
    )

    return " ".join(parts)


def generate_ai_insights(df_for_ai: pd.DataFrame) -> str:
    """
    Take the ranking dataframe and return a human-friendly, hybrid
    (professional + beginner-friendly) explanation of today's volatility.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return (
            "AI insights are disabled. Add `OPENAI_API_KEY` to your `.env` file "
            "to see an automatic explanation of today's volatility and trade ideas."
        )

    client = OpenAI(api_key=api_key)

    table = df_for_ai.copy()
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
2. **Watchlist Ideas (3–5 bullets)**
3. **Risk & Context (2–3 short bullets)**

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


def _infer_playbook_row(row: pd.Series) -> str:
    """
    Very simple rule-based mapping from metrics to strategy type.
    This is v1; we'll refine later.
    """
    iv_proxy = row.get("rv_20d", None)
    day_pct = row.get("day_pct", None)

    if iv_proxy is None or pd.isna(iv_proxy):
        return "Insufficient data – monitor only for now."

    high_vol = iv_proxy >= 60
    huge_move = (
        abs(day_pct) >= 3
        if (day_pct is not None and not pd.isna(day_pct))
        else False
    )
    trendish = (
        abs(day_pct) >= 1.5
        if (day_pct is not None and not pd.isna(day_pct))
        else False
    )

    if high_vol and not huge_move:
        return "Best suited for credit spreads / iron condors (elevated vol, non-crazy move)."
    if high_vol and huge_move:
        return "Possible volatility harvest after a big move – stick to defined-risk spreads."
    if trendish and not high_vol:
        return "Best suited for directional plays or debit spreads (trend with moderate vol)."

    return "No strong edge – keep this on a watchlist and size smaller."


# ---------------------------------------------------------------------
# Early exits if no tickers
# ---------------------------------------------------------------------
if not tickers:
    st.info("Add at least one ticker in the sidebar to run the screener.")
    st.stop()

# Universe summary strip
st.caption(
    f"Universe: **{preset_label}** · {len(tickers)} name(s) · "
    f"{'Demo snapshot (Recruiter / Demo Mode)' if recruiter_mode else 'Live data from yfinance + Polygon'}"
)

# ---------------------------------------------------------------------
# Run screener
# ---------------------------------------------------------------------
raw_df = _build_screener_table(tickers)
if raw_df.empty:
    st.error("No data returned for the given tickers.")
    st.stop()

# Persist results for this run into DuckDB
ensure_schema()
upsert_screener_snapshot(raw_df)

# ---------------- KPI row + narrative "Today at a Glance" ----------------
total_names = len(raw_df)

valid_edge = raw_df["edge_score"].replace([np.inf, -np.inf], np.nan)
max_edge = valid_edge.max(skipna=True)
max_edge_row = raw_df.loc[valid_edge.idxmax()] if not np.isnan(max_edge) else None

valid_rv20 = raw_df["rv_20d"].replace([np.inf, -np.inf], np.nan)
avg_rv20 = float(valid_rv20.mean(skipna=True)) if not valid_rv20.empty else float("nan")

valid_day_pct = raw_df["day_pct"].replace([np.inf, -np.inf], np.nan)
if valid_day_pct.dropna().empty:
    biggest_move_row = None
else:
    biggest_move_row = raw_df.loc[valid_day_pct.abs().idxmax()]

env_label, env_explainer = classify_vol_env(avg_rv20)

top_edge_name = (
    f"{max_edge_row['ticker']} with Daily Edge Score ≈ {max_edge:.2f}%"
    if max_edge_row is not None
    else "None – no usable edge scores yet."
)

biggest_move_text = (
    f"{biggest_move_row['ticker']} at {biggest_move_row['day_pct']:+.2f}% on the day"
    if biggest_move_row is not None
    else "No clear standout move yet."
)

st.markdown("### Today at a Glance")

st.info(
    f"**Environment:** {env_label}. {env_explainer}\n\n"
    f"- **Universe size:** {total_names} name(s) in the *{preset_label}* basket.\n"
    f"- **Top edge name:** {top_edge_name}.\n"
    f"- **Biggest mover:** {biggest_move_text}.",
    icon="📊",
)

# Compact KPI strip under the narrative card
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Names Screened", f"{total_names}")

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
        st.metric("Avg 20d RV", f"{avg_rv20:.2f}%")
    else:
        st.metric("Avg 20d RV", "None")

with col4:
    if biggest_move_row is not None:
        st.metric(
            "Biggest Mover (Day %) ",
            f"{biggest_move_row['day_pct']:+.2f}%",
            help=f"{biggest_move_row['ticker']}",
        )
    else:
        st.metric("Biggest Mover (Day %)", "None")

st.markdown("---")

# ---------------- Table & charts intro ----------------
st.subheader("Daily Edge Ranking")

st.caption(
    "Names are sorted by **Daily Edge Score** – start at the top when you’re "
    "looking for ideas or walking a recruiter through the dashboard."
)

# ---------------------------------------------------------------------
# Screener table
# ---------------------------------------------------------------------
st.markdown(
    """
Each row is a ticker in your basket.  
Higher **Daily Edge Score** = more interesting today (bigger move and/or higher realized volatility).
"""
)

styled = _styled_table(raw_df)
st.dataframe(styled, use_container_width=True)

csv_download = raw_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download today’s screener results (CSV)",
    data=csv_download,
    file_name="vae_screener_snapshot.csv",
    mime="text/csv",
    help="Save this slice for notebooks, Excel, or further analysis.",
)

st.markdown("---")

# ---------------------------------------------------------------------
# Volatility & Edge charts – side-by-side with explainers
# ---------------------------------------------------------------------
st.markdown("### Volatility & Edge Overview")

chart_col1, chart_col2 = st.columns(2)

# -------- Left: RV 20d vs Edge Score --------
with chart_col1:
    st.markdown("**Chart 1 – RV vs Edge Score**")

    chart_data = (
        raw_df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["rv_20d", "edge_score"])
    )

    if chart_data.empty:
        st.info("Not enough data to plot RV vs Edge Score.")
    else:
        scatter = (
            alt.Chart(chart_data)
            .mark_circle(size=80, opacity=0.9)
            .encode(
                x=alt.X("rv_20d", title="20d Realized Volatility (%)"),
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

    with st.expander("🧠 How to read this chart", expanded=False):
        st.markdown(
            """
- Each dot is a **ticker** in your basket.  
- **Right** = more volatile (higher 20-day realized volatility).  
- **Up** = higher Daily Edge Score (bigger move and/or higher vol).  
- Top-right names are your **highest-energy setups** for today.  
- Green dots are **up days**, red dots are **down days**.
"""
        )

# -------- Right: Edge Score by Ticker --------
with chart_col2:
    st.markdown("**Chart 2 – Edge Score by Ticker**")

    top_df = (
        raw_df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["edge_score"])
        .sort_values("edge_score", ascending=False)
    )

    if top_df.empty:
        st.info("Not enough data to plot Edge Score by Ticker.")
    else:
        bar = (
            alt.Chart(top_df)
            .mark_bar()
            .encode(
                x=alt.X("ticker:N", title="Ticker"),
                y=alt.Y("edge_score:Q", title="Daily Edge Score (%)"),
                color=alt.value("#6366f1"),
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

    with st.expander("🧠 How to read this chart", expanded=False):
        st.markdown(
            """
- Bars are sorted from **highest** to **lowest** Daily Edge Score.  
- Start on the **left** when you’re hunting for trade ideas.  
- Higher bars = names where **today’s move + volatility** look most interesting.  
- Use this like a **ranked watchlist** for further notebook or options work.
"""
        )

st.markdown("---")

# ---------------------------------------------------------------------
# Trade Ideas – Playbook & “Today’s Best Play”
# ---------------------------------------------------------------------
st.subheader("Trade Ideas")

# Best play = highest edge_score row (already computed)
best_play_row = max_edge_row

if best_play_row is not None:
    best_play_text = _infer_playbook_row(best_play_row)
    st.markdown("#### Today’s Best Play")
    st.success(
        f"**{best_play_row['ticker']}** – {best_play_text}",
        icon="🚀",
    )

st.caption(
    "These rule-based ideas mirror the regimes and edge buckets from the RL notebooks – "
    "they’re a human-readable version of that logic."
)

st.markdown("#### Strategy Fit by Ticker")

selected_ticker = st.selectbox(
    "Choose a ticker for today’s playbook:",
    options=raw_df["ticker"].unique().tolist(),
)

selected_row = raw_df[raw_df["ticker"] == selected_ticker].iloc[0]
playbook_text = _infer_playbook_row(selected_row)

st.info(f"**{selected_ticker} – {playbook_text}**", icon="🎯")

# ---- Quick Trade Ideas (Top 3 by Edge Score) ----
st.markdown("#### Quick Trade Ideas (Top 3 by Edge Score)")

ranked = raw_df.sort_values("edge_score", ascending=False).head(3)
for row in ranked.itertuples():
    vol_label = _classify_vol_env(float(row.rv_20d))
    idea_text = _infer_playbook_row(pd.Series(row._asdict()))
    st.markdown(
        f"""
**{row.ticker}**  
- Volatility: {vol_label}  
- Edge Score: **{row.edge_score:.2f}%**  
- Idea: {idea_text}
"""
    )

st.markdown("---")

# ---------------------------------------------------------------------
# Big Picture – AI summary (on button click)
# ---------------------------------------------------------------------
st.subheader("Big Picture – AI Summary")

ai_col1, ai_col2 = st.columns([0.25, 0.75])

with ai_col1:
    generate_clicked = st.button(
        "Generate AI Summary",
        help="Uses OpenAI to summarize today’s volatility landscape and highlight watchlist ideas.",
    )

with ai_col2:
    st.markdown(
        "Click to get a concise explanation of what this dashboard is saying, "
        "written for both traders and hiring managers."
    )

with st.expander("View AI Summary", expanded=False):
    if generate_clicked:
        # Use the styled table’s underlying data if available, otherwise raw_df
        df_for_ai = (
            styled.data if hasattr(styled, "data") else raw_df  # type: ignore[attr-defined]
        )
        ai_text = generate_ai_insights(df_for_ai.rename(columns=str))
        st.write(ai_text)
    else:
        st.write(
            "Click **Generate AI Summary** above to create a short narrative for today’s setup."
        )

st.markdown("---")

# ---------------------------------------------------------------------
# Help + Notebook workflow section
# ---------------------------------------------------------------------
with st.expander("How I walk this dashboard"):
    st.markdown(
        """
        - Start with **Today at a Glance** to explain the volatility environment.
        - Use **Daily Edge Ranking** to pick 1–2 tickers and narrate why they stand out.
        - Point to **Volatility & Edge Overview** to show you understand risk vs opportunity.
        - Close with **Trade Ideas** to connect the stats back to practical options structures.
        """
    )

st.subheader("Help – How to Read This Dashboard")

st.markdown(
    """
- **Last Price** – latest close from Yahoo Finance  
- **Day %** – today’s % move vs prior close (green = up, red = down)  
- **RV 20d / RV 60d** – annualized realized volatility from Polygon based on the last 20 / 60 trading days  
- **Daily Edge Score** – simple composite: average of |Day %| and RV 20d (for now)  

Future versions could add:

- IV Rank and ATM IV  
- Single-ticker deep dive with RV trend and options snapshot  
- Export / strategy notebook templates
"""
)

st.markdown("### Under the Hood – Research Workflow")

st.markdown(
    """
This live screener is backed by a full DuckDB + notebook pipeline:

- **00 – Backfill & Data Ingest** – loads raw OHLCV into DuckDB and builds a clean, gap-free daily dataset per ticker.  
- **01 – Volatility & EDA** – profiles returns, volatility regimes, and liquidity to make sure the universe and date ranges are tradeable.  
- **02 – Feature Engineering** – builds the core edge score and volatility/liquidity features that later drive baselines and RL.  
- **03 – Backtesting Signals** – turns features into simple rule-based strategies and equity curves to confirm they have economic signal.  
- **04 – RL Environment** – wraps engineered features into a `VAETradingEnv` so we can compare learned RL behavior to the baselines.  
- **05 – Baseline Policies** – benchmarks random / edge-threshold / regime-based policies on the same dataset.  
- **06 – RL Training** – trains a tabular Q-learning agent and produces RL equity curves vs cash.  
- **07 – Diagnostics & Interpretation** – validates that the RL agent behaves sensibly (by regime and edge bucket) and that the strategy is not a fluke.
"""
)

with st.expander("🔗 Open the research notebooks on GitHub"):
    st.markdown(
        """
- [00 – Backfill & Data Ingest](https://github.com/btheard3/volatility-alpha-engine/blob/main/notebooks/00_backfill.ipynb)
- [01 – Volatility & EDA](https://github.com/btheard3/volatility-alpha-engine/blob/main/notebooks/01_eda_volatility_alpha.ipynb)
- [02 – Feature Engineering](https://github.com/btheard3/volatility-alpha-engine/blob/main/notebooks/02_feature_engineering.ipynb)
- [03 – Backtesting Signals](https://github.com/btheard3/volatility-alpha-engine-main/blob/main/notebooks/03_backtesting_signals.ipynb)
- [04 – RL Environment](https://github.com/btheard3/volatility-alpha-engine/blob/main/notebooks/04_rl_environment.ipynb)
- [05 – Baseline Policies](https://github.com/btheard3/volatility-alpha-engine/blob/main/notebooks/05_rl_baseline_policies.ipynb)
- [06 – RL Training](https://github.com/btheard3/volatility-alpha-engine/blob/main/notebooks/06_rl_training_agent.ipynb)
- [07 – Diagnostics & Interpretation](https://github.com/btheard3/volatility-alpha-engine/blob/main/notebooks/07_rl_diagnostics.ipynb)
        """
    )

st.markdown("---")

st.caption(
    "Built by Brandon Theard as part of the **Volatility Alpha Engine** project. "
    "Source code: [GitHub repo](https://github.com/btheard3/volatility-alpha-engine) · "
    "Contact: [LinkedIn](https://www.linkedin.com/in/brandon-theard-811b38131)"
)
# ---------------------------------------------------------------------
