import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

APP_TITLE = "Volatility Alpha Engine – Option Screener V1"


def fetch_price_history(ticker: str, lookback_days: int = 60) -> pd.DataFrame:
    """Download recent daily price history for realized volatility calc."""
    try:
        df = yf.download(
            ticker,
            period=f"{lookback_days + 30}d",  # extra buffer
            interval="1d",
            progress=False,
        )
        if df.empty:
            return pd.DataFrame()
        df = df.tail(lookback_days + 1)
        return df
    except Exception:
        return pd.DataFrame()


def realized_volatility_annualized(df: pd.DataFrame) -> float:
    """Compute annualized realized volatility from daily close prices."""
    if df is None or df.empty or "Close" not in df.columns:
        return np.nan
    close = df["Close"].dropna()
    if len(close) < 10:
        return np.nan
    returns = np.log(close / close.shift(1)).dropna()
    if returns.empty:
        return np.nan
    return float(np.sqrt(252) * returns.std())


def get_underlying_snapshot(ticker: str) -> dict:
    """Pull latest price/volume and realized vol for a single ticker."""
    info = {"ticker": ticker}

    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="2d", interval="1d")
        if hist.empty:
            return info

        latest = hist.tail(1).iloc[0]
        prev = hist.head(-1).tail(1)
        prev_close = prev["Close"].iloc[0] if not prev.empty else np.nan

        last_price = float(latest["Close"])
        volume = int(latest.get("Volume", 0))

        if not math.isnan(prev_close) and prev_close != 0:
            day_pct = (last_price / prev_close - 1) * 100
        else:
            day_pct = np.nan

        # realized vol from ~60 days
        hist_60 = fetch_price_history(ticker, lookback_days=60)
        rv_20d = realized_volatility_annualized(hist_60.tail(20))
        rv_60d = realized_volatility_annualized(hist_60)

        info.update(
            {
                "last_price": last_price,
                "day_pct": day_pct,
                "volume": volume,
                "rv_20d": rv_20d,
                "rv_60d": rv_60d,
            }
        )
    except Exception:
        # leave partial info
        pass

    return info


def get_nearest_exp_chain(ticker: str):
    """Fetch nearest options expiration + option chain (calls, puts)."""
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return None, None, None
        nearest_exp = exps[0]
        chain = t.option_chain(nearest_exp)
        return nearest_exp, chain.calls, chain.puts
    except Exception:
        return None, None, None


def find_atm_options(calls: pd.DataFrame, puts: pd.DataFrame, spot: float):
    """Find roughly ATM call and put given spot price."""
    if math.isnan(spot) or spot <= 0:
        return None, None

    atm_call = None
    atm_put = None

    if calls is not None and not calls.empty:
        calls = calls.copy()
        calls["dist"] = (calls["strike"] - spot).abs()
        calls_sorted = calls.sort_values("dist")
        atm_call = calls_sorted.iloc[0]

    if puts is not None and not puts.empty:
        puts = puts.copy()
        puts["dist"] = (puts["strike"] - spot).abs()
        puts_sorted = puts.sort_values("dist")
        atm_put = puts_sorted.iloc[0]

    return atm_call, atm_put


def compute_iv_features(
    calls: pd.DataFrame, puts: pd.DataFrame, atm_call, atm_put
) -> tuple[float, float]:
    """
    Compute ATM IV (avg of call/put IV) and a simple IV Rank (0-100)
    within the distribution of IVs for this expiration.
    """
    atm_iv = np.nan

    call_iv = getattr(atm_call, "impliedVolatility", np.nan)
    put_iv = getattr(atm_put, "impliedVolatility", np.nan)

    vals = [v for v in [call_iv, put_iv] if pd.notna(v)]
    if vals:
        atm_iv = float(np.mean(vals)) * 100.0  # convert to %
    else:
        atm_iv = np.nan

    iv_rank = np.nan
    if calls is not None and not calls.empty and puts is not None and not puts.empty:
        iv_all = pd.concat(
            [calls["impliedVolatility"], puts["impliedVolatility"]]
        ).dropna()
        if not iv_all.empty and pd.notna(atm_iv):
            atm_iv_dec = atm_iv / 100.0
            iv_min = float(iv_all.min())
            iv_max = float(iv_all.max())
            if iv_max > iv_min:
                iv_rank = (atm_iv_dec - iv_min) / (iv_max - iv_min) * 100.0

    return atm_iv, iv_rank


def build_screener_df(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for tk in tickers:
        tk = tk.strip().upper()
        if not tk:
            continue

        snapshot = get_underlying_snapshot(tk)
        last = snapshot.get("last_price", np.nan)
        day_pct = snapshot.get("day_pct", np.nan)
        volume = snapshot.get("volume", np.nan)
        rv_20d = snapshot.get("rv_20d", np.nan)
        rv_60d = snapshot.get("rv_60d", np.nan)

        nearest_exp, calls, puts = get_nearest_exp_chain(tk)

        atm_call, atm_put = None, None
        atm_iv, iv_rank = np.nan, np.nan
        if nearest_exp is not None and calls is not None and puts is not None:
            atm_call, atm_put = find_atm_options(calls, puts, last)
            if atm_call is not None or atm_put is not None:
                atm_iv, iv_rank = compute_iv_features(calls, puts, atm_call, atm_put)

        row = {
            "Ticker": tk,
            "Last Price": last,
            "Day %": day_pct,
            "Volume": volume,
            "RV 20d": rv_20d,
            "RV 60d": rv_60d,
            "Nearest Exp": nearest_exp,
            "ATM IV %": atm_iv,
            "IV Rank %": iv_rank,
            "ATM Call Strike": getattr(atm_call, "strike", np.nan),
            "ATM Call Bid": getattr(atm_call, "bid", np.nan),
            "ATM Call Ask": getattr(atm_call, "ask", np.nan),
            "ATM Call Volume": getattr(atm_call, "volume", np.nan),
            "ATM Put Strike": getattr(atm_put, "strike", np.nan),
            "ATM Put Bid": getattr(atm_put, "bid", np.nan),
            "ATM Put Ask": getattr(atm_put, "ask", np.nan),
            "ATM Put Volume": getattr(atm_put, "volume", np.nan),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # "Daily Edge Score V1" based on move, volume, realized vol, and IV rank
    df["Abs Day %"] = df["Day %"].abs()

    rank_cols = ["Abs Day %", "Volume", "RV 20d", "IV Rank %"]
    for col in rank_cols:
        if col in df.columns:
            valid = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if valid.empty:
                df[col + " Rank"] = np.nan
            else:
                df[col + " Rank"] = df[col].rank(pct=True)
        else:
            df[col + " Rank"] = np.nan

    df["Edge Score"] = (
        df["Abs Day % Rank"].fillna(0)
        + df["Volume Rank"].fillna(0)
        + df["RV 20d Rank"].fillna(0)
        + df["IV Rank % Rank"].fillna(0)
    ) / 4 * 100

    return df


def layout_header():
    st.title(APP_TITLE)
    st.caption(
        "V1: Screener with realized volatility, ATM implied volatility, IV Rank, and a composite edge score."
    )
    st.markdown(
        """
This is your **daily volatility radar**:

- You type in a list of tickers  
- We pull live data with `yfinance`  
- We compute realized vol, ATM implied vol, and a simple IV Rank  
- We rank names by a **Daily Edge Score V1**
        """
    )


def layout_sidebar():
    st.sidebar.header("Screener Settings")

    tickers_input = st.sidebar.text_input(
        "Tickers (comma-separated)",
        value="SPY, QQQ, TSLA, NVDA, AMD",
        help="Enter any list of optionable tickers.",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Tip: Start with index ETFs + high-beta names to see where the action is."
    )

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    return tickers


def layout_main(df: pd.DataFrame):
    if df.empty:
        st.warning("No data loaded. Check your tickers and try again.")
        return

    st.subheader("Daily Edge Ranking (V1)")

    df_display = df[
        [
            "Ticker",
            "Last Price",
            "Day %",
            "Volume",
            "RV 20d",
            "RV 60d",
            "ATM IV %",
            "IV Rank %",
            "Nearest Exp",
            "Edge Score",
        ]
    ].copy()

    df_display = df_display.sort_values("Edge Score", ascending=False)

    st.dataframe(
        df_display.style.format(
            {
                "Last Price": "{:.2f}",
                "Day %": "{:+.2f}%",
                "RV 20d": "{:.2f}",
                "RV 60d": "{:.2f}",
                "ATM IV %": "{:.1f}",
                "IV Rank %": "{:.1f}",
                "Edge Score": "{:.1f}",
            }
        ),
        use_container_width=True,
    )

    st.markdown(
        """
**How to read this:**

- **ATM IV %** – average implied vol of the at-the-money call/put  
- **IV Rank %** – where today's ATM IV sits within the IV range for that expiration (0 = low, 100 = high)  
- **Edge Score** – composite of move, volume, realized vol, and IV Rank  
        """
    )

    st.markdown("---")
    st.subheader("Options Detail – Nearest Expiration")

    tickers = df_display["Ticker"].tolist()
    selected = st.selectbox("Select ticker for options chain", tickers)

    row = df.loc[df["Ticker"] == selected].iloc[0]
    st.markdown(
        f"""
**{selected} snapshot**

- Last Price: `{row['Last Price']:.2f}`
- Day %: `{row['Day %']:+.2f}%`
- Volume: `{int(row['Volume']):,}`  
- RV 20d: `{row['RV 20d']:.2f}`  
- RV 60d: `{row['RV 60d']:.2f}`  
- ATM IV %: `{row['ATM IV %']:.1f}`  
- IV Rank %: `{row['IV Rank %']:.1f}`  
- Nearest Expiration: `{row['Nearest Exp']}`
        """
    )

    nearest_exp, calls, puts = get_nearest_exp_chain(selected)
    if nearest_exp is None or calls is None or calls.empty:
        st.info("No options data available for this ticker/expiration via yfinance.")
        return

    st.markdown(f"### Calls – {nearest_exp}")
    calls_view = calls[
        [
            "contractSymbol",
            "strike",
            "lastPrice",
            "bid",
            "ask",
            "impliedVolatility",
            "volume",
            "openInterest",
        ]
    ].copy()
    calls_view["impliedVolatility"] = calls_view["impliedVolatility"] * 100.0
    calls_view = calls_view.rename(columns={"impliedVolatility": "IV %"})
    st.dataframe(calls_view, use_container_width=True)

    st.markdown(f"### Puts – {nearest_exp}")
    puts_view = puts[
        [
            "contractSymbol",
            "strike",
            "lastPrice",
            "bid",
            "ask",
            "impliedVolatility",
            "volume",
            "openInterest",
        ]
    ].copy()
    puts_view["impliedVolatility"] = puts_view["impliedVolatility"] * 100.0
    puts_view = puts_view.rename(columns={"impliedVolatility": "IV %"})
    st.dataframe(puts_view, use_container_width=True)


def main():
    layout_header()
    tickers = layout_sidebar()

    if not tickers:
        st.info("Enter at least one ticker to run the screener.")
        return

    with st.spinner("Running screener… pulling live market + options data."):
        df = build_screener_df(tickers)

    layout_main(df)


if __name__ == "__main__":
    main()
