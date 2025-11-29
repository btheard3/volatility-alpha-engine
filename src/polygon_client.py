import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient

# Load .env variables
load_dotenv()

_POLYGON_KEY = os.getenv("POLYGON_API_KEY")

# Project root & cache directories
BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_DIR / "data" / "cache_polygon"
EXP_CACHE_DIR = CACHE_DIR / "expirations"
CHAIN_CACHE_DIR = CACHE_DIR / "chains"

for p in [EXP_CACHE_DIR, CHAIN_CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)



def _get_client() -> RESTClient:
    """Return a Polygon REST client with API key loaded."""
    if not _POLYGON_KEY:
        raise RuntimeError("POLYGON_API_KEY not set in .env")
    return RESTClient(api_key=_POLYGON_KEY)


def get_underlying_bars(ticker: str, days: int = 60) -> pd.DataFrame:
    """
    Fetch daily OHLCV price bars for an underlying ticker.
    Returns a DataFrame indexed by timestamp.
    """
    client = _get_client()

    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 5)

    aggs = client.get_aggs(
        ticker=ticker,
        multiplier=1,
        timespan="day",
        from_=start.isoformat(),
        to=end.isoformat(),
    )

    if not aggs:
        return pd.DataFrame()

    df = pd.DataFrame(aggs)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()

    return df.tail(days)[["open", "high", "low", "close", "volume"]]


def get_underlying_last_price(ticker: str) -> float | None:
    """Return the most recent close price from Polygon."""
    df = get_underlying_bars(ticker, days=1)
    if df.empty:
        return None
    return float(df["close"].iloc[-1])

import numpy as np

def realized_volatility_from_polygon(ticker: str, days: int = 20) -> float | None:
    """
    Compute annualized realized volatility using Polygon daily bars.
    """
    df = get_underlying_bars(ticker, days=days+1)
    if df.empty or "close" not in df.columns:
        return None

    close = df["close"].dropna()
    if len(close) < 2:
        return None

    returns = np.log(close / close.shift(1)).dropna()
    if returns.empty:
        return None

    rv = np.sqrt(252) * returns.std()
    return float(rv)

# ---------------------------
# Options chain helpers with caching
# ---------------------------

def _expirations_cache_path(ticker: str) -> Path:
    return EXP_CACHE_DIR / f"{ticker.upper()}_expirations.csv"


def _chain_cache_paths(ticker: str, expiration: str) -> Tuple[Path, Path]:
    safe_exp = expiration.replace("-", "")
    calls_path = CHAIN_CACHE_DIR / f"{ticker.upper()}_{safe_exp}_calls.csv"
    puts_path = CHAIN_CACHE_DIR / f"{ticker.upper()}_{safe_exp}_puts.csv"
    return calls_path, puts_path


def list_option_expirations(
    ticker: str,
    use_cache_only: bool = False,
) -> List[str]:
    """
    Return a sorted list of available (non-expired) option expirations
    for a given underlying ticker using Polygon.

    Caching behavior:
      - If cache file exists, read & return it (no API call).
      - If use_cache_only=True and no cache, return [].
      - Else, call Polygon once, cache results to CSV, and return.
    """
    ticker = ticker.upper()
    cache_path = _expirations_cache_path(ticker)

    # 1) Try cache first
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path)
            exps = df["expiration"].astype(str).tolist()
            if exps:
                return sorted(set(exps))
        except Exception as e:
            print(f"[Polygon] expirations cache read error for {ticker}: {e}")

    if use_cache_only:
        # For portfolio/demo mode: never hit the API
        return []

    # 2) Fallback to live Polygon call + write cache
    client = _get_client()
    expirations = set()

    try:
        contracts = client.list_options_contracts(
            underlying_ticker=ticker,
            expired=False,
            limit=1000,
        )

        for c in contracts:
            if getattr(c, "expiration_date", None):
                expirations.add(c.expiration_date)

        exps_list = sorted(expirations)
        if exps_list:
            df = pd.DataFrame({"expiration": exps_list})
            df.to_csv(cache_path, index=False)

        return exps_list

    except Exception as e:
        print(f"[Polygon] list_option_expirations error for {ticker}: {e}")
        return []


def get_options_chain_for_expiration(
    ticker: str,
    expiration: str,
    use_cache_only: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch the full options chain (calls & puts) for a given underlying
    and expiration date from Polygon.

    Caching behavior:
      - If cache files exist, read & return (no API call).
      - If use_cache_only=True and cache missing, return empty DataFrames.
      - Else, call Polygon once, cache calls/puts separately, and return.
    """
    ticker = ticker.upper()
    calls_path, puts_path = _chain_cache_paths(ticker, expiration)

    # 1) Try cache first
    calls_df, puts_df = None, None
    if calls_path.exists():
        try:
            calls_df = pd.read_csv(calls_path)
        except Exception as e:
            print(f"[Polygon] calls cache read error for {ticker} {expiration}: {e}")

    if puts_path.exists():
        try:
            puts_df = pd.read_csv(puts_path)
        except Exception as e:
            print(f"[Polygon] puts cache read error for {ticker} {expiration}: {e}")

    if calls_df is not None and puts_df is not None:
        return calls_df, puts_df

    if use_cache_only:
        # In demo mode, don't ever hit the API
        return pd.DataFrame(), pd.DataFrame()

    # 2) Fallback to live Polygon call + write cache
    client = _get_client()
    rows = []

    try:
        contracts = client.list_options_contracts(
            underlying_ticker=ticker,
            expiration_date=expiration,
            expired=False,
            limit=1000,
        )

        for c in contracts:
            greeks = getattr(c, "greeks", None)

            rows.append(
                {
                    "symbol": c.ticker,
                    "type": c.contract_type,  # "call" or "put"
                    "strike": float(c.strike_price) if c.strike_price is not None else np.nan,
                    "expiration": c.expiration_date,
                    "bid": getattr(c, "bid_price", np.nan),
                    "ask": getattr(c, "ask_price", np.nan),
                    "mid": (
                        (getattr(c, "bid_price", np.nan) + getattr(c, "ask_price", np.nan)) / 2
                        if getattr(c, "bid_price", None) is not None
                        and getattr(c, "ask_price", None) is not None
                        else np.nan
                    ),
                    "last_trade_price": getattr(c, "last_trade_price", np.nan),
                    "volume": getattr(c, "trade_volume", np.nan),
                    "open_interest": getattr(c, "open_interest", np.nan),
                    "delta": getattr(greeks, "delta", np.nan) if greeks else np.nan,
                    "gamma": getattr(greeks, "gamma", np.nan) if greeks else np.nan,
                    "theta": getattr(greeks, "theta", np.nan) if greeks else np.nan,
                    "vega": getattr(greeks, "vega", np.nan) if greeks else np.nan,
                    "rho": getattr(greeks, "rho", np.nan) if greeks else np.nan,
                    "iv": getattr(greeks, "iv", np.nan) if greeks else np.nan,
                }
            )

    except Exception as e:
        print(f"[Polygon] get_options_chain_for_expiration error for {ticker} {expiration}: {e}")
        return pd.DataFrame(), pd.DataFrame()

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(rows)
    calls_df = df[df["type"] == "call"].reset_index(drop=True)
    puts_df = df[df["type"] == "put"].reset_index(drop=True)

    # Cache them for future / demo use
    try:
        calls_df.to_csv(calls_path, index=False)
        puts_df.to_csv(puts_path, index=False)
    except Exception as e:
        print(f"[Polygon] error writing chain cache for {ticker} {expiration}: {e}")

    return calls_df, puts_df


def get_nearest_expiration_chain(
    ticker: str,
    use_cache_only: bool = False,
) -> Tuple[Optional[str], pd.DataFrame, pd.DataFrame]:
    """
    Convenience helper:
      - finds the nearest (earliest) non-expired expiration
      - returns expiration string, calls_df, puts_df
    """
    expirations = list_option_expirations(ticker, use_cache_only=use_cache_only)
    if not expirations:
        return None, pd.DataFrame(), pd.DataFrame()

    nearest = expirations[0]
    calls, puts = get_options_chain_for_expiration(
        ticker, nearest, use_cache_only=use_cache_only
        )
    return nearest, calls, puts
