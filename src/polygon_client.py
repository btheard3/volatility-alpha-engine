from __future__ import annotations

import os
import json
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Massive / Polygon REST client ---------------------------------------------
# The package was historically "polygon-api-client" but the namespace
# rebranded to "massive". We try massive first, then polygon for safety.

try:  # new name
    from massive import RESTClient  # type: ignore
except ImportError:  # fallback for older installs
    from polygon import RESTClient  # type: ignore


load_dotenv()
_POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

if not _POLYGON_API_KEY:
    raise RuntimeError(
        "POLYGON_API_KEY is not set. "
        "Add it to your .env so the Volatility Alpha Engine can use Polygon."
    )

# Simple singleton client
_client: Optional[RESTClient] = None


def _get_client() -> RESTClient:
    global _client
    if _client is None:
        _client = RESTClient(api_key=_POLYGON_API_KEY)  # type: ignore[call-arg]
    return _client


# ---------------------------------------------------------------------------
# Disk caching helpers (so we don't hammer the free API tier)
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_DIR / "data" / "cache_polygon"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _read_cache(path: Path, max_age_hours: float) -> Optional[pd.DataFrame]:
    """Return DataFrame from cache if younger than `max_age_hours`."""
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    ts = payload.get("_cached_at")
    if ts is None:
        return None

    cached_at = dt.datetime.fromisoformat(ts)
    age = (dt.datetime.utcnow() - cached_at).total_seconds() / 3600.0
    if age > max_age_hours:
        return None

    df = pd.DataFrame(payload.get("data", []))
    if df.empty:
        return None

    # Restore timestamp index when present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

    return df


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    payload = {
        "_cached_at": dt.datetime.utcnow().isoformat(),
        "data": df.reset_index().to_dict(orient="records"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, default=str)


# ---------------------------------------------------------------------------
# Underlying helpers (daily bars + realized volatility)
# ---------------------------------------------------------------------------


def get_underlying_bars(ticker: str, days: int = 60) -> pd.DataFrame:
    """
    Fetch daily bars for an underlying from Massive/Polygon and return
    a tidy DataFrame indexed by timestamp with OHLCV columns.

    We over-request a bit to account for weekends/holidays, then trim.
    Results are cached on disk to avoid 429s.
    """
    cache_path = CACHE_DIR / f"underlying_{ticker.upper()}_{days}d.json"
    cached = _read_cache(cache_path, max_age_hours=4)
    if cached is not None:
        return cached

    client = _get_client()

    end_date = dt.date.today()
    # Overfetch to absorb non-trading days
    start_date = end_date - dt.timedelta(days=days * 3)

    aggs_iter: Iterable[Any] = client.list_aggs(  # type: ignore[attr-defined]
        ticker=ticker.upper(),
        multiplier=1,
        timespan="day",
        from_=start_date.isoformat(),
        to=end_date.isoformat(),
        limit=days * 3,
    )

    rows: List[Dict[str, Any]] = []
    for agg in aggs_iter:
        a = cast(Any, agg)
        ts = getattr(a, "timestamp", None) or getattr(a, "t", None)
        if isinstance(ts, (int, float)):
            ts_dt = dt.datetime.fromtimestamp(ts / 1000.0, tz=dt.timezone.utc)
        else:
            ts_dt = ts

        rows.append(
            {
                "timestamp": ts_dt,
                "open": getattr(a, "open", None),
                "high": getattr(a, "high", None),
                "low": getattr(a, "low", None),
                "close": getattr(a, "close", None),
                "volume": getattr(a, "volume", None),
            }
        )

    df = pd.DataFrame(rows).dropna(subset=["close"])
    if df.empty:
        return df

    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    # Keep the last `days` trading days
    df = df.tail(days)

    _write_cache(cache_path, df)
    return df


def get_underlying_last_price(ticker: str) -> float:
    """
    Get the latest trade price for an underlying.
    Fallback: use the last close from daily bars if snapshot call fails.
    """
    client = _get_client()
    try:
        # Massive client uses `get_current_price`, older polygon client
        # may only expose `get_last_trade`. We try both.
        try:
            price = client.get_current_price(ticker)  # type: ignore[attr-defined]
            if isinstance(price, (int, float)):
                return float(price)
        except Exception:
            trade = client.get_last_trade(ticker=ticker)  # type: ignore[arg-type]
            val = getattr(trade, "price", None)
            if isinstance(val, (int, float)):
                return float(val)
    except Exception:
        pass

    bars = get_underlying_bars(ticker, days=5)
    if bars.empty:
        raise RuntimeError(f"Could not fetch last price for {ticker}")
    return float(bars["close"].iloc[-1])


def compute_realized_vol(bars: pd.DataFrame, window: int = 20) -> float:
    """
    Compute annualized realized volatility from daily OHLCV bars.

    Parameters
    ----------
    bars : DataFrame
        Must contain a `close` column.
    window : int
        Rolling window in trading days.

    Returns
    -------
    float
        RV in percent (e.g. 25.3 == 25.3%).
    """
    if bars.empty or "close" not in bars.columns:
        return float("nan")

    # Make sure Pylance sees this as a Series, not a bare ndarray
    closes = pd.Series(
        pd.to_numeric(bars["close"], errors="coerce").dropna(),
        dtype="float64",
    )
    if len(closes) < window:
        return float("nan")

    # Series -> Series, Pylance is now happy with .diff()
    returns = np.log(closes).diff().dropna() # type: ignore
    rolling_std = returns.rolling(window=window).std().dropna()
    if rolling_std.empty:
        return float("nan")

    # Annualize: sqrt(252) * rolling std of daily log returns
    rv = float(rolling_std.iloc[-1] * np.sqrt(252.0) * 100.0)
    return rv



# ---------------------------------------------------------------------------
# Options helpers – expirations + simple chain snapshot
# ---------------------------------------------------------------------------


def list_option_expirations(ticker: str) -> List[str]:
    """
    Return a sorted list of unique expiration dates (YYYY-MM-DD) for the ticker.

    We only call the API once per day per underlying and cache the result.
    """
    cache_path = CACHE_DIR / f"expirations_{ticker.upper()}.json"
    cached = _read_cache(cache_path, max_age_hours=12)
    if cached is not None:
        return sorted(cached["expiration"].unique().tolist())

    client = _get_client()
    expirations: set[str] = set()

    # We fetch contracts once as of today and collect their expirations.
    today = dt.date.today()
    contracts: Iterable[Any] = client.list_options_contracts(  # type: ignore[attr-defined]
        underlying_ticker=ticker.upper(),
        as_of=today.isoformat(),
        limit=1000,
    )

    rows: List[Dict[str, Any]] = []
    for c in contracts:
        cc = cast(Any, c)
        exp = getattr(cc, "expiration_date", None)
        if isinstance(exp, dt.date):
            exp_str = exp.isoformat()
        else:
            exp_str = str(exp) if exp is not None else None

        if not exp_str:
            continue

        expirations.add(exp_str)
        rows.append({"expiration": exp_str})

    df = pd.DataFrame(rows)
    if not df.empty:
        _write_cache(cache_path, df)

    return sorted(expirations)


def _build_chain_df(contracts: Iterable[Any]) -> pd.DataFrame:
    """
    Turn an iterable of option contract objects into a tidy DataFrame
    with the fields we care about. Everything is treated as `Any`
    so Pylance stays happy.
    """
    rows: List[Dict[str, Any]] = []
    for c in contracts:
        cc = cast(Any, c)
        rows.append(
            {
                "ticker": getattr(cc, "ticker", None),
                "underlying": getattr(cc, "underlying_ticker", None),
                "expiration": getattr(cc, "expiration_date", None),
                "contract_type": getattr(cc, "contract_type", None),
                "strike_price": getattr(cc, "strike_price", None),
                "exercise_style": getattr(cc, "exercise_style", None),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalize expiration to string
    df["expiration"] = df["expiration"].apply(
        lambda x: x.isoformat() if isinstance(x, dt.date) else str(x)
    )

    return df


def get_options_chain_for_expiration(
    ticker: str, expiration: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch a simple options chain for a given expiration and split into
    calls / puts DataFrames.
    """
    cache_path = CACHE_DIR / f"chain_{ticker.upper()}_{expiration}.json"
    cached = _read_cache(cache_path, max_age_hours=6)
    if cached is not None:
        calls = cached[cached["contract_type"] == "call"].copy()
        puts = cached[cached["contract_type"] == "put"].copy()
        return calls, puts

    client = _get_client()

    contracts: Iterable[Any] = client.list_options_contracts(  # type: ignore[attr-defined]
        underlying_ticker=ticker.upper(),
        expiration_date=dt.datetime.strptime(expiration, "%Y-%m-%d").date(),
        as_of=dt.date.today().isoformat(),
        limit=2000,
    )

    df = _build_chain_df(contracts)
    if df.empty:
        empty = pd.DataFrame(
            columns=["ticker", "underlying", "expiration", "contract_type", "strike_price", "exercise_style"]
        )
        return empty.copy(), empty.copy()

    _write_cache(cache_path, df)

    calls = df[df["contract_type"] == "call"].copy()
    puts = df[df["contract_type"] == "put"].copy()
    return calls, puts


def get_nearest_expiration_chain(
    ticker: str,
) -> Tuple[Optional[str], pd.DataFrame, pd.DataFrame]:
    """
    Convenience helper:
    - find the nearest *future* expiration
    - fetch the options chain for that expiry
    """
    expirations = list_option_expirations(ticker)
    if not expirations:
        return None, pd.DataFrame(), pd.DataFrame()

    today = dt.date.today()

    def _parse(d: str) -> dt.date:
        return dt.datetime.strptime(d, "%Y-%m-%d").date()

    future_exps = [e for e in expirations if _parse(e) >= today]
    if not future_exps:
        exp = expirations[0]
    else:
        future_exps.sort(key=_parse)
        exp = future_exps[0]

    calls, puts = get_options_chain_for_expiration(ticker, exp)
    return exp, calls, puts
