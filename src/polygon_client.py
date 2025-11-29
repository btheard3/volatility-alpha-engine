import os
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient

# Load .env variables
load_dotenv()

_POLYGON_KEY = os.getenv("POLYGON_API_KEY")


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

