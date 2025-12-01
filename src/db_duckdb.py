from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


def _get_db_path() -> Path:
    path = Path("data/volatility_alpha.duckdb")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection() -> duckdb.DuckDBPyConnection:
    db_path = _get_db_path()
    conn = duckdb.connect(str(db_path))
    return conn


def ensure_schema() -> None:
    """
    Create the main table if it doesn't exist.

    One row per ticker per run_date.
    """
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS screener_snapshots (
            run_date      DATE,
            ticker        VARCHAR,
            last_price    DOUBLE,
            day_pct       DOUBLE,
            volume        DOUBLE,
            rv_20d        DOUBLE,
            rv_60d        DOUBLE,
            edge_score    DOUBLE,
            nearest_exp   VARCHAR,
            PRIMARY KEY (run_date, ticker)
        );
        """
    )
    conn.close()


def upsert_screener_snapshot(df: pd.DataFrame, run_date: Optional[pd.Timestamp] = None) -> None:
    """
    Save the current screener output to DuckDB.

    Expects df to have columns:
      ['ticker','last_price','day_pct','volume',
       'rv_20d','rv_60d','edge_score','nearest_exp']
    """
    if df.empty:
        return

    if run_date is None:
        run_date = pd.Timestamp.today().normalize()

    df_to_save = df.copy()
    df_to_save["run_date"] = run_date.date()

    cols = [
        "run_date",
        "ticker",
        "last_price",
        "day_pct",
        "volume",
        "rv_20d",
        "rv_60d",
        "edge_score",
        "nearest_exp",
    ]
    df_to_save = df_to_save[cols]

    conn = get_connection()
    ensure_schema()

    conn.register("tmp_screener", df_to_save)

    conn.execute(
        """
        INSERT OR REPLACE INTO screener_snapshots
        SELECT * FROM tmp_screener;
        """
    )
    conn.unregister("tmp_screener")
    conn.close()


def get_scanner_view_for_date(run_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Fetch the screener snapshot for the given run_date (defaults to today).
    """
    if run_date is None:
        run_date = pd.Timestamp.today().normalize()

    conn = get_connection()
    q = """
        SELECT
            run_date,
            ticker,
            last_price,
            day_pct,
            volume,
            rv_20d,
            rv_60d,
            edge_score,
            nearest_exp
        FROM screener_snapshots
        WHERE run_date = ?
        ORDER BY edge_score DESC NULLS LAST;
    """
    df = conn.execute(q, [run_date.date()]).fetchdf()
    conn.close()
    return df

