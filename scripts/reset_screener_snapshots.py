from pathlib import Path
import sys

# Make sure project root is on sys.path so `src` can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db_duckdb import connect_duckdb, DB_PATH


def main() -> None:
    con = connect_duckdb(DB_PATH, read_only=False)
    print("Connected to:", DB_PATH)

    # Nuke the old version of the table
    con.execute("DROP TABLE IF EXISTS screener_snapshots")
    con.close()
    print("Dropped screener_snapshots. It will be recreated on next run.")


if __name__ == "__main__":
    main()