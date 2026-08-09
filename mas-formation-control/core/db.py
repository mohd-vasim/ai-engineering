"""
SQLite Database utility for Multi-Agent Formation Control Telemetry Persistence.
Stores and queries mission telemetry records at data/telemetry.db.
"""
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import pandas as pd


DEFAULT_DB_PATH = "data/telemetry.db"


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Ensures database directory and table exist, then returns connection."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mission_runs (
            run_id       TEXT PRIMARY KEY,
            thread_id    TEXT,
            timestamp    TEXT,
            num_drones   INTEGER,
            formation    TEXT,
            spacing_m    REAL,
            max_dev_m    REAL,
            final_err_m  REAL,
            min_clear_m  REAL,
            mission_ok   INTEGER,
            summary      TEXT
        )
    """)
    conn.commit()
    return conn


def save_mission_run(
    num_drones: int,
    formation: str,
    spacing_m: float,
    max_dev_m: float,
    final_err_m: float,
    min_clear_m: float,
    mission_ok: bool,
    summary: str = "",
    thread_id: str = "mas-formation-thread-v2",
    run_id: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Inserts or replaces a mission run record in the SQLite database."""
    conn = get_db_connection(db_path)
    if not run_id:
        run_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """
        INSERT OR REPLACE INTO mission_runs 
        (run_id, thread_id, timestamp, num_drones, formation, spacing_m, max_dev_m, final_err_m, min_clear_m, mission_ok, summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            thread_id,
            ts,
            int(num_drones),
            str(formation),
            float(spacing_m),
            float(max_dev_m),
            float(final_err_m),
            float(min_clear_m),
            1 if mission_ok else 0,
            str(summary),
        )
    )
    conn.commit()
    conn.close()
    return run_id


def get_all_mission_runs(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Returns all historical mission runs as a Pandas DataFrame sorted by timestamp descending."""
    conn = get_db_connection(db_path)
    query = "SELECT * FROM mission_runs ORDER BY timestamp DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_mission_stats(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Calculates high-level statistical indicators across historical runs."""
    df = get_all_mission_runs(db_path)
    if df.empty:
        return {
            "total_runs": 0,
            "success_rate": 0.0,
            "avg_max_dev": 0.0,
            "avg_final_err": 0.0,
            "avg_min_clear": 0.0,
        }
    
    total = len(df)
    successes = (df["mission_ok"] == 1).sum()
    return {
        "total_runs": total,
        "success_rate": round(float(successes / total) * 100.0, 1),
        "avg_max_dev": round(float(df["max_dev_m"].mean()), 3),
        "avg_final_err": round(float(df["final_err_m"].mean()), 3),
        "avg_min_clear": round(float(df["min_clear_m"].mean()), 3),
    }
