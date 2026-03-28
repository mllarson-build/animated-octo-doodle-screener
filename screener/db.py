"""
SQLite data layer for the screener trade log.

Provides all CRUD operations for the trades table, CSV migration from the
legacy trade_log.csv, and connection management with WAL mode.

Schema:
  trades — one row per trade idea / paper trade / closed trade
  Lifecycle: Watching → Executed → Closed (Win/Loss) → Cancelled
"""

import os
import sqlite3
import time
import datetime
import pandas as pd

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "screener.db")
CSV_PATH = os.path.join(DB_DIR, "trade_log.csv")
CSV_BACKUP_PATH = os.path.join(DB_DIR, "trade_log.csv.bak")

# Holding period string → days mapping (constants, not magic strings)
HOLD_PERIOD_DAYS = {
    "short (2-4 weeks)": 28,
    "medium (1-3 months)": 90,
    "long (3-6 months)": 180,
}

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    ticker TEXT NOT NULL,
    entry REAL,
    target REAL,
    stop REAL,
    rr REAL,
    setup_score INTEGER,
    upside_pct REAL,
    shares INTEGER DEFAULT 0,
    summary TEXT DEFAULT '',
    actual_entry REAL,
    actual_exit REAL,
    outcome_notes TEXT DEFAULT '',
    status TEXT DEFAULT 'Watching',
    signal_source TEXT DEFAULT 'Unknown',
    exit_reason TEXT DEFAULT '',
    thesis TEXT DEFAULT '',
    conviction INTEGER DEFAULT 0 CHECK(conviction BETWEEN 0 AND 5),
    suggested_hold_period TEXT DEFAULT '',
    sector TEXT DEFAULT '',
    executed_at TEXT DEFAULT '',
    closed_at TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_trades_signal_source ON trades(signal_source);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);
"""

# Columns that should be numeric (empty string → None during CSV import)
_NUMERIC_COLS = {
    "entry", "target", "stop", "rr", "setup_score", "upside_pct",
    "shares", "actual_entry", "actual_exit",
}


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    """Get a SQLite connection with WAL mode enabled (set per-connection)."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def _retry_on_locked(fn, max_retries=1, delay=0.1):
    """Retry a callable once if the database is locked."""
    try:
        return fn()
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower() and max_retries > 0:
            time.sleep(delay)
            return fn()
        raise


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create the trades table and indexes if they don't exist."""
    conn = _get_connection()
    try:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CSV migration
# ---------------------------------------------------------------------------

def migrate_csv_if_needed() -> dict:
    """
    If trade_log.csv exists and screener.db does not have trades yet,
    import CSV rows into SQLite. Renames CSV to .bak after import.

    Returns {"imported": N, "skipped": M} or None if no migration needed.
    """
    if not os.path.exists(CSV_PATH):
        return None

    conn = _get_connection()
    try:
        count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        if count > 0:
            return None  # DB already has data, skip migration

        try:
            csv_df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
        except Exception:
            return {"imported": 0, "skipped": 0, "error": "Could not read CSV"}

        imported = 0
        skipped = 0

        for _, row in csv_df.iterrows():
            try:
                values = {}
                for col in row.index:
                    val = row[col]
                    if col in _NUMERIC_COLS:
                        # Map empty strings to None for numeric columns
                        if val == "" or val is None:
                            values[col] = None
                        else:
                            try:
                                values[col] = float(val)
                            except (ValueError, TypeError):
                                values[col] = None
                    else:
                        values[col] = val if val != "" else ""

                conn.execute(
                    """INSERT INTO trades (
                        timestamp, ticker, entry, target, stop, rr,
                        setup_score, upside_pct, shares, summary,
                        actual_entry, actual_exit, outcome_notes, status,
                        signal_source
                    ) VALUES (
                        :timestamp, :ticker, :entry, :target, :stop, :rr,
                        :setup_score, :upside_pct, :shares, :summary,
                        :actual_entry, :actual_exit, :outcome_notes, :status,
                        'Unknown'
                    )""",
                    {
                        "timestamp": values.get("timestamp", ""),
                        "ticker": values.get("ticker", ""),
                        "entry": values.get("entry"),
                        "target": values.get("target"),
                        "stop": values.get("stop"),
                        "rr": values.get("rr"),
                        "setup_score": values.get("setup_score"),
                        "upside_pct": values.get("upside_pct"),
                        "shares": values.get("shares"),
                        "summary": values.get("summary", ""),
                        "actual_entry": values.get("actual_entry"),
                        "actual_exit": values.get("actual_exit"),
                        "outcome_notes": values.get("outcome_notes", ""),
                        "status": values.get("status", "Watching"),
                    },
                )
                imported += 1
            except Exception:
                skipped += 1

        conn.commit()

        # Rename CSV to backup
        try:
            if os.path.exists(CSV_BACKUP_PATH):
                os.remove(CSV_BACKUP_PATH)
            os.rename(CSV_PATH, CSV_BACKUP_PATH)
        except OSError:
            pass

        return {"imported": imported, "skipped": skipped}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def save_trade(trade_dict: dict) -> int:
    """
    Insert a new trade row. Returns the new row id.
    Raises sqlite3.OperationalError (after retry) if DB is locked.
    """
    conn = _get_connection()
    try:
        def do_insert():
            cursor = conn.execute(
                """INSERT INTO trades (
                    timestamp, ticker, entry, target, stop, rr,
                    setup_score, upside_pct, shares, summary,
                    actual_entry, actual_exit, outcome_notes, status,
                    signal_source, thesis, conviction,
                    suggested_hold_period, sector
                ) VALUES (
                    :timestamp, :ticker, :entry, :target, :stop, :rr,
                    :setup_score, :upside_pct, :shares, :summary,
                    :actual_entry, :actual_exit, :outcome_notes, :status,
                    :signal_source, :thesis, :conviction,
                    :suggested_hold_period, :sector
                )""",
                trade_dict,
            )
            conn.commit()
            return cursor.lastrowid

        return _retry_on_locked(do_insert)
    finally:
        conn.close()


def load_trades(status_filter: str = None) -> pd.DataFrame:
    """
    Load trades as a DataFrame. Optionally filter by status.
    status_filter can be a prefix like 'Closed' to match 'Closed — Win', etc.
    """
    conn = _get_connection()
    try:
        if status_filter:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status LIKE ? ORDER BY id DESC",
                (f"{status_filter}%",),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY id DESC"
            ).fetchall()

        if not rows:
            return pd.DataFrame()

        columns = [desc[0] for desc in conn.execute("SELECT * FROM trades LIMIT 0").description]
        return pd.DataFrame([dict(r) for r in rows], columns=columns)
    finally:
        conn.close()


def update_trade(trade_id: int, fields: dict) -> None:
    """Update specific fields on a trade by id."""
    if not fields:
        return
    conn = _get_connection()
    try:
        set_clause = ", ".join(f"{k} = :{k}" for k in fields)
        fields["_id"] = trade_id

        def do_update():
            conn.execute(
                f"UPDATE trades SET {set_clause} WHERE id = :_id",
                fields,
            )
            conn.commit()

        _retry_on_locked(do_update)
    finally:
        conn.close()


def close_trade(trade_id: int, exit_price: float, exit_reason: str) -> dict:
    """
    Close an executed trade. Sets status to 'Closed — Win' or 'Closed — Loss'
    based on P&L sign. Returns {"status": ..., "pnl": ...} or raises.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT actual_entry, shares, status FROM trades WHERE id = ?",
            (trade_id,),
        ).fetchone()

        if row is None:
            raise ValueError(f"Trade {trade_id} not found")
        if row["status"] != "Executed":
            raise ValueError(f"Trade {trade_id} is not in Executed status (is: {row['status']})")

        entry_price = row["actual_entry"]
        shares = row["shares"] or 0
        pnl = (exit_price - entry_price) * shares if entry_price else 0
        new_status = "Closed — Win" if pnl >= 0 else "Closed — Loss"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        def do_close():
            conn.execute(
                """UPDATE trades SET
                    actual_exit = ?, exit_reason = ?, status = ?,
                    closed_at = ?
                WHERE id = ?""",
                (exit_price, exit_reason, new_status, now, trade_id),
            )
            conn.commit()

        _retry_on_locked(do_close)
        return {"status": new_status, "pnl": round(pnl, 2)}
    finally:
        conn.close()


def execute_trade(trade_id: int, entry_price: float) -> None:
    """Mark a Watching trade as Executed with the actual entry price."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    update_trade(trade_id, {
        "status": "Executed",
        "actual_entry": entry_price,
        "executed_at": now,
    })


def bulk_update_trades(updates: list[dict]) -> None:
    """
    Apply a list of field updates: [{"id": N, "fields": {...}}, ...].
    Used for the inline data_editor save.
    """
    conn = _get_connection()
    try:
        for upd in updates:
            trade_id = upd["id"]
            fields = upd["fields"]
            if not fields:
                continue
            set_clause = ", ".join(f"{k} = :{k}" for k in fields)
            fields["_id"] = trade_id
            conn.execute(
                f"UPDATE trades SET {set_clause} WHERE id = :_id",
                fields,
            )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Initialization: ensure DB + migration on import
# ---------------------------------------------------------------------------

def ensure_db() -> dict | None:
    """Initialize DB and run CSV migration if needed. Returns migration result."""
    init_db()
    return migrate_csv_if_needed()
