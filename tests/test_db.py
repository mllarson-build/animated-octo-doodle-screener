"""
Tests for screener.db — SQLite data layer.

Covers: init, CSV migration, CRUD, close trade flow, edge cases.
"""

import os
import sqlite3
import tempfile
import pytest
import pandas as pd

# Patch DB_PATH before importing db module
_test_dir = tempfile.mkdtemp()


@pytest.fixture(autouse=True)
def _isolate_db(monkeypatch, tmp_path):
    """Each test gets its own fresh database."""
    import screener.db as db_mod

    db_file = str(tmp_path / "test.db")
    csv_file = str(tmp_path / "trade_log.csv")
    csv_bak = str(tmp_path / "trade_log.csv.bak")

    monkeypatch.setattr(db_mod, "DB_DIR", str(tmp_path))
    monkeypatch.setattr(db_mod, "DB_PATH", db_file)
    monkeypatch.setattr(db_mod, "CSV_PATH", csv_file)
    monkeypatch.setattr(db_mod, "CSV_BACKUP_PATH", csv_bak)

    yield tmp_path


def _make_trade(**overrides):
    """Helper to build a trade dict with defaults."""
    base = {
        "timestamp": "2026-03-28 10:00",
        "ticker": "AAPL",
        "entry": 150.0,
        "target": 175.0,
        "stop": 138.0,
        "rr": 2.08,
        "setup_score": 3,
        "upside_pct": 16.7,
        "shares": 10,
        "summary": "Test trade",
        "actual_entry": None,
        "actual_exit": None,
        "outcome_notes": "",
        "status": "Watching",
        "signal_source": "Value Recovery",
        "thesis": "Testing oversold bounce",
        "conviction": 4,
        "suggested_hold_period": "medium (1-3 months)",
        "sector": "Technology",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestInitDb:
    def test_creates_tables(self):
        from screener.db import init_db, _get_connection

        init_db()
        conn = _get_connection()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        table_names = {r["name"] for r in tables}
        assert "trades" in table_names

    def test_wal_mode(self):
        from screener.db import init_db, _get_connection

        init_db()
        conn = _get_connection()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"

    def test_idempotent(self):
        from screener.db import init_db

        init_db()
        init_db()  # second call should not raise


# ---------------------------------------------------------------------------
# CSV Migration
# ---------------------------------------------------------------------------

class TestCsvMigration:
    def test_migrates_full_csv(self, tmp_path):
        from screener.db import init_db, migrate_csv_if_needed, load_trades, CSV_PATH

        # Write a CSV with all standard columns
        csv_data = pd.DataFrame([{
            "timestamp": "2026-03-20 09:30",
            "ticker": "MSFT",
            "entry": "350.0",
            "target": "400.0",
            "stop": "322.0",
            "rr": "1.79",
            "setup_score": "3",
            "upside_pct": "14.3",
            "shares": "5",
            "summary": "Test",
            "actual_entry": "",
            "actual_exit": "",
            "outcome_notes": "",
            "status": "Watching",
        }])
        csv_data.to_csv(CSV_PATH, index=False)

        init_db()
        result = migrate_csv_if_needed()

        assert result is not None
        assert result["imported"] == 1
        assert result["skipped"] == 0

        df = load_trades()
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "MSFT"
        assert df.iloc[0]["signal_source"] == "Unknown"

    def test_empty_strings_become_none(self, tmp_path):
        from screener.db import init_db, migrate_csv_if_needed, load_trades, CSV_PATH

        csv_data = pd.DataFrame([{
            "timestamp": "2026-03-20 09:30",
            "ticker": "AAPL",
            "entry": "150.0",
            "target": "175.0",
            "stop": "138.0",
            "rr": "2.08",
            "setup_score": "3",
            "upside_pct": "16.7",
            "shares": "10",
            "summary": "Test",
            "actual_entry": "",  # empty string
            "actual_exit": "",   # empty string
            "outcome_notes": "",
            "status": "Watching",
        }])
        csv_data.to_csv(CSV_PATH, index=False)

        init_db()
        migrate_csv_if_needed()

        df = load_trades()
        assert pd.isna(df.iloc[0]["actual_entry"]) or df.iloc[0]["actual_entry"] is None

    def test_no_csv_returns_none(self):
        from screener.db import init_db, migrate_csv_if_needed

        init_db()
        result = migrate_csv_if_needed()
        assert result is None

    def test_skips_if_db_has_data(self, tmp_path):
        from screener.db import init_db, migrate_csv_if_needed, save_trade, CSV_PATH

        init_db()
        save_trade(_make_trade())

        # Create CSV after DB has data
        pd.DataFrame([{"timestamp": "x", "ticker": "Y"}]).to_csv(CSV_PATH, index=False)

        result = migrate_csv_if_needed()
        assert result is None  # skipped

    def test_csv_renamed_to_bak(self, tmp_path):
        from screener.db import init_db, migrate_csv_if_needed, CSV_PATH, CSV_BACKUP_PATH

        pd.DataFrame([{
            "timestamp": "2026-03-20", "ticker": "X", "entry": "100",
            "target": "110", "stop": "90", "rr": "1", "setup_score": "1",
            "upside_pct": "10", "shares": "1", "summary": "", "actual_entry": "",
            "actual_exit": "", "outcome_notes": "", "status": "Watching",
        }]).to_csv(CSV_PATH, index=False)

        init_db()
        migrate_csv_if_needed()

        assert not os.path.exists(CSV_PATH)
        assert os.path.exists(CSV_BACKUP_PATH)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

class TestSaveTrade:
    def test_save_and_load(self):
        from screener.db import init_db, save_trade, load_trades

        init_db()
        trade_id = save_trade(_make_trade())

        assert trade_id > 0
        df = load_trades()
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "AAPL"
        assert df.iloc[0]["signal_source"] == "Value Recovery"
        assert df.iloc[0]["conviction"] == 4

    def test_missing_optional_fields(self):
        from screener.db import init_db, save_trade, load_trades

        init_db()
        trade = _make_trade(thesis="", conviction=0, sector="")
        save_trade(trade)

        df = load_trades()
        assert len(df) == 1

    def test_multiple_trades(self):
        from screener.db import init_db, save_trade, load_trades

        init_db()
        save_trade(_make_trade(ticker="AAPL"))
        save_trade(_make_trade(ticker="MSFT"))
        save_trade(_make_trade(ticker="GOOGL"))

        df = load_trades()
        assert len(df) == 3


class TestLoadTrades:
    def test_filter_by_status(self):
        from screener.db import init_db, save_trade, load_trades

        init_db()
        save_trade(_make_trade(status="Watching"))
        save_trade(_make_trade(status="Executed"))
        save_trade(_make_trade(status="Closed — Win"))

        watching = load_trades(status_filter="Watching")
        assert len(watching) == 1

        closed = load_trades(status_filter="Closed")
        assert len(closed) == 1

    def test_empty_table(self):
        from screener.db import init_db, load_trades

        init_db()
        df = load_trades()
        assert df.empty


class TestUpdateTrade:
    def test_update_single_field(self):
        from screener.db import init_db, save_trade, update_trade, load_trades

        init_db()
        tid = save_trade(_make_trade())
        update_trade(tid, {"outcome_notes": "Looking good"})

        df = load_trades()
        assert df.iloc[0]["outcome_notes"] == "Looking good"

    def test_update_nonexistent_id(self):
        from screener.db import init_db, update_trade

        init_db()
        # Should not raise, just updates 0 rows
        update_trade(9999, {"status": "Cancelled"})


class TestCloseTrade:
    def test_close_win(self):
        from screener.db import init_db, save_trade, execute_trade, close_trade, load_trades

        init_db()
        tid = save_trade(_make_trade())
        execute_trade(tid, entry_price=150.0)
        result = close_trade(tid, exit_price=170.0, exit_reason="Hit Target")

        assert result["status"] == "Closed — Win"
        assert result["pnl"] == 200.0  # (170 - 150) * 10 shares

        df = load_trades()
        row = df.iloc[0]
        assert row["status"] == "Closed — Win"
        assert row["exit_reason"] == "Hit Target"
        assert row["actual_exit"] == 170.0
        assert row["closed_at"] != ""

    def test_close_loss(self):
        from screener.db import init_db, save_trade, execute_trade, close_trade

        init_db()
        tid = save_trade(_make_trade())
        execute_trade(tid, entry_price=150.0)
        result = close_trade(tid, exit_price=130.0, exit_reason="Hit Stop")

        assert result["status"] == "Closed — Loss"
        assert result["pnl"] == -200.0

    def test_close_non_executed_raises(self):
        from screener.db import init_db, save_trade, close_trade

        init_db()
        tid = save_trade(_make_trade(status="Watching"))

        with pytest.raises(ValueError, match="not in Executed status"):
            close_trade(tid, exit_price=170.0, exit_reason="Hit Target")

    def test_execute_trade(self):
        from screener.db import init_db, save_trade, execute_trade, load_trades

        init_db()
        tid = save_trade(_make_trade())
        execute_trade(tid, entry_price=148.50)

        df = load_trades()
        row = df.iloc[0]
        assert row["status"] == "Executed"
        assert row["actual_entry"] == 148.50
        assert row["executed_at"] != ""
