"""
Tests for screener.analytics — My Edge computations.

Covers: signal scorecard, trade stats, behavioral patterns, sparklines.
Uses synthetic DataFrames (no DB or API calls).
"""

import numpy as np
import pandas as pd
import pytest


def _closed_trades_df(rows):
    """Build a closed-trades DataFrame from a list of dicts with sensible defaults."""
    defaults = {
        "id": 1, "ticker": "AAPL", "actual_entry": 150.0, "actual_exit": 165.0,
        "shares": 10, "status": "Closed — Win", "signal_source": "Value Recovery",
        "conviction": 3, "exit_reason": "Hit Target", "timestamp": "2026-03-01 09:30",
        "executed_at": "2026-03-01 09:30", "closed_at": "2026-03-15 15:00",
        "sector": "Technology",
    }
    data = []
    for i, row in enumerate(rows):
        r = {**defaults, "id": i + 1}
        r.update(row)
        data.append(r)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Signal Scorecard
# ---------------------------------------------------------------------------

class TestSignalScorecard:
    def test_empty_returns_empty(self):
        from screener.analytics import compute_signal_scorecard
        result = compute_signal_scorecard(pd.DataFrame())
        assert result.empty

    def test_below_threshold_shows_need_message(self):
        from screener.analytics import compute_signal_scorecard
        df = _closed_trades_df([
            {"signal_source": "Value Recovery", "actual_exit": 160},
            {"signal_source": "Value Recovery", "actual_exit": 155},
        ])
        result = compute_signal_scorecard(df)
        assert len(result) == 1
        assert "Need" in str(result.iloc[0]["Win Rate"])

    def test_above_threshold_shows_stats(self):
        from screener.analytics import compute_signal_scorecard
        df = _closed_trades_df([
            {"signal_source": "VR", "actual_entry": 100, "actual_exit": 110},
            {"signal_source": "VR", "actual_entry": 100, "actual_exit": 105},
            {"signal_source": "VR", "actual_entry": 100, "actual_exit": 95},
            {"signal_source": "VR", "actual_entry": 100, "actual_exit": 115},
            {"signal_source": "VR", "actual_entry": 100, "actual_exit": 108},
        ])
        result = compute_signal_scorecard(df)
        row = result.iloc[0]
        assert row["Trades"] == 5
        assert "80.0%" in row["Win Rate"]  # 4/5 wins

    def test_multiple_sources(self):
        from screener.analytics import compute_signal_scorecard
        rows = (
            [{"signal_source": "VR", "actual_entry": 100, "actual_exit": 110}] * 5
            + [{"signal_source": "GM", "actual_entry": 100, "actual_exit": 90}] * 5
        )
        df = _closed_trades_df(rows)
        result = compute_signal_scorecard(df)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Trade Stats
# ---------------------------------------------------------------------------

class TestTradeStats:
    def test_empty_returns_none(self):
        from screener.analytics import compute_trade_stats
        assert compute_trade_stats(pd.DataFrame()) is None

    def test_mixed_wins_losses(self):
        from screener.analytics import compute_trade_stats
        df = _closed_trades_df([
            {"actual_entry": 100, "actual_exit": 120, "shares": 10},  # +20%
            {"actual_entry": 100, "actual_exit": 90, "shares": 10},   # -10%
            {"actual_entry": 100, "actual_exit": 115, "shares": 10},  # +15%
        ])
        stats = compute_trade_stats(df)
        assert stats is not None
        assert stats["total_trades"] == 3
        assert stats["win_rate"] == pytest.approx(66.7, abs=0.1)
        assert stats["avg_win_pct"] > 0
        assert stats["avg_loss_pct"] < 0

    def test_all_wins(self):
        from screener.analytics import compute_trade_stats
        df = _closed_trades_df([
            {"actual_entry": 100, "actual_exit": 110, "shares": 10},
            {"actual_entry": 100, "actual_exit": 120, "shares": 10},
        ])
        stats = compute_trade_stats(df)
        assert stats["win_rate"] == 100.0
        assert stats["profit_factor"] == "∞"

    def test_all_losses(self):
        from screener.analytics import compute_trade_stats
        df = _closed_trades_df([
            {"actual_entry": 100, "actual_exit": 90, "shares": 10},
            {"actual_entry": 100, "actual_exit": 85, "shares": 10},
        ])
        stats = compute_trade_stats(df)
        assert stats["win_rate"] == 0.0
        assert stats["profit_factor"] == 0

    def test_equity_curve(self):
        from screener.analytics import compute_trade_stats
        df = _closed_trades_df([
            {"actual_entry": 100, "actual_exit": 110, "shares": 10, "closed_at": "2026-03-10"},
            {"actual_entry": 100, "actual_exit": 90, "shares": 10, "closed_at": "2026-03-15"},
        ])
        stats = compute_trade_stats(df)
        assert len(stats["equity_dates"]) == 2
        assert stats["equity_values"][0] == 100.0   # first trade: +$100
        assert stats["equity_values"][1] == 0.0      # net: +100 - 100 = 0


# ---------------------------------------------------------------------------
# Behavioral Patterns
# ---------------------------------------------------------------------------

class TestBehavioralPatterns:
    def test_empty_returns_none(self):
        from screener.analytics import compute_behavioral_patterns
        assert compute_behavioral_patterns(pd.DataFrame()) is None

    def test_conviction_scatter_excludes_zero(self):
        from screener.analytics import compute_behavioral_patterns
        df = _closed_trades_df([
            {"conviction": 0, "actual_entry": 100, "actual_exit": 110},
            {"conviction": 4, "actual_entry": 100, "actual_exit": 110},
        ])
        result = compute_behavioral_patterns(df)
        assert len(result["conviction_scatter"]) == 1
        assert result["conviction_scatter"][0]["conviction"] == 4

    def test_exit_reasons(self):
        from screener.analytics import compute_behavioral_patterns
        df = _closed_trades_df([
            {"exit_reason": "Hit Target"},
            {"exit_reason": "Hit Target"},
            {"exit_reason": "Hit Stop"},
        ])
        result = compute_behavioral_patterns(df)
        assert result["exit_reasons"]["Hit Target"] == 2
        assert result["exit_reasons"]["Hit Stop"] == 1


# ---------------------------------------------------------------------------
# Sparklines
# ---------------------------------------------------------------------------

class TestSparklines:
    def test_empty_returns_empty(self):
        from screener.analytics import compute_sparklines
        assert compute_sparklines(pd.DataFrame()) == {}

    def test_below_threshold_excluded(self):
        from screener.analytics import compute_sparklines
        df = _closed_trades_df([
            {"signal_source": "VR", "actual_entry": 100, "actual_exit": 110, "closed_at": "2026-03-10"},
            {"signal_source": "VR", "actual_entry": 100, "actual_exit": 105, "closed_at": "2026-03-15"},
        ])
        result = compute_sparklines(df)
        assert "VR" not in result  # only 2, need 3

    def test_above_threshold_included(self):
        from screener.analytics import compute_sparklines
        df = _closed_trades_df([
            {"signal_source": "VR", "actual_entry": 100, "actual_exit": 110, "closed_at": "2026-03-10"},
            {"signal_source": "VR", "actual_entry": 100, "actual_exit": 105, "closed_at": "2026-03-15"},
            {"signal_source": "VR", "actual_entry": 100, "actual_exit": 95, "closed_at": "2026-03-20"},
        ])
        result = compute_sparklines(df)
        assert "VR" in result
        assert len(result["VR"]) == 3
        assert result["VR"][0] == 10.0   # +10%
        assert result["VR"][2] == -5.0   # -5%
