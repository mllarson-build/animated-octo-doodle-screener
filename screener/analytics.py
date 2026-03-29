"""
My Edge analytics — signal scorecard, trade stats, behavioral patterns, sparklines.

All functions take a trades DataFrame (from db.load_trades) and return
display-ready data. No side effects, no API calls except SPY benchmark.
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

MIN_TRADES_FOR_SCORECARD = 5
MIN_TRADES_FOR_SPARKLINE = 3


# ---------------------------------------------------------------------------
# Signal Scorecard
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def _fetch_spy_series(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch SPY daily close for the full date range. One API call, reused per trade."""
    try:
        spy = yf.download("SPY", start=start_date, end=end_date, auto_adjust=True, progress=False)
        if spy.empty:
            return pd.DataFrame()
        return spy[["Close"]].copy()
    except Exception:
        return pd.DataFrame()


def compute_signal_scorecard(closed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group closed trades by signal_source. For each source compute:
    - trades: count
    - win_rate: % of winning trades
    - avg_return_pct: mean return %
    - edge_vs_spy: avg return minus SPY return over same holding period

    Returns a display-ready DataFrame. Sources with < MIN_TRADES get a note.
    """
    if closed_df.empty:
        return pd.DataFrame()

    df = closed_df.copy()

    # Ensure numeric columns
    for col in ("actual_entry", "actual_exit", "shares"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["return_pct"] = (
        (df["actual_exit"] - df["actual_entry"])
        / df["actual_entry"].replace(0, np.nan)
        * 100
    )
    df["is_win"] = df["return_pct"] > 0

    # Parse dates for SPY comparison
    df["open_date"] = pd.to_datetime(
        df["executed_at"].where(df["executed_at"].notna() & (df["executed_at"] != ""),
                                df["timestamp"]),
        errors="coerce",
    )
    df["close_date"] = pd.to_datetime(df["closed_at"], errors="coerce")

    # Fetch SPY once for full range
    spy_data = pd.DataFrame()
    valid_dates = df.dropna(subset=["open_date", "close_date"])
    if not valid_dates.empty:
        min_date = valid_dates["open_date"].min().strftime("%Y-%m-%d")
        max_date = (valid_dates["close_date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        spy_data = _fetch_spy_series(min_date, max_date)

    # Compute per-trade SPY return
    def _spy_return(row):
        if spy_data.empty or pd.isna(row.get("open_date")) or pd.isna(row.get("close_date")):
            return np.nan
        try:
            spy_close = spy_data["Close"]
            # Handle MultiIndex columns from yfinance
            if isinstance(spy_close, pd.DataFrame):
                spy_close = spy_close.iloc[:, 0]
            open_prices = spy_close.loc[:row["open_date"]].dropna()
            close_prices = spy_close.loc[:row["close_date"]].dropna()
            if open_prices.empty or close_prices.empty:
                return np.nan
            return (close_prices.iloc[-1] / open_prices.iloc[-1] - 1) * 100
        except Exception:
            return np.nan

    df["spy_return_pct"] = df.apply(_spy_return, axis=1)
    df["edge_vs_spy"] = df["return_pct"] - df["spy_return_pct"]

    # Group by signal source
    rows = []
    for source, grp in df.groupby("signal_source"):
        n = len(grp)
        if n < MIN_TRADES_FOR_SCORECARD:
            rows.append({
                "Signal Source": source,
                "Trades": n,
                "Win Rate": f"Need {MIN_TRADES_FOR_SCORECARD}+",
                "Avg Return %": f"Need {MIN_TRADES_FOR_SCORECARD}+",
                "Edge vs SPY": f"Need {MIN_TRADES_FOR_SCORECARD}+",
            })
        else:
            win_rate = grp["is_win"].sum() / n * 100
            avg_ret = grp["return_pct"].mean()
            edge = grp["edge_vs_spy"].mean()
            rows.append({
                "Signal Source": source,
                "Trades": n,
                "Win Rate": f"{win_rate:.1f}%",
                "Avg Return %": f"{avg_ret:+.2f}%",
                "Edge vs SPY": f"{edge:+.2f}%" if not np.isnan(edge) else "N/A",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Closed Trade Stats
# ---------------------------------------------------------------------------

def compute_trade_stats(closed_df: pd.DataFrame) -> dict:
    """
    Compute aggregate stats for closed trades:
    - win_rate, avg_win_pct, avg_loss_pct, profit_factor, expectancy
    - avg_hold_winners_days, avg_hold_losers_days
    - equity_curve: list of (date, cumulative_pnl) for plotting
    """
    if closed_df.empty:
        return None

    df = closed_df.copy()
    for col in ("actual_entry", "actual_exit", "shares"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["return_pct"] = (
        (df["actual_exit"] - df["actual_entry"])
        / df["actual_entry"].replace(0, np.nan)
        * 100
    )
    df["pnl"] = (df["actual_exit"] - df["actual_entry"]) * df["shares"].fillna(0)

    valid = df.dropna(subset=["return_pct"])
    if valid.empty:
        return None

    wins = valid[valid["return_pct"] > 0]
    losses = valid[valid["return_pct"] <= 0]

    n = len(valid)
    win_rate = len(wins) / n * 100 if n > 0 else 0
    avg_win_pct = wins["return_pct"].mean() if not wins.empty else 0
    avg_loss_pct = losses["return_pct"].mean() if not losses.empty else 0

    gross_wins = wins["pnl"].sum() if not wins.empty else 0
    gross_losses = abs(losses["pnl"].sum()) if not losses.empty else 0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf") if gross_wins > 0 else 0

    loss_rate = 100 - win_rate
    expectancy = (win_rate / 100 * avg_win_pct) + (loss_rate / 100 * avg_loss_pct)

    # Hold durations
    df["open_dt"] = pd.to_datetime(
        df["executed_at"].where(df["executed_at"].notna() & (df["executed_at"] != ""),
                                df["timestamp"]),
        errors="coerce",
    )
    df["close_dt"] = pd.to_datetime(df["closed_at"], errors="coerce")
    df["hold_days"] = (df["close_dt"] - df["open_dt"]).dt.days

    avg_hold_winners = wins.merge(df[["hold_days"]], left_index=True, right_index=True, how="left")["hold_days"].mean()
    avg_hold_losers = losses.merge(df[["hold_days"]], left_index=True, right_index=True, how="left")["hold_days"].mean()

    # Equity curve: cumulative realized P&L by close date
    eq = df.dropna(subset=["close_dt", "pnl"]).sort_values("close_dt")
    eq_dates = eq["close_dt"].dt.strftime("%Y-%m-%d").tolist()
    eq_values = eq["pnl"].cumsum().tolist()

    return {
        "total_trades": n,
        "win_rate": round(win_rate, 1),
        "avg_win_pct": round(avg_win_pct, 2),
        "avg_loss_pct": round(avg_loss_pct, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "∞",
        "expectancy": round(expectancy, 2),
        "avg_hold_winners": round(avg_hold_winners, 1) if pd.notna(avg_hold_winners) else None,
        "avg_hold_losers": round(avg_hold_losers, 1) if pd.notna(avg_hold_losers) else None,
        "equity_dates": eq_dates,
        "equity_values": eq_values,
    }


# ---------------------------------------------------------------------------
# Behavioral Patterns
# ---------------------------------------------------------------------------

def compute_behavioral_patterns(closed_df: pd.DataFrame) -> dict:
    """
    Returns data for behavioral analytics:
    - conviction_scatter: [(conviction, return_pct, ticker)] for scatter plot
    - exit_reasons: {reason: count} for pie chart
    - hold_by_reason: {reason: avg_hold_days} for bar chart
    - day_of_week: {day_name: avg_return_pct} for day-of-week analysis
    """
    if closed_df.empty:
        return None

    df = closed_df.copy()
    for col in ("actual_entry", "actual_exit", "conviction"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["return_pct"] = (
        (df["actual_exit"] - df["actual_entry"])
        / df["actual_entry"].replace(0, np.nan)
        * 100
    )

    # Conviction scatter — exclude conviction=0 (unscored/imported)
    scored = df[(df["conviction"] > 0) & df["return_pct"].notna()]
    conviction_scatter = []
    for _, row in scored.iterrows():
        conviction_scatter.append({
            "conviction": int(row["conviction"]),
            "return_pct": round(row["return_pct"], 2),
            "ticker": row["ticker"],
            "is_win": row["return_pct"] > 0,
        })

    # Exit reason distribution
    exit_reasons = {}
    if "exit_reason" in df.columns:
        reasons = df[df["exit_reason"].notna() & (df["exit_reason"] != "")]
        exit_reasons = reasons["exit_reason"].value_counts().to_dict()

    # Hold duration by exit reason
    df["open_dt"] = pd.to_datetime(
        df["executed_at"].where(df["executed_at"].notna() & (df["executed_at"] != ""),
                                df["timestamp"]),
        errors="coerce",
    )
    df["close_dt"] = pd.to_datetime(df["closed_at"], errors="coerce")
    df["hold_days"] = (df["close_dt"] - df["open_dt"]).dt.days

    hold_by_reason = {}
    if "exit_reason" in df.columns:
        for reason, grp in df[df["exit_reason"].notna() & (df["exit_reason"] != "")].groupby("exit_reason"):
            avg_days = grp["hold_days"].mean()
            if pd.notna(avg_days):
                hold_by_reason[reason] = round(avg_days, 1)

    # Day-of-week patterns (based on trade open date)
    day_of_week = {}
    with_day = df.dropna(subset=["open_dt", "return_pct"])
    if not with_day.empty:
        with_day["day_name"] = with_day["open_dt"].dt.day_name()
        for day, grp in with_day.groupby("day_name"):
            day_of_week[day] = round(grp["return_pct"].mean(), 2)

    return {
        "conviction_scatter": conviction_scatter,
        "exit_reasons": exit_reasons,
        "hold_by_reason": hold_by_reason,
        "day_of_week": day_of_week,
    }


# ---------------------------------------------------------------------------
# Signal Strength Sparklines
# ---------------------------------------------------------------------------

def compute_sparklines(closed_df: pd.DataFrame) -> dict:
    """
    For each signal_source with MIN_TRADES_FOR_SPARKLINE+ closed trades,
    return a list of sequential returns (chronological by close date).

    Returns: {signal_source: [return_pct, ...]}
    """
    if closed_df.empty:
        return {}

    df = closed_df.copy()
    for col in ("actual_entry", "actual_exit"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["return_pct"] = (
        (df["actual_exit"] - df["actual_entry"])
        / df["actual_entry"].replace(0, np.nan)
        * 100
    )
    df["close_dt"] = pd.to_datetime(df["closed_at"], errors="coerce")

    result = {}
    for source, grp in df.groupby("signal_source"):
        valid = grp.dropna(subset=["return_pct", "close_dt"]).sort_values("close_dt")
        if len(valid) >= MIN_TRADES_FOR_SPARKLINE:
            result[source] = [round(r, 2) for r in valid["return_pct"].tolist()]

    return result
