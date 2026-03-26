"""
Backtesting engine for screening signals.

All indicators are computed using only backward-looking rolling data —
no lookahead bias. Entry = close on signal date, exit = close after
holding_days trading days.

Limitations (also shown in UI):
- No slippage, commissions, or taxes modeled
- Survivorship bias: only tests tickers you supply (not delisted stocks)
- Growth Momentum uses a price-based proxy (momentum + MA) since
  historical revenue/analyst data requires a paid data provider
- Past performance does not guarantee future results
"""

import numpy as np
import pandas as pd
import streamlit as st
import ta as ta_lib
import yfinance as yf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOLDING_PERIOD_DAYS = {
    "2 weeks":  10,
    "1 month":  21,
    "3 months": 63,
    "6 months": 126,
}

LOOKBACK_PERIODS = {
    "1 year":  "1y",
    "2 years": "2y",
    "3 years": "3y",
    "5 years": "5y",
}

SIGNAL_DESCRIPTIONS = {
    "Value Recovery":    "Drawdown > 20% from 52W high  AND  RSI(14) < 40",
    "Growth Momentum":   "3-month price momentum > 10%  AND  price above 50-day MA  (price proxy — see disclaimer)",
    "RSI Oversold":      "RSI(14) below threshold (default 35)",
    "Near 52W Low":      "Price within 15% above rolling 52-week low",
    "Golden Cross":      "50-day MA crosses above 200-day MA",
    "Custom":            "User-defined RSI threshold  AND  drawdown threshold",
}

# Minimum trading days needed per signal (to ensure enough history for indicators)
SIGNAL_MIN_BARS = {
    "Value Recovery":  252 + 14,
    "Growth Momentum": 63  + 50,
    "RSI Oversold":    14  + 1,
    "Near 52W Low":    252 + 1,
    "Golden Cross":    200 + 1,
    "Custom":          252 + 14,
}

# Minimum gap between consecutive signals for the same ticker (trading days)
# Prevents counting the same "episode" (e.g. 20 consecutive oversold days) as 20 trades
SIGNAL_COOLDOWN_DAYS = 5


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

@st.cache_data(ttl=24 * 3600)
def fetch_historical_data(tickers: tuple, period: str) -> dict:
    """
    Bulk-fetch daily adjusted OHLCV for all requested tickers plus SPY.
    Always includes SPY for buy-and-hold comparison.

    Args:
        tickers: tuple of uppercase ticker strings (tuple is hashable for caching)
        period:  yfinance period string, e.g. "2y"

    Returns:
        dict mapping ticker -> DataFrame(Close, High, Low, Open, Volume)
        Missing / failed tickers are omitted silently.
    """
    all_tickers = sorted(set(tickers) | {"SPY"})

    try:
        if len(all_tickers) == 1:
            raw = yf.download(
                all_tickers[0], period=period, auto_adjust=True, progress=False
            )
            if raw.empty:
                return {}
            return {all_tickers[0]: raw.dropna(subset=["Close"])}

        raw = yf.download(
            all_tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
        if raw.empty:
            return {}

        result = {}
        lvl0 = set(raw.columns.get_level_values(0))
        for t in all_tickers:
            if t not in lvl0:
                continue
            try:
                df = raw[t].dropna(subset=["Close"])
                if not df.empty:
                    result[t] = df
            except Exception:
                pass
        return result

    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------

def _compute_indicators(close: pd.Series) -> dict:
    """
    Compute all indicators needed for signal detection.
    Every calculation uses only backward-looking data (rolling windows).

    Returns a dict of pd.Series aligned to the close index.
    NaN values in the warmup period are expected and are handled by
    the downstream signal masks.
    """
    ind = {}

    # RSI(14) — uses only the 14 most-recent returns at each point
    try:
        ind["rsi"] = ta_lib.momentum.rsi(close, window=14)
    except Exception:
        ind["rsi"] = pd.Series(np.nan, index=close.index)

    # Moving averages
    ind["ma50"]  = close.rolling(50,  min_periods=50).mean()
    ind["ma200"] = close.rolling(200, min_periods=200).mean()

    # Rolling 252-day (≈1 year) high and low
    ind["high_252"] = close.rolling(252, min_periods=20).max()
    ind["low_252"]  = close.rolling(252, min_periods=20).min()

    # Drawdown from rolling 252-day high (negative percentage)
    ind["drawdown"] = (close - ind["high_252"]) / ind["high_252"] * 100

    # Percentage above rolling 252-day low
    ind["above_low_pct"] = (close - ind["low_252"]) / ind["low_252"] * 100

    # 3-month (63-bar) price momentum percentage
    ind["momentum_3m"] = close.pct_change(63) * 100

    return ind


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------

def detect_signal(
    close: pd.Series,
    signal_type: str,
    custom_rsi: float = 35.0,
    custom_drawdown: float = 20.0,
) -> pd.Series:
    """
    Return a boolean Series where True = signal triggered on that date.
    Each condition uses only data available up to (and including) that date.

    Args:
        close:            daily adjusted close price series
        signal_type:      key from SIGNAL_DESCRIPTIONS
        custom_rsi:       RSI threshold for RSI Oversold / Custom signals
        custom_drawdown:  drawdown threshold (positive %) for Custom / Value Recovery
    """
    ind = _compute_indicators(close)

    if signal_type == "Value Recovery":
        mask = (ind["drawdown"] < -custom_drawdown) & (ind["rsi"] < 40)

    elif signal_type == "Growth Momentum":
        # Price proxy: stock showing upward momentum and above its medium-term trend
        mask = (ind["momentum_3m"] > 10) & (close > ind["ma50"])

    elif signal_type == "RSI Oversold":
        mask = ind["rsi"] < custom_rsi

    elif signal_type == "Near 52W Low":
        mask = ind["above_low_pct"] < 15

    elif signal_type == "Golden Cross":
        ma50  = ind["ma50"]
        ma200 = ind["ma200"]
        # Crossover: today above, yesterday at-or-below
        mask = (ma50 > ma200) & (ma50.shift(1) <= ma200.shift(1))

    elif signal_type == "Custom":
        mask = (ind["drawdown"] < -custom_drawdown) & (ind["rsi"] < custom_rsi)

    else:
        return pd.Series(False, index=close.index)

    return mask.fillna(False)


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    tickers: list,
    signal_type: str,
    holding_days: int,
    period: str,
    custom_rsi: float = 35.0,
    custom_drawdown: float = 20.0,
) -> dict:
    """
    Run the full backtest across all tickers.

    Returns a dict with:
        "trades"  — DataFrame of individual trade records (or absent on error)
        "metrics" — dict of aggregate performance metrics
        "spy_series" — normalized SPY price series for the cumulative chart
        "error"   — error message string (only present on failure)
    """
    hist = fetch_historical_data(tuple(sorted(tickers)), period)
    if not hist:
        return {"error": "Could not fetch historical data. Check ticker symbols and try again."}

    min_bars = SIGNAL_MIN_BARS.get(signal_type, 252) + holding_days
    trades = []

    for ticker in tickers:
        df = hist.get(ticker)
        if df is None or len(df) < min_bars:
            continue

        close = df["Close"]

        signal_mask = detect_signal(close, signal_type, custom_rsi, custom_drawdown)
        signal_dates = close.index[signal_mask]

        last_entry_idx = -SIGNAL_COOLDOWN_DAYS  # allow first signal immediately

        for signal_date in signal_dates:
            # Enforce cooldown between consecutive signals for the same ticker
            curr_idx = close.index.get_loc(signal_date)
            if curr_idx - last_entry_idx < SIGNAL_COOLDOWN_DAYS:
                continue

            # Ensure we have enough bars after the signal to hold
            future = close.iloc[curr_idx:]
            if len(future) < holding_days + 1:
                continue  # signal too close to end of data

            entry_price = float(future.iloc[0])
            exit_price  = float(future.iloc[holding_days])
            exit_date   = future.index[holding_days]
            ret         = (exit_price - entry_price) / entry_price * 100

            trades.append({
                "Ticker":       ticker,
                "Entry Date":   signal_date.date(),
                "Exit Date":    exit_date.date(),
                "Entry Price":  round(entry_price, 2),
                "Exit Price":   round(exit_price, 2),
                "Return %":     round(ret, 2),
                "Win":          ret > 0,
            })
            last_entry_idx = curr_idx

    if not trades:
        return {
            "error": (
                "No signals found in the selected period. "
                "Try a longer lookback, a different signal, or add more tickers."
            )
        }

    trades_df = pd.DataFrame(trades).sort_values("Entry Date").reset_index(drop=True)
    returns   = trades_df["Return %"]

    # ── SPY buy-and-hold metrics ──────────────────────────────────────────────
    spy_return = None
    spy_series = pd.Series(dtype=float)

    if "SPY" in hist:
        spy_close  = hist["SPY"]["Close"]
        spy_return = round(
            (float(spy_close.iloc[-1]) - float(spy_close.iloc[0]))
            / float(spy_close.iloc[0]) * 100,
            2,
        )
        # Normalize to 100 at start
        spy_series = (spy_close / float(spy_close.iloc[0])) * 100

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    wins   = returns[returns >  0]
    losses = returns[returns <= 0]

    # Max consecutive losses
    max_consec_loss = cur_loss = 0
    for w in trades_df["Win"]:
        if not w:
            cur_loss += 1
            max_consec_loss = max(max_consec_loss, cur_loss)
        else:
            cur_loss = 0

    # Profit factor: gross wins / gross losses (absolute)
    gross_wins   = wins.sum()
    gross_losses = abs(losses.sum())
    profit_factor = round(gross_wins / gross_losses, 2) if gross_losses > 0 else None

    # Sharpe-like ratio (mean / std dev of returns, not annualised)
    std = returns.std()
    sharpe_like = round(float(returns.mean()) / float(std), 3) if std > 0 else None

    # ── Cumulative strategy return series (for chart) ─────────────────────────
    # Each trade multiplies cumulative equity by (1 + ret/100); starts at 100
    cum_equity = [100.0]
    cum_dates  = [trades_df["Entry Date"].iloc[0]]
    running    = 100.0
    for _, row in trades_df.iterrows():
        running *= (1 + row["Return %"] / 100)
        cum_equity.append(round(running, 4))
        cum_dates.append(row["Exit Date"])

    metrics = {
        "total_signals":    len(trades_df),
        "win_rate":         round(float(returns.gt(0).mean()) * 100, 1),
        "avg_return":       round(float(returns.mean()), 2),
        "median_return":    round(float(returns.median()), 2),
        "avg_win":          round(float(wins.mean()),   2) if not wins.empty   else 0.0,
        "avg_loss":         round(float(losses.mean()), 2) if not losses.empty else 0.0,
        "profit_factor":    profit_factor,
        "best_trade":       round(float(returns.max()), 2),
        "worst_trade":      round(float(returns.min()), 2),
        "max_consec_losses": max_consec_loss,
        "sharpe_like":      sharpe_like,
        "spy_return":       spy_return,
        "cum_equity":       cum_equity,
        "cum_dates":        cum_dates,
    }

    return {
        "trades":     trades_df,
        "metrics":    metrics,
        "spy_series": spy_series,
    }
