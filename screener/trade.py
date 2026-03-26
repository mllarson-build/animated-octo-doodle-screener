"""
Trade Builder — single-ticker analysis producing a structured trade card
with price levels, entry/exit framework, position sizing, momentum signals,
hedge suggestions, and a fundamental snapshot.
"""

import os
import datetime
import pandas as pd
import ta as ta_lib
import yfinance as yf
import streamlit as st

from screener.utils import get_ticker_info, get_ticker_financials

PORTFOLIO_SIZE = 8_000.0
TRADE_LOG_PATH = "data/trade_log.csv"

TRADE_LOG_COLUMNS = [
    "timestamp", "ticker", "entry", "target", "stop", "rr",
    "setup_score", "upside_pct", "summary",
    "actual_entry", "actual_exit", "outcome_notes", "status",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_get(info: dict, key: str):
    val = info.get(key)
    return val if val not in (None, "N/A", "Infinity", float("inf")) else None


def _rev_earn_growth(fin: pd.DataFrame) -> tuple:
    """Return (revenue_growth_pct, earnings_growth_pct) or (None, None)."""
    if fin is None or fin.empty:
        return None, None

    rev_growth = earn_growth = None

    for label in ["Total Revenue", "TotalRevenue", "Revenue"]:
        if label in fin.index:
            row = fin.loc[label].sort_index(ascending=False)
            if len(row) >= 2:
                cur, prior = float(row.iloc[0]), float(row.iloc[1])
                if prior != 0:
                    rev_growth = (cur - prior) / abs(prior) * 100
            break

    for label in ["Net Income", "NetIncome", "Net Income Common Stockholders"]:
        if label in fin.index:
            row = fin.loc[label].sort_index(ascending=False)
            if len(row) >= 2:
                cur, prior = float(row.iloc[0]), float(row.iloc[1])
                if prior != 0:
                    earn_growth = (cur - prior) / abs(prior) * 100
            break

    return rev_growth, earn_growth


def _macd_bullish_crossover(close: pd.Series) -> bool:
    """True if MACD crossed above its signal line in the last 5 trading days."""
    try:
        macd_obj = ta_lib.trend.MACD(close)
        diff = macd_obj.macd_diff().dropna()
        if len(diff) < 2:
            return False
        recent = diff.iloc[-5:]
        for i in range(1, len(recent)):
            if recent.iloc[i - 1] <= 0 and recent.iloc[i] > 0:
                return True
        return False
    except Exception:
        return False


def _atr(high: pd.Series, low: pd.Series, close: pd.Series) -> float | None:
    """14-day Average True Range."""
    try:
        atr_series = ta_lib.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range()
        if atr_series is None or atr_series.empty:
            return None
        return float(atr_series.iloc[-1])
    except Exception:
        return None


def _find_protective_put(
    ticker: str, current_price: float, num_shares: int
) -> dict | None:
    """
    Find the nearest monthly put option 5-10% below current price.
    Returns a dict with cost info, or None if unavailable.
    """
    if num_shares <= 0:
        return None
    try:
        t = yf.Ticker(ticker)
        all_exps = t.options
        if not all_exps:
            return None
        exp = all_exps[0]
        puts = t.option_chain(exp).puts
        if puts.empty:
            return None

        lo, hi = current_price * 0.90, current_price * 0.95
        candidates = puts[(puts["strike"] >= lo) & (puts["strike"] <= hi)].copy()
        if candidates.empty:
            return None

        ideal = current_price * 0.925
        candidates["_dist"] = abs(candidates["strike"] - ideal)
        best = candidates.loc[candidates["_dist"].idxmin()]
        put_premium = float(best["lastPrice"])
        num_contracts = max(1, num_shares // 100)
        total_cost = put_premium * 100 * num_contracts
        position_cost = num_shares * current_price
        cost_pct = total_cost / position_cost * 100 if position_cost > 0 else 0
        breakeven = current_price + total_cost / num_shares

        return {
            "expiration": exp,
            "strike": round(float(best["strike"]), 2),
            "premium_per_share": round(put_premium, 2),
            "num_contracts": num_contracts,
            "total_cost": round(total_cost, 2),
            "cost_pct_of_position": round(cost_pct, 2),
            "breakeven_with_put": round(breakeven, 2),
        }
    except Exception:
        return None


def _hedge_etf(sector: str) -> tuple:
    """Return (etf_symbol, description) based on sector."""
    s = (sector or "").lower()
    if "tech" in s or "technology" in s or "software" in s or "semiconductor" in s:
        return "PSQ", "inverse QQQ (Nasdaq-100)"
    if "financ" in s or "bank" in s or "insurance" in s:
        return "SEF", "inverse Financials (XLF)"
    return "SH", "inverse SPY (S&P 500)"


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

@st.cache_data(ttl=4 * 3600)
def analyze_trade(ticker: str) -> dict:
    """
    Fetch and compute a full trade card for `ticker`.
    Returns a nested dict consumed by the Trade Builder tab in app.py.
    Raises ValueError if price history cannot be fetched.
    """
    t = yf.Ticker(ticker)
    hist = t.history(period="1y")
    if hist.empty:
        raise ValueError(f"No price history for {ticker}")

    close = hist["Close"]
    high_col = hist["High"]
    low_col = hist["Low"]
    current_price = float(close.iloc[-1])

    # ── Price levels ──────────────────────────────────────────────────────────
    high_52w = float(close.max())
    low_52w = float(close.min())
    support_20d = float(low_col.iloc[-20:].min())
    resistance_20d = float(high_col.iloc[-20:].max())
    ma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    ma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None

    # ── Entry / exit framework ────────────────────────────────────────────────
    entry = current_price
    entry_low = entry * 0.98            # 2% below for entry zone
    stop = entry * 0.92                 # 8% hard stop

    info = get_ticker_info(ticker)
    target_mean = _safe_get(info, "targetMeanPrice")
    if target_mean is not None and float(target_mean) > entry:
        target = float(target_mean)
        target_source = "analyst mean target"
    else:
        target = entry * 1.15
        target_source = "15% above entry (no analyst target)"

    risk_per_share = entry - stop       # always > 0 since stop = entry * 0.92
    reward_per_share = target - entry
    rr = reward_per_share / risk_per_share if risk_per_share > 0 else 0.0
    upside_pct = (target - current_price) / current_price * 100
    drawdown_from_high = (current_price - high_52w) / high_52w * 100

    # ── Position sizing ───────────────────────────────────────────────────────
    sizing = {}
    for label, risk_pct in [("conservative", 0.01), ("moderate", 0.02), ("aggressive", 0.03)]:
        dollar_risk = PORTFOLIO_SIZE * risk_pct
        shares = int(dollar_risk / risk_per_share) if risk_per_share > 0 else 0
        total_cost = shares * entry
        max_loss = shares * risk_per_share
        target_profit = shares * reward_per_share
        sizing[label] = {
            "risk_pct": risk_pct * 100,
            "shares": shares,
            "total_cost": round(total_cost, 2),
            "max_loss": round(max_loss, 2),
            "target_profit": round(target_profit, 2),
        }

    # ── Momentum & timing ─────────────────────────────────────────────────────
    rsi_series = ta_lib.momentum.rsi(close, window=14)
    rsi = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.empty else None

    macd_cross = _macd_bullish_crossover(close)
    atr = _atr(high_col, low_col, close)
    atr_pct = atr / current_price * 100 if atr is not None and current_price > 0 else None

    above_50ma = (current_price > ma_50) if ma_50 is not None else None
    above_200ma = (current_price > ma_200) if ma_200 is not None else None

    # Holding period heuristic
    if above_200ma and rsi is not None and rsi > 50:
        holding_period = "long (3-6 months)"
    elif rsi is not None and rsi < 40:
        holding_period = "medium (1-3 months)"
    else:
        holding_period = "short (2-4 weeks)"

    # ── RSI interpretation ────────────────────────────────────────────────────
    if rsi is None:
        rsi_label = "N/A"
    elif rsi < 30:
        rsi_label = "Oversold — strong value signal"
    elif rsi < 50:
        rsi_label = "Value zone — favorable entry"
    elif rsi < 70:
        rsi_label = "Neutral"
    else:
        rsi_label = "Overbought — wait for pullback"

    # ── Hedge suggestions ─────────────────────────────────────────────────────
    sector = info.get("sector", "") or ""
    hedge_symbol, hedge_desc = _hedge_etf(sector)
    hedge_alloc_10pct = round(PORTFOLIO_SIZE * 0.10, 2)

    moderate_shares = sizing["moderate"]["shares"]
    protective_put = None
    if sizing["moderate"]["total_cost"] > 500:
        protective_put = _find_protective_put(ticker, current_price, moderate_shares)

    # ── Fundamentals ──────────────────────────────────────────────────────────
    trailing_pe = _safe_get(info, "trailingPE")
    forward_pe = _safe_get(info, "forwardPE")
    peg = _safe_get(info, "pegRatio")
    pb = _safe_get(info, "priceToBook")
    div_yield_raw = _safe_get(info, "dividendYield")
    div_yield = float(div_yield_raw) * 100 if div_yield_raw is not None else None
    debt_equity = _safe_get(info, "debtToEquity")

    shares_short = _safe_get(info, "sharesShort")
    float_shares = _safe_get(info, "floatShares")
    short_pct = (
        shares_short / float_shares * 100
        if shares_short is not None and float_shares and float_shares > 0
        else None
    )

    fin = get_ticker_financials(ticker)
    rev_growth, earn_growth = _rev_earn_growth(fin)

    rec_mean = _safe_get(info, "recommendationMean")
    rec_key = info.get("recommendationKey", "") or ""
    num_analysts = _safe_get(info, "numberOfAnalystOpinions")

    if rec_mean is None:
        rec_label = "N/A"
    elif rec_mean <= 1.5:
        rec_label = "Strong Buy"
    elif rec_mean <= 2.5:
        rec_label = "Buy"
    elif rec_mean <= 3.5:
        rec_label = "Hold"
    elif rec_mean <= 4.5:
        rec_label = "Underperform"
    else:
        rec_label = "Sell"

    # ── Setup quality score 0-5 ───────────────────────────────────────────────
    setup_score = sum([
        rr >= 2.0,                                          # good risk/reward
        rsi is not None and 30 <= rsi <= 50,                # value entry zone
        above_200ma is True,                                # long-term uptrend
        upside_pct > 15,                                    # meaningful analyst upside
        rev_growth is not None and rev_growth > 10,         # revenue growing
    ])

    # ── Plain English summary ─────────────────────────────────────────────────
    direction = "down" if drawdown_from_high < 0 else "up"
    summary = (
        f"{ticker} is trading at ${current_price:.2f}, "
        f"{direction} {abs(drawdown_from_high):.1f}% from its 52-week high. "
        f"Analyst targets suggest {upside_pct:+.1f}% upside. "
        f"Risk/reward at current levels is {rr:.1f}:1. "
        f"Suggested entry between ${entry_low:.2f} and ${entry:.2f} with stop at ${stop:.2f}."
    )

    return {
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "price_levels": {
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),
            "support_20d": round(support_20d, 2),
            "resistance_20d": round(resistance_20d, 2),
            "ma_50": round(ma_50, 2) if ma_50 is not None else None,
            "ma_200": round(ma_200, 2) if ma_200 is not None else None,
        },
        "entry_exit": {
            "entry": round(entry, 2),
            "entry_low": round(entry_low, 2),
            "target": round(target, 2),
            "target_source": target_source,
            "stop": round(stop, 2),
            "rr": round(rr, 2),
            "good_setup": rr >= 2.0,
            "upside_pct": round(upside_pct, 1),
            "drawdown_from_high": round(drawdown_from_high, 1),
        },
        "sizing": sizing,
        "momentum": {
            "rsi": round(rsi, 1) if rsi is not None else None,
            "rsi_label": rsi_label,
            "macd_bullish_crossover": macd_cross,
            "above_50ma": above_50ma,
            "above_200ma": above_200ma,
            "atr": round(atr, 2) if atr is not None else None,
            "atr_pct": round(atr_pct, 2) if atr_pct is not None else None,
            "holding_period": holding_period,
        },
        "hedge": {
            "protective_put": protective_put,
            "hedge_etf_symbol": hedge_symbol,
            "hedge_etf_desc": hedge_desc,
            "hedge_alloc_10pct": hedge_alloc_10pct,
        },
        "fundamentals": {
            "trailing_pe": round(trailing_pe, 2) if trailing_pe is not None else None,
            "forward_pe": round(forward_pe, 2) if forward_pe is not None else None,
            "peg": round(peg, 2) if peg is not None else None,
            "pb": round(pb, 2) if pb is not None else None,
            "div_yield": round(div_yield, 2) if div_yield is not None else None,
            "debt_equity": round(float(debt_equity), 2) if debt_equity is not None else None,
            "rev_growth": round(rev_growth, 1) if rev_growth is not None else None,
            "earn_growth": round(earn_growth, 1) if earn_growth is not None else None,
            "short_pct": round(short_pct, 1) if short_pct is not None else None,
            "rec_mean": round(rec_mean, 2) if rec_mean is not None else None,
            "rec_label": rec_label,
            "num_analysts": int(num_analysts) if num_analysts is not None else None,
        },
        "setup_score": setup_score,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Trade log
# ---------------------------------------------------------------------------

def save_trade_idea(trade: dict) -> None:
    """Append a trade card to data/trade_log.csv."""
    ee = trade["entry_exit"]
    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ticker": trade["ticker"],
        "entry": ee["entry"],
        "target": ee["target"],
        "stop": ee["stop"],
        "rr": ee["rr"],
        "setup_score": trade["setup_score"],
        "upside_pct": ee["upside_pct"],
        "summary": trade["summary"],
        "actual_entry": "",
        "actual_exit": "",
        "outcome_notes": "",
        "status": "Open",
    }
    os.makedirs("data", exist_ok=True)
    df_new = pd.DataFrame([row])
    if os.path.exists(TRADE_LOG_PATH):
        existing = pd.read_csv(TRADE_LOG_PATH, dtype=str).fillna("")
        combined = pd.concat([existing, df_new], ignore_index=True)
    else:
        combined = df_new
    combined.to_csv(TRADE_LOG_PATH, index=False)


def load_trade_log() -> pd.DataFrame:
    """Load the trade log CSV. Returns an empty DataFrame if not found."""
    if not os.path.exists(TRADE_LOG_PATH):
        return pd.DataFrame(columns=TRADE_LOG_COLUMNS)
    try:
        return pd.read_csv(TRADE_LOG_PATH, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame(columns=TRADE_LOG_COLUMNS)


def save_trade_log(df: pd.DataFrame) -> None:
    """Write the full trade log DataFrame back to CSV (used for updates)."""
    try:
        os.makedirs("data", exist_ok=True)
        df.to_csv(TRADE_LOG_PATH, index=False)
    except Exception:
        pass
