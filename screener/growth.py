import pandas as pd
import yfinance as yf
import streamlit as st
from screener.utils import load_metadata_cache, save_metadata_cache


def _safe_get(info: dict, key: str):
    val = info.get(key)
    return val if val not in (None, "N/A", "Infinity", float("inf")) else None


def _revenue_growth(t: yf.Ticker) -> float | None:
    """YoY revenue growth from annual financials (most recent vs prior year)."""
    try:
        fin = t.financials  # columns are dates, rows are line items
        if fin is None or fin.empty:
            return None
        rev_row = None
        for label in ["Total Revenue", "TotalRevenue", "Revenue"]:
            if label in fin.index:
                rev_row = fin.loc[label]
                break
        if rev_row is None or len(rev_row) < 2:
            return None
        rev_row = rev_row.sort_index(ascending=False)
        current, prior = float(rev_row.iloc[0]), float(rev_row.iloc[1])
        if prior == 0:
            return None
        return (current - prior) / abs(prior) * 100
    except Exception:
        return None


def _earnings_growth(t: yf.Ticker) -> float | None:
    """YoY net income growth from annual financials."""
    try:
        fin = t.financials
        if fin is None or fin.empty:
            return None
        ni_row = None
        for label in ["Net Income", "NetIncome", "Net Income Common Stockholders"]:
            if label in fin.index:
                ni_row = fin.loc[label]
                break
        if ni_row is None or len(ni_row) < 2:
            return None
        ni_row = ni_row.sort_index(ascending=False)
        current, prior = float(ni_row.iloc[0]), float(ni_row.iloc[1])
        if prior == 0:
            return None
        return (current - prior) / abs(prior) * 100
    except Exception:
        return None


@st.cache_data(ttl=4 * 3600)
def fetch_growth_data(tickers: list[str]) -> pd.DataFrame:
    rows = []
    metadata_cache = load_metadata_cache()

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.info

            hist = t.history(period="1y")
            if hist.empty:
                raise ValueError("No price history")
            close = hist["Close"]
            current_price = float(close.iloc[-1])

            # 3-month momentum: 63 trading days
            if len(close) >= 63:
                momentum_pct = (
                    (current_price - float(close.iloc[-63])) / float(close.iloc[-63]) * 100
                )
            else:
                momentum_pct = (
                    (current_price - float(close.iloc[0])) / float(close.iloc[0]) * 100
                )

            rev_growth = _revenue_growth(t)
            earn_growth = _earnings_growth(t)

            target_mean = _safe_get(info, "targetMeanPrice")
            upside_pct = (
                (target_mean - current_price) / current_price * 100
                if target_mean is not None
                else None
            )

            shares_short = _safe_get(info, "sharesShort")
            float_shares = _safe_get(info, "floatShares")
            short_pct = (
                shares_short / float_shares * 100
                if shares_short is not None and float_shares and float_shares > 0
                else None
            )

            # Sector/industry — prefer live data, fall back to disk cache
            sector = info.get("sector") or None
            industry = info.get("industry") or None
            if sector or industry:
                metadata_cache[ticker] = {"sector": sector, "industry": industry}
            elif ticker in metadata_cache:
                cached = metadata_cache[ticker]
                sector = cached.get("sector")
                industry = cached.get("industry")

            score = sum([
                rev_growth is not None and rev_growth > 15,
                earn_growth is not None and earn_growth > 10,
                upside_pct is not None and upside_pct > 20,
                short_pct is None or short_pct < 10,
                momentum_pct > 0,
            ])

            rows.append({
                "Ticker": ticker,
                "Sector": sector,
                "Industry": industry,
                "Price": round(current_price, 2),
                "Rev Growth %": round(rev_growth, 1) if rev_growth is not None else None,
                "Earn Growth %": round(earn_growth, 1) if earn_growth is not None else None,
                "Analyst Target": round(target_mean, 2) if target_mean is not None else None,
                "Upside %": round(upside_pct, 1) if upside_pct is not None else None,
                "Short % Float": round(short_pct, 1) if short_pct is not None else None,
                "3M Momentum %": round(momentum_pct, 1),
                "growth_score": score,
            })

        except Exception:
            rows.append({
                "Ticker": ticker,
                "Sector": None,
                "Industry": None,
                "Price": None,
                "Rev Growth %": None,
                "Earn Growth %": None,
                "Analyst Target": None,
                "Upside %": None,
                "Short % Float": None,
                "3M Momentum %": None,
                "growth_score": 0,
            })

    save_metadata_cache(metadata_cache)

    df = pd.DataFrame(rows)
    return df.sort_values("growth_score", ascending=False).reset_index(drop=True)
