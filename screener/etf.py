import pandas as pd
import ta as ta_lib
import yfinance as yf
import streamlit as st
from screener.utils import (
    fetch_yf_chart,
    fetch_fred_10y_yield,
    file_cache_save_dict,
    file_cache_load_dict,
)

ETF_LIST = ["SPY", "QQQ", "IWM", "XLF", "XLK", "XLE", "XLV", "XBI", "GLD", "TLT", "HYG"]

ETF_NAMES = {
    "SPY": "SPDR S&P 500 ETF",
    "QQQ": "Invesco QQQ Trust",
    "IWM": "iShares Russell 2000",
    "XLF": "Financial Select SPDR",
    "XLK": "Technology Select SPDR",
    "XLE": "Energy Select SPDR",
    "XLV": "Health Care Select SPDR",
    "XBI": "SPDR S&P Biotech",
    "GLD": "SPDR Gold Shares",
    "TLT": "iShares 20+ Yr Treasury",
    "HYG": "iShares High Yield Corp Bond",
}


def _pct_change(series: pd.Series, periods: int) -> float | None:
    if len(series) < periods + 1:
        return None
    end = float(series.iloc[-1])
    start = float(series.iloc[-periods - 1])
    if start == 0:
        return None
    return (end - start) / start * 100


@st.cache_data(ttl=4 * 3600)
def fetch_etf_data() -> dict:
    """
    Returns:
      - etf_df: DataFrame with ETF performance metrics
      - macro: dict with VIX and TNX current levels and weekly changes

    ETF price history is fetched via the Yahoo Finance chart API directly
    (fetch_yf_chart), with yfinance as fallback.
    10Y Treasury yield comes from FRED DGS10 (no API key, more reliable than ^TNX).
    VIX still uses yfinance since there is no equivalent free public API.
    """
    rows = []
    for ticker in ETF_LIST:
        # Try direct Yahoo Finance chart API first, fall back to yfinance
        hist = fetch_yf_chart(ticker, period="3mo")
        if hist is None or hist.empty:
            try:
                raw = yf.Ticker(ticker).history(period="3mo")
                hist = raw[["Close", "Volume"]].copy() if not raw.empty else None
            except Exception:
                hist = None

        if hist is None or hist.empty:
            rows.append({
                "ETF": ticker,
                "Company": ETF_NAMES.get(ticker, ticker),
                "Price": None,
                "1D %": None,
                "1W %": None,
                "1M %": None,
                "RSI (14)": None,
                "Avg Vol (30d)": None,
            })
            continue

        try:
            close = hist["Close"]
            current_price = float(close.iloc[-1])

            ret_1d = _pct_change(close, 1)
            ret_1w = _pct_change(close, 5)
            ret_1m = _pct_change(close, 21)

            rsi_series = ta_lib.momentum.rsi(close, window=14)
            rsi = (
                float(rsi_series.iloc[-1])
                if rsi_series is not None and not rsi_series.empty
                else None
            )

            vol = hist["Volume"]
            avg_vol_30 = float(vol.iloc[-30:].mean()) if len(vol) >= 30 else float(vol.mean())

            rows.append({
                "ETF": ticker,
                "Company": ETF_NAMES.get(ticker, ticker),
                "Price": round(current_price, 2),
                "1D %": round(ret_1d, 2) if ret_1d is not None else None,
                "1W %": round(ret_1w, 2) if ret_1w is not None else None,
                "1M %": round(ret_1m, 2) if ret_1m is not None else None,
                "RSI (14)": round(rsi, 1) if rsi is not None else None,
                "Avg Vol (30d)": int(avg_vol_30),
            })
        except Exception:
            rows.append({
                "ETF": ticker,
                "Company": ETF_NAMES.get(ticker, ticker),
                "Price": None,
                "1D %": None,
                "1W %": None,
                "1M %": None,
                "RSI (14)": None,
                "Avg Vol (30d)": None,
            })

    etf_df = pd.DataFrame(rows)
    if not etf_df.empty and "1M %" in etf_df.columns:
        etf_df = etf_df.sort_values("1M %", ascending=False).reset_index(drop=True)

    # If all ETF rows failed, try file cache
    if etf_df.empty or etf_df["Price"].isna().all():
        cached_data, ts = file_cache_load_dict("etf_data")
        if cached_data is not None:
            st.warning(
                f"Showing cached ETF data from {ts} — live fetch temporarily unavailable."
            )
            return {
                "etf_df": pd.DataFrame(cached_data.get("etf_df", [])),
                "macro": cached_data.get("macro", {}),
            }

    # --- Macro indicators ---
    macro = {"vix": None, "vix_1w_chg": None, "tnx": None, "tnx_1w_chg": None}

    # 10Y yield from FRED — canonical source, no rate limit concerns
    tnx, tnx_chg = fetch_fred_10y_yield()
    macro["tnx"] = round(tnx, 2) if tnx is not None else None
    macro["tnx_1w_chg"] = round(tnx_chg, 2) if tnx_chg is not None else None

    # VIX from yfinance — no free equivalent public API
    try:
        vix_hist = yf.Ticker("^VIX").history(period="2wk")
        if not vix_hist.empty:
            vix_close = vix_hist["Close"]
            macro["vix"] = round(float(vix_close.iloc[-1]), 2)
            chg = _pct_change(vix_close, 5)
            macro["vix_1w_chg"] = round(chg, 2) if chg is not None else None
    except Exception:
        pass

    result = {"etf_df": etf_df, "macro": macro}

    # Persist to file cache for rate-limit fallback on next execution
    file_cache_save_dict("etf_data", {
        "etf_df": etf_df.to_dict("records"),
        "macro": macro,
    })

    return result
