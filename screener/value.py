import pandas as pd
import ta as ta_lib
import yfinance as yf
import streamlit as st
from screener.utils import load_metadata_cache, save_metadata_cache


def _safe_get(info: dict, key: str):
    val = info.get(key)
    return val if val not in (None, "N/A", "Infinity", float("inf")) else None


@st.cache_data(ttl=4 * 3600)
def fetch_value_data(tickers: list[str]) -> pd.DataFrame:
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
            high_52w = float(close.max())
            low_52w = float(close.min())
            drawdown = (current_price - high_52w) / high_52w * 100  # negative value
            return_52w = (current_price - float(close.iloc[0])) / float(close.iloc[0]) * 100

            rsi_series = ta_lib.momentum.rsi(close, window=14)
            rsi = (
                float(rsi_series.iloc[-1])
                if rsi_series is not None and not rsi_series.empty
                else None
            )

            volume = hist["Volume"]
            avg_vol_30 = (
                float(volume.iloc[-30:].mean())
                if len(volume) >= 30
                else float(volume.mean())
            )
            current_vol = float(volume.iloc[-1])
            volume_ratio = current_vol / avg_vol_30 if avg_vol_30 > 0 else None

            trailing_pe = _safe_get(info, "trailingPE")
            forward_pe = _safe_get(info, "forwardPE")
            pb = _safe_get(info, "priceToBook")
            peg = _safe_get(info, "pegRatio")
            div_yield_raw = _safe_get(info, "dividendYield")
            div_yield = float(div_yield_raw) * 100 if div_yield_raw is not None else None
            debt_equity = _safe_get(info, "debtToEquity")

            # Sector/industry — prefer live data, fall back to disk cache
            sector = info.get("sector") or None
            industry = info.get("industry") or None
            if sector or industry:
                metadata_cache[ticker] = {"sector": sector, "industry": industry}
            elif ticker in metadata_cache:
                cached = metadata_cache[ticker]
                sector = cached.get("sector")
                industry = cached.get("industry")

            # Recovery score 0-7
            score = sum([
                abs(drawdown) > 20,
                rsi is not None and rsi < 40,
                trailing_pe is not None and trailing_pe < 20,
                forward_pe is not None and forward_pe < 15,
                volume_ratio is not None and volume_ratio > 1.2,
                peg is not None and 0 < peg < 1,
                div_yield is not None and div_yield > 0,
            ])

            rows.append({
                "Ticker": ticker,
                "Sector": sector,
                "Industry": industry,
                "Price": round(current_price, 2),
                "52W High": round(high_52w, 2),
                "52W Low": round(low_52w, 2),
                "Drawdown %": round(drawdown, 1),
                "52W Return %": round(return_52w, 1),
                "RSI (14)": round(rsi, 1) if rsi is not None else None,
                "Trailing P/E": round(trailing_pe, 2) if trailing_pe is not None else None,
                "Forward P/E": round(forward_pe, 2) if forward_pe is not None else None,
                "P/B": round(pb, 2) if pb is not None else None,
                "PEG": round(peg, 2) if peg is not None else None,
                "Div Yield %": round(div_yield, 2) if div_yield is not None else None,
                "D/E": round(float(debt_equity), 2) if debt_equity is not None else None,
                "Avg Vol (30d)": avg_vol_30,
                "Current Vol": current_vol,
                "Vol Ratio": round(volume_ratio, 2) if volume_ratio is not None else None,
                "recovery_score": score,
            })

        except Exception:
            rows.append({
                "Ticker": ticker,
                "Sector": None,
                "Industry": None,
                "Price": None,
                "52W High": None,
                "52W Low": None,
                "Drawdown %": None,
                "52W Return %": None,
                "RSI (14)": None,
                "Trailing P/E": None,
                "Forward P/E": None,
                "P/B": None,
                "PEG": None,
                "Div Yield %": None,
                "D/E": None,
                "Avg Vol (30d)": None,
                "Current Vol": None,
                "Vol Ratio": None,
                "recovery_score": 0,
            })

    save_metadata_cache(metadata_cache)

    df = pd.DataFrame(rows)
    if "recovery_score" not in df.columns:
        df["recovery_score"] = 0
    try:
        return df.sort_values("recovery_score", ascending=False).reset_index(drop=True)
    except Exception:
        return df.reset_index(drop=True)
