import pandas as pd
import pandas_ta as ta
import yfinance as yf

ETF_LIST = ["SPY", "QQQ", "IWM", "XLF", "XLK", "XLE", "XLV", "XBI", "GLD", "TLT", "HYG"]


def _pct_change(series: pd.Series, periods: int) -> float | None:
    if len(series) < periods + 1:
        return None
    end = float(series.iloc[-1])
    start = float(series.iloc[-periods - 1])
    if start == 0:
        return None
    return (end - start) / start * 100


def fetch_etf_data() -> dict:
    """
    Returns:
      - etf_df: DataFrame with ETF performance metrics
      - macro: dict with VIX and TNX current levels and weekly changes
    """
    # --- ETF table ---
    rows = []
    for ticker in ETF_LIST:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="3mo")
            if hist.empty:
                raise ValueError("No history")

            close = hist["Close"]
            current_price = float(close.iloc[-1])

            ret_1d = _pct_change(close, 1)
            ret_1w = _pct_change(close, 5)
            ret_1m = _pct_change(close, 21)

            rsi_series = ta.rsi(close, length=14)
            rsi = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.empty else None

            vol = hist["Volume"]
            avg_vol_30 = float(vol.iloc[-30:].mean()) if len(vol) >= 30 else float(vol.mean())

            rows.append({
                "ETF": ticker,
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
                "Price": None,
                "1D %": None,
                "1W %": None,
                "1M %": None,
                "RSI (14)": None,
                "Avg Vol (30d)": None,
            })

    etf_df = pd.DataFrame(rows).sort_values("1M %", ascending=False).reset_index(drop=True)

    # --- Macro indicators ---
    macro = {"vix": None, "vix_1w_chg": None, "tnx": None, "tnx_1w_chg": None}
    for symbol, key in [("^VIX", "vix"), ("^TNX", "tnx")]:
        try:
            hist = yf.Ticker(symbol).history(period="2wk")
            if hist.empty:
                continue
            close = hist["Close"]
            macro[key] = round(float(close.iloc[-1]), 2)
            chg = _pct_change(close, 5)
            macro[f"{key}_1w_chg"] = round(chg, 2) if chg is not None else None
        except Exception:
            pass

    return {"etf_df": etf_df, "macro": macro}
