import pandas as pd
import pandas_ta as ta
import yfinance as yf


def _safe_get(info: dict, key: str):
    val = info.get(key)
    return val if val not in (None, "N/A", "Infinity", float("inf")) else None


def fetch_value_data(tickers: list[str]) -> pd.DataFrame:
    rows = []

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

            rsi_series = ta.rsi(close, length=14)
            rsi = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.empty else None

            volume = hist["Volume"]
            avg_vol_30 = float(volume.iloc[-30:].mean()) if len(volume) >= 30 else float(volume.mean())
            current_vol = float(volume.iloc[-1])
            volume_ratio = current_vol / avg_vol_30 if avg_vol_30 > 0 else None

            trailing_pe = _safe_get(info, "trailingPE")
            forward_pe = _safe_get(info, "forwardPE")
            pb = _safe_get(info, "priceToBook")

            # Score: count how many conditions are true
            score = sum([
                abs(drawdown) > 20,
                rsi is not None and rsi < 40,
                trailing_pe is not None and trailing_pe < 20,
                forward_pe is not None and forward_pe < 15,
                volume_ratio is not None and volume_ratio > 1.2,
            ])

            rows.append({
                "Ticker": ticker,
                "Price": round(current_price, 2),
                "52W High": round(high_52w, 2),
                "52W Low": round(low_52w, 2),
                "Drawdown %": round(drawdown, 1),
                "RSI (14)": round(rsi, 1) if rsi is not None else None,
                "Trailing P/E": round(trailing_pe, 1) if trailing_pe is not None else None,
                "Forward P/E": round(forward_pe, 1) if forward_pe is not None else None,
                "P/B": round(pb, 2) if pb is not None else None,
                "Vol Ratio": round(volume_ratio, 2) if volume_ratio is not None else None,
                "recovery_score": score,
            })

        except Exception as e:
            rows.append({
                "Ticker": ticker,
                "Price": None,
                "52W High": None,
                "52W Low": None,
                "Drawdown %": None,
                "RSI (14)": None,
                "Trailing P/E": None,
                "Forward P/E": None,
                "P/B": None,
                "Vol Ratio": None,
                "recovery_score": 0,
                "_error": str(e),
            })

    df = pd.DataFrame(rows)
    if "_error" in df.columns:
        df = df.drop(columns=["_error"])
    return df.sort_values("recovery_score", ascending=False).reset_index(drop=True)
