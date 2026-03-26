import pandas as pd
import yfinance as yf
import streamlit as st


@st.cache_data(ttl=4 * 3600)
def fetch_options_data(ticker: str) -> dict:
    """
    Returns a dict with:
      - current_price: float
      - expirations: list of the nearest 3 expiration date strings
      - chains: dict keyed by expiration date, each value is a dict with
                'calls' and 'puts' DataFrames
    """
    t = yf.Ticker(ticker)

    hist = t.history(period="5d")
    if hist.empty:
        raise ValueError(f"No price data for {ticker}")
    current_price = float(hist["Close"].iloc[-1])

    all_expirations = t.options
    if not all_expirations:
        raise ValueError(f"No options available for {ticker}")

    expirations = list(all_expirations[:3])

    chains = {}
    for exp in expirations:
        chain = t.option_chain(exp)
        chains[exp] = {
            "calls": _process_chain(chain.calls, "call", exp, current_price),
            "puts": _process_chain(chain.puts, "put", exp, current_price),
        }

    return {
        "current_price": current_price,
        "expirations": expirations,
        "chains": chains,
    }


def _process_chain(df: pd.DataFrame, option_type: str, expiration: str, current_price: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["Strike"] = df["strike"]
    out["Expiration"] = expiration
    out["Type"] = option_type
    out["Last Price"] = df["lastPrice"].round(2)
    out["IV"] = (df["impliedVolatility"] * 100).round(1)  # as percentage
    out["Volume"] = df["volume"].fillna(0).astype(int)
    out["Open Interest"] = df["openInterest"].fillna(0).astype(int)

    # Vol/OI ratio — guard against divide-by-zero
    out["Vol/OI"] = (
        out["Volume"] / out["Open Interest"].replace(0, pd.NA)
    ).round(2)

    # Flag unusual activity: all three conditions must be true
    near_money = (out["Strike"] >= current_price * 0.90) & (out["Strike"] <= current_price * 1.10)
    unusual = (
        (out["Vol/OI"] > 1.5) &
        near_money &
        (out["Volume"] > 100)
    )
    out["Flag"] = unusual

    return out.reset_index(drop=True)
