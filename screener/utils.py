import os
import json
import time
import random
import datetime
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

METADATA_CACHE_PATH = "data/ticker_metadata.csv"
FILE_CACHE_DIR = "data/cache"

# ---------------------------------------------------------------------------
# Request rotation — query1 and query2 are the two Yahoo Finance API endpoints;
# finance.yahoo.com is the browser website and does not serve JSON API responses.
# ---------------------------------------------------------------------------
YF_BASE_URLS = [
    "https://query1.finance.yahoo.com",
    "https://query2.finance.yahoo.com",
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Safari/605.1.15",
]

_url_index = 0


def next_yf_base_url() -> str:
    global _url_index
    url = YF_BASE_URLS[_url_index % len(YF_BASE_URLS)]
    _url_index += 1
    return url


def random_user_agent() -> str:
    return random.choice(USER_AGENTS)


def yf_headers() -> dict:
    return {
        "User-Agent": random_user_agent(),
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }


# ---------------------------------------------------------------------------
# File cache — secondary layer for rate-limit fallback
# ---------------------------------------------------------------------------

def file_cache_save(key: str, df: pd.DataFrame) -> None:
    """Persist a DataFrame to the file cache directory."""
    try:
        os.makedirs(FILE_CACHE_DIR, exist_ok=True)
        df.to_pickle(os.path.join(FILE_CACHE_DIR, f"{key}.pkl"))
        with open(os.path.join(FILE_CACHE_DIR, f"{key}.meta.json"), "w") as f:
            json.dump({"timestamp": datetime.datetime.now().isoformat()}, f)
    except Exception:
        pass


def file_cache_load(key: str) -> tuple:
    """
    Load a DataFrame from the file cache.
    Returns (df, timestamp_str) or (None, None).
    """
    data_path = os.path.join(FILE_CACHE_DIR, f"{key}.pkl")
    meta_path = os.path.join(FILE_CACHE_DIR, f"{key}.meta.json")
    if not os.path.exists(data_path):
        return None, None
    try:
        df = pd.read_pickle(data_path)
        timestamp = "unknown"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                timestamp = json.load(f).get("timestamp", "unknown")
        return df, timestamp
    except Exception:
        return None, None


def file_cache_save_dict(key: str, data: dict) -> None:
    """Persist a JSON-serializable dict to the file cache."""
    try:
        os.makedirs(FILE_CACHE_DIR, exist_ok=True)
        path = os.path.join(FILE_CACHE_DIR, f"{key}.json")
        with open(path, "w") as f:
            json.dump(
                {"timestamp": datetime.datetime.now().isoformat(), "data": data},
                f,
                default=str,
            )
    except Exception:
        pass


def file_cache_load_dict(key: str) -> tuple:
    """
    Load a dict from the file cache.
    Returns (data, timestamp_str) or (None, None).
    """
    path = os.path.join(FILE_CACHE_DIR, f"{key}.json")
    if not os.path.exists(path):
        return None, None
    try:
        with open(path) as f:
            payload = json.load(f)
        return payload.get("data"), payload.get("timestamp", "unknown")
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Cached per-ticker yfinance helpers (6h TTL — fundamentals rarely change intraday)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=6 * 3600)
def get_ticker_info(ticker: str) -> dict:
    """Fetch and cache yfinance .info for a single ticker."""
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=6 * 3600)
def get_ticker_financials(ticker: str) -> pd.DataFrame:
    """Fetch and cache yfinance .financials for a single ticker."""
    try:
        fin = yf.Ticker(ticker).financials
        return fin if fin is not None and not fin.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Alternative data sources
# ---------------------------------------------------------------------------

FRED_DGS10_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"


def fetch_fred_10y_yield() -> tuple:
    """
    Fetch current 10Y Treasury yield and 1-week change from FRED (no API key required).
    Returns (current_yield_float, pct_change_1w_float) or (None, None) on failure.
    Uses FRED DGS10 series — canonical source, more reliable than yfinance ^TNX.
    """
    try:
        r = requests.get(
            FRED_DGS10_URL,
            timeout=15,
            headers={"User-Agent": random_user_agent()},
        )
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        df.columns = ["date", "value"]
        df = df[df["value"] != "."].copy()
        df["value"] = df["value"].astype(float)
        if df.empty:
            return None, None
        current = float(df["value"].iloc[-1])
        chg = None
        if len(df) >= 6:
            prior = float(df["value"].iloc[-6])
            if prior != 0:
                chg = (current - prior) / abs(prior) * 100
        return current, chg
    except Exception:
        return None, None


def fetch_alpha_vantage_quote(ticker: str) -> dict | None:
    """
    Fetch EOD quote from Alpha Vantage free tier (25 calls/day).
    Requires ALPHA_VANTAGE_API_KEY environment variable; returns None if key not set.
    Set the key in a .env file: ALPHA_VANTAGE_API_KEY=your_key_here
    """
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return None
    try:
        url = (
            "https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}"
        )
        r = requests.get(url, timeout=15, headers={"User-Agent": random_user_agent()})
        r.raise_for_status()
        quote = r.json().get("Global Quote", {})
        if not quote or not quote.get("05. price"):
            return None
        return {
            "price": float(quote["05. price"]),
            "volume": int(float(quote.get("06. volume", 0))),
        }
    except Exception:
        return None


def fetch_yf_chart(ticker: str, period: str = "3mo") -> pd.DataFrame | None:
    """
    Fetch OHLCV data from Yahoo Finance chart API via direct HTTP (no yfinance).
    Rotates between query1 and query2 base URLs with randomised User-Agent headers.
    Returns a DataFrame with Close and Volume columns indexed by date, or None on failure.
    """
    base = next_yf_base_url()
    url = f"{base}/v8/finance/chart/{ticker}"
    params = {"interval": "1d", "range": period}
    try:
        r = requests.get(url, params=params, headers=yf_headers(), timeout=15)
        r.raise_for_status()
        data = r.json()
        result = data["chart"]["result"][0]
        timestamps = result.get("timestamp", [])
        quote = result["indicators"]["quote"][0]
        closes = quote.get("close", [])
        volumes = quote.get("volume", [])
        if not timestamps or not closes:
            return None
        df = pd.DataFrame(
            {
                "Close": [c if c is not None else float("nan") for c in closes],
                "Volume": [int(v) if v is not None else 0 for v in volumes],
            },
            index=pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None),
        )
        return df.dropna(subset=["Close"])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Batch fetch utility
# ---------------------------------------------------------------------------

def batch_fetch(
    tickers: list,
    fetch_one_fn,
    batch_size: int = 25,
    delay: float = 2.0,
    progress_bar=None,
) -> list:
    """
    Fetch data for each ticker by calling fetch_one_fn(ticker).
    Processes in batches of `batch_size` with `delay` seconds between batches.
    If a ticker raises an exception indicating a rate limit (429 / Too Many Requests),
    the rest of that batch is skipped rather than crashing the whole fetch.
    All other per-ticker errors should be handled inside fetch_one_fn itself.
    Returns a flat list of all non-None results from fetch_one_fn.
    """
    results = []
    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]
    total = len(batches)

    for idx, batch in enumerate(batches):
        rate_limited = False
        for ticker in batch:
            if rate_limited:
                break
            try:
                row = fetch_one_fn(ticker)
                if row is not None:
                    results.append(row)
            except Exception as exc:
                err_str = str(exc).lower()
                if "429" in err_str or "rate" in err_str or "too many" in err_str:
                    rate_limited = True
                # Non-rate-limit errors: fetch_one_fn is expected to handle them internally

        if progress_bar is not None:
            pct = (idx + 1) / total
            suffix = " (rate limited, skipping rest)" if rate_limited else ""
            progress_bar.progress(
                pct,
                text=f"Fetching: batch {idx + 1}/{total}{suffix} — {len(results)} tickers done",
            )

        if idx < total - 1:
            time.sleep(delay)

    return results


# ---------------------------------------------------------------------------
# Existing helpers
# ---------------------------------------------------------------------------

def fmt_volume(val) -> str:
    """Format a volume number as K/M/B string."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    val = float(val)
    if val >= 1e9:
        return f"{val / 1e9:.1f}B"
    if val >= 1e6:
        return f"{val / 1e6:.1f}M"
    if val >= 1e3:
        return f"{val / 1e3:.0f}K"
    return f"{int(val)}"


def load_metadata_cache() -> dict:
    """Load sector/industry cache from disk. Returns {ticker: {sector, industry}}."""
    if os.path.exists(METADATA_CACHE_PATH):
        try:
            df = pd.read_csv(METADATA_CACHE_PATH, dtype=str).fillna("")
            result = {}
            for _, row in df.iterrows():
                result[row["Ticker"]] = {
                    "sector": row.get("sector", "") or None,
                    "industry": row.get("industry", "") or None,
                }
            return result
        except Exception:
            pass
    return {}


def save_metadata_cache(cache: dict) -> None:
    """Save sector/industry cache to disk."""
    try:
        os.makedirs("data", exist_ok=True)
        rows = [
            {
                "Ticker": t,
                "sector": v.get("sector") or "",
                "industry": v.get("industry") or "",
            }
            for t, v in cache.items()
        ]
        pd.DataFrame(rows).to_csv(METADATA_CACHE_PATH, index=False)
    except Exception:
        pass
