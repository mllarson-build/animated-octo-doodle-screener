import os
import time
import pandas as pd
import requests
import streamlit as st

NASDAQ_URL = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
TICKER_CACHE_PATH = "data/tickers.csv"
FILTERED_CACHE_PATH = "data/filtered_tickers.csv"
FILTERED_CACHE_TTL = 3600  # 1 hour


@st.cache_data(ttl=86400)
def fetch_ticker_list() -> pd.DataFrame:
    """Download full US-listed ticker list from NASDAQ FTP (NYSE, NASDAQ, AMEX). Cached 24h."""
    dfs = []

    for url in [NASDAQ_URL, OTHER_URL]:
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            lines = r.text.strip().split("\n")
            if len(lines) < 2:
                continue
            header = lines[0].split("|")
            # Skip header (index 0) and trailer (last line which is file creation info)
            data_lines = lines[1:-1]
            rows = [ln.split("|") for ln in data_lines if "|" in ln]
            df = pd.DataFrame(rows, columns=header)
            if "Test Issue" in df.columns:
                df = df[df["Test Issue"] == "N"]
            sym_col = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
            if sym_col not in df.columns:
                continue
            out = df[[sym_col]].copy()
            out.columns = ["Ticker"]
            dfs.append(out)
        except Exception:
            pass

    if not dfs:
        # Fall back to disk cache
        if os.path.exists(TICKER_CACHE_PATH):
            try:
                return pd.read_csv(TICKER_CACHE_PATH)
            except Exception:
                pass
        return pd.DataFrame(columns=["Ticker"])

    combined = pd.concat(dfs, ignore_index=True)
    # Keep only clean single-class tickers (1-5 uppercase letters, no warrants/units)
    combined = combined[combined["Ticker"].str.match(r"^[A-Z]{1,5}$", na=False)]
    combined = combined.drop_duplicates("Ticker").reset_index(drop=True)

    try:
        os.makedirs("data", exist_ok=True)
        combined.to_csv(TICKER_CACHE_PATH, index=False)
    except Exception:
        pass

    return combined


def prefilter_universe(
    tickers: list[str],
    min_price: float = 1.0,
    min_avg_vol: int = 500_000,
    progress_bar=None,
) -> list[str]:
    """
    Batch-download 5 days of price/volume data (groups of 50) to filter
    out penny stocks and illiquid tickers. Returns list of tickers that pass.
    """
    import yfinance as yf

    passed = []
    batch_size = 50
    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]
    total = len(batches)

    for idx, batch in enumerate(batches):
        try:
            if len(batch) == 1:
                raw = yf.download(
                    batch[0], period="5d", auto_adjust=True, progress=False
                )
                if not raw.empty:
                    c = raw["Close"].dropna()
                    v = raw["Volume"].dropna()
                    if (
                        not c.empty
                        and not v.empty
                        and float(c.iloc[-1]) >= min_price
                        and float(v.mean()) >= min_avg_vol
                    ):
                        passed.extend(batch)
            else:
                raw = yf.download(
                    batch,
                    period="5d",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                    group_by="ticker",
                )
                if raw.empty:
                    continue
                lvl0 = set(raw.columns.get_level_values(0))
                for t in batch:
                    if t not in lvl0:
                        continue
                    try:
                        tc = raw[t]["Close"].dropna()
                        tv = raw[t]["Volume"].dropna()
                        if tc.empty or tv.empty:
                            continue
                        if float(tc.iloc[-1]) >= min_price and float(tv.mean()) >= min_avg_vol:
                            passed.append(t)
                    except Exception:
                        pass
        except Exception:
            pass

        if progress_bar is not None:
            pct = (idx + 1) / total
            progress_bar.progress(
                pct,
                text=f"Pre-filtering: batch {idx + 1}/{total} — {len(passed)} tickers passed so far",
            )
        time.sleep(0.05)  # avoid rate limiting

    return passed


def get_filtered_universe(progress_bar=None, status_placeholder=None) -> list[str]:
    """
    Return a pre-filtered list of liquid US tickers.
    Uses a 1-hour disk cache to avoid re-running pre-filtering on every refresh.
    """
    # Check disk cache first
    if os.path.exists(FILTERED_CACHE_PATH):
        try:
            mtime = os.path.getmtime(FILTERED_CACHE_PATH)
            if time.time() - mtime < FILTERED_CACHE_TTL:
                df = pd.read_csv(FILTERED_CACHE_PATH)
                cached_tickers = df["Ticker"].tolist()
                if status_placeholder is not None:
                    status_placeholder.text(
                        f"Loaded {len(cached_tickers)} tickers from cache (< 1h old)."
                    )
                return cached_tickers
        except Exception:
            pass

    # Fetch fresh ticker list
    if status_placeholder is not None:
        status_placeholder.text("Downloading ticker list from NASDAQ FTP...")
    ticker_df = fetch_ticker_list()
    all_tickers = ticker_df["Ticker"].tolist()

    if status_placeholder is not None:
        status_placeholder.text(
            f"Pre-filtering {len(all_tickers)} tickers (price > $1, avg vol > 500K)..."
        )

    filtered = prefilter_universe(all_tickers, progress_bar=progress_bar)

    # Save to disk cache
    try:
        os.makedirs("data", exist_ok=True)
        pd.DataFrame({"Ticker": filtered}).to_csv(FILTERED_CACHE_PATH, index=False)
    except Exception:
        pass

    return filtered
