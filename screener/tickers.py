import os
import time
import pandas as pd
import requests
import streamlit as st

NASDAQ_URL = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
SP500_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
TICKER_CACHE_PATH = "data/tickers.csv"
FILTERED_CACHE_PATH = "data/filtered_tickers.csv"
FILTERED_CACHE_TTL = 3600  # 1 hour

# Last-resort hardcoded list of highly liquid Russell 1000 names
LAST_RESORT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "LLY",
    "AVGO", "V", "JPM", "UNH", "XOM", "MA", "PG", "JNJ", "HD", "MRK",
    "COST", "ABBV", "CVX", "KO", "BAC", "NFLX", "PEP", "TMO", "WMT", "ADBE",
    "CSCO", "MCD", "ABT", "CRM", "ORCL", "ACN", "LIN", "DHR", "AMD", "AMGN",
    "NKE", "TXN", "PM", "QCOM", "NEE", "UPS", "CAT", "RTX", "HON", "IBM",
    "INTU", "SBUX", "T", "AMAT", "GS", "BKNG", "ELV", "LOW", "VRTX", "DE",
    "MDT", "AXP", "GE", "SPGI", "BLK", "GILD", "ADI", "NOW", "SYK", "ZTS",
    "REGN", "CI", "PLD", "MO", "DUK", "ADP", "SO", "BMY", "MMC", "CB",
    "ITW", "SCHW", "HCA", "ICE", "SHW", "CME", "WM", "PGR", "EOG", "NOC",
    "USB", "WFC", "C", "MS", "VZ", "INTC", "MMM", "BA", "LMT", "GD",
    "SPY", "QQQ", "IWM", "DIA", "GLD", "TLT", "SLV", "XLF", "XLE", "XLK",
]


@st.cache_data(ttl=86400)
def fetch_ticker_list() -> tuple[pd.DataFrame, str]:
    """Download full US-listed ticker list from NASDAQ FTP (NYSE, NASDAQ, AMEX). Cached 24h.
    Returns (DataFrame with Ticker column, source_label string).
    Falls back to S&P 500 CSV, then disk cache, then hardcoded list on failure."""
    dfs = []
    nasdaq_errors = []

    for url in [NASDAQ_URL, OTHER_URL]:
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            lines = r.text.strip().split("\n")
            if len(lines) < 2:
                nasdaq_errors.append(f"{url}: too few lines ({len(lines)})")
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
                nasdaq_errors.append(f"{url}: symbol column not found (cols: {list(df.columns)[:5]})")
                continue
            out = df[[sym_col]].copy()
            out.columns = ["Ticker"]
            dfs.append(out)
        except Exception as exc:
            nasdaq_errors.append(f"{url}: {exc}")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        # Keep only clean single-class tickers (1-5 uppercase letters, no warrants/units)
        combined = combined[combined["Ticker"].str.match(r"^[A-Z]{1,5}$", na=False)]
        combined = combined.drop_duplicates("Ticker").reset_index(drop=True)
        try:
            os.makedirs("data", exist_ok=True)
            combined.to_csv(TICKER_CACHE_PATH, index=False)
        except Exception:
            pass
        return combined, "NASDAQ FTP"

    # NASDAQ FTP failed — try S&P 500 CSV from GitHub
    try:
        r = requests.get(SP500_CSV_URL, timeout=20)
        r.raise_for_status()
        from io import StringIO
        sp500 = pd.read_csv(StringIO(r.text))
        sym_col = next((c for c in sp500.columns if c.lower() in ("symbol", "ticker")), None)
        if sym_col and not sp500.empty:
            out = sp500[[sym_col]].copy()
            out.columns = ["Ticker"]
            out = out[out["Ticker"].str.match(r"^[A-Z]{1,5}$", na=False)]
            out = out.drop_duplicates("Ticker").reset_index(drop=True)
            return out, "S&P 500 (GitHub fallback)"
    except Exception:
        pass

    # Try disk cache
    if os.path.exists(TICKER_CACHE_PATH):
        try:
            cached = pd.read_csv(TICKER_CACHE_PATH)
            if not cached.empty:
                return cached, "disk cache (NASDAQ FTP unavailable)"
        except Exception:
            pass

    # Last resort — hardcoded list
    return pd.DataFrame({"Ticker": LAST_RESORT_TICKERS}), "hardcoded fallback list (all sources failed)"


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


def get_filtered_universe(progress_bar=None, status_placeholder=None) -> tuple[list[str], str]:
    """
    Return (tickers, source_label) — a pre-filtered list of liquid US tickers and where they came from.
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
                        f"Loaded {len(cached_tickers)} tickers from filtered cache (< 1h old)."
                    )
                return cached_tickers, "filtered cache"
        except Exception:
            pass

    # Fetch fresh ticker list
    if status_placeholder is not None:
        status_placeholder.text("Downloading ticker list...")
    try:
        ticker_df, source_label = fetch_ticker_list()
        all_tickers = ticker_df["Ticker"].tolist()
    except Exception as exc:
        # fetch_ticker_list itself crashed — try data/tickers.csv then default list
        if os.path.exists(TICKER_CACHE_PATH):
            try:
                ticker_df = pd.read_csv(TICKER_CACHE_PATH)
                all_tickers = ticker_df["Ticker"].tolist()
                source_label = "disk cache (fetch error)"
            except Exception:
                all_tickers = LAST_RESORT_TICKERS
                source_label = f"hardcoded fallback (fetch crashed: {exc})"
        else:
            all_tickers = LAST_RESORT_TICKERS
            source_label = f"hardcoded fallback (fetch crashed: {exc})"

    if status_placeholder is not None:
        status_placeholder.text(
            f"Source: {source_label}. Pre-filtering {len(all_tickers)} tickers "
            "(price > $1, avg vol > 500K)..."
        )

    filtered = prefilter_universe(all_tickers, progress_bar=progress_bar)

    # Save to disk cache
    try:
        os.makedirs("data", exist_ok=True)
        pd.DataFrame({"Ticker": filtered}).to_csv(FILTERED_CACHE_PATH, index=False)
    except Exception:
        pass

    return filtered, source_label
