import os
import pandas as pd

METADATA_CACHE_PATH = "data/ticker_metadata.csv"


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
