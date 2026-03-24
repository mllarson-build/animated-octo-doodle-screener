import pandas as pd
import streamlit as st
from datetime import datetime

from screener.value import fetch_value_data
from screener.growth import fetch_growth_data
from screener.options import fetch_options_data
from screener.etf import fetch_etf_data
from screener.tickers import get_filtered_universe
from screener.utils import fmt_volume

st.set_page_config(page_title="Daily Screener", layout="wide")

DEFAULT_TICKERS = "AAPL\nMSFT\nJPM\nXOM\nBAC\nNVDA\nSPY\nQQQ\nIWM\nTLT\nGLD"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def color_score_value(val):
    if val >= 5:
        return "background-color: #1a7a1a; color: white"
    elif val >= 3:
        return "background-color: #7a7a00; color: white"
    else:
        return "background-color: #7a1a1a; color: white"


def color_vol_ratio(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if val > 1.5:
        return "background-color: #1a7a1a; color: white"
    elif val >= 1.0:
        return "background-color: #7a7a00; color: white"
    else:
        return "background-color: #444444; color: white"


def color_upside(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if val > 25:
        return "background-color: #1a7a1a; color: white"
    elif val >= 10:
        return "background-color: #7a7a00; color: white"
    elif val < 0:
        return "background-color: #7a1a1a; color: white"
    return ""


def color_return(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if val > 0:
        return "background-color: #1a7a1a; color: white"
    if val < 0:
        return "background-color: #7a1a1a; color: white"
    return ""


def add_volume_display_cols(df: pd.DataFrame, price_col: str, vol_col: str) -> pd.DataFrame:
    """Add formatted Volume and $ Volume string columns to a display copy."""
    df = df.copy()
    df["Volume"] = df[vol_col].apply(fmt_volume)
    df["$ Volume"] = df.apply(
        lambda row: fmt_volume(
            float(row[price_col]) * float(row[vol_col])
            if pd.notna(row[price_col]) and pd.notna(row[vol_col])
            else None
        ),
        axis=1,
    )
    return df


def sector_industry_filters(df: pd.DataFrame, key_prefix: str):
    """Render sector + industry multiselects; return filtered DataFrame."""
    all_sectors = sorted(df["Sector"].dropna().unique().tolist())
    selected_sectors = st.multiselect(
        "Filter by Sector",
        options=all_sectors,
        default=all_sectors,
        key=f"{key_prefix}_sectors",
    )

    if selected_sectors:
        sector_mask = df["Sector"].isin(selected_sectors) | df["Sector"].isna()
        sector_df = df[sector_mask]
    else:
        sector_df = df

    all_industries = sorted(sector_df["Industry"].dropna().unique().tolist())
    selected_industries = st.multiselect(
        "Filter by Industry",
        options=all_industries,
        default=all_industries,
        key=f"{key_prefix}_industries",
    )

    if selected_industries:
        industry_mask = sector_df["Industry"].isin(selected_industries) | sector_df["Industry"].isna()
        filtered = sector_df[industry_mask]
    else:
        filtered = sector_df

    return filtered


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Tickers")

    use_full_universe = st.toggle(
        "Use Full Universe",
        value=False,
        help="Scan all NYSE/NASDAQ/AMEX stocks. Filters to price > $1 and avg vol > 500K before screening.",
    )

    if not use_full_universe:
        ticker_input = st.text_area(
            "Paste tickers (one per line)",
            value=DEFAULT_TICKERS,
            height=250,
        )
    else:
        st.caption(
            "Scans NYSE, NASDAQ & AMEX. Pre-filters to price > $1 and avg daily volume > 500K. "
            "First run may take several minutes."
        )

    refresh = st.button("Refresh Data", use_container_width=True)

    if refresh:
        if use_full_universe:
            status_text = st.empty()
            progress_bar = st.progress(0, text="Starting…")
            tickers = get_filtered_universe(
                progress_bar=progress_bar,
                status_placeholder=status_text,
            )
            progress_bar.empty()
            status_text.text(f"Universe filtered to {len(tickers)} liquid tickers.")
        else:
            tickers = [t.strip().upper() for t in ticker_input.splitlines() if t.strip()]

        with st.spinner(f"Screening {len(tickers)} tickers…"):
            st.session_state["tickers"] = tickers
            st.session_state["value_df"] = fetch_value_data(tickers)
            st.session_state["growth_df"] = fetch_growth_data(tickers)
            st.session_state["etf_data"] = fetch_etf_data()
            st.session_state["last_refreshed"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    if "last_refreshed" in st.session_state:
        st.caption(f"Last refreshed: {st.session_state['last_refreshed']}")

# ---------------------------------------------------------------------------
# Column configs
# ---------------------------------------------------------------------------
VALUE_COLUMN_CONFIG = {
    "Ticker": st.column_config.TextColumn("Ticker"),
    "Sector": st.column_config.TextColumn("Sector"),
    "Industry": st.column_config.TextColumn("Industry"),
    "Price": st.column_config.NumberColumn(
        "Current Price",
        help="Current market price of the stock",
        format="$%.2f",
    ),
    "52W High": st.column_config.NumberColumn(
        "52W High",
        help="Highest price in the last 52 weeks",
        format="$%.2f",
    ),
    "52W Low": st.column_config.NumberColumn(
        "52W Low",
        help="Lowest price in the last 52 weeks",
        format="$%.2f",
    ),
    "Drawdown %": st.column_config.NumberColumn(
        "Drawdown %",
        help="Percent decline from 52-week high. Values over 20% may indicate oversold conditions",
        format="%.1f%%",
    ),
    "52W Return %": st.column_config.NumberColumn(
        "52W Return %",
        help="Price change over the last 52 weeks (annual momentum)",
        format="%.1f%%",
    ),
    "RSI (14)": st.column_config.NumberColumn(
        "RSI",
        help="Relative Strength Index (14-day). Below 40 suggests oversold, above 60 suggests overbought",
        format="%.1f",
    ),
    "Trailing P/E": st.column_config.NumberColumn(
        "Trailing P/E",
        help="Price divided by last 12 months earnings. Lower may indicate better value",
        format="%.2f",
    ),
    "Forward P/E": st.column_config.NumberColumn(
        "Forward P/E",
        help="Price divided by next 12 months estimated earnings. Lower may indicate better value",
        format="%.2f",
    ),
    "P/B": st.column_config.NumberColumn(
        "P/B Ratio",
        help="Price divided by book value. Below 1 may indicate undervalued assets",
        format="%.2f",
    ),
    "PEG": st.column_config.NumberColumn(
        "PEG Ratio",
        help="P/E divided by earnings growth rate. Below 1 considered undervalued",
        format="%.2f",
    ),
    "Div Yield %": st.column_config.NumberColumn(
        "Div Yield %",
        help="Annual dividend divided by stock price. A positive value means the company pays a dividend",
        format="%.1f%%",
    ),
    "D/E": st.column_config.NumberColumn(
        "D/E Ratio",
        help="Total debt divided by shareholder equity. Higher values indicate more financial leverage",
        format="%.2f",
    ),
    "Volume": st.column_config.TextColumn(
        "Volume",
        help="Most recent day's trading volume",
    ),
    "$ Volume": st.column_config.TextColumn(
        "$ Volume",
        help="Dollar volume (price × volume). More useful than share volume for assessing liquidity",
    ),
    "Vol Ratio": st.column_config.NumberColumn(
        "Volume Ratio",
        help="Current volume divided by 30-day average. Above 1.2 suggests above-average interest",
        format="%.2f",
    ),
    "recovery_score": st.column_config.NumberColumn(
        "Recovery Score",
        help="0–7 score counting how many value signals are triggered (drawdown, RSI, P/E, P/B, volume, PEG, dividend)",
        format="%d",
    ),
}

GROWTH_COLUMN_CONFIG = {
    "Ticker": st.column_config.TextColumn("Ticker"),
    "Sector": st.column_config.TextColumn("Sector"),
    "Industry": st.column_config.TextColumn("Industry"),
    "Price": st.column_config.NumberColumn(
        "Current Price",
        help="Current market price of the stock",
        format="$%.2f",
    ),
    "Rev Growth %": st.column_config.NumberColumn(
        "Revenue Growth YoY",
        help="Year over year revenue growth percent. Above 15% considered strong",
        format="%.1f%%",
    ),
    "Earn Growth %": st.column_config.NumberColumn(
        "Earnings Growth YoY",
        help="Year over year earnings growth percent. Above 10% considered strong",
        format="%.1f%%",
    ),
    "Analyst Target": st.column_config.NumberColumn(
        "Analyst Target",
        help="Mean analyst price target",
        format="$%.2f",
    ),
    "Upside %": st.column_config.NumberColumn(
        "Upside %",
        help="Percent difference between analyst target and current price",
        format="%.1f%%",
    ),
    "Short % Float": st.column_config.NumberColumn(
        "Short Interest %",
        help="Percent of float sold short. Above 10% indicates heavy short interest",
        format="%.1f%%",
    ),
    "3M Momentum %": st.column_config.NumberColumn(
        "3M Momentum",
        help="Price change over last 63 trading days",
        format="%.1f%%",
    ),
    "growth_score": st.column_config.NumberColumn(
        "Growth Score",
        help="0–5 score counting how many growth signals are triggered (revenue, earnings, upside, short interest, momentum)",
        format="%d",
    ),
}

OPTIONS_COLUMN_CONFIG = {
    "Strike": st.column_config.NumberColumn(
        "Strike",
        help="The price at which the option can be exercised",
        format="$%.2f",
    ),
    "Expiration": st.column_config.TextColumn(
        "Expiration",
        help="Date the option contract expires",
    ),
    "Type": st.column_config.TextColumn("Type"),
    "Last Price": st.column_config.NumberColumn(
        "Last Price",
        help="Most recent trade price of the option contract",
        format="$%.2f",
    ),
    "IV": st.column_config.NumberColumn(
        "Implied Volatility",
        help="Market's expectation of future volatility. Higher IV means more expensive options",
        format="%.1f%%",
    ),
    "Volume": st.column_config.TextColumn(
        "Volume",
        help="Number of contracts traded today",
    ),
    "Open Interest": st.column_config.TextColumn(
        "Open Interest",
        help="Total number of outstanding contracts",
    ),
    "Vol/OI": st.column_config.NumberColumn(
        "Volume/OI Ratio",
        help="Ratio of today's volume to open interest. Above 1.5 suggests unusual activity",
        format="%.2f",
    ),
}

ETF_COLUMN_CONFIG = {
    "ETF": st.column_config.TextColumn("ETF"),
    "Price": st.column_config.NumberColumn(
        "Price",
        help="Current market price",
        format="$%.2f",
    ),
    "1D %": st.column_config.NumberColumn(
        "1D Return",
        help="Price change over last trading day",
        format="%.2f%%",
    ),
    "1W %": st.column_config.NumberColumn(
        "1W Return",
        help="Price change over last 5 trading days",
        format="%.2f%%",
    ),
    "1M %": st.column_config.NumberColumn(
        "1M Return",
        help="Price change over last 21 trading days",
        format="%.2f%%",
    ),
    "RSI (14)": st.column_config.NumberColumn(
        "RSI",
        help="Relative Strength Index. Below 40 oversold, above 60 overbought",
        format="%.1f",
    ),
    "Avg Vol (30d)": st.column_config.TextColumn(
        "Avg Volume",
        help="30-day average daily trading volume",
    ),
}

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_value, tab_growth, tab_options, tab_etfs = st.tabs(
    ["Value Recovery", "Growth Momentum", "Options", "ETFs"]
)

# ── Value Recovery ──────────────────────────────────────────────────────────
with tab_value:
    st.subheader("Value Recovery")
    if "value_df" not in st.session_state:
        st.info("Press **Refresh Data** in the sidebar to run the screener.")
    else:
        raw_df = st.session_state["value_df"]

        # Sector / industry filters
        filtered_df = sector_industry_filters(raw_df, key_prefix="val")

        total = len(filtered_df)
        high_conviction = int((filtered_df["recovery_score"] >= 5).sum())
        col1, col2, col3 = st.columns(3)
        col1.metric("Tickers Shown", total)
        col2.metric("Scored 5+", high_conviction)
        col3.metric("Scored 3+", int((filtered_df["recovery_score"] >= 3).sum()))

        # Build display DataFrame: add formatted volume columns, drop raw volume
        display_df = add_volume_display_cols(filtered_df, price_col="Price", vol_col="Current Vol")
        display_df = display_df.drop(columns=["Avg Vol (30d)", "Current Vol"])

        styled = (
            display_df.style
            .applymap(color_score_value, subset=["recovery_score"])
            .applymap(color_vol_ratio, subset=["Vol Ratio"])
        )
        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config=VALUE_COLUMN_CONFIG,
        )

# ── Growth Momentum ──────────────────────────────────────────────────────────
with tab_growth:
    st.subheader("Growth Momentum")
    if "growth_df" not in st.session_state:
        st.info("Press **Refresh Data** in the sidebar to run the screener.")
    else:
        raw_df = st.session_state["growth_df"]

        # Sector / industry filters
        filtered_df = sector_industry_filters(raw_df, key_prefix="grw")

        total = len(filtered_df)
        high_conviction = int((filtered_df["growth_score"] >= 3).sum())
        col1, col2 = st.columns(2)
        col1.metric("Tickers Shown", total)
        col2.metric("Scored 3+", high_conviction)

        styled = filtered_df.style.applymap(color_upside, subset=["Upside %"])
        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config=GROWTH_COLUMN_CONFIG,
        )

# ── Options ──────────────────────────────────────────────────────────────────
with tab_options:
    st.subheader("Options")

    _default_tickers = [t for t in DEFAULT_TICKERS.splitlines() if t.strip()]
    watchlist = st.session_state.get("tickers", _default_tickers)

    col_l, col_r = st.columns([2, 1])
    with col_l:
        selected_ticker = st.selectbox("Ticker", watchlist, key="opt_ticker")
    with col_r:
        option_type = st.radio("Type", ["Calls", "Puts"], horizontal=True, key="opt_type")

    load_options = st.button("Load Options Chain", key="opt_load")

    if load_options:
        with st.spinner(f"Fetching options for {selected_ticker}…"):
            try:
                result = fetch_options_data(selected_ticker)
                st.session_state["opt_data"] = result
                st.session_state["opt_ticker_loaded"] = selected_ticker
            except Exception as e:
                st.error(f"Could not load options for {selected_ticker}: {e}")
                st.session_state.pop("opt_data", None)

    if "opt_data" in st.session_state:
        data = st.session_state["opt_data"]
        loaded_for = st.session_state.get("opt_ticker_loaded", "")
        st.caption(
            f"Showing options for **{loaded_for}** — current price: **${data['current_price']:.2f}**"
        )

        selected_exp = st.selectbox("Expiration", data["expirations"], key="opt_exp")

        chain_key = "calls" if option_type == "Calls" else "puts"
        df = data["chains"][selected_exp][chain_key]

        if df.empty:
            st.info("No contracts found for this selection.")
        else:
            flagged = int(df["Flag"].sum())
            if flagged:
                st.warning(
                    f"{flagged} contract(s) flagged for unusual activity "
                    "(Vol/OI > 1.5, near money, volume > 100)"
                )

            def highlight_flag(row):
                if df.loc[row.name, "Flag"]:
                    return ["background-color: #4a3800; color: white"] * len(row)
                return [""] * len(row)

            display_df = df.drop(columns=["Flag"]).copy()
            # Format volume columns as K/M/B strings
            display_df["Volume"] = display_df["Volume"].apply(
                lambda v: fmt_volume(v) if pd.notna(v) else "—"
            )
            display_df["Open Interest"] = display_df["Open Interest"].apply(
                lambda v: fmt_volume(v) if pd.notna(v) else "—"
            )

            styled = display_df.style.apply(highlight_flag, axis=1)
            st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
                column_config=OPTIONS_COLUMN_CONFIG,
            )

# ── ETFs ─────────────────────────────────────────────────────────────────────
with tab_etfs:
    st.subheader("ETFs")
    if "etf_data" not in st.session_state:
        st.info("Press **Refresh Data** in the sidebar to load ETF data.")
    else:
        etf_df = st.session_state["etf_data"]["etf_df"]
        macro = st.session_state["etf_data"]["macro"]

        # Format volume as K/M/B string
        display_etf = etf_df.copy()
        display_etf["Avg Vol (30d)"] = display_etf["Avg Vol (30d)"].apply(
            lambda v: fmt_volume(v) if v is not None else "—"
        )

        ret_cols = ["1D %", "1W %", "1M %"]
        styled = display_etf.style.applymap(color_return, subset=ret_cols)
        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config=ETF_COLUMN_CONFIG,
        )

        st.divider()

        # Macro panel
        st.subheader("Macro Indicators")
        mcol1, mcol2 = st.columns(2)

        vix = macro["vix"]
        vix_chg = macro["vix_1w_chg"]
        tnx = macro["tnx"]
        tnx_chg = macro["tnx_1w_chg"]

        mcol1.metric(
            "VIX",
            f"{vix:.2f}" if vix is not None else "N/A",
            delta=f"{vix_chg:+.2f}% (1W)" if vix_chg is not None else None,
            delta_color="inverse",
            help="CBOE Volatility Index measuring market fear. Above 25 is elevated, below 15 is calm",
        )
        mcol2.metric(
            "10Y Treasury Yield",
            f"{tnx:.2f}%" if tnx is not None else "N/A",
            delta=f"{tnx_chg:+.2f}% (1W)" if tnx_chg is not None else None,
            help="10-year US Treasury yield, key benchmark for risk-free rate",
        )

        if vix is not None:
            if vix > 25:
                st.warning("Elevated volatility — favor defensive positioning")
            elif vix < 15:
                st.success("Low volatility environment")
