from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import streamlit as st
from datetime import datetime

from screener.value import fetch_value_data
from screener.growth import fetch_growth_data
from screener.options import fetch_options_data
from screener.etf import fetch_etf_data
from screener.tickers import get_filtered_universe
from screener.utils import fmt_volume
from screener.trade import analyze_trade, save_trade_idea, load_trade_log, save_trade_log, compute_paper_stats
from screener.db import ensure_db
from screener.backtest import (
    run_backtest,
    HOLDING_PERIOD_DAYS,
    LOOKBACK_PERIODS,
    SIGNAL_DESCRIPTIONS,
)

st.set_page_config(page_title="Daily Screener", layout="wide")

# Initialize SQLite database (creates tables + migrates CSV on first run)
_migration = ensure_db()
if _migration and _migration.get("imported", 0) > 0:
    st.toast(f"Migrated {_migration['imported']} trades from CSV to SQLite.")

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
        if "tickers" in st.session_state and "ticker_source" in st.session_state:
            n = len(st.session_state["tickers"])
            src = st.session_state["ticker_source"]
            st.info(f"Full universe: {n} tickers loaded from {src}.")

    refresh = st.button("Refresh Data", use_container_width=True)

    if refresh:
        if use_full_universe:
            status_text = st.empty()
            progress_bar = st.progress(0, text="Starting…")
            tickers, ticker_source = get_filtered_universe(
                progress_bar=progress_bar,
                status_placeholder=status_text,
            )
            progress_bar.empty()
            if len(tickers) == 0:
                status_text.warning(
                    f"Universe filtered to 0 liquid tickers (source: {ticker_source}). "
                    "The pre-filter may be too strict or data could not be fetched. "
                    "Try refreshing or switching to manual ticker mode."
                )
            else:
                status_text.text(
                    f"Universe filtered to {len(tickers)} liquid tickers (source: {ticker_source})."
                )
            st.session_state["ticker_source"] = ticker_source
        else:
            tickers = [t.strip().upper() for t in ticker_input.splitlines() if t.strip()]
            st.session_state.pop("ticker_source", None)

        st.session_state["tickers"] = tickers
        st.session_state["last_refreshed"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Clear tab data so each tab loads its own data lazily
        for _key in ("value_df", "growth_df", "etf_data"):
            st.session_state.pop(_key, None)
        st.success(
            f"Tickers updated ({len(tickers)} tickers). "
            "Click **Load Data** inside each tab to fetch screener results."
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
tab_value, tab_growth, tab_options, tab_etfs, tab_trade, tab_backtest = st.tabs(
    ["Value Recovery", "Growth Momentum", "Options", "ETFs", "Trade Builder", "Backtest"]
)

# ── Value Recovery ──────────────────────────────────────────────────────────
with tab_value:
    st.subheader("Value Recovery")
    _val_tickers = st.session_state.get("tickers", [t.strip() for t in DEFAULT_TICKERS.splitlines() if t.strip()])
    if st.button(f"Load Value Data ({len(_val_tickers)} tickers)", key="btn_load_value"):
        with st.spinner(f"Screening {len(_val_tickers)} tickers…"):
            st.session_state["value_df"] = fetch_value_data(_val_tickers)
            st.session_state["last_screen"] = "Value Recovery"

    if "value_df" not in st.session_state:
        st.info(
            "Click **Load Value Data** above to fetch screener results. "
            "Use **Refresh Data** in the sidebar to change the ticker universe first."
        )
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
    _grw_tickers = st.session_state.get("tickers", [t.strip() for t in DEFAULT_TICKERS.splitlines() if t.strip()])
    if st.button(f"Load Growth Data ({len(_grw_tickers)} tickers)", key="btn_load_growth"):
        with st.spinner(f"Screening {len(_grw_tickers)} tickers…"):
            st.session_state["growth_df"] = fetch_growth_data(_grw_tickers)
            st.session_state["last_screen"] = "Growth Momentum"

    if "growth_df" not in st.session_state:
        st.info(
            "Click **Load Growth Data** above to fetch screener results. "
            "Use **Refresh Data** in the sidebar to change the ticker universe first."
        )
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
                st.session_state["last_screen"] = "Options"
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
    if st.button("Load ETF Data", key="btn_load_etf"):
        with st.spinner("Fetching ETF and macro data…"):
            st.session_state["etf_data"] = fetch_etf_data()

    if "etf_data" not in st.session_state:
        st.info("Click **Load ETF Data** above to fetch ETF performance and macro indicators.")
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

# ── Trade Builder ─────────────────────────────────────────────────────────────
with tab_trade:
    from screener.db import close_trade as db_close_trade, execute_trade as db_execute_trade, HOLD_PERIOD_DAYS

    st.subheader("Trade")

    # ── Daily P&L Summary (top of tab — glanceable health check) ─────────────
    _stats = compute_paper_stats()
    _open_df = _stats["open_df"]
    _closed_df = _stats["closed_df"]
    _watch_df = _stats["watching_df"]
    _has_any_trades = not _open_df.empty or not _closed_df.empty or not _watch_df.empty

    if _has_any_trades:
        _pm1, _pm2, _pm3, _pm4 = st.columns(4)
        _pm1.metric(
            "Unrealized P&L",
            f"${_stats['total_unrealized_pnl']:+,.2f}" if not _open_df.empty else "—",
            delta_color="normal" if _stats["total_unrealized_pnl"] >= 0 else "inverse",
        )
        _pm2.metric(
            "Realized P&L",
            f"${_stats['total_realized_pnl']:+,.2f}" if not _closed_df.empty else "—",
            delta_color="normal" if _stats["total_realized_pnl"] >= 0 else "inverse",
        )
        # Best / worst open position by unrealized P&L %
        _best_today = _worst_today = "—"
        if not _open_df.empty and "unreal_pnl_pct" in _open_df.columns:
            _valid_pnl = _open_df.dropna(subset=["unreal_pnl_pct"])
            if not _valid_pnl.empty:
                _best_row = _valid_pnl.loc[_valid_pnl["unreal_pnl_pct"].idxmax()]
                _worst_row = _valid_pnl.loc[_valid_pnl["unreal_pnl_pct"].idxmin()]
                _best_today = f"{_best_row['ticker']} ({_best_row['unreal_pnl_pct']:+.1f}%)"
                _worst_today = f"{_worst_row['ticker']} ({_worst_row['unreal_pnl_pct']:+.1f}%)"
        _pm3.metric("Best Open", _best_today, delta_color="off")
        _pm4.metric("Worst Open", _worst_today, delta_color="off")
        st.divider()

    # ── Open Positions with Close Trade Flow ──────────────────────────────────
    if not _open_df.empty:
        st.markdown("**Open Positions**")
        for _idx, _row in _open_df.iterrows():
            _live = _row.get("live_price")
            _upnl = _row.get("unreal_pnl")
            _upnl_p = _row.get("unreal_pnl_pct")
            _ae = _row.get("actual_entry_n")
            _shares = int(float(_row["shares_n"])) if _row.get("shares_n") else 0
            _tid = int(_row["id"]) if "id" in _row.index else None

            # Holding period countdown
            _hold_text = ""
            _held_days = 0
            if _row.get("executed_at"):
                try:
                    from datetime import datetime as _dt
                    _exec_date = _dt.strptime(str(_row["executed_at"])[:10], "%Y-%m-%d")
                    _held_days = (datetime.now() - _exec_date).days
                    _suggested = _row.get("suggested_hold_period", "")
                    _target_days = HOLD_PERIOD_DAYS.get(_suggested, 0)
                    if _target_days > 0:
                        _remaining = _target_days - _held_days
                        _pct_elapsed = _held_days / _target_days
                        if _pct_elapsed < 0.75:
                            _hold_color = "#1a7a1a"
                        elif _pct_elapsed < 1.0:
                            _hold_color = "#7a7a00"
                        else:
                            _hold_color = "#7a1a1a"
                        _hold_text = f"Held: {_held_days}d / {_target_days}d"
                    else:
                        _hold_text = f"Held: {_held_days}d"
                        _hold_color = "#444"
                except Exception:
                    _hold_text = ""
                    _hold_color = "#444"

            _oc1, _oc2, _oc3, _oc4, _oc5, _oc6 = st.columns([1.2, 1, 1, 1, 1.2, 1.5])
            _oc1.markdown(f"**{_row['ticker']}**")
            _oc2.markdown(f"Entry: ${float(_ae):.2f}" if _ae else "Entry: —")
            _oc3.markdown(f"Live: ${_live:.2f}" if _live else "Live: N/A")
            _pnl_str = f"${_upnl:+,.2f} ({_upnl_p:+.1f}%)" if _upnl is not None and _upnl == _upnl else "—"
            _oc4.markdown(f"P&L: {_pnl_str}")
            if _hold_text:
                _oc5.markdown(
                    f"<span style='color:{_hold_color}'>{_hold_text}</span>",
                    unsafe_allow_html=True,
                )
            else:
                _oc5.markdown("")

            # Two-step close: click Close → expand details → Confirm
            if _tid is not None:
                _close_key = f"close_{_tid}"
                if _oc6.button("Close", key=_close_key, disabled=st.session_state.get(f"closing_{_tid}", False)):
                    st.session_state[f"show_close_{_tid}"] = True

                if st.session_state.get(f"show_close_{_tid}", False):
                    st.session_state[f"closing_{_tid}"] = True
                    _cc1, _cc2, _cc3 = st.columns([1, 1, 1])
                    with _cc1:
                        _exit_price = st.number_input(
                            "Exit Price",
                            value=float(_live) if _live else 0.0,
                            format="%.2f",
                            key=f"exit_price_{_tid}",
                        )
                    with _cc2:
                        _exit_reason = st.selectbox(
                            "Exit Reason",
                            ["Hit Target", "Hit Stop", "Manual Exit", "Thesis Invalidated"],
                            key=f"exit_reason_{_tid}",
                        )
                    with _cc3:
                        st.markdown("")  # spacer
                        if st.button("Confirm Close", key=f"confirm_close_{_tid}", type="primary"):
                            try:
                                _result = db_close_trade(_tid, _exit_price, _exit_reason)
                                compute_paper_stats.clear()
                                st.success(
                                    f"{_row['ticker']} closed. "
                                    f"P&L: ${_result['pnl']:+,.2f} ({_result['status']})"
                                )
                                st.session_state.pop(f"show_close_{_tid}", None)
                                st.session_state.pop(f"closing_{_tid}", None)
                                st.rerun()
                            except Exception as _e:
                                st.error(f"Could not close trade: {_e}")
                        if st.button("Cancel", key=f"cancel_close_{_tid}"):
                            st.session_state.pop(f"show_close_{_tid}", None)
                            st.session_state.pop(f"closing_{_tid}", None)
                            st.rerun()

        st.divider()

    # ── Trade Analyzer ────────────────────────────────────────────────────────
    st.markdown("**Analyze a Trade**")
    _watchlist = st.session_state.get(
        "tickers", [t.strip() for t in DEFAULT_TICKERS.splitlines() if t.strip()]
    )
    _tb_col1, _tb_col2 = st.columns([2, 1])
    with _tb_col1:
        _tb_ticker_select = st.selectbox(
            "Select from watchlist", _watchlist, key="tb_ticker_select"
        )
    with _tb_col2:
        _tb_ticker_manual = st.text_input(
            "Or enter manually", placeholder="e.g. NVDA", key="tb_ticker_manual"
        )
    _tb_ticker = _tb_ticker_manual.strip().upper() if _tb_ticker_manual.strip() else _tb_ticker_select

    _tb_analyze = st.button("Analyze Trade", type="primary", key="tb_analyze")

    if _tb_analyze:
        with st.spinner(f"Analyzing {_tb_ticker}…"):
            try:
                _tc = analyze_trade(_tb_ticker)
                st.session_state["trade_card"] = _tc
                st.session_state["trade_ticker"] = _tb_ticker
            except Exception as _e:
                st.error(f"Could not analyze {_tb_ticker}: {_e}")
                st.session_state.pop("trade_card", None)

    if "trade_card" not in st.session_state:
        st.info("Enter a ticker above and click **Analyze Trade** to generate a trade card.")
    else:
        _tc = st.session_state["trade_card"]
        _sym = _tc["ticker"]
        _ee = _tc["entry_exit"]
        _pl = _tc["price_levels"]
        _mo = _tc["momentum"]
        _sz = _tc["sizing"]
        _hg = _tc["hedge"]
        _fu = _tc["fundamentals"]
        _score = _tc["setup_score"]

        st.divider()

        # ── Setup quality score & summary ─────────────────────────────────────
        _score_labels = {5: "Excellent", 4: "Strong", 3: "Moderate", 2: "Weak", 1: "Poor", 0: "Poor"}
        _score_label = _score_labels.get(_score, "Poor")
        _score_col, _sum_col = st.columns([1, 4])
        with _score_col:
            _score_color = "#1a7a1a" if _score >= 4 else "#7a7a00" if _score >= 2 else "#7a1a1a"
            st.markdown(
                f"<div style='background:{_score_color};padding:16px;border-radius:8px;"
                f"text-align:center'>"
                f"<div style='color:white;font-size:2rem;font-weight:bold'>{_score}/5</div>"
                f"<div style='color:white;font-size:0.9rem'>{_score_label} Setup</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with _sum_col:
            st.markdown(f"**{_sym}** — {_tc['summary']}")
            if _ee["good_setup"]:
                st.success("R/R >= 2.0 — qualifies as a good setup")
            else:
                st.warning(f"R/R = {_ee['rr']:.1f} — below 2.0 threshold, proceed with caution")

        st.divider()

        # ── Key metrics row ───────────────────────────────────────────────────
        st.markdown("**Entry / Exit Framework**")
        _m1, _m2, _m3, _m4, _m5 = st.columns(5)
        _m1.metric("Current Price", f"${_tc['current_price']:.2f}")
        _m2.metric("Entry Zone", f"${_ee['entry']:.2f}", delta=f"to ${_ee['entry_low']:.2f}", delta_color="off")
        _m3.metric("Price Target", f"${_ee['target']:.2f}", delta=f"+{_ee['upside_pct']:.1f}%")
        _m4.metric("Stop Loss", f"${_ee['stop']:.2f}", delta="-8.0%", delta_color="inverse")
        _m5.metric("Risk/Reward", f"{_ee['rr']:.1f}:1", delta="Good" if _ee["good_setup"] else "Weak", delta_color="normal" if _ee["good_setup"] else "inverse")
        st.caption(f"Target source: {_ee['target_source']}")

        st.divider()

        # ── Price levels ──────────────────────────────────────────────────────
        st.markdown("**Price Levels**")
        _lc1, _lc2, _lc3, _lc4, _lc5, _lc6 = st.columns(6)
        _lc1.metric("52W High", f"${_pl['high_52w']:.2f}", delta=f"{_ee['drawdown_from_high']:.1f}%", delta_color="inverse")
        _lc2.metric("52W Low", f"${_pl['low_52w']:.2f}")
        _lc3.metric("20d Resistance", f"${_pl['resistance_20d']:.2f}")
        _lc4.metric("20d Support", f"${_pl['support_20d']:.2f}")
        _lc5.metric("50-day MA", f"${_pl['ma_50']:.2f}" if _pl["ma_50"] else "N/A")
        _lc6.metric("200-day MA", f"${_pl['ma_200']:.2f}" if _pl["ma_200"] else "N/A")

        st.divider()

        # ── Position sizing ──────────────────────────────────────────────────
        st.markdown("**Position Sizing** — based on $8,000 portfolio")
        _sz_rows = []
        for _lbl, _s in _sz.items():
            _sz_rows.append({
                "Size": _lbl.capitalize(),
                "Risk %": f"{_s['risk_pct']:.0f}%",
                "Shares": _s["shares"],
                "Total Cost": f"${_s['total_cost']:,.2f}",
                "Max Loss (at stop)": f"-${_s['max_loss']:,.2f}",
                "Target Profit": f"+${_s['target_profit']:,.2f}",
            })
        st.dataframe(pd.DataFrame(_sz_rows), use_container_width=True, hide_index=True)

        st.divider()

        # ── Momentum & timing ────────────────────────────────────────────────
        st.markdown("**Momentum & Timing**")
        _mc1, _mc2, _mc3, _mc4, _mc5 = st.columns(5)
        _rsi_val = _mo["rsi"]
        _mc1.metric("RSI (14)", f"{_rsi_val:.1f}" if _rsi_val is not None else "N/A", delta=_mo["rsi_label"], delta_color="off")
        _mc2.metric("MACD Signal", "Bullish cross" if _mo["macd_bullish_crossover"] else "No cross", delta="Last 5 days", delta_color="off")
        _mc3.metric("vs 50-day MA", "Above" if _mo["above_50ma"] else "Below" if _mo["above_50ma"] is not None else "N/A", delta_color="off")
        _mc4.metric("vs 200-day MA", "Above" if _mo["above_200ma"] else "Below" if _mo["above_200ma"] is not None else "N/A", delta_color="off")
        _mc5.metric("ATR (14d)", f"${_mo['atr']:.2f}" if _mo["atr"] else "N/A", delta=f"{_mo['atr_pct']:.1f}% of price" if _mo["atr_pct"] else None, delta_color="off")
        if _mo["macd_bullish_crossover"]:
            st.success("MACD bullish crossover detected in the last 5 trading days.")
        if _mo["above_200ma"] is True:
            st.success("Price is above the 200-day MA — long-term uptrend intact.")
        elif _mo["above_200ma"] is False:
            st.warning("Price is below the 200-day MA — caution in longer-term entries.")
        st.info(f"Suggested holding period: **{_mo['holding_period']}**")

        st.divider()

        # ── Hedge suggestions ────────────────────────────────────────────────
        st.markdown("**Hedge Suggestions**")
        _hc1, _hc2 = st.columns(2)
        with _hc1:
            st.markdown("**Protective Put**")
            _pp = _hg["protective_put"]
            if _pp:
                st.markdown(f"Buy **{_pp['num_contracts']} put(s)** at **${_pp['strike']:.2f} strike** exp **{_pp['expiration']}**")
                st.markdown(f"- Premium: ${_pp['premium_per_share']:.2f}/share ({_pp['cost_pct_of_position']:.1f}% of position)")
                st.markdown(f"- Total put cost: **${_pp['total_cost']:,.2f}**")
                st.markdown(f"- Breakeven (inc. put cost): **${_pp['breakeven_with_put']:.2f}**")
            elif _sz["moderate"]["total_cost"] <= 500:
                st.caption("Position < $500 — protective put not warranted.")
            else:
                st.caption("No suitable put found near the 5-10% OTM range.")
        with _hc2:
            st.markdown(f"**Portfolio Hedge — {_hg['hedge_etf_symbol']}** ({_hg['hedge_etf_desc']})")
            st.markdown(f"A 10% portfolio allocation of **${_hg['hedge_alloc_10pct']:,.2f}** to **{_hg['hedge_etf_symbol']}** provides inverse market exposure.")
            st.caption("Inverse ETFs decay over time and are best used as short-term hedges during elevated volatility.")

        st.divider()

        # ── Fundamental snapshot ─────────────────────────────────────────────
        st.markdown("**Fundamental Snapshot**")
        _fc1, _fc2, _fc3 = st.columns(3)
        def _fmt(v, fmt=".2f", suffix=""):
            return f"{v:{fmt}}{suffix}" if v is not None else "N/A"
        with _fc1:
            st.markdown(f"**Trailing P/E:** {_fmt(_fu['trailing_pe'])}")
            st.markdown(f"**Forward P/E:** {_fmt(_fu['forward_pe'])}")
            st.markdown(f"**PEG Ratio:** {_fmt(_fu['peg'])}")
            st.markdown(f"**P/B Ratio:** {_fmt(_fu['pb'])}")
        with _fc2:
            st.markdown(f"**Div Yield:** {_fmt(_fu['div_yield'], '.2f', '%')}")
            st.markdown(f"**Debt/Equity:** {_fmt(_fu['debt_equity'])}")
            st.markdown(f"**Rev Growth (YoY):** {_fmt(_fu['rev_growth'], '.1f', '%')}")
            st.markdown(f"**Earn Growth (YoY):** {_fmt(_fu['earn_growth'], '.1f', '%')}")
        with _fc3:
            st.markdown(f"**Short Interest:** {_fmt(_fu['short_pct'], '.1f', '%')}")
            _analysts_str = f"{_fu['num_analysts']} analysts" if _fu["num_analysts"] else "N/A"
            st.markdown(f"**Analyst Consensus:** {_fu['rec_label']} ({_analysts_str})")
            if _fu["rec_mean"] is not None:
                st.markdown(f"**Rating Mean:** {_fu['rec_mean']:.2f} (1=Strong Buy, 5=Sell)")

        st.divider()

        # ── Save trade idea with signal source + thesis + conviction ─────────
        st.markdown("**Save Trade Idea**")
        _save_col1, _save_col2 = st.columns([2, 1])
        with _save_col1:
            _signal_source = st.selectbox(
                "Signal Source",
                ["Value Recovery", "Growth Momentum", "Options", "Manual"],
                index=["Value Recovery", "Growth Momentum", "Options", "Manual"].index(
                    st.session_state.get("last_screen", "Manual")
                ) if st.session_state.get("last_screen", "Manual") in ["Value Recovery", "Growth Momentum", "Options", "Manual"] else 3,
                key="tb_signal_source",
            )
        with _save_col2:
            _conviction = st.slider("Conviction", 1, 5, 3, key="tb_conviction")
        _thesis = st.text_area(
            "Trade Thesis",
            placeholder="Why are you taking this trade? What could invalidate it?",
            max_chars=500,
            key="tb_thesis",
        )

        if st.button("Save Trade Idea", key="tb_save", disabled=st.session_state.get("_save_pending", False)):
            save_trade_idea(
                _tc,
                signal_source=_signal_source,
                thesis=_thesis,
                conviction=_conviction,
            )
            st.success(f"Trade idea for {_sym} saved ({_signal_source}, conviction {_conviction}/5).")

    # ── Watchlist ─────────────────────────────────────────────────────────────
    if not _watch_df.empty:
        st.divider()
        with st.expander(f"Watchlist ({len(_watch_df)} ideas)", expanded=True):
            _watch_cols = ["timestamp", "ticker", "entry", "target", "stop", "rr",
                           "setup_score", "upside_pct", "shares", "signal_source", "conviction"]
            _avail_cols = [c for c in _watch_cols if c in _watch_df.columns]
            _watch_display = _watch_df[_avail_cols].copy()
            _watch_display.columns = [
                c.replace("_", " ").title() for c in _avail_cols
            ]
            st.dataframe(_watch_display, use_container_width=True, hide_index=True)

    # ── Closed trades ────────────────────────────────────────────────────────
    if not _closed_df.empty:
        with st.expander(f"Closed Trades ({len(_closed_df)})", expanded=False):
            _closed_rows = []
            for _, _row in _closed_df.iterrows():
                _rpnl = _row.get("real_pnl")
                _rpnl_p = _row.get("real_pnl_pct")
                _closed_rows.append({
                    "Ticker": _row["ticker"],
                    "Entry": f"${float(_row['actual_entry_n']):.2f}" if _row.get("actual_entry_n") else "—",
                    "Exit": f"${float(_row['actual_exit_n']):.2f}" if _row.get("actual_exit_n") else "—",
                    "Shares": int(float(_row["shares_n"])) if _row.get("shares_n") else 0,
                    "Real. P&L": f"${_rpnl:+,.2f}" if _rpnl is not None and _rpnl == _rpnl else "—",
                    "Real. %": f"{_rpnl_p:+.2f}%" if _rpnl_p is not None and _rpnl_p == _rpnl_p else "—",
                    "Reason": _row.get("exit_reason", ""),
                    "Signal": _row.get("signal_source", ""),
                    "Date": _row["timestamp"],
                })
            st.dataframe(pd.DataFrame(_closed_rows), use_container_width=True, hide_index=True)

    # ── Full editable log ────────────────────────────────────────────────────
    st.divider()
    _log_df = load_trade_log()
    if not _log_df.empty:
        with st.expander("Edit Trade Log", expanded=False):
            st.caption(
                "Update **Actual Entry**, **Outcome Notes**, and **Status** inline. "
                "Use the **Close** button above for executed positions. "
                "Click **Save Changes** to persist."
            )
            _edit_cols = [c for c in _log_df.columns if c in [
                "id", "timestamp", "ticker", "entry", "target", "stop", "rr",
                "setup_score", "shares",
                "actual_entry", "outcome_notes", "status",
            ]]
            _edited = st.data_editor(
                _log_df[_edit_cols] if _edit_cols else _log_df,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "id": st.column_config.NumberColumn("ID", disabled=True),
                    "timestamp": st.column_config.TextColumn("Date", disabled=True),
                    "ticker": st.column_config.TextColumn("Ticker", disabled=True),
                    "entry": st.column_config.NumberColumn("Plan Entry", format="$%.2f", disabled=True),
                    "target": st.column_config.NumberColumn("Target", format="$%.2f", disabled=True),
                    "stop": st.column_config.NumberColumn("Stop", format="$%.2f", disabled=True),
                    "rr": st.column_config.NumberColumn("R/R", format="%.1f", disabled=True),
                    "setup_score": st.column_config.NumberColumn("Score", disabled=True),
                    "shares": st.column_config.NumberColumn("Shares", disabled=True),
                    "actual_entry": st.column_config.NumberColumn("Actual Entry", format="$%.2f"),
                    "outcome_notes": st.column_config.TextColumn("Notes", width="medium"),
                    "status": st.column_config.SelectboxColumn(
                        "Status",
                        options=["Watching", "Executed", "Cancelled"],
                    ),
                },
                key="trade_log_editor",
            )
            if st.button("Save Changes", key="tb_log_save"):
                save_trade_log(_edited)
                st.success("Trade log updated.")

# ── Backtest ──────────────────────────────────────────────────────────────────
with tab_backtest:
    import plotly.graph_objects as go
    import plotly.express as px

    st.subheader("Signal Backtester")
    st.warning(
        "**Disclaimer:** Backtests use historical price data and do not account for "
        "slippage, commissions, taxes, or survivorship bias (delisted stocks are not "
        "included unless you add them manually). Past performance does not guarantee "
        "future results. Results are for research purposes only."
    )

    # ── Inputs ────────────────────────────────────────────────────────────────
    _bt_c1, _bt_c2 = st.columns([2, 2])

    with _bt_c1:
        _bt_raw = st.text_area(
            "Tickers to test (one per line)",
            value="AAPL\nMSFT\nJPM\nXOM\nNVDA",
            height=140,
            key="bt_tickers",
        )
        _bt_tickers = [t.strip().upper() for t in _bt_raw.splitlines() if t.strip()]

        _bt_signal = st.selectbox(
            "Signal to test",
            list(SIGNAL_DESCRIPTIONS.keys()),
            key="bt_signal",
        )
        st.caption(f"*{SIGNAL_DESCRIPTIONS[_bt_signal]}*")

    with _bt_c2:
        _bt_holding_label = st.selectbox(
            "Holding period after signal",
            list(HOLDING_PERIOD_DAYS.keys()),
            index=1,
            key="bt_holding",
        )
        _bt_holding_days = HOLDING_PERIOD_DAYS[_bt_holding_label]

        _bt_lookback_label = st.selectbox(
            "Lookback period",
            list(LOOKBACK_PERIODS.keys()),
            index=1,
            key="bt_lookback",
        )
        _bt_period = LOOKBACK_PERIODS[_bt_lookback_label]

        _bt_custom_rsi = 35.0
        _bt_custom_dd  = 20.0
        if _bt_signal in ("RSI Oversold", "Custom"):
            _bt_custom_rsi = float(st.slider(
                "RSI threshold (signal fires when RSI < X)", 20, 60, 35, key="bt_rsi"
            ))
        if _bt_signal in ("Value Recovery", "Custom"):
            _bt_custom_dd = float(st.slider(
                "Drawdown threshold % (signal fires when drawdown > X%)", 5, 50, 20, key="bt_dd"
            ))

    _bt_run = st.button("Run Backtest", type="primary", key="bt_run")

    if _bt_run:
        if not _bt_tickers:
            st.error("Enter at least one ticker.")
        else:
            with st.spinner(
                f"Fetching {_bt_lookback_label} of data for "
                f"{len(_bt_tickers)} ticker(s) and scanning for signals…"
            ):
                _bt_result = run_backtest(
                    tickers=_bt_tickers,
                    signal_type=_bt_signal,
                    holding_days=_bt_holding_days,
                    period=_bt_period,
                    custom_rsi=_bt_custom_rsi,
                    custom_drawdown=_bt_custom_dd,
                )
            st.session_state["bt_result"] = _bt_result
            st.session_state["bt_params"] = {
                "signal": _bt_signal,
                "holding": _bt_holding_label,
                "lookback": _bt_lookback_label,
                "tickers": _bt_tickers,
            }

    if "bt_result" not in st.session_state:
        st.info("Configure inputs above and click **Run Backtest**.")
    else:
        _bt_result = st.session_state["bt_result"]
        _bt_params = st.session_state.get("bt_params", {})

        if "error" in _bt_result:
            st.error(_bt_result["error"])
        else:
            _trades_df = _bt_result["trades"]
            _metrics   = _bt_result["metrics"]
            _spy_series = _bt_result["spy_series"]

            st.caption(
                f"Signal: **{_bt_params.get('signal','')}** | "
                f"Hold: **{_bt_params.get('holding','')}** | "
                f"Lookback: **{_bt_params.get('lookback','')}** | "
                f"Tickers: {', '.join(_bt_params.get('tickers',[]))}"
            )
            st.divider()

            # ── Summary metrics ───────────────────────────────────────────────
            st.markdown("**Performance Summary**")
            _sm1, _sm2, _sm3, _sm4 = st.columns(4)
            _sm1.metric("Total Signals", _metrics["total_signals"])
            _sm2.metric(
                "Win Rate",
                f"{_metrics['win_rate']:.1f}%",
                delta="Good" if _metrics["win_rate"] >= 55 else "Below 55%",
                delta_color="normal" if _metrics["win_rate"] >= 55 else "inverse",
            )
            _sm3.metric(
                "Avg Return / Signal",
                f"{_metrics['avg_return']:+.2f}%",
                delta_color="normal" if _metrics["avg_return"] > 0 else "inverse",
            )
            _sm4.metric(
                "vs SPY Buy & Hold",
                f"{_metrics['spy_return']:+.1f}%" if _metrics["spy_return"] is not None else "N/A",
                delta=(
                    f"Strategy avg {_metrics['avg_return']:+.2f}%"
                    if _metrics["spy_return"] is not None else None
                ),
                delta_color="off",
            )

            _sm5, _sm6, _sm7, _sm8 = st.columns(4)
            _sm5.metric("Median Return", f"{_metrics['median_return']:+.2f}%")
            _sm6.metric(
                "Profit Factor",
                f"{_metrics['profit_factor']:.2f}" if _metrics["profit_factor"] else "N/A",
                help="Gross wins / gross losses. Above 1.0 is profitable overall.",
            )
            _sm7.metric(
                "Sharpe-like Ratio",
                f"{_metrics['sharpe_like']:.3f}" if _metrics["sharpe_like"] is not None else "N/A",
                help="Mean return ÷ std deviation of returns. Higher is better.",
            )
            _sm8.metric("Max Consec. Losses", _metrics["max_consec_losses"])

            _sm9, _sm10, _sm11, _sm12 = st.columns(4)
            _sm9.metric("Best Trade",  f"{_metrics['best_trade']:+.2f}%")
            _sm10.metric("Worst Trade", f"{_metrics['worst_trade']:+.2f}%")
            _sm11.metric("Avg Win",  f"{_metrics['avg_win']:+.2f}%")
            _sm12.metric("Avg Loss", f"{_metrics['avg_loss']:+.2f}%")

            st.divider()

            # ── Charts ────────────────────────────────────────────────────────
            _chart_c1, _chart_c2 = st.columns([1, 1])

            with _chart_c1:
                st.markdown("**Return Distribution**")
                _fig_hist = px.histogram(
                    _trades_df,
                    x="Return %",
                    nbins=max(10, min(50, len(_trades_df) // 2)),
                    color_discrete_sequence=["#4a9eff"],
                    labels={"Return %": "Return per Trade (%)"},
                )
                _fig_hist.add_vline(
                    x=0,
                    line_dash="solid",
                    line_color="red",
                    annotation_text="0%",
                    annotation_position="top right",
                )
                _fig_hist.add_vline(
                    x=_metrics["avg_return"],
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"Avg {_metrics['avg_return']:+.1f}%",
                    annotation_position="top left",
                )
                _fig_hist.add_vline(
                    x=_metrics["median_return"],
                    line_dash="dot",
                    line_color="yellow",
                    annotation_text=f"Med {_metrics['median_return']:+.1f}%",
                    annotation_position="bottom right",
                )
                _fig_hist.update_layout(
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#1a1f2e",
                    font_color="#fafafa",
                    margin=dict(l=10, r=10, t=30, b=10),
                    showlegend=False,
                )
                st.plotly_chart(_fig_hist, use_container_width=True)

            with _chart_c2:
                st.markdown("**Cumulative Return: Strategy vs SPY Buy & Hold**")
                _fig_cum = go.Figure()

                # Strategy equity curve
                _fig_cum.add_trace(go.Scatter(
                    x=_metrics["cum_dates"],
                    y=_metrics["cum_equity"],
                    mode="lines+markers",
                    name="Strategy (sequential trades)",
                    line=dict(color="#4a9eff", width=2),
                    marker=dict(size=4),
                ))

                # SPY normalized line
                if not _spy_series.empty:
                    _fig_cum.add_trace(go.Scatter(
                        x=_spy_series.index,
                        y=_spy_series.values,
                        mode="lines",
                        name="SPY Buy & Hold",
                        line=dict(color="#ff9944", width=1.5, dash="dash"),
                    ))

                _fig_cum.add_hline(
                    y=100,
                    line_dash="dot",
                    line_color="gray",
                    annotation_text="Starting capital",
                )
                _fig_cum.update_layout(
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#1a1f2e",
                    font_color="#fafafa",
                    yaxis_title="Equity (start = 100)",
                    xaxis_title="Date",
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(_fig_cum, use_container_width=True)

            st.divider()

            # ── Trade table ───────────────────────────────────────────────────
            st.markdown("**All Signal Occurrences**")
            _display_trades = _trades_df.copy()
            _display_trades["Win/Loss"] = _display_trades["Win"].map(
                {True: "Win", False: "Loss"}
            )
            _display_trades = _display_trades.drop(columns=["Win"])

            def _color_return(val):
                if isinstance(val, (int, float)):
                    return "color: #4caf50" if val > 0 else "color: #f44336"
                return ""

            _styled_trades = _display_trades.style.applymap(
                _color_return, subset=["Return %"]
            )
            st.dataframe(
                _styled_trades,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Ticker":       st.column_config.TextColumn("Ticker"),
                    "Entry Date":   st.column_config.TextColumn("Entry Date"),
                    "Exit Date":    st.column_config.TextColumn("Exit Date"),
                    "Entry Price":  st.column_config.NumberColumn("Entry", format="$%.2f"),
                    "Exit Price":   st.column_config.NumberColumn("Exit",  format="$%.2f"),
                    "Return %":     st.column_config.NumberColumn("Return %", format="%+.2f%%"),
                    "Win/Loss":     st.column_config.TextColumn("Result"),
                },
            )
