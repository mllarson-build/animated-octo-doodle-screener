import pandas as pd
import streamlit as st
from datetime import datetime
from screener.value import fetch_value_data
from screener.growth import fetch_growth_data
from screener.options import fetch_options_data

st.set_page_config(page_title="Daily Screener", layout="wide")

DEFAULT_TICKERS = "AAPL\nMSFT\nJPM\nXOM\nBAC\nNVDA\nSPY\nQQQ\nIWM\nTLT\nGLD"

# --- Sidebar ---
with st.sidebar:
    st.header("Tickers")
    ticker_input = st.text_area(
        "Paste tickers (one per line)",
        value=DEFAULT_TICKERS,
        height=250,
    )
    refresh = st.button("Refresh Data", use_container_width=True)

    if refresh:
        tickers = [t.strip().upper() for t in ticker_input.splitlines() if t.strip()]
        with st.spinner(f"Loading data for {len(tickers)} tickers…"):
            st.session_state["tickers"] = tickers
            st.session_state["value_df"] = fetch_value_data(tickers)
            st.session_state["growth_df"] = fetch_growth_data(tickers)
            st.session_state["last_refreshed"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    if "last_refreshed" in st.session_state:
        st.caption(f"Last refreshed: {st.session_state['last_refreshed']}")

# --- Main tabs ---
tab_value, tab_growth, tab_options, tab_etfs = st.tabs(
    ["Value Recovery", "Growth Momentum", "Options", "ETFs"]
)

with tab_value:
    st.subheader("Value Recovery")
    if "value_df" not in st.session_state:
        st.info("Press **Refresh Data** in the sidebar to run the screener.")
    else:
        df = st.session_state["value_df"]
        total = len(df)
        high_conviction = int((df["recovery_score"] >= 3).sum())
        col1, col2 = st.columns(2)
        col1.metric("Tickers Screened", total)
        col2.metric("Scored 3+", high_conviction)

        def color_score(val):
            if val >= 4:
                return "background-color: #1a7a1a; color: white"
            elif val >= 2:
                return "background-color: #7a7a00; color: white"
            else:
                return "background-color: #7a1a1a; color: white"

        styled = df.style.applymap(color_score, subset=["recovery_score"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

with tab_growth:
    st.subheader("Growth Momentum")
    if "growth_df" not in st.session_state:
        st.info("Press **Refresh Data** in the sidebar to run the screener.")
    else:
        df = st.session_state["growth_df"]
        total = len(df)
        high_conviction = int((df["growth_score"] >= 3).sum())
        col1, col2 = st.columns(2)
        col1.metric("Tickers Screened", total)
        col2.metric("Scored 3+", high_conviction)

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

        styled = df.style.applymap(color_upside, subset=["Upside %"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

with tab_options:
    st.subheader("Options")

    # Ticker list comes from watchlist; fall back to defaults if no refresh yet
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
        st.caption(f"Showing options for **{loaded_for}** — current price: **${data['current_price']:.2f}**")

        selected_exp = st.selectbox("Expiration", data["expirations"], key="opt_exp")

        chain_key = "calls" if option_type == "Calls" else "puts"
        df = data["chains"][selected_exp][chain_key]

        if df.empty:
            st.info("No contracts found for this selection.")
        else:
            flagged = int(df["Flag"].sum())
            if flagged:
                st.warning(f"{flagged} contract(s) flagged for unusual activity (Vol/OI > 1.5, near money, volume > 100)")

            def highlight_flag(row):
                if df.loc[row.name, "Flag"]:
                    return ["background-color: #4a3800; color: white"] * len(row)
                return [""] * len(row)

            display_df = df.drop(columns=["Flag"])
            styled = display_df.style.apply(highlight_flag, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)

with tab_etfs:
    st.subheader("ETFs")
    st.info("ETF screener coming soon.")
