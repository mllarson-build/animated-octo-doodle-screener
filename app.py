import streamlit as st
from datetime import datetime
from screener.value import fetch_value_data

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
    st.info("Growth Momentum screener coming soon.")

with tab_options:
    st.subheader("Options")
    st.info("Options screener coming soon.")

with tab_etfs:
    st.subheader("ETFs")
    st.info("ETF screener coming soon.")
