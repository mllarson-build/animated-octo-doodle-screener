# Stock Screener Project

## What this is
A personal daily stock and options screening dashboard built in Streamlit.
Target deployment: Streamlit Community Cloud (requires public GitHub repo).
Data sources: yfinance only (free tier). No paid APIs.

## Owner context
- Personal Roth IRA, ~$8K portfolio
- Medium-term trades (weeks to months), not day trading
- Goal: identify 3-6 concentrated bets at a time
- Two strategies: Value Recovery and Growth Momentum
- Options: simple calls/puts only, near the money, liquid names
- ETFs: SPY, QQQ, IWM, XLF, XLK, XLE, XLV, XBI, GLD, TLT, HYG

## Architecture rules
- Entry point is app.py (Streamlit)
- All logic lives in screener/ modules, never inline in app.py
- No paid data APIs — yfinance and pandas-ta only
- requirements.txt must stay current after every new library added
- Python 3.11 compatible
- Keep UI simple and fast — this is a daily morning tool

## File structure
app.py                  # Streamlit entry point
screener/value.py       # Value recovery screen logic
screener/growth.py      # Growth momentum screen logic
screener/options.py     # Options chain scanner
screener/etf.py         # ETF dashboard
screener/utils.py       # Shared helpers
data/watchlist.csv      # Default ticker list

## Current status
- [ ] Session 1: App shell
- [ ] Session 2: Value recovery screen
- [ ] Session 3: Growth momentum screen
- [ ] Session 4: Options scanner
- [ ] Session 5: ETF dashboard
- [ ] Session 6: Streamlit Cloud deployment
