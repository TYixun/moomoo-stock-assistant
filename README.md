# Stock Assistant & Moomoo Trading Agent

A Streamlit market research dashboard and guarded automated/AI paper-trading assistant for US equities and ETFs.

> **Disclaimer:** This project is an analysis and paper-trading research tool, not financial advice. Past backtest or model performance does not predict future results.

## Features

- **Live Watchlist Strip**: Real-time/delayed US quotes (Yahoo Finance or Moomoo OpenD), intraday sparklines, market-hours status, and clickable cards that jump straight to the **Analysis** tab.
- **Light & Dark Themes**: Runtime **Dark mode** toggle in the top-right header bar.
- **Analysis**: Interactive Plotly candlestick + volume + SMA (20/50/200) + RSI (14) charts, KPI metrics, valuation/trend context, watchlist comparison table, and optional Claude summary.
- **Trading Agent**:
  - **Automated Strategy Runner**: Periodic or on-demand scanning using RSI mean reversion, Price vs SMA 50/200, or `ml_signal.latest_signal`.
  - **Claude AI Assistant (`agent.py`)**: Tool-using assistant that inspects quotes, positions, and technical indicators to propose cautious trades.
  - **Guardrails & Kill Switch (`guardrails.py`, `moomoo_exec.py`)**: Enforces allowed tickers, max order USD, daily order caps, price deviation limits, US market hours, a `STOP` kill-switch file, and `MOOMOO_LIVE=1` gating for real orders.
  - **Execution Modes**: Dry Run (local simulation), Paper (`SIMULATE` via Moomoo OpenD), and Real (`REAL`, gated by `MOOMOO_LIVE=1` and `MOOMOO_TRADE_PWD`).
- **Moomoo Account**: Read-only OpenD account balances, positions, Unrealised P/L, and holdings allocation chart.
- **News & Sentiment**: Yahoo Finance headlines with optional Claude sentiment scoring.
- **Backtest**: Strategy simulator for SMA crossover, Price above SMA, and RSI mean reversion against buy-and-hold after fees.
- **ML Signal**: Walk-forward out-of-sample Logistic Regression and HistGradientBoosting evaluation (`ml_signal.py`).
- **Manual Tracker**: Session position tracker with USD/MYR conversion and CSV import/export.

## Project Structure

- `app.py` — Main Streamlit application and UI
- `stock_assistant.py` — Core `yfinance` / `pandas` technical analysis helper & CLI script
- `ml_signal.py` — Walk-forward ML feature builder, training, and evaluation
- `agent.py` — Anthropic Claude tool-use agent for order proposals
- `guardrails.py` — Plain-Python trade validation rules and `STOP` kill switch
- `moomoo_client.py` — Read-only wrapper around Moomoo OpenD (`127.0.0.1:11111`)
- `moomoo_exec.py` — Order execution wrapper with `SIMULATE` default and `orders_log.csv` logging
- `paper_test.py` — Minimal Moomoo paper-order smoke test
- `test.py` — Minimal Moomoo quote context connectivity test
- `.streamlit/config.toml` — Streamlit base theme configuration

## Quick Start

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Optional Configuration

- **Claude AI features**: Set `ANTHROPIC_API_KEY` in your environment or paste it in the sidebar (stored in session only).
- **Moomoo OpenD**: Run and log into Moomoo OpenD locally on `127.0.0.1:11111` for live quotes, account balances, and paper (`SIMULATE`) orders.
- **Live Trading Gate (`REAL`)**: Live execution in `moomoo_exec.py` is disabled unless `MOOMOO_LIVE=1` and `MOOMOO_TRADE_PWD` are set in your environment.
