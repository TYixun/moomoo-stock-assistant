"""
Simple stock analysis assistant.
- Pulls price data with yfinance
- Computes indicators with pandas
- (Optional) asks Claude to summarize the numbers in plain language

Install:  pip install yfinance pandas anthropic
Optional: export ANTHROPIC_API_KEY=your_key   (skip it to just print the numbers)

This is an analysis tool, NOT a price predictor. Not financial advice.
"""
import os
import pandas as pd
import yfinance as yf

WATCHLIST = ["QQQ", "VOO", "NVDA"]


def rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return float(100 - 100 / (1 + rs.iloc[-1]))


def analyze(ticker: str) -> dict:
    df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
    close = df["Close"].squeeze().dropna()
    last = float(close.iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])
    year = close.iloc[-252:]
    daily_ret = close.pct_change().dropna()

    info = {}
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        pass  # fundamentals are optional

    return {
        "ticker": ticker,
        "price": round(last, 2),
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        "above_sma50": last > sma50,
        "above_sma200": last > sma200,
        "rsi14": round(rsi(close), 1),
        "high_52w": round(float(year.max()), 2),
        "low_52w": round(float(year.min()), 2),
        "pct_from_52w_high": round((last / float(year.max()) - 1) * 100, 1),
        "return_1m_pct": round((last / float(close.iloc[-21]) - 1) * 100, 1),
        "volatility_ann_pct": round(float(daily_ret.std() * (252 ** 0.5) * 100), 1),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
    }


def summarize_with_claude(rows: list[dict]) -> str:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    prompt = (
        "You are a cautious market analysis assistant. Using ONLY the data below, "
        "write a short plain-language summary for each ticker: trend, momentum, "
        "valuation (if available) and main risks. Do not predict prices, do not "
        "invent numbers, and remind me this is not financial advice.\n\n"
        f"{rows}"
    )
    msg = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


if __name__ == "__main__":
    results = [analyze(t) for t in WATCHLIST]
    print(pd.DataFrame(results).set_index("ticker").T.to_string())

    if os.getenv("ANTHROPIC_API_KEY"):
        print("\n--- Claude summary ---\n")
        print(summarize_with_claude(results))
    else:
        print("\n(Set ANTHROPIC_API_KEY to get a written summary.)")
