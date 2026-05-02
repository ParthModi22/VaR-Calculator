"""
Phase 1: Fetch 3 years of daily price data for Indian equities and compute returns.
Saves cleaned daily percentage returns to returns.csv for use by all subsequent scripts.
"""

import yfinance as yf
import pandas as pd

TICKERS = ["HDFCBANK.NS", "RELIANCE.NS", "SBIN.NS"]
PERIOD = "5y"
OUTPUT_FILE = "returns.csv"


def fetch_returns(tickers: list[str], period: str) -> pd.DataFrame:
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    returns = raw.pct_change().dropna()
    # Flatten multi-level columns if present
    returns.columns = [c.replace(".NS", "") for c in returns.columns]
    return returns


def portfolio_returns(returns: pd.DataFrame) -> pd.Series:
    """Equally weighted portfolio daily return."""
    weights = 1 / len(returns.columns)
    return returns.mul(weights).sum(axis=1).rename("Portfolio")


if __name__ == "__main__":
    print(f"Fetching {PERIOD} of data for: {', '.join(TICKERS)}")
    stock_returns = fetch_returns(TICKERS, PERIOD)
    port_returns = portfolio_returns(stock_returns)

    result = pd.concat([stock_returns, port_returns], axis=1)
    result.to_csv(OUTPUT_FILE)
    print(f"Saved {len(result)} rows to {OUTPUT_FILE}")
    print(result.tail())
