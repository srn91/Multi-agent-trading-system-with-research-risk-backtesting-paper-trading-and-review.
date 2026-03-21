"""
Data ingestion module.
Downloads and caches OHLCV data for US equities.
Uses yfinance for prototype. Replace with paid source for production.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_ohlcv(
    ticker: str,
    start: str = "2010-01-01",
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Download daily OHLCV for a single ticker.
    Caches to parquet. Set force=True to re-download.
    """
    ensure_dirs()
    cache_path = RAW_DIR / f"{ticker}.parquet"

    if cache_path.exists() and not force:
        df = pd.read_parquet(cache_path)
        logger.info(f"Loaded {ticker} from cache: {len(df)} rows")
        return df

    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"Downloading {ticker} from {start} to {end}")
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if data.empty:
        logger.warning(f"No data returned for {ticker}")
        return pd.DataFrame()

    # Flatten multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index.name = "Date"
    data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    data.columns = ["open", "high", "low", "close", "volume"]
    data = data.dropna()

    data.to_parquet(cache_path)
    logger.info(f"Saved {ticker}: {len(data)} rows")
    return data


def download_universe(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV for a list of tickers."""
    results = {}
    for ticker in tickers:
        try:
            df = download_ohlcv(ticker, start=start, end=end, force=force)
            if not df.empty:
                results[ticker] = df
        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")
    logger.info(f"Downloaded {len(results)}/{len(tickers)} tickers")
    return results


def get_spy(start: str = "2010-01-01", end: str | None = None, force: bool = False) -> pd.DataFrame:
    """Get SPY data for market regime filter."""
    return download_ohlcv("SPY", start=start, end=end, force=force)


# Default liquid US stock universe for testing
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "JNJ", "V", "UNH", "HD", "PG", "MA", "DIS", "ADBE", "CRM",
    "NFLX", "COST", "PEP", "TMO", "ABT", "ACN", "MRK", "LLY", "AVGO",
    "TXN", "QCOM", "ORCL", "AMD", "INTC", "CSCO", "IBM", "NOW", "AMAT",
    "ISRG", "BKNG", "ADP", "MDLZ", "GILD", "CME", "GS", "BLK", "SCHW",
    "MMM", "CAT", "DE", "BA", "RTX", "LMT", "GE", "HON", "UPS", "FDX",
    "LOW", "TGT", "WMT", "NKE", "SBUX", "MCD", "YUM", "CMG", "DHR",
    "SYK", "ZTS", "EL", "CL", "KO", "PFE", "ABBV", "BMY", "AMGN",
    "REGN", "VRTX", "MRNA", "PANW", "CRWD", "SNOW", "DDOG", "ZS",
    "NET", "FTNT", "MELI", "SQ", "PYPL", "SHOP", "ABNB", "UBER",
    "DASH", "COIN", "ROKU", "SNAP", "PINS", "TTD", "ENPH", "SEDG",
    "FSLR", "CEG", "VST", "XOM", "CVX", "COP", "SLB", "EOG",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    spy = get_spy()
    print(f"SPY: {len(spy)} rows, {spy.index[0]} to {spy.index[-1]}")
