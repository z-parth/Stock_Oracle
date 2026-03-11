import yfinance as yf
import pandas as pd
import os

# Folder where downloaded stock data will be saved
DATA_CACHE_DIR = "data/cache"
os.makedirs(DATA_CACHE_DIR, exist_ok=True)


def fetch_stock_data(ticker, period="2y", force_refresh=False):
    """
    Fetches stock data for a given ticker.
    - If data is already cached locally, loads it from disk (fast).
    - If not cached or force_refresh=True, downloads fresh data from Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker e.g. "AAPL", "TSLA", "RELIANCE.NS"
        period (str): How much history to fetch e.g. "1y", "2y", "5y"
        force_refresh (bool): If True, re-downloads even if cache exists

    Returns:
        pd.DataFrame: Stock data with Open, High, Low, Close, Volume columns
    """

    ticker = ticker.upper().strip()
    cache_file = os.path.join(DATA_CACHE_DIR, f"{ticker}.csv")

    # Step 1 - Check if cached data exists
    if os.path.exists(cache_file) and not force_refresh:
        print(f"[CACHE] Loading {ticker} data from cache...")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"[CACHE] Loaded {len(df)} rows for {ticker}")
        return df

    # Step 2 - Cache miss, download fresh data
    print(f"[DOWNLOAD] Fetching {ticker} data from Yahoo Finance...")
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            raise ValueError(f"No data found for ticker '{ticker}'. Please check the symbol.")

        # Keep only the columns we need
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.dropna(inplace=True)

        # Step 3 - Save to cache for future use
        df.to_csv(cache_file)
        print(f"[SAVED] {ticker} data cached to {cache_file} ({len(df)} rows)")

        return df

    except Exception as e:
        print(f"[ERROR] Could not fetch data for '{ticker}': {e}")
        return None


if __name__ == "__main__":
    # Test with a few tickers
    test_tickers = ["AAPL", "TSLA", "MSFT"]

    for ticker in test_tickers:
        print(f"\n{'='*40}")
        df = fetch_stock_data(ticker)
        if df is not None:
            print(df.tail(3))
            print(f"Shape: {df.shape}")