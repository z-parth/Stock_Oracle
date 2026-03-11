import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Folder where trained models will be saved
MODEL_CACHE_DIR = "data/models"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


def prepare_features(df):
    """
    Engineers features from raw stock data for the model to learn from.
    Instead of just feeding raw prices, we create indicators like moving averages,
    price changes, and volatility which help the model find patterns.

    Args:
        df (pd.DataFrame): Raw stock data with OHLCV columns

    Returns:
        pd.DataFrame: Data with added feature columns
    """
    df = df.copy()

    # Price-based features
    df['Return_1d']  = df['Close'].pct_change(1)   # 1 day % change
    df['Return_5d']  = df['Close'].pct_change(5)   # 5 day % change
    df['Return_10d'] = df['Close'].pct_change(10)  # 10 day % change

    # Moving averages
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    # Distance from moving averages (tells model if price is above/below average)
    df['Close_vs_MA10'] = df['Close'] / df['MA_10']
    df['Close_vs_MA20'] = df['Close'] / df['MA_20']

    # Volatility (how much the stock swings)
    df['Volatility_10d'] = df['Return_1d'].rolling(window=10).std()

    # Volume change
    df['Volume_Change'] = df['Volume'].pct_change(1)

    # Target — what we want to predict (next day's closing price)
    df['Target'] = df['Close'].shift(-1)

    # Drop rows with NaN values created by rolling calculations
    df.dropna(inplace=True)

    return df


def train_model(ticker, df, force_retrain=False):
    """
    Trains a Random Forest model for the given ticker.
    - If a saved model exists, loads it (fast).
    - If not, trains a new one and saves it.

    Args:
        ticker (str): Stock ticker symbol e.g. "AAPL"
        df (pd.DataFrame): Stock data from fetch_data.py
        force_retrain (bool): If True, retrains even if a saved model exists

    Returns:
        model: Trained RandomForestRegressor
        dict: Model performance metrics
    """

    ticker = ticker.upper().strip()
    model_file = os.path.join(MODEL_CACHE_DIR, f"{ticker}_model.pkl")

    # Step 1 - Check if model already exists
    if os.path.exists(model_file) and not force_retrain:
        print(f"[CACHE] Loading existing model for {ticker}...")
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print(f"[CACHE] Model loaded successfully for {ticker}")
        return model, None

    # Step 2 - Prepare features
    print(f"[TRAINING] Preparing features for {ticker}...")
    df = prepare_features(df)

    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Return_1d', 'Return_5d', 'Return_10d',
        'MA_10', 'MA_20', 'MA_50',
        'Close_vs_MA10', 'Close_vs_MA20',
        'Volatility_10d', 'Volume_Change'
    ]

    X = df[feature_cols]
    y = df['Target']

    # Step 3 - Split data (80% train, 20% test)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Step 4 - Train the Random Forest
    print(f"[TRAINING] Training Random Forest for {ticker}... (this may take 10-30 seconds)")
    model = RandomForestRegressor(
        n_estimators=200,    
        max_depth=10,        
        random_state=42,     
        n_jobs=-1            
    )
    model.fit(X_train, y_train)

    # Step 5 - Evaluate performance
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2  = r2_score(y_test, predictions)

    metrics = {
        "MAE": round(mae, 4),   
        "R2":  round(r2, 4),    
        "Ticker": ticker,
        "Training rows": len(X_train),
        "Test rows": len(X_test)
    }

    print(f"[DONE] Model trained for {ticker}")
    print(f"       MAE  : ${mae:.2f} average error")
    print(f"       R²   : {r2:.4f} (closer to 1.0 is better)")

    # Step 6 - Save model to disk
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"[SAVED] Model saved to {model_file}")

    return model, metrics


if __name__ == "__main__":
    # Test the full pipeline
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from src.data.fetch_data import fetch_stock_data

    test_tickers = ["AAPL", "TSLA"]

    for ticker in test_tickers:
        print(f"\n{'='*50}")
        print(f"Processing {ticker}...")
        df = fetch_stock_data(ticker)
        if df is not None:
            model, metrics = train_model(ticker, df)
            if metrics:
                print(f"\nMetrics: {metrics}")