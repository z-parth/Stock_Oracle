import os
import pickle
import numpy as np
import pandas as pd

MODEL_CACHE_DIR = "data/models"


def load_model(ticker):
    """
    Loads a saved model for the given ticker.

    Args:
        ticker (str): Stock ticker e.g. "AAPL"

    Returns:
        model: Trained RandomForestRegressor or None if not found
    """
    ticker = ticker.upper().strip()
    model_file = os.path.join(MODEL_CACHE_DIR, f"{ticker}_model.pkl")

    if not os.path.exists(model_file):
        print(f"[ERROR] No saved model found for {ticker}. Train it first.")
        return None

    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    print(f"[LOADED] Model loaded for {ticker}")
    return model


def prepare_latest_features(df):
    """
    Prepares features from the most recent row of stock data.
    This is what gets fed into the model to predict tomorrow's price.

    Args:
        df (pd.DataFrame): Stock data with OHLCV columns

    Returns:
        pd.DataFrame: Single row of features ready for prediction
    """
    df = df.copy()

    # Recreate the same features used during training
    df['Return_1d']  = df['Close'].pct_change(1)
    df['Return_5d']  = df['Close'].pct_change(5)
    df['Return_10d'] = df['Close'].pct_change(10)

    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    df['Close_vs_MA10'] = df['Close'] / df['MA_10']
    df['Close_vs_MA20'] = df['Close'] / df['MA_20']

    df['Volatility_10d'] = df['Return_1d'].rolling(window=10).std()
    df['Volume_Change']  = df['Volume'].pct_change(1)

    df.dropna(inplace=True)

    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Return_1d', 'Return_5d', 'Return_10d',
        'MA_10', 'MA_20', 'MA_50',
        'Close_vs_MA10', 'Close_vs_MA20',
        'Volatility_10d', 'Volume_Change'
    ]

    # Return only the last row (most recent data = basis for next day prediction)
    return df[feature_cols].iloc[[-1]]


def predict_next_price(ticker, df):
    """
    Main function — predicts the next day's closing price for a ticker.

    Args:
        ticker (str): Stock ticker e.g. "AAPL"
        df (pd.DataFrame): Recent stock data from fetch_data.py

    Returns:
        dict: Prediction result with current price, predicted price, and change %
    """
    ticker = ticker.upper().strip()

    # Load the saved model
    model = load_model(ticker)
    if model is None:
        return None

    # Prepare the latest features
    latest_features = prepare_latest_features(df)

    # Make prediction
    predicted_price = model.predict(latest_features)[0]
    current_price   = df['Close'].iloc[-1]
    change          = predicted_price - current_price
    change_pct      = (change / current_price) * 100
    direction       = "UP" if change > 0 else "DOWN"

    result = {
        "ticker":          ticker,
        "current_price":   round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "change":          round(change, 2),
        "change_pct":      round(change_pct, 2),
        "direction":       direction
    }

    return result


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from src.data.fetch_data import fetch_stock_data
    from src.model.train_model import train_model

    ticker = "AAPL"

    # Fetch data
    df = fetch_stock_data(ticker)

    # Train if not already trained
    train_model(ticker, df)

    # Predict
    result = predict_next_price(ticker, df)

    if result:
        print(f"\n{'='*40}")
        print(f"  Ticker          : {result['ticker']}")
        print(f"  Current Price   : ${result['current_price']}")
        print(f"  Predicted Price : ${result['predicted_price']}")
        print(f"  Change          : ${result['change']} ({result['change_pct']}%)")
        print(f"  Direction       : {result['direction']}")
        print(f"{'='*40}")