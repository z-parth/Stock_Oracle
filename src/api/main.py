import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data.fetch_data import fetch_stock_data
from src.model.train_model import train_model
from src.model.predict import predict_next_price


app = FastAPI(
    title="Stock Price Predictor API",
    description="Predicts next day closing price using Random Forest",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)



class PredictRequest(BaseModel):
    ticker: str             
    force_retrain: bool = False 
    force_refresh: bool = False  


class PredictResponse(BaseModel):
    ticker: str
    current_price: float
    predicted_price: float
    change: float
    change_pct: float
    direction: str          
    status: str             



@app.get("/")
def root():
    return {"message": "Stock Predictor API is running!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    
    ticker = request.ticker.upper().strip()

    #fetch data
    print(f"\n[API] Received prediction request for {ticker}")
    df = fetch_stock_data(ticker, force_refresh=request.force_refresh)

    if df is None or df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Could not fetch data for ticker '{ticker}'. Please check the symbol."
        )

    #train/load model
    model_path = f"data/models/{ticker}_model.pkl"
    already_trained = os.path.exists(model_path) and not request.force_retrain

    model, metrics = train_model(ticker, df, force_retrain=request.force_retrain)

    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to train or load model for '{ticker}'."
        )

    status = "loaded_from_cache" if already_trained else "freshly_trained"

    #predict
    result = predict_next_price(ticker, df)

    if result is None:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed for '{ticker}'."
        )

    #response
    return PredictResponse(
        ticker=result["ticker"],
        current_price=result["current_price"],
        predicted_price=result["predicted_price"],
        change=result["change"],
        change_pct=result["change_pct"],
        direction=result["direction"],
        status=status
    )


@app.get("/models")
def list_models():
    """Returns a list of all tickers that have been trained and cached."""
    model_dir = "data/models"
    if not os.path.exists(model_dir):
        return {"trained_models": []}

    models = [
        f.replace("_model.pkl", "")
        for f in os.listdir(model_dir)
        if f.endswith("_model.pkl")
    ]
    return {"trained_models": sorted(models)}