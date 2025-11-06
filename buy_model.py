import pandas as pd
import xgboost as xgb
import logging

# Saves Back to suggested db just the  symbol and prediction.

logging.basicConfig(level=logging.INFO)

# Required bought columns orchestrator passes
REQUIRED_COLUMNS = [
    "open", "high", "low", "close", "volume", "quote_asset_volume",
    "number_of_trades", "taker_buy_base", "taker_buy_quote", "macd",
    "macd_signal", "macd_histogram", "rsi", "rsi_sma", "ema_100",
    "ema_200", "atr", "relative_volume", "quote_volume_ratio",
    "buy_sell_pressure", "ema_ratio", "rsi_x_relative_volume",
    "macd_histogram_x_atr", "buy_sell_pressure_x_ema_ratio", "relative_volume", "rsi_x_relative_volume"
]

def run(inputs):
    """
    Stage 2 model prediction logic.

    inputs dict expects:
        - 'data': DataFrame with latest features per coin
        - 'model_file': path to trained XGBoost model JSON file
    """

    df = inputs.get("data")
    model_file = inputs.get("model_file")

    # -------------------------- Input validation --------------------------
    if df is None or df.empty:
        logging.error("Input data is missing or empty")
        return {"status": "error", "message": "No input data"}

    if not model_file:
        logging.error("Missing model_file path")
        return {"status": "error", "message": "model_file required"}

    # -------------------------- Load trained XGBoost model --------------------------
    try:
        model = xgb.Booster()
        model.load_model(model_file)
        logging.info(f"Loaded XGBoost model from {model_file}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return {"status": "error", "message": "Failed to load model"}

    # -------------------------- Ensure required columns --------------------------
    feature_cols = [c for c in REQUIRED_COLUMNS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        logging.error("No valid feature columns found from required columns.")
        return {"status": "error", "message": "No feature columns available for prediction"}

    # -------------------------- Prepare features for XGBoost --------------------------
    X = df[feature_cols]

    # -------------------------- Run prediction --------------------------
    try:
        dmatrix = xgb.DMatrix(X)
        preds = model.predict(dmatrix)  # probability/confidence
        df["confidence_score"] = preds
        df["prediction"] = "BUY"  # default action
        logging.info("Predictions and confidence scores generated for all symbols")
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return {"status": "error", "message": "Prediction failed"}

    # -------------------------- Prepare output --------------------------
    # Orchestrator will add symbol, so we only keep prediction & confidence_score here
    columns_to_return = ["prediction", "confidence_score"]
    df_to_return = df[columns_to_return]

    return {"status": "success", "suggested_trades": df_to_return}
