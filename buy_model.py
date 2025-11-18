import pandas as pd
import xgboost as xgb
import logging

logging.basicConfig(level=logging.INFO)

REQUIRED_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "macd", "macd_signal", "macd_histogram",
    "rsi", "rsi_sma",
    "ema_100", "ema_200",
    "atr", "ema_ratio",
    "macd_histogram_x_atr", "buy_sell_pressure_x_ema_ratio",
    "buy_sell_pressure", "relative_volume",
    "quote_volume_ratio", "rsi_x_relative_volume"
]


FEATURE_PREFIX = "feature_"  # <-- your prefix here

def run(inputs):

    df = inputs.get("data")
    model_file = inputs.get("model_file")

    if df is None or df.empty:
        logging.error("Input data is missing or empty")
        return {"status": "error", "message": "No input data"}

    if not model_file:
        logging.error("Missing model_file path")
        return {"status": "error", "message": "model_file required"}

    # Load model
    try:
        model = xgb.Booster()
        model.load_model(model_file)
        logging.info(f"Loaded XGBoost model from {model_file}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return {"status": "error", "message": "Failed to load model"}

    # -------------------- Ensure required numeric columns --------------------
    raw_feature_cols = [
        c for c in REQUIRED_COLUMNS
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not raw_feature_cols:
        return {"status": "error", "message": "No feature columns available for prediction"}

    # -------------------- Apply prefix to feature names --------------------
    prefixed_feature_cols = {c: FEATURE_PREFIX + c for c in raw_feature_cols}
    df = df.rename(columns=prefixed_feature_cols)

    X = df[list(prefixed_feature_cols.values())]

    # -------------------- Predict --------------------
    try:
        dmatrix = xgb.DMatrix(X)
        preds = model.predict(dmatrix)
        df["confidence_score"] = preds
        df["prediction"] = "BUY"
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return {"status": "error", "message": "Prediction failed"}

    # -------------------- Output --------------------
    return {"status": "success", "suggested_trades": df[["prediction", "confidence_score"]]}
