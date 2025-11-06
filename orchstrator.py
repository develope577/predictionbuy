# Predicts BUY signals,
# avoids duplicated by filter by prediction type

import os
import json
import logging
import pandas as pd
from pymongo import MongoClient
from buy_model import run as model_run  # your existing model logic function
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

# Define the columns to pass to the buy model (bought columns)
BOUGHT_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "macd",
    "macd_signal", "macd_histogram", "rsi", "rsi_sma", "ema_100",
    "ema_200", "atr", "relative_volume", "quote_volume_ratio",
    "buy_sell_pressure", "ema_ratio", "rsi_x_relative_volume",
    "macd_histogram_x_atr", "buy_sell_pressure_x_ema_ratio", "rsi_x_relative_volume", "relative_volume"
]


def orchestrator_stage2(config):
    # -------------------------- Load config --------------------------
    connection_str = config.get("connection_str")
    db_name = config.get("db_name")
    collection_name_str = config.get("collection_name")
    suggested_trades_collection_name = config.get("suggested_trades_collection")
    coins_file = config.get("coins_file")
    model_file = config.get("model_file")

    # -------------------------- Validate config --------------------------
    missing_keys = []
    for key, val in [("connection_str", connection_str), ("db_name", db_name),
                     ("collection_name", collection_name_str),
                     ("suggested_trades_collection", suggested_trades_collection_name),
                     ("coins_file", coins_file), ("model_file", model_file)]:
        if not val:
            missing_keys.append(key)
    if missing_keys:
        logging.error(f"Missing config values for: {missing_keys}")
        return

    # -------------------------- Make file paths absolute --------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    coins_file = os.path.join(base_dir, "..", coins_file)
    model_file = os.path.join(base_dir, "..", model_file)

    if not os.path.exists(coins_file):
        logging.error(f"Coins file not found: {coins_file}")
        return
    if not os.path.exists(model_file):
        logging.error(f"Model file not found: {model_file}")
        return

    # -------------------------- Load coins --------------------------
    with open(coins_file, "r") as f:
        coins = json.load(f)
    if not coins or not isinstance(coins, list):
        logging.error("Invalid coins list in coins_file")
        return

    # -------------------------- Connect to MongoDB --------------------------
    try:
        client = MongoClient(connection_str)
        db = client[db_name]
        collection_name = db[collection_name_str]
        suggested_trades_collection = db[suggested_trades_collection_name]
        logging.info(f"Connected to MongoDB database: {db_name}")

        # Ensure unique compound index on symbol + prediction (run once)
        suggested_trades_collection.create_index(
            [("symbol", 1), ("prediction", 1)],
            unique=True
        )
    except Exception as e:
        logging.error(f"Error connecting to MongoDB: {e}")
        return

    # -------------------------- Fetch latest features --------------------------
    logging.info(f"Fetching latest features for {len(coins)} coins from MongoDB")
    all_latest_rows = []

    for symbol in coins:
        doc = collection_name.find({"symbol": symbol}).sort("timestamp", -1).limit(1)
        df_symbol = pd.DataFrame(list(doc))
        if df_symbol.empty:
            logging.warning(f"No feature data found for {symbol}")
            continue
        all_latest_rows.append(df_symbol)

    if not all_latest_rows:
        logging.error("No data found for any coin. Exiting.")
        client.close()
        return

    # -------------------------- Combine data --------------------------
    df_latest = pd.concat(all_latest_rows, ignore_index=True)
    logging.info(f"Fetched latest data for {len(df_latest)} coins")

    # -------------------------- Filter DataFrame for buy model --------------------------
    df_model_input = df_latest[[c for c in BOUGHT_COLUMNS if c in df_latest.columns]]

    # -------------------------- Run model predictions --------------------------
    model_input = {"data": df_model_input, "model_file": model_file}
    result = model_run(model_input)

    if result.get("status") != "success":
        logging.error(f"Model run failed: {result.get('message')}")
        client.close()
        return

    suggested_trades = result.get("suggested_trades")
    if suggested_trades is None or suggested_trades.empty:
        logging.warning("Model returned no predictions; skipping all coins (no confidence_score).")
        client.close()
        return

    # -------------------------- Add symbol and buyid --------------------------
    suggested_trades["symbol"] = df_latest["symbol"].values
    suggested_trades["buyid"] = df_latest["_id"].astype(str).values  # link to source document

    # -------------------------- Set default prediction if missing --------------------------
    if "prediction" not in suggested_trades.columns:
        suggested_trades["prediction"] = "BUY"
    suggested_trades["prediction"].fillna("BUY", inplace=True)

    # -------------------------- Only keep rows with valid confidence_score --------------------------
    if "confidence_score" not in suggested_trades.columns:
        logging.warning("Model did not return confidence_score. Skipping all coins.")
        client.close()
        return

    suggested_trades = suggested_trades[suggested_trades["confidence_score"].notnull()]

    if suggested_trades.empty:
        logging.warning("No predictions with confidence_score available. Exiting.")
        client.close()
        return

    # -------------------------- Keep only required fields --------------------------
    columns_to_save = ["symbol", "buyid", "prediction", "confidence_score"]
    df_to_save = suggested_trades[[c for c in columns_to_save if c in suggested_trades.columns]]

    # -------------------------- Save to MongoDB (update existing, no duplicates) --------------------------
    try:
        for record in df_to_save.to_dict("records"):
            suggested_trades_collection.update_one(
                {"symbol": record["symbol"], "prediction": record["prediction"]},  # filter by symbol + prediction
                {"$set": record},                                               # update these fields
                upsert=True                                                     # insert if not exists
            )
        logging.info(
            f"Saved {len(df_to_save)} buy predictions (upserted by symbol + prediction) to collection {suggested_trades_collection_name}"
        )
    except Exception as e:
        logging.error(f"Error saving suggested trades to MongoDB: {e}")

    client.close()
    logging.info("Stage 2 prediction orchestrator completed successfully.")


# -------------------------- Main --------------------------
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, ".."))

    # Load environment variables from .env
    dotenv_path = os.path.join(root_dir, ".env")
    if not os.path.exists(dotenv_path):
        logging.error(f".env file not found at {dotenv_path}")
        exit(1)

    load_dotenv(dotenv_path)

    # Load config from environment variables
    config = {
        "connection_str": os.getenv("MONGO_CONN_STR"),
        "db_name": os.getenv("MONGO_DB_NAME"),
        "collection_name": os.getenv("MONGO_COLLECTION"),
        "suggested_trades_collection": os.getenv("SUGGESTED_TRADES_COLLECTION"),
        "coins_file": os.getenv("COINS_FILE"),
        "model_file": os.getenv("MODEL_FILE")
    }

    orchestrator_stage2(config)
