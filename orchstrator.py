# Predicts BUY signals,
# avoids duplicated by filter by trade_type

import os
import json
import logging
import pandas as pd
from pymongo import MongoClient
from buy_model import run as model_run  # your existing model logic function
from dotenv import load_dotenv
from datetime import datetime
from bson import ObjectId  # <-- import ObjectId

logging.basicConfig(level=logging.INFO)

# Define the columns to pass to the buy model (bought columns)
BOUGHT_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "macd", "macd_signal", "macd_histogram",
    "rsi", "rsi_sma",
    "ema_100", "ema_200",
    "atr", "ema_ratio",
    "macd_histogram_x_atr", "buy_sell_pressure_x_ema_ratio",
    "buy_sell_pressure", "relative_volume",
    "quote_volume_ratio", "rsi_x_relative_volume"
]

# Set minimum buy score threshold
MIN_BUY_SCORE = 0.7  # <-- adjust this to your preferred threshold

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
    coins_file = os.path.join(base_dir, coins_file)  # same folder as script
    model_file = os.path.join(base_dir, model_file)  # same folder as script

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

        # ------------------ Remove duplicates before creating unique index ------------------
        duplicates = suggested_trades_collection.aggregate([
            {
                "$group": {
                    "_id": {"symbol": "$symbol", "trade_type": "$trade_type"},
                    "ids": {"$push": "$_id"},
                    "count": {"$sum": 1}
                }
            },
            {"$match": {"count": {"$gt": 1}}}
        ])

        for doc in duplicates:
            ids_to_delete = doc["ids"][1:]
            suggested_trades_collection.delete_many({"_id": {"$in": ids_to_delete}})
            logging.info(f"Removed {len(ids_to_delete)} duplicate(s) for {doc['_id']}")

        # ------------------ Ensure unique compound index ------------------
        suggested_trades_collection.create_index(
            [("symbol", 1), ("trade_type", 1)],
            unique=True
        )
        logging.info("Unique index on (symbol, trade_type) ensured.")

    except Exception as e:
        logging.error(f"Error connecting to MongoDB or creating index: {e}")
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
        logging.warning("Model returned no predictions; skipping all coins (no buy_score).")
        client.close()
        return

    # -------------------------- Add symbol and buyid as ObjectId --------------------------
    suggested_trades["symbol"] = df_latest["symbol"].values
    suggested_trades["buyid"] = df_latest["_id"].values  # <-- keep as ObjectId

    # -------------------------- Set default trade_type if missing --------------------------
    if "prediction" in suggested_trades.columns:
        suggested_trades.rename(columns={"prediction": "trade_type"}, inplace=True)
    if "trade_type" not in suggested_trades.columns:
        suggested_trades["trade_type"] = "BUY"
    suggested_trades["trade_type"].fillna("BUY", inplace=True)

    # -------------------------- Only keep rows with valid buy_score --------------------------
    if "confidence_score" in suggested_trades.columns:
        suggested_trades.rename(columns={"confidence_score": "buy_score"}, inplace=True)

    if "buy_score" not in suggested_trades.columns:
        logging.warning("Model did not return buy_score. Skipping all coins.")
        client.close()
        return

    suggested_trades = suggested_trades[suggested_trades["buy_score"].notnull()]

    # -------------------------- Filter by minimum buy score --------------------------
    suggested_trades = suggested_trades[suggested_trades["buy_score"] >= MIN_BUY_SCORE]

    if suggested_trades.empty:
        logging.warning(f"No predictions with buy_score >= {MIN_BUY_SCORE}. Exiting.")
        client.close()
        return

    # -------------------------- Keep only required fields --------------------------
    columns_to_save = ["symbol", "buyid", "trade_type", "buy_score"]
    df_to_save = suggested_trades[[c for c in columns_to_save if c in suggested_trades.columns]]

    # -------------------------- Save to MongoDB with created_at updated every time --------------------------
    try:
        for record in df_to_save.to_dict("records"):
            now = datetime.utcnow()
            suggested_trades_collection.update_one(
                {"symbol": record["symbol"], "trade_type": record["trade_type"]},  # filter
                {
                    "$set": {**record, "created_at": now}  # overwrite created_at every run
                },
                upsert=True
            )
        logging.info(
            f"Saved {len(df_to_save)} trade predictions (upserted by symbol + trade_type) "
            f"to collection {suggested_trades_collection_name}"
        )
    except Exception as e:
        logging.error(f"Error saving suggested trades to MongoDB: {e}")

    client.close()
    logging.info("Stage 2 prediction orchestrator completed successfully.")


# -------------------------- Main --------------------------
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))  # script folder

    # Load environment variables from .env in the same folder as script
    dotenv_path = os.path.join(base_dir, ".env")
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
        "coins_file": os.getenv("coins_file"),
        "model_file": os.getenv("model_file")
    }

    orchestrator_stage2(config)

