# utils/helpers.py
# Shared utility functions — CSV merging, directory setup, logging helpers

import os
import glob
import pandas as pd
from datetime import datetime

from config.config import (
    EXTERNAL_DATA_DIR,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    SAVED_MODELS_DIR,
    TEXT_COLUMN,
    SENTIMENT_COLUMN,
    USER_COLUMN,
    SENTIMENT_MAP
)


# ─────────────────────────────────────────
# LOGGING HELPER
# ─────────────────────────────────────────

def log(message: str, level: str = "INFO"):
    """
    Simple console logger with timestamp.

    Args:
        message : Message to print
        level   : INFO / WARNING / ERROR / SUCCESS
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    icons = {
        "INFO"    : "ℹ️ ",
        "WARNING" : "⚠️ ",
        "ERROR"   : "❌",
        "SUCCESS" : "✅"
    }
    icon = icons.get(level, "ℹ️ ")
    print(f"[{timestamp}] {icon}  {message}")


# ─────────────────────────────────────────
# DIRECTORY SETUP
# ─────────────────────────────────────────

def ensure_directories():
    """
    Create all required project directories if they don't exist.
    Call this at the start of run.py
    """
    dirs = [
        os.path.join(os.path.dirname(RAW_DATA_PATH)),
        os.path.join(os.path.dirname(PROCESSED_DATA_PATH)),
        EXTERNAL_DATA_DIR,
        SAVED_MODELS_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    log("All directories verified/created", "SUCCESS")


# ─────────────────────────────────────────
# MERGE ALL 26 CSV FILES
# ─────────────────────────────────────────

def merge_csv_files() -> pd.DataFrame:
    """
    Scan data/external/ for all CSV files across Set-1 and Set-2,
    merge them into one DataFrame, clean up, and save to:
        data/raw/political_tweets.csv

    Expected columns in each CSV:
        - sentiment : positive / negative / neutral
        - text      : raw tweet text
        - user      : twitter username

    Returns:
        pd.DataFrame: merged and cleaned dataframe
    """
    log("Scanning data/external/ for CSV files...")

    pattern  = os.path.join(EXTERNAL_DATA_DIR, "**", "*.csv")
    all_files = glob.glob(pattern, recursive=True)

    if not all_files:
        raise FileNotFoundError(
            f"[ERROR] No CSV files found in: {EXTERNAL_DATA_DIR}\n"
            f"Make sure Set-1/ and Set-2/ folders are inside data/external/"
        )

    log(f"Found {len(all_files)} CSV files. Merging...")

    dfs = []
    skipped = 0

    for file in all_files:
        try:
            # ── KEY FIX: handle malformed CSVs ──
            df = pd.read_csv(
             file,
             on_bad_lines='skip',
             encoding='utf-8',
             encoding_errors='ignore'
            )

            # Check required columns exist
            required = [TEXT_COLUMN, SENTIMENT_COLUMN]
            missing  = [c for c in required if c not in df.columns]
            if missing:
                log(f"Skipping {os.path.basename(file)} — missing columns: {missing}", "WARNING")
                skipped += 1
                continue

            # Add source file name for traceability
            df['source_file'] = os.path.basename(file)
            dfs.append(df)
            log(f"Loaded: {os.path.basename(file)} → {len(df)} rows")

        except Exception as e:
            log(f"Could not read {os.path.basename(file)}: {e}", "WARNING")
            skipped += 1

    if not dfs:
        raise ValueError("[ERROR] No valid CSV files could be loaded.")

    log(f"Successfully loaded: {len(dfs)} files | Skipped: {skipped} files")

    # ── Merge all ─────────────────────────
    merged = pd.concat(dfs, ignore_index=True)
    log(f"Total rows before cleaning: {len(merged)}")

    # ── Keep only needed columns ──────────
    cols_to_keep  = [TEXT_COLUMN, SENTIMENT_COLUMN, USER_COLUMN, 'source_file']
    cols_available = [c for c in cols_to_keep if c in merged.columns]
    merged = merged[cols_available]

    # ── Drop nulls ────────────────────────
    merged.dropna(subset=[TEXT_COLUMN, SENTIMENT_COLUMN], inplace=True)

    # ── Standardize sentiment labels ──────
    merged[SENTIMENT_COLUMN] = (
    merged[SENTIMENT_COLUMN]
    .str.lower()
    .str.strip()
    .str.replace('"', '', regex=False)   # remove stray quote characters
    .str.replace("'", '', regex=False)   # remove stray apostrophes
    .str.strip()                          # strip again after removing quotes
)

    # ── Keep only valid sentiment values ──
    valid_sentiments = list(SENTIMENT_MAP.keys())
    merged = merged[merged[SENTIMENT_COLUMN].isin(valid_sentiments)]

    # ── Drop duplicate tweets ─────────────
    before_dedup = len(merged)
    merged.drop_duplicates(subset=[TEXT_COLUMN], inplace=True)
    after_dedup  = len(merged)
    log(f"Duplicates removed: {before_dedup - after_dedup}")

    # ── Reset index ───────────────────────
    merged.reset_index(drop=True, inplace=True)

    log(f"Total rows after cleaning: {len(merged)}")
    log(f"Sentiment distribution:\n{merged[SENTIMENT_COLUMN].value_counts().to_string()}")

    # ── Save ──────────────────────────────
    merged.to_csv(RAW_DATA_PATH, index=False)
    log(f"Merged dataset saved → {RAW_DATA_PATH}", "SUCCESS")

    return merged


# ─────────────────────────────────────────
# CHECK IF MODELS EXIST
# ─────────────────────────────────────────

def models_exist() -> bool:
    """
    Check if all three saved model files exist.
    Used by run.py to decide whether to retrain.
    """
    from config.config import LOGISTIC_MODEL_PATH, SVM_MODEL_PATH, VECTORIZER_PATH

    paths     = [LOGISTIC_MODEL_PATH, SVM_MODEL_PATH, VECTORIZER_PATH]
    all_exist = all(os.path.exists(p) for p in paths)

    if all_exist:
        log("All saved models found. Skipping training.", "SUCCESS")
    else:
        missing = [p for p in paths if not os.path.exists(p)]
        log(f"Missing models: {[os.path.basename(p) for p in missing]}", "WARNING")

    return all_exist


# ─────────────────────────────────────────
# DATA EXISTENCE CHECKS
# ─────────────────────────────────────────

def raw_data_exists() -> bool:
    """Check if merged political_tweets.csv exists."""
    exists = os.path.exists(RAW_DATA_PATH)
    if not exists:
        log(f"Raw data not found at: {RAW_DATA_PATH}", "WARNING")
    return exists


def processed_data_exists() -> bool:
    """Check if cleaned_data.csv exists."""
    return os.path.exists(PROCESSED_DATA_PATH)


# ─────────────────────────────────────────
# LOAD PROCESSED DATA
# ─────────────────────────────────────────

def load_processed_data() -> pd.DataFrame:
    """
    Load the cleaned dataset for use in training/evaluation.
    """
    if not processed_data_exists():
        raise FileNotFoundError(
            f"Processed data not found. Run clean_dataset() first.\n"
            f"Expected at: {PROCESSED_DATA_PATH}"
        )
    df = pd.read_csv(PROCESSED_DATA_PATH)
    log(f"Loaded processed data: {len(df)} rows", "SUCCESS")
    return df


# ─────────────────────────────────────────
# SENTIMENT SCORE CALCULATOR
# ─────────────────────────────────────────

def calculate_sentiment_score(predictions: list) -> float:
    """
    Calculate overall public opinion score.
    Score ranges from -1.0 (all negative) to +1.0 (all positive)

    Formula:
        score = (positive - negative) / total
    """
    total = len(predictions)
    if total == 0:
        return 0.0

    positive = predictions.count(2)
    negative = predictions.count(0)
    score    = (positive - negative) / total

    return round(score, 2)


def sentiment_score_label(score: float) -> str:
    """Convert numeric score to human-readable label."""
    if score >= 0.3:
        mood = "🟢 Positive Leaning"
    elif score <= -0.3:
        mood = "🔴 Negative Leaning"
    else:
        mood = "🟡 Neutral / Mixed"

    return f"{score:+.2f} → {mood}"


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    ensure_directories()
    df = merge_csv_files()
    print(df.head())
    print("\nSentiment Score Test:")
    preds = [2, 2, 1, 0, 2, 0, 1, 2]
    score = calculate_sentiment_score(preds)
    print(sentiment_score_label(score))