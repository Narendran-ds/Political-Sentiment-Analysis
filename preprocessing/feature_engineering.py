# preprocessing/feature_engineering.py
# Builds TF-IDF vectorizer and prepares train/test splits for ML training

import os
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    PROCESSED_DATA_PATH,
    VECTORIZER_PATH,
    TFIDF_MAX_FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
    SAVED_MODELS_DIR
)


# ─────────────────────────────────────────
# LOAD PROCESSED DATA
# ─────────────────────────────────────────

def load_cleaned_data() -> pd.DataFrame:
    """
    Load the cleaned dataset from processed path.

    Returns:
        pd.DataFrame: cleaned dataframe with
                      'cleaned_tweet' and 'label' columns
    """
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(
            f"[ERROR] Processed data not found at: {PROCESSED_DATA_PATH}\n"
            f"Run preprocessing/clean_text.py first!"
        )

    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"[INFO] Loaded cleaned data: {len(df)} rows")

    # Validate required columns
    for col in ['cleaned_tweet', 'label']:
        if col not in df.columns:
            raise ValueError(f"[ERROR] Missing column: '{col}'")

    # Drop any remaining nulls
    df.dropna(subset=['cleaned_tweet', 'label'], inplace=True)

    return df


# ─────────────────────────────────────────
# BUILD TFIDF VECTORIZER
# ─────────────────────────────────────────

def build_tfidf_vectorizer() -> TfidfVectorizer:
    """
    Create a TF-IDF vectorizer with optimal settings
    for tweet-level political text.

    Settings:
        - max_features : from config (default 5000)
        - ngram_range  : (1, 2) — unigrams + bigrams
        - sublinear_tf : True   — log normalization
        - min_df       : 2      — ignore very rare words
        - max_df       : 0.95   — ignore very common words

    Returns:
        TfidfVectorizer: configured but not yet fitted
    """
    vectorizer = TfidfVectorizer(
        max_features  = TFIDF_MAX_FEATURES,
        ngram_range   = (1, 2),
        sublinear_tf  = True,
        min_df        = 2,
        max_df        = 0.95,
        strip_accents = 'unicode',
        analyzer      = 'word'
    )
    return vectorizer


# ─────────────────────────────────────────
# SAVE VECTORIZER
# ─────────────────────────────────────────

def save_vectorizer(vectorizer: TfidfVectorizer):
    """
    Save fitted vectorizer to models/saved_models/vectorizer.pkl

    Args:
        vectorizer: fitted TfidfVectorizer instance
    """
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"[SUCCESS] Vectorizer saved → {VECTORIZER_PATH}")


# ─────────────────────────────────────────
# LOAD VECTORIZER
# ─────────────────────────────────────────

def load_vectorizer() -> TfidfVectorizer:
    """
    Load saved vectorizer from disk.

    Returns:
        TfidfVectorizer: fitted vectorizer
    """
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"[ERROR] Vectorizer not found at: {VECTORIZER_PATH}\n"
            f"Run feature_engineering.py first!"
        )

    vectorizer = joblib.load(VECTORIZER_PATH)
    print(f"[INFO] Vectorizer loaded from: {VECTORIZER_PATH}")
    return vectorizer


# ─────────────────────────────────────────
# MAIN PIPELINE — PREPARE FEATURES
# ─────────────────────────────────────────

def prepare_features(verbose=True):
    """
    Full feature engineering pipeline:

        1. Load cleaned_data.csv
        2. Validate dataset size
        3. Check class imbalance
        4. Split into train/test sets (stratified)
        5. Fit TF-IDF on training data ONLY
           (avoid data leakage)
        6. Transform both train and test sets
        7. Save vectorizer.pkl

    Returns:
        tuple: (X_train, X_test, y_train, y_test, vectorizer)
    """
    print("\n" + "="*50)
    print("  FEATURE ENGINEERING PIPELINE")
    print("="*50)

    # ── Step 1: Load data ─────────────────
    df = load_cleaned_data()

    X = df['cleaned_tweet'].values
    y = df['label'].values

    print(f"[INFO] Features shape : {X.shape}")
    print(f"[INFO] Labels shape   : {y.shape}")

    # ── Step 2: Safety check ──────────────
    # Improvement 2: handle very small datasets
    if len(df) < 10:
        raise ValueError(
            "[ERROR] Dataset too small for train/test split. "
            "Need at least 10 samples."
        )

    # ── Step 3: Label distribution ────────
    # Improvement 1: clean readable label print
    print("[INFO] Label distribution:")
    label_counts = pd.Series(y).value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = {0: "Negative", 1: "Neutral", 2: "Positive"}.get(int(label), str(label))
        print(f"       Class {int(label)} ({label_name}) → {count} samples")

    # ── Step 4: Class imbalance warning ───
    # Improvement 4: warn if any class is severely underrepresented
    if label_counts.min() < 5:
        print("[WARNING] Some classes have very few samples — model may be biased!")
    elif label_counts.max() / label_counts.min() > 5:
        print("[WARNING] Class imbalance detected — train_model will use class_weight='balanced'")

    # ── Step 5: Train/Test split ──────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y    # maintain class ratio in both sets
    )

    print(f"[INFO] Train size : {len(X_train)} samples")
    print(f"[INFO] Test size  : {len(X_test)}  samples")

    # ── Step 6: Build & Fit vectorizer ────
    # IMPORTANT: fit ONLY on training data — no data leakage
    print("[INFO] Fitting TF-IDF vectorizer on training data only...")
    vectorizer    = build_tfidf_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # ── Step 7: Transform test data ───────
    # Transform only — vectorizer already fitted on train
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"[INFO] TF-IDF matrix shape (train) : {X_train_tfidf.shape}")
    print(f"[INFO] TF-IDF matrix shape (test)  : {X_test_tfidf.shape}")

    # Improvement 3: vocabulary size check
    print(f"[INFO] Vocabulary size             : {len(vectorizer.vocabulary_)}")

    # ── Step 8: Save vectorizer ───────────
    save_vectorizer(vectorizer)

    print(f"\n[SUCCESS] Feature engineering complete!")
    print("="*50 + "\n")

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, vectorizer = prepare_features()

    print(f"X_train shape : {X_train.shape}")
    print(f"X_test shape  : {X_test.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"y_test shape  : {y_test.shape}")

    # Show sample features from vocabulary
    feature_names = vectorizer.get_feature_names_out()
    print(f"\nTotal features : {len(feature_names)}")
    print(f"Sample features: {list(feature_names[:20])}")