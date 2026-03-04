# models/train_model.py
# Trains Logistic Regression and SVM models on TF-IDF features

import os
import sys
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    LOGISTIC_MODEL_PATH,
    SVM_MODEL_PATH,
    SAVED_MODELS_DIR,
    RANDOM_STATE
)


# ─────────────────────────────────────────
# TRAIN LOGISTIC REGRESSION
# ─────────────────────────────────────────

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """
    Train Logistic Regression classifier.

    Settings:
        - class_weight='balanced' : handles class imbalance
        - max_iter=1000           : enough iterations to converge
        - C=1.0                   : regularization strength
        - solver='lbfgs'          : efficient for multiclass
    """
    print("[INFO] Training Logistic Regression...")

    model = LogisticRegression(
        class_weight = 'balanced',
        max_iter     = 1000,
        C            = 1.0,
        solver       = 'lbfgs',
        random_state = RANDOM_STATE
    )

    model.fit(X_train, y_train)
    print("[SUCCESS] Logistic Regression trained!")
    return model


# ─────────────────────────────────────────
# TRAIN SVM
# ─────────────────────────────────────────

def train_svm(X_train, y_train) -> LinearSVC:
    """
    Train Support Vector Machine classifier.
    Using LinearSVC — faster than SVC for large text datasets.

    Settings:
        - class_weight='balanced' : handles class imbalance
        - max_iter=2000           : enough for convergence
        - C=1.0                   : regularization
    """
    print("[INFO] Training SVM (LinearSVC)...")

    model = LinearSVC(
        class_weight = 'balanced',
        max_iter     = 2000,
        C            = 1.0,
        random_state = RANDOM_STATE
    )

    model.fit(X_train, y_train)
    print("[SUCCESS] SVM trained!")
    return model


# ─────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────

def save_model(model, path: str):
    """Save a trained model to disk using joblib."""
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    joblib.dump(model, path)
    print(f"[SUCCESS] Model saved → {path}")


# ─────────────────────────────────────────
# TRAIN ALL MODELS
# ─────────────────────────────────────────

def train_all_models(X_train, y_train) -> dict:
    """
    Train both models and save them to disk.

    Returns:
        dict: {'logistic': model, 'svm': model}
    """
    print("\n" + "="*50)
    print("  MODEL TRAINING PIPELINE")
    print("="*50)

    print(f"[INFO] Training samples : {X_train.shape[0]}")
    print(f"[INFO] Feature count    : {X_train.shape[1]}")

    # ── Label distribution ────────────────
    print("[INFO] Training label distribution:")
    label_counts = pd.Series(y_train).value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = {0: "Negative", 1: "Neutral", 2: "Positive"}.get(int(label), str(label))
        print(f"       Class {int(label)} ({label_name}) → {count} samples")

    # ── Train ─────────────────────────────
    logistic_model = train_logistic_regression(X_train, y_train)
    svm_model      = train_svm(X_train, y_train)

    # ── Save ──────────────────────────────
    save_model(logistic_model, LOGISTIC_MODEL_PATH)
    save_model(svm_model,      SVM_MODEL_PATH)

    print(f"\n[SUCCESS] All models trained and saved!")
    print("="*50 + "\n")

    return {
        'logistic' : logistic_model,
        'svm'      : svm_model
    }


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    from preprocessing.feature_engineering import prepare_features
    X_train, X_test, y_train, y_test, vectorizer = prepare_features()
    models = train_all_models(X_train, y_train)
    for name, model in models.items():
        print(f"  ✅ {name} → {type(model).__name__}")