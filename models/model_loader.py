# models/model_loader.py
# Loads saved models and vectorizer from disk
# Used by predict.py, shap_explainer.py and app.py

import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    LOGISTIC_MODEL_PATH,
    SVM_MODEL_PATH,
    VECTORIZER_PATH
)


# ─────────────────────────────────────────
# LOAD INDIVIDUAL MODELS
# ─────────────────────────────────────────

def load_logistic_model(verbose: bool = True):
    """
    Load saved Logistic Regression model from disk.

    Args:
        verbose : if True, prints load info (default: True)
                  set False in Streamlit to suppress prints

    Returns:
        LogisticRegression: fitted model

    Raises:
        FileNotFoundError: if logistic.pkl not found
    """
    if not os.path.exists(LOGISTIC_MODEL_PATH):
        raise FileNotFoundError(
            f"[ERROR] Logistic model not found at: {LOGISTIC_MODEL_PATH}\n"
            f"Run models/train_model.py first!"
        )

    model = joblib.load(LOGISTIC_MODEL_PATH)

    if verbose:
        print(f"[INFO] Logistic model loaded  ← {LOGISTIC_MODEL_PATH}")
        # Improvement 3: model metadata
        print(f"[INFO] Model classes          : {model.classes_}")

    return model


def load_svm_model(verbose: bool = True):
    """
    Load saved SVM model from disk.

    Args:
        verbose : if True, prints load info (default: True)
                  set False in Streamlit to suppress prints

    Returns:
        LinearSVC: fitted model

    Raises:
        FileNotFoundError: if svm.pkl not found
    """
    if not os.path.exists(SVM_MODEL_PATH):
        raise FileNotFoundError(
            f"[ERROR] SVM model not found at: {SVM_MODEL_PATH}\n"
            f"Run models/train_model.py first!"
        )

    model = joblib.load(SVM_MODEL_PATH)

    if verbose:
        print(f"[INFO] SVM model loaded       ← {SVM_MODEL_PATH}")
        # Improvement 3: model metadata
        print(f"[INFO] Model classes          : {model.classes_}")

    return model


def load_vectorizer(verbose: bool = True):
    """
    Load saved TF-IDF vectorizer from disk.

    Args:
        verbose : if True, prints load info (default: True)

    Returns:
        TfidfVectorizer: fitted vectorizer

    Raises:
        FileNotFoundError: if vectorizer.pkl not found
    """
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"[ERROR] Vectorizer not found at: {VECTORIZER_PATH}\n"
            f"Run preprocessing/feature_engineering.py first!"
        )

    vectorizer = joblib.load(VECTORIZER_PATH)

    if verbose:
        print(f"[INFO] Vectorizer loaded      ← {VECTORIZER_PATH}")
        # Improvement 3: vectorizer metadata
        print(f"[INFO] Vocabulary size        : {len(vectorizer.vocabulary_)}")
        print(f"[INFO] N-gram range           : {vectorizer.ngram_range}")
        print(f"[INFO] Max features           : {vectorizer.max_features}")

    return vectorizer


# ─────────────────────────────────────────
# LOAD ALL MODELS AT ONCE
# ─────────────────────────────────────────

def load_all_models(verbose: bool = True) -> dict:
    """
    Load all saved models and vectorizer in one call.

    Used by:
        - predict.py
        - shap_explainer.py
        - app.py  (with verbose=False to suppress prints)

    Args:
        verbose : if True prints load info (default: True)
                  pass verbose=False in Streamlit

    Returns:
        dict: {
            'logistic'   : LogisticRegression,
            'svm'        : LinearSVC,
            'vectorizer' : TfidfVectorizer
        }

    Raises:
        FileNotFoundError: if any model file is missing
    """
    if verbose:
        print("\n" + "="*50)
        print("  LOADING SAVED MODELS")
        print("="*50)

    try:
        logistic   = load_logistic_model(verbose=verbose)
        svm        = load_svm_model(verbose=verbose)
        vectorizer = load_vectorizer(verbose=verbose)

        if verbose:
            print(f"\n[SUCCESS] All models loaded successfully!")
            print("="*50 + "\n")

        return {
            'logistic'   : logistic,
            'svm'        : svm,
            'vectorizer' : vectorizer
        }

    except FileNotFoundError as e:
        print(f"\n[ERROR] Could not load models:\n{e}")
        print("[INFO]  Run: python run.py to train models first.")
        raise


# ─────────────────────────────────────────
# LOAD SPECIFIC MODEL BY NAME
# ─────────────────────────────────────────

def load_model_by_name(model_name: str, verbose: bool = True):
    """
    Load a specific model by name string.
    Useful for Streamlit dropdown selection.

    Args:
        model_name : 'logistic' or 'svm'
        verbose    : if True prints load info

    Returns:
        fitted sklearn model

    Raises:
        ValueError: if model_name not recognized
    """
    model_name = model_name.lower().strip()

    if model_name in ('logistic', 'logistic regression'):
        return load_logistic_model(verbose=verbose)
    elif model_name in ('svm', 'linearsvc', 'support vector machine'):
        return load_svm_model(verbose=verbose)
    else:
        raise ValueError(
            f"[ERROR] Unknown model: '{model_name}'\n"
            f"Valid options: 'logistic', 'svm'"
        )


# ─────────────────────────────────────────
# CHECK ALL MODELS EXIST
# ─────────────────────────────────────────

def all_models_exist() -> bool:
    """
    Check if all three model files exist on disk.
    Used by run.py to decide whether to retrain.

    Returns:
        bool: True if all files found
    """
    paths = {
        'logistic.pkl'   : LOGISTIC_MODEL_PATH,
        'svm.pkl'        : SVM_MODEL_PATH,
        'vectorizer.pkl' : VECTORIZER_PATH
    }

    all_exist = True
    for name, path in paths.items():
        if os.path.exists(path):
            print(f"[INFO] ✅ Found   : {name}")
        else:
            print(f"[INFO] ❌ Missing : {name}")
            all_exist = False

    return all_exist


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: check all model files exist
    print("Checking model files...")
    exists = all_models_exist()

    if exists:
        print("\n--- Verbose Mode (default) ---")
        components = load_all_models(verbose=True)

        # Display model info
        print("Model Types:")
        print(f"  Logistic   → {type(components['logistic']).__name__}")
        print(f"  SVM        → {type(components['svm']).__name__}")
        print(f"  Vectorizer → {type(components['vectorizer']).__name__}")

        # Test silent mode
        print("\n--- Silent Mode (verbose=False) ---")
        components_silent = load_all_models(verbose=False)
        print("  Loaded silently ✅ (no output — good for Streamlit)")

        # Test load by name
        print("\n--- Load by Name ---")
        m = load_model_by_name('svm', verbose=True)
        print(f"  Loaded: {type(m).__name__} ✅")

    else:
        print("\n[WARNING] Some models missing!")
        print("[INFO]    Run: python -m models.train_model")