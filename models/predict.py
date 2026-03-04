# models/predict.py
# Inference logic — single tweet and batch prediction

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    SENTIMENT_LABELS,
    TEXT_COLUMN
)
from preprocessing.clean_text import clean_text
from models.model_loader import (
    load_all_models,
    load_model_by_name
)


# ─────────────────────────────────────────
# SOFTMAX HELPER
# ─────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    """
    Convert decision function scores to probabilities
    using softmax normalization.

    Args:
        x : numpy array of raw scores

    Returns:
        numpy array of probabilities summing to 1.0
    """
    e_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return e_x / e_x.sum()


# ─────────────────────────────────────────
# GET PROBABILITIES (handles LinearSVC)
# ─────────────────────────────────────────

def _get_probabilities(model, X) -> dict:
    """
    Get per-class probabilities.

    Handles:
        - LogisticRegression → uses predict_proba()
        - LinearSVC          → uses decision_function()
                               converted via softmax

    Args:
        model : fitted sklearn model
        X     : vectorized input (sparse matrix)

    Returns:
        dict: {'Negative': float, 'Neutral': float, 'Positive': float}
    """
    # Use model.classes_ for safe class order
    # Improvement 3: class order safety
    class_indices = model.classes_
    label_names   = [SENTIMENT_LABELS.get(int(i), str(i)) for i in class_indices]

    try:
        # LogisticRegression — has predict_proba
        probs = model.predict_proba(X)[0]
        return {
            label_names[i]: round(float(probs[i]), 4)
            for i in range(len(label_names))
        }

    except AttributeError:
        # LinearSVC — use decision_function → softmax
        decision = model.decision_function(X)[0]
        probs    = _softmax(decision)
        return {
            label_names[i]: round(float(probs[i]), 4)
            for i in range(len(label_names))
        }


# ─────────────────────────────────────────
# PREDICT SINGLE TWEET
# ─────────────────────────────────────────

def predict_single(
    text       : str,
    model_name : str  = 'svm',
    components : dict = None,
    verbose    : bool = True
) -> dict:
    """
    Predict sentiment for a single raw tweet.

    Pipeline:
        raw text
            ↓ clean_text()
            ↓ vectorizer.transform()
            ↓ model.predict()
            ↓ sentiment label + confidence

    Args:
        text       : raw tweet string
        model_name : 'svm' or 'logistic'
        components : preloaded models dict (optional)
                     pass this to avoid reloading every call
                     — important for Streamlit performance
        verbose    : print prediction details

    Returns:
        dict: {
            'original_text' : str,
            'cleaned_text'  : str,
            'label'         : int,
            'sentiment'     : str,
            'confidence'    : float,
            'probabilities' : dict
        }
    """
    # ── Load models if not provided ───────
    # FIX: load_all_models() has no verbose param
    # so we always load silently here
    if components is None:
        components = load_all_models()

    vectorizer = components['vectorizer']

    # FIX: safely get model from components dict
    # load_model_by_name() has no verbose param
    model_key = model_name.lower().strip()
    if model_key in components:
        model = components[model_key]
    else:
        # fallback — load from disk
        model = load_model_by_name(model_name)

    # ── Clean text ────────────────────────
    cleaned = clean_text(text)

    # ── Handle empty text after cleaning ──
    if not cleaned.strip():
        return {
            'original_text' : text,
            'cleaned_text'  : cleaned,
            'label'         : 1,
            'sentiment'     : 'Neutral',
            'confidence'    : 0.0,
            'probabilities' : {
                'Negative' : 0.0,
                'Neutral'  : 1.0,
                'Positive' : 0.0
            }
        }

    # ── Vectorize ─────────────────────────
    X = vectorizer.transform([cleaned])

    # ── Predict ───────────────────────────
    label     = int(model.predict(X)[0])
    sentiment = SENTIMENT_LABELS.get(label, 'Neutral')

    # ── Confidence / Probabilities ────────
    probabilities = _get_probabilities(model, X)
    confidence    = probabilities.get(sentiment, 0.0)

    if verbose:
        print(f"\n{'─'*45}")
        print(f"  Input     : {text[:60]}")
        print(f"  Cleaned   : {cleaned[:60]}")
        print(f"  Sentiment : {sentiment}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Probs     : { {k: f'{v:.2%}' for k, v in probabilities.items()} }")
        print(f"{'─'*45}")

    return {
        'original_text' : text,
        'cleaned_text'  : cleaned,
        'label'         : label,
        'sentiment'     : sentiment,
        'confidence'    : round(confidence, 4),
        'probabilities' : probabilities
    }


# ─────────────────────────────────────────
# PREDICT BATCH (optimized)
# ─────────────────────────────────────────

def predict_batch(
    texts      : list,
    model_name : str  = 'svm',
    components : dict = None
) -> pd.DataFrame:
    """
    Predict sentiment for a list of tweets.

    Improvement 1: Optimized batch pipeline —
        - Clean all texts first
        - Vectorize in ONE batch call
        - model.predict() called ONCE
        - Much faster than looping predict_single()

    Args:
        texts      : list of raw tweet strings
        model_name : 'svm' or 'logistic'
        components : preloaded models dict (optional)

    Returns:
        pd.DataFrame with columns:
            original_text, cleaned_text, label,
            sentiment, confidence, Negative, Neutral, Positive
    """
    if not texts:
        raise ValueError("[ERROR] Empty list passed to predict_batch()")

    # ── Load models once ──────────────────
    if components is None:
        components = load_all_models()

    vectorizer = components['vectorizer']
    model_key  = model_name.lower().strip()
    model      = components.get(model_key) or load_model_by_name(model_name)

    print(f"[INFO] Batch prediction: {len(texts)} tweets using {model_name.upper()}...")

    # ── Step 1: Clean ALL texts first ─────
    cleaned_texts = [clean_text(t) for t in texts]

    # ── Step 2: Vectorize in ONE call ─────
    X_batch = vectorizer.transform(cleaned_texts)

    # ── Step 3: Predict in ONE call ───────
    labels = model.predict(X_batch)

    # ── Step 4: Get probabilities ─────────
    try:
        # LogisticRegression
        probs_matrix = model.predict_proba(X_batch)
        class_indices = model.classes_
        label_names   = [SENTIMENT_LABELS.get(int(i), str(i)) for i in class_indices]
    except AttributeError:
        # LinearSVC — decision function → softmax
        decisions    = model.decision_function(X_batch)
        probs_matrix = np.array([_softmax(d) for d in decisions])
        class_indices = model.classes_
        label_names   = [SENTIMENT_LABELS.get(int(i), str(i)) for i in class_indices]

    # ── Step 5: Build results ─────────────
    results = []
    for i, (text, cleaned, label) in enumerate(zip(texts, cleaned_texts, labels)):
        label     = int(label)
        sentiment = SENTIMENT_LABELS.get(label, 'Neutral')
        probs     = {label_names[j]: round(float(probs_matrix[i][j]), 4)
                     for j in range(len(label_names))}
        confidence = probs.get(sentiment, 0.0)

        results.append({
            'original_text' : text,
            'cleaned_text'  : cleaned,
            'label'         : label,
            'sentiment'     : sentiment,
            'confidence'    : confidence,
            'Negative'      : probs.get('Negative', 0.0),
            'Neutral'       : probs.get('Neutral',  0.0),
            'Positive'      : probs.get('Positive', 0.0),
        })

    df = pd.DataFrame(results)

    print(f"[INFO] Batch prediction complete!")
    print(f"[INFO] Sentiment distribution:")
    print(df['sentiment'].value_counts().to_string())

    return df


# ─────────────────────────────────────────
# PREDICT FROM DATAFRAME
# ─────────────────────────────────────────

def predict_from_dataframe(
    df         : pd.DataFrame,
    text_col   : str  = None,
    model_name : str  = 'svm',
    components : dict = None
) -> pd.DataFrame:
    """
    Run batch prediction directly on a DataFrame column.

    Args:
        df         : input DataFrame
        text_col   : column with tweet text
                     defaults to config TEXT_COLUMN
        model_name : 'svm' or 'logistic'
        components : preloaded models dict (optional)

    Returns:
        pd.DataFrame: original df + prediction columns appended
    """
    col = text_col or TEXT_COLUMN

    if col not in df.columns:
        raise ValueError(f"[ERROR] Column '{col}' not found in DataFrame")

    texts   = df[col].fillna('').tolist()
    pred_df = predict_batch(texts, model_name, components)

    # ── Append predictions to original df ─
    df = df.copy()
    df['cleaned_tweet'] = pred_df['cleaned_text'].values
    df['label']         = pred_df['label'].values
    df['sentiment']     = pred_df['sentiment'].values
    df['confidence']    = pred_df['confidence'].values
    df['Negative']      = pred_df['Negative'].values
    df['Neutral']       = pred_df['Neutral'].values
    df['Positive']      = pred_df['Positive'].values

    return df


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    # Load models ONCE — pass to all functions
    # Improvement 2: load once, reuse everywhere
    print("Loading models...")
    components = load_all_models()

    # ── Test 1: Single predictions ────────
    print("\n=== Single Prediction Test ===")
    sample_tweets = [
        "NATO forces are pushing back Russia successfully!",
        "This war is destroying innocent lives in Ukraine.",
        "The situation in Eastern Europe remains uncertain."
    ]

    for tweet in sample_tweets:
        predict_single(
            text       = tweet,
            model_name = 'svm',
            components = components,
            verbose    = True
        )

    # ── Test 2: Batch prediction ──────────
    print("\n=== Batch Prediction Test ===")
    df_results = predict_batch(
        texts      = sample_tweets,
        model_name = 'svm',
        components = components
    )
    print(df_results[['original_text', 'sentiment', 'confidence']])

    # ── Test 3: Both models on same tweet ─
    print("\n=== Model Comparison ===")
    test_tweet = "Ukraine is bravely defending against Russian aggression"
    for model_name in ['logistic', 'svm']:
        r = predict_single(
            text       = test_tweet,
            model_name = model_name,
            components = components,
            verbose    = False
        )
        print(f"  {model_name:10} → {r['sentiment']:10} (confidence: {r['confidence']:.2%})")