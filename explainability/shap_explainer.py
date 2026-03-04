# explainability/shap_explainer.py
# SHAP-based explainability for sentiment predictions

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import SENTIMENT_LABELS
from preprocessing.clean_text import clean_text
from models.model_loader import load_all_models


# ─────────────────────────────────────────
# CACHED EXPLAINER
# ─────────────────────────────────────────
_EXPLAINER_CACHE = {}


def get_explainer(model, vectorizer) -> shap.LinearExplainer:
    """
    Get SHAP explainer — builds ONCE, caches forever.
    Uses sparse zero matrix as background — avoids shape bugs.
    """
    global _EXPLAINER_CACHE
    cache_key = id(model)

    if cache_key not in _EXPLAINER_CACHE:
        print("[INFO] Building SHAP explainer (first time — will be cached)...")

        background = csr_matrix(
            np.zeros((1, len(vectorizer.vocabulary_)))
        )
        explainer = shap.LinearExplainer(model, background)

        _EXPLAINER_CACHE[cache_key] = explainer
        print("[SUCCESS] SHAP explainer built and cached!")

    else:
        print("[INFO] Using cached SHAP explainer ✅")

    return _EXPLAINER_CACHE[cache_key]


def clear_explainer_cache():
    """Clear cache — call after model retraining."""
    global _EXPLAINER_CACHE
    _EXPLAINER_CACHE.clear()
    print("[INFO] SHAP explainer cache cleared.")


# ─────────────────────────────────────────
# SAFE SHAP VALUE EXTRACTOR
# ─────────────────────────────────────────

def _extract_class_shap(shap_values, label: int) -> np.ndarray:
    """
    Safely extract 1D SHAP array for predicted class.

    Your SHAP version returns shape: (1, 2465, 3)
    Meaning: (n_samples, n_features, n_classes)

    So correct extraction is:
        shap_values[0, :, label]
        → all 2465 features for the predicted class

    All cases handled:
        (1, n_features, n_classes) → shap_values[0, :, label]  ✅ your case
        (n_samples, n_features)    → shap_values[0]
        (n_features,)              → shap_values
        list of arrays             → shap_values[label][0]

    Returns:
        np.ndarray: 1D array of shape (n_features,)
    """
    # ── List of arrays (older SHAP versions) ──
    if isinstance(shap_values, list):
        sv = shap_values[label]
        return sv[0] if sv.ndim == 2 else sv

    # ── 3D array ──────────────────────────────
    if shap_values.ndim == 3:
        # YOUR CASE: shape (1, 2465, 3)
        # axis 0 = samples, axis 1 = features, axis 2 = classes
        return shap_values[0, :, label]

    # ── 2D array ──────────────────────────────
    if shap_values.ndim == 2:
        return shap_values[0]

    # ── 1D array ──────────────────────────────
    return shap_values


# ─────────────────────────────────────────
# EXPLAIN SINGLE PREDICTION
# ─────────────────────────────────────────

def explain_prediction(
    text       : str,
    components : dict = None,
    top_n      : int  = 10
) -> dict:
    """
    Explain why a tweet received its predicted sentiment.

    Uses SHAP LinearExplainer on Logistic Regression.
    Identifies top words pushing toward/away from prediction.

    Args:
        text       : raw tweet string
        components : preloaded models dict (optional)
        top_n      : number of top features to show

    Returns:
        dict or None
    """
    if components is None:
        components = load_all_models()

    model      = components['logistic']
    vectorizer = components['vectorizer']

    # ── Clean ─────────────────────────────
    cleaned = clean_text(text)
    if not cleaned.strip():
        print("[WARNING] Empty text after cleaning.")
        return None

    # ── Vectorize ─────────────────────────
    X = vectorizer.transform([cleaned])

    # ── Predict ───────────────────────────
    label     = int(model.predict(X)[0])
    sentiment = SENTIMENT_LABELS.get(label, 'Neutral')

    # ── Get cached explainer ──────────────
    explainer = get_explainer(model, vectorizer)

    # ── Compute SHAP values ───────────────
    shap_values = explainer.shap_values(X)

    # ── Safe extraction ───────────────────
    class_shap = _extract_class_shap(shap_values, label)
    class_shap = np.array(class_shap).flatten()

    # ── Validate shape ────────────────────
    n_features = len(vectorizer.vocabulary_)
    if class_shap.shape[0] != n_features:
        print(f"[ERROR] Still wrong shape: {class_shap.shape}, expected ({n_features},)")
        return None

    # ── Feature names ─────────────────────
    feature_names = vectorizer.get_feature_names_out()

    # ── Non-zero features only ────────────
    X_dense     = X.toarray()[0]
    nonzero_idx = np.where(X_dense > 0)[0]

    if len(nonzero_idx) == 0:
        print("[WARNING] No features found.")
        return None

    # ── Build word → shap pairs ───────────
    word_shap_pairs = []
    for i in nonzero_idx:
        val = class_shap[i]
        val = float(val.flat[0]) if hasattr(val, '__len__') else float(val)
        word_shap_pairs.append((feature_names[i], val))

    # ── Sort ──────────────────────────────
    word_shap_pairs.sort(key=lambda x: x[1], reverse=True)

    # Push TOWARD predicted class
    top_positive = [(w, v) for w, v in word_shap_pairs if v > 0][:top_n]

    # Push AWAY from predicted class
    top_negative = sorted(
        [(w, v) for w, v in word_shap_pairs if v < 0],
        key=lambda x: x[1]
    )[:top_n]

    print(f"\n[INFO] Tweet     : {text[:60]}")
    print(f"[INFO] Predicted : {sentiment}")
    print(f"[INFO] Top +ve   : {[w for w, _ in top_positive[:5]]}")
    print(f"[INFO] Top -ve   : {[w for w, _ in top_negative[:5]]}")

    return {
        'text'            : text,
        'cleaned_text'    : cleaned,
        'predicted_label' : label,
        'predicted_class' : sentiment,
        'top_positive'    : top_positive,
        'top_negative'    : top_negative,
        'shap_values'     : class_shap,
        'feature_names'   : feature_names
    }


# ─────────────────────────────────────────
# PLOT SHAP BAR CHART
# ─────────────────────────────────────────

def plot_shap_explanation(
    explanation : dict,
    top_n       : int = 10
) -> plt.Figure:
    """
    Horizontal bar chart — top contributing words.
    Green → pushes TOWARD predicted class
    Red   → pushes AWAY from predicted class
    """
    if explanation is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No explanation available',
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    top_pos      = explanation['top_positive'][:top_n]
    top_neg      = explanation['top_negative'][:top_n]
    all_features = top_pos + top_neg

    if not all_features:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No contributing features found',
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    words  = [w for w, _ in all_features]
    values = [v for _, v in all_features]
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]

    sorted_pairs = sorted(
        zip(words, values, colors),
        key=lambda x: abs(x[1])
    )
    words, values, colors = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, max(6, len(words) * 0.45)))

    bars = ax.barh(
        words, values,
        color=colors, edgecolor='white', height=0.6
    )

    for bar, val in zip(bars, values):
        x_pos = val + (0.0005 if val >= 0 else -0.0005)
        ha    = 'left' if val >= 0 else 'right'
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f'{val:+.4f}',
            va='center', ha=ha, fontsize=9
        )

    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(
        f'SHAP Explanation — Predicted: {explanation["predicted_class"]}\n'
        f'"{explanation["text"][:70]}"',
        fontsize=12, fontweight='bold', pad=15
    )
    ax.set_xlabel('SHAP Value (contribution to prediction)', fontsize=11)
    ax.set_ylabel('Words / Features', fontsize=11)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Pushes toward predicted class'),
        Patch(facecolor='#e74c3c', label='Pushes away from predicted class')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    return fig


# ─────────────────────────────────────────
# PLOT SHAP SUMMARY (batch)
# ─────────────────────────────────────────

def plot_shap_summary(
    texts      : list,
    components : dict = None,
    top_n      : int  = 15
) -> plt.Figure:
    """
    Global feature importance across multiple tweets.
    Shows most influential words across entire dataset.
    """
    if components is None:
        components = load_all_models()

    model      = components['logistic']
    vectorizer = components['vectorizer']

    print(f"[INFO] Computing SHAP summary for {len(texts)} tweets...")

    cleaned_texts = [clean_text(t) for t in texts]
    cleaned_texts = [t for t in cleaned_texts if t.strip()]

    if not cleaned_texts:
        print("[WARNING] No valid texts.")
        return None

    X           = vectorizer.transform(cleaned_texts)
    explainer   = get_explainer(model, vectorizer)
    shap_values = explainer.shap_values(X)

    feature_names = vectorizer.get_feature_names_out()

    # ── Mean absolute SHAP ────────────────
    # Handle shape (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        mean_shap = np.mean(
            [np.abs(sv).mean(axis=0) for sv in shap_values],
            axis=0
        )
    elif shap_values.ndim == 3:
        # (n_samples, n_features, n_classes) → mean over samples & classes
        mean_shap = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)

    mean_shap = np.array(mean_shap).flatten()

    top_idx      = np.argsort(mean_shap)[-top_n:]
    top_features = [feature_names[i] for i in top_idx]
    top_values   = [float(mean_shap[i]) for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_features, top_values,
            color='#3498db', edgecolor='white', height=0.6)
    ax.set_title(
        f'Top {top_n} Most Influential Words\n'
        f'(Mean |SHAP| across {len(cleaned_texts)} tweets)',
        fontsize=13, fontweight='bold', pad=15
    )
    ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
    ax.set_ylabel('Words / Features',  fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    return fig


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    components = load_all_models()

    test_tweets = [
        "NATO is successfully defending Ukraine against Russian attacks",
        "This brutal war is causing massive destruction and civilian deaths",
        "The peace negotiations remain stalled with no clear outcome"
    ]

    print("\n=== Testing Explainer Cache ===")
    for i, tweet in enumerate(test_tweets):
        print(f"\n--- Tweet {i+1} ---")
        explanation = explain_prediction(
            text       = tweet,
            components = components,
            top_n      = 10
        )
        if explanation:
            print(f"Predicted : {explanation['predicted_class']}")
            print(f"Top +ve   : {explanation['top_positive'][:3]}")
            print(f"Top -ve   : {explanation['top_negative'][:3]}")

    print("\n=== SHAP Summary ===")
    fig = plot_shap_summary(test_tweets, components, top_n=15)
    if fig:
        plt.savefig("shap_summary.png", dpi=100, bbox_inches='tight')
        plt.close()
        print("[INFO] Summary saved!")