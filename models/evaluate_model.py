# models/evaluate_model.py
# Evaluates trained models — metrics + confusion matrix

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    SENTIMENT_LABELS,
    SENTIMENT_COLORS
)


# ─────────────────────────────────────────
# EVALUATE SINGLE MODEL
# ─────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a single trained model on test data.

    Computes:
        - Accuracy
        - Precision (weighted)
        - Recall    (weighted)
        - F1 Score  (weighted)

    Args:
        model      : trained sklearn model
        X_test     : TF-IDF test matrix
        y_test     : true labels
        model_name : display name (e.g. 'Logistic Regression')

    Returns:
        dict: metrics dictionary
    """
    # ── Predict ───────────────────────────
    y_pred = model.predict(X_test)

    # ── Compute metrics ───────────────────
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_test, y_pred,    average='weighted', zero_division=0)
    f1        = f1_score(y_test, y_pred,        average='weighted', zero_division=0)

    # ── Print results ─────────────────────
    print(f"\n{'─'*40}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'─'*40}")
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"{'─'*40}")

    # ── Full classification report ────────
    print(f"\n[INFO] Detailed Classification Report:")
    target_names = [SENTIMENT_LABELS[i] for i in sorted(SENTIMENT_LABELS.keys())]
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    return {
        'model_name' : model_name,
        'accuracy'   : round(accuracy,  4),
        'precision'  : round(precision, 4),
        'recall'     : round(recall,    4),
        'f1_score'   : round(f1,        4),
        'y_pred'     : y_pred
    }


# ─────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────

def plot_confusion_matrix(
    y_test,
    y_pred,
    model_name: str,
    save_path: str = None
) -> plt.Figure:
    """
    Plot a styled confusion matrix heatmap.

    Args:
        y_test     : true labels
        y_pred     : predicted labels
        model_name : title for the plot
        save_path  : optional path to save the figure

    Returns:
        plt.Figure: confusion matrix figure
    """
    labels      = sorted(SENTIMENT_LABELS.keys())
    label_names = [SENTIMENT_LABELS[i] for i in labels]

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot      = True,
        fmt        = 'd',
        cmap       = 'Blues',
        xticklabels = label_names,
        yticklabels = label_names,
        linewidths  = 0.5,
        ax          = ax
    )

    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label',      fontsize=12)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Confusion matrix saved → {save_path}")

    return fig


# ─────────────────────────────────────────
# COMPARE ALL MODELS
# ─────────────────────────────────────────

def compare_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate all models and return a comparison DataFrame.

    Args:
        models : dict of {'model_name': fitted_model}
        X_test : TF-IDF test matrix
        y_test : true labels

    Returns:
        pd.DataFrame: side-by-side metrics comparison
    """
    print("\n" + "="*50)
    print("  MODEL COMPARISON")
    print("="*50)

    results = []

    for model_name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name)
        results.append({
            'Model'     : metrics['model_name'],
            'Accuracy'  : metrics['accuracy'],
            'Precision' : metrics['precision'],
            'Recall'    : metrics['recall'],
            'F1 Score'  : metrics['f1_score']
        })

    # ── Build comparison table ────────────
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('F1 Score', ascending=False).reset_index(drop=True)

    print("\n[INFO] Model Comparison Table:")
    print(df_results.to_string(index=False))

    # ── Best model ────────────────────────
    best = df_results.iloc[0]
    print(f"\n[SUCCESS] Best Model → {best['Model']} (F1: {best['F1 Score']})")
    print("="*50 + "\n")

    return df_results


# ─────────────────────────────────────────
# PLOT MODEL COMPARISON BAR CHART
# ─────────────────────────────────────────

def plot_model_comparison(df_results: pd.DataFrame) -> plt.Figure:
    """
    Plot a grouped bar chart comparing model metrics.

    Args:
        df_results : DataFrame from compare_models()

    Returns:
        plt.Figure
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    x       = np.arange(len(metrics))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for i, (_, row) in enumerate(df_results.iterrows()):
        values = [row[m] for m in metrics]
        offset = (i - len(df_results) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, values, width, label=row['Model'], color=colors[i % len(colors)])

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=9
            )

    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    from preprocessing.feature_engineering import prepare_features
    from models.train_model import train_all_models

    # Prepare features
    X_train, X_test, y_train, y_test, vectorizer = prepare_features()

    # Train models
    models = train_all_models(X_train, y_train)

    # Evaluate and compare
    df_results = compare_models(models, X_test, y_test)

    # Plot confusion matrices
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        fig    = plot_confusion_matrix(y_test, y_pred, model_name)
        plt.show()

    # Plot comparison chart
    fig = plot_model_comparison(df_results)
    plt.show()