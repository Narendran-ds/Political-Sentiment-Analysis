# visualization/plots.py
# Confusion matrix, model comparison, and trend charts
# All outputs saved to outputs/plots/

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    SENTIMENT_LABELS,
    SENTIMENT_COLORS,
    PLOTS_OUTPUT_DIR
)


# ─────────────────────────────────────────
# OUTPUT DIRECTORY SETUP
# ─────────────────────────────────────────

def ensure_output_dir():
    """
    Create outputs/plots/ directory if it doesn't exist.
    Called automatically before every save operation.
    """
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)


def get_output_path(filename: str) -> str:
    """
    Get full path for a plot output file.

    Args:
        filename : e.g. 'confusion_matrix_svm.png'

    Returns:
        str: full path inside outputs/plots/
    """
    ensure_output_dir()
    return os.path.join(PLOTS_OUTPUT_DIR, filename)


def save_figure(fig: plt.Figure, filename: str):
    """
    Save a matplotlib figure to outputs/plots/

    Args:
        fig      : matplotlib Figure object
        filename : output filename e.g. 'confusion_matrix.png'
    """
    path = get_output_path(filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Plot saved → {path}")


# ─────────────────────────────────────────
# STYLING HELPER
# ─────────────────────────────────────────

def set_plot_style():
    """Apply consistent styling across all plots."""
    plt.rcParams.update({
        'figure.facecolor' : 'white',
        'axes.facecolor'   : 'white',
        'axes.grid'        : True,
        'grid.alpha'       : 0.3,
        'font.size'        : 11,
        'axes.titlesize'   : 13,
        'axes.labelsize'   : 11,
    })


# ─────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────

def plot_confusion_matrix(
    y_test,
    y_pred,
    model_name : str  = 'Model',
    save       : bool = True
) -> plt.Figure:
    """
    Plot styled confusion matrix heatmap.
    Shows both raw counts and normalized percentages.

    Args:
        y_test     : true labels (numeric)
        y_pred     : predicted labels (numeric)
        model_name : title label
        save       : if True auto-saves to outputs/plots/

    Returns:
        plt.Figure
    """
    from sklearn.metrics import confusion_matrix

    set_plot_style()

    labels      = sorted(SENTIMENT_LABELS.keys())
    label_names = [SENTIMENT_LABELS[i] for i in labels]

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Normalize for percentage display
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Raw counts ────────────────────────
    sns.heatmap(
        cm,
        annot       = True,
        fmt         = 'd',
        cmap        = 'Blues',
        xticklabels = label_names,
        yticklabels = label_names,
        linewidths  = 0.5,
        ax          = axes[0],
        cbar        = True
    )
    axes[0].set_title(
        f'Confusion Matrix — {model_name}\n(Raw Counts)',
        fontweight='bold', pad=12
    )
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_ylabel('True Label',      fontsize=11)

    # ── Normalized % ──────────────────────
    sns.heatmap(
        cm_norm,
        annot       = True,
        fmt         = '.2%',
        cmap        = 'Greens',
        xticklabels = label_names,
        yticklabels = label_names,
        linewidths  = 0.5,
        ax          = axes[1],
        cbar        = True
    )
    axes[1].set_title(
        f'Confusion Matrix — {model_name}\n(Normalized %)',
        fontweight='bold', pad=12
    )
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_ylabel('True Label',      fontsize=11)

    plt.tight_layout()

    if save:
        name = model_name.lower().replace(' ', '_')
        save_figure(fig, f"confusion_matrix_{name}.png")

    return fig


# ─────────────────────────────────────────
# MODEL COMPARISON BAR CHART
# ─────────────────────────────────────────

def plot_model_comparison(
    df_results : pd.DataFrame,
    save       : bool = True
) -> plt.Figure:
    """
    Grouped bar chart comparing model metrics side by side.

    Args:
        df_results : DataFrame with columns:
                     Model, Accuracy, Precision, Recall, F1 Score
        save       : if True auto-saves to outputs/plots/

    Returns:
        plt.Figure
    """
    set_plot_style()

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    x       = np.arange(len(metrics))
    n       = len(df_results)
    width   = 0.35
    colors  = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (_, row) in enumerate(df_results.iterrows()):
        values = [row[m] for m in metrics]
        offset = (i - n / 2 + 0.5) * width
        bars   = ax.bar(
            x + offset, values, width,
            label     = row['Model'],
            color     = colors[i % len(colors)],
            edgecolor = 'white',
            linewidth = 0.7
        )

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f'{val:.3f}',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold'
            )

    ax.set_title('Model Performance Comparison',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel('Score',  fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save:
        save_figure(fig, "model_comparison.png")

    return fig


# ─────────────────────────────────────────
# SENTIMENT DISTRIBUTION BAR CHART
# ─────────────────────────────────────────

def plot_sentiment_distribution(
    sentiments : list,
    title      : str  = 'Sentiment Distribution',
    save       : bool = True,
    filename   : str  = 'sentiment_distribution.png'
) -> plt.Figure:
    """
    Bar chart showing count of each sentiment class.

    Args:
        sentiments : list of sentiment strings
        title      : chart title
        save       : if True auto-saves to outputs/plots/
        filename   : custom filename for save

    Returns:
        plt.Figure
    """
    set_plot_style()

    counts = pd.Series(sentiments).value_counts()

    # Ensure all 3 classes shown
    for label in SENTIMENT_LABELS.values():
        if label not in counts:
            counts[label] = 0

    counts = counts[list(SENTIMENT_LABELS.values())]
    colors = [SENTIMENT_COLORS.get(s, '#95a5a6') for s in counts.index]
    total  = counts.sum()

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        counts.index, counts.values,
        color     = colors,
        edgecolor = 'white',
        linewidth = 0.7,
        width     = 0.5
    )

    for bar, (label, count) in zip(bars, counts.items()):
        pct = count / total * 100 if total > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{count}\n({pct:.1f}%)',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )

    ax.set_title(title,        fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Sentiment', fontsize=11)
    ax.set_ylabel('Count',     fontsize=11)
    ax.set_ylim(0, counts.max() * 1.25)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save:
        save_figure(fig, filename)

    return fig


# ─────────────────────────────────────────
# SENTIMENT PIE CHART
# ─────────────────────────────────────────

def plot_sentiment_pie(
    sentiments : list,
    title      : str  = 'Sentiment Breakdown',
    save       : bool = True,
    filename   : str  = 'sentiment_pie.png'
) -> plt.Figure:
    """
    Pie chart showing sentiment percentage breakdown.

    Args:
        sentiments : list of sentiment strings
        title      : chart title
        save       : if True auto-saves to outputs/plots/
        filename   : custom filename for save

    Returns:
        plt.Figure
    """
    set_plot_style()

    counts = pd.Series(sentiments).value_counts()
    for label in SENTIMENT_LABELS.values():
        if label not in counts:
            counts[label] = 0

    counts  = counts[counts > 0]
    colors  = [SENTIMENT_COLORS.get(s, '#95a5a6') for s in counts.index]
    explode = [0.05] * len(counts)

    fig, ax = plt.subplots(figsize=(7, 7))

    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels     = counts.index,
        colors     = colors,
        explode    = explode,
        autopct    = '%1.1f%%',
        startangle = 140,
        textprops  = {'fontsize': 11}
    )

    for autotext in autotexts:
        autotext.set_fontweight('bold')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()

    if save:
        save_figure(fig, filename)

    return fig


# ─────────────────────────────────────────
# CONFIDENCE SCORE HISTOGRAM
# ─────────────────────────────────────────

def plot_confidence_histogram(
    df    : pd.DataFrame,
    title : str  = 'Prediction Confidence Distribution',
    save  : bool = True
) -> plt.Figure:
    """
    Histogram of prediction confidence scores by sentiment.
    """
    set_plot_style()

    # ── FIX: reset index to avoid duplicate label error ──
    df = df.copy().reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    for sentiment, color in SENTIMENT_COLORS.items():
        # FIX: use boolean mask with reset index
        mask   = df['sentiment'].values == sentiment
        subset = df.loc[mask, 'confidence']

        if len(subset) > 0:
            ax.hist(
                subset,
                bins      = 20,
                alpha     = 0.6,
                color     = color,
                label     = sentiment,
                edgecolor = 'white'
            )

    ax.set_title(title,               fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Confidence Score', fontsize=11)
    ax.set_ylabel('Count',            fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save:
        save_figure(fig, "confidence_histogram.png")

    return fig

# ─────────────────────────────────────────
# METRICS SUMMARY TABLE
# ─────────────────────────────────────────

def plot_metrics_table(
    df_results : pd.DataFrame,
    save       : bool = True
) -> plt.Figure:
    """
    Render model metrics as a styled table figure.

    Args:
        df_results : DataFrame from compare_models()
        save       : if True auto-saves to outputs/plots/

    Returns:
        plt.Figure
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.axis('off')

    col_labels = df_results.columns.tolist()
    cell_text  = []

    for _, row in df_results.iterrows():
        cell_text.append([
            row['Model'],
            f"{row['Accuracy']:.4f}",
            f"{row['Precision']:.4f}",
            f"{row['Recall']:.4f}",
            f"{row['F1 Score']:.4f}"
        ])

    table = ax.table(
        cellText  = cell_text,
        colLabels = col_labels,
        cellLoc   = 'center',
        loc       = 'center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # Header styling
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Alternating row colors
    for i in range(1, len(cell_text) + 1):
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(
                '#ecf0f1' if i % 2 == 0 else 'white'
            )

    ax.set_title('Model Evaluation Summary',
                 fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()

    if save:
        save_figure(fig, "metrics_table.png")

    return fig


# ─────────────────────────────────────────
# LIST SAVED PLOTS
# ─────────────────────────────────────────

def list_saved_plots():
    """
    Print all saved plots in outputs/plots/ directory.
    Useful for debugging and verifying outputs.
    """
    if not os.path.exists(PLOTS_OUTPUT_DIR):
        print("[INFO] No plots saved yet.")
        return

    files = [f for f in os.listdir(PLOTS_OUTPUT_DIR)
             if f.endswith('.png')]

    if not files:
        print("[INFO] No PNG files in outputs/plots/")
        return

    print(f"\n[INFO] Saved plots in {PLOTS_OUTPUT_DIR}:")
    for f in sorted(files):
        path = os.path.join(PLOTS_OUTPUT_DIR, f)
        size = os.path.getsize(path) / 1024
        print(f"  📊 {f:<45} ({size:.1f} KB)")


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    import random

    print("="*50)
    print("  VISUALIZATION TEST")
    print("="*50)

    # ── Confusion matrix ──────────────────
    print("\n[1/5] Testing confusion matrix...")
    y_true = [0, 1, 2, 1, 0, 2, 1, 1, 0, 2] * 10
    y_pred = [0, 1, 1, 1, 0, 2, 0, 1, 2, 2] * 10
    fig = plot_confusion_matrix(y_true, y_pred, "SVM")
    plt.close()
    print("✅ Done")

    # ── Model comparison ──────────────────
    print("\n[2/5] Testing model comparison...")
    df_results = pd.DataFrame([
        {'Model': 'SVM',      'Accuracy': 0.7875, 'Precision': 0.7793,
         'Recall': 0.7875, 'F1 Score': 0.7810},
        {'Model': 'Logistic', 'Accuracy': 0.7456, 'Precision': 0.7547,
         'Recall': 0.7456, 'F1 Score': 0.7496},
    ])
    fig = plot_model_comparison(df_results)
    plt.close()
    print("✅ Done")

    # ── Sentiment distribution ────────────
    print("\n[3/5] Testing sentiment distribution...")
    sentiments = (['Positive'] * 306 +
                  ['Neutral']  * 978 +
                  ['Negative'] * 181)
    random.shuffle(sentiments)
    fig = plot_sentiment_distribution(sentiments)
    plt.close()
    fig = plot_sentiment_pie(sentiments)
    plt.close()
    print("✅ Done")

    # ── Confidence histogram ──────────────
    print("\n[4/5] Testing confidence histogram...")
    df_conf = pd.DataFrame({
        'sentiment'  : sentiments,
        'confidence' : [random.uniform(0.3, 0.9) for _ in sentiments]
    })
    fig = plot_confidence_histogram(df_conf)
    plt.close()
    print("✅ Done")

    # ── Metrics table ─────────────────────
    print("\n[5/5] Testing metrics table...")
    fig = plot_metrics_table(df_results)
    plt.close()
    print("✅ Done")

    # ── List all saved files ──────────────
    list_saved_plots()
    print("\n[SUCCESS] All plots saved to outputs/plots/")