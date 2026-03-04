# visualization/wordcloud_generator.py
# Generates word clouds for positive, negative and neutral tweets

import os
import sys

# ── Path fix — must be before any local imports ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

from config.config import (
    PROCESSED_DATA_PATH,
    SENTIMENT_LABELS,
    SENTIMENT_COLORS,
    PLOTS_OUTPUT_DIR
)


# ─────────────────────────────────────────
# OUTPUT HELPER
# ─────────────────────────────────────────

def get_output_path(filename: str) -> str:
    """Get full save path inside outputs/plots/"""
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
    return os.path.join(PLOTS_OUTPUT_DIR, filename)


# ─────────────────────────────────────────
# GENERATE SINGLE WORD CLOUD
# ─────────────────────────────────────────

def generate_wordcloud(
    text      : str,
    title     : str  = 'Word Cloud',
    color     : str  = '#2ecc71',
    save      : bool = True,
    filename  : str  = 'wordcloud.png',
    max_words : int  = 100
) -> plt.Figure:
    """
    Generate a styled word cloud from text.

    Args:
        text      : combined text string to visualize
        title     : chart title
        color     : sentiment color hex for colormap selection
        save      : if True saves to outputs/plots/
        filename  : output filename
        max_words : max number of words to display

    Returns:
        plt.Figure
    """
    # ── Handle empty text ─────────────────
    if not text or not text.strip():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(
            0.5, 0.5,
            f'No text available for\n{title}',
            ha='center', va='center',
            fontsize=14, color='gray'
        )
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold')
        return fig

    # ── Color map based on sentiment ──────
    color_map = {
        '#2ecc71' : 'Greens',    # Positive
        '#e74c3c' : 'Reds',      # Negative
        '#f39c12' : 'Oranges',   # Neutral
        '#3498db' : 'Blues',     # Default / All
    }
    cmap = color_map.get(color, 'Blues')

    # ── Build word cloud ──────────────────
    wc = WordCloud(
        width             = 900,
        height            = 450,
        background_color  = 'white',
        colormap          = cmap,
        max_words         = max_words,
        stopwords         = set(STOPWORDS),
        collocations      = False,
        prefer_horizontal = 0.85,
        min_font_size     = 10,
        max_font_size     = 90,
        random_state      = 42
    ).generate(text)

    # ── Plot ──────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()

    if save:
        path = get_output_path(filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Word cloud saved → {path}")

    return fig


# ─────────────────────────────────────────
# GENERATE PER-SENTIMENT WORD CLOUDS
# ─────────────────────────────────────────

def generate_sentiment_wordclouds(
    df   : pd.DataFrame,
    save : bool = True
) -> dict:
    """
    Generate separate word clouds for each sentiment class.

    Args:
        df   : DataFrame with 'cleaned_tweet' and 'sentiment' columns
        save : if True saves all to outputs/plots/

    Returns:
        dict: {
            'Positive' : plt.Figure,
            'Neutral'  : plt.Figure,
            'Negative' : plt.Figure
        }
    """
    print("\n[INFO] Generating per-sentiment word clouds...")

    figures = {}

    for label_num, sentiment in SENTIMENT_LABELS.items():

        # ── Filter by sentiment ───────────
        subset = df[df['sentiment'] == sentiment]['cleaned_tweet']
        subset = subset.dropna()

        if len(subset) == 0:
            print(f"[WARNING] No tweets for: {sentiment}")
            continue

        # ── Combine tweets ────────────────
        combined_text = ' '.join(subset.astype(str).tolist())

        print(f"[INFO] {sentiment:10} → {len(subset):4} tweets | "
              f"{len(combined_text.split()):5} words")

        # ── Generate ──────────────────────
        color    = SENTIMENT_COLORS.get(sentiment, '#3498db')
        filename = f"wordcloud_{sentiment.lower()}.png"

        fig = generate_wordcloud(
            text      = combined_text,
            title     = f'Word Cloud — {sentiment} Tweets',
            color     = color,
            save      = save,
            filename  = filename
        )

        figures[sentiment] = fig
        plt.close(fig)

    print(f"[SUCCESS] Generated {len(figures)} word clouds!")
    return figures


# ─────────────────────────────────────────
# GENERATE COMBINED WORD CLOUD
# ─────────────────────────────────────────

def generate_combined_wordcloud(
    df   : pd.DataFrame,
    save : bool = True
) -> plt.Figure:
    """
    Generate a single word cloud from ALL tweets combined.

    Args:
        df   : DataFrame with 'cleaned_tweet' column
        save : if True saves to outputs/plots/

    Returns:
        plt.Figure
    """
    print("[INFO] Generating combined word cloud...")

    all_text = ' '.join(
        df['cleaned_tweet'].dropna().astype(str).tolist()
    )

    fig = generate_wordcloud(
        text     = all_text,
        title    = 'Word Cloud — All Tweets Combined',
        color    = '#3498db',
        save     = save,
        filename = 'wordcloud_all.png'
    )

    return fig


# ─────────────────────────────────────────
# GENERATE SIDE-BY-SIDE COMPARISON
# ─────────────────────────────────────────

def generate_comparison_wordcloud(
    df   : pd.DataFrame,
    save : bool = True
) -> plt.Figure:
    """
    Generate positive vs negative word clouds
    side by side in one figure.

    Args:
        df   : DataFrame with 'cleaned_tweet' and 'sentiment' columns
        save : if True saves to outputs/plots/

    Returns:
        plt.Figure
    """
    print("[INFO] Generating comparison word cloud...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, sentiment, cmap in zip(
        axes,
        ['Positive', 'Negative'],
        ['Greens',   'Reds']
    ):
        subset = df[df['sentiment'] == sentiment]['cleaned_tweet']
        subset = subset.dropna()

        if len(subset) == 0:
            ax.text(0.5, 0.5, f'No {sentiment} tweets',
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            continue

        combined = ' '.join(subset.astype(str).tolist())

        wc = WordCloud(
            width             = 800,
            height            = 400,
            background_color  = 'white',
            colormap          = cmap,
            max_words         = 80,
            stopwords         = set(STOPWORDS),
            collocations      = False,
            prefer_horizontal = 0.85,
            random_state      = 42
        ).generate(combined)

        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')

        color = SENTIMENT_COLORS.get(sentiment, 'black')
        ax.set_title(
            f'{sentiment} Tweets  ({len(subset)} total)',
            fontsize  = 14,
            fontweight = 'bold',
            color     = color,
            pad       = 12
        )

    fig.suptitle(
        'Positive vs Negative — Word Cloud Comparison',
        fontsize   = 16,
        fontweight = 'bold',
        y          = 1.02
    )

    plt.tight_layout()

    if save:
        path = get_output_path('wordcloud_comparison.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Comparison word cloud saved → {path}")

    return fig


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("="*50)
    print("  WORD CLOUD GENERATOR TEST")
    print("="*50)

    # ── Load processed data ───────────────
    if not os.path.exists(PROCESSED_DATA_PATH):
        print("[ERROR] cleaned_data.csv not found!")
        print("[INFO]  Run: python -m preprocessing.clean_text")
        sys.exit(1)

    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"[INFO] Loaded {len(df)} rows")

    # ── Map numeric labels → sentiment names ──
    df['sentiment'] = df['label'].map(SENTIMENT_LABELS)
    print(f"[INFO] Sentiment distribution:")
    print(df['sentiment'].value_counts().to_string())

    # ── Individual word clouds ────────────
    print("\n[1/3] Per-sentiment word clouds...")
    figs = generate_sentiment_wordclouds(df, save=True)
    print(f"✅ Generated: {list(figs.keys())}")

    # ── Combined word cloud ───────────────
    print("\n[2/3] Combined word cloud...")
    fig = generate_combined_wordcloud(df, save=True)
    plt.close(fig)
    print("✅ Done")

    # ── Comparison word cloud ─────────────
    print("\n[3/3] Comparison word cloud...")
    fig = generate_comparison_wordcloud(df, save=True)
    plt.close(fig)
    print("✅ Done")

    # ── List saved files ──────────────────
    print(f"\n[INFO] Saved word clouds:")
    for f in sorted(os.listdir(PLOTS_OUTPUT_DIR)):
        if 'wordcloud' in f:
            path = os.path.join(PLOTS_OUTPUT_DIR, f)
            size = os.path.getsize(path) / 1024
            print(f"  🌥️  {f:<45} ({size:.1f} KB)")

    print("\n[SUCCESS] All word clouds generated!")