# visualization/sentiment_distribution.py
# Leader-wise sentiment breakdown charts

import os
import sys

# ── Path fix ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
# ──────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config.config import (
    PROCESSED_DATA_PATH,
    SENTIMENT_LABELS,
    SENTIMENT_COLORS,
    POLITICAL_LEADERS,
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
# EXTRACT LEADER MENTIONS
# ─────────────────────────────────────────

def extract_leader_sentiments(
    df      : pd.DataFrame,
    leaders : list = None
) -> pd.DataFrame:
    """
    Filter tweets mentioning each political leader/entity
    and compute sentiment distribution per leader.

    Args:
        df      : DataFrame with 'text' and 'sentiment' columns
        leaders : list of leader/entity names to search for
                  defaults to POLITICAL_LEADERS from config

    Returns:
        pd.DataFrame with columns:
            Leader, Positive, Neutral, Negative,
            Total, Positive%, Neutral%, Negative%
    """
    if leaders is None:
        leaders = POLITICAL_LEADERS

    print(f"[INFO] Analyzing {len(leaders)} political entities...")

    results = []

    for leader in leaders:
        # Case-insensitive search in tweet text
        mask   = df['text'].str.contains(
            leader, case=False, na=False
        )
        subset = df[mask]

        if len(subset) == 0:
            print(f"[WARNING] No tweets found for: {leader}")
            continue

        total    = len(subset)
        counts   = subset['sentiment'].value_counts()

        positive = int(counts.get('Positive', 0))
        neutral  = int(counts.get('Neutral',  0))
        negative = int(counts.get('Negative', 0))

        results.append({
            'Leader'    : leader,
            'Positive'  : positive,
            'Neutral'   : neutral,
            'Negative'  : negative,
            'Total'     : total,
            'Positive%' : round(positive / total * 100, 1),
            'Neutral%'  : round(neutral  / total * 100, 1),
            'Negative%' : round(negative / total * 100, 1),
        })

        print(f"[INFO] {leader:12} → {total:4} tweets | "
              f"+{positive/total*100:.0f}% "
              f"~{neutral/total*100:.0f}% "
              f"-{negative/total*100:.0f}%")

    if not results:
        print("[WARNING] No leader mentions found in dataset!")
        return pd.DataFrame()

    df_leaders = pd.DataFrame(results)
    df_leaders = df_leaders.sort_values('Total', ascending=False)
    df_leaders = df_leaders.reset_index(drop=True)

    return df_leaders


# ─────────────────────────────────────────
# STACKED BAR CHART — LEADER WISE
# ─────────────────────────────────────────

def plot_leader_sentiment_bars(
    df_leaders : pd.DataFrame,
    save       : bool = True
) -> plt.Figure:
    """
    Stacked horizontal bar chart showing sentiment
    breakdown per political leader/entity.

    Args:
        df_leaders : DataFrame from extract_leader_sentiments()
        save       : if True saves to outputs/plots/

    Returns:
        plt.Figure
    """
    if df_leaders.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, 'No leader data available',
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    fig, ax = plt.subplots(figsize=(11, max(5, len(df_leaders) * 0.9)))

    leaders   = df_leaders['Leader'].tolist()
    positives = df_leaders['Positive%'].tolist()
    neutrals  = df_leaders['Neutral%'].tolist()
    negatives = df_leaders['Negative%'].tolist()

    y = np.arange(len(leaders))

    # ── Stacked bars ──────────────────────
    bars_pos = ax.barh(y, positives, 0.6,
                       color=SENTIMENT_COLORS['Positive'],
                       label='Positive')
    bars_neu = ax.barh(y, neutrals,  0.6,
                       left=positives,
                       color=SENTIMENT_COLORS['Neutral'],
                       label='Neutral')
    bars_neg = ax.barh(y, negatives, 0.6,
                       left=[p + n for p, n in zip(positives, neutrals)],
                       color=SENTIMENT_COLORS['Negative'],
                       label='Negative')

    # ── Value labels inside bars ──────────
    for i, (pos, neu, neg) in enumerate(
        zip(positives, neutrals, negatives)
    ):
        if pos > 8:
            ax.text(pos / 2, i, f'{pos:.0f}%',
                    ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
        if neu > 8:
            ax.text(pos + neu / 2, i, f'{neu:.0f}%',
                    ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
        if neg > 8:
            ax.text(pos + neu + neg / 2, i, f'{neg:.0f}%',
                    ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')

    # ── Total tweet count labels ──────────
    for i, total in enumerate(df_leaders['Total'].tolist()):
        ax.text(101, i, f'n={total}',
                va='center', fontsize=9, color='gray')

    # ── Styling ───────────────────────────
    ax.set_yticks(y)
    ax.set_yticklabels(leaders, fontsize=11)
    ax.set_xlim(0, 115)
    ax.set_xlabel('Percentage (%)', fontsize=11)
    ax.set_title(
        'Leader-wise Sentiment Breakdown',
        fontsize=14, fontweight='bold', pad=15
    )
    ax.legend(loc='lower right', fontsize=10)
    ax.axvline(x=50, color='gray', linestyle='--',
               alpha=0.4, linewidth=0.8)
    ax.grid(axis='x', alpha=0.2)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save:
        path = get_output_path('leader_sentiment_bars.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Leader bar chart saved → {path}")

    return fig


# ─────────────────────────────────────────
# PIE CHARTS — PER LEADER
# ─────────────────────────────────────────

def plot_leader_sentiment_pies(
    df_leaders : pd.DataFrame,
    save       : bool = True
) -> plt.Figure:
    """
    Grid of pie charts — one per political leader.

    Args:
        df_leaders : DataFrame from extract_leader_sentiments()
        save       : if True saves to outputs/plots/

    Returns:
        plt.Figure
    """
    if df_leaders.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.axis('off')
        return fig

    n       = len(df_leaders)
    n_cols  = min(3, n)
    n_rows  = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 5, n_rows * 5)
    )

    # Flatten axes for easy iteration
    if n == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    colors = [
        SENTIMENT_COLORS['Positive'],
        SENTIMENT_COLORS['Neutral'],
        SENTIMENT_COLORS['Negative']
    ]

    for i, (_, row) in enumerate(df_leaders.iterrows()):
        ax     = axes[i]
        values = [row['Positive'], row['Neutral'], row['Negative']]
        labels = ['Positive', 'Neutral', 'Negative']

        # Remove zero slices
        filtered = [(v, l, c) for v, l, c in
                    zip(values, labels, colors) if v > 0]

        if not filtered:
            ax.text(0.5, 0.5, 'No data',
                    ha='center', va='center')
            ax.axis('off')
            continue

        vals, labs, cols = zip(*filtered)

        ax.pie(
            vals,
            labels    = labs,
            colors    = cols,
            autopct   = '%1.1f%%',
            startangle = 140,
            textprops  = {'fontsize': 9}
        )
        ax.set_title(
            f"{row['Leader']}\n(n={row['Total']})",
            fontsize=11, fontweight='bold'
        )

    # ── Hide empty subplots ───────────────
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(
        'Leader-wise Sentiment Pie Charts',
        fontsize=15, fontweight='bold', y=1.02
    )

    plt.tight_layout()

    if save:
        path = get_output_path('leader_sentiment_pies.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Leader pie charts saved → {path}")

    return fig


# ─────────────────────────────────────────
# GROUPED BAR CHART — SIDE BY SIDE
# ─────────────────────────────────────────

def plot_leader_grouped_bars(
    df_leaders : pd.DataFrame,
    save       : bool = True
) -> plt.Figure:
    """
    Grouped bar chart — Positive / Neutral / Negative
    counts side by side per leader.

    Args:
        df_leaders : DataFrame from extract_leader_sentiments()
        save       : if True saves to outputs/plots/

    Returns:
        plt.Figure
    """
    if df_leaders.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.axis('off')
        return fig

    x     = np.arange(len(df_leaders))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width, df_leaders['Positive'], width,
           label='Positive', color=SENTIMENT_COLORS['Positive'],
           edgecolor='white')
    ax.bar(x,          df_leaders['Neutral'],  width,
           label='Neutral',  color=SENTIMENT_COLORS['Neutral'],
           edgecolor='white')
    ax.bar(x + width, df_leaders['Negative'], width,
           label='Negative', color=SENTIMENT_COLORS['Negative'],
           edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(df_leaders['Leader'], fontsize=11)
    ax.set_ylabel('Tweet Count', fontsize=11)
    ax.set_title(
        'Leader-wise Sentiment Count Comparison',
        fontsize=14, fontweight='bold', pad=15
    )
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save:
        path = get_output_path('leader_grouped_bars.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Leader grouped bars saved → {path}")

    return fig


# ─────────────────────────────────────────
# OVERALL SENTIMENT SCORE TABLE
# ─────────────────────────────────────────

def plot_leader_score_table(
    df_leaders : pd.DataFrame,
    save       : bool = True
) -> plt.Figure:
    """
    Styled table showing sentiment score per leader.
    Score = (Positive - Negative) / Total

    Args:
        df_leaders : DataFrame from extract_leader_sentiments()
        save       : if True saves to outputs/plots/

    Returns:
        plt.Figure
    """
    if df_leaders.empty:
        fig, ax = plt.subplots()
        ax.axis('off')
        return fig

    # ── Compute score ─────────────────────
    df_leaders = df_leaders.copy()
    df_leaders['Score'] = (
        (df_leaders['Positive'] - df_leaders['Negative'])
        / df_leaders['Total']
    ).round(3)

    df_leaders['Mood'] = df_leaders['Score'].apply(
        lambda s: '🟢 Positive' if s >= 0.1
        else ('🔴 Negative' if s <= -0.1 else '🟡 Neutral')
    )

    df_display = df_leaders[[
        'Leader', 'Total', 'Positive%',
        'Neutral%', 'Negative%', 'Score', 'Mood'
    ]].copy()

    fig, ax = plt.subplots(
        figsize=(12, max(2.5, len(df_leaders) * 0.7))
    )
    ax.axis('off')

    table = ax.table(
        cellText  = df_display.values,
        colLabels = df_display.columns,
        cellLoc   = 'center',
        loc       = 'center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Header styling
    for j in range(len(df_display.columns)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(
            color='white', fontweight='bold'
        )

    # Row coloring based on mood
    for i, (_, row) in enumerate(df_leaders.iterrows(), start=1):
        color = (
            '#d5f5e3' if row['Score'] >= 0.1
            else '#fadbd8' if row['Score'] <= -0.1
            else '#fef9e7'
        )
        for j in range(len(df_display.columns)):
            table[i, j].set_facecolor(color)

    ax.set_title(
        'Leader Sentiment Score Summary',
        fontsize=13, fontweight='bold', pad=20
    )

    plt.tight_layout()

    if save:
        path = get_output_path('leader_score_table.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Score table saved → {path}")

    return fig


# ─────────────────────────────────────────
# RUN ALL LEADER VISUALIZATIONS
# ─────────────────────────────────────────

def generate_all_leader_charts(
    df   : pd.DataFrame,
    save : bool = True
) -> dict:
    """
    Generate all leader-wise charts in one call.

    Args:
        df   : DataFrame with 'text' and 'sentiment' columns
        save : if True saves all to outputs/plots/

    Returns:
        dict of plt.Figure objects
    """
    print("\n" + "="*50)
    print("  LEADER SENTIMENT ANALYSIS")
    print("="*50)

    # ── Extract leader data ───────────────
    df_leaders = extract_leader_sentiments(df)

    if df_leaders.empty:
        print("[WARNING] No leader data — charts skipped.")
        return {}

    print(f"\n[INFO] Leaders found: {df_leaders['Leader'].tolist()}")

    # ── Generate all charts ───────────────
    figures = {}

    print("\n[1/4] Stacked bar chart...")
    figures['bars']   = plot_leader_sentiment_bars(df_leaders, save)
    plt.close()

    print("[2/4] Pie charts...")
    figures['pies']   = plot_leader_sentiment_pies(df_leaders, save)
    plt.close()

    print("[3/4] Grouped bar chart...")
    figures['grouped'] = plot_leader_grouped_bars(df_leaders, save)
    plt.close()

    print("[4/4] Score table...")
    figures['table']  = plot_leader_score_table(df_leaders, save)
    plt.close()

    print(f"\n[SUCCESS] All leader charts generated!")
    print("="*50)

    return figures, df_leaders


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("="*50)
    print("  SENTIMENT DISTRIBUTION TEST")
    print("="*50)

    # ── Load processed data ───────────────
    if not os.path.exists(PROCESSED_DATA_PATH):
        print("[ERROR] Run preprocessing/clean_text.py first!")
        sys.exit(1)

    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"[INFO] Loaded {len(df)} rows")

    # ── Map numeric labels → names ────────
    df['sentiment'] = df['label'].map(SENTIMENT_LABELS)

    # ── Generate all charts ───────────────
    result = generate_all_leader_charts(df, save=True)

    if result:
        figures, df_leaders = result
        print("\n[INFO] Leader Summary:")
        print(df_leaders[[
            'Leader', 'Total', 'Positive%',
            'Neutral%', 'Negative%'
        ]].to_string(index=False))

    # ── List saved files ──────────────────
    print(f"\n[INFO] Saved leader charts:")
    for f in sorted(os.listdir(PLOTS_OUTPUT_DIR)):
        if 'leader' in f:
            path = os.path.join(PLOTS_OUTPUT_DIR, f)
            size = os.path.getsize(path) / 1024
            print(f"  📊 {f:<45} ({size:.1f} KB)")

    print("\n[SUCCESS] Leader analysis complete!")