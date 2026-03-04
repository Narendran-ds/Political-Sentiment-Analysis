# app.py
# Streamlit Dashboard — Political Sentiment Intelligence
# Run with: python run.py  OR  streamlit run app.py

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nltk
nltk.download("punkt")
nltk.download("stopwords")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import (
    PROCESSED_DATA_PATH,
    SENTIMENT_LABELS,
    SENTIMENT_COLORS,
    POLITICAL_LEADERS,
    APP_TITLE,
    APP_ICON,
    APP_LAYOUT
)


# ─────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────

st.set_page_config(
    page_title = APP_TITLE,
    page_icon  = APP_ICON,
    layout     = APP_LAYOUT
)


# ─────────────────────────────────────────
# CACHED RESOURCE LOADERS
# ─────────────────────────────────────────

@st.cache_resource
def load_models():
    """Load all models once — cached for session."""
    from models.model_loader import load_all_models
    return load_all_models(verbose=False)


@st.cache_data
def load_dataset():
    """Load processed dataset — cached for session."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        return pd.DataFrame()
    df = pd.read_csv(PROCESSED_DATA_PATH)


@st.cache_data
def run_batch_predictions(model_name: str):
    """
    Run predictions on entire dataset.
    Cached by model_name — reruns only if model changes.
    """
    from models.predict import predict_batch
    components = load_models()
    df         = load_dataset()

    if df.empty:
        return pd.DataFrame()

    texts      = df['text'].fillna('').tolist()
    pred_df    = predict_batch(texts, model_name, components)

    # Merge with original
    df = df.copy().reset_index(drop=True)
    df['predicted_sentiment'] = pred_df['sentiment'].values
    df['confidence']          = pred_df['confidence'].values
    df['Negative']            = pred_df['Negative'].values
    df['Neutral']             = pred_df['Neutral'].values
    df['Positive']            = pred_df['Positive'].values

    return df


# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
        /* Main header */
        .main-header {
            font-size: 2.2rem;
            font-weight: 800;
            color: #2c3e50;
            text-align: center;
            padding: 1rem 0 0.5rem 0;
        }
        .sub-header {
            font-size: 1rem;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 2rem;
        }

        /* Metric cards */
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 1.2rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border-left: 4px solid #3498db;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #2c3e50;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #7f8c8d;
            margin-top: 0.3rem;
        }

        /* Sentiment badges */
        .badge-positive {
            background: #d5f5e3;
            color: #1e8449;
            padding: 3px 10px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .badge-negative {
            background: #fadbd8;
            color: #922b21;
            padding: 3px 10px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .badge-neutral {
            background: #fef9e7;
            color: #9a7d0a;
            padding: 3px 10px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.85rem;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.95rem;
            font-weight: 600;
        }

        /* Sidebar */
        .sidebar-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1rem;
        }

        /* Footer */
        .footer {
            text-align: center;
            color: #bdc3c7;
            font-size: 0.8rem;
            padding: 2rem 0 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

def render_sidebar():
    """Render sidebar controls."""
    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        st.markdown("---")

        # ── Model selector ────────────────
        st.markdown("**🤖 Select Model**")
        model_name = st.selectbox(
            label      = "Model",
            options    = ["svm", "logistic"],
            index      = 0,
            format_func = lambda x: {
                "svm"      : "SVM (LinearSVC) — Best",
                "logistic" : "Logistic Regression"
            }.get(x, x),
            label_visibility = "collapsed"
        )

        st.markdown("---")

        # ── Live tweet section ────────────
        st.markdown("**🐦 Live Tweet Analysis**")
        keyword = st.text_input(
            "Search Keyword",
            value       = "Ukraine NATO",
            placeholder = "e.g. Russia, NATO, Zelensky"
        )
        tweet_count = st.slider(
            "Number of Tweets",
            min_value = 5,
            max_value = 50,
            value     = 10,
            step      = 5
        )

        fetch_btn = st.button(
            "🔍 Fetch & Analyze",
            use_container_width = True,
            type = "primary"
        )

        st.markdown("---")

        # ── Single tweet tester ───────────
        st.markdown("**✍️ Test Single Tweet**")
        test_tweet = st.text_area(
            "Enter tweet text",
            value       = "NATO is supporting Ukraine strongly",
            height      = 100,
            label_visibility = "visible"
        )
        predict_btn = st.button(
            "🔮 Predict Sentiment",
            use_container_width = True
        )

        st.markdown("---")
        st.markdown("""
        <div style='font-size:0.8rem; color:#95a5a6;'>
        📊 Dataset: Ukraine-Russia War Tweets<br>
        🗂️ 1,433 cleaned tweets<br>
        🤖 Models: SVM + Logistic Regression<br>
        🔍 Features: TF-IDF (2,465 features)
        </div>
        """, unsafe_allow_html=True)

    return model_name, keyword, tweet_count, fetch_btn, \
           test_tweet, predict_btn


# ─────────────────────────────────────────
# TAB 1 — MODEL PERFORMANCE
# ─────────────────────────────────────────

def render_model_performance():
    """Tab 1 — Model metrics and confusion matrix."""
    st.markdown("### 📊 Model Performance Comparison")

    from models.evaluate_model import evaluate_model, compare_models
    from visualization.plots  import (
        plot_confusion_matrix,
        plot_model_comparison,
        plot_metrics_table
    )

    components = load_models()
    df         = load_dataset()

    if df.empty:
        st.error("Dataset not found!")
        return

    # ── Rebuild features for evaluation ──
    from preprocessing.feature_engineering import prepare_features
    with st.spinner("Evaluating models..."):
        X_train, X_test, y_train, y_test, vectorizer = \
            prepare_features(verbose=False)

        df_results = compare_models(
            {
                'SVM'      : components['svm'],
                'Logistic' : components['logistic']
            },
            X_test, y_test
        )

    # ── Metric cards ──────────────────────
    st.markdown("#### Best Model — SVM (LinearSVC)")
    col1, col2, col3, col4 = st.columns(4)

    svm_row = df_results[df_results['Model'] == 'SVM'].iloc[0]

    with col1:
        st.metric("Accuracy",  f"{svm_row['Accuracy']:.2%}")
    with col2:
        st.metric("Precision", f"{svm_row['Precision']:.2%}")
    with col3:
        st.metric("Recall",    f"{svm_row['Recall']:.2%}")
    with col4:
        st.metric("F1 Score",  f"{svm_row['F1 Score']:.2%}")

    st.markdown("---")

    # ── Comparison chart ──────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Model Comparison")
        fig = plot_model_comparison(df_results, save=False)
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.markdown("#### Metrics Table")
        fig = plot_metrics_table(df_results, save=False)
        st.pyplot(fig)
        plt.close()

    # ── Confusion matrices ─────────────────
    st.markdown("---")
    st.markdown("#### Confusion Matrices")

    col1, col2 = st.columns(2)

    with col1:
        y_pred_svm = components['svm'].predict(X_test)
        fig = plot_confusion_matrix(
            y_test, y_pred_svm, "SVM", save=False
        )
        st.pyplot(fig)
        plt.close()

    with col2:
        y_pred_log = components['logistic'].predict(X_test)
        fig = plot_confusion_matrix(
            y_test, y_pred_log, "Logistic", save=False
        )
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────
# TAB 2 — LIVE TWEET ANALYSIS
# ─────────────────────────────────────────

def render_live_analysis(
    keyword     : str,
    tweet_count : int,
    fetch_btn   : bool,
    model_name  : str
):
    """Tab 2 — Live tweet fetching and prediction."""
    st.markdown("### 🐦 Live Tweet Analysis")

    from twitter.fetch_tweets   import fetch_tweets, validate_tweets
    from models.predict         import predict_batch
    from visualization.plots    import plot_sentiment_distribution

    components = load_models()

    if fetch_btn:
        with st.spinner(f"Fetching tweets for '{keyword}'..."):
            df_tweets = fetch_tweets(keyword, tweet_count)
            df_tweets = validate_tweets(df_tweets)

        if df_tweets.empty:
            st.warning("No tweets found. Try a different keyword.")
            return

        with st.spinner("Running sentiment analysis..."):
            pred_df = predict_batch(
                df_tweets['text'].tolist(),
                model_name,
                components
            )
            df_tweets['sentiment']  = pred_df['sentiment'].values
            df_tweets['confidence'] = pred_df['confidence'].values
            df_tweets['Negative']   = pred_df['Negative'].values
            df_tweets['Neutral']    = pred_df['Neutral'].values
            df_tweets['Positive']   = pred_df['Positive'].values

        # ── Summary metrics ───────────────
        st.success(f"✅ Analyzed {len(df_tweets)} tweets!")

        counts = df_tweets['sentiment'].value_counts()
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🟢 Positive",
                counts.get('Positive', 0),
                f"{counts.get('Positive',0)/len(df_tweets)*100:.1f}%"
            )
        with col2:
            st.metric(
                "🟡 Neutral",
                counts.get('Neutral', 0),
                f"{counts.get('Neutral',0)/len(df_tweets)*100:.1f}%"
            )
        with col3:
            st.metric(
                "🔴 Negative",
                counts.get('Negative', 0),
                f"{counts.get('Negative',0)/len(df_tweets)*100:.1f}%"
            )

        st.markdown("---")

        # ── Distribution chart ────────────
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Sentiment Distribution")
            fig = plot_sentiment_distribution(
                df_tweets['sentiment'].tolist(),
                title = f"Sentiment for '{keyword}'",
                save  = False
            )
            st.pyplot(fig)
            plt.close()

        with col_right:
            st.markdown("#### Tweet Results")
            display_df = df_tweets[
                ['text', 'sentiment', 'confidence']
            ].copy()
            display_df['confidence'] = display_df['confidence'].apply(
                lambda x: f"{x:.1%}"
            )
            display_df.columns = ['Tweet', 'Sentiment', 'Confidence']
            st.dataframe(display_df, use_container_width=True, height=350)

    else:
        st.info(
            "👈 Enter a keyword and click **Fetch & Analyze** "
            "in the sidebar to analyze live tweets."
        )


# ─────────────────────────────────────────
# TAB 3 — DATASET ANALYSIS
# ─────────────────────────────────────────

def render_dataset_analysis(model_name: str):
    """Tab 3 — Full dataset sentiment analysis."""
    st.markdown("### 📈 Dataset Sentiment Analysis")

    from visualization.plots import (
        plot_sentiment_distribution,
        plot_sentiment_pie,
        plot_confidence_histogram
    )

    with st.spinner("Running predictions on dataset..."):
        df = run_batch_predictions(model_name)

    if df.empty:
        st.error("Dataset not found!")
        return

    # ── Summary stats ─────────────────────
    col1, col2, col3, col4 = st.columns(4)
    counts = df['predicted_sentiment'].value_counts()
    total  = len(df)

    with col1:
        st.metric("Total Tweets", f"{total:,}")
    with col2:
        st.metric("Positive", f"{counts.get('Positive',0):,}",
                  f"{counts.get('Positive',0)/total*100:.1f}%")
    with col3:
        st.metric("Neutral",  f"{counts.get('Neutral',0):,}",
                  f"{counts.get('Neutral',0)/total*100:.1f}%")
    with col4:
        st.metric("Negative", f"{counts.get('Negative',0):,}",
                  f"{counts.get('Negative',0)/total*100:.1f}%")

    st.markdown("---")

    # ── Charts ────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Sentiment Distribution")
        fig = plot_sentiment_distribution(
            df['predicted_sentiment'].tolist(),
            title = 'Predicted Sentiment Distribution',
            save  = False
        )
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Sentiment Breakdown")
        fig = plot_sentiment_pie(
            df['predicted_sentiment'].tolist(),
            title = 'Sentiment Breakdown',
            save  = False
        )
        st.pyplot(fig)
        plt.close()

    # ── Confidence histogram ──────────────
    st.markdown("---")
    st.markdown("#### Prediction Confidence Distribution")
    fig = plot_confidence_histogram(
        df.rename(columns={'predicted_sentiment': 'sentiment'}).reset_index(drop=True),
        save=False
    )
    st.pyplot(fig)
    plt.close()

    # ── Sample predictions table ──────────
    st.markdown("---")
    st.markdown("#### Sample Predictions")
    sample = df[['text', 'predicted_sentiment', 'confidence']].head(20)
    sample.columns = ['Tweet', 'Sentiment', 'Confidence']
    sample['Confidence'] = sample['Confidence'].apply(
        lambda x: f"{x:.1%}"
    )
    st.dataframe(sample, use_container_width=True)


# ─────────────────────────────────────────
# TAB 4 — WORD CLOUDS
# ─────────────────────────────────────────

def render_wordclouds():
    """Tab 4 — Word cloud visualizations."""
    st.markdown("### 🌥️ Word Cloud Analysis")

    from visualization.wordcloud_generator import (
        generate_wordcloud,
        generate_comparison_wordcloud
    )

    df = load_dataset()
    if df.empty:
        st.error("Dataset not found!")
        return

    # ── Selector ──────────────────────────
    wc_type = st.radio(
        "Select Word Cloud",
        ["All Tweets", "Positive", "Neutral",
         "Negative", "Positive vs Negative"],
        horizontal = True
    )

    with st.spinner("Generating word cloud..."):

        if wc_type == "Positive vs Negative":
            fig = generate_comparison_wordcloud(df, save=False)
            st.pyplot(fig)
            plt.close()

        elif wc_type == "All Tweets":
            text  = ' '.join(df['cleaned_tweet'].dropna().tolist())
            fig   = generate_wordcloud(
                text     = text,
                title    = "All Tweets — Word Cloud",
                color    = '#3498db',
                save     = False
            )
            st.pyplot(fig)
            plt.close()

        else:
            subset = df[df['sentiment'] == wc_type]['cleaned_tweet']
            text   = ' '.join(subset.dropna().tolist())
            color  = SENTIMENT_COLORS.get(wc_type, '#3498db')
            fig    = generate_wordcloud(
                text     = text,
                title    = f"{wc_type} Tweets — Word Cloud",
                color    = color,
                save     = False
            )
            st.pyplot(fig)
            plt.close()


# ─────────────────────────────────────────
# TAB 5 — SHAP EXPLAINABILITY
# ─────────────────────────────────────────

def render_shap(test_tweet: str, predict_btn: bool):
    """Tab 5 — SHAP explanation for single tweet."""
    st.markdown("### 🔍 Prediction Explainability (SHAP)")

    from explainability.shap_explainer import (
        explain_prediction,
        plot_shap_explanation
    )
    from models.predict import predict_single

    components = load_models()

    st.markdown(
        "SHAP (SHapley Additive exPlanations) shows "
        "**which words** pushed the model toward its prediction."
    )

    if predict_btn or test_tweet:
        with st.spinner("Running prediction and SHAP analysis..."):
            # ── Single prediction ─────────
            result = predict_single(
                text       = test_tweet,
                model_name = 'svm',
                components = components,
                verbose    = False
            )

            # ── SHAP explanation ──────────
            explanation = explain_prediction(
                text       = test_tweet,
                components = components,
                top_n      = 10
            )

        # ── Prediction result ─────────────
        sentiment = result['sentiment']
        color_map = {
            'Positive': 'green',
            'Neutral' : 'orange',
            'Negative': 'red'
        }
        color = color_map.get(sentiment, 'gray')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Sentiment", sentiment)
        with col2:
            st.metric("Confidence", f"{result['confidence']:.1%}")
        with col3:
            st.metric("Cleaned Text", result['cleaned_text'][:30] + "...")

        st.markdown("---")

        # ── Probability bars ──────────────
        st.markdown("#### Class Probabilities")
        probs = result['probabilities']

        for label, prob in probs.items():
            color = SENTIMENT_COLORS.get(label, '#95a5a6')
            st.markdown(f"**{label}**")
            st.progress(prob)
            st.caption(f"{prob:.1%}")

        st.markdown("---")

        # ── SHAP chart ────────────────────
        if explanation:
            st.markdown("#### Word Contributions")
            fig = plot_shap_explanation(explanation, top_n=10)
            st.pyplot(fig)
            plt.close()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top Positive Words** (push toward prediction)")
                for word, val in explanation['top_positive'][:5]:
                    st.markdown(f"- `{word}` → `+{val:.4f}`")

            with col2:
                st.markdown("**Top Negative Words** (push away from prediction)")
                for word, val in explanation['top_negative'][:5]:
                    st.markdown(f"- `{word}` → `{val:.4f}`")
        else:
            st.warning("SHAP explanation not available for this tweet.")

    else:
        st.info(
            "👈 Enter a tweet in the sidebar and click "
            "**Predict Sentiment** to see the explanation."
        )


# ─────────────────────────────────────────
# TAB 6 — LEADER ANALYSIS
# ─────────────────────────────────────────

def render_leader_analysis():
    """Tab 6 — Leader-wise sentiment breakdown."""
    st.markdown("### 🏛️ Political Leader Sentiment Analysis")

    from visualization.sentiment_distribution import (
        extract_leader_sentiments,
        plot_leader_sentiment_bars,
        plot_leader_grouped_bars,
        plot_leader_score_table
    )

    df = load_dataset()
    if df.empty:
        st.error("Dataset not found!")
        return

    # ── Custom leader input ───────────────
    st.markdown("#### Search Political Entities")
    default_leaders = ', '.join(POLITICAL_LEADERS)
    leaders_input   = st.text_input(
        "Leaders / Keywords (comma separated)",
        value = default_leaders
    )
    leaders = [l.strip() for l in leaders_input.split(',') if l.strip()]

    with st.spinner("Analyzing leader sentiments..."):
        df_leaders = extract_leader_sentiments(df, leaders)

    if df_leaders.empty:
        st.warning("No mentions found for selected leaders.")
        return

    # ── Score table ───────────────────────
    st.markdown("#### Sentiment Score Summary")
    fig = plot_leader_score_table(df_leaders, save=False)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Stacked bars ──────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Sentiment % Breakdown")
        fig = plot_leader_sentiment_bars(df_leaders, save=False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Tweet Count Comparison")
        fig = plot_leader_grouped_bars(df_leaders, save=False)
        st.pyplot(fig)
        plt.close()

    # ── Raw data ──────────────────────────
    st.markdown("---")
    st.markdown("#### Raw Leader Data")
    st.dataframe(
        df_leaders[[
            'Leader', 'Total', 'Positive',
            'Neutral', 'Negative',
            'Positive%', 'Neutral%', 'Negative%'
        ]],
        use_container_width=True
    )


# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────

def main():
    """Main Streamlit app entry point."""

    # ── CSS ───────────────────────────────
    inject_css()

    # ── Header ────────────────────────────
    st.markdown(
        f'<div class="main-header">{APP_ICON} {APP_TITLE}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">'
        'Real-Time Political Sentiment Analysis using Machine Learning'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Sidebar ───────────────────────────
    model_name, keyword, tweet_count, fetch_btn, \
    test_tweet, predict_btn = render_sidebar()

    # ── Tabs ──────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Model Performance",
        "🐦 Live Analysis",
        "📈 Dataset Analysis",
        "🌥️ Word Clouds",
        "🔍 Explainability",
        "🏛️ Leader Analysis"
    ])

    with tab1:
        render_model_performance()

    with tab2:
        render_live_analysis(
            keyword, tweet_count, fetch_btn, model_name
        )

    with tab3:
        render_dataset_analysis(model_name)

    with tab4:
        render_wordclouds()

    with tab5:
        render_shap(test_tweet, predict_btn)

    with tab6:
        render_leader_analysis()

    # ── Footer ────────────────────────────
    st.markdown(
        '<div class="footer">'
        'Political Sentiment Intelligence Dashboard • '
        'Built with Streamlit + scikit-learn + SHAP'
        '</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    main()