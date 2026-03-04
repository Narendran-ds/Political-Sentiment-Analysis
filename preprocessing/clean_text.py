# preprocessing/clean_text.py
# Handles all text cleaning logic for raw political tweets

import re
import string
import nltk
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    STOPWORDS_PATH,
    TEXT_COLUMN,
    SENTIMENT_COLUMN,
    USER_COLUMN,
    SENTIMENT_MAP
)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ─────────────────────────────────────────
# NLTK RESOURCE MANAGER
# ─────────────────────────────────────────

def ensure_nltk_resources():
    """
    Download required NLTK data only when explicitly called.
    Never triggers on module import — production best practice.
    """
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet',   quiet=True)
    nltk.download('omw-1.4',   quiet=True)


# ─────────────────────────────────────────
# Initialize Lemmatizer
# ─────────────────────────────────────────
lemmatizer = WordNetLemmatizer()


# ─────────────────────────────────────────
# LAZY STOPWORDS LOADER
# ─────────────────────────────────────────
_STOP_WORDS = None  # not loaded until first use


def load_custom_stopwords() -> set:
    """
    Load custom stopwords — downloads NLTK data if needed.
    Merges NLTK english stopwords + custom stopwords.txt
    """
    # Always ensure NLTK data is available before loading
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet',   quiet=True)
    nltk.download('omw-1.4',   quiet=True)

    base_stopwords = set(stopwords.words('english'))

    try:
        with open(STOPWORDS_PATH, 'r') as f:
            custom = set(line.strip().lower() for line in f if line.strip())
        return base_stopwords.union(custom)
    except FileNotFoundError:
        # No custom stopwords file — use NLTK only
        return base_stopwords


def get_stop_words() -> set:
    """
    Returns stopwords set using lazy loading.
    Downloads and loads only on first call.
    """
    global _STOP_WORDS
    if _STOP_WORDS is None:
        _STOP_WORDS = load_custom_stopwords()
    return _STOP_WORDS


# ─────────────────────────────────────────
# INDIVIDUAL CLEANING FUNCTIONS
# ─────────────────────────────────────────

def remove_urls(text: str) -> str:
    """Remove http/https URLs and t.co links."""
    return re.sub(r'http\S+|www\S+|https\S+', '', text)


def remove_mentions(text: str) -> str:
    """Remove @username mentions."""
    return re.sub(r'@\w+', '', text)


def remove_hashtag_symbol(text: str) -> str:
    """Remove # symbol but keep the word."""
    return re.sub(r'#', '', text)


def remove_rt(text: str) -> str:
    """Remove RT (retweet prefix) from tweets."""
    return re.sub(r'\bRT\b', '', text)


def remove_emojis(text: str) -> str:
    """Remove emojis and unicode symbols."""
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"
        u"\u3030"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)


def remove_special_characters(text: str) -> str:
    """Remove punctuation and special characters."""
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_numbers(text: str) -> str:
    """Remove standalone numbers."""
    return re.sub(r'\b\d+\b', '', text)


def remove_extra_spaces(text: str) -> str:
    """Strip and reduce multiple spaces to one."""
    return re.sub(r'\s+', ' ', text).strip()


def remove_stopwords(text: str) -> str:
    """Remove English stopwords using lazy loader."""
    stop_words = get_stop_words()  # lazy load — safe at any call time
    return ' '.join(
        word for word in text.split()
        if word not in stop_words
    )


def lemmatize_text(text: str) -> str:
    """Lemmatize each word to its base form."""
    return ' '.join(
        lemmatizer.lemmatize(word)
        for word in text.split()
    )


def remove_non_ascii(text: str) -> str:
    """Remove non-ASCII characters (handles mixed language tweets)."""
    return text.encode('ascii', errors='ignore').decode('ascii')


# ─────────────────────────────────────────
# MASTER CLEANING PIPELINE
# ─────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Full cleaning pipeline for a single tweet.

    Steps:
        1.  Lowercase
        2.  Remove URLs
        3.  Remove RT prefix
        4.  Remove mentions (@user)
        5.  Remove hashtag symbols
        6.  Remove emojis
        7.  Remove non-ASCII characters
        8.  Remove special characters
        9.  Remove numbers
        10. Remove stopwords
        11. Lemmatize
        12. Remove extra spaces

    Args:
        text: raw tweet string

    Returns:
        str: cleaned tweet string
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = remove_urls(text)
    text = remove_rt(text)
    text = remove_mentions(text)
    text = remove_hashtag_symbol(text)
    text = remove_emojis(text)
    text = remove_non_ascii(text)
    text = remove_special_characters(text)
    text = remove_numbers(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    text = remove_extra_spaces(text)

    return text


# ─────────────────────────────────────────
# CLEAN ENTIRE DATASET
# ─────────────────────────────────────────

def clean_dataset() -> pd.DataFrame:
    """
    Load raw dataset → clean tweets → handle labels
    → save to processed path.

    Works for BOTH:
        - String labels  : 'positive', 'negative', 'neutral'
        - Numeric labels : 0, 1, 2

    Flow:
        data/raw/political_tweets.csv
            ↓ clean text
            ↓ map/validate labels
            ↓
        data/processed/cleaned_data.csv

    Returns:
        pd.DataFrame: cleaned and labeled dataframe
    """
    print("\n" + "="*50)
    print("  CLEANING PIPELINE")
    print("="*50)

    # ── Load raw data ─────────────────────
    print(f"[INFO] Loading: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"[INFO] Loaded: {len(df)} rows")

    # ── Validate required columns ─────────
    for col in [TEXT_COLUMN, SENTIMENT_COLUMN]:
        if col not in df.columns:
            raise ValueError(f"[ERROR] Missing column: '{col}'")

    # ── Drop nulls ────────────────────────
    before = len(df)
    df.dropna(subset=[TEXT_COLUMN, SENTIMENT_COLUMN], inplace=True)
    print(f"[INFO] Dropped {before - len(df)} null rows")

    # ── Handle sentiment labels ───────────
    # FIX: check dtype before applying string operations
    if df[SENTIMENT_COLUMN].dtype == object:
        # String labels → clean → map to numeric
        print("[INFO] Detected string labels → mapping to numeric")

        df[SENTIMENT_COLUMN] = (
            df[SENTIMENT_COLUMN]
            .str.lower()
            .str.strip()
            .str.replace('"', '', regex=False)
            .str.replace("'", '', regex=False)
        )

        # Keep only valid labels
        valid  = list(SENTIMENT_MAP.keys())
        before = len(df)
        df     = df[df[SENTIMENT_COLUMN].isin(valid)]
        print(f"[INFO] Removed {before - len(df)} rows with invalid labels")

        # Map string → numeric
        df['label'] = df[SENTIMENT_COLUMN].map(SENTIMENT_MAP)

    else:
        # Already numeric → use directly
        print("[INFO] Detected numeric labels → using directly")
        df['label'] = df[SENTIMENT_COLUMN]

    # ── Label distribution ────────────────
    print(f"[INFO] Label distribution:")
    print(df[SENTIMENT_COLUMN].value_counts().to_string())

    # ── Clean tweet text ──────────────────
    print(f"[INFO] Cleaning {len(df)} tweets...")
    df['cleaned_tweet'] = df[TEXT_COLUMN].apply(clean_text)

    # ── Drop empty tweets ─────────────────
    before = len(df)
    df     = df[df['cleaned_tweet'].str.strip() != '']
    print(f"[INFO] Removed {before - len(df)} empty tweets after cleaning")

    # ── Drop very short tweets (noise) ────
    before = len(df)
    df     = df[df['cleaned_tweet'].str.len() > 3]
    print(f"[INFO] Removed {before - len(df)} very short tweets (≤3 chars)")

    # ── Final columns ─────────────────────
    cols = [TEXT_COLUMN, 'cleaned_tweet', SENTIMENT_COLUMN, 'label']
    if USER_COLUMN in df.columns:
        cols.append(USER_COLUMN)

    df = df[cols].reset_index(drop=True)

    # ── Save ──────────────────────────────
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"\n[SUCCESS] Saved → {PROCESSED_DATA_PATH}")
    print(f"[SUCCESS] Final dataset: {len(df)} rows")
    print("="*50 + "\n")

    return df


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: ensure NLTK resources before anything
    ensure_nltk_resources()

    # Step 2: test single tweet cleaning
    sample = "RT @NATO: Check https://t.co/abc123 🔥 #Ukraine is fighting back! Biden said so @user 123"
    print("Original :", sample)
    print("Cleaned  :", clean_text(sample))
    print()

    # Step 3: run full dataset cleaning pipeline
    df = clean_dataset()
    print(df.head())