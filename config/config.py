# config/config.py
import os

# ─────────────────────────────────────────
# BASE DIRECTORY
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────
# DATA PATHS
# ─────────────────────────────────────────
EXTERNAL_DATA_DIR   = os.path.join(BASE_DIR, "data", "external")
RAW_DATA_PATH       = os.path.join(BASE_DIR, "data", "raw", "political_tweets.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")

# ─────────────────────────────────────────
# MODEL PATHS
# ─────────────────────────────────────────
SAVED_MODELS_DIR    = os.path.join(BASE_DIR, "models", "saved_models")
LOGISTIC_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "logistic.pkl")
SVM_MODEL_PATH      = os.path.join(SAVED_MODELS_DIR, "svm.pkl")
VECTORIZER_PATH     = os.path.join(SAVED_MODELS_DIR, "vectorizer.pkl")

# ─────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────
STOPWORDS_PATH = os.path.join(BASE_DIR, "preprocessing", "stopwords.txt")

# ─────────────────────────────────────────
# DATASET COLUMN NAMES
# ─────────────────────────────────────────
TEXT_COLUMN      = "text"
SENTIMENT_COLUMN = "sentiment"
USER_COLUMN      = "user"

# ─────────────────────────────────────────
# SENTIMENT LABEL MAPPING
# ─────────────────────────────────────────
SENTIMENT_MAP = {
    "negative" : 0,
    "neutral"  : 1,
    "positive" : 2
}

SENTIMENT_LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# ─────────────────────────────────────────
# SENTIMENT COLORS
# ─────────────────────────────────────────
SENTIMENT_COLORS = {
    "Positive" : "#2ecc71",
    "Neutral"  : "#f39c12",
    "Negative" : "#e74c3c"
}

# ─────────────────────────────────────────
# POLITICAL ENTITIES
# ─────────────────────────────────────────
POLITICAL_LEADERS = [
    "Putin",
    "NATO",
    "Zelensky",
    "Biden",
    "Ukraine",
    "Russia"
]

# ─────────────────────────────────────────
# MODEL SETTINGS
# ─────────────────────────────────────────
TFIDF_MAX_FEATURES = 5000
TEST_SIZE          = 0.2
RANDOM_STATE       = 42

# ─────────────────────────────────────────
# TWITTER API KEYS
# ─────────────────────────────────────────
TWITTER_API_KEY             = "YOUR_API_KEY"
TWITTER_API_SECRET          = "YOUR_API_SECRET"
TWITTER_ACCESS_TOKEN        = "YOUR_ACCESS_TOKEN"
TWITTER_ACCESS_TOKEN_SECRET = "YOUR_ACCESS_TOKEN_SECRET"
TWITTER_BEARER_TOKEN        = "YOUR_BEARER_TOKEN"

# ─────────────────────────────────────────
# TWITTER FETCH SETTINGS
# ─────────────────────────────────────────
DEFAULT_TWEET_COUNT = 50
MAX_TWEET_COUNT     = 200

# ─────────────────────────────────────────
# VISUALIZATION OUTPUT PATHS
# ─────────────────────────────────────────
PLOTS_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "plots")

# ─────────────────────────────────────────
# APP SETTINGS
# ─────────────────────────────────────────
APP_TITLE  = "Political Sentiment Intelligence Dashboard"
APP_ICON   = "🏛️"
APP_LAYOUT = "wide"