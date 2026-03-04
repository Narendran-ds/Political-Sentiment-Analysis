# twitter/fetch_tweets.py
# Twitter API integration using Tweepy
# Fetches live tweets by keyword for real-time sentiment analysis

import os
import sys

# ── Path fix ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
# ──────────────────────────────────────────────────────

import pandas as pd
import tweepy
from datetime import datetime

from config.config import (
    TWITTER_API_KEY,
    TWITTER_API_SECRET,
    TWITTER_ACCESS_TOKEN,
    TWITTER_ACCESS_TOKEN_SECRET,
    TWITTER_BEARER_TOKEN,
    DEFAULT_TWEET_COUNT,
    MAX_TWEET_COUNT,
    TEXT_COLUMN
)


# ─────────────────────────────────────────
# CHECK API KEYS
# ─────────────────────────────────────────

def api_keys_configured() -> bool:
    """
    Check if Twitter API keys are properly configured.
    Returns False if any key is still placeholder.

    Returns:
        bool: True if all keys look real
    """
    placeholders = [
        "YOUR_API_KEY",
        "YOUR_API_SECRET",
        "YOUR_ACCESS_TOKEN",
        "YOUR_ACCESS_TOKEN_SECRET",
        "YOUR_BEARER_TOKEN"
    ]

    keys = [
        TWITTER_API_KEY,
        TWITTER_API_SECRET,
        TWITTER_ACCESS_TOKEN,
        TWITTER_ACCESS_TOKEN_SECRET,
        TWITTER_BEARER_TOKEN
    ]

    for key in keys:
        if key in placeholders or not key or key.strip() == "":
            return False

    return True


# ─────────────────────────────────────────
# BUILD TWEEPY CLIENT
# ─────────────────────────────────────────

def get_twitter_client() -> tweepy.Client:
    """
    Build and return authenticated Tweepy v2 Client.

    Uses Bearer Token for app-only authentication.
    This allows searching recent tweets without
    user-level OAuth.

    Returns:
        tweepy.Client: authenticated client

    Raises:
        ValueError  : if API keys not configured
        tweepy.errors.TweepyException : if auth fails
    """
    if not api_keys_configured():
        raise ValueError(
            "[ERROR] Twitter API keys not configured!\n"
            "Update config/config.py with your real API keys.\n"
            "Get keys at: https://developer.twitter.com"
        )

    client = tweepy.Client(
        bearer_token        = TWITTER_BEARER_TOKEN,
        consumer_key        = TWITTER_API_KEY,
        consumer_secret     = TWITTER_API_SECRET,
        access_token        = TWITTER_ACCESS_TOKEN,
        access_token_secret = TWITTER_ACCESS_TOKEN_SECRET,
        wait_on_rate_limit  = True   # auto-wait on rate limits
    )

    print("[INFO] Twitter client authenticated ✅")
    return client


# ─────────────────────────────────────────
# FETCH TWEETS BY KEYWORD
# ─────────────────────────────────────────

def fetch_tweets(
    keyword     : str,
    count       : int  = DEFAULT_TWEET_COUNT,
    lang        : str  = 'en',
    exclude_rts : bool = True
) -> pd.DataFrame:
    """
    Fetch recent tweets matching a keyword.

    Uses Twitter API v2 recent search endpoint.
    Requires Bearer Token (Free tier supported).

    Args:
        keyword     : search term e.g. 'Ukraine NATO'
        count       : number of tweets to fetch
                      capped at MAX_TWEET_COUNT
        lang        : language filter (default: 'en')
        exclude_rts : if True excludes retweets

    Returns:
        pd.DataFrame with columns:
            - text       : raw tweet text
            - user       : username
            - created_at : tweet timestamp
            - tweet_id   : unique tweet ID
            - source     : always 'twitter_api'

        Returns empty DataFrame if API unavailable.
    """
    # ── Validate inputs ───────────────────
    if not keyword or not keyword.strip():
        raise ValueError("[ERROR] Keyword cannot be empty!")

    count = min(max(1, count), MAX_TWEET_COUNT)

    # ── Check API keys ────────────────────
    if not api_keys_configured():
        print("[WARNING] Twitter API keys not configured.")
        print("[INFO]    Returning sample data for demonstration.")
        return _get_sample_tweets(keyword, count)

    try:
        # ── Build client ──────────────────
        client = get_twitter_client()

        # ── Build query ───────────────────
        query_parts = [keyword, f'lang:{lang}']
        if exclude_rts:
            query_parts.append('-is:retweet')

        query = ' '.join(query_parts)
        print(f"[INFO] Fetching tweets for: '{query}'")
        print(f"[INFO] Count: {count}")

        # ── Fetch tweets ──────────────────
        response = client.search_recent_tweets(
            query        = query,
            max_results  = min(count, 100),  # API max per page
            tweet_fields = ['created_at', 'author_id', 'text'],
            expansions   = ['author_id'],
            user_fields  = ['username']
        )

        if not response.data:
            print(f"[WARNING] No tweets found for: '{keyword}'")
            return pd.DataFrame()

        # ── Build user map ────────────────
        user_map = {}
        if response.includes and 'users' in response.includes:
            for user in response.includes['users']:
                user_map[user.id] = user.username

        # ── Build DataFrame ───────────────
        records = []
        for tweet in response.data:
            records.append({
                TEXT_COLUMN   : tweet.text,
                'user'        : user_map.get(tweet.author_id, 'unknown'),
                'created_at'  : tweet.created_at,
                'tweet_id'    : tweet.id,
                'source'      : 'twitter_api'
            })

        df = pd.DataFrame(records)
        print(f"[SUCCESS] Fetched {len(df)} tweets for '{keyword}'")

        return df

    except tweepy.errors.TweepyException as e:
        print(f"[ERROR] Twitter API error: {e}")
        print("[INFO]  Returning sample data instead.")
        return _get_sample_tweets(keyword, count)

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────
# FETCH MULTIPLE KEYWORDS
# ─────────────────────────────────────────

def fetch_multiple_keywords(
    keywords : list,
    count    : int = DEFAULT_TWEET_COUNT
) -> pd.DataFrame:
    """
    Fetch tweets for multiple keywords and combine.

    Args:
        keywords : list of search terms
        count    : tweets per keyword

    Returns:
        pd.DataFrame: combined results with 'keyword' column
    """
    if not keywords:
        raise ValueError("[ERROR] Keywords list is empty!")

    all_dfs = []

    for keyword in keywords:
        print(f"\n[INFO] Fetching: '{keyword}'...")
        df = fetch_tweets(keyword, count)

        if not df.empty:
            df['keyword'] = keyword
            all_dfs.append(df)

    if not all_dfs:
        print("[WARNING] No tweets fetched for any keyword.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.drop_duplicates(subset=[TEXT_COLUMN], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    print(f"\n[SUCCESS] Total tweets fetched: {len(combined)}")
    return combined


# ─────────────────────────────────────────
# SAMPLE TWEETS (fallback when no API key)
# ─────────────────────────────────────────

def _get_sample_tweets(
    keyword : str,
    count   : int = 10
) -> pd.DataFrame:
    """
    Return realistic sample tweets for demonstration.
    Used when Twitter API keys are not configured.
    Allows dashboard to work without real API access.

    Args:
        keyword : search keyword (used in tweet text)
        count   : number of sample tweets

    Returns:
        pd.DataFrame
    """
    sample_templates = [
        f"The situation around {keyword} continues to develop rapidly.",
        f"Breaking: New developments in the {keyword} conflict reported.",
        f"International community responds to {keyword} crisis.",
        f"NATO allies discuss implications of {keyword} situation.",
        f"Civilians affected by ongoing {keyword} conflict need support.",
        f"Peace talks regarding {keyword} show some progress today.",
        f"Military movements near {keyword} region raise concerns.",
        f"Economic sanctions related to {keyword} taking effect.",
        f"Humanitarian aid efforts in {keyword} area continue.",
        f"World leaders meet to address the {keyword} crisis.",
        f"Latest updates on the {keyword} situation from reporters.",
        f"Analysis: What does the {keyword} development mean?",
        f"Protests around the world show support for {keyword}.",
        f"UN Security Council debates response to {keyword}.",
        f"Refugees flee due to {keyword} conflict escalation.",
    ]

    import random
    random.seed(42)

    records = []
    for i in range(min(count, len(sample_templates))):
        records.append({
            TEXT_COLUMN  : sample_templates[i],
            'user'       : f'user_{i+1}',
            'created_at' : datetime.now(),
            'tweet_id'   : f'sample_{i+1}',
            'source'     : 'sample_data'
        })

    df = pd.DataFrame(records)
    print(f"[INFO] Returned {len(df)} sample tweets "
          f"(API keys not configured)")

    return df


# ─────────────────────────────────────────
# VALIDATE FETCHED DATA
# ─────────────────────────────────────────

def validate_tweets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate fetched tweet DataFrame.

    Removes:
        - Empty tweets
        - Very short tweets (< 5 chars)
        - Duplicate tweets

    Args:
        df : raw fetched DataFrame

    Returns:
        pd.DataFrame: cleaned DataFrame
    """
    if df.empty:
        return df

    before = len(df)

    # Drop nulls
    df = df.dropna(subset=[TEXT_COLUMN])

    # Drop very short
    df = df[df[TEXT_COLUMN].str.len() > 5]

    # Drop duplicates
    df = df.drop_duplicates(subset=[TEXT_COLUMN])

    df = df.reset_index(drop=True)

    print(f"[INFO] Validated: {before} → {len(df)} tweets")
    return df


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("="*50)
    print("  TWITTER FETCH TEST")
    print("="*50)

    # ── Check API config ──────────────────
    print(f"\n[INFO] API keys configured: {api_keys_configured()}")

    # ── Test single keyword ───────────────
    print("\n[1/2] Testing single keyword fetch...")
    df = fetch_tweets(
        keyword = "Ukraine NATO",
        count   = 10
    )

    if not df.empty:
        df = validate_tweets(df)
        print(f"\nFetched DataFrame:")
        print(df[[TEXT_COLUMN, 'user', 'source']].head())

    # ── Test multiple keywords ────────────
    print("\n[2/2] Testing multiple keywords...")
    df_multi = fetch_multiple_keywords(
        keywords = ["Putin", "Zelensky"],
        count    = 5
    )

    if not df_multi.empty:
        print(f"\nMulti-keyword results:")
        print(df_multi[[TEXT_COLUMN, 'keyword']].head())

    print("\n[SUCCESS] Twitter fetch test complete!")