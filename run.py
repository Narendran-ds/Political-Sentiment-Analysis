# run.py
# Main execution controller
# Checks all dependencies → trains if needed → launches Streamlit
# Run this file to start the entire project:
#   python run.py

import os
import sys
import subprocess

# ── Must be first — adds project root to path ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    LOGISTIC_MODEL_PATH,
    SVM_MODEL_PATH,
    VECTORIZER_PATH,
    SAVED_MODELS_DIR,
    APP_TITLE
)


# ─────────────────────────────────────────
# DISPLAY BANNER
# ─────────────────────────────────────────

def print_banner():
    """Print project banner on startup."""
    print("\n" + "="*60)
    print("  🏛️  POLITICAL SENTIMENT INTELLIGENCE DASHBOARD")
    print("="*60)
    print(f"  {APP_TITLE}")
    print("="*60)
    print("  Initializing system...\n")


# ─────────────────────────────────────────
# CHECK STEPS
# ─────────────────────────────────────────

def check_raw_data() -> bool:
    """Check if merged political_tweets.csv exists."""
    return os.path.exists(RAW_DATA_PATH)


def check_processed_data() -> bool:
    """Check if cleaned_data.csv exists."""
    return os.path.exists(PROCESSED_DATA_PATH)


def check_models() -> bool:
    """Check if all 3 model files exist."""
    paths = [LOGISTIC_MODEL_PATH, SVM_MODEL_PATH, VECTORIZER_PATH]
    return all(os.path.exists(p) for p in paths)


# ─────────────────────────────────────────
# PIPELINE STEPS
# ─────────────────────────────────────────

def step_merge_data():
    """Step 1 — Merge all CSVs into political_tweets.csv"""
    print("[STEP 1] Merging CSV files...")
    try:
        from utils.helpers import merge_csv_files, ensure_directories
        ensure_directories()
        merge_csv_files()
        print("[STEP 1] ✅ Data merged successfully!\n")
    except Exception as e:
        print(f"[STEP 1] ❌ Failed to merge data: {e}")
        print("[INFO]   Make sure Set-1/ and Set-2/ are in data/external/")
        sys.exit(1)


def step_clean_data():
    """Step 2 — Clean and preprocess tweets"""
    print("[STEP 2] Cleaning tweet data...")
    try:
        from preprocessing.clean_text import (
            ensure_nltk_resources,
            clean_dataset
        )
        ensure_nltk_resources()
        clean_dataset()
        print("[STEP 2] ✅ Data cleaned successfully!\n")
    except Exception as e:
        print(f"[STEP 2] ❌ Failed to clean data: {e}")
        sys.exit(1)


def step_train_models():
    """Step 3 — Feature engineering + Train models"""
    print("[STEP 3] Training ML models...")
    try:
        from preprocessing.feature_engineering import prepare_features
        from models.train_model import train_all_models
        from models.evaluate_model import compare_models

        # Feature engineering
        X_train, X_test, y_train, y_test, vectorizer = prepare_features()

        # Train both models
        models = train_all_models(X_train, y_train)

        # Evaluate and compare
        model_display = {
            'Logistic Regression' : models['logistic'],
            'SVM'                 : models['svm']
        }
        df_results = compare_models(model_display, X_test, y_test)

        print("[STEP 3] ✅ Models trained and evaluated!\n")
        return df_results

    except Exception as e:
        print(f"[STEP 3] ❌ Failed to train models: {e}")
        sys.exit(1)


# ─────────────────────────────────────────
# SYSTEM STATUS CHECK
# ─────────────────────────────────────────

def print_system_status():
    """Print current status of all pipeline components."""
    print("\n" + "─"*60)
    print("  SYSTEM STATUS CHECK")
    print("─"*60)

    checks = {
        "Raw data (political_tweets.csv)" : check_raw_data(),
        "Processed data (cleaned_data.csv)": check_processed_data(),
        "Vectorizer (vectorizer.pkl)"      : os.path.exists(VECTORIZER_PATH),
        "Logistic model (logistic.pkl)"    : os.path.exists(LOGISTIC_MODEL_PATH),
        "SVM model (svm.pkl)"              : os.path.exists(SVM_MODEL_PATH),
    }

    all_ok = True
    for name, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"  {icon}  {name}")
        if not status:
            all_ok = False

    print("─"*60)
    return all_ok


# ─────────────────────────────────────────
# LAUNCH STREAMLIT
# ─────────────────────────────────────────

def launch_streamlit():
    """Launch the Streamlit dashboard."""
    app_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "app.py"
    )

    if not os.path.exists(app_path):
        print("[ERROR] app.py not found!")
        print("[INFO]  Make sure app.py exists in project root.")
        sys.exit(1)

    print("\n" + "="*60)
    print("  🚀 LAUNCHING STREAMLIT DASHBOARD")
    print("="*60)
    print("  Open your browser at: http://localhost:8501")
    print("  Press Ctrl+C to stop the server")
    print("="*60 + "\n")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        app_path,
        "--server.port", "8501",
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false"
    ])


# ─────────────────────────────────────────
# MAIN CONTROLLER
# ─────────────────────────────────────────

def main():
    """
    Main execution controller.

    Flow:
        1. Print banner
        2. Check system status
        3. Merge CSVs if needed
        4. Clean data if needed
        5. Train models if needed
        6. Launch Streamlit
    """
    # ── Banner ────────────────────────────
    print_banner()

    # ── Status check ──────────────────────
    print_system_status()

    # ── Step 1: Raw data ──────────────────
    if not check_raw_data():
        print("\n[INFO] Raw data missing → merging CSVs...")
        step_merge_data()
    else:
        print("\n[INFO] ✅ Raw data found → skipping merge")

    # ── Step 2: Processed data ────────────
    if not check_processed_data():
        print("[INFO] Processed data missing → cleaning...")
        step_clean_data()
    else:
        print("[INFO] ✅ Processed data found → skipping cleaning")

    # ── Step 3: Models ────────────────────
    if not check_models():
        print("[INFO] Models missing → training now...")
        step_train_models()
    else:
        print("[INFO] ✅ All models found → skipping training")

    # ── Final status ──────────────────────
    print("\n")
    all_ready = print_system_status()

    if not all_ready:
        print("\n[ERROR] System not ready — check errors above!")
        sys.exit(1)

    print("\n[SUCCESS] All systems ready!")

    # ── Launch app ────────────────────────
    launch_streamlit()


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    main()