from pathlib import Path

# -----------------------------
# Input Parameters
# -----------------------------

RAW_DIR = Path.cwd() / 'data' / 'raw'
DATASET_DIR = Path.cwd() / "data" / "processed"

# Pickles
PICKLE_BLOCK = DATASET_DIR / "block_daily_means.pkl"
PICKLE_HOUSE = DATASET_DIR / "house_daily_means.pkl"

TARGET = "energy_mean"
LAGS = [1, 30, 90]
FEATURES = ["DOY"]   #, "day_of_week", "is_holiday"]  # Base features only, lag features added automatically

SETUPS = [(1, 1, 0.2), (30, 30, 0.2), (90, 90, 0.5)]  # (window_size, horizon, validation_ratio)

OUTMODELS_DIR = Path.cwd() / "outmodels"
