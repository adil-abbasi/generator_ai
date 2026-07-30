from pathlib import Path

# ==========================================================
# Project Root
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================================================
# Dataset Paths
# ==========================================================
DATA_DIR = BASE_DIR / "data" / "prepared"

TRAIN_PATH = DATA_DIR / "train.csv"
VALID_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"

# ==========================================================
# Columns to Exclude
# ==========================================================
DROP_COLUMNS = [
    "timestamp"
]

# ==========================================================
# Models Directory
# ==========================================================
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ==========================================================
# Models directory
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Metrics directory
METRICS_DIR = BASE_DIR / "metrics"
METRICS_DIR.mkdir(exist_ok=True)
# ==========================================================
# CatBoost Hyperparameters
# ==========================================================
CATBOOST_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 8,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": 100
}