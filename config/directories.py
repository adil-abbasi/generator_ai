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
# Models Directory
# ==========================================================
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ==========================================================
# Metrics Directory
# ==========================================================
METRICS_DIR = BASE_DIR / "metrics"
METRICS_DIR.mkdir(exist_ok=True)

# ==========================================================
# Feature Importance Directory
# ==========================================================
FEATURE_IMPORTANCE_DIR = BASE_DIR / "feature_importance"
FEATURE_IMPORTANCE_DIR.mkdir(exist_ok=True)