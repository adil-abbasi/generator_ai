from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent
DROP_COLUMNS = [
    "timestamp"
]

TRAIN_PATH = BASE_DIR / "data" / "prepared" / "train.csv"
VALID_PATH = BASE_DIR / "data" / "prepared" / "validation.csv"
TEST_PATH = BASE_DIR / "data" / "prepared" / "test.csv"
# Model Folder
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
# CatBoost Parameters
CATBOOST_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 8,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": 100
}