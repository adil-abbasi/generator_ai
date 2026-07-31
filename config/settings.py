# ==========================================================
# Columns to Exclude
# ==========================================================
DROP_COLUMNS = [
    "timestamp"
]

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

# ==========================================================
# Categorical Features
# ==========================================================
CATEGORICAL_COLUMNS = [
    "generator_id",
    "site_name",
    "status"
]