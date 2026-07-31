import pandas as pd
import numpy as np
import json

from pathlib import Path
from datetime import datetime

from catboost import CatBoostRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from config import (
    TRAIN_PATH,
    VALID_PATH,
    TEST_PATH,
    MODEL_DIR,
    METRICS_DIR,
    FEATURE_IMPORTANCE_DIR,
    CATBOOST_PARAMS,
    DROP_COLUMNS
)


def train_target(TARGET):
    """
    Train a CatBoost model for one target column.
    """

    print("=" * 60)
    print(f"Training Target : {TARGET}")
    print("=" * 60)

    # ==========================================================
    # Load datasets
    # ==========================================================

    print("\nLoading datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Train Rows      : {len(train_df):,}")
    print(f"Validation Rows : {len(valid_df):,}")
    print(f"Test Rows       : {len(test_df):,}")

    # ==========================================================
    # Identify target columns
    # ==========================================================

    TARGET_COLUMNS = [
        column
        for column in train_df.columns
        if column.startswith("target_")
    ]

    # ==========================================================
    # Feature selection
    # ==========================================================

    FEATURE_COLUMNS = [
        column
        for column in train_df.columns
        if column not in TARGET_COLUMNS + DROP_COLUMNS
    ]

    # ==========================================================
    # Split Features / Labels
    # ==========================================================

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET]

    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df[TARGET]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET]

    print("\nDataset Shapes")
    print("-" * 50)

    print(f"X_train : {X_train.shape}")
    print(f"y_train : {y_train.shape}")

    print(f"X_valid : {X_valid.shape}")
    print(f"y_valid : {y_valid.shape}")

    print(f"X_test  : {X_test.shape}")
    print(f"y_test  : {y_test.shape}")

    # ==========================================================
    # Categorical Features
    # ==========================================================

    print("\nDetecting categorical features...")

    CATEGORICAL_COLUMNS = [
        "generator_id",
        "site_name",
        "status"
    ]

    categorical_features = [
        col
        for col in CATEGORICAL_COLUMNS
        if col in X_train.columns
    ]

    print("\nCategorical Columns")
    print("-" * 50)

    for column in categorical_features:
        print(column)

    # ==========================================================
    # Model
    # ==========================================================

    print("\nCreating CatBoost Model...")

    model = CatBoostRegressor(
        **CATBOOST_PARAMS
    )

    # ==========================================================
    # Train
    # ==========================================================

    print("\nStarting Training...")
    print("=" * 60)

    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        cat_features=categorical_features,
        use_best_model=True
    )

    # ==========================================================
    # Prediction
    # ==========================================================

    print("\nMaking Predictions on Test Dataset...")

    y_pred = model.predict(X_test)

    # ==========================================================
    # Metrics
    # ==========================================================

    print("\nCalculating Evaluation Metrics...")

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    print("\nEvaluation Results")
    print("=" * 60)

    print(f"Target : {TARGET}")
    print(f"MAE    : {mae:.3f}")
    print(f"RMSE   : {rmse:.3f}")
    print(f"R²     : {r2:.4f}")

    # ==========================================================
    # Save Metrics
    # ==========================================================

    metrics = {
        "target": TARGET,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "best_iteration": model.get_best_iteration(),
        "training_rows": len(train_df),
        "validation_rows": len(valid_df),
        "test_rows": len(test_df),
        "training_date": datetime.now().isoformat()
    }

    METRICS_PATH = METRICS_DIR / f"{TARGET}.json"

    with open(METRICS_PATH, "w") as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    print(f"\nMetrics Saved : {METRICS_PATH}")

    # ==========================================================
    # Feature Importance
    # ==========================================================

    print("\nCalculating Feature Importance...")

    importance = model.get_feature_importance()

    importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": importance
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    IMPORTANCE_PATH = (
        FEATURE_IMPORTANCE_DIR /
        f"{TARGET}.csv"
    )

    importance_df.to_csv(
        IMPORTANCE_PATH,
        index=False
    )

    print(f"Feature Importance Saved : {IMPORTANCE_PATH}")

    # ==========================================================
    # Save Model
    # ==========================================================

    MODEL_PATH = MODEL_DIR / f"{TARGET}.cbm"

    print("\nSaving Model...")

    model.save_model(MODEL_PATH)

    print(f"Model Saved : {MODEL_PATH}")

    print("\nTraining Completed Successfully.")

    return {
        "target": TARGET,
        "model_path": MODEL_PATH,
        "metrics_path": METRICS_PATH,
        "feature_importance_path": IMPORTANCE_PATH,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generic Generator Model Trainer"
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Target column to train"
    )

    args = parser.parse_args()

    train_target(args.target)