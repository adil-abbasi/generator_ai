import pandas as pd
import math
from catboost import CatBoostRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import os
print("=" * 60)
print("Fuel Forecast Model Training")
print("=" * 60)

print("\nLoading datasets...")

train_df = pd.read_csv("./data/prepared/train.csv")
valid_df = pd.read_csv("./data/prepared/validation.csv")
test_df = pd.read_csv("./data/prepared/test.csv")

print("Training Rows   :", len(train_df))
print("Validation Rows :", len(valid_df))
print("Test Rows       :", len(test_df))

print("\nDataset Columns")
print("-" * 60)

for column in train_df.columns:
    print(column)

print("\nPreparing Features and Target...")

# Columns that should NOT be used as input features
drop_columns = [
    "timestamp",
    "site_name",
    "status",

    "target_fuel_level_l",
    "target_fuel_rate_lph",
    "target_load_pct",
    "target_load_kw",
    "target_rpm",
    "target_current",
    "target_frequency",
    "target_oil_pressure_bar",
    "target_coolant_temp_c",
    "target_battery_voltage"
]

# Training Features
X_train = train_df.drop(columns=drop_columns)

# Training Target
y_train = train_df["target_fuel_level_l"]

# Validation Features
X_valid = valid_df.drop(columns=drop_columns)

# Validation Target
y_valid = valid_df["target_fuel_level_l"]

# Test Features
X_test = test_df.drop(columns=drop_columns)

# Test Target
y_test = test_df["target_fuel_level_l"]

print("\nDataset Shapes")
print("-" * 50)

print("X_train :", X_train.shape)
print("y_train :", y_train.shape)

print("X_valid :", X_valid.shape)
print("y_valid :", y_valid.shape)

print("X_test  :", X_test.shape)
print("y_test  :", y_test.shape)

print("\nCreating CatBoost Model...")

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,

    loss_function="RMSE",
    eval_metric="RMSE",

    random_seed=42,

    early_stopping_rounds=100,

    verbose=100
)

print("\nStarting Training...")
print("=" * 60)

model.fit(
    X_train,
    y_train,

    eval_set=(X_valid, y_valid),

    cat_features=["generator_id"]
)

print("\nTraining Completed!")

print("\nMaking Predictions on Test Dataset...")

y_pred = model.predict(X_test)

print("\nCalculating Evaluation Metrics...")

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

print("\nEvaluation Results")
print("=" * 60)

print(f"MAE  : {mae:.3f}")

print(f"RMSE : {rmse:.3f}")

print(f"R²   : {r2:.4f}")

import os

save_dir = os.path.abspath("../models")
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "fuel_model.cbm")

print("Saving model to:")
print(save_path)

model.save_model(save_path)

print("Model saved successfully!")