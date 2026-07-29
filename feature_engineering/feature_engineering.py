import pandas as pd
import numpy as np

print("Loading dataset...")

df = pd.read_csv("../data/generator_clean.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

# --------------------------------------------------
# Sort correctly
# --------------------------------------------------

df = df.sort_values(
    ["generator_id", "timestamp"]
).reset_index(drop=True)

print("Dataset Loaded:", len(df))

df["hour"] = df["timestamp"].dt.hour

df["minute"] = df["timestamp"].dt.minute

df["day"] = df["timestamp"].dt.day

df["weekday"] = df["timestamp"].dt.weekday

df["month"] = df["timestamp"].dt.month

df["is_weekend"] = (
    df["weekday"] >= 5
).astype(int)

status_map = {
    "OFF":0,
    "STARTING":1,
    "WARMUP":2,
    "RUNNING":3,
    "COOLDOWN":4
}

df["status_code"] = df["status"].map(status_map)

lag_columns = [
    "fuel_level_l",
    "fuel_rate_lph",
    "load_pct",
    "load_kw",
    "rpm",
    "current",
    "frequency",
    "oil_pressure_bar",
    "coolant_temp_c",
    "battery_voltage"
]

for col in lag_columns:

    df[f"{col}_lag1"] = (
        df.groupby("generator_id")[col]
        .shift(1)
    )

    df[f"{col}_lag2"] = (
        df.groupby("generator_id")[col]
        .shift(2)
    )

    df[f"{col}_lag3"] = (
        df.groupby("generator_id")[col]
        .shift(3)
    )

    rolling_columns = [
    "fuel_level_l",
    "load_pct",
    "rpm",
    "coolant_temp_c"
]

for col in rolling_columns:

    df[f"{col}_rolling3"] = (
        df.groupby("generator_id")[col]
        .rolling(3)
        .mean()
        .reset_index(level=0,drop=True)
    )

    difference_columns = [
    "fuel_level_l",
    "load_pct",
    "rpm",
    "coolant_temp_c"
]

for col in difference_columns:

    df[f"{col}_diff"] = (
        df.groupby("generator_id")[col]
        .diff()
    )       

    df["fuel_percentage"] = (
    df["fuel_level_l"] /
    (df["capacity_kva"] * 1.5)
) * 100

    df["cooling_margin"] = (
    df["coolant_temp_c"]
    -
    df["ambient_temp_c"]
)
    df = df.bfill()

    output = "../data/generator_features.csv"

df.to_csv(output,index=False)

print("Saved:",output)

print(df.head())