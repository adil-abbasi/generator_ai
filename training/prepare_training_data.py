import pandas as pd
from pathlib import Path

print("=" * 60)
print("Preparing Forecast Training Dataset")
print("=" * 60)

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "generator_features.csv"

OUTPUT_DIR = BASE_DIR / "data" / "prepared"

OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

print("\nLoading feature engineered dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Rows Loaded : {len(df):,}")

# --------------------------------------------------
# Sort Dataset
# --------------------------------------------------

print("\nSorting dataset...")

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values(
    ["generator_id", "timestamp"]
).reset_index(drop=True)

# --------------------------------------------------
# Forecast Targets
# --------------------------------------------------

TARGET_COLUMNS = [
    "fuel_level_l",
    "fuel_rate_lph",
    "load_pct",
    "load_kw",
    "rpm",
    "current",
    "frequency",
    "oil_pressure_bar",
    "coolant_temp_c",
    "battery_voltage",
]

print("\nCreating future target columns...")

for col in TARGET_COLUMNS:
    df[f"target_{col}"] = (
        df.groupby("generator_id")[col]
          .shift(-1)
    )

# --------------------------------------------------
# Remove Last Record of Every Generator
# --------------------------------------------------

before = len(df)

df = df.dropna().reset_index(drop=True)

after = len(df)

print(f"Removed {before-after} rows")

# --------------------------------------------------
# Train / Validation / Test Split
# --------------------------------------------------

print("\nSplitting dataset...")

train_parts = []
val_parts = []
test_parts = []

for gen in df["generator_id"].unique():

    gen_df = df[df["generator_id"] == gen]

    n = len(gen_df)

    train_end = int(n * 0.70)

    val_end = int(n * 0.85)

    train_parts.append(gen_df.iloc[:train_end])

    val_parts.append(gen_df.iloc[train_end:val_end])

    test_parts.append(gen_df.iloc[val_end:])

train_df = pd.concat(train_parts)

val_df = pd.concat(val_parts)

test_df = pd.concat(test_parts)

# --------------------------------------------------
# Save
# --------------------------------------------------

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

val_df.to_csv(
    OUTPUT_DIR / "validation.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)

print("\nSaved Files")

print(f"Train      : {len(train_df):,}")
print(f"Validation : {len(val_df):,}")
print(f"Test       : {len(test_df):,}")

print("\nDone.")