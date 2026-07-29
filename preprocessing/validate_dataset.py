"""
Dataset Validation Script
-------------------------
Validates generator telemetry before ML training.

Checks:
1. Dataset schema
2. Missing values
3. Duplicate rows
4. Physical limits
5. Physics consistency
6. Time-series consistency
7. Correlation summary

Author: Adil Abbasi
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


class DatasetValidator:

    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.df = None

        self.report = {
            "rows": 0,
            "columns": 0,
            "generators": 0,
            "missing_values": {},
            "duplicate_rows": 0,
            "duplicate_timestamp_generator": 0,
            "physics_violations": [],
            "range_violations": [],
            "time_series_violations": [],
            "overall_status": "PASS"
        }

    # -------------------------------------------------------
    # Load Dataset
    # -------------------------------------------------------

    def load(self):

        print("Loading dataset...")

        self.df = pd.read_csv(self.csv_path)

        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        self.report["rows"] = len(self.df)
        self.report["columns"] = len(self.df.columns)
        self.report["generators"] = self.df["generator_id"].nunique()

        print(f"Loaded {len(self.df):,} rows")

    # -------------------------------------------------------
    # Schema
    # -------------------------------------------------------

    def validate_schema(self):

        required_columns = [

            "timestamp",
            "generator_id",
            "site_name",
            "capacity_kva",
            "status",
            "fuel_level_l",
            "fuel_rate_lph",
            "load_pct",
            "load_kw",
            "rpm",
            "voltage",
            "current",
            "frequency",
            "power_factor",
            "oil_pressure_bar",
            "coolant_temp_c",
            "ambient_temp_c",
            "battery_voltage",
            "running_hours"

        ]

        missing = []

        for col in required_columns:
            if col not in self.df.columns:
                missing.append(col)

        if missing:
            self.report["overall_status"] = "FAIL"
            raise ValueError(f"Missing columns: {missing}")

    # -------------------------------------------------------
    # Missing Values
    # -------------------------------------------------------

    def validate_missing(self):

        missing = self.df.isna().sum()

        self.report["missing_values"] = missing.to_dict()

    # -------------------------------------------------------
    # Duplicate Rows
    # -------------------------------------------------------

    def validate_duplicates(self):

        self.report["duplicate_rows"] = int(self.df.duplicated().sum())

        dup = self.df.duplicated(
            subset=["generator_id", "timestamp"]
        ).sum()

        self.report["duplicate_timestamp_generator"] = int(dup)

    # -------------------------------------------------------
    # Physical Ranges
    # -------------------------------------------------------

    def validate_ranges(self):

        checks = {

            "fuel_level_l":
                (self.df["fuel_level_l"] < 0),

            "fuel_rate_lph":
                (self.df["fuel_rate_lph"] < 0),

            "load_pct":
                (
                    (self.df["load_pct"] < 0) |
                    (self.df["load_pct"] > 100)
                ),

            "power_factor":
                (
                    (self.df["power_factor"] < 0) |
                    (self.df["power_factor"] > 1)
                ),

            "frequency":
                (
                    (self.df["status"] == "RUNNING") &
                    (
                        (self.df["frequency"] < 48) |
                        (self.df["frequency"] > 52)
                    )
                ),

            "battery_voltage":
                (
                    (self.df["battery_voltage"] < 22) |
                    (self.df["battery_voltage"] > 30)
                )

        }

        for column, condition in checks.items():

            count = int(condition.sum())

            if count > 0:

                self.report["range_violations"].append({

                    "column": column,
                    "count": count

                })

    # -------------------------------------------------------
    # Physics Rules
    # -------------------------------------------------------

    def validate_physics(self):

        off = self.df["status"] == "OFF"

        rules = [

            ("RPM while OFF", off & (self.df["rpm"] != 0)),

            ("Voltage while OFF", off & (self.df["voltage"] != 0)),

            ("Current while OFF", off & (self.df["current"] != 0)),

            ("Frequency while OFF", off & (self.df["frequency"] != 0)),

            ("Fuel rate while OFF", off & (self.df["fuel_rate_lph"] != 0)),

            ("Oil pressure while OFF", off & (self.df["oil_pressure_bar"] != 0)),

            ("Power factor while OFF", off & (self.df["power_factor"] != 0))

        ]

        for name, mask in rules:

            count = int(mask.sum())

            if count > 0:

                self.report["physics_violations"].append({

                    "rule": name,
                    "count": count

                })

    # -------------------------------------------------------
    # Time-Series Checks
    # -------------------------------------------------------

    def validate_timeseries(self):

        grouped = self.df.sort_values("timestamp").groupby("generator_id")

        for gid, data in grouped:

            if not data["timestamp"].is_monotonic_increasing:

                self.report["time_series_violations"].append(

                    f"{gid}: timestamps not ordered"

                )

            diff = data["running_hours"].diff()

            bad = (diff < 0).sum()

            if bad > 0:

                self.report["time_series_violations"].append(

                    f"{gid}: running hours decreased"

                )

    # -------------------------------------------------------
    # Correlations
    # -------------------------------------------------------

    def correlation_summary(self):

        cols = [

            "load_pct",
            "load_kw",
            "fuel_rate_lph",
            "current",
            "rpm",
            "frequency",
            "coolant_temp_c"

        ]

        corr = self.df[cols].corr()

        print("\nCorrelation Matrix\n")

        print(corr.round(3))

    # -------------------------------------------------------
    # Save Report
    # -------------------------------------------------------

    def save_report(self):

        Path("reports").mkdir(exist_ok=True)

        with open("reports/validation_report.json", "w") as f:
            json.dump(self.report, f, indent=4)

        print("\nValidation report saved.")

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------

    def summary(self):

        print("\n========== DATASET SUMMARY ==========")

        print(f"Rows                : {self.report['rows']:,}")
        print(f"Columns             : {self.report['columns']}")
        print(f"Generators          : {self.report['generators']}")
        print(f"Duplicate Rows      : {self.report['duplicate_rows']}")
        print(f"Physics Violations  : {len(self.report['physics_violations'])}")
        print(f"Range Violations    : {len(self.report['range_violations'])}")
        print(f"Time Violations     : {len(self.report['time_series_violations'])}")

        print("\nDataset Status:", self.report["overall_status"])

        print("====================================")


if __name__ == "__main__":

    validator = DatasetValidator(
        "../data/generator_clean.csv"
    )

    validator.load()

    validator.validate_schema()

    validator.validate_missing()

    validator.validate_duplicates()

    validator.validate_ranges()

    validator.validate_physics()

    validator.validate_timeseries()

    validator.correlation_summary()

    validator.save_report()

    validator.summary()