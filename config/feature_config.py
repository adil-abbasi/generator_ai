"""
Feature Configuration

This file contains all feature definitions used by the
Feature Builder during inference.

The goal is to generate exactly the same features that
were used during model training.
"""

# ==========================================================
# Raw Telemetry Features
# (Received directly from simulator)
# ==========================================================

RAW_FEATURES = [

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

# ==========================================================
# Lag Features
# ==========================================================

LAG_FEATURES = [

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

# ==========================================================
# Rolling Average Features
# ==========================================================

ROLLING_FEATURES = [

    "fuel_level_l",

    "load_pct",

    "rpm",

    "coolant_temp_c",
]

# ==========================================================
# Difference Features
# ==========================================================

DIFF_FEATURES = [

    "fuel_level_l",

    "load_pct",

    "rpm",

    "coolant_temp_c",
]