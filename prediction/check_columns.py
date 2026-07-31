"""
Feature Configuration

This file defines how inference features
should be generated.

Changing this file is enough to update the
entire Feature Builder.
"""

# ==========================================================
# Raw telemetry received from simulator
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

    "battery_voltage"
]

# ==========================================================
# Features requiring Lag1 Lag2 Lag3
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

    "battery_voltage"
]

# ==========================================================
# Features requiring Rolling Average
# ==========================================================

ROLLING_FEATURES = [

    "fuel_level_l",

    "load_pct",

    "rpm",

    "coolant_temp_c"
]

# ==========================================================
# Features requiring Difference
# ==========================================================

DIFF_FEATURES = [

    "fuel_level_l",

    "load_pct",

    "rpm",

    "coolant_temp_c"
]