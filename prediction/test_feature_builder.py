from prediction.feature_builder import FeatureBuilder

builder = FeatureBuilder()

# ==========================================================
# Packet 1 (10:00)
# ==========================================================

packet1 = {
    "generator_id": "GEN001",
    "site_name": "Site_A",
    "capacity_kva": 500,
    "status": "Running",

    "fuel_level_l": 850,
    "fuel_rate_lph": 18.0,

    "load_pct": 60,
    "load_kw": 300,

    "rpm": 1500,

    "voltage": 400,
    "current": 115,
    "frequency": 50.0,
    "power_factor": 0.92,

    "oil_pressure_bar": 4.2,
    "coolant_temp_c": 81,
    "ambient_temp_c": 34,
    "battery_voltage": 26.8,

    "running_hours": 1450,

    "timestamp": "2026-08-01 10:00:00"
}

# ==========================================================
# Packet 2 (10:05)
# ==========================================================

packet2 = {
    **packet1,

    "fuel_level_l": 848,
    "fuel_rate_lph": 18.1,

    "load_pct": 62,
    "load_kw": 310,

    "rpm": 1498,

    "current": 117,

    "oil_pressure_bar": 4.1,
    "coolant_temp_c": 82,
    "battery_voltage": 26.7,

    "running_hours": 1450.08,

    "timestamp": "2026-08-01 10:05:00"
}

# ==========================================================
# Packet 3 (10:10)
# ==========================================================

packet3 = {
    **packet2,

    "fuel_level_l": 846,
    "fuel_rate_lph": 18.3,

    "load_pct": 64,
    "load_kw": 320,

    "rpm": 1499,

    "current": 118,

    "oil_pressure_bar": 4.0,
    "coolant_temp_c": 83,
    "battery_voltage": 26.6,

    "running_hours": 1450.16,

    "timestamp": "2026-08-01 10:10:00"
}

# ==========================================================
# Packet 4 (10:15)
# ==========================================================

packet4 = {
    **packet3,

    "fuel_level_l": 844,
    "fuel_rate_lph": 18.4,

    "load_pct": 65,
    "load_kw": 325,

    "rpm": 1497,

    "current": 119,

    "oil_pressure_bar": 3.9,
    "coolant_temp_c": 84,
    "battery_voltage": 26.5,

    "running_hours": 1450.25,

    "timestamp": "2026-08-01 10:15:00"
}

# ==========================================================
# Feed packets
# ==========================================================

builder.add_packet(packet1)
builder.add_packet(packet2)
builder.add_packet(packet3)
builder.add_packet(packet4)

features = builder.build_features()

print("=" * 70)
print("FEATURES")
print("=" * 70)

for key in sorted(features.keys()):
    print(f"{key:35} {features[key]}")