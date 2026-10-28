
"""
Industrial Edge-AI Predictive Maintenance
Target: Crompton Greaves CG-IPM-15 centrifugal pump motor
Controller: ESP32 student-level prototype

Generates synthetic multi-sensor data for:
1. Voltage
2. Current
3. Temperature
4. Vibration

Outputs:
- cg_ipm15_synthetic_dataset.csv
- fault_label_dictionary.csv

IMPORTANT:
This is synthetic educational data, not measured motor data.
Replace illustrative baseline parameters with verified nameplate
values and measurements before conducting real-world validation.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

RANDOM_SEED = 42
SAMPLES_PER_FAULT = 1000

rng = np.random.default_rng(RANDOM_SEED)

# Illustrative baseline assumptions only — NOT verified CG-IPM-15 data.
BASE_VOLTAGE = 415.0       # V, three-phase line voltage
BASE_CURRENT = 3.0         # A, illustrative operating current
BASE_AMBIENT_TEMP = 30.0   # °C
BASE_STATOR_TEMP = 55.0    # °C, surface-temperature proxy
BASE_BEARING_TEMP = 45.0   # °C, housing-temperature proxy
BASE_VIBRATION = 0.12      # g RMS, illustrative acceleration
BASE_LOAD = 0.70           # 70% illustrative load fraction

# Severity levels:
# 0 = normal
# 1 = mild
# 2 = moderate
# 3 = severe

SEVERITY_NAMES = {
    0: "Normal",
    1: "Mild",
    2: "Moderate",
    3: "Severe"
}

SEVERITY_FACTORS = {
    0: 0.0,
    1: 0.4,
    2: 0.7,
    3: 1.0
}

# 8 fault categories + normal operation = 9 classes.
FAULTS = {
    "Normal": 0,
    "Bearing_Wear": 1,
    "Shaft_Misalignment": 2,
    "Rotor_Impeller_Imbalance": 3,
    "Cavitation": 4,
    "Dry_Run_Low_Suction": 5,
    "Overload": 6,
    "Phase_Voltage_Imbalance": 7,
    "Winding_Overheating": 8
}

# Simplified illustrative fault effects.
# Effects overlap intentionally; they are not diagnostic laws.
FAULT_EFFECTS = {
    "Normal": {
        "current": 1.00, "temperature": 0,
        "vibration": 1.00, "voltage_imbalance": 0.2
    },
    "Bearing_Wear": {
        "current": 1.04, "temperature": 7,
        "vibration": 2.00, "voltage_imbalance": 0.3
    },
    "Shaft_Misalignment": {
        "current": 1.10, "temperature": 6,
        "vibration": 2.10, "voltage_imbalance": 0.4
    },
    "Rotor_Impeller_Imbalance": {
        "current": 1.08, "temperature": 4,
        "vibration": 2.35, "voltage_imbalance": 0.3
    },
    "Cavitation": {
        "current": 0.94, "temperature": 3,
        "vibration": 1.95, "voltage_imbalance": 0.5
    },
    "Dry_Run_Low_Suction": {
        "current": 0.68, "temperature": 12,
        "vibration": 1.45, "voltage_imbalance": 0.3
    },
    "Overload": {
        "current": 1.48, "temperature": 15,
        "vibration": 1.40, "voltage_imbalance": 0.4
    },
    "Phase_Voltage_Imbalance": {
        "current": 1.15, "temperature": 9,
        "vibration": 1.20, "voltage_imbalance": 7.0
    },
    "Winding_Overheating": {
        "current": 1.12, "temperature": 22,
        "vibration": 1.10, "voltage_imbalance": 0.4
    }
}


# ---------------------------------------------------------
# SYNTHETIC DATA GENERATION
# ---------------------------------------------------------

def generate_dataset():
    records = []
    sample_id = 0

    for fault_name, fault_code in FAULTS.items():

        # Healthy data has severity 0.
        # Each fault gets mild, moderate and severe examples.
        severity_levels = (
            [0] if fault_name == "Normal"
            else [1, 2, 3]
        )

        samples_per_level = max(
            1, SAMPLES_PER_FAULT // len(severity_levels)
        )

        effect = FAULT_EFFECTS[fault_name]

        for severity in severity_levels:
            severity_factor = SEVERITY_FACTORS[severity]

            for _ in range(samples_per_level):
                sample_id += 1

                # Operating-condition variation
                load = np.clip(
                    rng.normal(BASE_LOAD, 0.12), 0.25, 1.0
                )

                ambient_temp = np.clip(
                    rng.normal(BASE_AMBIENT_TEMP, 3.0), 18, 42
                )

                # ---------------- VOLTAGE ----------------
                voltage_base = rng.normal(BASE_VOLTAGE, 3.0)

                voltage_1 = voltage_base + rng.normal(0, 0.8)
                voltage_2 = voltage_base + rng.normal(0, 0.8)
                voltage_3 = voltage_base + rng.normal(0, 0.8)

                if fault_name == "Phase_Voltage_Imbalance":
                    voltage_2 *= (
                        1 - 0.006 * severity_factor
                    )
                    voltage_3 *= (
                        1 + 0.005 * severity_factor
                    )

                mean_voltage = (
                    voltage_1 + voltage_2 + voltage_3
                ) / 3.0

                voltage_unbalance = (
                    max(voltage_1, voltage_2, voltage_3)
                    - min(voltage_1, voltage_2, voltage_3)
                ) / max(mean_voltage, 1.0) * 100

                # ---------------- CURRENT ----------------
                current_base = (
                    BASE_CURRENT
                    * (0.55 + 0.65 * load)
                    * effect["current"]
                )

                current_base *= (
                    1 + 0.13 * severity_factor
                    * (effect["current"] - 1)
                )

                current_1 = max(
                    0.03,
                    rng.normal(current_base, 0.12)
                )
                current_2 = max(
                    0.03,
                    rng.normal(current_base, 0.12)
                )
                current_3 = max(
                    0.03,
                    rng.normal(current_base, 0.12)
                )

                if fault_name == "Phase_Voltage_Imbalance":
                    current_2 *= 1 + 0.035 * severity_factor
                    current_3 *= 1 - 0.025 * severity_factor

                if fault_name == "Dry_Run_Low_Suction":
                    current_1 *= 1 - 0.08 * severity_factor
                    current_2 *= 1 - 0.08 * severity_factor
                    current_3 *= 1 - 0.08 * severity_factor

                mean_current = (
                    current_1 + current_2 + current_3
                ) / 3.0

                current_unbalance = (
                    max(current_1, current_2, current_3)
                    - min(current_1, current_2, current_3)
                ) / max(mean_current, 0.01) * 100

                # ---------------- TEMPERATURE ----------------
                stator_temp = (
                    ambient_temp
                    + BASE_STATOR_TEMP - BASE_AMBIENT_TEMP
                    + effect["temperature"] * severity_factor
                    + 7.0 * (load - BASE_LOAD)
                )

                stator_temp = rng.normal(stator_temp, 2.0)

                bearing_temp = (
                    ambient_temp
                    + BASE_BEARING_TEMP - BASE_AMBIENT_TEMP
                )

                if fault_name == "Bearing_Wear":
                    bearing_temp += 8 * severity_factor

                if fault_name == "Shaft_Misalignment":
                    bearing_temp += 4 * severity_factor

                bearing_temp += 2 * (load - BASE_LOAD)
                bearing_temp = rng.normal(bearing_temp, 1.8)

                # ---------------- VIBRATION ----------------
                vibration_base = (
                    BASE_VIBRATION
                    * (0.65 + 0.60 * load)
                    * effect["vibration"]
                )

                vibration_base *= (
                    1 + 0.55 * severity_factor
                    * (effect["vibration"] - 1)
                )

                vibration_x = max(
                    0.002,
                    rng.normal(vibration_base, 0.025)
                )

                vibration_y = max(
                    0.002,
                    rng.normal(vibration_base * 0.82, 0.025)
                )

                vibration_z = max(
                    0.002,
                    rng.normal(vibration_base * 0.68, 0.025)
                )

                vibration_rms = np.sqrt(
                    (
                        vibration_x**2
                        + vibration_y**2
                        + vibration_z**2
                    ) / 3
                )

                # Approximate severity label for model development.
                # Not a certified safety or maintenance threshold.
                severity_label = SEVERITY_NAMES[severity]

                records.append({
                    "sample_id": sample_id,

                    "load_fraction_simulated": load,
                    "ambient_temp_c": ambient_temp,

                    "phase1_voltage_v": voltage_1,
                    "phase2_voltage_v": voltage_2,
                    "phase3_voltage_v": voltage_3,
                    "mean_voltage_v": mean_voltage,
                    "voltage_unbalance_pct": voltage_unbalance,

                    "phase1_current_a": current_1,
                    "phase2_current_a": current_2,
                    "phase3_current_a": current_3,
                    "mean_current_a": mean_current,
                    "current_unbalance_pct": current_unbalance,

                    "stator_surface_temp_c": stator_temp,
                    "bearing_housing_temp_c": bearing_temp,

                    "vibration_x_rms_g": vibration_x,
                    "vibration_y_rms_g": vibration_y,
                    "vibration_z_rms_g": vibration_z,
                    "vibration_vector_rms_g": vibration_rms,

                    "fault_code": fault_code,
                    "fault_type": fault_name,
                    "severity_level": severity,
                    "severity_label": severity_label,

                    "data_origin":
                        "SYNTHETIC_SIMULATED_NOT_MEASURED"
                })

    df = pd.DataFrame(records)

    # Shuffle records for ML training
    df = df.sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)

    return df


# ---------------------------------------------------------
# SAVE FILES
# ---------------------------------------------------------

if __name__ == "__main__":

    dataset = generate_dataset()

    dataset.to_csv(
        "cg_ipm15_synthetic_dataset.csv",
        index=False
    )

    labels = pd.DataFrame([
        {
            "fault_code": code,
            "fault_type": name
        }
        for name, code in FAULTS.items()
    ])

    labels.to_csv(
        "fault_label_dictionary.csv",
        index=False
    )

    print("\nDATASET GENERATED SUCCESSFULLY")
    print("--------------------------------")
    print("File: cg_ipm15_synthetic_dataset.csv")
    print("Total records:", len(dataset))
    print("Classification classes:", dataset["fault_type"].nunique())
    print("Fault types excluding normal:", len(FAULTS) - 1)

    print("\nFault-class counts:")
    print(dataset["fault_type"].value_counts().sort_index())

    print("\nSeverity-class counts:")
    print(dataset["severity_label"].value_counts())

    print("\nIMPORTANT:")
    print("This dataset is synthetic, not real motor test data.")
    print("Verify motor nameplate values and validate with measured data.")