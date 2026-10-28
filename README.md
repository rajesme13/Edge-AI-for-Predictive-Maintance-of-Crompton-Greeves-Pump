# Industrial Edge-AI Framework for Predictive Maintenance of Electric Motors Using Multi-Sensor Fusion

## 1. Project Overview

This project develops an **Industrial Edge-AI framework for predictive maintenance of electric motors** using multi-sensor data fusion and machine learning.

The prototype focuses on monitoring a Crompton Greaves CG-IPM-15 three-phase centrifugal pump motor using an ESP32-based edge monitoring system. Electrical, thermal, and vibration parameters are used to identify potential motor operating conditions, classify faults, and estimate their severity.

The current dataset is **synthetic**, generated using a Python-based simulation approach. It supports initial software development, preprocessing, model training, and proof-of-concept evaluation. It must not be treated as real motor measurements or validated manufacturer data.

## 2. Project Objectives

- Monitor motor voltage, current, temperature, and vibration parameters.
- Develop a labeled multisensor dataset for machine learning.
- Classify normal operation and different simulated fault conditions.
- Estimate fault severity using four severity categories.
- Establish a foundation for ESP32-based edge monitoring.
- Support future deployment of predictive maintenance models in industrial environments.

## 3. System Architecture

The proposed framework follows this workflow:

1. **Sensor acquisition:** Collect voltage, current, temperature, and vibration measurements.
2. **Data preprocessing:** Clean, normalize, and validate sensor readings.
3. **Feature engineering:** Extract statistical and signal-processing features.
4. **Fault classification:** Train a machine learning model to identify operating conditions.
5. **Severity estimation:** Estimate the severity category of a detected condition.
6. **Edge deployment:** Optimize the trained model for suitable embedded hardware.
7. **Monitoring and alerts:** Display motor health information and issue alerts when required.

### Proposed Hardware

| Component | Purpose |
|---|---|
| ESP32 | Sensor acquisition and edge processing |
| Isolated three-phase voltage transducers | Phase-voltage measurement |
| Current transformers or Hall-effect current sensors | Phase-current measurement |
| DS18B20 or suitable industrial temperature sensor | Temperature monitoring |
| ADXL345 or suitable industrial accelerometer | Vibration measurement |
| Optional pressure and flow sensors | Improved pump cavitation and dry-run detection |

**Electrical safety:** Never connect mains voltage directly to the ESP32 or its ADC pins. Use appropriately rated, isolated sensing circuits, protective enclosures, and qualified supervision.

## 4. Dataset Description

The synthetic dataset is generated using:

`generate_cg_ipm15_synthetic_data.py`

### Dataset files

| File | Description |
|---|---|
| `generate_cg_ipm15_synthetic_data.py` | Python script for generating the synthetic dataset |
| `cg_ipm15_synthetic_multisensor.csv` | Generated multisensor dataset |
| `fault_label_dictionary.csv` | Fault codes, labels, and descriptions |
| `dataset_metadata.json` | Dataset configuration and metadata |

The default generator configuration creates approximately **9,000 records**, with 1,000 samples for each of nine classes. The exact count depends on the generator configuration used.

### Sensor parameters and units

| Parameter | Description | Unit |
|---|---|---|
| `voltage_phase1_v` | Phase 1 voltage | V |
| `voltage_phase2_v` | Phase 2 voltage | V |
| `voltage_phase3_v` | Phase 3 voltage | V |
| `voltage_mean_v` | Mean phase voltage | V |
| `voltage_imbalance_pct` | Voltage imbalance | % |
| `current_phase1_a` | Phase 1 current | A |
| `current_phase2_a` | Phase 2 current | A |
| `current_phase3_a` | Phase 3 current | A |
| `current_mean_a` | Mean phase current | A |
| `current_imbalance_pct` | Current imbalance | % |
| `stator_surface_temp_c` | Stator surface temperature | °C |
| `bearing_housing_temp_c` | Bearing housing temperature | °C |
| `vibration_x_rms_g` | X-axis RMS acceleration | g |
| `vibration_y_rms_g` | Y-axis RMS acceleration | g |
| `vibration_z_rms_g` | Z-axis RMS acceleration | g |
| `vibration_vector_rms_g` | Vibration vector RMS | g |
| `fault_code` | Encoded fault class | Unitless |
| `fault_type` | Fault class label | Categorical |
| `severity_level` | Severity category | Ordinal |
| `severity_label` | Severity name | Categorical |
| `data_origin` | Data source indicator | Categorical |

**Unit note:** Voltage is measured in volts (V), current in amperes (A), and acceleration in SI units of m/s². The generator's vibration columns use g, where 1 g = 9.80665 m/s². Temperature is stored in °C for practical monitoring; the SI unit of thermodynamic temperature is kelvin (K).

## 5. Fault Classification

The generator supports nine classes:

| Fault Code | Fault Class | Description |
|---:|---|---|
| 0 | Normal | Simulated normal operating condition |
| 1 | Bearing_Wear | Simulated bearing wear |
| 2 | Shaft_Misalignment | Simulated shaft misalignment |
| 3 | Rotor_Impeller_Imbalance | Simulated rotor or impeller imbalance |
| 4 | Cavitation | Simulated pump cavitation |
| 5 | Dry_Run_Low_Suction | Simulated dry-run or low-suction condition |
| 6 | Overload | Simulated motor overload |
| 7 | Phase_Voltage_Imbalance | Simulated three-phase voltage imbalance |
| 8 | Winding_Overheating | Simulated winding overheating |

These labels describe the conditions represented by the generator. They do not establish that the corresponding fault can be reliably identified from real sensor measurements.

## 6. Fault Severity Levels

The dataset uses four severity categories:

| Severity Level | Label | Interpretation |
|---:|---|---|
| 0 | Normal | No simulated fault |
| 1 | Mild | Mild simulated abnormality |
| 2 | Moderate | Moderate simulated abnormality |
| 3 | Severe | Severe simulated abnormality |

Severity is a **synthetic label**, not a validated safety threshold or clinical/industrial risk rating. Real-world thresholds must be established from motor specifications, engineering standards, measured data, and appropriate validation.

## 7. Technology Stack

- **Programming:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Edge hardware:** ESP32
- **Sensors:** Voltage, current, temperature, and vibration sensors
- **Potential deployment:** TensorFlow Lite for Microcontrollers or another compatible embedded inference framework, depending on model size and hardware constraints

## 8. Requirements

Install Python 3.10 or a compatible version and the required packages.

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

If you are using a virtual environment, activate it before installing dependencies.

## 9. How to Run the Dataset Generator

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Run the generator:

```bash
python generate_cg_ipm15_synthetic_data.py
```

The script should create the CSV dataset and associated metadata files in the output directory configured in the code.

Check the script's output path and configuration if the files are saved to a different location.

## 10. Loading the Dataset

Use the following Python example to load and inspect the generated CSV file:

```python
import pandas as pd

df = pd.read_csv("cg_ipm15_synthetic_multisensor.csv")

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nFault distribution:")
print(df["fault_type"].value_counts())

print("\nSeverity distribution:")
print(df["severity_label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())
```

Update the CSV path if the file is stored in a subdirectory.

## 11. Machine Learning Workflow

A recommended initial workflow is:

1. Load the generated CSV file.
2. Verify missing values, data types, class counts, and feature ranges.
3. Select sensor-derived features as model inputs.
4. Encode `fault_type` as the classification target.
5. Split data into training, validation, and test sets.
6. Train baseline models such as Random Forest, Decision Tree, and Gradient Boosting.
7. Evaluate performance using accuracy, precision, recall, macro F1-score, and a confusion matrix.
8. Train a separate severity model if severity prediction is required.
9. Evaluate generalization using independently generated scenarios and, later, measured motor data.
10. Optimize the model for edge deployment and test its memory, latency, and reliability on the target hardware.

**Important:** Randomly splitting samples generated from the same simulation rules can produce overly optimistic results. Test on independently generated operating scenarios and, when available, data from separate operating sessions or physical motors.

## 12. Important Research Limitations

- The dataset is synthetic and does not represent verified field measurements.
- The baseline motor parameters in the generator are illustrative assumptions, not confirmed CG-IPM-15 nameplate specifications.
- Synthetic fault patterns may not fully represent real motor dynamics or interactions between faults.
- Voltage, current, temperature, and low-rate vibration RMS values alone may not uniquely identify every fault.
- Cavitation and dry-run detection can benefit from pump pressure, suction, flow, and related operating measurements.
- Bearing and mechanical fault diagnosis may require high-frequency vibration acquisition, spectral analysis, and suitable sensor mounting.
- Fault severity requires calibration against actual measurements and validated engineering criteria.
- Machine learning performance on synthetic data does not prove reliable performance on a physical motor.

For these reasons, results should be reported as **synthetic-data proof-of-concept results** until validated using real measurements.

## 13. Future Enhancements

- Acquire real motor data using the ESP32 and calibrated sensor interfaces.
- Add vibration waveform acquisition and FFT-based feature extraction.
- Integrate pressure and flow sensors for pump fault discrimination.
- Develop sensor-fusion models using Random Forest, XGBoost, LightGBM, or suitable deep-learning architectures.
- Implement anomaly detection for previously unseen operating conditions.
- Add live dashboards, historical trends, and maintenance alerts.
- Quantize and deploy a suitable model on edge hardware.
- Validate the system across different loads, speeds, operating sessions, and fault conditions.

## 14. Expected Outcome

The project aims to establish a modular prototype for motor-condition monitoring, simulated fault classification, severity estimation, and future edge-AI deployment. Its final reliability will depend on sensor calibration, representative physical measurements, suitable test procedures, and independent validation.

## 15. Author and Project Information

**Project Title:** Industrial Edge-AI Framework for Predictive Maintenance of Electric Motors Using Multi-Sensor Fusion

**Application:** Three-phase centrifugal pump motor condition monitoring

**Prototype Platform:** ESP32 with voltage, current, temperature, and vibration sensing

**Dataset Type:** Synthetic multisensor dataset

**Project Stage:** Software proof of concept; physical validation required

---

## License

Choose a license before publishing this repository. For example, the MIT License permits reuse subject to its terms. Do not add a license unless you intend to grant those permissions.

## Disclaimer

This repository is intended for educational and research purposes. Synthetic data and prototype predictions must not be used as the sole basis for industrial safety decisions, fault isolation, or maintenance actions. Validate the system with appropriate measurements, engineering review, and safety procedures before operational use.
