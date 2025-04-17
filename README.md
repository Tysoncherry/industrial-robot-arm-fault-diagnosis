# Mobile Robot Fault Diagnosis

This project simulates and diagnoses actuator and sensor faults in a differential-drive mobile robot using Python. It includes:
- Custom simulation of robot behavior
- Fault injection for 10 different fault types
- Feature extraction pipeline
- Machine learning-based fault classification
- ROS 2 & Gazebo validation (optional, separate modules)

---

## 🗂 Directory Structure

```
mobile_robot_fault_diagnosis/
├── actuators.py               # DC motor with fault logic
├── sensors.py                 # Encoder and IMU sensor models
├── robot_model.py             # Differential drive kinematic robot
├── logging_simulation.py      # Run simulation and export raw data
├── extract_features.py        # Extract features per window
├── train_model.py             # Train and save ML model
├── integration_pipeline.py    # Full pipeline: simulate + predict
├── dataset/
│   ├── raw/                   # CSVs of simulated time-series data
│   └── processed/             # Extracted feature CSVs
├── ml_model/                  # Trained model and scaler
├── results/                   # Prediction outputs per fault
└── README.md
```

---

## ✅ Faults Simulated

| Fault Type                 | Component   |
|---------------------------|-------------|
| voltage_drop              | DC Motor    |
| torque_degradation        | DC Motor    |
| partial_stall             | DC Motor    |
| overspeed                 | DC Motor    |
| encoder_drift             | Encoder     |
| encoder_skip              | Encoder     |
| gyro_bias                 | IMU (Gyro)  |
| accelerometer_offset      | IMU (Accel) |
| accelerometer_vibration   | IMU (Accel) |
| normal                    | No Fault    |

---

## 🧪 How to Run (Step-by-Step)

1. Generate Raw Data:
```
python logging_simulation.py
```

2. Extract Features:
```
python extract_features.py
```

3. Train the ML Model:
```
python train_model.py
```

4. Run Full Integration (Test One Fault):
```
python integration_pipeline.py
```

---

## 📦 Requirements
- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, scipy, joblib

Install via:
```
pip install -r requirements.txt
```

---

## 📈 Output
- Raw simulation data → dataset/raw/
- Feature CSVs → dataset/processed/
- Trained model → ml_model/
- Prediction results → results/

---

## 🧠 Notes
- All faults are simulated in software only (runs on any laptop)
- ROS 2 + Gazebo extensions available for validation
- Window size and simulation time can be adjusted

---

© Ram Charan Vuyyala | M.Eng Mechatronics and Robotics 
