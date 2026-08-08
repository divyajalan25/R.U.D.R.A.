# 🚀 R.U.D.R.A. - Physics-Informed Digital Twin for Turbojet Engines

![RUDRA Dashboard Demo](./public/preview.png) *(Note: Please ensure `preview.png` is placed here to display the preview)*

**R.U.D.R.A.** is an advanced, web-based digital twin dashboard tailored for turbojet engines. It integrates real-time thermodynamic simulation, interactive 3D visualizations, and predictive machine learning models to monitor engine health degradation and forecast Remaining Useful Life (RUL).

---

## ✨ Key Features

### ⚙️ 1. Physics-Informed Thermodynamic Modeling (Brayton Cycle)
- **Standardized Gas Turbine Station Notations:** Strictly adheres to physical definitions ($P_3 > P_2$) for Compressor Inlet (2), Compressor Exit (3), Combustor Exit (4), and Turbine Exit (5).
- **Realistic Energy Boundaries:** Calculates exact isentropic efficiencies for the compressor and turbine (clipped within strict $0.7 - 0.95$ thermodynamic bounds).
- **Live T-s Diagram:** Computes and visually overlays the actual irreversible cycle onto the theoretical isentropic cycle using real-time gas properties ($c_{p,c}$, $\gamma_c$).

### 🧠 2. Machine Learning Predictive Maintenance (PdM)
- **Multi-Target Prediction:** Evaluates 6 distinct health metrics: `HI_compressor`, `HI_turbine`, `HI_combustor`, `HI_overall`, `thrust_N`, and `RUL`.
- **Continuous-Time EMA (Exponential Moving Average):** Calculates dynamic time-weights based on flight cycles to handle sparse and non-continuous telemetry effectively.
- **Adaptive Kalman Filtering (AKF):** Smooths noisy predictions to tighten confidence intervals for Overall Health Index (OHI). State histories are persisted using a robust backend SQLite database.

### 🖥️ 3. 3D Interactive Visualization & UI
- **Three.js Engine Rendering:** Generates a lightweight 3D turbojet model.
- **Dynamic Vertex Coloring & Raycasting:** Allows users to interact with components (compressor, combustor, turbine) directly to isolate performance metrics, with colors dynamically shifting based on real-time temperature telemetry.
- **Highly Optimized Canvas & SVG Displays:** Custom-built circular gauges and high-performance canvas sparklines avoid heavy charting libraries to guarantee a buttery-smooth 60 FPS dashboard experience.

---

## 🛠️ Technical Stack

- **Frontend:** Pure HTML5, CSS3, Vanilla JavaScript, Three.js
- **Backend:** Python 3.10+, FastAPI, Uvicorn, SQLite
- **Machine Learning:** XGBoost Ensemble Regression, Scikit-Learn

---

## 🚀 Getting Started

### 1. Prerequisites
Make sure you have Python 3.10 or later installed on your system. It is highly recommended to use a virtual environment.

```bash
# Install the required dependencies
pip install -r requirements.txt
```

### 2. Launching the Twin
To run the full application—including the FastAPI prediction server and the frontend dashboard—simply execute the startup script:

```bash
python3 start_all.py
```

This will:
1. Verify port availability and initialize the FastAPI backend on `http://localhost:8001`.
2. Automatically load the pre-trained XGBoost machine learning models.
3. Open the R.U.D.R.A. digital twin dashboard in your default web browser.

---

## 📂 Data Format

The Digital Twin accepts batch data uploads in CSV format for offline predictions. Ensure your dataset includes the following columns:
`EngineID, Cycle, Altitude_m, Mach, Tamb_K, Pamb_Pa, RPM_rev_min, FuelFlow_kg_s, P2_Pa, T2_K, P3_Pa, T3_K, P4_Pa, T4_K`

*(Note: Raw telemetry `P2_Pa` aligns with the compressor exit in this dataset, which the API securely translates to standard $P_3$ for the predictive models).* 

---

*Created by [Divya](mailto:divya.25bai10900@vitbhopal.ac.in)*
