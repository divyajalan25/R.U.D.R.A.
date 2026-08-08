# Turbojet Digital Twin — Rudra

## Overview
A web-based digital twin dashboard for a turbojet engine. It provides real-time simulation, monitoring, and visualization of thermodynamic cycles, engine health degradation, and sensor telemetry using an advanced Machine Learning backend.

## Technical Stack
- **Frontend**: Pure HTML, CSS, and Vanilla JavaScript. Designed for maximum performance with a lightweight footprint.
- **3D Visualization**: **Three.js** is used for rendering the 3D engine geometry, enabling dynamic camera controls and interactive data visualization.
- **Machine Learning Backend**: FastAPI backend (`server.py`) serving a robust XGBoost ensemble model. The backend maintains state across engine cycles to predict health components and Remaining Useful Life (RUL).
- **Time-Series Processing**: Uses Continuous-Time Exponential Moving Averages (EMA) to handle non-continuous test sets, meaning it accurately tracks degradation even when flight cycles are skipped.
- **Kalman Filtering**: Includes an Adaptive Kalman Filter (AKF) that smooths raw predictions, reducing noise and improving confidence intervals for the Overall Health Index (OHI).

## Key Technical Features

### 1. Thermodynamic Brayton Cycle Simulation
The core physics engine computes both the ideal and actual Brayton cycles based on current telemetry:
- Uses accurate gas properties ($c_{p,c}$, $\gamma_c$, etc.).
- Generates geometrically exact logarithmic thermodynamic curves for the T-s Diagram.
- Visually overlays the irreversible (actual) cycle on top of the isentropic (ideal) cycle.

### 2. Machine Learning Predictive Maintenance
- The backend evaluates 6 target variables: `HI_compressor`, `HI_turbine`, `HI_combustor`, `HI_overall`, `thrust_N`, and `RUL`.
- **Continuous-Time EMA**: Calculates dynamic alpha weights based on the time step (`dt`) between the last known flight cycle and the current one, providing robust predictions on sparse telemetry data.
- **Stateful DB**: Uses SQLite to persist Kalman Filter states across server restarts for multiple active engines.

### 3. 3D Engine Visualization
- **Dynamic Vertex Coloring**: The Three.js implementation dynamically alters the color properties of the engine geometry based on real-time temperature telemetry.
- **Raycasting**: Allows users to interact with and click on specific engine components (compressor, combustor, turbine) to pull up localized performance metrics.

### 4. Telemetry & Analytics Dashboard
- Custom-built, highly optimized SVG circular gauges for real-time parameter tracking.
- Canvas-based sparklines for immediate historical context windowing.

## Running the Project

To run the full application (including the machine learning prediction API and the frontend dashboard), execute the startup script from the root directory:

```bash
python3 start_all.py
```

This script will automatically start the FastAPI backend on port `8001`, load the XGBoost machine learning models, and open the dashboard in your default web browser (`http://localhost:8001`).
