# Turbojet Digital Twin — Dashboard Mockup

## Overview
A web-based digital twin dashboard for a turbojet engine. It provides real-time simulation, monitoring, and visualization of thermodynamic cycles, engine health degradation, and sensor telemetry.

## Technical Stack
- **Frontend**: Pure HTML, CSS, and Vanilla JavaScript. Designed for maximum performance with a lightweight footprint.
- **3D Visualization**: **Three.js** is used for rendering the 3D engine geometry, enabling dynamic camera controls and interactive data visualization.
- **Data Visualization**: Custom HTML5 Canvas and SVG implementations for rendering complex graphs (e.g., T-s diagrams, sparklines, and trend charts) without relying on heavy external charting libraries.
- **Backend/Serving**: FastAPI backend (`server.py`) serving a machine learning API and the frontend dashboard, along with a convenient `start_all.py` script.

## Key Technical Features

### 1. Thermodynamic Brayton Cycle Simulation
The core physics engine computes both the ideal and actual Brayton cycles based on current telemetry:
- Uses accurate gas properties:
  - **Cold Section (Compressor)**: $c_{p,c} = 1005$ J/(kg·K), $\gamma_c = 1.4$
  - **Hot Section (Combustor/Turbine)**: $c_{p,h} = 1148$ J/(kg·K), $\gamma_h = 1.33$
- **T-s Diagram (Temperature vs. Specific Entropy)**:
  - Dynamically calculates process entropy via $\Delta s = c_p \ln(T_2/T_1) - R \ln(P_2/P_1)$.
  - Generates geometrically exact logarithmic thermodynamic curves rather than simple straight-line approximations.
  - Visually overlays the irreversible (actual) cycle on top of the isentropic (ideal) cycle and calculates area-based entropy generation (irreversibility losses).

### 2. 3D Engine Visualization
- **Dynamic Vertex Coloring**: The Three.js implementation dynamically alters the color properties of the engine geometry based on real-time temperature telemetry.
- **Raycasting**: Allows users to interact with and click on specific engine components (compressor, combustor, turbine) to pull up localized performance metrics.

### 3. Degradation & Health Monitoring
- Simulates long-term degradation of aerodynamic components across multiple flight cycles.
- Computes an **Overall Health Index (OHI)** and estimates **Remaining Useful Life (RUL)** with statistical confidence bounds ($\pm \sigma$).
- Includes automated **Physics Constraints Checking** to flag anomalous data (e.g., efficiencies exceeding 100% or non-monotonic degradation behavior).

### 4. Telemetry & Analytics Dashboard
- Custom-built, highly optimized SVG circular gauges for real-time parameter tracking.
- Canvas-based sparklines for immediate historical context windowing.
- Extensive specific work, thermal efficiency, and back-work ratio comparisons between the real-world and ideal thermodynamic cycles.

## Running the Project

To run the full application (including the machine learning prediction API and the frontend dashboard), ensure you have Python installed and execute the startup script from the root directory:

```bash
python3 start_all.py
```

This script will automatically start the FastAPI backend on port `8001`, load the XGBoost machine learning models, and open the dashboard in your default web browser (`http://localhost:8001`).

*(Note: If you only want to view the static frontend mockup without the ML prediction features, you can optionally run `python3 launcher.py` to start a simple HTTP server on port `8000`).*
