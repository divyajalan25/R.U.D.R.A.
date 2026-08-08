"""
Turbojet Digital Twin — FastAPI Backend
========================================
Loads trained XGBoost models and serves health-monitoring predictions.

Endpoints:
  GET  /health          → 200 when models loaded
  POST /predict/single  → JSON body with 13 sensors + EngineID + Cycle
  POST /predict/batch   → multipart CSV upload

Physics constants: gamma=1.4, cp=1005.0 (matches training notebook).
"""

import os
import json
import io
import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

# ──────────────────────── Constants ────────────────────────
GAMMA = 1.4
CP = 1005.0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "results", "models")
DB_PATH = os.path.join(BASE_DIR, "digital_twin.db")

# ──────────────────────── Global State ────────────────────────
models: dict = {}
scaler = None
feature_columns: list = []
fleet_defaults: dict = {}

# New 32-feature Digital Twin models for all 6 targets
new_ensemble_models: dict = {}
new_feature_names = None
new_baselines = None

# AKF state per engine — loaded from DB on startup, persisted on every update
# Global variables for Stateful Machine Learning context
akf_states: dict = {}
ema_states: dict = {}
last_cycle: dict = {}


# ──────────────────────── Database Layer ────────────────────────
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't exist. Safe to call on every startup."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS engines (
            engine_id        TEXT PRIMARY KEY,
            serial_number    TEXT,
            model            TEXT,
            aircraft_reg     TEXT,
            position         TEXT,
            base             TEXT,
            cycles_at_reg    INTEGER DEFAULT 0,
            last_overhaul    TEXT,
            warn_threshold   REAL    DEFAULT 0.82,
            crit_threshold   REAL    DEFAULT 0.70,
            registered_at    TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS akf_multi_states (
            engine_id             TEXT,
            target                TEXT,
            x                     REAL,
            P                     REAL,
            Q                     REAL,
            epoch                 INTEGER DEFAULT 1,
            cold_start_active     INTEGER DEFAULT 0,
            cold_start_remaining  INTEGER DEFAULT 0,
            updated_at            TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (engine_id, target)
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            engine_id      TEXT,
            cycle          INTEGER,
            hi_overall     REAL,
            hi_compressor  REAL,
            hi_turbine     REAL,
            hi_combustor   REAL,
            kf_health      REAL,
            uncertainty    REAL,
            rul            REAL,
            thrust_n       REAL,
            predicted_at   TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS maintenance_logs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            engine_id        TEXT,
            technician       TEXT,
            maintenance_type TEXT,
            method           TEXT,
            health_before    REAL,
            health_after     REAL,
            logged_at        TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()
    print("  ✓ SQLite DB ready:", DB_PATH)


def load_akf_states_from_db():
    """Restore AKF state for all engines and targets from the DB into the in-memory dict."""
    conn = get_db()
    # If the table doesn't exist yet (first run after migration), it will fail or return empty, safe to handle
    try:
        rows = conn.execute("SELECT * FROM akf_multi_states").fetchall()
        for row in rows:
            eid = row["engine_id"]
            target = row["target"]
            if eid not in akf_states:
                akf_states[eid] = {}
            akf_states[eid][target] = {
                "x": row["x"],
                "P": row["P"],
                "Q": row["Q"],
                "epoch": row["epoch"],
                "cold_start_active": bool(row["cold_start_active"]),
                "cold_start_cycles_remaining": row["cold_start_remaining"],
            }
        print(f"  ✓ Restored AKF state for {len(akf_states)} engines from DB")
    except Exception as e:
        print(f"  ⚠ Could not load multi-states from DB: {e}")
    finally:
        conn.close()


def save_akf_state(engine_id: str, target: str):
    """Upsert the current in-memory AKF state for one engine and target into the DB."""
    s = akf_states.get(engine_id, {}).get(target)
    if not s:
        return
    conn = get_db()
    conn.execute("""
        INSERT INTO akf_multi_states (engine_id, target, x, P, Q, epoch, cold_start_active, cold_start_remaining, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(engine_id, target) DO UPDATE SET
            x=excluded.x, P=excluded.P, Q=excluded.Q, epoch=excluded.epoch,
            cold_start_active=excluded.cold_start_active,
            cold_start_remaining=excluded.cold_start_remaining,
            updated_at=excluded.updated_at
    """, (engine_id, target, s["x"], s["P"], s["Q"], s["epoch"],
           int(s["cold_start_active"]), s["cold_start_cycles_remaining"]))
    conn.commit()
    conn.close()


def log_prediction(row: dict, kf_health: float = None, uncertainty: float = None):
    """Write one prediction row to the predictions table."""
    conn = get_db()
    conn.execute("""
        INSERT INTO predictions
            (engine_id, cycle, hi_overall, hi_compressor, hi_turbine, hi_combustor,
             kf_health, uncertainty, rul, thrust_n)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(row.get("engine_id", "")),
        int(row.get("cycle", 0)),
        float(row.get("HI_overall", 0)),
        float(row.get("HI_compressor", 0)),
        float(row.get("HI_turbine", 0)),
        float(row.get("HI_combustor", 0)),
        kf_health,
        uncertainty,
        float(row.get("RUL", 0)),
        float(row.get("thrust_N", 0)),
    ))
    conn.commit()
    conn.close()


def update_kalman_for_prediction(engine_id: str, target: str, cycle: int, raw_prediction: float) -> tuple[float, float]:
    """
    Update the recursive Adaptive Health Kalman Filter state for an engine and target.
    Scales process noise by dt (cycle gap) to correctly handle sparse/random cycles.
    """
    global akf_states
    
    # 1. Get or initialize state
    if engine_id not in akf_states:
        akf_states[engine_id] = {}
        
    state = akf_states[engine_id].get(target)
    if not state:
        state = {
            "x": 1.0,
            "P": 0.01,
            "Q": 0.005,  # Tuned process noise
            "epoch": 0,
            "cold_start_active": False,
            "cold_start_cycles_remaining": 0,
        }
        akf_states[engine_id][target] = state

    # 2. Query last prediction cycle from DB to compute dt
    conn = get_db()
    last_pred = conn.execute(
        "SELECT cycle FROM predictions WHERE engine_id=? ORDER BY cycle DESC LIMIT 1",
        (engine_id,)
    ).fetchone()
    conn.close()

    if last_pred:
        dt = max(1, cycle - last_pred["cycle"])
    else:
        dt = 1

    # 3. Predict step (scale base process noise by dt)
    state["P"] = state["P"] + state["Q"] * dt

    # 4. Update step
    measurement_variance = 0.2  # Tuned uncertainty standard deviation squared (R = 0.2)
    R = measurement_variance
    S = state["P"] + R
    K = state["P"] / S
    
    state["x"] = state["x"] + K * (raw_prediction - state["x"])
    state["P"] = (1 - K) * state["P"]
    state["epoch"] += 1

    # 5. Persist the updated state to database
    save_akf_state(engine_id, target)

    return float(state["x"]), float(np.sqrt(state["P"]))


# ──────────────────────── Column Rename Map ────────────────────────
COL_RENAME = {
    "EngineID":       "engine_id",
    "Cycle":          "cycle",
    "Altitude_m":     "altitude",
    "Mach":           "mach",
    "Tamb_K":         "T_amb",
    "Pamb_Pa":        "P_amb",
    "RPM_rev_min":    "rpm",
    "FuelFlow_kg_s":  "fuel_flow",
    "P2_Pa":          "P2",
    "T2_K":           "T2",
    "P3_Pa":          "P3",
    "T3_K":           "T3",
    "P4_Pa":          "P4",
    "T4_K":           "T4",
}


def _load_all_models():
    """Load models, scaler, feature list, fleet defaults into globals."""
    global models, scaler, feature_columns, fleet_defaults

    # Feature column ordering
    path = os.path.join(MODELS_DIR, "feature_columns.json")
    with open(path) as f:
        feature_columns = json.load(f)
    print(f"  Feature columns ({len(feature_columns)}): {feature_columns}")

    # Fleet baseline fallback values
    path = os.path.join(MODELS_DIR, "fleet_baseline_defaults.json")
    with open(path) as f:
        fleet_defaults = json.load(f)
    print(f"  Fleet defaults: {fleet_defaults}")

    # Sklearn StandardScaler
    scaler = joblib.load(os.path.join(MODELS_DIR, "feature_scaler.joblib"))
    print(f"  Scaler loaded (n_features={scaler.n_features_in_})")

# XGBoost Booster models — 6 production models
    model_files = [
        "xgb_HI_compressor",
        "xgb_HI_turbine",
        "xgb_HI_combustor",
        "xgb_HI_overall",
        "xgb_thrust",
        "xgb_RUL",
    ]
    for name in model_files:
        booster = xgb.Booster()
        booster.load_model(os.path.join(MODELS_DIR, f"{name}.json"))
        models[name] = booster
        print(f"  ✓ {name}.json loaded")

    # Load new 32-feature ensemble models and baselines
    global new_ensemble_models, new_feature_names, new_baselines
    new_models_dir = os.path.join(BASE_DIR, "models")
    try:
        if os.path.exists(os.path.join(new_models_dir, "feature_names.pkl")):
            new_feature_names = joblib.load(os.path.join(new_models_dir, "feature_names.pkl"))
        if os.path.exists(os.path.join(new_models_dir, "baselines.pkl")):
            new_baselines = joblib.load(os.path.join(new_models_dir, "baselines.pkl"))
        
        # Load all 6 target models if they exist
        targets_list = ['HI_compressor', 'HI_turbine', 'HI_combustor', 'HI_overall', 'thrust_N', 'RUL']
        for target in targets_list:
            base_path = os.path.join(new_models_dir, f"base_models_{target}.pkl")
            meta_path = os.path.join(new_models_dir, f"meta_model_{target}.pkl")
            if os.path.exists(base_path) and os.path.exists(meta_path):
                new_ensemble_models[target] = {
                    'base_models': joblib.load(base_path),
                    'meta_model': joblib.load(meta_path)
                }
                print(f"  ✓ Loaded new ensemble models for target: {target}")
                
        # If HI_overall was not loaded via target-specific file, try old base_models.pkl fallback
        if 'HI_overall' not in new_ensemble_models:
            old_base_path = os.path.join(new_models_dir, "base_models.pkl")
            old_meta_path = os.path.join(new_models_dir, "meta_model.pkl")
            if os.path.exists(old_base_path) and os.path.exists(old_meta_path):
                new_ensemble_models['HI_overall'] = {
                    'base_models': joblib.load(old_base_path),
                    'meta_model': joblib.load(old_meta_path)
                }
                print("  ✓ Fallback loaded old overall health ensemble models (base_models.pkl)")
        print(f"  ✓ Loaded new ensemble models and baselines from {new_models_dir}")
    except Exception as e:
        print(f"  ⚠ Failed to load new models from {new_models_dir}: {e}")

    print(f"\n✓ All {len(models)} old models and new ensemble loaded and ready.")


# ──────────────────────── Lifespan ────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n── Initialising database ──")
    init_db()
    load_akf_states_from_db()
    print("\n── Loading model artifacts ──")
    _load_all_models()
    yield
    print("\n── Shutting down ──")


# ──────────────────────── App Setup ────────────────────────
app = FastAPI(
    title="Turbojet Digital Twin API",
    description="Health monitoring predictions for single-spool turbojet engines",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────── Pydantic Schema ────────────────────────
class SensorInput(BaseModel):
    EngineID: str = "NEW_1"
    Cycle: int = 1
    Altitude_m: float
    Mach: float
    Tamb_K: float
    Pamb_Pa: float
    RPM_rev_min: float
    FuelFlow_kg_s: float
    P2_Pa: float
    T2_K: float
    P3_Pa: float
    T3_K: float
    P4_Pa: float
    T4_K: float


# ──────────────────────── Core Pipeline ────────────────────────
def compute_features_and_predict(df: pd.DataFrame, mode: str = "batch") -> pd.DataFrame:
    """
    Full inference pipeline — shared by /predict/single and /predict/batch.

    Steps:
      1. Compute physics features from raw sensor data
      2. Apply baseline logic (Case A / B / C)
      3. Assemble feature vector, scale, predict with all 6 models
      4. Derive TSFC from predicted thrust

    Args:
        df: DataFrame with internal column names (engine_id, cycle, altitude, etc.)
        mode: "single" (always fleet fallback) or "batch" (per-engine baseline check)
    """

    # Store original row index to restore output order later, and sort chronologically
    df['orig_index'] = df.index
    df = df.sort_values(['engine_id', 'cycle'])
    
    global ema_states, last_cycle

    # ── Step 1: Physics feature engineering ──

    # Stagnation (total) conditions from flight state
    df["T_t_amb"] = df["T_amb"] * (1.0 + (GAMMA - 1.0) / 2.0 * df["mach"] ** 2)
    df["P_t_amb"] = df["P_amb"] * (
        (1.0 + (GAMMA - 1.0) / 2.0 * df["mach"] ** 2) ** (GAMMA / (GAMMA - 1.0))
    )

    # Pressure ratios
    df["PR_compressor"] = df["P2"] / df["P_t_amb"]
    df["PR_combustor"]  = df["P3"] / df["P2"]
    df["PR_turbine"]    = df["P3"] / df["P4"]
    df["PR_overall"]    = df["P4"] / df["P_t_amb"]

    # Ideal isentropic exit temperatures
    exp = (GAMMA - 1.0) / GAMMA  # 0.2857 for gamma=1.4

    T2_ideal = df["T_t_amb"] * df["PR_compressor"] ** exp

    # NOTE: positive exponent here matches training-time code.
    # This makes T4_ideal > T3 (since PR_turbine > 1), which produces
    # negative eta_turbine. The fleet default (-0.804) confirms this convention.
    T4_ideal = df["T3"] * df["PR_turbine"] ** exp

    # Isentropic efficiencies
    denom_c = (df["T2"] - df["T_t_amb"]).replace(0.0, np.nan)
    denom_t = (df["T3"] - T4_ideal).replace(0.0, np.nan)

    df["eta_compressor"] = (T2_ideal - df["T_t_amb"]) / denom_c
    df["eta_turbine"]    = (df["T3"] - df["T4"]) / denom_t

    # Guard against NaN/inf from degenerate inputs
    for col in ["eta_compressor", "eta_turbine"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Work terms (specific, J/kg)
    df["compressor_work"]   = CP * (df["T2"] - df["T_t_amb"])
    df["turbine_work"]      = CP * (df["T3"] - df["T4"])
    df["net_specific_work"] = df["turbine_work"] - df["compressor_work"]

    # Combustor temperature rise
    df["T_rise_combustor"] = df["T3"] - df["T2"]

    # ── Step 2: Baseline feature logic ──

    # Initialize all rows with fleet fallback
    df["eta_compressor_baseline_input"] = fleet_defaults["eta_compressor_baseline_input"]
    df["eta_turbine_baseline_input"]    = fleet_defaults["eta_turbine_baseline_input"]
    df["PR_combustor_baseline_input"]   = fleet_defaults["PR_combustor_baseline_input"]
    df["baseline_source"] = "fleet_fallback"

    if mode == "batch":
        # Check each engine for sufficient early-cycle history
        for eid in df["engine_id"].unique():
            engine_mask = df["engine_id"] == eid
            early_rows  = df.loc[engine_mask & (df["cycle"] <= 10)]

            if len(early_rows) >= 10:
                # Case A: engine has ≥10 rows with cycle ≤ 10 → use own baseline
                df.loc[engine_mask, "eta_compressor_baseline_input"] = early_rows["eta_compressor"].mean()
                df.loc[engine_mask, "eta_turbine_baseline_input"]    = early_rows["eta_turbine"].mean()
                df.loc[engine_mask, "PR_combustor_baseline_input"]   = early_rows["PR_combustor"].mean()
                df.loc[engine_mask, "baseline_source"] = "engine_history"
            # else: Case B — stays fleet_fallback (already set)
    # mode == "single": Case C — always fleet fallback (already set)

    # ── Step 3: Assemble features, scale, predict ──

    # Build feature matrix in exact column order from feature_columns.json
    X = df[feature_columns].values.astype(np.float64)

    # Standardize with the training-time scaler
    X_scaled = scaler.transform(X)

    # Create XGBoost DMatrix
    dm = xgb.DMatrix(X_scaled, feature_names=feature_columns)

    # Run all 6 models
    df["HI_compressor"] = models["xgb_HI_compressor"].predict(dm).astype(float)
    df["HI_turbine"]    = models["xgb_HI_turbine"].predict(dm).astype(float)
    df["HI_combustor"]  = models["xgb_HI_combustor"].predict(dm).astype(float)
    df["HI_overall"]    = models["xgb_HI_overall"].predict(dm).astype(float)
    df["thrust_N"]      = models["xgb_thrust"].predict(dm).astype(float)
    df["RUL"]           = models["xgb_RUL"].predict(dm).astype(float)

    # ── Step 4: Derive TSFC ──
    # TSFC = (fuel_flow_kg_s × 1000) / predicted_thrust_N  →  g/(N·s)
    df["TSFC"] = (df["fuel_flow"] * 1000.0) / df["thrust_N"].clip(lower=1.0)

    # ── Step 5: New 32-Feature Pipeline for Ensemble Models ──
    global new_ensemble_models, new_feature_names, new_baselines
    if len(new_ensemble_models) > 0 and new_feature_names is not None:
        try:
            LHV_FUEL = 43000000
            new_df = df.copy()
            
            # Physics features
            new_df["a_amb"] = np.sqrt(GAMMA * 287 * new_df["T_amb"])
            new_df["V_aircraft"] = new_df["mach"] * new_df["a_amb"]
            new_df["T_t_amb"] = new_df["T_amb"] * (1.0 + (GAMMA - 1.0) / 2.0 * new_df["mach"] ** 2)
            new_df["P_t_amb"] = new_df["P_amb"] * ((1.0 + (GAMMA - 1.0) / 2.0 * new_df["mach"] ** 2) ** (GAMMA / (GAMMA - 1.0)))
            
            new_df['PR_compressor'] = new_df['P2'] / (new_df['P_t_amb'] + 1e-8)
            new_df['PR_combustor'] = new_df['P3'] / (new_df['P2'] + 1e-8)
            new_df['PR_turbine'] = new_df['P3'] / (new_df['P4'] + 1e-8)
            
            T2_ideal = new_df['T_t_amb'] * (new_df['PR_compressor']) ** ((GAMMA - 1.0) / GAMMA)
            T4_ideal = new_df['T3'] * (new_df['PR_turbine']) ** ((GAMMA - 1.0) / GAMMA)
            
            new_df['eta_compressor'] = np.clip(np.abs((T2_ideal - new_df['T_t_amb']) / (new_df['T2'] - new_df['T_t_amb'] + 1e-8)), 0.7, 0.95)
            new_df['eta_turbine'] = np.clip(np.abs((new_df['T3'] - new_df['T4']) / (new_df['T3'] - T4_ideal + 1e-8)), 0.7, 0.95)
            
            T3_ideal = new_df['T3'] * 1.05
            new_df['eta_combustor'] = np.clip(np.abs((new_df['T3'] - new_df['T2']) / (T3_ideal - new_df['T2'] + 1e-8)), 0.85, 1.0)
            
            new_df['residual_T2'] = np.abs(new_df['T2'] - T2_ideal)
            new_df['residual_T4'] = np.abs(new_df['T4'] - T4_ideal)
            
            Wc = CP * (new_df['T2'] - new_df['T_t_amb'])
            Wt = CP * (new_df['T3'] - new_df['T4'])
            new_df['work_ratio'] = np.abs(Wt / (Wc + 1e-8))
            
            Qin = new_df['fuel_flow'] * LHV_FUEL
            new_df['thermal_efficiency'] = np.abs((Wt - Wc) / (Qin + 1e-8))
            
            physics_features = [
                'eta_compressor', 'eta_turbine', 'eta_combustor',
                'residual_T2', 'residual_T4', 'work_ratio',
                'thermal_efficiency', 'PR_compressor'
            ]
            
            # Normalize features using dynamic or training baselines
            dynamic_baselines = {}
            for eid in new_df['engine_id'].unique():
                engine_mask = new_df['engine_id'] == eid
                early_rows = new_df.loc[engine_mask & (new_df['cycle'] <= 10)]
                if len(early_rows) > 0:
                    dynamic_baselines[eid] = early_rows[physics_features].mean().to_dict()
                else:
                    str_eid = str(eid)
                    int_eid = int(eid) if isinstance(eid, (int, float)) or (isinstance(eid, str) and eid.isdigit()) else None
                    
                    if new_baselines is not None and str_eid in new_baselines:
                        dynamic_baselines[eid] = new_baselines[str_eid]
                    elif new_baselines is not None and int_eid in new_baselines:
                        dynamic_baselines[eid] = new_baselines[int_eid]
                    else:
                        dynamic_baselines[eid] = {feat: 1.0 for feat in physics_features}
                        
            for feature in physics_features:
                baseline_vals = new_df['engine_id'].map(lambda x: dynamic_baselines.get(x, {}).get(feature, 1.0))
                new_df[f'{feature}_norm'] = new_df[feature] / (baseline_vals + 1e-8)
            
            # ── Step 2.5: Compute Continuous-Time EMA Features ──
            lambda_val = 0.2
            norm_features = [f'{feat}_norm' for feat in physics_features]
            ema_cols = [f'{feat}_ema' for feat in norm_features]
            delta_cols = [f'{feat}_delta' for feat in norm_features]
            
            # Create empty arrays for fast assignment
            ema_data = {col: np.zeros(len(new_df)) for col in ema_cols}
            
            for i, (idx, row) in enumerate(new_df.iterrows()):
                eid = str(row['engine_id'])
                current_cyc = int(row['cycle'])
                
                if eid not in ema_states:
                    ema_states[eid] = {feat: row[feat] for feat in norm_features}
                    last_cycle[eid] = current_cyc
                    dt = 1
                else:
                    dt = current_cyc - last_cycle[eid]
                    if dt <= 0:
                        dt = 1
                    last_cycle[eid] = current_cyc
                    
                alpha_eff = 1.0 - (1.0 - lambda_val)**dt
                
                for feat, ema_col in zip(norm_features, ema_cols):
                    old_ema = ema_states[eid][feat]
                    new_ema = old_ema * (1 - alpha_eff) + row[feat] * alpha_eff
                    ema_states[eid][feat] = new_ema
                    ema_data[ema_col][i] = new_ema
                    
            for ema_col in ema_cols:
                new_df[ema_col] = ema_data[ema_col]
                
            for feat, ema_col, delta_col in zip(norm_features, ema_cols, delta_cols):
                new_df[delta_col] = new_df[feat] - new_df[ema_col]
                
            new_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            new_df.fillna(0, inplace=True)
            
            # ML Inference
            X_inf = new_df[new_feature_names].copy()
            
            for target, model_group in new_ensemble_models.items():
                OOF_preds = pd.DataFrame({
                    'xgb': model_group['base_models']['xgb'].predict(X_inf),
                    'lgb': model_group['base_models']['lgb'].predict(X_inf),
                    'cat': model_group['base_models']['cat'].predict(X_inf),
                    'rf': model_group['base_models']['rf'].predict(X_inf)
                })
                pred_val = model_group['meta_model'].predict(OOF_preds)
                
                # Assign predictions to the main DataFrame
                if target == 'HI_compressor':
                    df["HI_compressor"] = pred_val.astype(float)
                elif target == 'HI_turbine':
                    df["HI_turbine"] = pred_val.astype(float)
                elif target == 'HI_combustor':
                    df["HI_combustor"] = pred_val.astype(float)
                elif target == 'HI_overall':
                    df["Predicted_Health_Raw"] = pred_val.astype(float)
                elif target == 'thrust_N':
                    df["thrust_N"] = pred_val.astype(float)
                elif target == 'RUL':
                    df["RUL"] = pred_val.astype(float)
            
            if 'HI_overall' not in new_ensemble_models:
                df['Predicted_Health_Raw'] = df['HI_overall']
        except Exception as e:
            print(f"  ⚠ Failed to compute new physics features/predict targets: {e}")
            df['Predicted_Health_Raw'] = df['HI_overall']
    else:
        df['Predicted_Health_Raw'] = df['HI_overall']

    # Restore original sorting order to match input CSV exactly, then drop the index column
    df = df.sort_values('orig_index').drop(columns=['orig_index'])

    return df


def _df_to_response(df: pd.DataFrame) -> list[dict]:
    """Convert prediction DataFrame rows to JSON-serializable dicts."""
    records = []
    for _, r in df.iterrows():
        records.append({
            # Identity
            "engine_id":        str(r.get("engine_id", "")),
            "cycle":            int(r["cycle"]),
            # Raw sensors
            "altitude":         float(r["altitude"]),
            "mach":             float(r["mach"]),
            "T_amb":            float(r["T_amb"]),
            "P_amb":            float(r["P_amb"]),
            "rpm":              float(r["rpm"]),
            "fuel_flow":        float(r["fuel_flow"]),
            "P2":               float(r["P2"]),
            "T2":               float(r["T2"]),
            "P3":               float(r["P3"]),
            "T3":               float(r["T3"]),
            "P4":               float(r["P4"]),
            "T4":               float(r["T4"]),
            # Computed physics
            "T_t_amb":          float(r["T_t_amb"]),
            "P_t_amb":          float(r["P_t_amb"]),
            "PR_compressor":    float(r["PR_compressor"]),
            "PR_combustor":     float(r["PR_combustor"]),
            "PR_turbine":       float(r["PR_turbine"]),
            "PR_overall":       float(r["PR_overall"]),
            "eta_compressor":   float(r["eta_compressor"]),
            "eta_turbine":      float(r["eta_turbine"]),
            "compressor_work":  float(r["compressor_work"]),
            "turbine_work":     float(r["turbine_work"]),
            "net_specific_work":float(r["net_specific_work"]),
            "T_rise_combustor": float(r["T_rise_combustor"]),
            # Model predictions
            "HI_compressor":    float(r["HI_compressor"]),
            "HI_turbine":       float(r["HI_turbine"]),
            "HI_combustor":     float(r["HI_combustor"]),
            "HI_overall":       float(r["HI_overall"]),
            "Predicted_Health_Raw": float(r.get("Predicted_Health_Raw", r["HI_overall"])),
            "thrust_N":         float(r["thrust_N"]),
            "RUL":              float(r["RUL"]),
            "TSFC":             float(r["TSFC"]),
            "baseline_source":  str(r["baseline_source"]),
            "eta_compressor_norm": float(r.get("eta_compressor_norm", 1.0)),
            "eta_turbine_norm": float(r.get("eta_turbine_norm", 1.0)),
            "eta_combustor_norm": float(r.get("eta_combustor_norm", 1.0)),
            "work_ratio_norm": float(r.get("work_ratio_norm", 1.0)),
        })
    return records


# ──────────────────────── Endpoints ────────────────────────

@app.get("/health")
def health_check():
    """Returns 200 once all models are loaded and ready."""
    if not models or scaler is None:
        raise HTTPException(status_code=503, detail="Models not yet loaded")
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
        "feature_count": len(feature_columns),
    }


@app.post("/predict/single")
def predict_single(sensor: SensorInput):
    """
    Single-row prediction from manual slider input (Case C — always fleet fallback).
    """
    row = {
        "engine_id": sensor.EngineID,
        "cycle":     sensor.Cycle,
        "altitude":  sensor.Altitude_m,
        "mach":      sensor.Mach,
        "T_amb":     sensor.Tamb_K,
        "P_amb":     sensor.Pamb_Pa,
        "rpm":       sensor.RPM_rev_min,
        "fuel_flow": sensor.FuelFlow_kg_s,
        "P2":        sensor.P2_Pa,
        "T2":        sensor.T2_K,
        "P3":        sensor.P3_Pa,
        "T3":        sensor.T3_K,
        "P4":        sensor.P4_Pa,
        "T4":        sensor.T4_K,
    }
    df = pd.DataFrame([row])
    df = compute_features_and_predict(df, mode="single")
    results = _df_to_response(df)
    # Persist to DB
    for r in results:
        eid = str(r.get("engine_id"))
        cyc = int(r.get("cycle"))
        targets_list = ['HI_compressor', 'HI_turbine', 'HI_combustor', 'HI_overall', 'thrust_N', 'RUL']
        for t in targets_list:
            if t == 'HI_overall':
                raw_val = float(r.get("Predicted_Health_Raw", r["HI_overall"]))
            else:
                raw_val = float(r.get(t, 0.0))
                
            if t.startswith('HI_'):
                kf_val, unc = update_kalman_for_prediction(eid, t, cyc, raw_val)
                r[t] = kf_val
                if t == 'HI_overall':
                    r["kf_health"] = kf_val
                    r["uncertainty"] = unc
            else:
                r[t] = raw_val
                
        log_prediction(r, kf_health=r["HI_overall"], uncertainty=r["uncertainty"])
    return JSONResponse({"predictions": results})


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Batch prediction from CSV upload. Groups by EngineID and applies
    per-engine baseline logic. Uses high-performance bulk database transactions.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    # Validate required columns
    required_cols = set(COL_RENAME.keys())
    actual_cols = set(df.columns)
    missing = required_cols - actual_cols
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {sorted(missing)}. "
                   f"Expected: {sorted(required_cols)}"
         )

    # Rename to internal column names
    df = df.rename(columns=COL_RENAME)

    # Ensure engine_id is string
    df["engine_id"] = df["engine_id"].astype(str)

    # Run full pipeline
    df = compute_features_and_predict(df, mode="batch")
    results = _df_to_response(df)
    
    # Sort results by engine and cycle for proper Kalman state updates
    results.sort(key=lambda x: (x["engine_id"], x["cycle"]))

    # 1. Query the last recorded cycle for each engine in a single query
    conn = get_db()
    try:
        last_cycles_rows = conn.execute(
            "SELECT engine_id, MAX(cycle) as max_cycle FROM predictions GROUP BY engine_id"
        ).fetchall()
        last_cycles = {r["engine_id"]: r["max_cycle"] for r in last_cycles_rows}
    except Exception:
        last_cycles = {}

    predictions_to_insert = []
    akf_to_update = {}

    # 2. Run Kalman Filter in-memory to avoid row-by-row DB queries
    targets_list = ['HI_compressor', 'HI_turbine', 'HI_combustor', 'HI_overall', 'thrust_N', 'RUL']
    for r in results:
        eid = str(r["engine_id"])
        cyc = int(r["cycle"])
        
        if eid not in akf_states:
            akf_states[eid] = {}
        if eid not in akf_to_update:
            akf_to_update[eid] = {}

        prev_cycle = last_cycles.get(eid)
        if prev_cycle is None:
            dt = 1
        else:
            if cyc <= prev_cycle:
                dt = 1
                if eid in akf_states:
                    del akf_states[eid]
            else:
                dt = max(1, cyc - prev_cycle)

        for t in targets_list:
            if t == 'HI_overall':
                raw_val = float(r.get("Predicted_Health_Raw", r["HI_overall"]))
            else:
                raw_val = float(r.get(t, 0.0))

            if t.startswith('HI_'):
                if eid not in akf_states:
                    akf_states[eid] = {}
                state = akf_states[eid].get(t)
                if not state:
                    state = {
                        "x": 1.0,
                        "P": 0.01,
                        "Q": 0.005,
                        "epoch": 0,
                        "cold_start_active": False,
                        "cold_start_cycles_remaining": 0,
                    }
                    akf_states[eid][t] = state

                # Predict step
                state["P"] = state["P"] + state["Q"] * dt

                # Update step
                measurement_variance = 0.2
                R = measurement_variance
                S = state["P"] + R
                K = state["P"] / S
                
                state["x"] = state["x"] + K * (raw_val - state["x"])
                state["P"] = (1 - K) * state["P"]
                state["epoch"] += 1

                r[t] = float(state["x"])
                if t == 'HI_overall':
                    r["kf_health"] = float(state["x"])
                    r["uncertainty"] = float(np.sqrt(state["P"]))
                    
                akf_to_update[eid][t] = state
            else:
                r[t] = raw_val

        # Save cycle index for next iteration
        last_cycles[eid] = cyc

        predictions_to_insert.append((
            str(r.get("engine_id")),
            int(r.get("cycle")),
            float(r["HI_overall"]),
            float(r["HI_compressor"]),
            float(r["HI_turbine"]),
            float(r["HI_combustor"]),
            float(r["kf_health"]),
            float(r["uncertainty"]),
            float(r["RUL"]),
            float(r["thrust_N"])
        ))

    # 3. Perform batch inserts inside a single database transaction
    try:
        conn.executemany("""
            INSERT INTO predictions
                (engine_id, cycle, hi_overall, hi_compressor, hi_turbine, hi_combustor,
                 kf_health, uncertainty, rul, thrust_n)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, predictions_to_insert)

        for eid, targets in akf_to_update.items():
            for t, s in targets.items():
                conn.execute("""
                    INSERT INTO akf_multi_states (engine_id, target, x, P, Q, epoch, cold_start_active, cold_start_remaining, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    ON CONFLICT(engine_id, target) DO UPDATE SET
                        x=excluded.x, P=excluded.P, Q=excluded.Q, epoch=excluded.epoch,
                        cold_start_active=excluded.cold_start_active,
                        cold_start_remaining=excluded.cold_start_remaining,
                        updated_at=excluded.updated_at
                """, (eid, t, s["x"], s["P"], s["Q"], s["epoch"],
                       int(s["cold_start_active"]), s["cold_start_cycles_remaining"]))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error persisting batch predictions: {e}")
    finally:
        conn.close()

    return JSONResponse({"predictions": results})

@app.post("/maintenance/reset")
async def maintenance_reset(
    engine_id: str = Form(...),
    technician_name: str = Form(...),
    maintenance_type: str = Form(...),
    notes: str = Form(""),
    test_cell_file: UploadFile = File(None)
):
    """
    Manual post-maintenance health reset.
    
    Method 2 (Test Cell): Upload a test_cell_file CSV with sensor readings from
    the ground test rig. The physics pipeline computes the objective post-overhaul
    health from first principles. No human health estimate is accepted.
    
    Method 3 (Cold Start): No file uploaded. AKF uncertainty is widened to 30%.
    The system learns the new health from the first 10 live post-maintenance flights.
    Fault alarms are suppressed during the cold start window.
    """
    method_used = "cold_start"
    computed_health = None

    # FastAPI returns an empty UploadFile object even when no file is sent;
    # treat it as absent if the filename is empty/None
    has_file = test_cell_file is not None and bool(test_cell_file.filename)

    if has_file:
        # ── Method 2: Compute health from test cell physics ──
        contents = await test_cell_file.read()
        try:
            tc_df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not parse test cell CSV: {e}")

        # Rename columns and run physics pipeline
        tc_df = tc_df.rename(columns=COL_RENAME)
        tc_df["engine_id"] = tc_df["engine_id"].astype(str)
        tc_df = compute_features_and_predict(tc_df, mode="batch")

        # Use median HI_overall as the objective post-overhaul health
        computed_health = float(tc_df["HI_overall"].median())

        # Hard state injection with low uncertainty (we trust the test cell)
        akf_states[engine_id] = {
            "x": computed_health,
            "P": 0.002,
            "Q": 1e-5,
            "epoch": akf_states.get(engine_id, {}).get("epoch", 0) + 1,
            "cold_start_cycles_remaining": 0,
            "cold_start_active": False,
        }
        method_used = "test_cell"
    else:
        # ── Method 3: Cold Start Protocol ──
        akf_states[engine_id] = {
            "x": akf_states.get(engine_id, {}).get("x", 1.0),
            "P": 0.09,
            "Q": 1e-5,
            "epoch": akf_states.get(engine_id, {}).get("epoch", 0) + 1,
            "cold_start_cycles_remaining": 10,
            "cold_start_active": True,
        }

    # Persist AKF state to DB (survives server restart)
    save_akf_state(engine_id)

    # Log the maintenance event permanently
    health_before = None
    conn = get_db()
    prev = conn.execute(
        "SELECT hi_overall FROM predictions WHERE engine_id=? ORDER BY predicted_at DESC LIMIT 1",
        (engine_id,)
    ).fetchone()
    if prev:
        health_before = prev["hi_overall"]
    conn.execute(
        "INSERT INTO maintenance_logs (engine_id, technician, maintenance_type, method, health_before, health_after) VALUES (?, ?, ?, ?, ?, ?)",
        (engine_id, technician_name, maintenance_type, method_used, health_before, computed_health)
    )
    conn.commit()
    conn.close()

    return JSONResponse({
        "status": "ok",
        "engine_id": engine_id,
        "method_used": method_used,
        "computed_health": computed_health,
        "health_before": health_before,
        "cold_start_active": akf_states[engine_id]["cold_start_active"],
        "cycles_until_nominal": akf_states[engine_id]["cold_start_cycles_remaining"],
        "maintenance_type": maintenance_type,
        "logged_by": technician_name,
        "epoch": akf_states[engine_id]["epoch"],
        "message": (
            f"Engine {engine_id} reset via test cell. Health: {computed_health:.3f}"
            if method_used == "test_cell"
            else f"Engine {engine_id} cold start activated. Fault detection suppressed for 10 cycles."
        )
    })


@app.get("/maintenance/status/{engine_id}")
def maintenance_status(engine_id: str):
    """Returns the current AKF state and cold start status for an engine."""
    state = akf_states.get(engine_id)
    if state is None:
        return JSONResponse({"engine_id": engine_id, "akf_initialized": False})
    return JSONResponse({
        "engine_id": engine_id,
        "akf_initialized": True,
        "current_health_estimate": state["x"],
        "uncertainty_std": float(state["P"] ** 0.5),
        "cold_start_active": state["cold_start_active"],
        "cycles_until_nominal": state["cold_start_cycles_remaining"],
        "epoch": state["epoch"],
    })


# ──────────────────────── Engine Registry Endpoints ────────────────────────

@app.post("/engines/register")
def register_engine(
    engine_id:      str  = Form(...),
    serial_number:  str  = Form(""),
    model:          str  = Form(""),
    aircraft_reg:   str  = Form(""),
    position:       str  = Form(""),
    base:           str  = Form(""),
    cycles_at_reg:  int  = Form(0),
    last_overhaul:  str  = Form(""),
    warn_threshold: float = Form(0.82),
    crit_threshold: float = Form(0.70),
):
    """Register a new engine in the fleet database."""
    conn = get_db()
    try:
        conn.execute("""
            INSERT INTO engines
                (engine_id, serial_number, model, aircraft_reg, position, base,
                 cycles_at_reg, last_overhaul, warn_threshold, crit_threshold)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(engine_id) DO UPDATE SET
                serial_number=excluded.serial_number, model=excluded.model,
                aircraft_reg=excluded.aircraft_reg, position=excluded.position,
                base=excluded.base, cycles_at_reg=excluded.cycles_at_reg,
                last_overhaul=excluded.last_overhaul,
                warn_threshold=excluded.warn_threshold, crit_threshold=excluded.crit_threshold
        """, (engine_id, serial_number, model, aircraft_reg, position, base,
               cycles_at_reg, last_overhaul, warn_threshold, crit_threshold))
        conn.commit()
    finally:
        conn.close()

    # Initialize AKF state for this engine at healthy baseline
    if engine_id not in akf_states:
        akf_states[engine_id] = {
            "x": 1.0, "P": 0.01, "Q": 1e-5,
            "epoch": 1, "cold_start_active": True,
            "cold_start_cycles_remaining": 10,
        }
        save_akf_state(engine_id)

    return JSONResponse({"status": "ok", "engine_id": engine_id,
                         "message": f"Engine {engine_id} registered successfully."})


@app.get("/engines")
def list_engines():
    """List all registered engines with their current AKF health estimate."""
    conn = get_db()
    rows = conn.execute("SELECT * FROM engines ORDER BY registered_at DESC").fetchall()
    conn.close()
    result = []
    for row in rows:
        eid = row["engine_id"]
        state = akf_states.get(eid, {})
        result.append({
            **{k: row[k] for k in row.keys()},
            "current_health": state.get("x"),
            "cold_start_active": state.get("cold_start_active", False),
        })
    return JSONResponse({"engines": result})


@app.get("/engines/{engine_id}/history")
def engine_history(engine_id: str, limit: int = 300):
    """Return the prediction history and maintenance log for one engine."""
    conn = get_db()
    preds = conn.execute(
        "SELECT cycle, hi_overall as HI_overall, hi_compressor as HI_compressor, "
        "hi_turbine as HI_turbine, hi_combustor as HI_combustor, "
        "kf_health, rul as RUL, thrust_n as thrust_N, predicted_at "
        "FROM predictions WHERE engine_id=? ORDER BY cycle ASC LIMIT ?",
        (engine_id, limit)
    ).fetchall()
    logs = conn.execute(
        "SELECT maintenance_type, technician, method, health_before, health_after, logged_at "
        "FROM maintenance_logs WHERE engine_id=? ORDER BY logged_at ASC",
        (engine_id,)
    ).fetchall()
    meta = conn.execute(
        "SELECT * FROM engines WHERE engine_id=?", (engine_id,)
    ).fetchone()
    conn.close()

    return JSONResponse({
        "engine_id": engine_id,
        "metadata": dict(meta) if meta else {},
        "predictions": [dict(p) for p in preds],
        "maintenance_logs": [dict(l) for l in logs],
    })


# ──────────────────────── Static Files ────────────────────────
# This must come AFTER all API route definitions so that
# /health, /predict/single, /predict/batch are matched first.
public_dir = os.path.join(BASE_DIR, "public")
if os.path.isdir(public_dir):
    app.mount("/", StaticFiles(directory=public_dir, html=True), name="static")


# ──────────────────────── Entry Point ────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info",
    )
