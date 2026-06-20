<div align="center">

# 🚦 ASTRAM — Traffic Intelligence Platform

### Flipkart GRIDLOCK Hackathon 2.0

**Problem Statement:** *Event-Driven Congestion (Planned & Unplanned)*

> *How can historical and real-time data be used to forecast event-related traffic impact and recommend optimal manpower, barricading, and diversion plans?*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)

</div>

---

## 📌 Problem Statement

Political rallies, festivals, sports events, construction activities, and sudden gatherings create localized traffic breakdowns. Today:

- **Event impact is not quantified in advance** — authorities have no way to know how bad a traffic jam will be before it happens.
- **Resource deployment is experience-driven** — how many officers to send, whether to set up barricades, or reroute traffic is decided by gut-feel.
- **No post-event learning system** — there's no feedback loop to improve future predictions from past outcomes.

**ASTRAM solves all three.**

Given an incoming traffic event (accident, VIP movement, protest, construction, etc.), ASTRAM instantly predicts:

| What We Predict | How We Predict It | Output |
|---|---|---|
| **Congestion Severity** | Multi-class Classification (0–3) | Low / Moderate / High / Severe |
| **Event Duration** | Regression (minutes) | Estimated time-to-clear |
| **Manpower Required** | Multi-class Classification (0–4) | 2 / 4 / 8 / 12 / 20 officers |
| **Barricade Level** | Multi-class Classification (0–3) | None / Light (2) / Medium (5) / Heavy (10) |
| **Diversion Needed** | Binary Classification | Yes / No |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ASTRAM — Full System                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   ┌──────────────────┐    ┌───────────────────┐   ┌─────────────────┐ │
│   │   ML Pipeline    │    │  FastAPI Backend   │   │ React Dashboard │ │
│   │  (src/)          │───▶│  (backend/)        │◀──│ (frontend/)     │ │
│   │                  │    │                    │   │                 │ │
│   │ • Preprocessing  │    │ • /api/kpis        │   │ • Overview KPIs │ │
│   │ • Feature Eng.   │    │ • /api/predict     │   │ • Hotspot Map   │ │
│   │ • Optuna Tuning  │    │ • /api/map         │   │ • Predict UI    │ │
│   │ • Classification │    │ • /api/charts/*    │   │ • Model Results │ │
│   │ • Regression     │    │ • /api/history     │   │ • Feature Imp.  │ │
│   │ • Post-Event     │    │ • /api/models/*    │   │ • History Table │ │
│   └──────────────────┘    └───────────────────┘   └─────────────────┘ │
│          │                        │                                    │
│          ▼                        ▼                                    │
│   ┌──────────────────┐    ┌───────────────────┐                       │
│   │ models/saved/    │    │   PostgreSQL DB    │                       │
│   │ (5 .joblib files)│    │  (traffic_events)  │                       │
│   └──────────────────┘    └───────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 ML Pipeline — Deep Dive

The entire pipeline is orchestrated by `run_pipeline.py` and runs through 7 automated stages:

### Stage 1: Preprocessing (`src/preprocessing.py`)
- Parses 6 datetime columns (`start_datetime`, `end_datetime`, `closed_datetime`, `resolved_datetime`, etc.)
- **Derives event duration** using a priority chain: `closed_datetime > resolved_datetime > end_datetime`, with 99th percentile capping to remove outliers
- **Builds 5 target variables** from raw data using domain-specific rules:
  - `congestion_severity` is computed from `event_cause` base score + `priority` modifier + `unplanned` flag + `duration > 120min` flag, then binned into 4 classes. Road closures automatically escalate to Severe (3).
  - `manpower_req` maps severity 0→2 officers, 1→4, 2→8, 3→12, with road closure + severe bumping to class 4 (20 officers).
  - `barricade_req` distinguishes None (0) / Light (2 units) / Medium (5) / Heavy (10) based on severity and closure.
  - `diversion_req` is flagged for road closures or high-priority events from causes like `congestion`, `construction`, `vip_movement`, `procession`, `protest`, `public_event`, `accident`.
- **Zone imputation** via KMeans clustering on lat/lon coordinates — missing zones are predicted from spatial proximity to known zones
- Drops 16 useless metadata columns and applies IQR-based outlier clipping on spatial coordinates

### Stage 2: Feature Engineering (`src/feature_engineering.py`)
Constructs **46 features** across 5 categories:

| Category | Features | Examples |
|---|---|---|
| **Temporal (17)** | Hour, minute, day, weekday, month, week-of-year, `is_weekend`, `is_peak`, `is_night`, `is_morning_peak`, `is_evening_peak`, `time_since_midnight`, cyclical sine/cosine for hour/weekday/month | `hour_sin = sin(2π × hour/24)` |
| **Spatial (4)** | Haversine distance from Bangalore city center (12.9716°N, 77.5946°E), KMeans spatial cluster (k=15), cluster density, junction presence | `dist_from_center_km` |
| **Categorical (12)** | Label encoding for zone, police_station, corridor, event_cause, veh_type; binary flags for event_type, priority, road_closure, authentication; frequency encoding for corridor; target encoding of cause→severity | `cause_target_enc` |
| **Historical (7)** | Rolling 1-day/7-day/30-day event counts per zone; previous event severity at same zone; mean severity per zone; avg duration per cause×zone and per police station | `zone_events_7d` |
| **Interaction (8)** | Cause×Peak, Cause×Weekend, Zone×Priority, Distance×Priority, Cluster×Cause, Hour×EventType, Closure×Priority, Severity×ClusterDensity | `closure_x_priority` |

### Stage 3: EDA (`src/eda.py`)
Generates 10 analysis plots saved to `models/reports/`:
- Missing values heatmap, feature correlation matrix, event cause distribution
- Congestion severity analysis (distribution, by cause, by hour-of-day)
- Duration distribution and median duration by event cause
- Geographic scatter plot of events colored by severity (Bangalore)
- Zone-wise mean severity and event count comparison
- Numeric feature distributions

### Stage 4: Optuna Hyperparameter Tuning (`src/optuna_tuning.py`)
- Runs **60 trials** using TPE (Tree-structured Parzen Estimator) sampler with Median Pruner
- Tunes **3 gradient boosting models** on the primary target (`congestion_severity`):
  - **XGBoost**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda
  - **LightGBM**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_samples, num_leaves, reg_alpha, reg_lambda
  - **CatBoost**: iterations, depth, learning_rate, l2_leaf_reg, border_count, bagging_temperature
- Uses **5-fold Stratified Cross Validation** with `balanced_accuracy` scoring
- Uses 70% of data for faster tuning, then applies best params to full training

### Stage 5: Classification Training (`src/train_classification.py`)
Trains on **4 targets** independently: `congestion_severity`, `manpower_req`, `barricade_req`, `diversion_req`

**9 models trained per target:**
1. RandomForest (balanced class weights)
2. ExtraTrees (balanced class weights)
3. HistGradientBoosting
4. XGBoost (with Optuna params)
5. LightGBM (balanced + Optuna params)
6. CatBoost (auto balanced weights)
7. BalancedRandomForest (from imbalanced-learn)
8. **Soft Voting Ensemble** (top-3 models)
9. **Stacking Ensemble** (top-3 models + LogisticRegression meta-learner)

- **SMOTE oversampling** applied to handle class imbalance
- **70/15/15 train/val/test split** with stratification
- Saves confusion matrix plots per target to `models/reports/`

### Stage 6: Regression Training (`src/train_regression.py`)
Trains on `duration_min` using 7 regressors:
1. RandomForest, ExtraTrees, GradientBoosting
2. XGBoost, LightGBM, CatBoost
3. **Stacking Regressor** (top-3 + Ridge meta-learner)

- Generates residual plots and predicted-vs-actual scatter plots
- Runs **5-fold Cross Validation** on the best model

### Stage 7: Prediction (`src/predict.py`)
- Reconstructs all 46 features from raw event parameters in real-time
- Loads 5 trained `.joblib` models and runs inference
- Falls back to rule-based heuristics if models aren't available
- Outputs a structured prediction with severity, duration, manpower count, barricade level, and diversion recommendation

---

## 📊 Model Performance (Actual Results)

### Classification Results

| Target | Best Model | Accuracy | Balanced Acc | F1 | ROC-AUC | Cohen's κ |
|---|---|---|---|---|---|---|
| **Congestion Severity** | StackingEnsemble | 100% | 100% | 1.000 | 1.000 | 1.000 |
| **Manpower Required** | HistGradBoost | 99.9% | 99.8% | 0.999 | 1.000 | 0.999 |
| **Barricade Level** | HistGradBoost | 100% | 100% | 1.000 | 1.000 | 1.000 |
| **Diversion Required** | RandomForest | 100% | 100% | 1.000 | 1.000 | 1.000 |

### Regression Results (Duration Prediction)

| Metric | Value |
|---|---|
| **Best Model** | StackingRegressor |
| **R²** | 0.451 |
| **CV R² (5-fold)** | 0.383 ± 0.046 |

### Optuna Tuning Scores (5-fold CV Balanced Accuracy)

| Model | Best Score |
|---|---|
| XGBoost | 99.81% |
| LightGBM | 99.91% |
| CatBoost | 99.80% |

---

## 🖥️ Dashboard — 6 Interactive Pages

### 1. 📊 Overview
Real-time KPI cards (Total Events, Avg Severity, Median Duration, Severe Events, Road Closures, Closed %) and 3 interactive Recharts visualizations: Event Cause Distribution, Zone-wise Mean Severity, and Duration Distribution.

### 2. 🗺 Hotspot Map
Full-screen Leaflet map centered on Bangalore (12.97°N, 77.59°E) using CartoDB dark tiles. Each of the last 1,000 events is rendered as a `CircleMarker` colored by severity (Green→Yellow→Orange→Red). Popups show cause, zone, and severity class.

### 3. 🤖 Predict Event
Dispatcher-facing form with inputs for Event Cause (13 categories), Event Type (planned/unplanned), Priority (High/Low), Road Closure toggle, GPS coordinates, and Hour-of-Day slider. Submits to `POST /api/predict` and displays the full resource recommendation: severity badge, duration, officer count, barricade class, and diversion status. Each prediction is persisted to the database.

### 4. 📈 Model Results
Grouped bar chart comparing Balanced Accuracy across all 9 models for each of the 4 classification targets. Detailed metrics table with Accuracy, Balanced Accuracy, F1, AUC, and Cohen's Kappa. Regression section showing best model, MAE, R², and the residual scatter plot.

### 5. 🔍 Feature Analysis
Horizontal bar chart of the Top-20 feature importances from the severity model. Embedded correlation matrix heatmap from the EDA pipeline.

### 6. 📋 Historical Data
Full-featured data table powered by `@tanstack/react-table` showing the last 500 events with columns: Start Time, Cause, Type, Priority (with color badges), Zone, Police Station, Severity (color-coded), Duration, and Status (open/closed).

---

## 🔄 Post-Event Learning System (`src/post_event_learning.py`)

ASTRAM implements a **closed-loop feedback system** — the piece that directly addresses the problem statement's requirement for post-event learning:

1. **`log_prediction()`** — Stores every prediction (severity, duration, officers, barricades, diversion) alongside the input event parameters
2. **`record_actual()`** — After the event is resolved, dispatchers record what actually happened (actual severity, actual duration, actual officers deployed)
3. **`compute_drift_report()`** — Compares predictions vs. actuals, computes accuracy and MAE, and flags **model drift** if severity accuracy drops below 70%, recommending retraining

---

## 🗂️ Complete Project Structure

```
ASTRAM/
├── src/                          # ML Pipeline
│   ├── preprocessing.py          # Data cleaning, target engineering, imputation
│   ├── feature_engineering.py    # 46 features: temporal, spatial, categorical, historical, interaction
│   ├── eda.py                    # 10 EDA visualizations
│   ├── optuna_tuning.py          # 60-trial hyperparameter optimization (XGB, LGB, CatBoost)
│   ├── train_classification.py   # 9 models × 4 targets with SMOTE + ensemble stacking
│   ├── train_regression.py       # 7 models for duration prediction with stacking
│   ├── predict.py                # Real-time inference with 46-feature reconstruction
│   ├── post_event_learning.py    # Feedback loop and drift detection
│   └── utils.py                  # Shared paths, mappings, I/O helpers
│
├── backend/                      # FastAPI Web Server
│   ├── main.py                   # 8 REST endpoints (KPIs, charts, map, predict, history, models)
│   ├── database.py               # SQLAlchemy engine (SQLite local / PostgreSQL production)
│   ├── models.py                 # ORM models: TrafficEvent (13 cols), ModelData (JSON store)
│   ├── migrate_to_postgres.py    # Bulk migration from parquet → PostgreSQL + model metrics
│   └── requirements.txt          # FastAPI, uvicorn, gunicorn, sqlalchemy, psycopg2-binary
│
├── frontend/                     # React + Vite + Tailwind CSS Dashboard
│   ├── src/
│   │   ├── api/client.js         # Axios client with env-based URL (VITE_API_BASE_URL)
│   │   ├── App.jsx               # 6-tab navigation with ASTRAM header
│   │   └── pages/
│   │       ├── Overview.jsx      # KPI cards + 3 Recharts visualizations
│   │       ├── MapView.jsx       # Leaflet map with severity-colored CircleMarkers
│   │       ├── Predict.jsx       # Event input form + prediction result cards
│   │       ├── Models.jsx        # Model comparison chart + metrics table
│   │       ├── Features.jsx      # Feature importance bar chart + correlation matrix
│   │       └── History.jsx       # TanStack React Table with 500 records
│   ├── vercel.json               # SPA rewrite rules for Vercel deployment
│   └── package.json              # React 19, Recharts, Leaflet, TanStack Table, Axios
│
├── models/
│   ├── saved/                    # 5 trained models (.joblib)
│   │   ├── clf_congestion_severity.joblib
│   │   ├── clf_manpower_req.joblib
│   │   ├── clf_barricade_req.joblib
│   │   ├── clf_diversion_req.joblib
│   │   └── reg_duration.joblib
│   └── reports/                  # JSON metrics + 15 analysis PNG plots
│
├── data/
│   ├── raw/                      # Original ASTRAM event dataset (CSV)
│   └── processed/                # Cleaned parquet, features, targets
│
├── run_pipeline.py               # Master orchestrator (7 stages)
├── render.yaml                   # Render Blueprint: PostgreSQL + FastAPI web service
├── requirements.txt              # ML pipeline dependencies (16 packages)
└── .gitignore                    # Excludes venv, __pycache__, *.csv, *.log, *.db
```

---

## 🚀 Local Development Setup

### Prerequisites
- Python 3.10+
- Node.js 18+

### 1. ML Pipeline (Train Models)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install ML dependencies
pip install -r requirements.txt

# Place the dataset at data/raw/Astram event data.csv

# Run the full pipeline (preprocessing → features → EDA → Optuna → classification → regression → predictions)
python run_pipeline.py --mode full
```

### 2. Backend (FastAPI)
```bash
pip install -r backend/requirements.txt

# Migrate training data + model metrics into the database
cd backend
python migrate_to_postgres.py

# Start the API server
python -m uvicorn main:app --reload --port 8000
```

### 3. Frontend (React Dashboard)
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

---

## 🌐 Production Deployment

| Component | Platform | Config |
|---|---|---|
| **Backend + Database** | Render | `render.yaml` (auto-provisions PostgreSQL + gunicorn web service) |
| **Frontend** | Vercel | Root Directory = `frontend`, Env: `VITE_API_BASE_URL` = Render URL |

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **ML / Data Science** | XGBoost, LightGBM, CatBoost, Scikit-Learn, Optuna, Imbalanced-Learn (SMOTE), Pandas, NumPy, Matplotlib, Seaborn |
| **Backend** | FastAPI, SQLAlchemy, PostgreSQL, Gunicorn + Uvicorn |
| **Frontend** | React 19, Vite 8, Tailwind CSS 4, Recharts, React-Leaflet, TanStack React Table, Axios |
| **Deployment** | Render (Backend + DB), Vercel (Frontend) |

---

## 👥 Team

Built for **Flipkart GRIDLOCK Hackathon 2.0**
