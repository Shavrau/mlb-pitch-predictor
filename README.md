# ⚾ MLB Pitch Predictor — Multimodal Statcast & Biomechanical AI

> **Next-Generation Multimodal Tactical & Biomechanical Pitch Prediction Platform**  
> Powered by **Databricks Serverless**, **Lakeflow Spark Declarative Pipelines**, **Computer Vision Pose Estimation (MediaPipe 3D)**, **SHAP Explainability**, and an interactive multi-page **Streamlit** application.

---

## 📌 Project Overview

**MLB Pitch Predictor** is an enterprise-grade multimodal artificial intelligence platform designed to analyze high-stakes pitcher-batter duels in Major League Baseball (MLB). By integrating real-time **Statcast physical telemetry** (count, baserunners, historical arsenal priors) with **computer vision biomechanics** (3D skeletal landmark tracking and release point pitch tipping analysis), the system predicts in real time which tactical pitch category will be delivered:

- 🚀 **Fastball** (*Four-Seam, Sinker, Cutter*)
- 🌪️ **Breaking Ball** (*Slider, Sweeper, Curveball, Knuckle Curve*)
- 🛑 **Offspeed** (*Changeup, Splitter*)

### Key Architectural Pillars

1. **Zero-Leakage Pre-Pitch Feature Engineering**: Strict intra-at-bat historical sequence partitioning (`(game_pk, at_bat_number)` ordered by `pitch_number`) and causally masked historical pitcher priors with Bayesian league smoothing—guaranteeing 0% future data leakage.
2. **Multimodal Fusion Engine**: Late-stage weighted fusion combining situational context from Statcast with 33 upper-body 3D anatomical landmarks extracted via computer vision at the exact moment of ball release.
3. **Computer Vision & Pitch Tipping Detection**: Deep biomechanical analysis diagnosing arm slot angles, elbow flexion, release height, and trunk tilt, powered by SHAP tree explainability to detect physical "tells" before the ball leaves the pitcher's hand.
4. **Serverless Infrastructure as Code (IaC)**: End-to-end declarative asset bundles (DABs), Unity Catalog governance (Bronze streaming tables, Silver materialized views, Gold biomechanical tables), and Lakeflow Jobs on Databricks Serverless.
5. **Real-Time Model Serving**: Low-latency custom MLflow PyFunc model served on Databricks Model Serving Serverless with scale-to-zero capabilities and automatic offline fallback resilience.
6. **Robust Testing & Engineering Standards**: **208 automated unit and integration tests** passing with 100% success rate across all pipelines, computer vision extractors, and UI components.

---

## 🗂️ Repository Structure

```text
pitch_predictor/
├── .agents/                        # Autonomous AI agent guidelines, invariants, and operational skills
├── config/
│   └── config.yaml                 # Catalog, schema, volume, and serving endpoint definitions
├── notebooks/
│   ├── 00_setup_environment.py     # Idempotent Unity Catalog metastore and schema initialization
│   ├── 01_data_ingestion.py        # Statcast data acquisition via PyBaseball into Delta Bronze
│   ├── 02_automl_run.py            # Databricks AutoML execution with temporal date-based splitting
│   ├── 03_model_inference.py       # Batch scoring validation and MLflow model registry verification
│   └── 04_video_biomechanics_pipeline.py # Lakeflow Job task for Savant video and 3D pose ingestion
├── pages/
│   ├── 00_home_apresentacao.py     # Executive overview, live pitch prediction & star player profiles
│   ├── 02_scout_report.py          # MLB Scouting Report, 4x3 12-count matrix & tactical game plans
│   ├── 03_minigame.py              # Retro Arcade Duel: You vs. AI on the Mound (8-bit sound & RPG stats)
│   └── 04_pitch_vision_lab.py      # Computer Vision Lab: MediaPipe 3D pose overlay, tipping & ablation
├── resources/
│   ├── pitch_predictor_app.yml     # Databricks App serverless deployment specification
│   ├── pitch_predictor_endpoint.yml# Databricks Model Serving Serverless endpoint configuration
│   ├── pitch_predictor_job.yml     # 6-task orchestrated Lakeflow Serverless workflow
│   └── pitch_predictor_pipeline.pipeline.yml # Lakeflow Spark Declarative Pipeline definition
├── scripts/
│   ├── download_pitch_videos.py    # Incremental Baseball Savant video scraper and manifest builder
│   ├── extract_pitch_poses.py      # MediaPipe 3D landmark extraction and kinematic angle computation
│   ├── generate_all_15_release_frames.py # Statcast neon skeletal release frame calibration
│   ├── ingest_multiple_pitchers.py # Multi-pitcher incremental ingestion (Skenes, Burnes, Webb, Sale, Yamamoto)
│   ├── register_multimodal_model.py # MLflow PyFunc Unity Catalog registration with @prod alias
│   ├── setup_unity_catalog.py      # Idempotent Unity Catalog setup CLI utility
│   ├── setup_unity_catalog.sql     # Parametrized DDL templates (${catalog}, ${schema})
│   ├── train_multimodal_model.py   # Hybrid model training and multimodal lift ablation study
│   ├── train_tipping_model.py      # Random Forest tipping classifier training with SHAP analysis
│   ├── verify_browser_dom.py       # Automated Playwright DOM inspector for Streamlit views
│   ├── verify_minigame_browser.py  # End-to-end headless arcade minigame browser test
│   └── verify_ui_server.py         # HTTP health check probe for the Streamlit server
├── src/pitch_predictor/            # Core reusable Python package
│   ├── __init__.py                 # Package initialization
│   ├── config.py                   # Dynamic configuration loader & Databricks WorkspaceClient
│   ├── fallback_engine.py          # Unified offline inference engine & tactical/multimodal fallback
│   ├── features.py                 # Silver feature engineering, mappings, and zero-leakage transforms
│   ├── ingestion.py                # Statcast data acquisition, schema validation, and sanitization
│   ├── minigame_engine.py          # Turn-based arcade state machine, RPG attributes, and pitch tiers
│   ├── multimodal.py               # Deterministic multimodal fusion, ablation study, and lift evaluation
│   ├── scenarios.py                # Single source of truth for official MLB pitcher and batter profiles
│   ├── scout_analytics.py          # 12-count matrix aggregations, go-to pitch, and put-away heuristics
│   ├── serving_client.py           # Resilient Databricks Model Serving client with exponential backoff
│   ├── serving_model.py            # MLflow Custom PyFunc wrapper with schema imputation & alignment status
│   ├── training.py                 # Databricks AutoML orchestrator and MLflow experiment tracker
│   ├── ui_components.py            # SVG visual components (Baseball Field, 3x3 Strike Zone, Neon Cards)
│   ├── utils.py                    # Formatted logging and local Spark session bootstrap
│   └── vision/                     # Computer Vision & Biomechanics submodule
│       ├── __init__.py             # Unified vision exports and graceful dependency detection
│       ├── incremental_pipeline.py # Serverless Databricks video-to-landmarks ETL orchestrator
│       ├── overlay.py              # Real-time Statcast neon skeletal rendering over broadcast video
│       ├── pose_extractor.py       # 33 MediaPipe 3D landmark extractor and joint angle calculus
│       ├── tipping_model.py        # PitchTippingClassifier & SHAP TreeExplainer diagnostics
│       └── video_ingestion.py      # Baseball Savant MP4 downloader and UC Volume sync
├── tests/                          # 208 automated unit and integration tests (pytest)
│   ├── test_automl_training.py     # AutoML orchestration and temporal splitting tests
│   ├── test_config.py              # Configuration resolution and catalog path tests
│   ├── test_fallback_engine.py     # Offline fallback heuristics and multimodal consistency tests
│   ├── test_features_multimodal.py # Multimodal dataset fusion and feature alignment tests
│   ├── test_incremental_pipeline.py# Video ETL pipeline incremental state and manifest tests
│   ├── test_ingestion.py           # Statcast raw data validation and bounds checking
│   ├── test_minigame_engine.py     # Arcade state machine, tiers, batter timing, and duel logic
│   ├── test_multimodal.py          # Hybrid classifier inference, ablation study, and lift metrics
│   ├── test_overlay.py             # Skeletal overlay rendering and video frame canvas tests
│   ├── test_pose_extractor.py      # MediaPipe landmark extraction, angle physics, and arm slot tests
│   ├── test_scenarios.py           # Pitcher catalog, priors, and single-source-of-truth tests
│   ├── test_scout_analytics.py     # 4x3 12-count matrix calculations and game plan tests
│   ├── test_serving_client.py      # REST endpoint client, retries, and offline mock fallback tests
│   ├── test_serving_model.py       # MLflow PyFunc input/output schema and imputation tests
│   ├── test_silver_features.py     # Intra-at-bat sequence isolation and zero future leakage tests
│   ├── test_streamlit_app.py       # Headless Streamlit page rendering and navigation tests
│   ├── test_tipping_model.py       # Biomechanical pitch tipping model and SHAP attribution tests
│   ├── test_ui_components.py       # SVG integrity and visual card component tests
│   └── test_video_ingestion.py     # Video scraper, URL resolver, and volume upload tests
├── AGENTS.md                       # Architectural standards, invariants, and developer operational guide
├── app.py                          # Streamlit multi-page application entrypoint
├── app.yaml                        # Databricks Apps runtime packaging specification
├── databricks.yml                  # Databricks Asset Bundle (DAB) declaration
├── pyproject.toml                  # Python package specifications and build metadata
├── requirements.txt                # Pinned production and development dependencies
└── sample_inference_request.json   # Sample payload for Model Serving endpoint testing
```

---

## 🚀 Interactive Multi-Page Application

The web interface is built with **Streamlit** and features 4 high-impact modules:

### 1. 🏠 Executive Overview & Live Predictor (`pages/00_home_apresentacao.py`)
- Live pitch prediction tracker with official MLB stars (**Paul Skenes**, **Corbin Burnes**, **Logan Webb**, **Chris Sale**, **Yoshinobu Yamamoto**).
- Dynamic count, baserunner, and inning controls with real-time probability dials and tactical contextual explanations.
- Direct navigation links to specialized analytical and arcade sub-modules.

### 2. 📋 MLB Scouting Report & 4x3 Matrix (`pages/02_scout_report.py`)
- Full tactical dossier mapping pitcher tendencies across all 12 baseball counts (`0-0` to `3-2`).
- Algorithmic detection of "Go-To Pitch" (*Confidence*) and "Put-Away Pitch" (*Strikeout finisher*).
- Automated batter game plan recommendations tailored to pitcher profiles and count leverage.

### 3. 🎮 Retro Arcade Duel: You vs. AI (`pages/03_minigame.py`)
- 90s arcade aesthetic with 8-bit / 16-bit styling, *Press Start 2P* typography, and neon CRT scanlines.
- 5 vertical star pitcher selection cards with official MLB headshots and tier-calibrated arsenals (Tier S, A, B).
- Real MLB batters (**Shohei Ohtani**, **Aaron Judge**, **Juan Soto**, **Mookie Betts**, **Bryce Harper**).
- Batter Heat Timing meter (anti-spam system) and live synthesized chiptune sound effects via browser Web Audio API.

### 4. 🔬 Computer Vision & Biomechanics Lab (`pages/04_pitch_vision_lab.py`)
- High-definition video player featuring official MLB broadcast footage synchronized at release points.
- 33-landmark 3D MediaPipe skeletal overlay in neon Statcast styling with automatic left/right arm slot detection.
- Biomechanical pitch tipping diagnostics powered by SHAP feature attribution (elbow angle, release height, trunk tilt).
- Interactive Ablation Study measuring the information lift of fusing physical mechanics with Statcast telemetry.

---

## 🛠️ Local Setup & Execution

### 1. Configure Python Virtual Environment
```powershell
# Create and activate virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

### 2. Run Test Suite (208 tests)
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

### 3. Launch Streamlit Application
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Databricks Serverless Deployment (DABs)

### 1. Authentication & Bundle Validation
```bash
# Authenticate with Databricks CLI
databricks auth login --profile dev

# Validate the Databricks Asset Bundle
databricks bundle validate -p dev
```

### 2. Deploy Cloud Resources
```bash
# Deploy notebooks, declarative pipelines, Lakeflow jobs, and Databricks App
databricks bundle deploy -p dev
```

### 3. Run Serverless Lakeflow Workflow
```bash
# Execute end-to-end Lakeflow Job (Setup -> Ingestion -> DLT -> Pose Extraction -> AutoML -> Inference)
databricks bundle run pitch_predictor_job -p dev
```

---

## 📄 License & Attribution

This project is licensed under the Apache 2.0 License.  
Statcast telemetry and video footage are provided courtesy of MLB Advanced Media / Baseball Savant for analytical and research purposes.
