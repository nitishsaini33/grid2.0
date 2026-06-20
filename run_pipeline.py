"""
run_pipeline.py — Master Orchestrator
Usage:
    python run_pipeline.py                  # Full end-to-end pipeline
    python run_pipeline.py --mode eda       # EDA only
    python run_pipeline.py --mode train     # Train only (requires features)
    python run_pipeline.py --mode predict   # Single prediction demo
"""
import warnings
warnings.filterwarnings("ignore")

import argparse, sys, time, json
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from utils import get_logger, DATA_PROC, print_section
log = get_logger("run_pipeline")


def step_preprocessing():
    from src.preprocessing import run_preprocessing
    return run_preprocessing()

def step_feature_engineering(df):
    from src.feature_engineering import run_feature_engineering
    return run_feature_engineering(df)

def step_eda(full_df):
    from src.eda import run_eda
    run_eda(full_df)

def step_optuna(X, Y):
    from src.optuna_tuning import run_optuna_tuning
    return run_optuna_tuning(X, Y)

def step_classification(X, Y, best_params=None):
    from src.train_classification import run_classification_training
    return run_classification_training(X, Y, best_params)

def step_regression(X, Y, best_params=None):
    from src.train_regression import run_regression_training
    return run_regression_training(X, Y, best_params)

def step_predict_demo():
    from src.predict import predict_event
    print_section("DEMO PREDICTIONS")
    scenarios = [
        dict(event_cause="accident",    priority="High", requires_road_closure=True,
             latitude=12.9716, longitude=77.5946),
        dict(event_cause="vip_movement",priority="High", requires_road_closure=True,
             event_type="planned", latitude=12.9352, longitude=77.6245),
        dict(event_cause="vehicle_breakdown", priority="Low", requires_road_closure=False,
             latitude=12.9200, longitude=77.5800),
        dict(event_cause="public_event", priority="High", requires_road_closure=False,
             latitude=13.0200, longitude=77.5700),
    ]
    for s in scenarios:
        predict_event(**s)
        print()


def run_full_pipeline():
    t0 = time.time()
    print_section("ASTRAM ML PIPELINE - FULL RUN")

    # 1. Preprocessing
    print_section("STEP 1 — Preprocessing")
    df = step_preprocessing()

    # 2. Feature Engineering
    print_section("STEP 2 — Feature Engineering")
    X, Y = step_feature_engineering(df)
    import pandas as pd
    full_df = pd.read_parquet(DATA_PROC / "full_featured.parquet")

    # 3. EDA
    print_section("STEP 3 — EDA")
    step_eda(full_df)

    # 4. Optuna (60 trials)
    print_section("STEP 4 — Optuna Hyperparameter Tuning (60 trials)")
    best_params = step_optuna(X, Y)

    # 5. Classification
    print_section("STEP 5 — Classification Models")
    clf_results = step_classification(X, Y, best_params)

    # 6. Regression
    print_section("STEP 6 — Regression Models")
    reg_results = step_regression(X, Y, best_params)

    # 7. Demo predictions
    print_section("STEP 7 — Demo Predictions")
    step_predict_demo()

    elapsed = time.time() - t0
    print_section(f"PIPELINE COMPLETE  ({elapsed/60:.1f} minutes)")

    # Print summary
    print("\n[CLASSIFICATION SUMMARY]:")
    for tgt, info in clf_results.items():
        tm = info.get("test_metrics", {})
        print(f"  {tgt:25s}: best={info.get('best_model','?'):20s} "
              f"bal_acc={tm.get('balanced_accuracy',0)*100:.1f}%  "
              f"f1={tm.get('f1',0):.3f}")

    print("\n[REGRESSION SUMMARY]:")
    if reg_results:
        tm = reg_results.get("test_metrics",{})
        print(f"  duration_min: best={reg_results.get('best_model','?')}  "
              f"MAE={tm.get('MAE',0):.1f}min  R²={tm.get('R2',0):.3f}")



# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASTRAM ML Pipeline Runner")
    parser.add_argument("--mode", default="full",
                        choices=["full","eda","train","optuna","predict"])
    args = parser.parse_args()

    if args.mode == "full":
        run_full_pipeline()

    elif args.mode == "eda":
        df = step_preprocessing()
        X, Y = step_feature_engineering(df)
        import pandas as pd
        full_df = pd.read_parquet(DATA_PROC / "full_featured.parquet")
        step_eda(full_df)

    elif args.mode == "optuna":
        import pandas as pd
        X = pd.read_parquet(DATA_PROC / "features.parquet")
        Y = pd.read_parquet(DATA_PROC / "targets.parquet")
        step_optuna(X, Y)

    elif args.mode == "train":
        import pandas as pd
        X = pd.read_parquet(DATA_PROC / "features.parquet")
        Y = pd.read_parquet(DATA_PROC / "targets.parquet")
        best_params = {}
        p = BASE_DIR / "models" / "reports" / "optuna_best_params.json"
        if p.exists():
            with open(p) as f:
                best_params = json.load(f).get("best_params", {})
        step_classification(X, Y, best_params)
        step_regression(X, Y, best_params)

    elif args.mode == "predict":
        step_predict_demo()
