"""
train_regression.py — Duration Regression Training
Target: duration_min
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import (get_logger, DATA_PROC, save_model, save_json,
                   print_section, print_metrics, REPORTS_DIR)

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor, StackingRegressor)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

log = get_logger("train_regression")


def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_reg(model, X_test, y_test, name: str) -> dict:
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)
    metrics = {
        "MAE":  mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAPE": mape(np.array(y_test), y_pred),
        "R2":   r2_score(y_test, y_pred),
    }
    print_metrics(metrics, title=name)
    return metrics


def build_regressors(seed: int = 42):
    return {
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=seed),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbosity=0),
        "LightGBM": LGBMRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbosity=-1, n_jobs=-1),
        "CatBoost": CatBoostRegressor(
            iterations=300, depth=6, learning_rate=0.05,
            random_seed=seed, verbose=0, thread_count=-1),
    }


def run_regression_training(X: pd.DataFrame, Y: pd.DataFrame,
                             best_params: dict = None):
    print_section("REGRESSION PIPELINE — duration_min")

    if "duration_min" not in Y.columns:
        log.warning("duration_min not in targets, skipping regression")
        return {}

    y = Y["duration_min"].dropna()
    X_ = X.loc[y.index]

    # 70/15/15 split
    X_trv, X_test, y_trv, y_test = train_test_split(
        X_, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trv, y_trv, test_size=0.15/0.85, random_state=42)

    log.info(f"Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    regs = build_regressors()

    # Apply Optuna params if available
    if best_params:
        if "XGBoost" in best_params:
            regs["XGBoost"].set_params(**best_params["XGBoost"])
        if "LightGBM" in best_params:
            regs["LightGBM"].set_params(**best_params["LightGBM"])

    results = {}
    trained = {}

    for name, reg in regs.items():
        log.info(f"  Fitting {name}...")
        try:
            reg.fit(X_train, y_train)
            m = evaluate_reg(reg, X_val, y_val, f"duration/{name}/val")
            results[name] = m
            trained[name] = reg
        except Exception as e:
            log.warning(f"  {name} failed: {e}")

    best_name = max(results, key=lambda k: results[k].get("R2", -999))
    best_reg  = trained[best_name]
    log.info(f"\n  ★ Best regressor: {best_name}  R²={results[best_name]['R2']:.4f}")

    # Final test evaluation
    test_m = evaluate_reg(best_reg, X_test, y_test, f"duration/BEST({best_name})-TEST")

    # Residual plot
    y_pred_test = np.clip(best_reg.predict(X_test), 0, None)
    residuals = y_test.values - y_pred_test
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_pred_test, residuals, alpha=0.3, s=10, color="#3b82f6")
    axes[0].axhline(0, color="red", lw=1)
    axes[0].set_xlabel("Predicted (min)"); axes[0].set_ylabel("Residual")
    axes[0].set_title(f"Residuals — {best_name}")

    axes[1].scatter(y_test, y_pred_test, alpha=0.3, s=10, color="#10b981")
    mn = min(y_test.min(), y_pred_test.min())
    mx = max(y_test.max(), y_pred_test.max())
    axes[1].plot([mn, mx], [mn, mx], "r--", lw=1)
    axes[1].set_xlabel("Actual (min)"); axes[1].set_ylabel("Predicted (min)")
    axes[1].set_title("Predicted vs Actual")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "duration_residuals.png", dpi=120)
    plt.close(fig)

    # 5-fold CV
    log.info("Running 5-fold cross validation on best regressor...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(best_reg, X_.values, y.values, cv=kf, scoring="r2", n_jobs=-1)
    log.info(f"CV R² scores: {cv_r2}  mean={cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    save_model(best_reg, "reg_duration")
    reg_report = {
        "best_model":   best_name,
        "all_models":   results,
        "test_metrics": test_m,
        "cv_r2_mean":   float(cv_r2.mean()),
        "cv_r2_std":    float(cv_r2.std()),
    }
    save_json(reg_report, "regression_results")
    log.info("Regression training complete. Results saved.")
    return reg_report


if __name__ == "__main__":
    X = pd.read_parquet(DATA_PROC / "features.parquet")
    Y = pd.read_parquet(DATA_PROC / "targets.parquet")
    run_regression_training(X, Y)
