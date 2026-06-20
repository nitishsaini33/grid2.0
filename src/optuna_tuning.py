"""
optuna_tuning.py — Hyperparameter Optimization with Optuna (60 trials)
Tunes XGBoost & LightGBM for the most complex classification target.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.INFO)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import (get_logger, DATA_PROC, save_json, print_section, REPORTS_DIR)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

log = get_logger("optuna_tuning")

N_TRIALS  = 60
N_CV      = 5
SEED      = 42


# ──────────────────────────────────────────────────────────────────────────────
def objective_xgb(trial, X, y, n_classes):
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 100, 500),
        "max_depth":       trial.suggest_int("max_depth", 3, 10),
        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":trial.suggest_int("min_child_weight", 1, 10),
        "gamma":           trial.suggest_float("gamma", 0, 5),
        "reg_alpha":       trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda":      trial.suggest_float("reg_lambda", 0.5, 2.0),
        "random_state": SEED, "verbosity": 0,
        "use_label_encoder": False, "eval_metric": "mlogloss",
    }
    clf = XGBClassifier(**params)
    cv  = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=SEED)
    scores = cross_val_score(clf, X, y, cv=cv,
                              scoring="balanced_accuracy", n_jobs=-1)
    return scores.mean()


def objective_lgb(trial, X, y, n_classes):
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 100, 500),
        "max_depth":       trial.suggest_int("max_depth", 3, 10),
        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples":trial.suggest_int("min_child_samples", 10, 100),
        "num_leaves":      trial.suggest_int("num_leaves", 20, 150),
        "reg_alpha":       trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda":      trial.suggest_float("reg_lambda", 0.0, 1.0),
        "class_weight": "balanced",
        "random_state": SEED, "verbosity": -1, "n_jobs": -1,
    }
    clf = LGBMClassifier(**params)
    cv  = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=SEED)
    scores = cross_val_score(clf, X, y, cv=cv,
                              scoring="balanced_accuracy", n_jobs=-1)
    return scores.mean()


def objective_cat(trial, X, y, n_classes):
    params = {
        "iterations":   trial.suggest_int("iterations", 100, 400),
        "depth":        trial.suggest_int("depth", 3, 10),
        "learning_rate":trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg":  trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "auto_class_weights": "Balanced",
        "random_seed": SEED, "verbose": 0, "thread_count": -1,
    }
    clf = CatBoostClassifier(**params)
    cv  = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=SEED)
    scores = cross_val_score(clf, X, y, cv=cv,
                              scoring="balanced_accuracy", n_jobs=-1)
    return scores.mean()


# ──────────────────────────────────────────────────────────────────────────────
def tune_model(name: str, objective_fn, X, y, n_classes):
    print_section(f"OPTUNA TUNING — {name}  ({N_TRIALS} trials)")
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{name}_study",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )

    from functools import partial
    obj = partial(objective_fn, X=X, y=y, n_classes=n_classes)

    study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=True,
                   n_jobs=1)

    log.info(f"  Best {name} score: {study.best_value:.4f}")
    log.info(f"  Best params: {study.best_params}")
    return study.best_params, study.best_value


# ──────────────────────────────────────────────────────────────────────────────
def run_optuna_tuning(X: pd.DataFrame, Y: pd.DataFrame) -> dict:
    """Tune on the primary target: congestion_severity"""
    print_section("OPTUNA HYPERPARAMETER OPTIMIZATION")

    target = "congestion_severity"
    if target not in Y.columns:
        log.warning(f"{target} not found, skipping Optuna")
        return {}

    y = Y[target].dropna().astype(int)
    X_ = X.loc[y.index].values

    # Use 70% for tuning (faster)
    from sklearn.model_selection import train_test_split
    X_tune, _, y_tune, _ = train_test_split(
        X_, y, test_size=0.30, random_state=SEED, stratify=y)

    n_classes = int(y_tune.nunique())
    log.info(f"Tuning on {len(X_tune):,} samples, {n_classes} classes")

    best_params = {}

    xgb_params, xgb_score = tune_model("XGBoost", objective_xgb, X_tune, y_tune, n_classes)
    best_params["XGBoost"] = xgb_params

    lgb_params, lgb_score = tune_model("LightGBM", objective_lgb, X_tune, y_tune, n_classes)
    best_params["LightGBM"] = lgb_params

    cat_params, cat_score = tune_model("CatBoost", objective_cat, X_tune, y_tune, n_classes)
    best_params["CatBoost"] = cat_params

    summary = {
        "target": target,
        "n_trials": N_TRIALS,
        "best_params": best_params,
        "scores": {
            "XGBoost":  xgb_score,
            "LightGBM": lgb_score,
            "CatBoost": cat_score,
        }
    }
    save_json(summary, "optuna_best_params")
    log.info("Optuna tuning complete. Best params saved.")
    return best_params


if __name__ == "__main__":
    X = pd.read_parquet(DATA_PROC / "features.parquet")
    Y = pd.read_parquet(DATA_PROC / "targets.parquet")
    run_optuna_tuning(X, Y)
