"""
train_classification.py — Multi-Target Classification Training
Targets: congestion_severity, manpower_req, barricade_req, diversion_req
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import json
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import (get_logger, DATA_PROC, save_model, save_json,
                   print_section, print_metrics, MODELS_DIR, REPORTS_DIR)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               VotingClassifier, StackingClassifier,
                               HistGradientBoostingClassifier)
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, classification_report,
                              cohen_kappa_score, balanced_accuracy_score,
                              precision_score, recall_score)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

log = get_logger("train_classification")

CLF_TARGETS = ["congestion_severity", "manpower_req", "barricade_req", "diversion_req"]


# ──────────────────────────────────────────────────────────────────────────────
def evaluate_clf(model, X_test, y_test, target_name: str, n_classes: int) -> dict:
    y_pred = model.predict(X_test)
    avg = "binary" if n_classes == 2 else "weighted"
    auc_avg = "macro"

    metrics = {
        "accuracy":          accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1":                f1_score(y_test, y_pred, average=avg, zero_division=0),
        "precision":         precision_score(y_test, y_pred, average=avg, zero_division=0),
        "recall":            recall_score(y_test, y_pred, average=avg, zero_division=0),
        "cohen_kappa":       cohen_kappa_score(y_test, y_pred),
    }

    # ROC-AUC
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if n_classes == 2:
                metrics["roc_auc"] = roc_auc_score(y_test, proba[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(y_test, proba,
                                                     multi_class="ovr", average=auc_avg)
    except Exception:
        metrics["roc_auc"] = np.nan

    print_metrics(metrics, title=target_name)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
def build_classifiers(n_classes: int, seed: int = 42) -> dict:
    """Return dict of {name: estimator}"""
    common_xgb = dict(random_state=seed, eval_metric="mlogloss",
                      use_label_encoder=False, verbosity=0)
    common_lgb = dict(random_state=seed, verbosity=-1, n_jobs=-1)
    common_cat = dict(random_seed=seed, verbose=0, thread_count=-1)

    clfs = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=seed, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=seed, n_jobs=-1),
        "HistGradBoost": HistGradientBoostingClassifier(
            max_iter=200, random_state=seed),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            **common_xgb),
        "LightGBM": LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", **common_lgb),
        "CatBoost": CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.05,
            auto_class_weights="Balanced", **common_cat),
        "BalancedRF": BalancedRandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1),
    }
    return clfs


# ──────────────────────────────────────────────────────────────────────────────
def train_target(X_train, X_val, X_test, y_train, y_val, y_test,
                 target_name: str, best_params: dict = None):
    print_section(f"TRAINING: {target_name.upper()}")

    n_classes  = int(y_train.nunique())
    log.info(f"Classes: {sorted(y_train.unique())}  n={n_classes}")

    # ── Balance with SMOTE ────────────────────────────────────────────────
    try:
        smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.value_counts().min()-1))
        X_res, y_res = smote.fit_resample(X_train, y_train)
        log.info(f"SMOTE: {len(X_train)} → {len(X_res)} rows")
    except Exception as e:
        log.warning(f"SMOTE failed ({e}), using original data")
        X_res, y_res = X_train.copy(), y_train.copy()

    clfs = build_classifiers(n_classes)

    # ── Apply best_params from Optuna if available ────────────────────────
    if best_params and "XGBoost" in best_params:
        p = best_params["XGBoost"]
        clfs["XGBoost"].set_params(**p)
    if best_params and "LightGBM" in best_params:
        p = best_params["LightGBM"]
        clfs["LightGBM"].set_params(**p)

    results = {}
    trained = {}

    for name, clf in clfs.items():
        log.info(f"  Fitting {name}...")
        try:
            clf.fit(X_res, y_res)
            m = evaluate_clf(clf, X_val, y_val, f"{target_name}/{name}", n_classes)
            results[name] = m
            trained[name] = clf
        except Exception as e:
            log.warning(f"  {name} failed: {e}")

    # ── Voting Ensemble ───────────────────────────────────────────────────
    top3 = sorted(results, key=lambda k: results[k].get("balanced_accuracy", 0),
                  reverse=True)[:3]
    log.info(f"Top-3 for ensemble: {top3}")
    try:
        voting = VotingClassifier(
            estimators=[(n, trained[n]) for n in top3],
            voting="soft", n_jobs=-1)
        voting.fit(X_res, y_res)
        m = evaluate_clf(voting, X_val, y_val, f"{target_name}/VotingEnsemble", n_classes)
        results["VotingEnsemble"] = m
        trained["VotingEnsemble"] = voting
    except Exception as e:
        log.warning(f"VotingEnsemble failed: {e}")

    # ── Stacking Ensemble ─────────────────────────────────────────────────
    try:
        stacking = StackingClassifier(
            estimators=[(n, trained[n]) for n in top3],
            final_estimator=LogisticRegression(max_iter=500, C=0.5),
            passthrough=True, n_jobs=-1)
        stacking.fit(X_res, y_res)
        m = evaluate_clf(stacking, X_val, y_val, f"{target_name}/StackingEnsemble", n_classes)
        results["StackingEnsemble"] = m
        trained["StackingEnsemble"] = stacking
    except Exception as e:
        log.warning(f"StackingEnsemble failed: {e}")

    # ── Select Best ───────────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k].get("balanced_accuracy", 0))
    best_clf  = trained[best_name]
    log.info(f"\n  ★ Best model for {target_name}: {best_name}  "
             f"bal_acc={results[best_name]['balanced_accuracy']:.4f}")

    # Final evaluation on test set
    test_m = evaluate_clf(best_clf, X_test, y_test,
                          f"{target_name}/BEST({best_name})-TEST", n_classes)

    # Confusion matrix
    y_pred_test = best_clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix — {target_name} ({best_name})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"cm_{target_name}.png", dpi=120)
    plt.close(fig)

    # Save
    save_model(best_clf, f"clf_{target_name}")
    return best_name, best_clf, results, test_m


# ──────────────────────────────────────────────────────────────────────────────
def run_classification_training(X: pd.DataFrame, Y: pd.DataFrame,
                                 best_params: dict = None):
    from sklearn.model_selection import train_test_split

    print_section("CLASSIFICATION PIPELINE START")
    all_results = {}

    for target in CLF_TARGETS:
        if target not in Y.columns:
            log.warning(f"Target {target} not found, skipping")
            continue

        y = Y[target].dropna().astype(int)
        X_ = X.loc[y.index]

        # 70 / 15 / 15 split
        X_trv, X_test, y_trv, y_test = train_test_split(
            X_, y, test_size=0.15, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trv, y_trv, test_size=0.15/0.85, random_state=42, stratify=y_trv)

        log.info(f"{target}: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        best_name, best_clf, results, test_m = train_target(
            X_train, X_val, X_test, y_train, y_val, y_test,
            target, best_params)

        all_results[target] = {
            "best_model":    best_name,
            "all_models":    results,
            "test_metrics":  test_m,
        }

    save_json(all_results, "classification_results")
    log.info("Classification training complete. Results saved.")
    return all_results


if __name__ == "__main__":
    from feature_engineering import FEATURE_COLS, TARGET_COLS
    X = pd.read_parquet(DATA_PROC / "features.parquet")
    Y = pd.read_parquet(DATA_PROC / "targets.parquet")
    run_classification_training(X, Y)
