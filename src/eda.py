"""
eda.py — Exploratory Data Analysis with Rich Visualizations
Generates: heatmap, correlation matrix, distributions, event-wise analysis
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import get_logger, DATA_PROC, REPORTS_DIR, print_section

log = get_logger("eda")
STYLE = "dark_background"
PALETTE = "viridis"


def run_eda(df: pd.DataFrame):
    print_section("EDA — EXPLORATORY DATA ANALYSIS")
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    # ── 1. Basic Stats ────────────────────────────────────────────────────
    log.info(f"Shape: {df.shape}")
    print("\n=== BASIC STATS ===")
    print(df.describe(include="all").to_string())

    # ── 2. Missing Value Report ───────────────────────────────────────────
    print("\n=== MISSING VALUES ===")
    miss = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    miss_df = pd.DataFrame({"Missing": miss, "Pct": miss_pct})
    miss_df = miss_df[miss_df["Missing"] > 0].sort_values("Pct", ascending=False)
    print(miss_df.to_string())

    # ── 3. Missing Values Heatmap ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    miss_hm = df[numeric_cols].isnull().astype(int)
    if miss_hm.sum().sum() > 0:
        sns.heatmap(miss_hm.T, cmap="Reds", cbar=False, ax=ax, yticklabels=True)
        ax.set_title("Missing Values Heatmap", fontsize=14, fontweight="bold")
        ax.set_xlabel("Row index")
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / "missing_heatmap.png", dpi=120)
    plt.close(fig)
    log.info("Saved missing_heatmap.png")

    # ── 4. Correlation Matrix ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 12))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                annot=False, linewidths=0.3, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "correlation_matrix.png", dpi=120)
    plt.close(fig)
    log.info("Saved correlation_matrix.png")

    # ── 5. Event Cause Distribution ───────────────────────────────────────
    if "event_cause" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        cause_counts = df["event_cause"].value_counts()
        colors = sns.color_palette("husl", len(cause_counts))
        bars = ax.barh(cause_counts.index, cause_counts.values, color=colors)
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.set_title("Event Cause Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Count")
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / "event_cause_dist.png", dpi=120)
        plt.close(fig)

    # ── 6. Congestion Severity Distribution ──────────────────────────────
    if "congestion_severity" in df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Severity count
        sev_counts = df["congestion_severity"].value_counts().sort_index()
        sev_labels = ["Low", "Moderate", "High", "Severe"]
        colors     = ["#22c55e", "#eab308", "#f97316", "#ef4444"]
        axes[0].bar([sev_labels[i] for i in sev_counts.index],
                    sev_counts.values, color=colors[:len(sev_counts)])
        axes[0].set_title("Congestion Severity Distribution")
        axes[0].set_ylabel("Count")

        # Severity by event cause
        if "event_cause" in df.columns:
            pivot = df.groupby(["event_cause", "congestion_severity"]).size().unstack(fill_value=0)
            pivot.plot(kind="bar", ax=axes[1], stacked=True,
                       colormap="RdYlGn_r", legend=True)
            axes[1].set_title("Severity by Event Cause")
            axes[1].set_xlabel(""); axes[1].tick_params(axis="x", rotation=45)

        # Severity by hour
        if "hour" in df.columns:
            df.groupby(["hour", "congestion_severity"]).size().unstack(fill_value=0).plot(
                kind="line", ax=axes[2], colormap="RdYlGn_r", linewidth=2)
            axes[2].set_title("Severity by Hour of Day")
            axes[2].set_xlabel("Hour")

        fig.suptitle("Congestion Severity Analysis", fontsize=15, fontweight="bold")
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / "severity_analysis.png", dpi=120)
        plt.close(fig)

    # ── 7. Duration Analysis ──────────────────────────────────────────────
    if "duration_min" in df.columns:
        dur = df["duration_min"].dropna()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(dur, bins=50, color="#3b82f6", edgecolor="white", alpha=0.8)
        axes[0].set_title("Event Duration Distribution (minutes)")
        axes[0].set_xlabel("Duration (min)")
        axes[0].set_ylabel("Frequency")

        if "event_cause" in df.columns:
            dur_by_cause = df.groupby("event_cause")["duration_min"].median().sort_values()
            axes[1].barh(dur_by_cause.index, dur_by_cause.values,
                         color=sns.color_palette("Blues_d", len(dur_by_cause)))
            axes[1].set_title("Median Duration by Event Cause")
            axes[1].set_xlabel("Median Duration (min)")

        fig.tight_layout()
        fig.savefig(REPORTS_DIR / "duration_analysis.png", dpi=120)
        plt.close(fig)

    # ── 8. Spatial Distribution ────────────────────────────────────────────
    if "latitude" in df.columns and "longitude" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        scat = ax.scatter(df["longitude"], df["latitude"],
                          c=df.get("congestion_severity", pd.Series([1]*len(df))),
                          cmap="RdYlGn_r", alpha=0.3, s=5)
        plt.colorbar(scat, ax=ax, label="Severity")
        ax.set_title("Geographic Distribution of Events (Bangalore)", fontsize=14)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / "spatial_distribution.png", dpi=120)
        plt.close(fig)

    # ── 9. Zone-wise Stats ────────────────────────────────────────────────
    if "zone" in df.columns and "congestion_severity" in df.columns:
        zone_stats = df.groupby("zone")["congestion_severity"].agg(["mean", "count"])
        zone_stats = zone_stats.sort_values("mean", ascending=False)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].barh(zone_stats.index, zone_stats["mean"],
                     color=sns.color_palette("Reds_d", len(zone_stats)))
        axes[0].set_title("Mean Severity by Zone")
        axes[0].set_xlabel("Mean Severity")
        axes[1].barh(zone_stats.index, zone_stats["count"],
                     color=sns.color_palette("Blues_d", len(zone_stats)))
        axes[1].set_title("Event Count by Zone")
        axes[1].set_xlabel("Event Count")
        fig.suptitle("Zone-wise Analysis", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / "zone_analysis.png", dpi=120)
        plt.close(fig)

    # ── 10. Feature Distributions (numeric) ──────────────────────────────
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in ["congestion_severity","manpower_req","barricade_req","diversion_req"]][:12]
    if num_cols:
        n = len(num_cols)
        ncols = 4; nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows*3))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            axes[i].hist(df[col].dropna(), bins=30, color="#6366f1", edgecolor="white", alpha=0.8)
            axes[i].set_title(col, fontsize=9)
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / "feature_distributions.png", dpi=120)
        plt.close(fig)

    log.info("EDA complete — all plots saved to models/reports/")
    print("\n✓ EDA plots saved to models/reports/")


if __name__ == "__main__":
    from preprocessing import run_preprocessing
    from feature_engineering import run_feature_engineering
    df_clean = run_preprocessing()
    X, Y = run_feature_engineering(df_clean)
    full = pd.read_parquet(DATA_PROC / "full_featured.parquet")
    run_eda(full)
