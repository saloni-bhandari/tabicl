import os
import sys
import time
import json
import warnings
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.datasets import fetch_openml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from tabicl import TabICLClassifier
from localized_tabicl import LocalizedTabICLClassifier
from row_selection import select_rows

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# Configuration
# ============================================================================

RANDOM_SEED = 42
N_RUNS = 5  # Number of runs per configuration for statistics
TEST_SIZE = 0.2
N_ESTIMATORS = 8

# Datasets: (name, openml_id, description)
# Datasets: (name, openml_id, description)
DATASETS = [
    ("iris", 61, "small-multiclass"),
    ("credit-g", 31, "small-binary"),
    ("blood-transfusion", 1464, "medium-binary"),
    ("phoneme", 1489, "medium-binary"),
    ("MagicTelescope", 1120, "large-binary"),
    ("bank-marketing", 1461, "large-binary"),
    ("electricity", 44120, "large-binary"),
    ("adult", 1590, "large-binary"),
    ("jannis", 41168, "large-multiclass"),
    ("numerai28.6", 23517, "xlarge-binary"),
    ("Diabetes130US", 4541, "xlarge-multiclass"),
    ("MiniBooNE", 41150, "xlarge-binary"),
    ("credit-card-fraud", 42175, "xxlarge-binary"),
]

# k values to test
K_VALUES = [16, 32, 64, 128, 256]

# Selection methods to test
SELECTION_CONFIGS = [
    {"name": "full_tabicl", "type": "baseline"},
    {"name": "random", "type": "localized", "method": "random", "space": "embedding"},
    {"name": "raw_euclidean", "type": "localized", "method": "euclidean", "space": "raw"},
    {"name": "raw_cosine", "type": "localized", "method": "cosine", "space": "raw"},
    {"name": "emb_euclidean", "type": "localized", "method": "euclidean", "space": "embedding"},
    {"name": "emb_cosine", "type": "localized", "method": "cosine", "space": "embedding"},
]

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================================
# Dataset Loading
# ============================================================================

def load_dataset(name, openml_id):
    """Load and split a dataset from OpenML."""
    print(f"  Loading {name} (OpenML ID: {openml_id})...")
    dataset = fetch_openml(data_id=openml_id, as_frame=True, parser="auto")
    X = dataset.data
    y = dataset.target

    # Convert to numpy
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values

    # Encode categorical string columns to numeric
    try:
        X = X.astype(np.float64)
    except (ValueError, TypeError):
        from sklearn.preprocessing import OrdinalEncoder
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X = enc.fit_transform(X).astype(np.float64)

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)

    print(f"    Shape: {X.shape}, Classes: {len(np.unique(y))}")
    return X, y


def split_dataset(X, y, seed):
    """Split into train/test with a given seed."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)


# ============================================================================
# Experiment 1: Full TabICL Baselines
# ============================================================================

def run_baseline(X_train, y_train, X_test, y_test, seed):
    """Run full TabICL baseline and return metrics."""
    clf = TabICLClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=seed,
        verbose=False,
    )

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    t0 = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - t0

    t0 = time.time()
    proba = clf.predict_proba(X_test)
    predict_time = time.time() - t0

    y_pred = np.argmax(proba, axis=1)
    # Map predictions back through label encoder for accuracy
    y_pred_labels = clf.y_encoder_.inverse_transform(y_pred)

    acc = accuracy_score(y_test, y_pred_labels)

    # Log loss needs probability for each class in the test set
    # Ensure classes align
    classes = clf.classes_
    try:
        ll = log_loss(y_test, proba, labels=classes)
    except Exception:
        ll = float('nan')

    peak_mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0

    return {
        "accuracy": acc,
        "log_loss": ll,
        "fit_time": fit_time,
        "predict_time": predict_time,
        "total_time": fit_time + predict_time,
        "peak_memory_mb": peak_mem,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ============================================================================
# Experiment 2: Localized TabICL
# ============================================================================

def run_localized(X_train, y_train, X_test, y_test, k, method, space, seed):
    """Run localized TabICL and return metrics."""
    clf = LocalizedTabICLClassifier(
        k=k,
        selection_method=method,
        selection_space=space,
        n_estimators=N_ESTIMATORS,
        device=None,
        verbose=False,
    )

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    t0 = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - t0

    t0 = time.time()
    proba = clf.predict_proba(X_test)
    predict_time = time.time() - t0

    y_pred = np.argmax(proba, axis=1)
    y_pred_labels = clf.y_encoder_.inverse_transform(y_pred)

    acc = accuracy_score(y_test, y_pred_labels)

    classes = clf.classes_
    try:
        ll = log_loss(y_test, proba, labels=classes)
    except Exception:
        ll = float('nan')

    peak_mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    timing = clf.get_timing()

    return {
        "accuracy": acc,
        "log_loss": ll,
        "fit_time": fit_time,
        "predict_time": predict_time,
        "total_time": fit_time + predict_time,
        "peak_memory_mb": peak_mem,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "k": k,
        "method": method,
        "space": space,
        "t_stage12": timing.get("stage12", 0),
        "t_retrieval": timing.get("retrieval", 0),
        "t_stage3": timing.get("stage3", 0),
    }


# ============================================================================
# Main Experiment Loop
# ============================================================================

def run_all_experiments():
    """Run all experiments and save results."""

    all_results = []

    for ds_name, ds_id, ds_desc in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({ds_desc})")
        print(f"{'='*60}")

        try:
            X, y = load_dataset(ds_name, ds_id)
        except Exception as e:
            print(f"  ERROR loading {ds_name}: {e}")
            continue

        n_train_total = int(len(X) * (1 - TEST_SIZE))
        valid_k_values = [k for k in K_VALUES if k < n_train_total]

        for run_idx in range(N_RUNS):
            seed = RANDOM_SEED + run_idx
            X_train, X_test, y_train, y_test = split_dataset(X, y, seed)
            print(f"\n  Run {run_idx+1}/{N_RUNS} (seed={seed})")
            print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

            # --- Baseline ---
            print(f"    Running full TabICL baseline...")
            try:
                result = run_baseline(X_train, y_train, X_test, y_test, seed)
                result.update({
                    "dataset": ds_name,
                    "method_name": "full_tabicl",
                    "k": len(X_train),
                    "run": run_idx,
                    "selection_method": "none",
                    "selection_space": "none",
                })
                all_results.append(result)
                print(f"      Acc: {result['accuracy']:.4f}, LogLoss: {result['log_loss']:.4f}, "
                      f"Time: {result['total_time']:.2f}s")
            except Exception as e:
                print(f"      ERROR: {e}")

            # --- Localized variants ---
            for config in SELECTION_CONFIGS:
                if config["type"] == "baseline":
                    continue  # Already ran

                method = config["method"]
                space = config["space"]
                config_name = config["name"]

                for k in valid_k_values:
                    print(f"    Running {config_name} (k={k})...")
                    try:
                        result = run_localized(
                            X_train, y_train, X_test, y_test,
                            k=k, method=method, space=space, seed=seed,
                        )
                        result.update({
                            "dataset": ds_name,
                            "method_name": config_name,
                            "run": run_idx,
                            "selection_method": method,
                            "selection_space": space,
                        })
                        all_results.append(result)
                        print(f"      Acc: {result['accuracy']:.4f}, LogLoss: {result['log_loss']:.4f}, "
                              f"Time: {result['total_time']:.2f}s "
                              f"(S12: {result['t_stage12']:.2f}, Retr: {result['t_retrieval']:.2f}, "
                              f"S3: {result['t_stage3']:.2f})")
                    except Exception as e:
                        print(f"      ERROR: {e}")

            # Save intermediate results
            df = pd.DataFrame(all_results)
            df.to_csv(RESULTS_DIR / "results_intermediate.csv", index=False)

    # Final save
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "results_final.csv", index=False)
    print(f"\n\nResults saved to {RESULTS_DIR / 'results_final.csv'}")

    return df


# ============================================================================
# Analysis & Visualization
# ============================================================================

def analyze_results(df: pd.DataFrame):
    """Generate summary tables and print analysis."""

    print("\n" + "="*80)
    print("EXPERIMENT 1: Does localization work?")
    print("="*80)

    # Group by dataset and method, average over runs
    summary = df.groupby(["dataset", "method_name", "k"]).agg({
        "accuracy": ["mean", "std"],
        "log_loss": ["mean", "std"],
        "total_time": ["mean"],
        "peak_memory_mb": ["mean"],
    }).round(4)
    print(summary.to_string())

    print("\n" + "="*80)
    print("EXPERIMENT 2: Raw vs Embedding retrieval")
    print("="*80)

    retrieval_methods = df[df["method_name"].isin(["raw_cosine", "emb_cosine"])]
    if not retrieval_methods.empty:
        comparison = retrieval_methods.groupby(["dataset", "method_name", "k"]).agg({
            "accuracy": ["mean", "std"],
        }).round(4)
        print(comparison.to_string())

    print("\n" + "="*80)
    print("EXPERIMENT 3: Accuracy vs k tradeoff")
    print("="*80)

    for ds_name in df["dataset"].unique():
        ds_data = df[df["dataset"] == ds_name]
        baseline_acc = ds_data[ds_data["method_name"] == "full_tabicl"]["accuracy"].mean()
        print(f"\n  {ds_name} (baseline acc: {baseline_acc:.4f}):")

        localized = ds_data[ds_data["method_name"] == "emb_cosine"]
        if not localized.empty:
            for k in sorted(localized["k"].unique()):
                k_data = localized[localized["k"] == k]
                acc_mean = k_data["accuracy"].mean()
                acc_std = k_data["accuracy"].std()
                print(f"    k={k:4d}: {acc_mean:.4f} ± {acc_std:.4f} "
                      f"(Δ = {acc_mean - baseline_acc:+.4f})")

    print("\n" + "="*80)
    print("EXPERIMENT 4: Runtime breakdown")
    print("="*80)

    timing_cols = ["t_stage12", "t_retrieval", "t_stage3"]
    localized_data = df[df["method_name"].isin(["emb_cosine", "random"])]
    if not localized_data.empty and all(c in localized_data.columns for c in timing_cols):
        timing = localized_data.groupby(["dataset", "method_name", "k"])[timing_cols + ["total_time"]].mean().round(3)
        print(timing.to_string())


def generate_plots(df: pd.DataFrame):
    """Generate plots for the experiments."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    # --- Plot 1: Accuracy vs k for each dataset ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    datasets = df["dataset"].unique()
    methods_to_plot = ["emb_cosine", "raw_cosine", "random"]
    colors = {"emb_cosine": "blue", "raw_cosine": "orange", "random": "gray"}
    labels = {"emb_cosine": "Embedding kNN", "raw_cosine": "Raw-feature kNN", "random": "Random"}

    for idx, ds_name in enumerate(datasets):
        if idx >= len(axes):
            break
        ax = axes[idx]
        ds_data = df[df["dataset"] == ds_name]

        # Baseline
        baseline = ds_data[ds_data["method_name"] == "full_tabicl"]
        if not baseline.empty:
            baseline_acc = baseline.groupby("run")["accuracy"].mean().mean()
            ax.axhline(y=baseline_acc, color="red", linestyle="--", label="Full TabICL", linewidth=2)

        # Localized methods
        for method in methods_to_plot:
            method_data = ds_data[ds_data["method_name"] == method]
            if method_data.empty:
                continue
            grouped = method_data.groupby("k")["accuracy"]
            means = grouped.mean()
            stds = grouped.std()
            ax.errorbar(means.index, means.values, yerr=stds.values,
                       marker='o', label=labels.get(method, method),
                       color=colors.get(method, "black"), capsize=3)

        ax.set_xlabel("k (selected training rows)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{ds_name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "accuracy_vs_k.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plots_dir / 'accuracy_vs_k.png'}")

    # --- Plot 2: Runtime vs k ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, ds_name in enumerate(datasets):
        if idx >= len(axes):
            break
        ax = axes[idx]
        ds_data = df[df["dataset"] == ds_name]

        baseline = ds_data[ds_data["method_name"] == "full_tabicl"]
        if not baseline.empty:
            baseline_time = baseline["total_time"].mean()
            ax.axhline(y=baseline_time, color="red", linestyle="--", label="Full TabICL", linewidth=2)

        for method in methods_to_plot:
            method_data = ds_data[ds_data["method_name"] == method]
            if method_data.empty:
                continue
            grouped = method_data.groupby("k")["total_time"]
            means = grouped.mean()
            ax.plot(means.index, means.values, marker='o',
                   label=labels.get(method, method),
                   color=colors.get(method, "black"))

        ax.set_xlabel("k (selected training rows)")
        ax.set_ylabel("Total time (s)")
        ax.set_title(f"{ds_name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "runtime_vs_k.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plots_dir / 'runtime_vs_k.png'}")

    # --- Plot 3: Runtime breakdown for emb_cosine ---
    emb_data = df[df["method_name"] == "emb_cosine"]
    if not emb_data.empty and "t_stage12" in emb_data.columns:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, ds_name in enumerate(datasets):
            if idx >= len(axes):
                break
            ax = axes[idx]
            ds = emb_data[emb_data["dataset"] == ds_name]
            if ds.empty:
                continue

            grouped = ds.groupby("k")[["t_stage12", "t_retrieval", "t_stage3"]].mean()
            grouped.plot(kind="bar", stacked=True, ax=ax,
                        color=["#2196F3", "#FF9800", "#4CAF50"])
            ax.set_xlabel("k")
            ax.set_ylabel("Time (s)")
            ax.set_title(f"{ds_name} — Runtime Breakdown")
            ax.legend(["Stage 1-2", "Retrieval", "Stage 3"], fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(plots_dir / "runtime_breakdown.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plots_dir / 'runtime_breakdown.png'}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    print("Localized TabICL Experiments")
    print("="*60)
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Datasets: {[d[0] for d in DATASETS]}")
    print(f"K values: {K_VALUES}")
    print(f"Selection methods: {[c['name'] for c in SELECTION_CONFIGS]}")
    print(f"Runs per config: {N_RUNS}")
    print()

    # Run experiments
    df = run_all_experiments()

    # Analyze
    if df is not None and not df.empty:
        analyze_results(df)
        print("\nGenerating plots...")
        generate_plots(df)

    print("\nDone!")
