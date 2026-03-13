#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def rolling_mean(values: list[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return arr
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window, dtype=float) / window
    valid = np.convolve(arr, kernel, mode="valid")
    prefix = arr[: window - 1]
    return np.concatenate([prefix, valid])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_scalar_metrics(step_records: list[dict], output_dir: Path, window: int) -> None:
    if not step_records:
        return

    x = np.arange(1, len(step_records) + 1)
    losses = [r["loss"] for r in step_records]
    accs = [r["accuracy"] for r in step_records]
    lrs = [r["learning_rate"] for r in step_records]
    logit_std = [r["logit_std"] for r in step_records]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(x, losses, alpha=0.35, label="raw")
    axes[0, 0].plot(x, rolling_mean(losses, window), label=f"rolling mean ({window})")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Cross Entropy")
    axes[0, 0].legend()

    axes[0, 1].plot(x, accs, alpha=0.35, label="raw")
    axes[0, 1].plot(x, rolling_mean(accs, window), label=f"rolling mean ({window})")
    axes[0, 1].set_title("Training Accuracy")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()

    axes[1, 0].plot(x, lrs)
    axes[1, 0].set_title("Learning Rate")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("LR")

    axes[1, 1].plot(x, logit_std, alpha=0.35, label="raw")
    axes[1, 1].plot(x, rolling_mean(logit_std, window), label=f"rolling mean ({window})")
    axes[1, 1].set_title("Logit Std")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Std")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "scalar_metrics.png", dpi=160)
    plt.close(fig)


def plot_shape_metadata(step_records: list[dict], output_dir: Path) -> None:
    if not step_records:
        return

    x = np.arange(1, len(step_records) + 1)
    seq_len = [r["seq_len"] for r in step_records]
    train_size = [r["train_size"] for r in step_records]
    max_features = [r["max_features"] for r in step_records]
    active_features = [r["active_features_max"] for r in step_records]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(x, seq_len, label="seq_len")
    axes[0].plot(x, train_size, label="train_size")
    axes[0].set_title("Sequence and Train Sizes")
    axes[0].set_ylabel("Length")
    axes[0].legend()

    axes[1].plot(x, max_features, label="tensor feature dim")
    axes[1].plot(x, active_features, label="active features")
    axes[1].set_title("Feature Counts")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Features")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "shape_metadata.png", dpi=160)
    plt.close(fig)


def plot_class_distributions(dist_records: list[dict], output_dir: Path) -> None:
    if not dist_records:
        return

    steps = [r["step"] for r in dist_records]
    n_classes = max(len(r["pred_distribution"]) for r in dist_records)
    pred = np.array(
        [np.array(r["pred_distribution"] + [0] * (n_classes - len(r["pred_distribution"])), dtype=float) for r in dist_records]
    )
    true = np.array(
        [np.array(r["true_distribution"] + [0] * (n_classes - len(r["true_distribution"])), dtype=float) for r in dist_records]
    )

    pred = pred / np.clip(pred.sum(axis=1, keepdims=True), 1e-12, None)
    true = true / np.clip(true.sum(axis=1, keepdims=True), 1e-12, None)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    im0 = axes[0].imshow(pred.T, aspect="auto", origin="lower", interpolation="nearest")
    axes[0].set_title("Predicted Class Distribution Over Time")
    axes[0].set_ylabel("Class")
    plt.colorbar(im0, ax=axes[0], label="Probability")

    im1 = axes[1].imshow(true.T, aspect="auto", origin="lower", interpolation="nearest")
    axes[1].set_title("Ground Truth Class Distribution Over Time")
    axes[1].set_xlabel("Logged Distribution Step Index")
    axes[1].set_ylabel("Class")
    plt.colorbar(im1, ax=axes[1], label="Probability")

    tick_positions = np.linspace(0, len(steps) - 1, min(6, len(steps)), dtype=int)
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels([str(steps[i]) for i in tick_positions])

    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution_heatmaps.png", dpi=160)
    plt.close(fig)


def plot_class_usage(dist_records: list[dict], output_dir: Path) -> None:
    if not dist_records:
        return

    steps = [r["step"] for r in dist_records]
    pred_unique = []
    true_unique = []
    dominance = []

    for record in dist_records:
        pred = np.asarray(record["pred_distribution"], dtype=float)
        true = np.asarray(record["true_distribution"], dtype=float)
        pred_unique.append(int((pred > 0).sum()))
        true_unique.append(int((true > 0).sum()))
        total = pred.sum()
        dominance.append(float(pred.max() / total) if total > 0 else 0.0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(steps, pred_unique, label="predicted unique classes")
    axes[0].plot(steps, true_unique, label="true unique classes")
    axes[0].set_title("Unique Class Usage")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].plot(steps, dominance)
    axes[1].set_title("Prediction Dominance")
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Largest predicted class share")

    fig.tight_layout()
    fig.savefig(output_dir / "class_usage.png", dpi=160)
    plt.close(fig)


def write_text_summary(step_records: list[dict], dist_records: list[dict], summary: dict | None, output_dir: Path) -> None:
    lines = []
    lines.append(f"num_step_records: {len(step_records)}")
    lines.append(f"num_distribution_records: {len(dist_records)}")

    if step_records:
        losses = np.asarray([r["loss"] for r in step_records], dtype=float)
        accs = np.asarray([r["accuracy"] for r in step_records], dtype=float)
        lines.append(f"loss_mean: {losses.mean():.6f}")
        lines.append(f"loss_min: {losses.min():.6f}")
        lines.append(f"loss_max: {losses.max():.6f}")
        lines.append(f"accuracy_mean: {accs.mean():.6f}")
        lines.append(f"accuracy_min: {accs.min():.6f}")
        lines.append(f"accuracy_max: {accs.max():.6f}")

    if dist_records:
        last = dist_records[-1]
        pred = np.asarray(last["pred_distribution"], dtype=float)
        true = np.asarray(last["true_distribution"], dtype=float)
        lines.append(f"last_logged_step: {last['step']}")
        lines.append(f"last_pred_unique_classes: {(pred > 0).sum()}")
        lines.append(f"last_true_unique_classes: {(true > 0).sum()}")
        if pred.sum() > 0:
            lines.append(f"last_pred_dominance: {pred.max() / pred.sum():.6f}")

    if summary:
        lines.append("summary_keys: " + ", ".join(sorted(summary.keys())))

    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training metrics saved by train.py.")
    parser.add_argument("--analysis_dir", type=str, default="analysis", help="Directory containing saved metrics.")
    parser.add_argument("--window", type=int, default=25, help="Rolling-average window for scalar plots.")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    output_dir = analysis_dir / "plots"
    ensure_dir(output_dir)

    step_records = load_jsonl(analysis_dir / "train_metrics_steps.jsonl")
    dist_records = load_jsonl(analysis_dir / "train_metrics_distributions.jsonl")

    summary_path = analysis_dir / "train_metrics_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None

    plot_scalar_metrics(step_records, output_dir, args.window)
    plot_shape_metadata(step_records, output_dir)
    plot_class_distributions(dist_records, output_dir)
    plot_class_usage(dist_records, output_dir)
    write_text_summary(step_records, dist_records, summary, output_dir)

    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
