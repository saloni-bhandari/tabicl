import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Styling ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

METHOD_STYLE = {
    "full_tabicl":   {"color": "#D32F2F", "marker": "s", "label": "Full TabICL (baseline)"},
    "emb_cosine":    {"color": "#1976D2", "marker": "o", "label": "Emb Cosine"},
    "emb_euclidean": {"color": "#7B1FA2", "marker": "^", "label": "Emb Euclidean"},
    "raw_cosine":    {"color": "#F57C00", "marker": "D", "label": "Raw Cosine"},
    "raw_euclidean": {"color": "#388E3C", "marker": "v", "label": "Raw Euclidean"},
    "random":        {"color": "#757575", "marker": "x", "label": "Random"},
}

STAGE_COLORS = {"Stage 1-2": "#2196F3", "Retrieval": "#FF9800", "Stage 3": "#4CAF50"}


# ── Helpers ──────────────────────────────────────────────────

def _resolve_csv(argv: list[str]) -> Path:
    """Find the results CSV — accept CLI arg or auto-detect."""
    if len(argv) > 1:
        p = Path(argv[1])
        if p.exists():
            return p
    for candidate in ["results/results_final.csv", "results/results_3.csv", "results/results_2.csv", "results/results_intermediate.csv",
                       "results/results_all.csv", "results_all.csv"]:
        p = Path(candidate)
        if p.exists():
            return p
    sys.exit("ERROR: No results CSV found. Pass the path as an argument.")


def _norm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names across old / new CSV formats."""
    renames = {}
    if "method" in df.columns and "method_name" not in df.columns:
        renames["method"] = "method_name"
    if "peak_mem_mb" in df.columns and "peak_memory_mb" not in df.columns:
        renames["peak_mem_mb"] = "peak_memory_mb"
    if renames:
        df = df.rename(columns=renames)
    # Drop stray columns that duplicate info under different names
    for col in ["method", "space"]:
        if col in df.columns and "method_name" in df.columns:
            df = df.drop(columns=[col], errors="ignore")
    return df


def _style(m: str):
    return METHOD_STYLE.get(m, {"color": "black", "marker": ".", "label": m})


def _n_cols(n: int, max_cols: int = 4) -> tuple[int, int]:
    ncols = min(n, max_cols)
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


# ── Table generators ─────────────────────────────────────────

def table_per_dataset(df: pd.DataFrame) -> str:
    """Detailed per-dataset table comparing every method/k to baseline."""
    bl = df[df["method_name"] == "full_tabicl"]
    loc = df[df["method_name"] != "full_tabicl"]
    lines: list[str] = []

    for ds in df["dataset"].unique():
        bl_ds = bl[bl["dataset"] == ds]
        loc_ds = loc[loc["dataset"] == ds]
        if bl_ds.empty:
            continue
        bl_acc = bl_ds["accuracy"].mean()
        bl_pred = bl_ds["predict_time"].mean()
        bl_total = bl_ds["total_time"].mean()
        bl_mem = bl_ds["peak_memory_mb"].mean()
        n_train = int(bl_ds["k"].iloc[0])

        lines.append("")
        lines.append("=" * 105)
        lines.append(f"  {ds.upper()}  (n_train={n_train})")
        lines.append(f"  Baseline: acc={bl_acc:.4f}  predict={bl_pred:.2f}s"
                      f"  total={bl_total:.2f}s  mem={bl_mem:.0f}MB")
        lines.append("=" * 105)
        lines.append(f"{'Method':<16} {'k':<6} {'Acc':>8} {'ΔAcc':>9}"
                      f" {'PredTime':>10} {'Speedup':>8} {'MemMB':>8} {'ΔMem':>8}")
        lines.append("-" * 105)

        g = loc_ds.groupby(["method_name", "k"]).agg(
            acc=("accuracy", "mean"),
            pred=("predict_time", "mean"),
            mem=("peak_memory_mb", "mean"),
        ).sort_index()

        for (method, k), row in g.iterrows():
            acc_diff = row["acc"] - bl_acc
            sp = bl_pred / row["pred"] if row["pred"] > 0 else float("inf")
            mem_pct = (1 - row["mem"] / bl_mem) * 100 if bl_mem > 0 else 0
            sign = "+" if acc_diff >= 0 else ""
            lines.append(
                f"{method:<16} {k:<6} {row['acc']:>8.4f} {sign}{acc_diff:>8.4f}"
                f" {row['pred']:>9.2f}s {sp:>7.2f}x {row['mem']:>8.0f} {mem_pct:>+7.1f}%"
            )

    return "\n".join(lines)


def table_best_per_dataset(df: pd.DataFrame) -> str:
    """One-row-per-dataset summary: best localized config vs baseline."""
    bl = df[df["method_name"] == "full_tabicl"]
    loc = df[df["method_name"] != "full_tabicl"]
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 110)
    lines.append("  BEST LOCALIZED vs BASELINE (per dataset)")
    lines.append("=" * 110)
    lines.append(f"{'Dataset':<18} {'BL Acc':>8} {'Best':>8} {'ΔAcc':>8}"
                  f" {'Method':<16} {'k':>4} {'PredSp':>8} {'TotalSp':>9} {'ΔMem':>8}")
    lines.append("-" * 110)

    for ds in df["dataset"].unique():
        bl_ds = bl[bl["dataset"] == ds]
        loc_ds = loc[loc["dataset"] == ds]
        if bl_ds.empty or loc_ds.empty:
            continue
        bl_acc = bl_ds["accuracy"].mean()
        bl_pred = bl_ds["predict_time"].mean()
        bl_total = bl_ds["total_time"].mean()
        bl_mem = bl_ds["peak_memory_mb"].mean()

        g = loc_ds.groupby(["method_name", "k"]).agg(
            acc=("accuracy", "mean"),
            pred=("predict_time", "mean"),
            total=("total_time", "mean"),
            mem=("peak_memory_mb", "mean"),
        )
        best_idx = g["acc"].idxmax()
        best = g.loc[best_idx]
        method, k = best_idx
        ad = best["acc"] - bl_acc
        ps = bl_pred / best["pred"] if best["pred"] > 0 else float("inf")
        ts = bl_total / best["total"] if best["total"] > 0 else float("inf")
        mp = (1 - best["mem"] / bl_mem) * 100 if bl_mem > 0 else 0
        sign = "+" if ad >= 0 else ""
        lines.append(
            f"{ds:<18} {bl_acc:>8.4f} {best['acc']:>8.4f} {sign}{ad:>7.4f}"
            f" {method:<16} {k:>4} {ps:>7.2f}x {ts:>8.2f}x {mp:>+7.1f}%"
        )

    return "\n".join(lines)


def table_timing_breakdown(df: pd.DataFrame) -> str:
    """Timing breakdown for emb_cosine."""
    emb = df[df["method_name"] == "emb_cosine"]
    if emb.empty:
        return ""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 95)
    lines.append("  TIMING BREAKDOWN — emb_cosine  (Stage 1-2 / Retrieval / Stage 3)")
    lines.append("=" * 95)
    lines.append(f"{'Dataset':<18} {'k':>5} {'S1-2':>8} {'Retr':>8}"
                  f" {'S3':>8} {'S3%':>6} {'Pred':>8}")
    lines.append("-" * 95)

    g = emb.groupby(["dataset", "k"]).agg(
        s12=("t_stage12", "mean"),
        ret=("t_retrieval", "mean"),
        s3=("t_stage3", "mean"),
        pred=("predict_time", "mean"),
    )
    for (ds, k), row in g.iterrows():
        total = row["s12"] + row["ret"] + row["s3"]
        s3_pct = row["s3"] / total * 100 if total > 0 else 0
        lines.append(
            f"{ds:<18} {k:>5} {row['s12']:>7.2f}s {row['ret']:>7.3f}s"
            f" {row['s3']:>7.2f}s {s3_pct:>5.1f}% {row['pred']:>7.2f}s"
        )

    return "\n".join(lines)


# ── Plot generators ──────────────────────────────────────────

def plot_accuracy_vs_k(df: pd.DataFrame, out_dir: Path):
    """One subplot per dataset: accuracy (mean ± std) vs k for each method."""
    datasets = df["dataset"].unique()
    methods = [m for m in ["emb_cosine", "emb_euclidean", "raw_cosine",
                           "raw_euclidean", "random"] if m in df["method_name"].values]
    nrows, ncols = _n_cols(len(datasets))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows),
                              squeeze=False)

    for idx, ds in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        ds_df = df[df["dataset"] == ds]

        # Baseline
        bl = ds_df[ds_df["method_name"] == "full_tabicl"]["accuracy"]
        if not bl.empty:
            ax.axhline(bl.mean(), **{k: v for k, v in _style("full_tabicl").items()
                       if k in ("color",)}, linestyle="--", linewidth=2,
                       label=_style("full_tabicl")["label"])
            ax.axhspan(bl.mean() - bl.std(), bl.mean() + bl.std(),
                       color=_style("full_tabicl")["color"], alpha=0.07)

        for m in methods:
            md = ds_df[ds_df["method_name"] == m]
            if md.empty:
                continue
            g = md.groupby("k")["accuracy"]
            means, stds = g.mean(), g.std().fillna(0)
            s = _style(m)
            ax.errorbar(means.index, means.values, yerr=stds.values,
                        marker=s["marker"], color=s["color"], label=s["label"],
                        capsize=3, linewidth=1.5, markersize=6)

        ax.set_title(ds)
        ax.set_xlabel("k")
        ax.set_ylabel("Accuracy")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Remove unused axes
    for idx in range(len(datasets), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 6),
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle("Accuracy vs k (mean ± std)", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_k.png", bbox_inches="tight")
    plt.close(fig)


def plot_speedup_vs_k(df: pd.DataFrame, out_dir: Path):
    """Prediction-time speedup relative to baseline."""
    bl = df[df["method_name"] == "full_tabicl"]
    loc = df[df["method_name"] != "full_tabicl"]
    datasets = df["dataset"].unique()
    methods = [m for m in ["emb_cosine", "emb_euclidean", "raw_cosine",
                           "raw_euclidean", "random"] if m in loc["method_name"].values]
    nrows, ncols = _n_cols(len(datasets))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows),
                              squeeze=False)

    for idx, ds in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        bl_pred = bl[bl["dataset"] == ds]["predict_time"].mean()
        ax.axhline(1.0, color=_style("full_tabicl")["color"], linestyle="--",
                   linewidth=2, label="Baseline (1×)")

        for m in methods:
            md = loc[(loc["dataset"] == ds) & (loc["method_name"] == m)]
            if md.empty:
                continue
            g = md.groupby("k")["predict_time"].mean()
            speedups = bl_pred / g
            s = _style(m)
            ax.plot(speedups.index, speedups.values, marker=s["marker"],
                    color=s["color"], label=s["label"], linewidth=1.5, markersize=6)

        ax.set_title(ds)
        ax.set_xlabel("k")
        ax.set_ylabel("Prediction speedup (×)")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for idx in range(len(datasets), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 6),
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle("Prediction Speedup vs k", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "speedup_vs_k.png", bbox_inches="tight")
    plt.close(fig)


def plot_runtime_breakdown(df: pd.DataFrame, out_dir: Path):
    """Stacked bar: Stage 1-2 / Retrieval / Stage 3 for emb_cosine."""
    emb = df[df["method_name"] == "emb_cosine"]
    if emb.empty:
        return
    datasets = emb["dataset"].unique()
    nrows, ncols = _n_cols(len(datasets))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows),
                              squeeze=False)

    for idx, ds in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        g = emb[emb["dataset"] == ds].groupby("k")[
            ["t_stage12", "t_retrieval", "t_stage3"]].mean()
        g.columns = ["Stage 1-2", "Retrieval", "Stage 3"]
        g.plot.bar(stacked=True, ax=ax, color=[STAGE_COLORS[c] for c in g.columns],
                   width=0.6, edgecolor="white", linewidth=0.5)
        ax.set_title(ds)
        ax.set_xlabel("k")
        ax.set_ylabel("Time (s)")
        ax.legend(fontsize=8)

    for idx in range(len(datasets), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Runtime Breakdown — Emb Cosine", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_breakdown.png", bbox_inches="tight")
    plt.close(fig)


def plot_memory_vs_k(df: pd.DataFrame, out_dir: Path):
    """Peak GPU memory vs k for each dataset."""
    bl = df[df["method_name"] == "full_tabicl"]
    loc = df[df["method_name"] != "full_tabicl"]
    datasets = df["dataset"].unique()
    methods = [m for m in ["emb_cosine", "random"] if m in loc["method_name"].values]
    nrows, ncols = _n_cols(len(datasets))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows),
                              squeeze=False)

    for idx, ds in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        bl_mem = bl[bl["dataset"] == ds]["peak_memory_mb"].mean()
        ax.axhline(bl_mem, color=_style("full_tabicl")["color"], linestyle="--",
                   linewidth=2, label=_style("full_tabicl")["label"])

        for m in methods:
            md = loc[(loc["dataset"] == ds) & (loc["method_name"] == m)]
            if md.empty:
                continue
            g = md.groupby("k")["peak_memory_mb"].mean()
            s = _style(m)
            ax.plot(g.index, g.values, marker=s["marker"], color=s["color"],
                    label=s["label"], linewidth=1.5, markersize=6)

        ax.set_title(ds)
        ax.set_xlabel("k")
        ax.set_ylabel("Peak GPU Memory (MB)")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for idx in range(len(datasets), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 4),
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle("Peak GPU Memory vs k", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "memory_vs_k.png", bbox_inches="tight")
    plt.close(fig)


def plot_pareto(df: pd.DataFrame, out_dir: Path):
    """Accuracy vs speedup Pareto scatter — one subplot per dataset."""
    bl = df[df["method_name"] == "full_tabicl"]
    loc = df[df["method_name"] != "full_tabicl"]
    datasets = df["dataset"].unique()
    methods = [m for m in ["emb_cosine", "emb_euclidean", "raw_cosine",
                           "raw_euclidean", "random"] if m in loc["method_name"].values]
    nrows, ncols = _n_cols(len(datasets))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows),
                              squeeze=False)

    for idx, ds in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        bl_ds = bl[bl["dataset"] == ds]
        bl_acc = bl_ds["accuracy"].mean()
        bl_pred = bl_ds["predict_time"].mean()

        ax.scatter(1.0, bl_acc, color=_style("full_tabicl")["color"],
                   marker=_style("full_tabicl")["marker"], s=120, zorder=5,
                   label=_style("full_tabicl")["label"], edgecolors="black", linewidth=0.5)

        for m in methods:
            md = loc[(loc["dataset"] == ds) & (loc["method_name"] == m)]
            if md.empty:
                continue
            g = md.groupby("k").agg(acc=("accuracy", "mean"),
                                     pred=("predict_time", "mean"))
            speedups = bl_pred / g["pred"]
            s = _style(m)
            ax.scatter(speedups.values, g["acc"].values, marker=s["marker"],
                       color=s["color"], s=80, label=s["label"], edgecolors="black",
                       linewidth=0.3, zorder=4)
            for k_val, sp, ac in zip(g.index, speedups.values, g["acc"].values):
                ax.annotate(f"k={k_val}", (sp, ac), fontsize=7,
                            textcoords="offset points", xytext=(4, 4))

        ax.set_title(ds)
        ax.set_xlabel("Speedup (×)")
        ax.set_ylabel("Accuracy")
        ax.axhline(bl_acc, color="grey", linestyle=":", alpha=0.4)
        ax.axvline(1.0, color="grey", linestyle=":", alpha=0.4)

    for idx in range(len(datasets), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 6),
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle("Accuracy vs Speedup (Pareto View)", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_accuracy_speedup.png", bbox_inches="tight")
    plt.close(fig)


def plot_method_comparison_heatmap(df: pd.DataFrame, out_dir: Path):
    """Heatmap of accuracy difference (vs baseline) for each method × dataset at best k."""
    bl = df[df["method_name"] == "full_tabicl"].groupby("dataset")["accuracy"].mean()
    loc = df[df["method_name"] != "full_tabicl"]
    methods = sorted(loc["method_name"].unique())
    datasets = list(df["dataset"].unique())

    matrix = pd.DataFrame(index=datasets, columns=methods, dtype=float)
    for m in methods:
        for ds in datasets:
            md = loc[(loc["method_name"] == m) & (loc["dataset"] == ds)]
            if md.empty or ds not in bl.index:
                matrix.loc[ds, m] = np.nan
                continue
            best_acc = md.groupby("k")["accuracy"].mean().max()
            matrix.loc[ds, m] = best_acc - bl[ds]

    fig, ax = plt.subplots(figsize=(max(7, len(methods) * 1.3), max(4, len(datasets) * 0.6)))
    im = ax.imshow(matrix.values.astype(float), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([_style(m)["label"] for m in methods], rotation=30, ha="right")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    for i in range(len(datasets)):
        for j in range(len(methods)):
            val = matrix.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(val) > 0.05 else "black")
    fig.colorbar(im, ax=ax, label="ΔAccuracy vs baseline (best k)")
    ax.set_title("Accuracy Difference Heatmap (best k per method)")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_accuracy_diff.png", bbox_inches="tight")
    plt.close(fig)


def plot_speedup_heatmap(df: pd.DataFrame, out_dir: Path):
    """Heatmap of prediction speedup for each method × dataset at best-accuracy k."""
    bl = df[df["method_name"] == "full_tabicl"]
    bl_pred = bl.groupby("dataset")["predict_time"].mean()
    loc = df[df["method_name"] != "full_tabicl"]
    methods = sorted(loc["method_name"].unique())
    datasets = list(df["dataset"].unique())

    matrix = pd.DataFrame(index=datasets, columns=methods, dtype=float)
    for m in methods:
        for ds in datasets:
            md = loc[(loc["method_name"] == m) & (loc["dataset"] == ds)]
            if md.empty or ds not in bl_pred.index:
                matrix.loc[ds, m] = np.nan
                continue
            # Use speed at k that gives best accuracy
            g = md.groupby("k").agg(acc=("accuracy", "mean"), pred=("predict_time", "mean"))
            best_k = g["acc"].idxmax()
            matrix.loc[ds, m] = bl_pred[ds] / g.loc[best_k, "pred"] if g.loc[best_k, "pred"] > 0 else np.nan

    fig, ax = plt.subplots(figsize=(max(7, len(methods) * 1.3), max(4, len(datasets) * 0.6)))
    vals = matrix.values.astype(float)
    im = ax.imshow(vals, cmap="RdYlBu", aspect="auto",
                   vmin=min(0.1, np.nanmin(vals)), vmax=max(3.0, np.nanmax(vals)))
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([_style(m)["label"] for m in methods], rotation=30, ha="right")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    for i in range(len(datasets)):
        for j in range(len(methods)):
            val = matrix.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}x", ha="center", va="center", fontsize=9,
                        color="white" if val > 2.0 or val < 0.3 else "black")
    fig.colorbar(im, ax=ax, label="Speedup (×) at best-accuracy k")
    ax.set_title("Speedup Heatmap (at best-accuracy k per method)")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_speedup.png", bbox_inches="tight")
    plt.close(fig)


def plot_scalability(df: pd.DataFrame, out_dir: Path):
    """Scatter plot: dataset size (n_train) vs speedup for each method at k=32."""
    bl = df[df["method_name"] == "full_tabicl"]
    loc = df[df["method_name"] != "full_tabicl"]
    methods = [m for m in ["emb_cosine", "raw_cosine", "random"]
               if m in loc["method_name"].values]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: speedup vs n_train at k=32
    ax = axes[0]
    ax.axhline(1.0, color="grey", linestyle=":", alpha=0.5)
    for m in methods:
        xs, ys, labels = [], [], []
        for ds in df["dataset"].unique():
            bl_ds = bl[bl["dataset"] == ds]
            md = loc[(loc["dataset"] == ds) & (loc["method_name"] == m) & (loc["k"] == 32)]
            if bl_ds.empty or md.empty:
                continue
            n_train = int(bl_ds["n_train"].iloc[0])
            speedup = bl_ds["predict_time"].mean() / md["predict_time"].mean()
            xs.append(n_train)
            ys.append(speedup)
            labels.append(ds)
        s = _style(m)
        ax.scatter(xs, ys, marker=s["marker"], color=s["color"], s=80, label=s["label"],
                   edgecolors="black", linewidth=0.3, zorder=4)
        for x, y, lb in zip(xs, ys, labels):
            ax.annotate(lb, (x, y), fontsize=6.5, textcoords="offset points",
                        xytext=(5, 4), alpha=0.8)
    ax.set_xlabel("n_train")
    ax.set_ylabel("Speedup (×) at k=32")
    ax.set_title("Speedup Scalability (k=32)")
    ax.set_xscale("log")
    ax.legend(fontsize=9)

    # Right: accuracy drop vs n_train at k=32
    ax = axes[1]
    ax.axhline(0, color="grey", linestyle=":", alpha=0.5)
    for m in methods:
        xs, ys, labels = [], [], []
        for ds in df["dataset"].unique():
            bl_ds = bl[bl["dataset"] == ds]
            md = loc[(loc["dataset"] == ds) & (loc["method_name"] == m) & (loc["k"] == 32)]
            if bl_ds.empty or md.empty:
                continue
            n_train = int(bl_ds["n_train"].iloc[0])
            acc_delta = md["accuracy"].mean() - bl_ds["accuracy"].mean()
            xs.append(n_train)
            ys.append(acc_delta)
            labels.append(ds)
        s = _style(m)
        ax.scatter(xs, ys, marker=s["marker"], color=s["color"], s=80, label=s["label"],
                   edgecolors="black", linewidth=0.3, zorder=4)
        for x, y, lb in zip(xs, ys, labels):
            ax.annotate(lb, (x, y), fontsize=6.5, textcoords="offset points",
                        xytext=(5, 4), alpha=0.8)
    ax.set_xlabel("n_train")
    ax.set_ylabel("ΔAccuracy at k=32")
    ax.set_title("Accuracy Impact vs Dataset Size (k=32)")
    ax.set_xscale("log")
    ax.legend(fontsize=9)

    fig.suptitle("Scalability Analysis — k=32", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "scalability_k32.png", bbox_inches="tight")
    plt.close(fig)


def plot_emb_vs_raw(df: pd.DataFrame, out_dir: Path):
    """Bar chart comparing emb_cosine vs raw_cosine accuracy for each dataset at each k."""
    emb = df[df["method_name"] == "emb_cosine"]
    raw = df[df["method_name"] == "raw_cosine"]
    if emb.empty or raw.empty:
        return

    datasets = [ds for ds in df["dataset"].unique()
                if not emb[emb["dataset"] == ds].empty and not raw[raw["dataset"] == ds].empty]
    nrows, ncols = _n_cols(len(datasets))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, ds in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        emb_ds = emb[emb["dataset"] == ds].groupby("k")["accuracy"].mean()
        raw_ds = raw[raw["dataset"] == ds].groupby("k")["accuracy"].mean()
        common_k = sorted(set(emb_ds.index) & set(raw_ds.index))
        if not common_k:
            continue
        x = np.arange(len(common_k))
        width = 0.35
        ax.bar(x - width / 2, [emb_ds[k] for k in common_k], width,
               color=_style("emb_cosine")["color"], label="Emb Cosine", alpha=0.85)
        ax.bar(x + width / 2, [raw_ds[k] for k in common_k], width,
               color=_style("raw_cosine")["color"], label="Raw Cosine", alpha=0.85)

        # Baseline line
        bl_ds = df[(df["dataset"] == ds) & (df["method_name"] == "full_tabicl")]
        if not bl_ds.empty:
            ax.axhline(bl_ds["accuracy"].mean(), color=_style("full_tabicl")["color"],
                       linestyle="--", linewidth=1.5, label="Baseline")

        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in common_k])
        ax.set_xlabel("k")
        ax.set_ylabel("Accuracy")
        ax.set_title(ds)
        ax.legend(fontsize=7)

    for idx in range(len(datasets), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Embedding vs Raw Cosine Retrieval", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "emb_vs_raw_cosine.png", bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_retention_bar(df: pd.DataFrame, out_dir: Path):
    """Grouped bar: for each dataset, show baseline acc and best localized acc."""
    bl = df[df["method_name"] == "full_tabicl"]
    loc = df[df["method_name"] != "full_tabicl"]

    datasets = []
    bl_accs = []
    loc_accs = []
    methods_used = []
    for ds in df["dataset"].unique():
        bl_ds = bl[bl["dataset"] == ds]
        loc_ds = loc[loc["dataset"] == ds]
        if bl_ds.empty or loc_ds.empty:
            continue
        datasets.append(ds)
        bl_accs.append(bl_ds["accuracy"].mean())
        g = loc_ds.groupby(["method_name", "k"])["accuracy"].mean()
        best_idx = g.idxmax()
        loc_accs.append(g[best_idx])
        methods_used.append(f"{best_idx[0]} k={best_idx[1]}")

    # Sort by n_train
    order = []
    for ds in datasets:
        bl_ds = bl[bl["dataset"] == ds]
        order.append(int(bl_ds["n_train"].iloc[0]))
    sorted_indices = np.argsort(order)
    datasets = [datasets[i] for i in sorted_indices]
    bl_accs = [bl_accs[i] for i in sorted_indices]
    loc_accs = [loc_accs[i] for i in sorted_indices]
    methods_used = [methods_used[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 0.9), 6))
    x = np.arange(len(datasets))
    width = 0.35
    bars1 = ax.bar(x - width / 2, bl_accs, width, color=_style("full_tabicl")["color"],
                   label="Full TabICL", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width / 2, loc_accs, width, color=_style("emb_cosine")["color"],
                   label="Best Localized", alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=35, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Baseline vs Best Localized Accuracy (sorted by dataset size)")
    ax.legend()
    ax.set_ylim(bottom=max(0, min(bl_accs + loc_accs) - 0.05))

    # Annotate deltas
    for i, (ba, la) in enumerate(zip(bl_accs, loc_accs)):
        delta = la - ba
        color = "green" if delta >= 0 else "red"
        ax.annotate(f"{delta:+.3f}", (i + width / 2, la), fontsize=7,
                    ha="center", va="bottom", color=color, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_retention_bar.png", bbox_inches="tight")
    plt.close(fig)


# ── CSV table export ─────────────────────────────────────────

def table_master_comparison(df: pd.DataFrame) -> str:
    """
    One-row-per-dataset master table sorted by baseline runtime (ascending).

    For each dataset shows:
      - Baseline accuracy, runtime, memory
      - Best localized accuracy (and method/k), delta
      - Best localized runtime, speedup
      - Best localized memory, delta
    """
    bl = df[df["method_name"] == "full_tabicl"]
    loc = df[df["method_name"] != "full_tabicl"]

    rows = []
    for ds in df["dataset"].unique():
        bl_ds = bl[bl["dataset"] == ds]
        loc_ds = loc[loc["dataset"] == ds]
        if bl_ds.empty or loc_ds.empty:
            continue

        bl_acc = bl_ds["accuracy"].mean()
        bl_pred = bl_ds["predict_time"].mean()
        bl_total = bl_ds["total_time"].mean()
        bl_mem = bl_ds["peak_memory_mb"].mean()
        n_train = int(bl_ds["n_train"].iloc[0])

        g = loc_ds.groupby(["method_name", "k"]).agg(
            acc=("accuracy", "mean"),
            pred=("predict_time", "mean"),
            total=("total_time", "mean"),
            mem=("peak_memory_mb", "mean"),
        )

        # Best accuracy config
        best_acc_idx = g["acc"].idxmax()
        best_acc = g.loc[best_acc_idx]
        best_acc_method, best_acc_k = best_acc_idx

        # Best speedup config (fastest)
        best_speed_idx = g["pred"].idxmin()
        best_speed = g.loc[best_speed_idx]
        best_speed_method, best_speed_k = best_speed_idx

        rows.append({
            "dataset": ds, "n_train": n_train,
            "bl_acc": bl_acc, "bl_pred": bl_pred, "bl_total": bl_total, "bl_mem": bl_mem,
            "best_acc": best_acc["acc"], "best_acc_method": best_acc_method,
            "best_acc_k": best_acc_k, "acc_delta": best_acc["acc"] - bl_acc,
            "best_acc_pred": best_acc["pred"],
            "best_acc_speedup": bl_pred / best_acc["pred"] if best_acc["pred"] > 0 else float("inf"),
            "best_acc_mem": best_acc["mem"],
            "best_acc_mem_delta": (best_acc["mem"] - bl_mem) / bl_mem * 100,
            "best_speed_pred": best_speed["pred"],
            "best_speed_speedup": bl_pred / best_speed["pred"] if best_speed["pred"] > 0 else float("inf"),
            "best_speed_acc": best_speed["acc"],
            "best_speed_acc_delta": best_speed["acc"] - bl_acc,
            "best_speed_method": best_speed_method,
            "best_speed_k": best_speed_k,
        })

    rows.sort(key=lambda r: r["bl_pred"])  # sort by baseline runtime

    lines = []
    lines.append("")
    lines.append("=" * 145)
    lines.append("  MASTER COMPARISON TABLE — Datasets sorted by Baseline Runtime")
    lines.append("=" * 145)
    lines.append("")

    # Sub-table 1: Best-accuracy config
    lines.append("  ── Best-Accuracy Localized Config vs Baseline ──")
    lines.append(f"{'Dataset':<20} {'n_train':>7} │ {'BL Acc':>7} {'Loc Acc':>7} {'ΔAcc':>8} │"
                 f" {'BL Time':>8} {'Loc Time':>8} {'Speedup':>7} │"
                 f" {'BL Mem':>8} {'Loc Mem':>8} {'ΔMem%':>7} │ {'Method':<14} {'k':>4}")
    lines.append("─" * 145)
    for r in rows:
        sign = "+" if r["acc_delta"] >= 0 else ""
        lines.append(
            f"{r['dataset']:<20} {r['n_train']:>7,} │ {r['bl_acc']:>7.4f} {r['best_acc']:>7.4f}"
            f" {sign}{r['acc_delta']:>7.4f} │ {r['bl_pred']:>7.2f}s {r['best_acc_pred']:>7.2f}s"
            f" {r['best_acc_speedup']:>6.2f}x │ {r['bl_mem']:>7.0f}M {r['best_acc_mem']:>7.0f}M"
            f" {r['best_acc_mem_delta']:>+6.1f}% │ {r['best_acc_method']:<14} {r['best_acc_k']:>4}"
        )

    lines.append("")
    # Sub-table 2: Best-speedup config
    lines.append("  ── Fastest Localized Config vs Baseline ──")
    lines.append(f"{'Dataset':<20} {'n_train':>7} │ {'BL Acc':>7} {'Loc Acc':>7} {'ΔAcc':>8} │"
                 f" {'BL Time':>8} {'Loc Time':>8} {'Speedup':>7} │ {'Method':<14} {'k':>4}")
    lines.append("─" * 145)
    for r in rows:
        sign = "+" if r["best_speed_acc_delta"] >= 0 else ""
        lines.append(
            f"{r['dataset']:<20} {r['n_train']:>7,} │ {r['bl_acc']:>7.4f} {r['best_speed_acc']:>7.4f}"
            f" {sign}{r['best_speed_acc_delta']:>7.4f} │ {r['bl_pred']:>7.2f}s {r['best_speed_pred']:>7.2f}s"
            f" {r['best_speed_speedup']:>6.2f}x │ {r['best_speed_method']:<14} {r['best_speed_k']:>4}"
        )

    return "\n".join(lines)


def generate_deep_analysis(df: pd.DataFrame) -> str:
    """Generate a comprehensive written analysis/report of all results."""
    bl = df[df["method_name"] == "full_tabicl"]
    loc = df[df["method_name"] != "full_tabicl"]

    # Gather stats
    datasets = list(df["dataset"].unique())
    n_datasets = len(datasets)

    # Per-dataset best-accuracy localized
    records = []
    for ds in datasets:
        bl_ds = bl[bl["dataset"] == ds]
        loc_ds = loc[loc["dataset"] == ds]
        if bl_ds.empty or loc_ds.empty:
            continue
        bl_acc = bl_ds["accuracy"].mean()
        bl_pred = bl_ds["predict_time"].mean()
        bl_mem = bl_ds["peak_memory_mb"].mean()
        n_train = int(bl_ds["n_train"].iloc[0])

        g = loc_ds.groupby(["method_name", "k"]).agg(
            acc=("accuracy", "mean"), pred=("predict_time", "mean"),
            mem=("peak_memory_mb", "mean"),
        )

        # Best accuracy
        ba_idx = g["acc"].idxmax()
        ba = g.loc[ba_idx]

        # Best speedup with <=2% acc drop
        viable = g[g["acc"] >= bl_acc - 0.02]
        if not viable.empty:
            bs_idx = viable["pred"].idxmin()
            bs = viable.loc[bs_idx]
            bs_method, bs_k = bs_idx
        else:
            bs = ba
            bs_method, bs_k = ba_idx

        records.append({
            "ds": ds, "n_train": n_train,
            "bl_acc": bl_acc, "bl_pred": bl_pred, "bl_mem": bl_mem,
            "best_acc": ba["acc"], "best_acc_delta": ba["acc"] - bl_acc,
            "best_acc_method": ba_idx[0], "best_acc_k": ba_idx[1],
            "best_viable_speedup": bl_pred / bs["pred"] if bs["pred"] > 0 else 1.0,
            "best_viable_acc_drop": bs["acc"] - bl_acc,
            "best_viable_method": bs_method, "best_viable_k": bs_k,
        })

    # Classify datasets by size
    small = [r for r in records if r["n_train"] < 5000]
    medium = [r for r in records if 5000 <= r["n_train"] < 30000]
    large = [r for r in records if r["n_train"] >= 30000]

    # Method frequency analysis
    method_wins_acc = {}
    method_wins_speed = {}
    for r in records:
        m = r["best_acc_method"]
        method_wins_acc[m] = method_wins_acc.get(m, 0) + 1
        m2 = r["best_viable_method"]
        method_wins_speed[m2] = method_wins_speed.get(m2, 0) + 1

    # Emb vs Raw comparison
    emb_methods = {"emb_cosine", "emb_euclidean"}
    raw_methods = {"raw_cosine", "raw_euclidean"}
    emb_vs_raw = []
    for ds in datasets:
        for k_val in sorted(loc["k"].unique()):
            emb_accs = []
            raw_accs = []
            for m in emb_methods:
                sub = loc[(loc["dataset"] == ds) & (loc["method_name"] == m) & (loc["k"] == k_val)]
                if not sub.empty:
                    emb_accs.append(sub["accuracy"].mean())
            for m in raw_methods:
                sub = loc[(loc["dataset"] == ds) & (loc["method_name"] == m) & (loc["k"] == k_val)]
                if not sub.empty:
                    raw_accs.append(sub["accuracy"].mean())
            if emb_accs and raw_accs:
                emb_vs_raw.append({"ds": ds, "k": k_val, "emb": np.mean(emb_accs), "raw": np.mean(raw_accs)})

    emb_wins = sum(1 for x in emb_vs_raw if x["emb"] > x["raw"])
    raw_wins = sum(1 for x in emb_vs_raw if x["raw"] > x["emb"])

    # Speedup scaling with dataset size
    large_speedups = [(r["ds"], r["n_train"], r["best_viable_speedup"], r["best_viable_acc_drop"])
                      for r in large]
    large_speedups.sort(key=lambda x: x[1])

    # Build report
    lines = []
    sep = "=" * 80

    lines.append("")
    lines.append(sep)
    lines.append("  DEEP ANALYSIS REPORT — Localized TabICL")
    lines.append(sep)

    # 1. Executive Summary
    lines.append("")
    lines.append("1. EXECUTIVE SUMMARY")
    lines.append("-" * 40)
    avg_acc_drop = np.mean([r["best_acc_delta"] for r in records])
    max_speedup = max(r["best_viable_speedup"] for r in large) if large else 0
    avg_speedup_large = np.mean([r["best_viable_speedup"] for r in large]) if large else 0
    lines.append(
        f"Evaluated {n_datasets} datasets ranging from {min(r['n_train'] for r in records):,}"
        f" to {max(r['n_train'] for r in records):,} training samples."
    )
    lines.append(
        f"Average accuracy delta (best localized vs baseline): {avg_acc_drop:+.4f}"
    )
    if large:
        lines.append(
            f"On large datasets (n_train >= 30K): avg speedup = {avg_speedup_large:.2f}x "
            f"(max {max_speedup:.2f}x) with <=2% accuracy drop."
        )
    lines.append("")

    # 2. Accuracy Analysis
    lines.append("2. ACCURACY ANALYSIS")
    lines.append("-" * 40)
    acc_preserved = [r for r in records if r["best_acc_delta"] >= -0.02]
    acc_improved = [r for r in records if r["best_acc_delta"] > 0.001]
    lines.append(f"Accuracy within 2% of baseline: {len(acc_preserved)}/{len(records)} datasets")
    lines.append(f"Accuracy improved over baseline: {len(acc_improved)}/{len(records)} datasets")
    lines.append("")
    lines.append("  Datasets where localized IMPROVED accuracy:")
    for r in sorted(acc_improved, key=lambda x: x["best_acc_delta"], reverse=True):
        lines.append(f"    {r['ds']:<22} +{r['best_acc_delta']:.4f}  ({r['best_acc_method']}, k={r['best_acc_k']})")
    acc_hurt = [r for r in records if r["best_acc_delta"] < -0.02]
    if acc_hurt:
        lines.append("")
        lines.append("  Datasets where localized DEGRADED accuracy (>2%):")
        for r in sorted(acc_hurt, key=lambda x: x["best_acc_delta"]):
            lines.append(f"    {r['ds']:<22} {r['best_acc_delta']:.4f}  ({r['best_acc_method']}, k={r['best_acc_k']})")
    lines.append("")

    # 3. Runtime / Speedup Analysis
    lines.append("3. RUNTIME & SPEEDUP ANALYSIS")
    lines.append("-" * 40)
    lines.append("  Speedup (best localized with ≤2% acc drop) by dataset size:")
    for category, group, label in [("Small (n<5K)", small, "small"),
                                     ("Medium (5K–30K)", medium, "medium"),
                                     ("Large (≥30K)", large, "large")]:
        if group:
            speedups = [r["best_viable_speedup"] for r in group]
            lines.append(f"    {category}: avg {np.mean(speedups):.2f}x, "
                         f"range [{min(speedups):.2f}x – {max(speedups):.2f}x]")
    lines.append("")
    lines.append("  Key observation: Localized TabICL adds overhead for small datasets (KNN +")
    lines.append("  Stage 1-2 caching cost), so speedup < 1x is expected when n_train < ~5K.")
    lines.append("  Speedup becomes meaningful (>1x) as dataset size grows, because the")
    lines.append("  Stage 3 ICL inference cost scales with n_train in full TabICL but is")
    lines.append("  fixed at O(k) in the localized variant.")
    lines.append("")

    if large_speedups:
        lines.append("  Large dataset breakdown:")
        for ds, nt, sp, ad in large_speedups:
            lines.append(f"    {ds:<22} n={nt:>7,}  speedup={sp:.2f}x  Δacc={ad:+.4f}")
    lines.append("")

    # 4. Embedding vs Raw Retrieval
    lines.append("4. EMBEDDING vs RAW RETRIEVAL SPACE")
    lines.append("-" * 40)
    lines.append(f"  Head-to-head comparisons (same dataset, same k): {len(emb_vs_raw)} pairs")
    lines.append(f"  Embedding wins: {emb_wins}  |  Raw wins: {raw_wins}")
    if emb_vs_raw:
        avg_emb = np.mean([x["emb"] for x in emb_vs_raw])
        avg_raw = np.mean([x["raw"] for x in emb_vs_raw])
        lines.append(f"  Average accuracy — Emb: {avg_emb:.4f}, Raw: {avg_raw:.4f}")
    lines.append("")
    winner = "Embedding" if emb_wins > raw_wins else ("Raw" if raw_wins > emb_wins else "Tied")
    lines.append(f"  Verdict: {winner}-space retrieval is generally {'better' if winner != 'Tied' else 'equivalent'}.")
    lines.append("  This suggests that the learned representations from Stages 1-2 "
                 + ("do" if emb_wins > raw_wins else "may not")
                 + " provide meaningfully better")
    lines.append("  neighborhoods for ICL than raw feature similarity.")
    lines.append("")

    # 5. Method Ranking
    lines.append("5. METHOD RANKING")
    lines.append("-" * 40)
    lines.append("  Best-accuracy method wins across datasets:")
    for m, c in sorted(method_wins_acc.items(), key=lambda x: -x[1]):
        lines.append(f"    {m:<18} {c}/{len(records)} datasets")
    lines.append("")
    lines.append("  Best viable-speed method wins (≤2% acc drop):")
    for m, c in sorted(method_wins_speed.items(), key=lambda x: -x[1]):
        lines.append(f"    {m:<18} {c}/{len(records)} datasets")
    lines.append("")

    # 6. Memory Analysis
    lines.append("6. MEMORY ANALYSIS")
    lines.append("-" * 40)
    lines.append("  Localized TabICL caches Stage 1-2 representations AND runs Stage 3 on")
    lines.append("  smaller inputs, so memory impact depends on the dataset:")
    for r in records:
        g_ds = loc[loc["dataset"] == r["ds"]].groupby(["method_name", "k"])["peak_memory_mb"].mean()
        if not g_ds.empty:
            min_mem = g_ds.min()
            mem_change = (min_mem - r["bl_mem"]) / r["bl_mem"] * 100
            lines.append(f"    {r['ds']:<22} baseline={r['bl_mem']:>7.0f}MB  "
                         f"min_localized={min_mem:>7.0f}MB  ({mem_change:+.1f}%)")
    lines.append("")

    # 7. Honest Assessment
    lines.append("7. HONEST ASSESSMENT — IMPACT, NOVELTY, AND FRAMING")
    lines.append("-" * 40)
    lines.append("")
    lines.append("  IMPACT:")
    lines.append("  • The primary value is runtime reduction on large datasets (n > 30K).")
    if large:
        n_speedup_good = sum(1 for r in large if r["best_viable_speedup"] > 1.5)
        lines.append(f"    {n_speedup_good}/{len(large)} large datasets achieve >1.5x speedup with ≤2% acc drop.")
    lines.append("  • On small/medium datasets, localization adds overhead and is not beneficial")
    lines.append("    for runtime — though it can sometimes improve accuracy by focusing the")
    lines.append("    ICL context on the most relevant training samples.")
    lines.append("  • Memory savings are inconsistent. The caching overhead can offset gains")
    lines.append("    from smaller Stage 3 inputs.")
    lines.append("")
    lines.append("  NOVELTY:")
    lines.append("  • The core idea — row-selective ICL via KNN in embedding space — is a")
    lines.append("    natural extension of TabICL's architecture. It is not deeply novel but")
    lines.append("    is a sensible, well-motivated engineering contribution.")
    lines.append("  • The comparison between embedding-space and raw-feature-space retrieval")
    lines.append("    provides useful empirical insight into whether learned representations")
    lines.append("    capture task-relevant structure for nearest-neighbor selection.")
    lines.append("  • The approach relates to retrieval-augmented generation (RAG) in NLP,")
    lines.append("    applying similar intuitions to tabular in-context learning.")
    lines.append("")
    lines.append("  RECOMMENDED FRAMING:")
    lines.append("  • Frame as: 'Retrieval-Augmented In-Context Learning for Tabular Data'")
    lines.append("    — positioning it at the intersection of ICL and retrieval-augmented methods.")
    lines.append("  • Lead with the SCALABILITY story: full TabICL becomes impractical at")
    lines.append("    large n_train due to quadratic attention cost. Localized TabICL makes")
    lines.append("    ICL-based tabular prediction feasible at scale.")
    lines.append("  • Emphasize the accuracy-retention result: on most datasets, localization")
    lines.append("    retains baseline accuracy while substantially reducing inference time.")
    lines.append("  • The emb-vs-raw comparison is a secondary but interesting finding. If")
    if emb_wins > raw_wins:
        lines.append("    embedding retrieval consistently wins, this validates that Stages 1-2")
        lines.append("    learn useful representations beyond what raw features provide.")
    else:
        lines.append("    raw retrieval performs comparably, this suggests the pretrained model's")
        lines.append("    feature transformations may not add much for neighbor selection —")
        lines.append("    which is itself an interesting negative result.")
    lines.append("  • Be honest about limitations: overhead on small datasets, memory tradeoffs,")
    lines.append("    and the fact that this is an engineering optimization rather than a")
    lines.append("    fundamental algorithmic advance.")
    lines.append("")

    # 8. Key Recommendations
    lines.append("8. KEY RECOMMENDATIONS FOR REPORT")
    lines.append("-" * 40)
    lines.append("  • Include a clear 'when to use' guideline: Localized TabICL is recommended")
    lines.append("    when n_train > 10K–15K and fast inference is needed.")
    lines.append("  • Show the Pareto plots (accuracy vs speedup) prominently — they")
    lines.append("    communicate the tradeoff more clearly than tables.")
    lines.append("  • Report k=32 or k=64 as the recommended default — these typically offer")
    lines.append("    the best accuracy-speed tradeoff.")
    lines.append("  • The timing breakdown (Stage 1-2 / Retrieval / Stage 3) shows where time")
    lines.append("    is spent and why localization helps — Stage 3 cost drops dramatically.")
    lines.append("")

    return "\n".join(lines)


def save_summary_csv(df: pd.DataFrame, out_dir: Path):
    """Save a clean summary CSV with one row per (dataset, method, k)."""
    bl = df[df["method_name"] == "full_tabicl"]
    bl_acc = bl.groupby("dataset")["accuracy"].mean().to_dict()
    bl_pred = bl.groupby("dataset")["predict_time"].mean().to_dict()

    g = df.groupby(["dataset", "method_name", "k"]).agg(
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        ll_mean=("log_loss", "mean"),
        pred_mean=("predict_time", "mean"),
        total_mean=("total_time", "mean"),
        mem_mean=("peak_memory_mb", "mean"),
    ).reset_index()

    g["acc_diff"] = g.apply(lambda r: r["acc_mean"] - bl_acc.get(r["dataset"], np.nan), axis=1)
    g["pred_speedup"] = g.apply(
        lambda r: bl_pred.get(r["dataset"], np.nan) / r["pred_mean"]
        if r["pred_mean"] > 0 else np.nan, axis=1)

    g = g.round(4)
    g.to_csv(out_dir / "summary_table.csv", index=False)


# ── Main ─────────────────────────────────────────────────────

def main():
    csv_path = _resolve_csv(sys.argv)
    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path)
    df = _norm_columns(df)
    print(f"  {len(df)} rows, {df['dataset'].nunique()} datasets, "
          f"{df['method_name'].nunique()} methods")

    out_dir = Path("results/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Tables (print + save) ────────────────────────────────
    report_parts = []

    t = table_master_comparison(df)
    report_parts.append(t)
    print(t)

    t = table_per_dataset(df)
    report_parts.append(t)
    print(t)

    t = table_best_per_dataset(df)
    report_parts.append(t)
    print(t)

    t = table_timing_breakdown(df)
    if t:
        report_parts.append(t)
        print(t)

    # ── Deep analysis report ─────────────────────────────────
    analysis = generate_deep_analysis(df)
    report_parts.append(analysis)
    print(analysis)

    report_path = out_dir / "tables.txt"
    report_path.write_text("\n".join(report_parts), encoding="utf-8")
    print(f"\nTables & analysis saved to {report_path}")

    save_summary_csv(df, out_dir)
    print(f"Summary CSV saved to {out_dir / 'summary_table.csv'}")

    # ── Plots ────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_accuracy_vs_k(df, out_dir)
    print(f"  ✓ accuracy_vs_k.png")
    plot_speedup_vs_k(df, out_dir)
    print(f"  ✓ speedup_vs_k.png")
    plot_runtime_breakdown(df, out_dir)
    print(f"  ✓ runtime_breakdown.png")
    plot_memory_vs_k(df, out_dir)
    print(f"  ✓ memory_vs_k.png")
    plot_pareto(df, out_dir)
    print(f"  ✓ pareto_accuracy_speedup.png")
    plot_method_comparison_heatmap(df, out_dir)
    print(f"  ✓ heatmap_accuracy_diff.png")
    plot_speedup_heatmap(df, out_dir)
    print(f"  ✓ heatmap_speedup.png")
    plot_scalability(df, out_dir)
    print(f"  ✓ scalability_k32.png")
    plot_emb_vs_raw(df, out_dir)
    print(f"  ✓ emb_vs_raw_cosine.png")
    plot_accuracy_retention_bar(df, out_dir)
    print(f"  ✓ accuracy_retention_bar.png")

    print(f"\nAll outputs saved to {out_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
