#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics as stats
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.tabicl import TabICL
from prior.dataset import PriorDataset


def str2bool(value: str) -> bool:
    return value.lower() == "true"


def train_size_type(value: str):
    value = float(value)
    if 0 < value < 1:
        return value
    if value.is_integer():
        return int(value)
    raise argparse.ArgumentTypeError(
        "Train size must be either an integer (absolute position) or a float between 0 and 1."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze class-collapse behavior for a TabICL checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint to analyze.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for model inference.")
    parser.add_argument("--prior_device", type=str, default="cpu", help="Device for prior generation.")
    parser.add_argument("--num_batches", type=int, default=12, help="Number of prior batches to sample.")
    parser.add_argument("--batch_size", type=int, default=32, help="Prior batch size.")
    parser.add_argument("--batch_size_per_gp", type=int, default=4, help="Prior group batch size.")
    parser.add_argument("--min_features", type=int, default=5)
    parser.add_argument("--max_features", type=int, default=20)
    parser.add_argument("--max_classes", type=int, default=10)
    parser.add_argument("--min_seq_len", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=100)
    parser.add_argument("--log_seq_len", type=str2bool, default=False)
    parser.add_argument("--seq_len_per_gp", type=str2bool, default=False)
    parser.add_argument("--min_train_size", type=train_size_type, default=0.1)
    parser.add_argument("--max_train_size", type=train_size_type, default=0.9)
    parser.add_argument("--replay_small", type=str2bool, default=False)
    parser.add_argument("--prior_type", type=str, default="mix_scm")
    parser.add_argument("--pretty", type=str2bool, default=True, help="Pretty-print the JSON output.")
    return parser


def summarize_matrix(name: str, mat: torch.Tensor) -> dict[str, Any]:
    row_norms = mat.norm(dim=1)
    normalized = F.normalize(mat, dim=1)
    cosine = normalized @ normalized.T
    cosine.fill_diagonal_(-1.0)
    return {
        "name": name,
        "shape": list(mat.shape),
        "min_row_norm": row_norms.min().item(),
        "max_row_norm": row_norms.max().item(),
        "mean_row_norm": row_norms.mean().item(),
        "norm_ratio": row_norms.max().item() / max(row_norms.min().item(), 1e-12),
        "max_pairwise_cosine": cosine.max().item(),
    }


def to_serializable_counter(counter: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda kv: kv[0])}


def align_batch(
    X: torch.Tensor,
    y: torch.Tensor,
    d: torch.Tensor,
    seq_len: torch.Tensor,
    train_size: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    seq = int(seq_len[0].item())
    tr = int(train_size[0].item())
    max_features = int(d.max().item())
    return X[:, :seq, :max_features], y[:, :seq], d, tr


def main() -> None:
    args = build_parser().parse_args()

    checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=True)
    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain 'config' and 'state_dict'.")

    model = TabICL(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device)
    model.eval()

    dataset = PriorDataset(
        batch_size=args.batch_size,
        batch_size_per_gp=args.batch_size_per_gp,
        min_features=args.min_features,
        max_features=args.max_features,
        max_classes=args.max_classes,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        log_seq_len=args.log_seq_len,
        seq_len_per_gp=args.seq_len_per_gp,
        min_train_size=args.min_train_size,
        max_train_size=args.max_train_size,
        replay_small=args.replay_small,
        prior_type=args.prior_type,
        device=args.prior_device,
        n_jobs=1,
    )
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
    data_iter = iter(dataloader)

    pred_counter: Counter = Counter()
    true_counter: Counter = Counter()
    top3_counter: Counter = Counter()
    unique_pred_counts = []
    entropies = []
    max_probs = []
    margins = []
    class_prob_mass = None
    batch_examples = []

    with torch.no_grad():
        for batch_idx in range(args.num_batches):
            X, y, d, seq_len, train_size = next(data_iter)
            batch = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in (X, y, d, seq_len, train_size)]
            X, y, d, seq_len, train_size = batch
            X, y, d, tr = align_batch(X, y, d, seq_len, train_size)

            X = X.to(args.device)
            y = y.to(args.device)
            d = d.to(args.device)

            logits = model(X, y[:, :tr], d=d, return_logits=True)
            probs = logits.softmax(dim=-1)
            preds = probs.argmax(dim=-1)
            y_test = y[:, tr:]

            flat_probs = probs.reshape(-1, probs.shape[-1]).cpu()
            flat_preds = preds.flatten().cpu()
            flat_true = y_test.flatten().cpu()

            pred_counter.update(flat_preds.tolist())
            true_counter.update(flat_true.tolist())
            unique_pred_counts.append(int(flat_preds.unique().numel()))
            entropies.append((-(flat_probs * flat_probs.clamp_min(1e-12).log()).sum(dim=-1)).mean().item())
            max_probs.append(flat_probs.max(dim=-1).values.mean().item())

            top2_vals = flat_probs.topk(2, dim=-1).values
            margins.append((top2_vals[:, 0] - top2_vals[:, 1]).mean().item())

            top3 = flat_probs.topk(min(3, flat_probs.shape[-1]), dim=-1).indices
            top3_counter.update(top3.reshape(-1).tolist())

            class_prob_mass = flat_probs.sum(dim=0) if class_prob_mass is None else class_prob_mass + flat_probs.sum(dim=0)
            batch_examples.append(
                {
                    "batch_idx": batch_idx,
                    "seq_len": int(seq_len[0].item()),
                    "train_size": tr,
                    "unique_pred_classes": int(flat_preds.unique().numel()),
                    "first_predictions": flat_preds[:10].tolist(),
                }
            )

    mean_class_probability = (class_prob_mass / class_prob_mass.sum()).tolist()

    head = model.icl_predictor.prediction_MLP[2].weight.detach().cpu()
    head_bias = model.icl_predictor.prediction_MLP[2].bias.detach().cpu()
    col_label_embed = model.col_embedder.label_embedding.weight.detach().cpu()
    icl_label_embed = model.icl_predictor.embedding.weight.detach().cpu()

    per_class = []
    for cls in range(head.shape[0]):
        per_class.append(
            {
                "class": cls,
                "prediction_head_norm": head[cls].norm().item(),
                "icl_label_embedding_norm": icl_label_embed[cls].norm().item(),
                "prediction_head_bias": head_bias[cls].item(),
                "mean_probability": mean_class_probability[cls],
                "argmax_count": int(pred_counter.get(cls, 0)),
                "true_count": int(true_counter.get(float(cls), true_counter.get(cls, 0))),
            }
        )

    report = {
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "model_config": checkpoint["config"],
        "analysis_config": {
            "device": args.device,
            "prior_device": args.prior_device,
            "num_batches": args.num_batches,
            "batch_size": args.batch_size,
            "prior_type": args.prior_type,
        },
        "prediction_behavior": {
            "total_pred_tokens": int(sum(pred_counter.values())),
            "classes_used_overall": int(len(pred_counter)),
            "pred_histogram": to_serializable_counter(pred_counter),
            "true_histogram": to_serializable_counter(true_counter),
            "avg_unique_pred_classes_per_batch": stats.mean(unique_pred_counts),
            "min_unique_pred_classes_per_batch": min(unique_pred_counts),
            "max_unique_pred_classes_per_batch": max(unique_pred_counts),
            "mean_entropy": stats.mean(entropies),
            "mean_max_probability": stats.mean(max_probs),
            "mean_top1_top2_margin": stats.mean(margins),
            "mean_class_probability": {str(i): p for i, p in enumerate(mean_class_probability)},
            "top3_class_frequency": to_serializable_counter(top3_counter),
            "batch_examples": batch_examples[:5],
        },
        "weights": {
            "prediction_head": summarize_matrix("prediction_head", head),
            "column_label_embedding": summarize_matrix("column_label_embedding", col_label_embed),
            "icl_label_embedding": summarize_matrix("icl_label_embedding", icl_label_embed),
            "prediction_head_bias": {
                "min": head_bias.min().item(),
                "max": head_bias.max().item(),
                "mean": head_bias.mean().item(),
                "std": head_bias.std().item(),
            },
            "per_class": per_class,
        },
    }

    indent = 2 if args.pretty else None
    print(json.dumps(report, indent=indent))


if __name__ == "__main__":
    main()
