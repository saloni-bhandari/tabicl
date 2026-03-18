import torch
import numpy as np
import pandas as pd
import random
import time
import os
import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score

from model.tabicl import TabICL


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "step-10000.ckpt"

MAX_CLASSES = 10

N_PERMUTATIONS = 32
N_SPLITS = 5

CUSTOM_DATASETS_DIR = "text_datasets"

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def load_custom_datasets(root_dir):
    datasets = {}
    for folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        x_files = glob.glob(os.path.join(folder_path, "*_X.npy"))
        y_files = glob.glob(os.path.join(folder_path, "*_y.npy"))

        if len(x_files) != 1 or len(y_files) != 1:
            print(f"  [SKIP] {folder}: expected 1 *_X.npy and 1 *_y.npy, "
                  f"found {len(x_files)} and {len(y_files)}")
            continue

        X_raw = np.load(x_files[0], allow_pickle=True)
        X = pd.DataFrame(X_raw).apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64).copy()

        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        X = X.astype(np.float32)

        y = LabelEncoder().fit_transform(
            np.load(y_files[0], allow_pickle=True).astype(str)
        ).astype(np.int64)

        datasets[folder] = (X, y)
        print(f"  [OK] {folder}: X={X.shape}, classes={len(np.unique(y))}")

    return datasets


def preprocess(X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    d = X_train.shape[1]

    return X_train, X_test, y_train, y_test, d


def permute_labels(y):
    classes = np.unique(y)
    perm = np.random.permutation(classes)

    mapping = {c: perm[i] for i, c in enumerate(classes)}
    inv_mapping = {perm[i]: c for i, c in enumerate(classes)}

    y_perm = np.array([mapping[v] for v in y])

    return y_perm, mapping, inv_mapping


def invert_probs(probs, inv_mapping):
    reordered = np.zeros_like(probs)
    for perm_class, original_class in inv_mapping.items():
        reordered[:, original_class] = probs[:, perm_class]
    return reordered


def compute_roc_auc(y_true, probs, num_classes):
    probs = probs[:, :num_classes]

    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    probs = probs / row_sums

    if num_classes == 2:
        try:
            return roc_auc_score(y_true, probs[:, 1])
        except ValueError:
            return float("nan")
    else:
        try:
            return roc_auc_score(
                y_true,
                probs,
                multi_class="ovo",
                average="macro",
                labels=list(range(num_classes)),
            )
        except ValueError:
            return float("nan")


def evaluate_dataset(model, X_train, y_train, X_test, y_test, d):
    model.eval()

    num_classes = len(np.unique(y_train))

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    d_tensor = torch.tensor([d], device=DEVICE)

    probs_all = []

    for _ in range(N_PERMUTATIONS):
        col_perm = torch.randperm(d)

        X_train_perm = X_train_t.clone()
        X_test_perm = X_test_t.clone()
        X_train_perm[:, :d] = X_train_t[:, col_perm]
        X_test_perm[:, :d] = X_test_t[:, col_perm]

        y_perm, _, inv_mapping = permute_labels(y_train_t.cpu().numpy())
        y_perm_t = torch.tensor(y_perm, dtype=torch.long, device=DEVICE)

        X_input = torch.cat([X_train_perm, X_test_perm], dim=0).unsqueeze(0)
        y_input = y_perm_t.unsqueeze(0)

        with torch.no_grad():
            logits = model(X_input, y_input, d=d_tensor)
            probs = torch.softmax(logits[0], dim=-1).cpu().numpy()

        probs = invert_probs(probs, inv_mapping)
        probs_all.append(probs)

    probs_mean = np.mean(np.stack(probs_all), axis=0)

    preds = np.argmax(probs_mean[:, :num_classes], axis=1)
    acc = (preds == y_test).mean()
    auc = compute_roc_auc(y_test, probs_mean, num_classes)

    return acc, auc


def main():
    print("Loading model...")

    model = TabICL(
        max_classes=MAX_CLASSES,
        embed_dim=128,
        col_num_blocks=3,
        col_nhead=4,
        col_num_inds=128,
        row_num_blocks=3,
        row_nhead=8,
        row_num_cls=4,
        icl_num_blocks=12,
        icl_nhead=4,
        ff_factor=2,
        dropout=0.0,
        activation="gelu",
        norm_first=True,
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    print("Model loaded.\n")

    print(f"Scanning datasets in: {CUSTOM_DATASETS_DIR}\n")
    datasets = load_custom_datasets(CUSTOM_DATASETS_DIR)

    results = {}

    for name, (X, y) in datasets.items():
        print(f"\n========== {name.upper()} ==========")

        num_classes = len(np.unique(y))
        if num_classes > MAX_CLASSES:
            print(f"  [SKIP] {num_classes} classes exceeds MAX_CLASSES={MAX_CLASSES}")
            continue

        split_accs, split_aucs = [], []

        for split in range(N_SPLITS):
            print(f"  Split {split+1}/{N_SPLITS}", end=" ")

            X_train, X_test, y_train, y_test, d = preprocess(X, y, seed=split)

            start = time.time()
            acc, auc = evaluate_dataset(model, X_train, y_train, X_test, y_test, d)
            elapsed = time.time() - start

            print(f"| Acc: {acc:.4f}  AUC: {auc:.4f}  ({elapsed:.1f}s)")
            split_accs.append(acc)
            split_aucs.append(auc)

        results[name] = {
            "acc": split_accs,
            "auc": split_aucs,
            "num_classes": num_classes,
        }

        mean_acc = np.mean(split_accs)
        std_acc = np.std(split_accs)
        mean_auc = np.nanmean(split_aucs)
        std_auc = np.nanstd(split_aucs)
        print(f"  -> Acc: {mean_acc:.4f} ± {std_acc:.4f}   AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    print("\n\n" + "=" * 72)
    print(f"{'DATASET':<30} {'CLASSES':>7} {'ACC':>8} {'±':>6} {'AUC':>8} {'±':>6}")
    print("=" * 72)

    all_accs, all_aucs = [], []

    for name, res in results.items():
        mean_acc = np.mean(res["acc"])
        std_acc = np.std(res["acc"])
        mean_auc = np.nanmean(res["auc"])
        std_auc = np.nanstd(res["auc"])
        nc = res["num_classes"]

        print(f"{name:<30} {nc:>7} {mean_acc:>8.4f} {std_acc:>6.4f} {mean_auc:>8.4f} {std_auc:>6.4f}")
        all_accs.append(mean_acc)
        all_aucs.append(mean_auc)

    print("=" * 72)
    print(f"{'MEAN':<30} {'':>7} {np.mean(all_accs):>8.4f} {'':>6} {np.nanmean(all_aucs):>8.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()