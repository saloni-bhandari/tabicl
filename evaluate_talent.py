import torch
import numpy as np
import random
import time

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score

from model.tabicl import TabICL


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "step-10000.ckpt"

MAX_FEATURES = 100
MAX_CLASSES = 10

N_PERMUTATIONS = 32
N_SPLITS = 5


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


DATASETS = {
    "balance_scale": 11,
    "mfeat_fourier": 14,
    "breast_w": 15,
    "mfeat_karhunen": 16,
    "mfeat_morphological": 18,
    "mfeat_zernike": 22,
    "cmc": 23,
    "credit_approval": 29,
    "credit_g": 31,
    "diabetes": 37,
    "tic_tac_toe": 50,
    "vehicle": 54,
    "eucalyptus": 188,
    "analcatdata_authorship": 458,
    "analcatdata_dmft": 469,
    "pc4": 1049,
    "pc3": 1050,
    "kc2": 1063,
    "pc1": 1068,
    "banknote_authentication": 1462,
    "blood_transfusion": 1464,
    "ilpd": 1480,
    "qsar_biodeg": 1494,
    "wdc": 1510,
    "cylinder_bands": 1523,
    "steel_plates_fault": 40982,
    "climate_model_crashes": 40994,
    "car": 40975,
    "iris": 61,
}


def load_dataset(dataset_id):
    data = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")

    df = data.frame
    target_col = data.target_names[0]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    y = LabelEncoder().fit_transform(y.astype(str))

    return X.values.astype(np.float32), y.astype(np.int64)


def pad_features(X):
    if X.shape[1] < MAX_FEATURES:
        pad = np.zeros((X.shape[0], MAX_FEATURES - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])
    else:
        X = X[:, :MAX_FEATURES]
    return X


def preprocess(X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    d = X_train.shape[1]

    X_train = pad_features(X_train)
    X_test = pad_features(X_test)

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

    # Renormalize to avoid floating point issues after ensembling
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

    results = {}

    for name, dataset_id in DATASETS.items():
        print(f"\n========== {name.upper()} ==========")

        X, y = load_dataset(dataset_id)
        num_classes = len(np.unique(y))

        if num_classes > MAX_CLASSES:
            continue

        split_accs = []
        split_aucs = []

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

    all_accs = []
    all_aucs = []

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