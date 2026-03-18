import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import random

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.dummy import DummyClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

MAX_FEATURES = 100
MAX_CLASSES = 10
N_SPLITS = 5

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


def preprocess(X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if X_train.shape[1] > MAX_FEATURES:
        X_train = X_train[:, :MAX_FEATURES]
        X_test = X_test[:, :MAX_FEATURES]

    return X_train, X_test, y_train, y_test

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


def get_full_probs(probs, classes, num_classes):
    full_probs = np.zeros((probs.shape[0], num_classes))
    for i, cls in enumerate(classes):
        full_probs[:, int(cls)] = probs[:, i]
    return full_probs

def run_majority_class(X_train, y_train, X_test, y_test, num_classes):
    clf = DummyClassifier(strategy="most_frequent", random_state=0)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = get_full_probs(clf.predict_proba(X_test), clf.classes_, num_classes)

    acc = (preds == y_test).mean()
    auc = compute_roc_auc(y_test, probs, num_classes)
    return acc, auc

def run_lightgbm(X_train, y_train, X_test, y_test, num_classes):
    clf = LGBMClassifier(n_estimators=100, random_state=0, verbose=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = get_full_probs(clf.predict_proba(X_test), clf.classes_, num_classes)

    acc = (preds == y_test).mean()
    auc = compute_roc_auc(y_test, probs, num_classes)
    return acc, auc

def run_xgboost(X_train, y_train, X_test, y_test, num_classes):
    clf = XGBClassifier(
        n_estimators=100,
        random_state=0,
        verbosity=0,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    if probs.shape[1] < num_classes:
        probs = get_full_probs(probs, clf.classes_, num_classes)

    acc = (preds == y_test).mean()
    auc = compute_roc_auc(y_test, probs, num_classes)
    return acc, auc

def main():
    methods = ["majority", "lightgbm", "xgboost"]
    all_results = {m: {} for m in methods}

    for name, dataset_id in DATASETS.items():
        print(f"\n========== {name.upper()} ==========")

        X, y = load_dataset(dataset_id)
        num_classes = len(np.unique(y))

        if num_classes > MAX_CLASSES:
            print("Skipping (too many classes)")
            continue

        split_results = {m: {"acc": [], "auc": []} for m in methods}

        for split in range(N_SPLITS):
            print(f"  Split {split+1}/{N_SPLITS}", end=" | ")

            X_train, X_test, y_train, y_test = preprocess(X, y, seed=split)

            acc, auc = run_majority_class(X_train, y_train, X_test, y_test, num_classes)
            split_results["majority"]["acc"].append(acc)
            split_results["majority"]["auc"].append(auc)
            print(f"Majority: {acc:.4f}/{auc:.4f}", end=" | ")

            acc, auc = run_lightgbm(X_train, y_train, X_test, y_test, num_classes)
            split_results["lightgbm"]["acc"].append(acc)
            split_results["lightgbm"]["auc"].append(auc)
            print(f"LGBM: {acc:.4f}/{auc:.4f}", end=" | ")

            acc, auc = run_xgboost(X_train, y_train, X_test, y_test, num_classes)
            split_results["xgboost"]["acc"].append(acc)
            split_results["xgboost"]["auc"].append(auc)
            print(f"XGB: {acc:.4f}/{auc:.4f}")

        for m in methods:
            all_results[m][name] = {
                "acc": split_results[m]["acc"],
                "auc": split_results[m]["auc"],
                "num_classes": num_classes,
            }

        for m in methods:
            mean_acc = np.nanmean(split_results[m]["acc"])
            mean_auc = np.nanmean(split_results[m]["auc"])
            print(f"  {m:<12} -> Acc: {mean_acc:.4f}  AUC: {mean_auc:.4f}")


    col_w = 10
    sep = "=" * (33 + len(methods) * (col_w + 2))

    print(f"\n\n{sep}")
    print(f"{'DATASET':<30} {'C':>3}", end="")
    for m in methods:
        print(f"  {m.upper():>{col_w}}", end="")
    print()
    print(sep)

    method_means = {m: [] for m in methods}

    for name in DATASETS:
        if name not in all_results["majority"]:
            continue

        nc = all_results["majority"][name]["num_classes"]
        print(f"{name:<30} {nc:>3}", end="")

        for m in methods:
            mean_auc = np.nanmean(all_results[m][name]["auc"])
            print(f"  {mean_auc:>{col_w}.4f}", end="")
            method_means[m].append(mean_auc)

        print()

    print(sep)
    print(f"{'MEAN':<30} {'':>3}", end="")
    for m in methods:
        print(f"  {np.nanmean(method_means[m]):>{col_w}.4f}", end="")
    print()
    print(sep)


if __name__ == "__main__":
    main()