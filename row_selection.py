from typing import Literal, Optional
import torch
from torch import Tensor
import numpy as np


def select_rows(
    train_embeddings: Tensor,
    test_embedding: Tensor,
    k: int,
    method: Literal["random", "cosine", "euclidean"] = "cosine",
    train_labels: Optional[Tensor] = None,
    class_balanced: bool = False,
) -> Tensor:
    """Select top-k training row indices for a batch of test samples.

    Parameters
    ----------
    train_embeddings : Tensor
        Training row embeddings of shape (N_train, D).
    test_embedding : Tensor
        Test row embeddings of shape (N_test, D).
    k : int
        Number of training rows to select per test sample.
    method : str
        Selection strategy: "random", "cosine", or "euclidean".
    train_labels : Optional[Tensor]
        Training labels of shape (N_train,). Required if class_balanced=True.
    class_balanced : bool
        If True, select top-k/C nearest per class (C = num_classes).

    Returns
    -------
    Tensor
        Indices of shape (N_test, k) into the training set.
    """
    N_train = train_embeddings.shape[0]
    k = min(k, N_train)

    if method == "random":
        return _select_random(N_train, test_embedding.shape[0], k, test_embedding.device)
    elif method == "cosine":
        scores = _cosine_similarity(train_embeddings, test_embedding)
    elif method == "euclidean":
        scores = _neg_euclidean_distance(train_embeddings, test_embedding)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'random', 'cosine', or 'euclidean'.")

    if class_balanced and train_labels is not None:
        return _select_class_balanced(scores, train_labels, k)
    else:
        _, indices = torch.topk(scores, k, dim=1)
        return indices


def _select_random(n_train: int, n_test: int, k: int, device: torch.device) -> Tensor:
    """Random selection: pick k random training rows per test sample."""
    indices = torch.stack([torch.randperm(n_train, device=device)[:k] for _ in range(n_test)])
    return indices


def _cosine_similarity(train: Tensor, test: Tensor) -> Tensor:
    """Compute cosine similarity between test and train embeddings.

    Returns
    -------
    Tensor of shape (N_test, N_train) with similarity scores.
    """
    train_norm = torch.nn.functional.normalize(train, dim=1)
    test_norm = torch.nn.functional.normalize(test, dim=1)
    return test_norm @ train_norm.T


def _neg_euclidean_distance(train: Tensor, test: Tensor) -> Tensor:
    """Compute negative Euclidean distance (so higher = closer).

    Returns
    -------
    Tensor of shape (N_test, N_train) with negative distances.
    """
    # Efficient batch pairwise distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    train_sq = (train ** 2).sum(dim=1, keepdim=True)  # (N_train, 1)
    test_sq = (test ** 2).sum(dim=1, keepdim=True)    # (N_test, 1)
    cross = test @ train.T                             # (N_test, N_train)
    dist_sq = test_sq + train_sq.T - 2 * cross        # (N_test, N_train)
    return -dist_sq


def _select_class_balanced(
    scores: Tensor, train_labels: Tensor, k: int
) -> Tensor:
    """Select top-k training rows with class balance.

    For each class, select k // num_classes nearest rows, filling remainder
    from the globally nearest remaining rows.
    """
    device = scores.device
    n_test = scores.shape[0]
    classes = torch.unique(train_labels)
    n_classes = len(classes)
    per_class_k = k // n_classes
    remainder = k - per_class_k * n_classes

    all_indices = []
    for i in range(n_test):
        row_scores = scores[i]  # (N_train,)
        selected = []
        used = set()

        # Select per_class_k from each class
        for c in classes:
            class_mask = (train_labels == c).nonzero(as_tuple=True)[0]
            class_scores = row_scores[class_mask]
            n_select = min(per_class_k, len(class_mask))
            if n_select > 0:
                _, top_idx = torch.topk(class_scores, n_select)
                global_idx = class_mask[top_idx]
                selected.append(global_idx)
                used.update(global_idx.tolist())

        # Fill remainder from globally nearest unused rows
        if remainder > 0:
            mask = torch.ones(scores.shape[1], dtype=torch.bool, device=device)
            for idx in used:
                mask[idx] = False
            remaining_scores = row_scores.clone()
            remaining_scores[~mask] = float('-inf')
            _, top_idx = torch.topk(remaining_scores, remainder)
            selected.append(top_idx)

        selected = torch.cat(selected)[:k]
        all_indices.append(selected)

    return torch.stack(all_indices)
