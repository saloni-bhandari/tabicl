from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Optional, Literal, List, Dict
from collections import OrderedDict

import numpy as np
import torch
from torch import Tensor
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from tabicl import TabICLClassifier, TabICL, TabICLCache, InferenceConfig
from row_selection import select_rows


class LocalizedTabICLClassifier:
    """Localized TabICL that selects informative training rows per test sample.

    Parameters
    ----------
    k : int
        Number of training rows to select per test sample.
    selection_method : str
        Row selection strategy: "random", "cosine", "euclidean".
    selection_space : str
        Feature space for retrieval: "embedding" (use row representations)
        or "raw" (use original input features).
    class_balanced : bool
        Whether to enforce class-balanced selection.
    base_kwargs : dict
        Keyword arguments passed to the underlying TabICLClassifier.
    """

    def __init__(
        self,
        k: int = 128,
        selection_method: Literal["random", "cosine", "euclidean"] = "cosine",
        selection_space: Literal["embedding", "raw"] = "embedding",
        class_balanced: bool = False,
        n_estimators: int = 8,
        softmax_temperature: float = 0.9,
        icl_chunk_size: int = 8192,
        device: Optional[str] = None,
        verbose: bool = False,
        **base_kwargs,
    ):
        self.k = k
        self.selection_method = selection_method
        self.selection_space = selection_space
        self.class_balanced = class_balanced
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self._icl_chunk_size = icl_chunk_size
        self.device = device
        self.verbose = verbose
        self.base_kwargs = base_kwargs

        # Will be populated during fit
        self.base_clf_ = None
        self.train_repr_cache_ = None   # {norm_method: Tensor (n_est, n_train, D)}
        self.train_labels_cache_ = None  # {norm_method: Tensor (n_est, n_train)}
        self.train_raw_cache_ = None    # {norm_method: Tensor (n_est, n_train, H)}
        self.timing_ = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LocalizedTabICLClassifier":
        """Fit the localized classifier.

        This builds the base TabICL model with "repr" cache mode to extract
        and cache the training row representations (Stage 2 output with
        labels baked in).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Training labels.

        Returns
        -------
        self
        """
        t0 = time.time()

        # Create base classifier with repr cache mode
        self.base_clf_ = TabICLClassifier(
            n_estimators=self.n_estimators,
            kv_cache="repr",
            softmax_temperature=self.softmax_temperature,
            device=self.device,
            verbose=self.verbose,
            **self.base_kwargs,
        )
        self.base_clf_.fit(X, y)

        # Store references
        self.classes_ = self.base_clf_.classes_
        self.n_classes_ = self.base_clf_.n_classes_
        self.y_encoder_ = self.base_clf_.y_encoder_

        # Extract cached representations per norm method
        # model_kv_cache_ is OrderedDict[str, TabICLCache]
        # Each TabICLCache has row_repr of shape (n_estimators_for_method, n_train, D)
        self.train_repr_cache_ = OrderedDict()
        self.train_labels_cache_ = OrderedDict()

        for norm_method, cache in self.base_clf_.model_kv_cache_.items():
            # row_repr has labels baked in (from prepare_repr_cache)
            self.train_repr_cache_[norm_method] = cache.row_repr.clone()

        # Also store the raw training data for raw-feature retrieval
        if self.selection_space == "raw":
            train_data = self.base_clf_.ensemble_generator_.transform(X=None, mode="train")
            self.train_raw_cache_ = OrderedDict()
            for norm_method, (Xs, ys) in train_data.items():
                self.train_raw_cache_[norm_method] = torch.from_numpy(Xs).float()

        # Store training labels for class-balanced selection
        train_data = self.base_clf_.ensemble_generator_.transform(X=None, mode="train")
        self.train_labels_cache_ = OrderedDict()
        for norm_method, (Xs, ys) in train_data.items():
            self.train_labels_cache_[norm_method] = torch.from_numpy(ys).float()

        self.timing_["fit"] = time.time() - t0
        return self

    def _get_test_representations(self, X: np.ndarray) -> OrderedDict:
        """Run Stages 1-2 on test data using cached col_cache from training.

        Uses col_embedder.forward_with_cache with the stored col_cache to
        properly embed test data in the context of training data, then runs
        row_interactor to get row representations.

        Returns
        -------
        OrderedDict mapping norm_method to Tensor of shape
        (n_estimators_for_method, n_test, D).
        """
        model = self.base_clf_.model_
        device = self.base_clf_.device_
        inference_config = self.base_clf_.inference_config_

        X_enc = self.base_clf_.X_encoder_.transform(X)
        test_data = self.base_clf_.ensemble_generator_.transform(X_enc, mode="test")

        test_reprs = OrderedDict()
        for norm_method, (Xs_test,) in test_data.items():
            kv_cache = self.base_clf_.model_kv_cache_[norm_method]
            batch_size = self.base_clf_.batch_size or Xs_test.shape[0]
            n_batches = int(np.ceil(Xs_test.shape[0] / batch_size))
            Xs_split = np.array_split(Xs_test, n_batches)

            batch_reprs = []
            offset = 0
            for X_batch in Xs_split:
                bs = X_batch.shape[0]
                cache_subset = kv_cache.slice_batch(offset, offset + bs)
                offset += bs

                X_t = torch.from_numpy(X_batch).float().to(device)
                with torch.no_grad():
                    embeddings = model.col_embedder.forward_with_cache(
                        X_t,
                        col_cache=cache_subset.col_cache,
                        y_train=None,
                        use_cache=True,
                        store_cache=False,
                        mgr_config=inference_config.COL_CONFIG,
                    )
                    representations = model.row_interactor(
                        embeddings,
                        mgr_config=inference_config.ROW_CONFIG,
                    )
                batch_reprs.append(representations.cpu())

            test_reprs[norm_method] = torch.cat(batch_reprs, dim=0)

        return test_reprs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using localized ICL.

        For each test sample, selects top-k training rows and runs the
        ICL transformer (Stage 3) on only those rows.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self.base_clf_)

        model = self.base_clf_.model_
        device = self.base_clf_.device_
        inference_config = self.base_clf_.inference_config_

        t_retrieval = 0.0
        t_stage12 = 0.0
        t_stage3 = 0.0

        # Step 1: Get test row representations (Stages 1-2)
        t0 = time.time()
        test_reprs = self._get_test_representations(X)
        t_stage12 = time.time() - t0

        # Step 2: For each norm method, select rows and run Stage 3
        # Get preprocessed test data for raw-feature retrieval
        if self.selection_space == "raw":
            X_enc = self.base_clf_.X_encoder_.transform(X)
            test_raw_data = self.base_clf_.ensemble_generator_.transform(X_enc, mode="test")

        all_outputs = []

        for norm_method in test_reprs:
            test_repr = test_reprs[norm_method].to(device=device, dtype=torch.float32)
            train_repr = self.train_repr_cache_[norm_method].to(device=device, dtype=torch.float32)
            train_labels = self.train_labels_cache_[norm_method].to(device)  # (n_est, n_train)
            n_est, n_test, D = test_repr.shape
            n_train = train_repr.shape[1]

            if self.selection_space == "raw":
                train_raw = self.train_raw_cache_[norm_method].to(device)
                test_raw_tuple = test_raw_data[norm_method]
                test_raw = torch.from_numpy(test_raw_tuple[0]).float().to(device)

            # Retrieval for all estimators
            all_indices = []
            for e in range(n_est):
                t0 = time.time()
                if self.selection_method == "random":
                    indices = select_rows(
                        train_repr[e], test_repr[e], self.k,
                        method="random",
                    )
                elif self.selection_space == "raw":
                    indices = select_rows(
                        train_raw[e], test_raw[e], self.k,
                        method=self.selection_method,
                        train_labels=train_labels[e] if self.class_balanced else None,
                        class_balanced=self.class_balanced,
                    )
                else:
                    indices = select_rows(
                        train_repr[e], test_repr[e], self.k,
                        method=self.selection_method,
                        train_labels=train_labels[e] if self.class_balanced else None,
                        class_balanced=self.class_balanced,
                    )
                t_retrieval += time.time() - t0
                all_indices.append(indices)

            # Stage 3: Batched ICL for ALL estimators at once
            t0 = time.time()
            est_preds = self._run_localized_icl(
                model, train_repr, test_repr, all_indices, device,
            )
            t_stage3 += time.time() - t0

            all_outputs.append(est_preds)

        outputs = np.concatenate(all_outputs, axis=0)  # (total_estimators, n_test, n_classes)

        # Aggregate predictions with class shuffle correction
        class_shuffles = []
        for shuffles in self.base_clf_.ensemble_generator_.class_shuffles_.values():
            class_shuffles.extend(shuffles)
        n_estimators = len(class_shuffles)

        avg = np.zeros_like(outputs[0])
        for i, shuffle in enumerate(class_shuffles):
            out = outputs[i]
            avg += out[..., shuffle]
        avg /= n_estimators

        if self.base_clf_.average_logits:
            avg = self.base_clf_.softmax(avg, axis=-1, temperature=self.softmax_temperature)

        self.timing_["stage12"] = t_stage12
        self.timing_["retrieval"] = t_retrieval
        self.timing_["stage3"] = t_stage3
        self.timing_["total_predict"] = t_stage12 + t_retrieval + t_stage3

        # Normalize
        proba = avg / avg.sum(axis=1, keepdims=True)
        return proba

    def _run_localized_icl(
        self,
        model: TabICL,
        train_repr: Tensor,
        test_repr: Tensor,
        all_indices: List[Tensor],
        device: torch.device,
    ) -> np.ndarray:
        """Run Stage 3 (ICL) for ALL estimators in one batched forward pass.

        Bypasses InferenceManager and calls the ICL transformer directly,
        batching all estimators together to minimize GPU kernel launches
        and per-call overhead.

        Parameters
        ----------
        model : TabICL
            The TabICL model.
        train_repr : Tensor
            Training representations of shape (n_est, N_train, D), labels baked in.
        test_repr : Tensor
            Test representations of shape (n_est, N_test, D).
        all_indices : list of Tensor
            List of n_est tensors, each of shape (N_test, k).
        device : torch.device

        Returns
        -------
        np.ndarray of shape (n_est, N_test, n_classes)
        """
        n_est = len(all_indices)
        N_test = test_repr.shape[1]
        k = all_indices[0].shape[1]
        num_classes = self.n_classes_
        return_logits = self.base_clf_.average_logits
        icl = model.icl_predictor

        # Build R tensors for all estimators and stack into one batch
        R_parts = []
        for e in range(n_est):
            sel_train = train_repr[e][all_indices[e]]  # (N_test, k, D)
            test_col = test_repr[e].unsqueeze(1).to(dtype=sel_train.dtype)
            R = torch.cat([sel_train, test_col], dim=1)  # (N_test, k+1, D)
            R_parts.append(R)

        R_all = torch.cat(R_parts, dim=0)  # (n_est * N_test, k+1, D)
        total = R_all.shape[0]
        chunk_size = max(1, self._icl_chunk_size)
        all_preds = []

        # Direct transformer call — bypass InferenceManager to avoid
        # per-call overhead (memory estimation, offload resolution, etc.)
        amp_ctx = torch.autocast(device_type="cuda") if device.type == "cuda" else nullcontext()
        with torch.no_grad(), amp_ctx:
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                R_chunk = R_all[start:end].to(device)

                src = icl.tf_icl(R_chunk, train_size=k)
                if icl.norm_first:
                    src = icl.ln(src)
                out = icl.decoder(src)

                # Extract test predictions: (chunk, k+1, out_dim) → (chunk, num_classes)
                out = out[:, k:, :num_classes]
                if not return_logits:
                    out = torch.softmax(out / self.softmax_temperature, dim=-1)
                all_preds.append(out.squeeze(1).cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)  # (n_est * N_test, num_classes)
        return preds.reshape(n_est, N_test, -1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for test samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        array-like of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        y = np.argmax(proba, axis=1)
        return self.y_encoder_.inverse_transform(y)

    def get_timing(self) -> Dict[str, float]:
        """Return timing breakdown from the last predict call.

        Returns
        -------
        dict with keys: fit, stage12, retrieval, stage3, total_predict
        """
        return self.timing_.copy()
