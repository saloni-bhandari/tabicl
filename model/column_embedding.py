from model.encoders import SetTransformer
import torch
import torch.nn as nn
from typing import Optional, List

from .layers import OneHotAndLinear


class ColEmbedding(nn.Module):

    def __init__(
        self,
        embedding_dim,
        nhead,
        num_classes,
        num_blocks=3,
        num_inds=128,
        dim_feedforward=64,
        dropout=0.0,
        activation="gelu",
        debug=False,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.nhead = nhead
        self.num_classes = num_classes
        self.debug = debug

        self.label_embedding = OneHotAndLinear(
            num_classes=self.num_classes,
            embed_dim=self.embedding_dim
        )

        self.embed_cells_to_dim = nn.Linear(1, embedding_dim)

        self.encode_column = SetTransformer(
            num_blocks=num_blocks,
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_inds=num_inds,
            dropout=dropout,
            activation=activation,
        )

        self.generate_W = nn.Linear(embedding_dim, embedding_dim)
        self.generate_B = nn.Linear(embedding_dim, embedding_dim)

    def _debug_print(self, *args):
        if self.debug:
            print(*args)

    def forward(
        self,
        X: torch.Tensor,
        y_train: torch.Tensor,
        d: Optional[torch.Tensor] = None,
        embed_with_test: bool = False,
        feature_shuffles: Optional[List[List[int]]] = None,
        mgr_config=None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        X : (B, T, Hmax)
            Possibly padded input table batch.
        y_train : (B, train_size)
        d : (B,)
            Number of valid features in each table.
        embed_with_test : bool
            If False, the column encoder should only condition on train rows.

        Returns
        -------
        X_out : (B, T, Hmax, E)
        """
        B, T, H = X.shape
        train_size = y_train.shape[1]
        device = X.device

        self._debug_print("Input X:", X.shape)
        self._debug_print("Input y_train:", y_train.shape)
        self._debug_print("Input d:", None if d is None else d.shape)

        y_full = torch.zeros(B, T, dtype=y_train.dtype, device=device)
        y_full[:, :train_size] = y_train

        y_emb = self.label_embedding(y_full.long().reshape(-1, 1))
        y_emb = y_emb.reshape(B, T, self.embedding_dim)

        # zero out test rows explicitly
        y_emb[:, train_size:, :] = 0.0

        X_t = X.transpose(1, 2)  # (B, H, T)

        if d is None:
            # all columns valid
            valid_mask = torch.ones(B, H, dtype=torch.bool, device=device)
        else:
            col_idx = torch.arange(H, device=device).unsqueeze(0).expand(B, H)
            valid_mask = col_idx < d.unsqueeze(1)  # (B, H)

        # Extract valid columns only
        valid_cols = X_t[valid_mask]  # (N_valid, T)
        valid_cols_scalar = valid_cols.unsqueeze(-1)  # (N_valid, T, 1)

        self._debug_print("Valid columns:", valid_cols_scalar.shape)

        valid_cols_emb = self.embed_cells_to_dim(valid_cols_scalar)  # (N_valid, T, E)

        # We need matching y embeddings for each valid column
        # Expand y_emb from (B, T, E) -> (B, H, T, E), then index by valid_mask
        y_expand = y_emb.unsqueeze(1).expand(B, H, T, self.embedding_dim)
        valid_y_emb = y_expand[valid_mask]  # (N_valid, T, E)

        valid_cols_emb = valid_cols_emb + valid_y_emb

        valid_cols_encoded = self.encode_column(
            valid_cols_emb,
            train_size=None if embed_with_test else train_size,
        )  # (N_valid, T, E)

        self._debug_print("Encoded valid columns:", valid_cols_encoded.shape)

        W = self.generate_W(valid_cols_encoded)      # (N_valid, T, E)
        B_bias = self.generate_B(valid_cols_encoded) # (N_valid, T, E)

        # Apply affine transform to ORIGINAL scalar values
        valid_cols_out = valid_cols_scalar * W + B_bias  # (N_valid, T, E)

        X_out_cols = torch.zeros(B, H, T, self.embedding_dim, device=device, dtype=valid_cols_out.dtype)
        X_out_cols[valid_mask] = valid_cols_out

        X_out = X_out_cols.permute(0, 2, 1, 3)  # (B, T, H, E)

        self._debug_print("Output embeddings:", X_out.shape)

        return X_out
    
if __name__ == "__main__":
    batch_size = 2
    num_cols = 4
    num_rows = 8
    embedding_dim = 32
    nhead = 2
    train_size = 4

    model = ColEmbedding(embedding_dim, nhead, num_classes=10, debug=True)
    X = torch.randn(batch_size, num_rows, num_cols)
    y_train = torch.randint(0, 10, (batch_size, train_size))
    d = torch.tensor([4, 3])

    output = model(X, y_train, d=d)
    print(output.shape)
