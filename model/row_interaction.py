from typing import Optional
import torch
import torch.nn as nn
from .encoders import Encoder


class RowInteraction(nn.Module):

    def __init__(
        self,
        num_attention_blocks,
        embedding_dim=8,
        nhead=1,
        dim_feedforward=2048,
        num_cls=4,
        debug=False,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_cls = num_cls
        self.debug = debug

        # learnable CLS tokens
        self.cls_token = nn.Parameter(torch.randn(1, num_cls, embedding_dim))

        # transformer encoder
        self.tfrow = Encoder(
            num_blocks=num_attention_blocks,
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="gelu",
            use_rope=True,
            rope_base=100000,
            rope_interleaved=True,
            recompute=False,
        )

    def _debug_print(self, *args):
        if self.debug:
            print(*args)

    def forward(
        self,
        embeddings: torch.Tensor,
        d: Optional[torch.Tensor] = None,
        mgr_config=None,
    ):
        """
        Parameters
        ----------
        embeddings : (B, T, H, E)
        d : unused (kept for compatibility)
        mgr_config : unused (kept for compatibility)

        Returns
        -------
        row_representations : (B, T, C*E)
        """

        B, T, H, E = embeddings.shape

        self._debug_print("Input embeddings:", embeddings.shape)

        # collapse rows so each row becomes its own sequence
        X = embeddings.reshape(B * T, H, E)

        self._debug_print("Row sequences:", X.shape)

        # add CLS tokens
        cls_tokens = self.cls_token.expand(B * T, self.num_cls, E)

        X = torch.cat([cls_tokens, X], dim=1)

        self._debug_print("After adding CLS:", X.shape)

        # transformer blocks
        for block in self.tfrow.blocks:
            X = block(X)

        self._debug_print("After transformer:", X.shape)

        # extract CLS outputs
        cls_outputs = X[:, : self.num_cls, :]

        self._debug_print("CLS outputs:", cls_outputs.shape)

        # flatten CLS tokens
        cls_outputs = cls_outputs.reshape(B, T, self.num_cls * E)

        self._debug_print("Final row representation:", cls_outputs.shape)

        return cls_outputs
    
if __name__ == "__main__":
    # testing
    batch_size = 2
    num_cols = 8
    num_rows = 5
    embedding_dim = 32
    num_attention_blocks = 4
    nhead=2

    model = RowInteraction(num_rows, num_attention_blocks, embedding_dim, nhead, debug=True)
    X = torch.randn(batch_size, num_rows, num_cols, embedding_dim) # this is the output after column embedding
    model(X)
    