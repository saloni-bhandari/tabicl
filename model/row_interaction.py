import torch
import torch.nn as nn
from .encoders import Encoder


class RowInteraction(nn.Module):
    def __init__(self, num_rows, num_attention_blocks, embedding_dim=8, nhead=1, dim_feedforward=2048, debug=False):
        super(RowInteraction, self).__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim)) # shape: (1, 1, embedding_dim)
        self.debug = debug
    
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

    def forward(self, X):

        batch_size, num_cols, num_rows, embedding_dim = X.size()

        self._debug_print(f"Input X shape: {X.shape}")

        X = X.view(batch_size * num_rows, num_cols, embedding_dim)

        self._debug_print(f"Collapsed X shape: {X.shape}")

        cls_tokens = self.cls_token.expand(batch_size * num_rows, 4, embedding_dim)
        self._debug_print(f"CLS tokens shape: {cls_tokens.shape}")

        X = torch.cat((cls_tokens, X), dim=1)

        self._debug_print(f"X shape after adding CLS tokens: {X.shape}")

        X = X.view(batch_size, num_rows, -1, embedding_dim)
        self._debug_print(f"X shape after reshaping back: {X.shape}")

        X = X.view(batch_size * num_rows, -1, embedding_dim)
        self._debug_print(f"X shape after reshaping for attention layer: {X.shape}")
        
        for block in self.tfrow.blocks:
            X = block(X)

        self._debug_print(f"After doing multi_head_attention with ROPE: {X.shape}")

        cls_outputs = X[:, :4, :]
        self._debug_print(f"Class output tokens: {cls_outputs.shape}")

        cls_outputs = cls_outputs.view(batch_size, num_rows, -1)
        self._debug_print(
            f"CLS outputs after concatenating the 4 CLS tokens together: {cls_outputs.shape}"
        )

        return cls_outputs

if __name__ == "__main__":
    # testing
    batch_size = 2
    num_cols = 8
    num_rows = 5
    embedding_dim = 32
    num_attention_blocks = 4
    nhead=2

    model = RowInteraction(num_rows, num_attention_blocks, embedding_dim, nhead)
    X = torch.randn(batch_size, num_cols, num_rows, embedding_dim) # this is the output after column embedding
    model(X)
    