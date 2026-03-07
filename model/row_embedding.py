import torch
import torch.nn as nn

from .rope import RotaryEmbedding
from .attention import multi_head_attention_forward


class RoPEAttnLayer(nn.Module):
    """
    A version of an Attention-Encoder Layer with Rotary Positional Encodings
    (RoPE) as described in [1].

    [1] https://arxiv.org/pdf/2104.09864.pdf - RoPE Paper
    [2] https://github.com/karpathy/nanoGPT/blob/master/model.py - Guidance for
            loop-free implementation of multi-head architecture.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_pos_enc_len: int,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.inv_sqrt_d_head = 1.0 / torch.sqrt(torch.tensor(self.d_head))

        self.multi_head_in_projection = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.multi_head_out_projection = nn.Linear(d_model, d_model, bias=bias)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.n_heads = n_heads
        self.d_model = d_model
        self._construct_rope_matrices(max_pos_enc_len)

    def _construct_rope_matrices(self, max_pos_enc_len):
        """Constructs rotary embedding matrices for additive version
        [1, p. 7, eq. (34)]. Configured for x beeing of shape
        (batch_size, seqlen, d_model).
        """
        assert self.d_head % 2 == 0
        # [t1, t1, t2, t2, t3, t3, ...]
        thetas = 1000 ** (
            -2.0 * torch.arange(1, self.d_head / 2 + 1) / self.d_head
        ).repeat_interleave(2)
        positions = torch.arange(1, max_pos_enc_len + 1).float()
        # [ [1t1, 1t1, 1t2, 1t2, ...],
        #   [2t1, 2t1, 2t2, 2t2, ...],
        #   [3t1, 3t1, 3t2, 3t2, ...],
        #   ...                       ]
        args = positions.reshape(-1, 1) @ thetas.reshape(1, -1)
        self.register_buffer("rope_sin", torch.sin(args))
        self.register_buffer("rope_cos", torch.cos(args))

    def _reorder_for_rope_sin(self, x):
        """Reorders the inputs according to [1, p. 7, eq. (34)] for the
        multiplication with the sinus-part of the RoPE. Configured for x beeing
        having d_head as last dimension, should be of shape
        (batch_size, n_heads, seqlen, d_head).
        """
        # [x1, x3, x5, ...]
        x_odd = x[..., ::2]
        # [x2, x4, x6, ...]
        x_even = x[..., 1::2]
        # [[-x2, x1], [-x4, x3], [-x6, x5], ...]
        x_stacked = torch.stack([-x_even, x_odd], dim=-1)
        # [-x2, x1, -x4, x3, ...]
        return x_stacked.flatten(start_dim=-2)

    def _apply_rope(self, x):
        """Applies RoPE the inputs according to [1, p. 7, eq. (34)].
        Configured for x being of shape (batch_size, n_heads, seqlen, d_head).
        """
        T = x.shape[2]
        x_sin = self._reorder_for_rope_sin(x)
        x_rope = x * self.rope_cos[:T, :] + x_sin * self.rope_sin[:T, :]
        return x_rope

    def forward(self, x):
        B, T, C = x.size()  # batch_size, seqlen, d_model

        # apply key, query, value projections
        q, k, v = self.multi_head_in_projection(x).split(self.d_model, dim=2)

        # separate heads (batch_size, n_heads, seqlen, d_head)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # apply RoPE transformation [1, p. 7]
        q_rope = self._apply_rope(q)
        k_rope = self._apply_rope(k)

        # RoPE self attention:
        #   (batch_size, n_heads, seqlen, d_head) x
        #   (batch_size, n_heads, d_head, seqlen)
        #       -> (batch_size, n_heads, seqlen, seqlen)
        #  This is the place, where the rotations get "inserted" into the
        #  attention mechanism as presented in [1, p. 6, eq. 19]. I stick to
        #  the basic `exp` as non-negativities.
        att_numerator = torch.exp(
            (q_rope @ k_rope.transpose(-2, -1)) * self.inv_sqrt_d_head
        )
        att_denominator = torch.exp((q @ k.transpose(-2, -1)) * self.inv_sqrt_d_head)
        att_denominator = torch.sum(att_denominator, dim=-1, keepdim=True)
        att = att_numerator / att_denominator
        # (batch_size, n_heads, seqlen, seqlen) x
        #   (batch_size, n_heads, seqlen, d_head)
        # -> (batch_size, n_heads, seqlen, d_head)
        y = att @ v
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.multi_head_out_projection(y)

        # skip-connection and regularization
        y = self.layer_norm(y + x)
        y = self.dropout(y)
        return y
    
class RowInteraction(nn.Module):
    def __init__(self, num_rows, num_attention_blocks, embedding_dim=8, nhead=1, debug=False):
        super(RowInteraction, self).__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim)) # shape: (1, 1, embedding_dim)
        self.debug = debug
        
        rope_layers = []
        for _ in range(num_attention_blocks):
            rope_layers.append(RoPEAttnLayer(d_model=embedding_dim, n_heads=nhead, max_pos_enc_len=512)) # max_pos_enc_len is num_rows + 4 CLS tokens

        self.RopeAttnLayer = nn.Sequential(
            *rope_layers
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

        X = self.RopeAttnLayer(X)

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
    