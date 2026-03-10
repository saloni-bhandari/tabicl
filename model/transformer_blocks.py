import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .attention import multi_head_attention_forward
from .ssmax import create_ssmax_layer
from .rope import RotaryEmbedding

from typing import Union, Optional

class MAB(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        ssmax: Union[bool, str] = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        dim_feedforward = dim_feedforward or embed_dim * 4

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(ssmax, bool):
            ssmax = "qassmax-mlp-elementwise" if ssmax else "none"
        self.ssmax_layer = create_ssmax_layer(
            ssmax_type=ssmax,
            num_heads=num_heads,
            embed_dim=embed_dim,
        )
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.ff[3].weight)
        nn.init.zeros_(self.ff[3].bias)

    def forward(
        self,
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        rope: Optional[RotaryEmbedding] = None,
    ) -> Tensor:
        k = q if k is None else k
        v = q if v is None else v

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=q.dtype,
        )
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=q.dtype,
            check_other=False,
        )

        q_norm = self.norm1(q)
        k_norm = self.norm1(k) if k is not q else q_norm
        v_norm = k_norm if v is k else self.norm1(v)

        attn_out = multi_head_attention_forward(
            query=q_norm,
            num_heads=self.num_heads,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            dropout_p=self.dropout,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            key=k_norm,
            value=v_norm,
            training=self.training,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            rope=rope,
            ssmax_layer=self.ssmax_layer,
        )

        x = q + self.dropout1(attn_out)
        x = x + self.dropout2(self.ff(self.norm2(x)))

        return x

class ISAB(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int,
        num_inducing: int = 128,
        dropout: float = 0.0,
        activation: str = "gelu",
        ssmax: Union[bool, str] = False,
        skip_value: float = -100.0,
    ):
        super().__init__()
        self.skip_value = skip_value

        if isinstance(ssmax, bool):
            ssmax = "qassmax-mlp-elementwise" if ssmax else "none"

        # only first attention uses ssmax
        self.mab1 = MAB(embed_dim, num_heads, dim_feedforward, dropout, activation, ssmax=ssmax)
        self.mab2 = MAB(embed_dim, num_heads, dim_feedforward, dropout, activation, ssmax=False)

        self.num_inducing = num_inducing
        self.inducing_points = nn.Parameter(torch.empty(num_inducing, embed_dim))
        nn.init.trunc_normal_(self.inducing_points, std=0.02)

    def _induced_attention(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
        *batch, _, model_dims = src.shape
        VI = self.inducing_points.expand(*batch, self.num_inducing, model_dims)

        src_train = src if train_size is None else src[..., :train_size, :]
        attn = self.mab1(q=VI, k=src_train, v=src_train)  # (*batch, num_inducing, d)

        out = self.mab2(q=src, k=attn, v=attn)               # (*batch, n_total, d)

        return out

    def forward(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
        # handle skip values 
        skip_mask = (src == self.skip_value).all(dim=(-2, -1))
        if skip_mask.any():
            if skip_mask.all():
                return torch.full_like(src, self.skip_value)
            out = torch.empty_like(src)
            out[~skip_mask] = self._induced_attention(src[~skip_mask], train_size)
            out[skip_mask] = self.skip_value
            return out

        return self._induced_attention(src, train_size) 