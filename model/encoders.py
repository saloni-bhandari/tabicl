from __future__ import annotations

from typing import Optional, Union
from functools import partial

from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

from .rope import RotaryEmbedding
from .transformer_blocks import MAB, ISAB

class Encoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_rope: bool = False,
        rope_base: int = 100000,
        rope_interleaved: bool = True,
        ssmax: Union[bool, str] = False,
        recompute: bool = False,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.blocks = nn.ModuleList([
            MAB(
                embed_dim=d_model,
                num_heads=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                ssmax=ssmax,
            )
            for _ in range(num_blocks)
        ])

        self.rope = (
            RotaryEmbedding(dim=d_model // nhead, theta=rope_base, interleaved=rope_interleaved)
            if use_rope else None
        )
        self.recompute = recompute

    def forward(self, src: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        out = src
        for block in self.blocks:
            if self.recompute:
                out = checkpoint(partial(block, rope=self.rope, attn_mask=attn_mask), out, use_reentrant=False)
            else:
                out = block(q=out, rope=self.rope, attn_mask=attn_mask)
        return out

class SetTransformer(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int = 128,
        dropout: float = 0.0,
        activation: str = "gelu",
        ssmax: Union[bool, str] = False,
        recompute: bool = False,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.blocks = nn.ModuleList([
            ISAB(
                embed_dim=d_model,
                num_heads=nhead,
                dim_feedforward=dim_feedforward,
                num_inducing=num_inds,
                dropout=dropout,
                activation=activation,
                ssmax=ssmax,
            )
            for _ in range(num_blocks)
        ])
        self.recompute = recompute

    def forward(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
        out = src
        for block in self.blocks:
            if self.recompute:
                out = checkpoint(partial(block, train_size=train_size), out, use_reentrant=False)
            else:
                out = block(src=out, train_size=train_size)
        return out