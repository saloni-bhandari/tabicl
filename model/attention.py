from .rope import RotaryEmbedding
from .kv_cache import KVCacheEntry

import torch
from torch import Tensor, nn
from torch.nn import functional as F

import contextlib
from typing import Optional, Union, Tuple

HAS_FLASH_ATT_3 = False
try:
    from flash_attn.flash_attn_interface import flash_attn_func
    HAS_FLASH_ATT_3 = True
except ImportError:
    pass

#  TOGGLE IF NEEDED
_use_flash_attn3 = True

@contextlib.contextmanager
def flash_attn3_toggle(state: bool):
    global _use_flash_attn3
    old_state = _use_flash_attn3
    _use_flash_attn3 = state
    try:
        yield
    finally:
        _use_flash_attn3 = old_state

def split_heads(x: Tensor, num_heads: int) -> Tensor:
    # [.., seq_len, embed_dim] -> [..., num_heads, seq_len, head_dim]
    *batch, seq_len, embed_dim = x.shape
    head_dim = embed_dim // num_heads
    x = x.reshape(*batch, seq_len, num_heads, head_dim)
    return x.transpose(-3, -2)

def merge_heads(x: Tensor) -> Tensor:
    # [..., num_heads, seq_len, head_dim] -> [.., seq_len, embed_dim]
    *batch, num_heads, seq_len, head_dim = x.shape
    x = x.transpose(-3, -2).contiguous()
    return x.reshape(*batch, seq_len, num_heads * head_dim) 

def sdpa_with_flattened_batch(q: Tensor,k:Tensor,v:Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    ssmax_layer: Optional[nn.Module] = None):
    original_shape = q.shape

    # reshape: [batches, num_heads, seq_len, head_dims]
    q = q.view(-1, *q.shape[-3:])
    k = k.view(-1, *k.shape[-3:])
    v = v.view(-1, *v.shape[-3:])

    # reshape attn mask if exists
    if attn_mask is not None:
        attn_mask = attn_mask.view(-1, *attn_mask.shape[-3:])

    # apply scalable softmax if needed
    if ssmax_layer is not None:
        q = ssmax_layer(q, k.size(-2))

    head_dims = q.shape[-1]
    # use flash attention 3 if possible
    use_fa3 = (HAS_FLASH_ATT_3 and _use_flash_attn3 and q.is_cuda 
        and head_dims % 8 == 0 
        and attn_mask is None and dropout_p == 0.0)
    
    if use_fa3:
        orig_dtype = q.dtype

        # flash attention only works for (b)float 16
        if orig_dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)

        # reshape to [batches, seq_len, num_heads, head_dims]
        # .contiguous helps with memory issues
        q_flash_att = q.transpose(-3,-2).contiguous()
        k_flash_att = k.transpose(-3,-2).contiguous()
        v_flash_att = v.transpose(-3,-2).contiguous()

        out = flash_attn_func(q_flash_att, k_flash_att, v_flash_att, causal=False)
        # reshape back to [batches, num_heads, seq_len, head_dims]
        out = out.transpose(-3,-2)

        if orig_dtype not in (torch.float16, torch.bfloat16):
            out = out.to(orig_dtype)
    
    # if not use flash attention, use pytorch
    else:
        out = F.scaled_dot_product_attention(q,k,v,attn_mask=attn_mask,dropout_p=dropout_p)
    
    return out.view(original_shape)

def multi_head_attention_forward(
    query: Tensor,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    key: Optional[Tensor] = None,
    value: Optional[Tensor] = None,
    cached_kv: Optional[KVCacheEntry] = None,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    attn_mask: Optional[Tensor] = None,
    rope: Optional[RotaryEmbedding] = None,
    ssmax_layer: Optional[nn.Module] = None,
    need_kv: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    
    # query: [batch, seq_len, embed_dim]
    *batch, seq_len, embed_dim = query.shape
    head_dim = embed_dim//num_heads

    if cached_kv is None:
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)
    else:
        q_proj_bias = in_proj_bias[:embed_dim] if in_proj_bias is not None else None
        q = F.linear(query, in_proj_weight[:embed_dim], q_proj_bias)
        k, v = cached_kv.key, cached_kv.value

    q = split_heads(q, num_heads)

    if rope is not None:
        q = rope.rotate_queries_or_keys(q)
        # Only apply RoPE to k if it was freshly generated.
        # Tensors pulled from the KV cache were already rotated prior to storage
        if cached_kv is None:
            k = rope.rotate_queries_or_keys(k)

    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.view(*batch, 1, 1, k.shape[-2])
        if attn_mask is not None:
            attn_mask += key_padding_mask
        else:
            attn_mask = key_padding_mask

    attn_output = sdpa_with_flattened_batch(q, k, v, attn_mask, 
                                            dropout_p if training else 0.0, 
                                            ssmax_layer)

    attn_output = merge_heads(attn_output)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    
    if need_kv and cached_kv is None:
        return attn_output, k, v
    
    return attn_output
