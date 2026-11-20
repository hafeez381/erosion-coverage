"""Directional ablation, guide Eq. (8) left: h <- h - v v^T h at every layer (Arditi et al. 2024).

Applied to each decoder block's input and to every attention and MLP output, so no layer can write
the direction back into the residual stream.
"""
from contextlib import contextmanager

import torch


def _proj(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    v = v.to(x.dtype)
    return x - (x @ v).unsqueeze(-1) * v


@contextmanager
def ablated(model, v: torch.Tensor):
    """Context manager: every forward inside removes unit direction v (d,) everywhere."""
    v = (v / v.norm()).to(next(model.parameters()).device)
    handles = []

    def pre(mod, args, kwargs):
        if args:
            return (_proj(args[0], v), *args[1:]), kwargs
        kwargs["hidden_states"] = _proj(kwargs["hidden_states"], v)
        return args, kwargs

    def post(mod, inp, out):
        if isinstance(out, tuple):
            return (_proj(out[0], v), *out[1:])
        return _proj(out, v)

    for layer in model.model.layers:
        handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(post))
        handles.append(layer.mlp.register_forward_hook(post))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
