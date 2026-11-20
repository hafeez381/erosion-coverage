"""Dual-Null calibration covariances (guide C7, Decision 5).

For each adapted projection: C_in = sum z z^T over the inputs it reads, C_out = sum g g^T over the loss
gradients at its output, pooled over every token of the 256 demonstrations (prompt and response), loss =
summed cross-entropy times `scale`. q/k/v share one C_in, gate/up share one (same input tensor).
Accumulated in fp32 on the GPU, a few layers per pass.
"""
import torch
import torch.nn.functional as F

MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
IN_GROUP = {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv", "o_proj": "o",
            "gate_proj": "gate_up", "up_proj": "gate_up", "down_proj": "down"}
GROUP_OWNER = {"q_proj", "o_proj", "gate_proj", "down_proj"}   # the one module per group that records C_in


def module(model, layer: int, name: str):
    blk = model.model.layers[layer]
    return getattr(blk.self_attn if name in ("q_proj", "k_proj", "v_proj", "o_proj") else blk.mlp, name)


def demo_ids(tok, prompt: str, response: str, max_len: int = 512) -> torch.Tensor:
    """Chat with Qwen's default system prompt, the prompt as user turn and the response as assistant turn."""
    text = tok.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
                                   tokenize=False)
    return tok(text, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=max_len).input_ids


def _loss(model, ids, scale):
    logits = model(input_ids=ids).logits[0, :-1].float()
    return F.cross_entropy(logits, ids[0, 1:], reduction="sum") * scale


def grad_max(model, demos) -> float:
    """Largest |g| at any adapted output over all demos at scale 1 (fp16 overflow guard for the real pass)."""
    m = torch.zeros((), device=next(model.parameters()).device)
    handles = []

    def record(g):          # a tensor hook must return None, or its return value replaces the gradient
        m.copy_(torch.maximum(m, g.detach().abs().max().float()))

    def fwd(mod, inp, out):
        out.register_hook(record)

    for l in range(len(model.model.layers)):
        for name in MODULES:
            handles.append(module(model, l, name).register_forward_hook(fwd))
    try:
        for ids in demos:
            _loss(model, ids.to(m.device), 1.0).backward()
    finally:
        for h in handles:
            h.remove()
    return float(m)


def accumulate(model, demos, layers, scale: float):
    """Returns ({(l, group): C_in}, {(l, module): C_out}, n_tokens). Model frozen, input grads enabled."""
    dev = next(model.parameters()).device
    C_in, C_out, handles = {}, {}, []

    def make(l, name):
        def fwd(mod, inp, out):
            if name in GROUP_OWNER:
                z = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                key = (l, IN_GROUP[name])
                if key not in C_in:
                    C_in[key] = torch.zeros(z.shape[1], z.shape[1], device=dev)
                C_in[key].addmm_(z.T, z)

            def bwd(g, key=(l, name)):
                gf = g.detach().reshape(-1, g.shape[-1]).float()
                if key not in C_out:
                    C_out[key] = torch.zeros(gf.shape[1], gf.shape[1], device=dev)
                C_out[key].addmm_(gf.T, gf)
            out.register_hook(bwd)
        return fwd

    for l in layers:
        for name in MODULES:
            handles.append(module(model, l, name).register_forward_hook(make(l, name)))
    n_tok = 0
    try:
        for ids in demos:
            ids = ids.to(dev)
            _loss(model, ids, scale).backward()
            n_tok += ids.shape[1]
            bad = [k for k, C in (*C_in.items(), *C_out.items()) if not torch.isfinite(C).all()]
            assert not bad, f"non-finite covariance {bad[:3]} (fp16 overflow: lower the loss scale)"
    finally:
        for h in handles:
            h.remove()
    return C_in, C_out, n_tok
