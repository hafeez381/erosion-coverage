"""DualNullLoRA (guide C8, Decisions 3 and 4) around a PEFT LoRA model.

Each adapted projection f(z) = Wz + s B A z keeps A out of span(Q_in) and B out of span(Q_out):
gradients are projected before the optimizer step and parameters after it. Initialisation is the top-r
SVD of P_out W P_in, subtracted from the frozen weight so the model is unchanged at step 0.

Bases are held in fp16 on the GPU (4.3 GB for the 3B model in fp32 would not fit beside training).
fp16 columns are not exactly orthonormal, so each projection is applied `passes` times: the residual
X Q shrinks by a factor ||I - Q^T Q|| per pass and converges to exactly X Q = 0.
"""
from pathlib import Path

import torch

from ec.dualnull.bases import load_Q
from ec.dualnull.calibrate import IN_GROUP, MODULES

ATTN = {"q_proj", "k_proj", "v_proj", "o_proj"}


def lora_module(peft_model, layer: int, name: str):
    blk = peft_model.base_model.model.model.layers[layer]
    return getattr(blk.self_attn if name in ATTN else blk.mlp, name)


def load_bases(path, n_layers: int, device) -> dict:
    """{(layer, module): (Q_in, Q_out)} in fp16; projections sharing an input share one Q_in tensor."""
    out = {}
    for l in range(n_layers):
        d = torch.load(Path(path) / f"layer_{l:02d}.pt", map_location="cpu")
        q_in = {g: load_Q(v["Q"], device).half() for g, v in d["in"].items()}
        for m in MODULES:
            out[(l, m)] = (q_in[IN_GROUP[m]], load_Q(d["out"][m]["Q"], device).half())
    return out


@torch.no_grad()
def proj_rows(X: torch.Tensor, Q: torch.Tensor, passes: int = 3) -> torch.Tensor:
    """X (r, d) -> X (I - Q Q^T), so that X Q = 0. Returns fp32."""
    X, Qf = X.float(), Q.float()
    for _ in range(passes):
        X = X - (X @ Qf) @ Qf.T
    return X


@torch.no_grad()
def proj_cols(X: torch.Tensor, Q: torch.Tensor, passes: int = 3) -> torch.Tensor:
    """X (d, r) -> (I - Q Q^T) X, so that Q^T X = 0. Returns fp32."""
    X, Qf = X.float(), Q.float()
    for _ in range(passes):
        X = X - Qf @ (Qf.T @ X)
    return X


class DualNullLoRA:
    def __init__(self, peft_model, bases: dict, scale: float, passes: int = 3):
        self.model, self.bases, self.s, self.passes = peft_model, bases, scale, passes
        self.mods = {k: lora_module(peft_model, *k) for k in bases}
        self.hooks = []

    def A(self, k):
        return self.mods[k].lora_A["default"].weight

    def B(self, k):
        return self.mods[k].lora_B["default"].weight

    @torch.no_grad()
    def init_balanced_svd_(self) -> dict:
        """B0 A0 = top-r SVD of P_out W P_in / s; frozen weight <- W - s B0 A0. Returns {key: (A0, B0)}."""
        init = {}
        for k, mod in self.mods.items():
            Qi, Qo = self.bases[k]
            W = mod.base_layer.weight
            M = proj_cols(proj_rows(W, Qi, self.passes), Qo, self.passes)        # fp32 (d_out, d_in)
            r = self.A(k).shape[0]
            U, S, V = torch.svd_lowrank(M, q=min(r + 8, min(M.shape)), niter=4)
            root = S[:r].sqrt()
            B0 = U[:, :r] * root / self.s ** 0.5
            A0 = (root[:, None] * V[:, :r].T) / self.s ** 0.5
            A0, B0 = proj_rows(A0, Qi, self.passes), proj_cols(B0, Qo, self.passes)
            self.A(k).copy_(A0.to(self.A(k).dtype))
            self.B(k).copy_(B0.to(self.B(k).dtype))
            W.copy_((W.float() - self.s * B0 @ A0).to(W.dtype))
            init[k] = (A0.cpu(), B0.cpu())
            del M, U, S, V
        return init

    def add_grad_hooks(self):
        """Project .grad of every A and B before the optimizer sees it (the paper's rule)."""
        for k in self.mods:
            Qi, Qo = self.bases[k]
            self.hooks.append(self.A(k).register_hook(lambda g, Q=Qi: proj_rows(g, Q, self.passes).to(g.dtype)))
            self.hooks.append(self.B(k).register_hook(lambda g, Q=Qo: proj_cols(g, Q, self.passes).to(g.dtype)))

    @torch.no_grad()
    def project_(self):
        """Re-project the parameters after optimizer.step() (Decision 3: AdamW leaks otherwise)."""
        for k in self.mods:
            Qi, Qo = self.bases[k]
            A, B = self.A(k), self.B(k)
            A.copy_(proj_rows(A, Qi, self.passes).to(A.dtype))
            B.copy_(proj_cols(B, Qo, self.passes).to(B.dtype))

    @torch.no_grad()
    def residual(self) -> tuple[float, float]:
        """max over modules of ||A Q_in||_F / ||A||_F and ||Q_out^T B||_F / ||B||_F."""
        ra = rb = 0.0
        for k in self.mods:
            Qi, Qo = self.bases[k]
            A, B = self.A(k).float(), self.B(k).float()
            if Qi.shape[1]:
                ra = max(ra, float((A @ Qi.float()).norm() / A.norm().clamp_min(1e-30)))
            if Qo.shape[1]:
                rb = max(rb, float((Qo.float().T @ B).norm() / B.norm().clamp_min(1e-30)))
        return ra, rb


@torch.no_grad()
def fidelity(peft_model, keys, batches) -> dict:
    """Per module: mean ||A z|| / (||A||_2 ||z||) over tokens (unweighted), and the ||z||^2-weighted mean
    of ||A z||^2 / (||A||_2^2 ||z||^2). `batches` are input_ids tensors (batch 1, no padding)."""
    acc = {k: [0.0, 0.0, 0.0, 0] for k in keys}        # sum ratio, sum ||Az||^2/||A||^2, sum ||z||^2, n
    handles = []

    def make(k, mod):
        A = mod.lora_A["default"].weight
        def hook(m, inp, out):
            z = inp[0].reshape(-1, inp[0].shape[-1]).float()
            An = torch.linalg.matrix_norm(A.float(), 2)
            Az = (z @ A.float().T).norm(dim=1)
            zn = z.norm(dim=1).clamp_min(1e-30)
            a = acc[k]
            a[0] += float((Az / (An * zn)).sum())
            a[1] += float((Az ** 2).sum() / An ** 2)
            a[2] += float((zn ** 2).sum())
            a[3] += z.shape[0]
        return hook

    for k in keys:
        mod = lora_module(peft_model, *k)
        handles.append(mod.register_forward_hook(make(k, mod)))
    try:
        dev = next(peft_model.parameters()).device
        for ids in batches:
            peft_model(input_ids=ids.to(dev))
    finally:
        for h in handles:
            h.remove()
    return {k: {"unweighted": a[0] / a[3], "weighted_sq": a[1] / a[2]} for k, a in acc.items()}
