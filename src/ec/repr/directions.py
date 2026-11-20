"""Directions (guide Eq. 2), per-layer AUROC and the layer band (Decision 6)."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from ec.repr.scores import T_INST, T_POST


def fit(h_harm: torch.Tensor, h_harmless: torch.Tensor) -> dict:
    """Class means on the fit split and unit directions. Caches are (N, L, 2, d).

    v_harm is taken at t_inst, v_ref at t_post (Zhao et al.: harmfulness vs refusal)."""
    mu_h, mu_n = h_harm.float().mean(0), h_harmless.float().mean(0)       # (L, 2, d)
    return {"mu_harmful": mu_h, "mu_harmless": mu_n,
            "v_harm": F.normalize(mu_h[:, T_INST] - mu_n[:, T_INST], dim=-1),
            "v_ref": F.normalize(mu_h[:, T_POST] - mu_n[:, T_POST], dim=-1)}


def layer_scores(h: torch.Tensor, d: dict, pos: int) -> torch.Tensor:
    """s^l for every layer: (N, L)."""
    x = h[:, :, pos].float()
    return (F.cosine_similarity(x, d["mu_harmful"][:, pos], dim=-1)
            - F.cosine_similarity(x, d["mu_harmless"][:, pos], dim=-1))


def layer_auroc(s_pos: torch.Tensor, s_neg: torch.Tensor) -> np.ndarray:
    y = np.r_[np.ones(len(s_pos)), np.zeros(len(s_neg))]
    s = torch.cat([s_pos, s_neg]).numpy()
    return np.array([roc_auc_score(y, s[:, l]) for l in range(s.shape[1])])


def choose_band(auroc: np.ndarray, thr: float = 0.95, max_frac: float = 0.6, width: int = 8) -> list[int]:
    """Decision 6. Cache indices (index l is block l+1). The last layer is excluded (final RMSNorm).

    Longest contiguous run of layers with AUROC >= thr; if more than max_frac of layers pass, the
    width-layer window centred on the best layer instead."""
    L = len(auroc)
    a = auroc[: L - 1]
    ok = a >= thr
    if ok.sum() > max_frac * L:
        lo = int(np.clip(int(a.argmax()) - width // 2, 0, L - 1 - width))
        return list(range(lo, lo + width))
    best, run = [], []
    for l in range(L - 1):
        run = run + [l] if ok[l] else []
        if len(run) > len(best):
            best = run
    if not best:
        raise ValueError(f"no layer reaches AUROC {thr}; max is {a.max():.3f} (Decision 6 wrong-choice signature)")
    return best
