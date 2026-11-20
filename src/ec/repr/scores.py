"""Class means and the Zhao scores H and R, guide Eq. (1)."""
import torch
import torch.nn.functional as F

T_INST, T_POST = 0, 1  # index on the position axis of a cache


def class_means(h: torch.Tensor) -> torch.Tensor:
    """(N, L, 2, d) cache slice of one class -> (L, 2, d) fp32 means."""
    return h.float().mean(0)


def score(h: torch.Tensor, mu_pos: torch.Tensor, mu_neg: torch.Tensor, band: list[int], pos: int) -> torch.Tensor:
    """Mean over the band of cos(h, mu_pos) - cos(h, mu_neg). pos=T_INST gives H, T_POST gives R."""
    hb = h[:, band, pos].float()                       # (N, |B|, d)
    s = F.cosine_similarity(hb, mu_pos[band, pos], dim=-1) - F.cosine_similarity(hb, mu_neg[band, pos], dim=-1)
    return s.mean(1)
