"""Protected bases from a covariance under budget b (guide C7, Decision 4).

Q holds the leading eigenvectors: the fewest k such that the trailing eigenvalues sum to at most b * tr C.
"""
import torch


def basis(C: torch.Tensor, b: float = 0.05) -> dict:
    """C: (d, d) symmetric fp32. Returns Q (d, k) fp32, evals (d,) descending, k, tail_frac, neg_ratio."""
    evals, evecs = torch.linalg.eigh(C)                 # ascending
    evals, evecs = evals.flip(0), evecs.flip(1)         # descending
    neg_ratio = float(evals.min() / evals.max())        # Decision 5 check: >= -1e-6
    ev = evals.clamp_min(0)
    tail = ev.flip(0).cumsum(0).flip(0)                 # tail[i] = sum_{j >= i} ev_j, mass left if i vectors kept
    budget = b * ev.sum()
    k = int((tail > budget).sum())                      # tail is non-increasing, so this is the smallest valid k
    tail_k = float(tail[k]) if k < len(ev) else 0.0
    assert tail_k <= budget + 1e-6 * float(ev.sum()), "budget rule violated"
    assert k == 0 or float(tail[k - 1]) > budget, "k is not minimal"
    return {"Q": evecs[:, :k].contiguous(), "evals": evals, "k": k,
            "tail_frac": tail_k / float(ev.sum()), "neg_ratio": neg_ratio}


def load_Q(Q_stored: torch.Tensor, device=None) -> torch.Tensor:
    """Bases are stored in fp16 to fit Kaggle's output limit; re-orthonormalise in fp32 on load.

    shortcut: QR keeps the span up to fp16 rounding (~1e-3), fine for a 5%-budget subspace; store fp32 if
    a fidelity check ever needs the exact eigenvectors.
    """
    Q = Q_stored.to(device=device, dtype=torch.float32)
    return torch.linalg.qr(Q)[0] if Q.shape[1] else Q
