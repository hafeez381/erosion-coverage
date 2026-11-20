import torch


@torch.no_grad()
def project_rows_(A: torch.Tensor, Q: torch.Tensor) -> None:
    """A <- A (I - Q Q^T) in place, guide Eq. (5). A is (r, d_in), Q is (d_in, k) orthonormal fp32."""
    A.sub_(((A.float() @ Q) @ Q.T).to(A.dtype))


@torch.no_grad()
def project_cols_(B: torch.Tensor, Q: torch.Tensor) -> None:
    """B <- (I - Q Q^T) B in place. B is (d_out, r), Q is (d_out, k)."""
    B.sub_((Q @ (Q.T @ B.float())).to(B.dtype))


def random_basis(d: int, k: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Random orthonormal (d, k) fp32 basis, a stand-in for calibration bases in the C0 skeleton."""
    return torch.linalg.qr(torch.randn(d, k, generator=generator))[0]
