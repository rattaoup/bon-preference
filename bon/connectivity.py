"""Connectivity-degree metric for Best-of-N preference datasets.

The connectivity degree is the smallest generalized eigenvalue of
``sigma_test`` with respect to ``sigma_train``, where each ``sigma`` is
the covariance of paired-embedding differences. Intuitively, it
captures how well the training distribution covers directions that
matter under the target test distribution.
"""

from __future__ import annotations

import torch
from tqdm import tqdm


@torch.no_grad()
def compute_sigma(embeddings: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Covariance of (chosen - rejected) for a ``(N, 2, d)`` embedding tensor."""
    n, _, d = embeddings.shape
    sigma = torch.zeros((d, d), dtype=torch.float64, device=embeddings.device)
    for start in tqdm(range(0, n, block_size), desc="Computing sigma"):
        end = min(start + block_size, n)
        block = embeddings[start:end]                   # (B, 2, d)
        diff = (block[:, [0], :] - block[:, [1], :]).double()  # (B, 1, d)
        sigma += (diff.transpose(1, 2) @ diff).sum(dim=0)
    sigma /= n
    return 0.5 * (sigma + sigma.T)


def smallest_generalized_eigenvalue(A: torch.Tensor, B: torch.Tensor,
                                    eps: float = 1e-6) -> torch.Tensor:
    """Smallest eigenvalue of ``A^{-1/2} B A^{-1/2}`` via Cholesky whitening.

    ``A`` must be symmetric positive definite; ``B`` must be symmetric.
    """
    I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    L = torch.linalg.cholesky(A + eps * I)
    Linv = torch.linalg.inv(L)
    M = Linv @ B @ Linv.mT
    return torch.linalg.eigvalsh(M)[..., 0]
