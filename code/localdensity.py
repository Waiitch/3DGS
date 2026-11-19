#!/usr/bin/env python3
"""Per-gaussian local spacing (mean kNN distance), for the plane-residual term.

Computed once per pass from the KNN indices that the pipeline already builds.
"""
import torch


def local_spacing(mu, nbr, chunk=200_000):
    """mu: (N,3) tensor, nbr: (N,k) int tensor.  Returns (N,) float32."""
    N, k = nbr.shape
    out = torch.empty(N, dtype=torch.float32, device=mu.device)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        nb = mu[nbr[s:e]]                       # (B,k,3)
        d = (nb - mu[s:e][:, None, :]).norm(dim=-1)
        out[s:e] = d.mean(1)
    return out
