"""Combinatorially Symmetric Cross-Validation — PBO (López de Prado 2014).

Ported from src/backtest.py::cscv_pbo. Stays here as a wagie utility because
no maintained PyPI package implements it (verified Wave 1).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def cscv_pbo(
    returns_matrix: np.ndarray,
    *,
    n_chunks: int = 16,
    eps: float = 1e-12,
) -> dict:
    """Backtest-overfit probability via Combinatorially Symmetric CV.

    Input: (n_strategies, n_periods) matrix of per-period returns.
    Output: dict with `pbo`, `median_logit`, `is_best_strategy_modal_index`, etc.
    """
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim != 2:
        raise ValueError(f"returns_matrix must be 2D; got {R.ndim}D")
    n_strats, n_periods = R.shape
    if n_strats < 2:
        raise ValueError(f"need >= 2 strategies; got {n_strats}")
    if n_chunks < 2 or n_chunks % 2 != 0:
        raise ValueError(f"n_chunks must be a positive even number; got {n_chunks}")
    chunk_size = n_periods // n_chunks
    if chunk_size < 1:
        raise ValueError(f"n_periods {n_periods} too small for n_chunks {n_chunks}")
    n_total = chunk_size * n_chunks
    R = R[:, :n_total]
    R_chunks = R.reshape(n_strats, n_chunks, chunk_size)

    half = n_chunks // 2
    logits: list[float] = []
    rel_ranks: list[float] = []
    is_best_counts = np.zeros(n_strats, dtype=int)

    for IS_idx in combinations(range(n_chunks), half):
        IS_set = set(IS_idx)
        OOS_idx = tuple(c for c in range(n_chunks) if c not in IS_set)

        IS_flat = R_chunks[:, IS_idx, :].reshape(n_strats, -1)
        OOS_flat = R_chunks[:, OOS_idx, :].reshape(n_strats, -1)

        IS_mean = IS_flat.mean(axis=1)
        IS_std = IS_flat.std(axis=1, ddof=1)
        OOS_mean = OOS_flat.mean(axis=1)
        OOS_std = OOS_flat.std(axis=1, ddof=1)

        IS_sharpe = IS_mean / np.maximum(IS_std, eps)
        OOS_sharpe = OOS_mean / np.maximum(OOS_std, eps)

        best = int(np.argmax(IS_sharpe))
        is_best_counts[best] += 1
        rank = int((OOS_sharpe < OOS_sharpe[best]).sum()) + 1
        omega = (rank - 0.5) / n_strats
        omega_clip = float(np.clip(omega, eps, 1.0 - eps))
        logits.append(float(np.log(omega_clip / (1.0 - omega_clip))))
        rel_ranks.append(omega)

    logits_arr = np.asarray(logits, dtype=float)
    rel_ranks_arr = np.asarray(rel_ranks, dtype=float)
    pbo = float((logits_arr < 0).mean())
    return {
        "pbo": pbo,
        "n_combinations": int(len(logits)),
        "n_chunks": int(n_chunks),
        "chunk_size": int(chunk_size),
        "n_strategies": int(n_strats),
        "median_logit": float(np.median(logits_arr)),
        "mean_logit": float(np.mean(logits_arr)),
        "median_rel_rank": float(np.median(rel_ranks_arr)),
        "is_best_strategy_modal_index": int(is_best_counts.argmax()),
        "is_best_strategy_counts": is_best_counts.tolist(),
    }


__all__ = ["cscv_pbo"]
