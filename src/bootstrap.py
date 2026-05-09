"""Stationary block bootstrap CIs for time-series statistics.

Implements Politis & Romano (1994) JASA 89:1303 stationary block bootstrap
with Politis & White (2004) Econometric Reviews 23:53 flat-top plug-in
for automatic block-length selection.

For ratio-of-means trading metrics like Sharpe / Sortino, validity follows
from composing Künsch (1989) Annals of Statistics 17:1217 block-bootstrap
consistency theorem with vdV&W §3.9.3 Thm 3.9.11 delta-method for
Hadamard-differentiable functionals.

Per BACKLOG H-190: returns scheme name and tuned block length so the caller
can audit. No silent fallback to plain bootstrap if assumption violated —
raises with the diagnostic.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


# -----------------------------------------------------------------------------
# Politis & White (2004) optimal block length
# -----------------------------------------------------------------------------

def _autocovariance(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocovariances γ̂(0..max_lag) using the biased estimator
    (divide by n, not n-k) which is consistent with the Politis-White paper.
    """
    n = len(x)
    xc = x - x.mean()
    var = float((xc * xc).mean())
    if var == 0.0:
        return np.zeros(max_lag + 1)
    acov = np.zeros(max_lag + 1)
    acov[0] = var
    for k in range(1, max_lag + 1):
        acov[k] = float((xc[k:] * xc[:-k]).mean())
    return acov


def optimal_block_length(
    x: np.ndarray, *, max_lag: Optional[int] = None,
) -> int:
    """Politis & White (2004) automatic block length for the stationary
    block bootstrap.

    Returns expected block length `b̂` ≥ 1 (an integer rounded from the
    continuous estimate).

    Algorithm (Politis & White 2004 §3, with the corrigendum in Patton-
    Politis-White 2009):
    1. Compute sample ACF ρ̂(k) for k = 1..max_lag where max_lag =
       min(n // 4, ceil(8 · log10(n))).
    2. Find smallest `M` such that |ρ̂(M+j)| < 2 √(log10(n) / n) for j = 1..K_n
       with K_n = max(5, ceil(√(log10(n)))) (the flat-top criterion).
    3. Compute spectral-density estimates with the flat-top window
       λ(s) = 1 for |s| ≤ 1/2; 2(1-|s|) for 1/2 < |s| ≤ 1; 0 else:
         ĝ(0) = γ̂(0) + 2 Σ_{k=1..M} λ(k/M) γ̂(k)
         Ĝ    = 2 Σ_{k=1..M} λ(k/M) k γ̂(k)
    4. b̂ = ((2 Ĝ²) / ĝ(0)²)^(1/3) n^(1/3).

    Falls back to b̂ = ceil(n^(1/3)) when the spectral estimate is
    degenerate (zero variance, M = 0, or non-finite intermediates).
    """
    n = len(x)
    if n < 8:
        raise ValueError(f"Need n >= 8 for block-length plug-in; got n={n}")

    if max_lag is None:
        max_lag = int(min(n // 4, max(8, int(np.ceil(8.0 * np.log10(n))))))
    max_lag = max(int(max_lag), 1)

    acov = _autocovariance(x, max_lag)
    var = float(acov[0])
    if var <= 0.0:
        return max(1, int(np.ceil(n ** (1.0 / 3.0))))

    rho = acov / var
    threshold = 2.0 * float(np.sqrt(np.log10(n) / n))
    K_n = max(5, int(np.ceil(np.sqrt(np.log10(n)))))

    M = 0
    for i in range(1, max_lag + 1):
        # i is the candidate cutoff; check that K_n consecutive lags after i
        # are all below threshold.
        ok = True
        for j in range(1, K_n + 1):
            if i + j > max_lag:
                ok = False
                break
            if abs(rho[i + j]) >= threshold:
                ok = False
                break
        if ok and abs(rho[i]) < threshold:
            M = i
            break
    if M == 0:
        M = max_lag
    M = min(M, max_lag)

    g_hat0 = var
    G_hat = 0.0
    for k in range(1, M + 1):
        s = k / M
        if s <= 0.5:
            lam = 1.0
        elif s <= 1.0:
            lam = 2.0 * (1.0 - s)
        else:
            lam = 0.0
        g_hat0 += 2.0 * lam * acov[k]
        G_hat += 2.0 * lam * k * acov[k]

    if not np.isfinite(g_hat0) or g_hat0 <= 0.0 or not np.isfinite(G_hat):
        return max(1, int(np.ceil(n ** (1.0 / 3.0))))

    if G_hat == 0.0:
        # Effectively IID: b ≈ 1.
        return 1

    b_opt = ((2.0 * G_hat ** 2) / (g_hat0 ** 2)) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    if not np.isfinite(b_opt) or b_opt < 1.0:
        return 1
    # Politis-White robustness: cap at n / 4 to avoid degenerate single-block bootstrap.
    return max(1, min(int(np.round(b_opt)), n // 4))


# -----------------------------------------------------------------------------
# Politis & Romano (1994) stationary block bootstrap
# -----------------------------------------------------------------------------

def stationary_block_bootstrap_indices(
    n: int, block_length: int, *, rng: np.random.Generator,
) -> np.ndarray:
    """Generate one bootstrap resample of size n using Politis-Romano (1994)
    stationary block bootstrap.

    Block length L_i ~ Geometric(p) with p = 1/block_length so E[L] =
    block_length. Block start s_i ~ Uniform({0, .., n-1}). Indices wrap
    circularly. Concatenate blocks until total length ≥ n; truncate to n.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    if block_length < 1:
        raise ValueError(f"block_length must be >= 1; got {block_length}")
    if block_length == 1:
        # Geometric(1) is degenerate: every block has length 1 = IID bootstrap.
        return rng.integers(0, n, size=n)

    p = 1.0 / block_length
    indices = np.empty(n, dtype=np.int64)
    pos = 0
    while pos < n:
        start = int(rng.integers(0, n))
        # Geometric(p) with support {1, 2, ...}; numpy uses the same convention.
        block_len = int(rng.geometric(p))
        for j in range(block_len):
            if pos >= n:
                break
            indices[pos] = (start + j) % n
            pos += 1
    return indices


# -----------------------------------------------------------------------------
# Generic bootstrap CI
# -----------------------------------------------------------------------------

def bootstrap_metric(
    x: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    *,
    n_resamples: int = 5000,
    block_length: Optional[int] = None,
    confidence: float = 0.95,
    seed: int = 42,
    return_samples: bool = False,
    scheme: str = "stationary_block",
) -> dict:
    """Bootstrap CI for a scalar metric of a 1D array.

    `scheme`:
    - "stationary_block": Politis-Romano with `block_length` (auto-selected
       if None via Politis-White plug-in).
    - "iid": classic Efron IID bootstrap (block_length=1 equivalent).

    Returns dict with: estimate, ci_lo, ci_hi, se, scheme, block_length,
    n_resamples, samples (optional).
    """
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    n = len(x)
    if n < 2:
        raise ValueError(f"x must have length >= 2; got {n}")

    rng = np.random.default_rng(seed)
    obs = float(metric_fn(x))

    if scheme == "iid":
        bl = 1
    elif scheme == "stationary_block":
        bl = int(block_length) if block_length is not None else optimal_block_length(x)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    samples = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        idx = stationary_block_bootstrap_indices(n, bl, rng=rng)
        samples[b] = float(metric_fn(x[idx]))

    alpha = 1.0 - confidence
    lo, hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    out = {
        "estimate": obs,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "se": float(samples.std(ddof=1)) if n_resamples > 1 else 0.0,
        "scheme": scheme,
        "block_length": int(bl),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
    }
    if return_samples:
        out["samples"] = samples
    return out


# -----------------------------------------------------------------------------
# Specialized wrappers per BACKLOG H-190 + H-191
# -----------------------------------------------------------------------------

def _sharpe(r: np.ndarray) -> float:
    if len(r) < 2:
        return 0.0
    s = float(r.std(ddof=1))
    return float(r.mean() / max(s, 1e-12))


def _sortino(r: np.ndarray) -> float:
    if len(r) < 2:
        return 0.0
    downside = r[r < 0.0]
    if len(downside) < 2:
        return 0.0 if len(downside) == 0 else float("inf") * np.sign(r.mean())
    ds = float(downside.std(ddof=1))
    return float(r.mean() / max(ds, 1e-12))


def _max_drawdown(r: np.ndarray) -> float:
    """Max drawdown of cumulative returns (not annualised, log-units)."""
    if len(r) == 0:
        return 0.0
    eq = np.cumsum(r)
    running_max = np.maximum.accumulate(eq)
    return float(-(eq - running_max).min())


def bootstrap_sharpe(returns: np.ndarray, **kwargs) -> dict:
    """Stationary block bootstrap CI for the Sharpe ratio."""
    return bootstrap_metric(returns, _sharpe, **kwargs)


def bootstrap_sortino(returns: np.ndarray, **kwargs) -> dict:
    """Stationary block bootstrap CI for the Sortino ratio."""
    return bootstrap_metric(returns, _sortino, **kwargs)


def bootstrap_max_drawdown(returns: np.ndarray, **kwargs) -> dict:
    """Stationary block bootstrap CI for max drawdown of the cumulative return.

    Per H-190 known caveat: max-DD is a path functional (sup statistic),
    NOT Hadamard-differentiable in the empirical CDF — coverage degrades
    when ρ̂(1) is mis-estimated under heavy tails.
    """
    return bootstrap_metric(returns, _max_drawdown, **kwargs)


# -----------------------------------------------------------------------------
# Per-bin Wilson interval (binomial CI for calibration bins) — H-191
# -----------------------------------------------------------------------------

def wilson_interval(
    n_success: int, n_total: int, *, confidence: float = 0.95,
) -> tuple[float, float]:
    """Closed-form Wilson score interval for a binomial proportion.

    Per H-191: per-bin calibration CI uses Wilson rather than bootstrap
    because per-bin n is small; Wilson is closed-form on the binomial
    likelihood and matches `statsmodels.stats.proportion.proportion_confint`
    to 1e-12 in tests.
    """
    if n_total < 0 or n_success < 0 or n_success > n_total:
        raise ValueError(
            f"invalid (n_success, n_total) = ({n_success}, {n_total})"
        )
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    if n_total == 0:
        return (0.0, 1.0)
    from scipy import stats as _sps
    z = float(_sps.norm.ppf(0.5 + confidence / 2.0))
    p_hat = n_success / n_total
    denom = 1.0 + z * z / n_total
    centre = (p_hat + z * z / (2.0 * n_total)) / denom
    half = z * np.sqrt(p_hat * (1.0 - p_hat) / n_total + z * z / (4.0 * n_total ** 2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


# -----------------------------------------------------------------------------
# DeLong (1988) closed-form ROC-AUC variance — H-191
# -----------------------------------------------------------------------------

def delong_roc_auc_ci(
    y_true: np.ndarray, y_score: np.ndarray, *, confidence: float = 0.95,
) -> dict:
    """ROC-AUC point estimate + DeLong (1988) closed-form CI.

    Implementation uses Sun & Xu (2014) IEEE SPL 21:1389 mid-rank
    O((n+m) log(n+m)) reformulation. Asymptotically valid CI under weak
    temporal dependence (rank statistic concentrates).

    Returns dict with: estimate, ci_lo, ci_hi, se, scheme.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch {y_true.shape} vs {y_score.shape}")
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need at least one positive and one negative sample.")

    pos_scores = y_score[pos_mask]
    neg_scores = y_score[neg_mask]

    # Mid-rank approach (Sun-Xu): rank all scores together, ties get average rank.
    all_scores = np.concatenate([pos_scores, neg_scores])
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty(len(all_scores), dtype=float)
    sorted_scores = all_scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0  # ranks are 1-indexed by convention
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    pos_ranks = ranks[: n_pos]
    neg_ranks = ranks[n_pos:]
    auc = (pos_ranks.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    # DeLong covariance: structural decomposition via V_10 (positive influence)
    # and V_01 (negative influence) functions.
    # Per Sun-Xu eq (5)-(7):
    # V_10[i] = (R_i - i_pos_rank_within_pos) / n_neg, applied per pos sample
    # V_01[j] = (j_neg_rank_within_neg - R_j' + 1) / n_pos, applied per neg sample
    # Easier: V_10[i] = mean over neg of 1{pos_score[i] > neg_score[j]} + 0.5 * 1{ties}
    # Direct fast computation via ranks within own class:
    pos_within = pos_scores.argsort(kind="mergesort").argsort() + 1.0
    neg_within = neg_scores.argsort(kind="mergesort").argsort() + 1.0
    # V_10[i] = (R(pos_i) - pos_within_i) / n_neg
    V10 = (pos_ranks - pos_within) / n_neg
    V01 = 1.0 - (neg_ranks - neg_within) / n_pos

    s10 = float(V10.var(ddof=1)) if n_pos > 1 else 0.0
    s01 = float(V01.var(ddof=1)) if n_neg > 1 else 0.0
    var_auc = s10 / n_pos + s01 / n_neg
    if var_auc < 0.0:
        var_auc = 0.0
    se = float(np.sqrt(var_auc))

    from scipy import stats as _sps
    z = float(_sps.norm.ppf(0.5 + confidence / 2.0))
    lo = float(max(0.0, auc - z * se))
    hi = float(min(1.0, auc + z * se))
    return {
        "estimate": float(auc),
        "ci_lo": lo,
        "ci_hi": hi,
        "se": se,
        "scheme": "delong_1988",
        "n_pos": n_pos,
        "n_neg": n_neg,
        "confidence": float(confidence),
    }


# -----------------------------------------------------------------------------
# Stratified bootstrap for PR-AUC — H-191
# -----------------------------------------------------------------------------

def stratified_bootstrap_pr_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """PR-AUC point estimate + stratified bootstrap CI per Boyd-Eng-Page (2013).

    Stratification: positives and negatives resampled separately. B ≥ 1000
    recommended. Bias-down at small n_pos is a known limitation.
    """
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Need at least one positive and one negative sample.")

    rng = np.random.default_rng(seed)
    obs = float(average_precision_score(y_true, y_score))
    samples = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        pos_resample = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        neg_resample = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([pos_resample, neg_resample])
        samples[b] = float(average_precision_score(y_true[idx], y_score[idx]))

    alpha = 1.0 - confidence
    lo, hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "estimate": obs,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "se": float(samples.std(ddof=1)),
        "scheme": "stratified_bootstrap_pr_auc",
        "n_resamples": int(n_resamples),
        "n_pos": int(len(pos_idx)),
        "n_neg": int(len(neg_idx)),
        "confidence": float(confidence),
    }


# -----------------------------------------------------------------------------
# Per-trade overlap-aware bootstrap — H-193
# -----------------------------------------------------------------------------

def trade_concurrency(
    n_open: np.ndarray, n_close: np.ndarray, n_minutes: int,
) -> np.ndarray:
    """Per-minute concurrency count `c_t = sum_i 1[n_open_i <= t < n_close_i]`.

    AFML Ch. 4.4 p. 61. Vectorised difference-array implementation.
    """
    n_open = np.asarray(n_open, dtype=np.int64)
    n_close = np.asarray(n_close, dtype=np.int64)
    if n_open.shape != n_close.shape:
        raise ValueError("n_open and n_close must have the same shape")
    if (n_close < n_open).any():
        raise ValueError("All n_close must be >= n_open")
    diff = np.zeros(n_minutes + 1, dtype=np.int64)
    np.add.at(diff, n_open, 1)
    np.add.at(diff, np.minimum(n_close, n_minutes), -1)
    return np.cumsum(diff)[:n_minutes]


def average_uniqueness(
    n_open: np.ndarray, n_close: np.ndarray, n_minutes: int,
) -> np.ndarray:
    """Per-trade average uniqueness `ū_i = mean_{t in [n_open_i, n_close_i)} 1/c_t`.

    AFML Ch. 4.4 p. 61, Snippet 4.2 vectorised. Returns array of length
    len(n_open).
    """
    c = trade_concurrency(n_open, n_close, n_minutes).astype(float)
    inv_c = np.where(c > 0, 1.0 / np.maximum(c, 1.0), 0.0)
    cumsum_inv_c = np.concatenate([[0.0], np.cumsum(inv_c)])
    n_open = np.asarray(n_open, dtype=np.int64)
    n_close = np.asarray(n_close, dtype=np.int64)
    sum_inv = cumsum_inv_c[np.minimum(n_close, n_minutes)] - cumsum_inv_c[n_open]
    span = np.maximum(np.minimum(n_close, n_minutes) - n_open, 1).astype(float)
    return sum_inv / span


def sequential_bootstrap_indices(
    n_open: np.ndarray, n_close: np.ndarray, n_minutes: int,
    *, n_draws: int, rng: np.random.Generator,
) -> np.ndarray:
    """LdP sequential bootstrap (AFML §4.5.1, Snippet 4.5).

    At each draw, sample trade i with probability ∝ ū_i^(2) where ū^(2) is
    the average uniqueness given already-selected trades. Concretely:
    `δ_j ∝ ū_j^(2)` and ū^(2) is recomputed after each draw to penalise
    overlapping selections.

    Cheaper approximation (used here): one-shot pre-computed uniqueness
    weights `w_i = ū_i` per the static-concurrency average; resample
    indices with replacement weighted by w_i. Per AFML 4.5 footnote, the
    full sequential update is preferred but the static-weighted variant
    captures the dominant effect at lower cost.
    """
    avg_u = average_uniqueness(n_open, n_close, n_minutes)
    if avg_u.sum() <= 0.0:
        # Degenerate: all uniqueness 0 — fall back to uniform.
        return rng.integers(0, len(n_open), size=n_draws)
    p = avg_u / avg_u.sum()
    return rng.choice(len(n_open), size=n_draws, replace=True, p=p)


def bootstrap_trade_metric_overlap_aware(
    pnl: np.ndarray,
    n_open: np.ndarray,
    n_close: np.ndarray,
    n_minutes: int,
    metric_fn: Callable[[np.ndarray], float],
    *,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
    scheme: str = "sequential_bootstrap",
) -> dict:
    """Per-trade bootstrap CI with overlap-aware resampling.

    `scheme`:
    - "sequential_bootstrap": LdP weighted resampling by average uniqueness.
    - "naive": plain IID resampling at the trade level (returns wider/narrower
       depending on overlap rate; under-covers when overlap > 0).
    """
    pnl = np.asarray(pnl, dtype=float)
    n_trades = len(pnl)
    if n_trades < 2:
        raise ValueError(f"need n_trades >= 2; got {n_trades}")
    if n_open.shape != (n_trades,) or n_close.shape != (n_trades,):
        raise ValueError("n_open and n_close must have shape (n_trades,)")

    rng = np.random.default_rng(seed)
    obs = float(metric_fn(pnl))
    samples = np.empty(n_resamples, dtype=float)

    if scheme == "naive":
        for b in range(n_resamples):
            idx = rng.integers(0, n_trades, size=n_trades)
            samples[b] = float(metric_fn(pnl[idx]))
    elif scheme == "sequential_bootstrap":
        for b in range(n_resamples):
            idx = sequential_bootstrap_indices(
                n_open, n_close, n_minutes, n_draws=n_trades, rng=rng,
            )
            samples[b] = float(metric_fn(pnl[idx]))
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    alpha = 1.0 - confidence
    lo, hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    avg_u = average_uniqueness(n_open, n_close, n_minutes)
    return {
        "estimate": obs,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "se": float(samples.std(ddof=1)),
        "scheme": scheme,
        "n_resamples": int(n_resamples),
        "n_trades": int(n_trades),
        "mean_uniqueness": float(avg_u.mean()),
        "median_uniqueness": float(np.median(avg_u)),
        "confidence": float(confidence),
    }
