"""Time-series feature compute_* functions.

Implements:
- compute_hurst_rs: rolling Hurst exponent via Rescaled Range analysis (R/S)
  per Hurst (1951) — memory exponent ∈ [0, 1]; 0.5 = uncorrelated, > 0.5 =
  persistent, < 0.5 = mean-reverting. Online-compatible via cumulative
  range tracking. (H-040)
- compute_dfa_alpha: rolling Detrended Fluctuation Analysis exponent
  per Peng et al. (1994) — robust under non-stationary trends. (H-040)
- compute_sample_entropy: rolling sample entropy per Richman & Moorman (2000).
  Small-sample-biased below n ≈ 100; m=2, r = 0.2·std default. (H-040)
- compute_permutation_entropy: rolling Bandt-Pompe (2002) normalised
  permutation entropy on m=3, τ=1 ordinal patterns. (H-115)

All functions follow CONSTITUTION I.1 (no future data), I.4 (NaN flag-and-impute
discipline — return NaN at boundaries; let caller add `undef__*` flags), and
the existing `compute_*` registry contract. Tests in `tests/test_features.py`.

References (per RESEARCH/literature/INDEX.md):
- Hurst (1951) "Long-term storage capacity of reservoirs" Trans. ASCE 116:770.
- Peng et al. (1994) Phys. Rev. E 49:1685 — original DFA.
- Bandt & Pompe (2002) Phys. Rev. Lett. 88:174102 — permutation entropy.
- Richman & Moorman (2000) Am. J. Physiol. 278:H2039 — sample entropy.
- Cover & Thomas Ch. 8 (entropy bound) local Desktop.
- López de Prado, AFML Ch. 5 (frac-diff for memory residualisation).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Hurst / R/S
# -----------------------------------------------------------------------------

def _rs_single_window(x: np.ndarray) -> float:
    """Rescaled Range R/S for a single window of length n.

    `R/S(n) = (max cum(x - mean) - min cum(x - mean)) / std(x)`. Returns NaN
    when std == 0.
    """
    n = len(x)
    if n < 4:
        return float("nan")
    mu = float(np.mean(x))
    deviations = x - mu
    cum = np.cumsum(deviations)
    R = float(cum.max() - cum.min())
    S = float(np.std(x, ddof=1))
    if S == 0.0:
        return float("nan")
    return R / S


def compute_hurst_rs(
    series: pd.Series, *, window: int, scales: Sequence[int] | None = None,
) -> pd.Series:
    """Rolling Hurst exponent via R/S analysis.

    Slope of log(R/S(scale)) vs log(scale) over `scales` ⊂ [4, window/2],
    fit by least-squares. Returns NaN for the first `window-1` indices.

    Parameters
    ----------
    series : pd.Series
        Input 1D series (returns or prices in log-space).
    window : int
        Rolling window length.
    scales : sequence of int, optional
        Scales at which to compute R/S. Defaults to a geometric grid
        between 8 and `window // 2`.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if window < 16:
        raise ValueError(f"window must be >= 16 for Hurst stability; got {window}")
    if scales is None:
        max_scale = max(window // 2, 8)
        # 5-8 geometric scales between 8 and max_scale
        scales = sorted(set(int(round(s)) for s in
                            np.geomspace(8, max_scale, num=6)))
    scales = [s for s in scales if 4 <= s <= window // 2]
    if len(scales) < 2:
        raise ValueError(f"need >= 2 valid scales; got {scales} for window={window}")

    out = np.full(n, np.nan, dtype=float)
    for i in range(window - 1, n):
        win = x[i - window + 1: i + 1]
        if not np.all(np.isfinite(win)):
            continue
        log_rs = []
        log_n = []
        for s in scales:
            n_segments = window // s
            if n_segments < 1:
                continue
            rs_values = []
            for seg in range(n_segments):
                seg_x = win[seg * s: (seg + 1) * s]
                rs = _rs_single_window(seg_x)
                if np.isfinite(rs):
                    rs_values.append(rs)
            if len(rs_values) == 0:
                continue
            log_rs.append(np.log(np.mean(rs_values)))
            log_n.append(np.log(s))
        if len(log_rs) < 2:
            continue
        slope, _ = np.polyfit(log_n, log_rs, 1)
        out[i] = float(slope)
    return pd.Series(out, index=series.index, name=f"hurst_rs_{window}")


# -----------------------------------------------------------------------------
# DFA
# -----------------------------------------------------------------------------

def _dfa_single_window(x: np.ndarray, scales: Sequence[int]) -> float:
    """DFA exponent for a single window.

    1. Cumulative profile y = cumsum(x - mean(x)).
    2. For each scale s, partition y into floor(n/s) non-overlapping segments.
    3. Linearly detrend each segment; compute RMS of residuals = F(s).
    4. DFA exponent α = slope of log F(s) vs log s.
    """
    n = len(x)
    if n < 16:
        return float("nan")
    y = np.cumsum(x - np.mean(x))
    log_F = []
    log_s = []
    for s in scales:
        n_segments = n // s
        if n_segments < 1 or s < 4:
            continue
        residuals_sq = []
        for seg in range(n_segments):
            seg_y = y[seg * s: (seg + 1) * s]
            t = np.arange(s)
            slope, intercept = np.polyfit(t, seg_y, 1)
            trend = slope * t + intercept
            residuals_sq.append(np.mean((seg_y - trend) ** 2))
        if len(residuals_sq) == 0:
            continue
        F_s = np.sqrt(np.mean(residuals_sq))
        if F_s <= 0.0:
            continue
        log_F.append(np.log(F_s))
        log_s.append(np.log(s))
    if len(log_F) < 2:
        return float("nan")
    slope, _ = np.polyfit(log_s, log_F, 1)
    return float(slope)


def compute_dfa_alpha(
    series: pd.Series, *, window: int, scales: Sequence[int] | None = None,
) -> pd.Series:
    """Rolling Detrended Fluctuation Analysis exponent.

    Parameters
    ----------
    series : pd.Series
        Input 1D series.
    window : int
        Rolling window length (must be >= 16).
    scales : sequence of int, optional
        Scales at which to compute F(s). Defaults to a geometric grid
        between 4 and `window // 4`.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if window < 16:
        raise ValueError(f"window must be >= 16 for DFA stability; got {window}")
    if scales is None:
        max_scale = max(window // 4, 4)
        scales = sorted(set(int(round(s)) for s in
                            np.geomspace(4, max_scale, num=6)))
    scales = [s for s in scales if 4 <= s <= window // 2]
    if len(scales) < 2:
        raise ValueError(f"need >= 2 valid scales; got {scales}")

    out = np.full(n, np.nan, dtype=float)
    for i in range(window - 1, n):
        win = x[i - window + 1: i + 1]
        if not np.all(np.isfinite(win)):
            continue
        out[i] = _dfa_single_window(win, scales)
    return pd.Series(out, index=series.index, name=f"dfa_alpha_{window}")


# -----------------------------------------------------------------------------
# Sample entropy
# -----------------------------------------------------------------------------

def _sample_entropy_single_window(
    x: np.ndarray, m: int, r: float,
) -> float:
    """Sample entropy per Richman & Moorman (2000)."""
    n = len(x)
    if n < m + 2:
        return float("nan")
    if r <= 0.0:
        return float("nan")

    def _phi(m_local: int) -> float:
        templates = np.array([x[i: i + m_local] for i in range(n - m_local)])
        if len(templates) < 2:
            return float("nan")
        # Pairwise Chebyshev distance ≤ r counts.
        count = 0
        for i in range(len(templates) - 1):
            diff = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            count += int((diff <= r).sum())
        # Symmetric pair count: total = N*(N-1)/2 where N=len(templates).
        total = len(templates) * (len(templates) - 1) // 2
        return count / max(total, 1)

    A = _phi(m + 1)
    B = _phi(m)
    if not np.isfinite(A) or not np.isfinite(B) or B == 0.0:
        return float("nan")
    if A == 0.0:
        # Per Richman & Moorman: SampEn = -ln(0/B) is undefined;
        # return a large but finite value.
        return float("inf")
    return -float(np.log(A / B))


def compute_sample_entropy(
    series: pd.Series, *, window: int, m: int = 2, r_factor: float = 0.2,
) -> pd.Series:
    """Rolling sample entropy.

    Per Richman & Moorman (2000), with `r = r_factor * std(window)`.
    Default m=2, r_factor=0.2 are standard for finance applications.
    Small-sample-biased below n ≈ 100 — caller should ensure window >= 100.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if window < 32:
        raise ValueError(f"window must be >= 32 for SampEn stability; got {window}")
    if m < 1:
        raise ValueError(f"m must be >= 1; got {m}")
    if r_factor <= 0.0:
        raise ValueError(f"r_factor must be > 0; got {r_factor}")

    out = np.full(n, np.nan, dtype=float)
    for i in range(window - 1, n):
        win = x[i - window + 1: i + 1]
        if not np.all(np.isfinite(win)):
            continue
        sigma = float(np.std(win, ddof=1))
        if sigma == 0.0:
            continue
        r = r_factor * sigma
        out[i] = _sample_entropy_single_window(win, m, r)
    return pd.Series(out, index=series.index, name=f"sampen_{window}_m{m}")


# -----------------------------------------------------------------------------
# Permutation entropy (Bandt-Pompe)
# -----------------------------------------------------------------------------

def _permutation_entropy_single_window(
    x: np.ndarray, m: int, tau: int,
) -> float:
    """Bandt-Pompe permutation entropy with stable mergesort tie-breaking.

    Returns normalised permutation entropy in [0, 1], where 1 = uniform
    over m! ordinal patterns (max disorder), 0 = single dominating pattern.
    """
    from math import factorial

    n = len(x)
    n_patterns = n - (m - 1) * tau
    if n_patterns < 2:
        return float("nan")

    # Build ordinal patterns. Use mergesort for stable tie-breaking
    # per the H-115 BACKLOG spec.
    pattern_strs = np.empty(n_patterns, dtype=object)
    for i in range(n_patterns):
        embed = x[i: i + (m - 1) * tau + 1: tau]
        if not np.all(np.isfinite(embed)):
            pattern_strs[i] = None
            continue
        order = np.argsort(embed, kind="mergesort")
        pattern_strs[i] = tuple(int(o) for o in order)

    valid = pattern_strs[pattern_strs != None]  # noqa: E711
    if len(valid) < 2:
        return float("nan")

    unique, counts = np.unique(valid, return_counts=True)
    p = counts / counts.sum()
    H = float(-np.sum(p * np.log(p)))
    H_max = float(np.log(factorial(m)))
    return H / H_max if H_max > 0.0 else float("nan")


def compute_permutation_entropy(
    series: pd.Series, *, window: int, m: int = 3, tau: int = 1,
) -> pd.Series:
    """Rolling normalised Bandt-Pompe permutation entropy.

    Per H-115 spec: m=3, tau=1 default with stable mergesort tie-breaking.
    Returns values in [0, 1]; 1.0 = uniform over m! = 6 patterns
    (maximum disorder); 0 = pure trend.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if window < (m - 1) * tau + 4:
        raise ValueError(
            f"window must be >= (m-1)*tau + 4 = {(m-1)*tau + 4}; got {window}"
        )
    if m < 2:
        raise ValueError(f"m must be >= 2; got {m}")
    if tau < 1:
        raise ValueError(f"tau must be >= 1; got {tau}")

    out = np.full(n, np.nan, dtype=float)
    for i in range(window - 1, n):
        win = x[i - window + 1: i + 1]
        if not np.all(np.isfinite(win)):
            continue
        out[i] = _permutation_entropy_single_window(win, m, tau)
    return pd.Series(out, index=series.index, name=f"permen_{window}_m{m}_tau{tau}")


# -----------------------------------------------------------------------------
# H-130: derived microstructure flow features (taker buy ratio, signed flow)
# -----------------------------------------------------------------------------

def compute_taker_buy_ratio(
    bars: pd.DataFrame, *, windows: Sequence[int],
) -> pd.DataFrame:
    """Bar-aggregated VPIN imbalance proxy.

    Per AFML §18.8.4 p. 276: `|2·v^B − 1|` is the canonical VPIN form.
    We implement the signed (non-absolute) variant alongside the absolute
    form, since the upper-barrier label is one-sided.

    Requires `taker_buy_base` and `volume` columns.

    Returns: DataFrame with columns
        `flow_taker_buy_ratio_w{w}` for each w in windows (rolling mean).
        `flow_taker_buy_ratio_abs_w{w}` for each w (rolling abs).

    Strict CONSTITUTION I.1: rolling mean uses past bars only.
    """
    if "taker_buy_base" not in bars.columns or "volume" not in bars.columns:
        raise ValueError("bars must have 'taker_buy_base' and 'volume' columns")
    out = pd.DataFrame(index=bars.index)
    instant_ratio = bars["taker_buy_base"] / bars["volume"].replace(0, np.nan)
    instant_signed = 2.0 * instant_ratio - 1.0  # in [-1, 1]
    for w in windows:
        out[f"flow_taker_buy_ratio_w{w}"] = (
            instant_signed.rolling(w, min_periods=w).mean()
        )
        out[f"flow_taker_buy_ratio_abs_w{w}"] = (
            instant_signed.rolling(w, min_periods=w).mean().abs()
        )
    return out


def compute_signed_dollar_flow(
    bars: pd.DataFrame, *, windows: Sequence[int],
) -> pd.DataFrame:
    """Signed dollar flow per bar = `taker_buy_quote − (quote_volume −
    taker_buy_quote) = 2·taker_buy_quote − quote_volume`.

    Returns rolling mean over each window in windows; sign in {−1, 0, +1}
    on each bar by sign of signed flow.

    Strict CONSTITUTION I.1: rolling uses past bars only.
    """
    needed = {"taker_buy_quote", "quote_volume"}
    if not needed.issubset(bars.columns):
        raise ValueError(f"bars must have columns {needed}")
    out = pd.DataFrame(index=bars.index)
    instant = 2.0 * bars["taker_buy_quote"] - bars["quote_volume"]
    instant_normalized = instant / bars["quote_volume"].replace(0, np.nan)
    for w in windows:
        out[f"flow_signed_dollar_w{w}"] = (
            instant.rolling(w, min_periods=w).mean()
        )
        out[f"flow_signed_dollar_norm_w{w}"] = (
            instant_normalized.rolling(w, min_periods=w).mean()
        )
    return out


# -----------------------------------------------------------------------------
# H-131: BPV/RV jump-detector + signed semivariance asymmetry + vov
# -----------------------------------------------------------------------------

def compute_bpv_ratio(
    log_returns: pd.Series, *, windows: Sequence[int],
) -> pd.DataFrame:
    """Bipower-variation / Realized-Variance ratio across windows.

    Barndorff-Nielsen & Shephard (2004): `BPV = (π/2) · Σ |r_{i-1}| · |r_i|`
    is jump-robust; `RV = Σ r_i^2`. Ratio `BPV / RV ∈ (0, 1]` is a
    continuous-vs-jump regime indicator (1.0 = no jumps, < 1 = jumps present).

    Strict t ≤ k−1 (the BACKLOG H-131 leakage rule): rolling uses past
    returns only.

    Returns DataFrame with columns `bpv_rv_ratio_w{w}` for each w.
    """
    r = log_returns.to_numpy(dtype=float)
    n = len(r)
    out = pd.DataFrame(index=log_returns.index)
    abs_r = np.abs(r)
    rv_per_bar = r ** 2
    bpv_per_bar = np.full(n, np.nan, dtype=float)
    bpv_per_bar[1:] = (np.pi / 2.0) * abs_r[:-1] * abs_r[1:]
    s_rv = pd.Series(rv_per_bar, index=log_returns.index)
    s_bpv = pd.Series(bpv_per_bar, index=log_returns.index)
    for w in windows:
        rv = s_rv.rolling(w, min_periods=w).sum()
        bpv = s_bpv.rolling(w, min_periods=w).sum()
        ratio = bpv / rv.replace(0, np.nan)
        out[f"bpv_rv_ratio_w{w}"] = ratio
    return out


def compute_signed_semivariance(
    log_returns: pd.Series, *, windows: Sequence[int],
) -> pd.DataFrame:
    """Signed semivariance asymmetry: `SV_up − SV_down`.

    Per Barndorff-Nielsen-Kinnebrock-Shephard (2010): `SV^- = Σ r_i^2 1{r_i<0}`,
    `SV^+ = Σ r_i^2 1{r_i>0}`. Asymmetry `SV_up − SV_down` is direction-
    aware variance; mechanically aligned with one-sided upper-barrier label
    (high SV_up → upward variance dominates → label more likely 1).

    Strict t ≤ k−1: rolling uses past returns only. Returns:
        `sv_up_w{w}` per window
        `sv_down_w{w}` per window
        `sv_asymmetry_w{w}` = SV_up − SV_down

    Failure mode (H-131): under simultaneous price+volume jumps, both
    BPV terms inflate and asymmetry over-estimates direction.
    """
    r = log_returns.to_numpy(dtype=float)
    sv_up_per = np.where(r > 0.0, r ** 2, 0.0)
    sv_down_per = np.where(r < 0.0, r ** 2, 0.0)
    s_up = pd.Series(sv_up_per, index=log_returns.index)
    s_down = pd.Series(sv_down_per, index=log_returns.index)
    out = pd.DataFrame(index=log_returns.index)
    for w in windows:
        sv_up = s_up.rolling(w, min_periods=w).sum()
        sv_down = s_down.rolling(w, min_periods=w).sum()
        out[f"sv_up_w{w}"] = sv_up
        out[f"sv_down_w{w}"] = sv_down
        out[f"sv_asymmetry_w{w}"] = sv_up - sv_down
    return out


def compute_vol_of_vol(
    log_returns: pd.Series, *, vol_window: int, vov_window: int,
) -> pd.Series:
    """Vol-of-vol: rolling std of rolling std of returns.

    Two-stage: first compute rolling std with `vol_window`, then std of that
    with `vov_window`. Returns NaN for the first `vol_window + vov_window − 1`
    indices.

    Strict CONSTITUTION I.1: each stage uses past data only.
    """
    rv = log_returns.rolling(vol_window, min_periods=vol_window).std()
    vov = rv.rolling(vov_window, min_periods=vov_window).std()
    vov.name = f"vov_{vol_window}_{vov_window}"
    return vov
