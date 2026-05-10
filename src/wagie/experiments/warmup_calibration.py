"""Warmup-phase calibration — compute σ_max / r*_init / τ_init / t_max once.

This module is the **pure, testable** statistical reducer that runs ONCE before
trading begins. The caller (the experiment driver) is responsible for streaming
the warmup bars through the pipeline so that each :class:`Observation` arrives
here already carrying ``sigma_ve`` and ``p_online``. This module:

  1. consumes those Observations + their hindsight labels,
  2. (optionally) checks ARF rolling-Brier stability across the warmup,
  3. computes the four parameters the adaptive strategy needs:

         σ_max     — 75th percentile of warmup ``sigma_ve`` (gate threshold)
         r*_init   — empirical event rate over the warmup window
         τ_init    — quantile of ``p_online`` such that P(p ≥ τ) ≈ r*_init,
                     floored at the EV-breakeven ``tau_floor``
         t_max     — high-quantile of training-set time-to-first-barrier
                     (passed in as an array; not derived from warmup)

  4. records baseline Brier scores for both the online and offline probabilities
     so the integration step can later quantify the value the online layer adds.

All gating decisions are explicit. The function never silently absorbs degenerate
input — it raises :class:`CalibrationError` with a descriptive message when:

  * the warmup window had no opportunities (``r*_init < r_min``), or
  * the σ_max gate is non-restrictive (``σ_max ≥ 95% of max(sigma_ve)``).

Non-convergence of the ARF rolling Brier is recorded on the result
(``converged=False``) but does **not** raise — the caller decides how to react.

Wire-up contract
----------------
* The function takes pre-computed Observations. It does **not** invoke the
  pipeline. The driver runs ``pipeline.update()`` in its own loop and yields
  ``(observation, label)`` pairs into this module.
* Brier is implemented inline (mean squared error between probability and
  binary label). No dependency on ``wagie.metrics`` so this module stays free
  of the heavy battery import graph.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Public errors
# ---------------------------------------------------------------------------


class CalibrationError(Exception):
    """Raised when warmup statistics fail a sanity gate.

    The message is intended for an operator: it states the gate that tripped,
    the observed value, and the floor / threshold that was violated.
    """


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class WarmupCalibration:
    """Calibrated parameters and diagnostics produced by the warmup pass.

    Every field is a plain Python scalar / list of floats so the value
    round-trips through JSON without numpy-scalar leakage. The driver persists
    ``to_dict()`` under the run's artifacts directory; the strategy reads it
    back in.
    """

    tau_init: float
    sigma_max: float
    r_star_init: float
    t_max: int
    brier_baseline: float            # Brier of p_online over warmup
    brier_offline_baseline: float    # Brier of p_offline over warmup
    warmup_bars_used: int
    arf_stability_trace: list[float] = field(default_factory=list)
    converged: bool = False

    def to_dict(self) -> dict:
        """JSON-serializable view (all values are floats / ints / lists)."""
        d = asdict(self)
        # Defensive cast — asdict already returns Python primitives but the
        # caller may have passed in numpy scalars upstream.
        d["tau_init"] = float(d["tau_init"])
        d["sigma_max"] = float(d["sigma_max"])
        d["r_star_init"] = float(d["r_star_init"])
        d["t_max"] = int(d["t_max"])
        d["brier_baseline"] = float(d["brier_baseline"])
        d["brier_offline_baseline"] = float(d["brier_offline_baseline"])
        d["warmup_bars_used"] = int(d["warmup_bars_used"])
        d["arf_stability_trace"] = [float(x) for x in d["arf_stability_trace"]]
        d["converged"] = bool(d["converged"])
        return d


# ---------------------------------------------------------------------------
# Inline Brier
# ---------------------------------------------------------------------------


def _brier(probs: list[float], labels: list[int]) -> float:
    """Mean squared error between probability and binary label.

    Defined for the empty input as 0.0 (caller already gates on bar count).
    """
    if not probs:
        return 0.0
    n = len(probs)
    s = 0.0
    for p, y in zip(probs, labels):
        d = float(p) - float(y)
        s += d * d
    return s / n


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calibrate_from_warmup(
    warmup_observations: Iterable,
    training_label_ttbarrier,
    *,
    quantile_sigma: float = 0.75,
    stability_window: int = 50,
    stability_threshold: float = 0.05,
    min_warmup_bars: int = 100,
    max_warmup_bars: int | None = 200,
    tau_floor: float = 0.524,
    r_min: float = 0.02,
    t_max_quantile: float = 0.95,
) -> WarmupCalibration:
    """Reduce a stream of warmup ``(Observation, label)`` pairs to calibrated knobs.

    Streaming behaviour
    -------------------
    The function pulls observations one at a time. After ``min_warmup_bars`` it
    starts checking ARF rolling-Brier stability every
    ``stability_window // 2`` bars. Stability is

        |brier(recent_window) - brier(prev_window)| / brier(prev_window) < threshold

    where ``recent_window`` is the most recent ``stability_window`` p_online /
    label pairs and ``prev_window`` is the window immediately preceding it. The
    very first check (which has no ``prev_window`` Brier yet) seeds the
    comparison and does not flip ``converged``.

    Termination
    -----------
    The loop terminates as soon as one of the following is true:

      * ``converged`` is True (after at least one prev/recent comparison), or
      * ``max_warmup_bars`` is reached.

    If the iterable runs dry before either fires, the loop exits with whatever
    bars it has and ``converged`` reflects whether the last comparison passed.

    Parameters
    ----------
    warmup_observations
        Iterable of ``(Observation, label)`` pairs. Each ``Observation`` MUST
        already carry ``sigma_ve`` and ``p_online`` (computed by the caller's
        pipeline). ``label`` is an int / bool indicating whether the upper
        barrier was hit first within the look-ahead horizon.
    training_label_ttbarrier
        Array / sequence of time-to-first-barrier values from the training set.
        Used only for the ``t_max`` quantile.
    quantile_sigma
        Quantile of ``sigma_ve`` used as ``σ_max``. Default 0.75 — the strategy
        will pause when ``sigma_ve > σ_max``.
    stability_window
        Bars per rolling Brier window for the convergence check.
    stability_threshold
        Maximum relative change in Brier between adjacent windows that still
        counts as "stable".
    min_warmup_bars
        Minimum bars before convergence is even evaluated.
    max_warmup_bars
        Hard cap on warmup length. ``None`` means consume the full iterable.
    tau_floor
        EV-breakeven floor on ``τ_init`` (must satisfy probability ≥ this to be
        worth crossing the spread + fees on average).
    r_min
        Sanity floor on the empirical event rate; below this the warmup window
        had no opportunities and the experiment cannot proceed.
    t_max_quantile
        Quantile of ``training_label_ttbarrier`` taken as the pause-budget
        ``t_max``.

    Returns
    -------
    WarmupCalibration
        Populated with all four parameters, baseline Briers, the rolling-Brier
        trace, and the convergence flag.

    Raises
    ------
    CalibrationError
        If ``r_star_init < r_min`` or ``σ_max`` is not strictly below
        ``0.95 * max(sigma_ve)``.
    """
    # --- Stream collect -----------------------------------------------------
    sigma_arr: list[float] = []
    p_online_arr: list[float] = []
    p_offline_arr: list[float] = []
    label_arr: list[int] = []

    stability_trace: list[float] = []
    converged = False
    half_window = max(1, stability_window // 2)
    next_check_at = min_warmup_bars

    for obs, label in warmup_observations:
        if obs.sigma_ve is None:
            raise CalibrationError(
                f"observation missing sigma_ve at bar {len(sigma_arr)}; "
                "the caller must run the pipeline so sigma_ve is set"
            )
        if obs.p_online is None:
            raise CalibrationError(
                f"observation missing p_online at bar {len(sigma_arr)}; "
                "the caller must run the pipeline so p_online is set"
            )
        sigma_arr.append(float(obs.sigma_ve))
        p_online_arr.append(float(obs.p_online))
        # p_offline may be None on the first observation of a fresh stage; treat
        # missing as p_online so the offline baseline is at worst as bad as the
        # online one (rather than fabricating a zero).
        if obs.p_offline is None:
            p_offline_arr.append(float(obs.p_online))
        else:
            p_offline_arr.append(float(obs.p_offline))
        label_arr.append(int(label))

        n = len(sigma_arr)

        # --- Convergence check (rolling Brier) -----------------------------
        if n >= max(min_warmup_bars, 2 * stability_window) and n >= next_check_at:
            recent = _brier(
                p_online_arr[-stability_window:],
                label_arr[-stability_window:],
            )
            prev = _brier(
                p_online_arr[-2 * stability_window: -stability_window],
                label_arr[-2 * stability_window: -stability_window],
            )
            stability_trace.append(recent)
            if prev > 0.0:
                rel = abs(recent - prev) / prev
                if rel < stability_threshold:
                    converged = True
            next_check_at = n + half_window

        if converged:
            break
        if max_warmup_bars is not None and n >= max_warmup_bars:
            break

    n_used = len(sigma_arr)
    if n_used == 0:
        raise CalibrationError("warmup iterable yielded zero observations")

    # --- Scalar reductions --------------------------------------------------
    sigma_np = np.asarray(sigma_arr, dtype=float)
    p_online_np = np.asarray(p_online_arr, dtype=float)

    sigma_max = float(np.quantile(sigma_np, quantile_sigma))
    r_star_init = float(np.mean(label_arr))

    # τ_init: quantile of p_online such that P(p ≥ τ) ≈ r*_init. For empirical
    # rate r* we want the (1 - r*) quantile of the predicted-probability
    # distribution. Floored at the EV-breakeven.
    tau_q = max(0.0, min(1.0, 1.0 - r_star_init))
    tau_init = float(np.quantile(p_online_np, tau_q))
    tau_init = max(tau_init, float(tau_floor))

    # t_max from training labels — never from warmup (warmup is too short to
    # produce a stable tail estimate).
    tt_arr = np.asarray(training_label_ttbarrier, dtype=float)
    if tt_arr.size == 0:
        raise CalibrationError(
            "training_label_ttbarrier is empty; cannot compute t_max"
        )
    t_max = int(np.ceil(np.quantile(tt_arr, t_max_quantile)))

    brier_baseline = _brier(p_online_arr, label_arr)
    brier_offline_baseline = _brier(p_offline_arr, label_arr)

    # --- Sanity gates -------------------------------------------------------
    if r_star_init < r_min:
        raise CalibrationError(
            f"r_star_init={r_star_init:.4f} < r_min={r_min:.4f}: "
            "warmup window had no (or insufficient) opportunities; either the "
            "selected window is degenerate or the labeller is mis-configured"
        )

    sigma_observed_max = float(np.max(sigma_np))
    sigma_max_ceiling = sigma_observed_max * 0.95
    # Degenerate σ_VE distribution (e.g., single-member ensemble where every
    # σ_VE = 0) — the gate is intentionally non-restrictive. Set sigma_max
    # to +inf so the strategy never gates on this signal, and emit a warning
    # rather than failing. This is the correct behavior when offline
    # uncertainty information is genuinely unavailable.
    if sigma_observed_max <= 0.0:
        import warnings as _w
        _w.warn(
            "warmup_calibration: σ_VE distribution is degenerate (all zeros); "
            "ensemble_n likely == 1. Disabling the σ_max gate (sigma_max=inf).",
            stacklevel=2,
        )
        sigma_max = float("inf")
    elif sigma_max >= sigma_max_ceiling:
        raise CalibrationError(
            f"sigma_max={sigma_max:.6g} is not below 0.95 * max(sigma_ve)="
            f"{sigma_max_ceiling:.6g}: the ensemble-dispersion gate would be "
            "non-restrictive (almost no bars rejected). Lower quantile_sigma "
            "or revisit the sigma_ve distribution before trading."
        )

    return WarmupCalibration(
        tau_init=float(tau_init),
        sigma_max=float(sigma_max),
        r_star_init=float(r_star_init),
        t_max=int(t_max),
        brier_baseline=float(brier_baseline),
        brier_offline_baseline=float(brier_offline_baseline),
        warmup_bars_used=int(n_used),
        arf_stability_trace=[float(x) for x in stability_trace],
        converged=bool(converged),
    )


__all__ = ["WarmupCalibration", "CalibrationError", "calibrate_from_warmup"]
