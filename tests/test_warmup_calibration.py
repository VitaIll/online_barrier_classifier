"""Tests for ``wagie.experiments.warmup_calibration``.

The module is the pure statistical reducer that runs once before trading begins.
It must:

  * compute σ_max / r*_init / τ_init / t_max from the warmup stream,
  * detect ARF rolling-Brier convergence,
  * raise ``CalibrationError`` on degenerate input,
  * stop at ``max_warmup_bars`` even if not converged,
  * round-trip through JSON without numpy-scalar leakage.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from wagie.core.event import DecisionBar
from wagie.core.identity import DEFAULT_INSTRUMENT
from wagie.core.numeric import Price, Probability, Quantity
from wagie.core.observation import Observation
from wagie.core.time import Duration, Timestamp
from wagie.experiments.warmup_calibration import (
    CalibrationError,
    WarmupCalibration,
    calibrate_from_warmup,
)


_NS_PER_MIN = 60_000_000_000


def _bar(close: float = 100.0, ts_min: int = 1) -> DecisionBar:
    return DecisionBar(
        ts_init=Timestamp(_NS_PER_MIN * ts_min),
        instrument=DEFAULT_INSTRUMENT,
        open=Price(close),
        high=Price(close * 1.01),
        low=Price(close * 0.99),
        close=Price(close),
        volume=Quantity(1.0),
        duration=Duration.from_minutes(20),
        segment_id=0,
    )


def _obs(
    *,
    sigma_ve: float,
    p_online: float,
    p_offline: float | None = None,
    ts_min: int = 1,
) -> Observation:
    """Construct a fully-populated Observation for the warmup stream."""
    o = (
        Observation(bar=_bar(ts_min=ts_min))
        .with_sigma_ve(sigma_ve)
        .with_p_online(Probability(p_online))
    )
    if p_offline is not None:
        o = o.with_p_offline(Probability(p_offline))
    return o


def _stream(sigmas, p_onlines, labels, p_offlines=None):
    """Yield ``(Observation, label)`` pairs from parallel arrays."""
    if p_offlines is None:
        p_offlines = [None] * len(sigmas)
    for i, (s, p, y, po) in enumerate(zip(sigmas, p_onlines, labels, p_offlines)):
        yield _obs(sigma_ve=s, p_online=p, p_offline=po, ts_min=i + 1), int(y)


# ---------------------------------------------------------------------------
# Happy path — synthetic stable signal
# ---------------------------------------------------------------------------


def test_synthetic_stable_signal_converges_and_returns_expected_quantiles():
    """Constant p_online=0.1, label rate ≈ 0.1, varied sigma_ve.

    With the signal stable from the first bar the rolling-Brier check fires
    and ``converged`` becomes True. σ_max is the 75th-percentile of the
    sigma_ve array. With r*≈0.1 the τ-quantile is the 90th-pct of p_online,
    which is 0.1 — but the EV-breakeven floor (0.524) wins.
    """
    rng = np.random.default_rng(0)
    n = 200
    sigmas = rng.uniform(0.01, 0.20, size=n).tolist()
    p_onlines = [0.1] * n
    labels = [1 if rng.random() < 0.1 else 0 for _ in range(n)]

    cal = calibrate_from_warmup(
        _stream(sigmas, p_onlines, labels),
        training_label_ttbarrier=np.arange(1, 101),
    )

    expected_sigma_max = float(np.quantile(np.asarray(sigmas[: cal.warmup_bars_used]), 0.75))
    assert cal.sigma_max == pytest.approx(expected_sigma_max)
    # τ_init: floor wins because constant p_online=0.1 < tau_floor=0.524.
    assert cal.tau_init == pytest.approx(0.524)
    # Brier of constant prediction 0.1 vs rate≈0.1: ≈ 0.1 * 0.81 + 0.9 * 0.01 = 0.09.
    assert cal.brier_baseline == pytest.approx(0.09, abs=0.05)
    assert cal.warmup_bars_used >= 100
    assert cal.r_star_init > 0.0


def test_tau_init_uses_quantile_when_above_floor():
    """If the (1 - r*)-quantile of p_online sits above the EV floor, use it."""
    n = 200
    # Spread p_online widely so the 90th percentile is well above the floor.
    p_onlines = np.linspace(0.0, 1.0, n).tolist()
    sigmas = np.linspace(0.01, 0.20, n).tolist()
    # ~10% events distributed uniformly.
    rng = np.random.default_rng(1)
    labels = [1 if rng.random() < 0.1 else 0 for _ in range(n)]

    cal = calibrate_from_warmup(
        _stream(sigmas, p_onlines, labels),
        training_label_ttbarrier=np.arange(1, 101),
        max_warmup_bars=n,
    )
    # τ should be approximately the (1 - r*)-quantile of np.linspace(0, 1, n).
    used = cal.warmup_bars_used
    expected_tau = float(np.quantile(np.asarray(p_onlines[:used]), 1.0 - cal.r_star_init))
    expected_tau = max(expected_tau, 0.524)
    assert cal.tau_init == pytest.approx(expected_tau, abs=1e-9)
    assert cal.tau_init > 0.524  # the quantile won, not the floor


# ---------------------------------------------------------------------------
# Degenerate input → CalibrationError
# ---------------------------------------------------------------------------


def test_all_labels_zero_raises_calibration_error():
    """r_star_init = 0 < r_min must raise."""
    n = 200
    sigmas = np.linspace(0.01, 0.20, n).tolist()
    p_onlines = [0.1] * n
    labels = [0] * n

    with pytest.raises(CalibrationError, match="r_star_init"):
        calibrate_from_warmup(
            _stream(sigmas, p_onlines, labels),
            training_label_ttbarrier=np.arange(1, 101),
        )


def test_identical_sigma_ve_raises_calibration_error():
    """When all sigma_ve are equal, the 75th-percentile EQUALS the max,
    so the σ_max gate would be non-restrictive — must raise."""
    n = 200
    sigmas = [0.05] * n
    p_onlines = [0.1] * n
    rng = np.random.default_rng(2)
    labels = [1 if rng.random() < 0.10 else 0 for _ in range(n)]

    with pytest.raises(CalibrationError, match="sigma_max"):
        calibrate_from_warmup(
            _stream(sigmas, p_onlines, labels),
            training_label_ttbarrier=np.arange(1, 101),
            max_warmup_bars=n,
        )


def test_missing_sigma_ve_raises():
    """Caller bug: forgot to set sigma_ve before passing in the observation."""
    bar = _bar()
    obs = Observation(bar=bar).with_p_online(Probability(0.1))  # no sigma_ve

    def gen():
        yield obs, 1

    with pytest.raises(CalibrationError, match="sigma_ve"):
        calibrate_from_warmup(gen(), training_label_ttbarrier=np.arange(1, 101))


def test_missing_p_online_raises():
    bar = _bar()
    obs = Observation(bar=bar).with_sigma_ve(0.1)  # no p_online

    def gen():
        yield obs, 1

    with pytest.raises(CalibrationError, match="p_online"):
        calibrate_from_warmup(gen(), training_label_ttbarrier=np.arange(1, 101))


def test_empty_training_ttbarrier_raises():
    n = 200
    sigmas = np.linspace(0.01, 0.20, n).tolist()
    p_onlines = [0.1] * n
    rng = np.random.default_rng(3)
    labels = [1 if rng.random() < 0.10 else 0 for _ in range(n)]

    with pytest.raises(CalibrationError, match="training_label_ttbarrier"):
        calibrate_from_warmup(
            _stream(sigmas, p_onlines, labels),
            training_label_ttbarrier=np.array([], dtype=float),
        )


# ---------------------------------------------------------------------------
# Convergence detection
# ---------------------------------------------------------------------------


def test_convergence_detection_with_two_stable_blocks():
    """Feed 100 stable bars then 100 more stable bars — must report converged
    and produce a non-empty stability trace."""
    rng = np.random.default_rng(4)
    n = 200
    sigmas = rng.uniform(0.01, 0.20, size=n).tolist()
    p_onlines = [0.1] * n
    labels = [1 if rng.random() < 0.10 else 0 for _ in range(n)]

    cal = calibrate_from_warmup(
        _stream(sigmas, p_onlines, labels),
        training_label_ttbarrier=np.arange(1, 101),
        stability_window=50,
        min_warmup_bars=100,
        max_warmup_bars=n,
    )
    assert cal.converged is True
    assert len(cal.arf_stability_trace) >= 1
    # All trace values are finite and non-negative (Brier ∈ [0, 1]).
    for x in cal.arf_stability_trace:
        assert 0.0 <= x <= 1.0


# ---------------------------------------------------------------------------
# Hard cap
# ---------------------------------------------------------------------------


def test_hard_cap_returns_at_max_warmup_bars_even_if_unconverged():
    """With max_warmup_bars=50 < 2*stability_window=100, we cannot converge.
    The function must return at exactly 50 bars with converged=False."""
    rng = np.random.default_rng(5)
    n = 500
    sigmas = rng.uniform(0.01, 0.20, size=n).tolist()
    # Drifting probability so the stability check would fail anyway.
    p_onlines = [0.1 + 0.001 * i for i in range(n)]
    labels = [1 if rng.random() < 0.10 else 0 for _ in range(n)]

    cal = calibrate_from_warmup(
        _stream(sigmas, p_onlines, labels),
        training_label_ttbarrier=np.arange(1, 101),
        stability_window=50,
        min_warmup_bars=40,
        max_warmup_bars=50,
    )
    assert cal.warmup_bars_used == 50
    assert cal.converged is False


# ---------------------------------------------------------------------------
# t_max
# ---------------------------------------------------------------------------


def test_t_max_quantile_matches_numpy_for_simple_array():
    """Spec: training_label_ttbarrier=[1..100], q=0.95 → 95."""
    rng = np.random.default_rng(6)
    n = 200
    sigmas = rng.uniform(0.01, 0.20, size=n).tolist()
    p_onlines = [0.1] * n
    labels = [1 if rng.random() < 0.10 else 0 for _ in range(n)]

    cal = calibrate_from_warmup(
        _stream(sigmas, p_onlines, labels),
        training_label_ttbarrier=np.arange(1, 101),
        t_max_quantile=0.95,
    )
    # np.quantile on [1..100] at 0.95 → 95.05; ceil → 96.
    expected = int(np.ceil(np.quantile(np.arange(1, 101), 0.95)))
    assert cal.t_max == expected
    # Sanity: this is 96 (ceil of 95.05), close to the spec hint of 95.
    assert cal.t_max in (95, 96)


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_to_dict_is_json_serializable_with_no_numpy_leakage():
    rng = np.random.default_rng(7)
    n = 200
    sigmas = rng.uniform(0.01, 0.20, size=n).tolist()
    p_onlines = [0.1] * n
    p_offlines = [0.12] * n
    labels = [1 if rng.random() < 0.10 else 0 for _ in range(n)]

    cal = calibrate_from_warmup(
        _stream(sigmas, p_onlines, labels, p_offlines),
        training_label_ttbarrier=np.arange(1, 101),
    )
    d = cal.to_dict()

    # Every scalar field is a Python primitive, not a numpy scalar.
    assert type(d["tau_init"]) is float
    assert type(d["sigma_max"]) is float
    assert type(d["r_star_init"]) is float
    assert type(d["t_max"]) is int
    assert type(d["brier_baseline"]) is float
    assert type(d["brier_offline_baseline"]) is float
    assert type(d["warmup_bars_used"]) is int
    assert type(d["converged"]) is bool
    assert isinstance(d["arf_stability_trace"], list)
    for x in d["arf_stability_trace"]:
        assert type(x) is float

    # And the whole thing serializes cleanly.
    s = json.dumps(d)
    round_trip = json.loads(s)
    assert round_trip == d


def test_brier_offline_falls_back_to_p_online_when_offline_missing():
    """Missing p_offline must NOT fabricate a zero — we copy p_online so the
    offline baseline is at worst as bad as the online one."""
    n = 200
    sigmas = np.linspace(0.01, 0.20, n).tolist()
    p_onlines = [0.1] * n
    rng = np.random.default_rng(8)
    labels = [1 if rng.random() < 0.10 else 0 for _ in range(n)]

    cal = calibrate_from_warmup(
        _stream(sigmas, p_onlines, labels),  # no p_offlines
        training_label_ttbarrier=np.arange(1, 101),
    )
    # When offline falls back to online, the two Briers are equal.
    assert cal.brier_offline_baseline == pytest.approx(cal.brier_baseline)


def test_warmup_calibration_dataclass_fields_minimal_construction():
    """Direct construction works for downstream consumers that build the
    dataclass from a persisted dict."""
    cal = WarmupCalibration(
        tau_init=0.6,
        sigma_max=0.1,
        r_star_init=0.1,
        t_max=42,
        brier_baseline=0.09,
        brier_offline_baseline=0.10,
        warmup_bars_used=144,
        arf_stability_trace=[0.09, 0.092],
        converged=True,
    )
    assert cal.to_dict()["t_max"] == 42
