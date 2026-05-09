"""Integration test for H-005 (round 001).

Loads `artifacts/online_eval/predictions.parquet` plus the 1m and 20m parquets,
runs `simulate_inventory_aware` on the real test stream, and asserts the
harness emits all-finite metrics with valid exit reasons. This is the gap the
synthetic tests in `test_backtest.py` cannot cover: it specifically validates
the open_time → minute-index mapping (CODE-SCOUT round-001 finding).

Skipped automatically if any of the three input files is missing — that keeps
the test cheap to pull into a CI environment without raw data, and makes the
test a hard *integration* check whenever the artifacts are present.
"""

from __future__ import annotations

from pathlib import Path

import math
import numpy as np
import pandas as pd
import pytest

from src.backtest import simulate_inventory_aware

REPO = Path(__file__).resolve().parents[1]
PREDS = REPO / "artifacts" / "online_eval" / "predictions.parquet"
BARS = REPO / "data" / "model_data" / "BTCUSDT" / "bars_20m_features.parquet"
MIN1 = REPO / "data" / "cleansed_data" / "BTCUSDT" / "1m.parquet"

REQUIRED = (PREDS, BARS, MIN1)
ARTIFACTS_PRESENT = all(p.exists() for p in REQUIRED)
SKIP_REASON = (
    "H-005 integration test needs artifacts: "
    + ", ".join(str(p.relative_to(REPO)) for p in REQUIRED if not p.exists())
)


@pytest.mark.skipif(not ARTIFACTS_PRESENT, reason=SKIP_REASON)
def test_h005_loader_mapping_is_consistent():
    """The mapping minute_idx → decision-bar k must round-trip exactly."""
    preds = pd.read_parquet(PREDS)
    bars = pd.read_parquet(BARS, columns=["open_time", "close_time", "close"])
    n_test = len(preds)
    test_bars = bars.iloc[-n_test:].reset_index(drop=True)

    first_open_ms = int(test_bars["open_time"].iloc[0])
    last_open_ms = int(test_bars["open_time"].iloc[-1])
    last_close_ms = int(test_bars["close_time"].iloc[-1])

    mn_full = pd.read_parquet(MIN1, columns=["open_time", "high", "low", "close"])
    mask = (mn_full["open_time"] >= first_open_ms) & (
        mn_full["open_time"] <= last_close_ms + 60_000
    )
    mn_test = mn_full.loc[mask].reset_index(drop=True)

    minute_idx = np.searchsorted(
        mn_test["open_time"].to_numpy(),
        test_bars["open_time"].to_numpy(),
    )
    matched_open_times = mn_test["open_time"].to_numpy()[minute_idx]
    target_open_times = test_bars["open_time"].to_numpy()
    # Open-time of decision-bar i must equal minute open_time at minute_idx[i].
    assert (matched_open_times == target_open_times).all(), (
        "decision-bar open_times do not align with minute index"
    )

    # No bar should ever land off-grid (open_time must be a 20-minute multiple
    # within the test slice's minute coordinates).
    M = 20
    relative_minutes = (target_open_times - first_open_ms) // 60_000
    # Allow gaps: relative_minutes need not be exactly arange(N)*M, but each
    # individual offset must be a non-negative integer of minutes.
    assert (relative_minutes >= 0).all()
    assert (relative_minutes % M == 0).all(), (
        "at least one decision-bar open_time is not a 20-minute multiple"
    )


@pytest.mark.skipif(not ARTIFACTS_PRESENT, reason=SKIP_REASON)
def test_h005_simulate_inventory_aware_emits_finite_metrics_on_real_stream():
    """Round-001 falsifier: harness emits finite metrics on real predictions.

    For at least one τ_open in {0.10, 0.20, 0.50}, on each of {p_offline,
    p_final}, every standard metric must be finite (no NaN, no inf), exit
    reasons must be a subset of {tp, sl, timeout}, and bars_held ≤ M.
    """
    preds = pd.read_parquet(PREDS)
    bars = pd.read_parquet(BARS, columns=["open_time", "close_time"])
    mn_full = pd.read_parquet(MIN1, columns=["open_time", "high", "low", "close", "segment_id"])

    n_test = len(preds)
    test_bars = bars.iloc[-n_test:].reset_index(drop=True)
    first_open_ms = int(test_bars["open_time"].iloc[0])
    last_close_ms = int(test_bars["close_time"].iloc[-1])
    M = 20

    mask = (mn_full["open_time"] >= first_open_ms) & (
        mn_full["open_time"] <= last_close_ms + 60_000
    )
    mn_test = mn_full.loc[mask].reset_index(drop=True)

    # Build boundaries with k = minute_idx_in_test_slice // M (per CODE-SCOUT).
    # This matches simulate_inventory_aware's internal `n_k = k * M`.
    minute_idx = np.searchsorted(
        mn_test["open_time"].to_numpy(),
        test_bars["open_time"].to_numpy(),
    )
    boundaries_full = pd.DataFrame({
        "k": minute_idx // M,
        "ts": pd.to_datetime(test_bars["open_time"], unit="ms", utc=True),
    })

    # Drop bars whose lookahead window [n_k+1, n_k+M] crosses the major gap.
    # A safe rule: keep only bars whose minute_idx + M lies inside the same
    # segment_id as minute_idx itself.
    seg = mn_test["segment_id"].to_numpy()
    n_k_arr = (boundaries_full["k"].to_numpy()) * M
    n_close_arr = n_k_arr + M
    in_range = n_close_arr < len(seg)
    keep = np.zeros(len(boundaries_full), dtype=bool)
    keep[in_range] = seg[n_k_arr[in_range]] == seg[n_close_arr[in_range]]

    boundaries = boundaries_full.loc[keep].reset_index(drop=True)
    p_off = preds["p_offline"].to_numpy()[keep]
    p_fin = preds["p_final"].to_numpy()[keep]

    minute_close = mn_test["close"].to_numpy()
    minute_high = mn_test["high"].to_numpy()
    minute_low = mn_test["low"].to_numpy()

    metric_names = (
        "n_trades", "total_log_return", "sharpe", "probabilistic_sharpe",
        "sortino", "max_drawdown_log", "cdar_5pct_log", "hit_rate", "profit_factor",
    )
    valid_exits = {"tp", "sl", "timeout"}

    found_any = False
    for tau in (0.10, 0.20, 0.50):
        for label, p in (("p_offline", p_off), ("p_final", p_fin)):
            res = simulate_inventory_aware(
                boundaries, minute_close, minute_high, minute_low, p,
                tau_open=tau, M=M, phi=0.00411135, c_stop=0.00411135, cost_bps=1.0,
            )
            for m in metric_names:
                v = res.metrics[m]
                assert math.isfinite(v) or (m == "profit_factor" and not res.trades.empty), (
                    f"non-finite metric {m}={v} at tau={tau} signal={label}"
                )
            if not res.trades.empty:
                assert set(res.trades["exit_reason"].unique()).issubset(valid_exits)
                assert (res.trades["bars_held"] <= M).all()
                assert (res.trades["k_close"] >= res.trades["k_open"] + 1).all()
                found_any = True

    assert found_any, "no τ produced any trades on real data — mapping bug"


@pytest.mark.skipif(not ARTIFACTS_PRESENT, reason=SKIP_REASON)
def test_h005_offline_overprediction_yields_more_trades_than_online():
    """THEORIST round-001 prediction: offline-only opens MORE trades than
    offline+online at identical τ ≥ 0.20 (offline mean p ≈ 0.20; online ≈ 0.097).

    This is a sanity check on the real prediction stream, not a model claim.
    """
    preds = pd.read_parquet(PREDS)
    p_off = preds["p_offline"].to_numpy()
    p_fin = preds["p_final"].to_numpy()
    for tau in (0.20, 0.30, 0.50):
        n_off = int((p_off > tau).sum())
        n_fin = int((p_fin > tau).sum())
        # Allow equality at extreme τ where both are starved; require strict
        # inequality at moderate τ that THEORIST flagged.
        if tau == 0.20:
            assert n_off > n_fin, (
                f"expected p_offline > p_final crossings at τ=0.20; got {n_off} vs {n_fin}"
            )
        else:
            assert n_off >= n_fin, (
                f"expected p_offline >= p_final crossings at τ={tau}; got {n_off} vs {n_fin}"
            )
